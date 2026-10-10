/**
 * WebTraffic (Early et al. 2024, "Inherently interpretable time series classification via multiple instance
 * learning", ICLR, App. C.1): a week of synthetic web traffic per series, with daily and weekly seasonality, into which
 * one of nine signatures is injected (class 0, None, has none). The location of every signature is known, so an
 * interpretation can be scored against it (NDCG@n). Every series is a MIL bag of time points.
 *
 * The base series (Eqs. A.3–A.7), for time index $j = 1, \dots, t$ with $t = 7d$ ($d$ the `samplesPerDay`), is
 * $\operatorname{series}(j) = (\operatorname{RateDay}(j) + \sigma'\varepsilon_j) \cdot \operatorname{RateWeek}(j)$,
 * $\varepsilon_j \sim \Gauss(0, 1)$, where
 *
 * - $\operatorname{WarpedSin}(a, b, p, s, x) = \frac{a}{2} \sin(x' - \sin(x')/s) + b$ with $x' = 2\pi(x - p)$;
 * - $\operatorname{RateDay}(j) = \operatorname{WarpedSin}(a_D, b, p + 0.55, s, j/d)$;
 * - $\operatorname{RateWeek}(j) = \operatorname{WarpedSin}(a_W, 1, 0.6, 2, j/t)$;
 *
 * with $a_D \sim \Unif(2, 4)$, $a_W \sim \Unif(0.8, 1.2)$, $b \sim \Unif(2.5, 5)$, $p \sim \Unif(-0.05, 0.05)$,
 * $s \sim \Unif(1, 3)$ and $\sigma \sim \Unif(2, 4)$ per series, and $\sigma' = \sigma/10$ (see below). Signatures
 * (classes 2–9) go into one window whose length $l$ is a whole number drawn uniformly from a quarter day to two days,
 * starting at a whole number drawn uniformly from $0, \dots, t - l$; class 1 adds spikes at random points. Values are
 * clipped at 0 afterwards.
 *
 * Readings (logged in the progress file): the paper writes the noise as $\Gauss(\operatorname{RateDay}, \sigma)$ with
 * $\sigma \sim \Unif(2, 4)$, but its Fig. A.1 shows noise of about $\pm 0.3$ around rates of 1–5, so the standard
 * deviation used is $\sigma/10$. Peak and trough multiply the window by $1 + c\,\phi(z)$, $z$ from $-5$ to 5 across
 * the window ($\phi$ the standard normal density), which leaves the window's ends unchanged as in Fig. A.2. With fewer
 * samples per day than the paper's 144 (a browser-sized week), the spike probability keeps the paper's expected count
 * ($0.01 \times 1008$ spikes per week) and the average signature's window scales with the sampling rate (at least 2
 * points).
 */

import { child, normal, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkCount, labels, matrix, type DatasetMeta } from '../types'

/** The ten WebTraffic classes, in the paper's order. */
export const WEB_TRAFFIC_CLASSES = [
  'none',
  'spikes',
  'flip',
  'skew',
  'noise',
  'cutoff',
  'average',
  'wander',
  'peak',
  'trough',
] as const

/** A WebTraffic class name. */
export type WebTrafficClass = (typeof WEB_TRAFFIC_CLASSES)[number]

/** Options of `webTraffic`. */
export interface WebTrafficOptions {
  /** Series per class (default 50, as the paper's training and test sets). */
  perClass?: number
  /**
   * Which classes to draw, by index into `WEB_TRAFFIC_CLASSES` (default all ten); labels are $0, \dots, k - 1$ in this
   * order.
   */
  classes?: readonly number[]
  /** Samples per day $d$, an integer of at least 4 (default 144: every ten minutes, $t = 1008$ as the paper). */
  samplesPerDay?: number
}

/** A WebTraffic sample: series as rows, labels, and where each signature is. */
export interface WebTrafficSample {
  /** Marks a dataset. */
  kind: 'dataset'
  /** Series, $n \times t$. */
  x: Tensor
  /** Labels $0, \dots, k - 1$, indexing `classNames`. */
  y: Tensor
  /** The names of the classes drawn, in label order. */
  classNames: readonly WebTrafficClass[]
  /**
   * 1 at the time points the signature changed (the discriminatory points), $n \times t$ (stored as float64 0s and
   * 1s); all 0 for `none`.
   */
  discriminatory: Tensor
  /** The start of each series' signature window (int32; $-1$ for `none` and `spikes`). */
  windowStart: Tensor
  /** The length of each series' signature window (int32; $-1$ for `none` and `spikes`). */
  windowLength: Tensor
  /** The base series before injection, $n \times t$. */
  base: Tensor
  /** The name, description, feature and label names, and the stream's key. */
  meta: DatasetMeta
}

/**
 * The paper's warped sine, $\frac{a}{2} \sin(x' - \sin(x')/s) + b$ with $x' = 2\pi(x - p)$: a sine of period 1 whose
 * peaks lean to one side.
 *
 * @param a The peak-to-trough amplitude.
 * @param b The offset (the mean level).
 * @param p The phase, in periods.
 * @param s The warp: the smaller, the stronger the lean.
 * @param x The position, in periods.
 * @returns The value at `x`.
 */
const warpedSin = (a: number, b: number, p: number, s: number, x: number) => {
  const xp = 2 * Math.PI * (x - p)
  return (a / 2) * Math.sin(xp - Math.sin(xp) / s) + b
}

/**
 * The standard normal density $\phi(z)$.
 *
 * @param z The point.
 * @returns $\phi(z) = e^{-z^2/2}/\sqrt{2\pi}$.
 */
const normalPdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)

/**
 * A number drawn uniformly between two ends.
 *
 * @param s The stream of the draw.
 * @param lo The low end.
 * @param hi The high end.
 * @returns A draw from the uniform distribution between `lo` and `hi`.
 */
const between = (s: Stream, lo: number, hi: number) => lo + (hi - lo) * uniform(s)
/**
 * An integer drawn uniformly from `lo` to `hi` inclusive (the paper's $U_{\integers}$).
 *
 * @param s The stream of the draw.
 * @param lo The smallest value, an integer.
 * @param hi The largest value, an integer at least `lo`.
 * @returns The integer.
 */
const integer = (s: Stream, lo: number, hi: number) => lo + Math.min(hi - lo, Math.floor(uniform(s) * (hi - lo + 1)))

/**
 * One base series (Eqs. A.3–A.7; see the file comment), with its own random parameters.
 *
 * @param s The series' stream: its parameters and noise are drawn from children `'aD'`, `'aW'`, `'b'`, `'p'`, `'s'`,
 *   `'sigma'` and `'noise'`.
 * @param t The number of time points, a week.
 * @param perDay The samples per day $d$.
 * @returns The $t$ values of the series (not yet clipped at 0).
 */
function baseSeries(s: Stream, t: number, perDay: number): Float64Array {
  const aD = between(child(s, 'aD'), 2, 4)
  const aW = between(child(s, 'aW'), 0.8, 1.2)
  const b = between(child(s, 'b'), 2.5, 5)
  const p = between(child(s, 'p'), -0.05, 0.05)
  const skew = between(child(s, 's'), 1, 3)
  const sigma = between(child(s, 'sigma'), 2, 4)
  const z = toFlat(normals(child(s, 'noise'), t))
  return Float64Array.from({ length: t }, (_, k) => {
    const j = k + 1
    const day = warpedSin(aD, b, p + 0.55, skew, j / perDay) + (sigma / 10) * z[k]
    return day * warpedSin(aW, 1, 0.6, 2, j / t)
  })
}

/**
 * Linear interpolation of `v` at a fractional position, clamped to its ends.
 *
 * @param v The values, at positions $0, 1, \dots$.
 * @param x The position, fractional.
 * @returns The value interpolated between the two neighbours of `x`, or the end value outside the range.
 */
const at = (v: ArrayLike<number>, x: number) => {
  const i = Math.max(0, Math.min(v.length - 1, Math.floor(x)))
  const f = Math.min(1, Math.max(0, x - i))
  return i + 1 < v.length ? v[i] * (1 - f) + v[i + 1] * f : v[i]
}

/**
 * Inject class `c`'s signature into a copy of `base`; returns the series and its discriminatory mask. `perDay` scales
 * the window lengths (a quarter day to 2 days), the spike probability and the averaging width.
 *
 * @param s The stream of the signature's random choices.
 * @param base The base series; not modified.
 * @param c The class whose signature is injected (`none` injects nothing).
 * @param perDay The samples per day $d$.
 * @returns `y`, the series with the signature, clipped at 0; `mask`, 1 at the points the signature changed; and
 *   `start` and `length`, the signature's window ($-1$ for `none` and `spikes`).
 */
function inject(s: Stream, base: Float64Array, c: WebTrafficClass, perDay: number) {
  const t = base.length
  const y = Float64Array.from(base)
  const mask = new Int32Array(t)
  let start = -1
  let length = -1
  if (c === 'spikes') {
    const p = Math.min(1, (0.01 * 1008) / t)
    const u = toFlat(uniform(child(s, 'spike'), 0, 1, { shape: [t] }))
    for (let j = 0; j < t; j++)
      if (u[j] < p) {
        const m = 3 + 2 * normal(child(s, 'size', j))
        y[j] += uniform(child(s, 'sign', j)) < 0.5 ? m : -m
        mask[j] = 1
      }
  } else if (c !== 'none') {
    const lo = Math.max(2, Math.round(perDay / 4))
    const hi = Math.max(lo, Math.min(t, Math.round(2 * perDay)))
    length = integer(child(s, 'length'), lo, hi)
    start = integer(child(s, 'start'), 0, t - length)
    const w = base.slice(start, start + length)
    const l = length
    let out: Float64Array
    if (c === 'flip') out = w.reverse()
    else if (c === 'skew') {
      const amount = between(child(s, 'skew'), 0.25, 0.45)
      const ws = uniform(child(s, 'skewSign')) < 0.5 ? 0.5 - amount : 0.5 + amount
      // The window's midpoint moves to ⌊w·l⌋: stretch one side, compress the other (piecewise-linear time warp).
      const mid = Math.max(1, Math.min(l - 1, Math.floor(ws * l)))
      const half = (l - 1) / 2
      out = Float64Array.from({ length: l }, (_, i) =>
        at(w, i <= mid ? (i * half) / mid : half + ((i - mid) * (l - 1 - half)) / Math.max(1, l - 1 - mid)),
      )
    } else if (c === 'noise') {
      const sd = between(child(s, 'sdNoise'), 0.5, 1)
      const z = toFlat(normals(child(s, 'added'), l))
      out = Float64Array.from(w, (v, i) => v + sd * z[i])
    } else if (c === 'cutoff') {
      const cut = between(child(s, 'cut'), 0, 0.2)
      const z = toFlat(normals(child(s, 'cutNoise'), l))
      out = Float64Array.from(w, (_, i) => cut + 0.1 * z[i])
    } else if (c === 'average') {
      const width = Math.max(2, Math.round((integer(child(s, 'width'), 5, 10) * perDay) / 144))
      out = Float64Array.from(w, (_, i) => {
        let sum = 0
        let count = 0
        for (let k = i - Math.floor(width / 2); k < i - Math.floor(width / 2) + width; k++)
          if (start + k >= 0 && start + k < t) {
            sum += base[start + k]
            count++
          }
        return sum / count
      })
    } else if (c === 'wander') {
      const top = between(child(s, 'trend'), 2, 3) * (uniform(child(s, 'trendSign')) < 0.5 ? 1 : -1)
      out = Float64Array.from(w, (v, i) => v + (top * i) / Math.max(1, l - 1))
    } else {
      const scale = between(child(s, 'peak'), 1.5, 2.5) * (c === 'trough' ? -1 : 1)
      out = Float64Array.from(w, (v, i) => v * (1 + scale * normalPdf(-5 + (10 * i) / Math.max(1, l - 1))))
    }
    y.set(out, start)
    mask.fill(1, start, start + l)
  }
  for (let j = 0; j < t; j++) y[j] = Math.max(0, y[j])
  return { y, mask, start, length }
}

/**
 * Draw WebTraffic (see the file comment): `perClass` series of every requested class, in class order; every series has
 * its own substream, so the same seed gives the same series for a class whatever other classes are drawn. Throws
 * `DomainError` when `perClass` is not a non-negative integer, `samplesPerDay` is not an integer of at least 4, or a
 * class index is not in $0, \dots, 9$.
 *
 * @param s The stream the series are drawn from (series $r$ of class `name` from `child(s, name, r)`).
 * @param options The series per class, the classes, and the samples per day.
 * @returns The series as rows, their labels, the class names, the discriminatory points, the signature windows and
 *   the base series.
 *
 * @example Signatures sit in their windows; the rest is the base series
 * const d = webTraffic(stream(1), { perClass: 2, samplesPerDay: 12 })
 * print('x:', d.x.shape, ' y:', toArray(d.y))
 * print('classes:', d.classNames)
 * print('first series, first day:', toArray(d.x)[0].slice(0, 12))
 * const [x, base, mask] = [toArray(d.x), toArray(d.base), toArray(d.discriminatory)]
 * const [start, length] = [toArray(d.windowStart), toArray(d.windowLength)]
 * print('windows (start, length) of the peak series:', [16, 17].map((i) => [start[i], length[i]]))
 * print('marked points of the peak series:', [16, 17].map((i) => mask[i].reduce((a, v) => a + v, 0)))
 * const changedOutside = x.filter((row, i) => row.some((v, j) => !mask[i][j] && v !== Math.max(0, base[i][j]))).length
 * print('series changed outside their marked points:', changedOutside)
 */
export function webTraffic(s: Stream, options: WebTrafficOptions = {}): WebTrafficSample {
  const { perClass = 50, classes = WEB_TRAFFIC_CLASSES.map((_, i) => i), samplesPerDay = 144 } = options
  checkCount(perClass, 'webTraffic')
  if (!(Number.isInteger(samplesPerDay) && samplesPerDay >= 4))
    throw new DomainError('webTraffic', 'webTraffic: samplesPerDay must be an integer of at least 4')
  for (const c of classes)
    if (!(Number.isInteger(c) && c >= 0 && c < WEB_TRAFFIC_CLASSES.length))
      throw new DomainError('webTraffic', `webTraffic: class ${c} is not in 0 … 9`)
  const t = 7 * samplesPerDay
  const n = perClass * classes.length
  const x = new Float64Array(n * t)
  const base = new Float64Array(n * t)
  const mask = new Int32Array(n * t)
  const y = new Int32Array(n)
  const starts = new Int32Array(n)
  const lengths = new Int32Array(n)
  classes.forEach((c, k) => {
    const name = WEB_TRAFFIC_CLASSES[c]
    for (let r = 0; r < perClass; r++) {
      const i = k * perClass + r
      const series = child(s, name, r)
      const b = baseSeries(child(series, 'base'), t, samplesPerDay)
      const made = inject(child(series, 'signature'), b, name, samplesPerDay)
      x.set(made.y, i * t)
      base.set(b, i * t)
      mask.set(made.mask, i * t)
      y[i] = k
      starts[i] = made.start
      lengths[i] = made.length
    }
  })
  const names = classes.map((c) => WEB_TRAFFIC_CLASSES[c])
  return {
    kind: 'dataset',
    x: matrix(x, n, t),
    y: labels(y),
    classNames: names,
    discriminatory: matrix(Float64Array.from(mask), n, t),
    windowStart: labels(starts),
    windowLength: labels(lengths),
    base: matrix(base, n, t),
    meta: {
      name: 'WebTraffic',
      description: `${n} week-long series of synthetic web traffic (${t} points, ${samplesPerDay} a day), ${perClass} per class: ${names.join(', ')}.`,
      task: 'classification',
      featureNames: Array.from({ length: t }, (_, j) => `t${j}`),
      labelNames: names,
      key: s.key,
    } as DatasetMeta,
  }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'webTraffic',
    name: 'WebTraffic (MILLET)',
    summary:
      'A week of seasonal synthetic web traffic per series with one of nine signatures injected at a known place.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      perClass: int(1, 1000, { default: 50 }),
      samplesPerDay: int(4, 288, { default: 144 }),
    }),
    truth: false,
    random: true,
    notes: ['millet', 'multiple-instance-learning'],
    cite: ['early2024'],
  },
  webTraffic,
)
