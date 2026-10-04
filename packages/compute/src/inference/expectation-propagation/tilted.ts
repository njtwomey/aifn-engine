/**
 * Tilted distributions: a Gaussian cavity N(θ; m, v) times one exact factor, summarised by its log normaliser and
 * first two moments (the moment-matching step of EP and ADF). Closed forms for step, probit, interval (truncated
 * Gaussian, the TrueSkill draw) and clutter factors (Minka 2001, thesis §3.2–3.3, §5; Herbrich, Minka & Graepel 2007,
 * "TrueSkill", NIPS), and a quadrature fallback for any factor raised to a power.
 *
 * Every closed form broadcasts over its arguments (numbers in, numbers out; tensors elementwise). The truncated-normal
 * ratios v = φ/Φ and w = v(v + t) come from `aifn-compute/numerics/special`, accurate far into both tails, so extreme cavities do not
 * give 0/0. These functions are not differentiable primitives: EP uses them for updates, not under autodiff.
 */

import {
  normalLogCdf,
  normalLogIntervalProbability,
  normalLogPdf,
  truncatedNormalV,
  truncatedNormalW,
} from 'aifn-compute/numerics/special'
import {
  broadcastShapes,
  broadcastTo,
  fromData,
  isTensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

/** The tilted distribution's log normaliser log Z = log ∫ N(θ; m, v) f(θ) dθ and its mean and variance. */
export interface Tilted<T extends Value = number> {
  logZ: T
  mean: T
  variance: T
}

type Numbers = readonly (number | Tensor)[]

/**
 * Apply a scalar kernel returning several named numbers elementwise over broadcast arguments: numbers in give numbers
 * out, otherwise each named output is a tensor of the broadcast shape. The tilted moments are built on it.
 */
export function lift<K extends string>(
  args: Numbers,
  keys: readonly K[],
  f: (...x: number[]) => Record<K, number>,
): Record<K, number | Tensor> {
  if (args.every((a) => typeof a === 'number')) return f(...(args as number[]))
  const shape = broadcastShapes(...args.map((a) => (isTensor(a) ? a.shape : [])))
  const flats = args.map((a) => (isTensor(a) ? toFlat(broadcastTo(a, shape)) : null))
  const n = shape.reduce((p, s) => p * s, 1)
  const out = Object.fromEntries(keys.map((k) => [k, new Float64Array(n)])) as Record<K, Float64Array>
  for (let i = 0; i < n; i++) {
    const r = f(...args.map((a, j) => (flats[j] ? flats[j]![i] : (a as number))))
    for (const k of keys) out[k][i] = r[k]
  }
  return Object.fromEntries(keys.map((k) => [k, fromData(out[k], shape)])) as Record<K, Tensor>
}

/** The output type of a lifted kernel: a number when every argument is a number, else a tensor. */
export type Out<T> = [T] extends [number] ? number : Tensor

const v = (t: number) => truncatedNormalV(t) as number
const w = (t: number) => truncatedNormalW(t) as number

/**
 * Step factor 𝟙(θ > threshold): the cavity truncated below. With t = (m − threshold)/√v:
 * log Z = log Φ(t), mean = m + √v · v(t), variance = v (1 − w(t)).
 */
export function stepTilted<M extends number | Tensor, S extends number | Tensor>(
  mean: M,
  variance: S,
  threshold: number | Tensor = 0,
): Tilted<Out<M | S>> {
  return lift([mean, variance, threshold], ['logZ', 'mean', 'variance'], (m, s2, e) => {
    const s = Math.sqrt(s2)
    const t = (m - e) / s
    return { logZ: normalLogCdf(t) as number, mean: m + s * v(t), variance: s2 * (1 - w(t)) }
  }) as Tilted<Out<M | S>>
}

/** Options of {@link probitTilted}. */
export interface ProbitOptions {
  /** c in Φ(y(θ − c)/s). Default 0. */
  offset?: number | Tensor
  /** s², the variance of the Gaussian noise the probit integrates out. Default 1. */
  noiseVariance?: number | Tensor
}

/**
 * Probit factor Φ(y(θ − c)/s), y = ±1: the step factor on θ plus N(0, s²) noise. With d = √(s² + v) and
 * z = y(m − c)/d: log Z = log Φ(z), mean = m + y v · v(z)/d, variance = v − v² w(z)/(s² + v).
 */
export function probitTilted<M extends number | Tensor, S extends number | Tensor>(
  mean: M,
  variance: S,
  y: number | Tensor = 1,
  { offset = 0, noiseVariance = 1 }: ProbitOptions = {},
): Tilted<Out<M | S>> {
  return lift([mean, variance, y, offset, noiseVariance], ['logZ', 'mean', 'variance'], (m, s2, yy, c, n2) => {
    const d = Math.sqrt(n2 + s2)
    const z = (yy * (m - c)) / d
    return {
      logZ: normalLogCdf(z) as number,
      mean: m + (yy * s2 * v(z)) / d,
      variance: s2 - (s2 * s2 * w(z)) / (n2 + s2),
    }
  }) as Tilted<Out<M | S>>
}

/**
 * Interval factor 𝟙(lower < θ < upper): the cavity truncated to an interval (either end may be infinite). With
 * a = (lower − m)/√v, b = (upper − m)/√v, Z = Φ(b) − Φ(a) (computed without cancellation), r_a = φ(a)/Z,
 * r_b = φ(b)/Z: mean = m + √v (r_a − r_b), variance = v [1 + a r_a − b r_b − (r_a − r_b)²].
 */
export function intervalTilted<M extends number | Tensor, S extends number | Tensor>(
  mean: M,
  variance: S,
  lower: number | Tensor,
  upper: number | Tensor,
): Tilted<Out<M | S>> {
  return lift([mean, variance, lower, upper], ['logZ', 'mean', 'variance'], (m, s2, lo, hi) => {
    const s = Math.sqrt(s2)
    const a = (lo - m) / s
    const b = (hi - m) / s
    const logZ = normalLogIntervalProbability(a, b) as number
    const ra = Number.isFinite(a) ? Math.exp((normalLogPdf(a) as number) - logZ) : 0
    const rb = Number.isFinite(b) ? Math.exp((normalLogPdf(b) as number) - logZ) : 0
    const ara = Number.isFinite(a) ? a * ra : 0
    const brb = Number.isFinite(b) ? b * rb : 0
    return { logZ, mean: m + s * (ra - rb), variance: s2 * (1 + ara - brb - (ra - rb) ** 2) }
  }) as Tilted<Out<M | S>>
}

/** log N(x; m, s2), through the one standard normal log-density of `aifn-compute/numerics/special`. */
const logNormal = (x: number, m: number, s2: number) =>
  (normalLogPdf((x - m) / Math.sqrt(s2)) as number) - 0.5 * Math.log(s2)

/** Options of {@link tiltedByQuadrature}. */
export interface QuadratureTiltOptions {
  /** Raise the factor to this power (power EP). Default 1. */
  power?: number
  /** Grid points (default 801) spanning ± `width` cavity standard deviations (default 10). */
  points?: number
  width?: number
}

/**
 * Moments of N(θ; m, v) · f(θ)^power by the trapezoid rule on a grid of ± width cavity standard deviations, for
 * factors without closed forms. `logFactor` is log f. Accurate when the tilted mass lies within the grid.
 */
export function tiltedByQuadrature(
  mean: number,
  variance: number,
  logFactor: (theta: number) => number,
  { power = 1, points = 801, width = 10 }: QuadratureTiltOptions = {},
): Tilted {
  const s = Math.sqrt(variance)
  const lo = mean - width * s
  const h = (2 * width * s) / (points - 1)
  const logs = new Float64Array(points)
  let top = -Infinity
  for (let i = 0; i < points; i++) {
    const t = lo + i * h
    logs[i] = logNormal(t, mean, variance) + power * logFactor(t)
    if (logs[i] > top) top = logs[i]
  }
  let z = 0
  let m1 = 0
  let m2 = 0
  for (let i = 0; i < points; i++) {
    const t = lo + i * h
    const f = Math.exp(logs[i] - top) * (i === 0 || i === points - 1 ? 0.5 : 1)
    z += f
    m1 += f * t
    m2 += f * t * t
  }
  const m = m1 / z
  return { logZ: top + Math.log(z * h), mean: m, variance: m2 / z - m * m }
}
