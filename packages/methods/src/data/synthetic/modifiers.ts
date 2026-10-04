/**
 * Modifiers: pure functions Dataset → Dataset that change one thing about a dataset (its labels, class balance,
 * features or rows) and record themselves in `meta.recipe`. Random ones take a stream first. Each keeps `meta.truth`
 * consistent where it can: label noise composes the noise matrix into the Bayes posterior, prevalence changes reweight
 * it, covariate shift leaves it unchanged (only the marginal of x moves), and a map of the features maps it along.
 */

import { normal, permutation, type Stream, child, integers, uniform } from 'aifn-compute/foundation/random'
import { selectRows } from '../rows'
import { classCounts } from '../sizes'
import {
  REFERENCE_SIZE,
  classificationTruth,
  points,
  pointsFrom,
  remodelRegression,
  type ClassificationTruth,
  type ClassModel,
  type Reference,
  type RegressionTruth,
  type Row,
  type Truth,
} from '../truth'
import { appendStep, labels, matrix, values, vector, type Dataset, type DatasetMeta } from '../types'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { inverse, logDet } from 'aifn-compute/numerics/linalg'
import { sigmoid } from 'aifn-compute/numerics/special'
import { MultivariateNormal, Normal, Uniform } from 'aifn-compute/probability/distributions'
import type { ModifierInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space, when } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Metadata fields a modifier sets. */
type MetaEdit = { -readonly [K in keyof DatasetMeta]?: DatasetMeta[K] }

// ── Helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────

function lazy<T>(f: () => T): () => T {
  let cached: { value: T } | undefined
  return () => (cached ??= { value: f() }).value
}

function step(
  d: Dataset,
  op: string,
  params: Record<string, unknown>,
  sentence: string,
  meta: MetaEdit = {},
): DatasetMeta {
  return {
    ...d.meta,
    ...meta,
    description: `${d.meta.description} ${sentence}`,
    recipe: appendStep(d.meta.recipe, { op, params }),
  }
}

function intLabels(d: Dataset, what: string): Int32Array {
  if (!d.y || d.y.dtype !== 'int32') throw new TypeError(`${what}: needs integer class labels y`)
  return Int32Array.from(values(d.y))
}

/** The number of classes: from the truth, the label names or the largest label. */
function classCount(d: Dataset, y: Int32Array): number {
  const t = d.meta.truth
  if (t?.task === 'classification' && 'classes' in t) return t.classes
  if (d.meta.labelNames) return d.meta.labelNames.length
  return y.reduce((a, b) => Math.max(a, b), -1) + 1
}

function countsOf(y: Int32Array, k: number): number[] {
  const c = new Array<number>(k).fill(0)
  y.forEach((v) => c[v]++)
  return c
}

/** Column means and standard deviations, ignoring NaN (missing) entries. */
function columnStats(x: Float64Array, n: number, d: number): { mean: number[]; sd: number[] } {
  const mean = new Array<number>(d).fill(0)
  const sd = new Array<number>(d).fill(0)
  for (let c = 0; c < d; c++) {
    let m = 0
    let count = 0
    for (let i = 0; i < n; i++) {
      const v = x[i * d + c]
      if (!Number.isNaN(v)) [m, count] = [m + v, count + 1]
    }
    m /= count || 1
    let ss = 0
    for (let i = 0; i < n; i++) {
      const v = x[i * d + c]
      if (!Number.isNaN(v)) ss += (v - m) ** 2
    }
    mean[c] = m
    sd[c] = Math.sqrt(ss / Math.max(count - 1, 1))
  }
  return { mean, sd }
}

/** The intercept α with mean σ(α + zᵢ) = target, by bisection (the mean is increasing in α). */
function calibrate(z: readonly number[], target: number): number {
  if (target <= 0) return -Infinity
  if (target >= 1) return Infinity
  let [lo, hi] = [-60, 60]
  for (let it = 0; it < 100; it++) {
    const mid = (lo + hi) / 2
    const m = z.reduce((a, v) => a + sigmoid(mid + v), 0) / z.length
    if (m < target) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

function head(x: Row, d: number): Row {
  return x.length === d ? x : Float64Array.from({ length: d }, (_, i) => x[i])
}

/** Columns [from, to) of a batch of points. */
function columns(x: Tensor, from: number, to: number): Tensor {
  const { data, n, d } = points(x)
  const out = new Float64Array(n * (to - from))
  for (let i = 0; i < n; i++) out.set(data.subarray(i * d + from, i * d + to), i * (to - from))
  return pointsFrom(out, n, to - from)
}

/** log(eᵃ + eᵇ). */
function logAddExp(a: number, b: number): number {
  const m = Math.max(a, b)
  return m === -Infinity ? -Infinity : m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}

/** The sum of each row of an [n, m] tensor. */
function rowSums(t: Tensor): Float64Array {
  const [n, m] = t.shape
  const v = values(t)
  return Float64Array.from({ length: n }, (_, i) => v.subarray(i * m, (i + 1) * m).reduce((a, b) => a + b, 0))
}

function classTruth(d: Dataset): ClassificationTruth | undefined {
  const t = d.meta.truth
  // A regime truth (`regimeTruth`) is a classification truth without class densities: modifiers leave it alone.
  return t?.task === 'classification' && 'priors' in t ? t : undefined
}

/** A Gaussian regression truth (an additive GAM truth has its own family, which these modifiers do not edit). */
function regTruth(d: Dataset): RegressionTruth | undefined {
  const t = d.meta.truth
  return t?.task === 'regression' && 'noiseSd' in t ? t : undefined
}

/** A new truth from an edited model. */
function remodel(t: ClassificationTruth, edit: Partial<ClassModel>): ClassificationTruth {
  return classificationTruth({ ...t.model, ...edit })
}

function percent(v: number): string {
  return `${+(100 * v).toFixed(1)}%`
}

// ── Label noise ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Label noise: `{ rate }` flips each label with probability `rate` to one of the other classes chosen uniformly
 * (symmetric noise); `{ matrix }` gives class-conditional noise, matrix[i][j] = P(observed j | label i), rows summing
 * to one.
 */
export type LabelNoiseOptions = { rate: number } | { matrix: readonly (readonly number[])[] }

/** The noise matrix of symmetric label noise at `rate` over `k` classes. */
export function symmetricNoise(rate: number, k: number): number[][] {
  if (!(rate >= 0 && rate <= 1))
    throw new DomainError('symmetricNoise', `label noise rate must be in [0, 1], got ${rate}`)
  return Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 - rate : rate / (k - 1))))
}

/**
 * Flip labels at random. Each observed label yᵢ becomes j with probability matrix[yᵢ][j], one uniform draw per row
 * from `s`. `meta.cleanLabels` keeps the labels before the first flip, so `flippedMask` shows which changed. The
 * truth's posterior becomes P(ỹ = j | x) = Σᵢ P(y = i | x) matrix[i][j] (Natarajan et al., 2013, "Learning with
 * noisy labels", NeurIPS), and its Bayes error follows.
 */
export function withLabelNoise(s: Stream, d: Dataset, options: LabelNoiseOptions): Dataset {
  const y = intLabels(d, 'withLabelNoise')
  const k = classCount(d, y)
  const m = 'rate' in options ? symmetricNoise(options.rate, k) : options.matrix.map((r) => [...r])
  if (m.length !== k || m.some((r) => r.length !== k))
    throw new ShapeError('withLabelNoise', `withLabelNoise: the noise matrix must be ${k} × ${k}`)
  m.forEach((r, i) => {
    const sum = r.reduce((a, b) => a + b, 0)
    if (r.some((v) => !(v >= 0)) || Math.abs(sum - 1) > 1e-9)
      throw new DomainError(
        'withLabelNoise',
        `withLabelNoise: row ${i} of the noise matrix must be a probability vector`,
      )
  })
  const out = new Int32Array(y.length)
  for (let i = 0; i < y.length; i++) {
    const u = uniform(s)
    const row = m[y[i]]
    let j = 0
    let c = row[0]
    while (u >= c && j < k - 1) c += row[++j]
    out[i] = j
  }
  const t = classTruth(d)
  const flipped = out.reduce((a, v, i) => a + (v !== y[i] ? 1 : 0), 0)
  const sentence =
    'rate' in options
      ? `Labels flipped at random with probability ${options.rate} (${flipped} flipped).`
      : `Labels flipped by a class-conditional noise matrix (${flipped} flipped).`
  return {
    ...d,
    y: labels(out),
    meta: step(d, 'withLabelNoise', 'rate' in options ? { rate: options.rate } : { matrix: m }, sentence, {
      cleanLabels: d.meta.cleanLabels ?? d.y,
      truth: t ? remodel(t, { ops: [...t.model.ops, { kind: 'noise', matrix: m }] }) : d.meta.truth,
    }),
  }
}

/** 1 where the observed label differs from the clean one (int32, length n); all 0 without label noise. */
export function flippedMask(d: Dataset): Int32Array {
  const y = intLabels(d, 'flippedMask')
  const clean = d.meta.cleanLabels ? values(d.meta.cleanLabels) : undefined
  return Int32Array.from(y, (v, i) => (clean && clean[i] !== v ? 1 : 0))
}

// ── Prevalence and label shift ───────────────────────────────────────────────────────────────────────────────────────

/** Options for `withPrevalence` and `withLabelShift`. */
export interface PrevalenceOptions {
  /** Two classes: the target share of class 1, in [0, 1]. */
  prevalence?: number
  /** Any number of classes: one target weight per class (normalised); in place of `prevalence`. */
  weights?: readonly number[]
  /**
   * `subsample` (default): drop points of the over-represented classes, keeping as many points as the target allows.
   * `oversample`: keep every point and repeat draws (with replacement) of the under-represented classes.
   */
  method?: 'subsample' | 'oversample'
  /** The number of points to return; by default the most (subsample) or fewest (oversample) that hit the target. */
  n?: number
}

function targetWeights(target: number | readonly number[], k: number, what: string): number[] {
  if (typeof target === 'number') {
    if (k !== 2) throw new DomainError(what, `${what}: a single prevalence needs two classes; give ${k} weights`)
    if (!(target >= 0 && target <= 1)) throw new DomainError(what, `${what}: prevalence must be in [0, 1]`)
    return [1 - target, target]
  }
  if (target.length !== k) throw new ShapeError(what, `${what}: ${target.length} weights for ${k} classes`)
  const sum = target.reduce((a, b) => a + b, 0)
  if (!(sum > 0) || target.some((v) => !(v >= 0))) throw new DomainError(what, `${what}: weights must be non-negative`)
  return target.map((v) => v / sum)
}

function resample(s: Stream, d: Dataset, options: PrevalenceOptions, op: string) {
  const { method = 'subsample' } = options
  const target = options.weights ?? options.prevalence
  if (target === undefined) throw new DomainError(op, `${op}: give a prevalence or class weights`)
  const y = intLabels(d, op)
  const k = classCount(d, y)
  const pi = targetWeights(target, k, op)
  const have = countsOf(y, k)
  pi.forEach((p, j) => {
    if (p > 0 && have[j] === 0) throw new DomainError(op, `${op}: no points of class ${j} to resample`)
  })
  let counts: number[]
  if (method === 'subsample') {
    let total = options.n ?? Math.min(...pi.map((p, j) => (p > 0 ? Math.floor(have[j] / p) : Infinity)))
    counts = classCounts(total, pi)
    if (options.n !== undefined && counts.some((c, j) => c > have[j]))
      throw new DomainError(op, `${op}: not enough points for n = ${options.n} by subsampling; use method 'oversample'`)
    while (counts.some((c, j) => c > have[j])) counts = classCounts(--total, pi)
  } else {
    const total = options.n ?? Math.max(...pi.map((p, j) => (p > 0 ? Math.ceil(have[j] / p) : 0)))
    counts = classCounts(total, pi)
  }
  const byClass: number[][] = Array.from({ length: k }, () => [])
  y.forEach((v, i) => byClass[v].push(i))
  const index: number[] = []
  byClass.forEach((rows, j) => {
    const c = counts[j]
    const order = Array.from(permutation(child(s, 'class', j), rows.length).data)
    if (c <= rows.length)
      index.push(
        ...order
          .slice(0, c)
          .map((i) => rows[i])
          .sort((a, b) => a - b),
      )
    else {
      index.push(...rows)
      const extra = child(s, 'extra', j)
      for (let e = rows.length; e < c; e++) index.push(rows[integers(extra, rows.length)])
    }
  })
  const out = selectRows(d, index)
  const t = classTruth(d)
  let truth: Truth | undefined = d.meta.truth
  if (t) {
    // Resampling by observed label multiplies the joint p(x, ỹ = j) by wⱼ = πⱼ / prevalenceⱼ.
    const prev = t.prevalence
    const w = pi.map((p, j) => (prev[j] > 0 ? p / prev[j] : 0))
    const reference = lazy((): Reference => {
      const ref = t.model.reference()
      // The new marginal of x is p(x) Σⱼ wⱼ P(ỹ = j | x).
      const post = values(t.posterior(ref.x))
      const weights = ref.weights.map((v, i) => {
        let a = 0
        for (let j = 0; j < k; j++) if (!Number.isNaN(post[i * k + j])) a += post[i * k + j] * w[j]
        return v * a
      })
      const sum = weights.reduce((a, b) => a + b, 0)
      return { x: ref.x, weights: weights.map((v) => v / sum) }
    })
    truth = remodel(t, { ops: [...t.model.ops, { kind: 'weights', weights: w }], reference })
  }
  const sentence = `Resampled (${method}) to class shares ${pi.map((p) => percent(p)).join(' : ')}.`
  return { ...out, meta: step(out, op, { target: pi, method, n: index.length }, sentence, { truth }) }
}

/**
 * Change the class balance to `prevalence` (two classes: the share of class 1) or `weights` (one per class) by
 * resampling rows within each class, with exact counts by largest remainder. The truth becomes the posterior under
 * the new priors: P'(j | x) ∝ wⱼ P(j | x) with wⱼ = targetⱼ / prevalenceⱼ (Saerens et al., 2002, Neural Computation
 * 14(1)).
 */
export function withPrevalence(s: Stream, d: Dataset, options: PrevalenceOptions): Dataset {
  return resample(s, d, options, 'withPrevalence')
}

/**
 * Label shift, for a test split: p(y) changes while p(x | y) stays fixed. The same resampling as `withPrevalence`,
 * recorded as a shift.
 */
export function withLabelShift(s: Stream, d: Dataset, options: PrevalenceOptions): Dataset {
  return resample(s, d, options, 'withLabelShift')
}

// ── Outliers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `withOutliers`. */
export interface OutlierOptions {
  /** Fraction of rows replaced, in [0, 1]; the count is rounded. */
  fraction: number
  /** Outlier spread in units of each column's standard deviation. Default 4. */
  scale?: number
  /** Replace features (`x`, the default for labelled data) or targets (`y`, the default for real targets). */
  target?: 'x' | 'y'
}

/**
 * Gross outliers: a random `fraction` of rows get new values from a broad Gaussian centred on the data mean with
 * `scale` times each column's standard deviation, keeping their labels. `meta.outliers` marks them. For features,
 * the truth's class-conditional densities become the mixture (1 − ε) p(x | j) + ε q(x), a contamination model
 * (Huber, 1964, Annals of Mathematical Statistics 35(1)); for regression targets the regression function is unchanged
 * and `truth.outlierFraction` records ε.
 */
export function withOutliers(s: Stream, d: Dataset, options: OutlierOptions): Dataset {
  const { fraction, scale = 4 } = options
  if (!(fraction >= 0 && fraction <= 1))
    throw new DomainError('withOutliers', 'withOutliers: fraction must be in [0, 1]')
  const realTarget = d.y !== undefined && d.y.dtype !== 'int32'
  const target = options.target ?? (realTarget ? 'y' : 'x')
  const [n, dim] = d.x.shape
  const m = Math.round(fraction * n)
  const rows = Array.from(permutation(child(s, 'rows'), n).data).slice(0, m)
  const mask = d.meta.outliers ? Int32Array.from(values(d.meta.outliers)) : new Int32Array(n)
  rows.forEach((i) => (mask[i] = 1))
  const draw = child(s, 'points')
  const eps = n > 0 ? m / n : 0
  const sentence = `${m} ${target === 'x' ? 'points' : 'targets'} replaced by outliers (${scale} sd).`
  if (target === 'y') {
    if (!realTarget) throw new TypeError('withOutliers: target y needs real-valued targets')
    const y = values(d.y!)
    const { mean, sd } = columnStats(y, n, 1)
    rows.forEach((i) => (y[i] = mean[0] + scale * sd[0] * normal(draw)))
    const t = regTruth(d)
    return {
      ...d,
      y: vector(y),
      meta: step(d, 'withOutliers', { fraction, scale, target }, sentence, {
        outliers: labels(mask),
        truth: t ? remodelRegression(t, { outlierFraction: 1 - (1 - t.outlierFraction) * (1 - eps) }) : d.meta.truth,
      }),
    }
  }
  const x = values(d.x)
  const { mean, sd } = columnStats(x, n, dim)
  const spread = sd.map((v) => scale * (v || 1))
  for (const i of rows) for (let c = 0; c < dim; c++) x[i * dim + c] = mean[c] + spread[c] * normal(draw)
  const t = classTruth(d)
  let truth: Truth | undefined = t ? undefined : d.meta.truth
  if (t) {
    // The contaminating law q: independent normals at the column means, `scale` column sds wide.
    const q = MultivariateNormal(fromData(Float64Array.from(mean), [dim]), {
      covariance: fromData(
        Float64Array.from({ length: dim * dim }, (_, e) => (e % (dim + 1) === 0 ? spread[e / (dim + 1)] ** 2 : 0)),
        [dim, dim],
      ),
    })
    const [l1, l2] = [Math.log(1 - eps), Math.log(eps)]
    const base = t.model
    const k = base.classes
    const reference = lazy((): Reference => {
      const ref = base.reference()
      const r = child(s, 'reference')
      const extra = Math.round(REFERENCE_SIZE / 3)
      const m0 = ref.x.shape[0]
      const rowsOut = new Float64Array((m0 + extra) * dim)
      rowsOut.set(points(ref.x).data)
      for (let i = 0; i < extra; i++)
        for (let c = 0; c < dim; c++) rowsOut[(m0 + i) * dim + c] = mean[c] + spread[c] * normal(r)
      const weights = new Float64Array(m0 + extra)
      ref.weights.forEach((w, i) => (weights[i] = w * (1 - eps)))
      weights.fill(eps / extra, m0)
      return { x: pointsFrom(rowsOut, m0 + extra, dim), weights }
    })
    truth = remodel(t, {
      logDensity: (x) => {
        const l = values(base.logDensity(x))
        const lq = values(q.logProb(x) as Tensor)
        for (let i = 0; i < lq.length; i++)
          for (let j = 0; j < k; j++) l[i * k + j] = logAddExp(l1 + l[i * k + j], l2 + lq[i])
        return fromData(l, [lq.length, k])
      },
      reference,
      closedForm: undefined,
      family: `${base.family} with ${percent(eps)} outliers`,
    })
  }
  return {
    ...d,
    x: matrix(x, n, dim),
    meta: step(d, 'withOutliers', { fraction, scale, target }, sentence, { outliers: labels(mask), truth }),
  }
}

// ── Nuisance features ────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `withNuisanceFeatures`. */
export interface NuisanceOptions {
  /** How many features to append. */
  count: number
  /** `gaussian` (default): N(0, scale²); `uniform`: uniform on ±√3 · scale (same sd). */
  kind?: 'gaussian' | 'uniform'
  /** Standard deviation of each nuisance feature; by default the mean sd of the existing features. */
  scale?: number
}

/**
 * Append `count` features drawn independently of everything else, named `noise 1`, `noise 2`, …. They carry no
 * information about y, so the truth's posterior depends only on the original features and the Bayes error is
 * unchanged; a fitted model that uses them overfits.
 */
export function withNuisanceFeatures(s: Stream, d: Dataset, options: NuisanceOptions): Dataset {
  const { count, kind = 'gaussian' } = options
  if (!(Number.isInteger(count) && count >= 0))
    throw new DomainError('withNuisanceFeatures', 'withNuisanceFeatures: count must be an integer ≥ 0')
  const [n, d0] = d.x.shape
  const old = values(d.x)
  const scale = options.scale ?? (columnStats(old, n, d0).sd.reduce((a, b) => a + b, 0) / d0 || 1)
  const half = Math.sqrt(3) * scale
  const drawOne = (r: Stream) => (kind === 'gaussian' ? scale * normal(r) : half * (2 * uniform(r) - 1))
  const d1 = d0 + count
  const widenRows = (src: Float64Array, rows: number, fill: (i: number, c: number) => number) => {
    const out = new Float64Array(rows * d1)
    for (let i = 0; i < rows; i++) {
      out.set(src.subarray(i * d0, (i + 1) * d0), i * d1)
      for (let c = 0; c < count; c++) out[i * d1 + d0 + c] = fill(i, c)
    }
    return out
  }
  const widen = (src: Float64Array, fill: (i: number, c: number) => number) => widenRows(src, n, fill)
  const x = widen(old, () => drawOne(s))
  // The law of each nuisance feature; they are independent of each other and of (x, y).
  const nuisance = kind === 'gaussian' ? Normal(0, scale) : Uniform(-half, half)
  const logQ = (x: Tensor) => rowSums(nuisance.logProb(columns(x, d0, d1)) as Tensor)
  let truth: Truth | undefined = d.meta.truth
  const t = classTruth(d)
  if (t) {
    const base = t.model
    const k = base.classes
    truth = remodel(t, {
      logDensity: (x) => {
        const q = logQ(x)
        const l = values(base.logDensity(columns(x, 0, d0)))
        for (let i = 0; i < q.length; i++) for (let j = 0; j < k; j++) l[i * k + j] += q[i]
        return fromData(l, [q.length, k])
      },
      reference: lazy((): Reference => {
        const ref = base.reference()
        const r = child(s, 'reference')
        const m0 = ref.x.shape[0]
        return {
          x: pointsFrom(
            widenRows(points(ref.x).data, m0, () => drawOne(r)),
            m0,
            d1,
          ),
          weights: ref.weights,
        }
      }),
    })
  }
  const rt = regTruth(d)
  if (rt)
    truth = remodelRegression(rt, {
      mean: (p) => rt.model.mean(head(p, d0)),
      sdAt: (p) => rt.model.sdAt(head(p, d0)),
    })
  const extra: MetaEdit = {
    featureNames: [...d.meta.featureNames, ...Array.from({ length: count }, (_, c) => `noise ${c + 1}`)],
    truth,
  }
  if (d.meta.missing) {
    const mask = values(d.meta.missing)
    extra.missing = labels(widen(mask, () => 0))
  }
  if (d.meta.complete)
    extra.complete = matrix(
      widen(values(d.meta.complete), (i, c) => x[i * d1 + d0 + c]),
      n,
      d1,
    )
  return {
    ...d,
    x: matrix(x, n, d1),
    meta: step(
      d,
      'withNuisanceFeatures',
      { count, kind, scale },
      `${count} uninformative ${kind} features appended.`,
      extra,
    ),
  }
}

// ── Linear maps ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** The 2 × 2 rotation by `theta` radians anticlockwise. */
export function rotation2d(theta: number): number[][] {
  return [
    [Math.cos(theta), -Math.sin(theta)],
    [Math.sin(theta), Math.cos(theta)],
  ]
}

/** The 2 × 2 shear x₁ ↦ x₁ + k x₂. */
export function shear2d(k: number): number[][] {
  return [
    [1, k],
    [0, 1],
  ]
}

/** Inverse and log |det| of a small square matrix (`aifn-compute/numerics/linalg`); undefined when it is singular. */
function invert(a: readonly (readonly number[])[]): { inverse: number[][]; logAbsDet: number } | undefined {
  const d = a.length
  const m = fromData(Float64Array.from(a.flat()), [d, d])
  const logAbsDet = logDet(m) as number
  if (!Number.isFinite(logAbsDet)) return undefined
  const inv = values(inverse(m) as Tensor)
  return { inverse: Array.from({ length: d }, (_, r) => Array.from(inv.subarray(r * d, (r + 1) * d))), logAbsDet }
}

/** Options for `withTransform`: a matrix and offset, or a rotation, shear and stretch of the first two features. */
export interface TransformOptions {
  /** A (m × d). When given, `rotation`, `shear` and `stretch` are not used. */
  matrix?: readonly (readonly number[])[]
  /** b (length m). Default zero. */
  offset?: readonly number[]
  /** Without `matrix`: rotate the first two features by this angle (radians, anticlockwise). Default 0. */
  rotation?: number
  /** Without `matrix`: shear x₁ ↦ x₁ + k x₂ before the rotation. Default 0. */
  shear?: number
  /** Without `matrix`: scale x₁ by this factor before the shear. Default 1. */
  stretch?: number
}

/** The matrix of a `TransformOptions`: A itself, or R(rotation) · shear · diag(stretch, 1, …) on d features. */
function transformMatrix(options: TransformOptions, d: number): readonly (readonly number[])[] {
  if (options.matrix) return options.matrix
  const { rotation = 0, shear = 0, stretch = 1 } = options
  const a: number[][] = Array.from({ length: d }, (_, r) => Array.from({ length: d }, (_, c) => (r === c ? 1 : 0)))
  if (d < 2) {
    if (rotation !== 0 || shear !== 0)
      throw new DomainError('withTransform', 'withTransform: a rotation or shear needs two features')
    a[0][0] = stretch
    return a
  }
  const rs = rotation2d(rotation)
  const sh = shear2d(shear)
  // (R · S · D) restricted to the first two coordinates, D = diag(stretch, 1).
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 2; c++) a[r][c] = (rs[r][0] * sh[0][c] + rs[r][1] * sh[1][c]) * (c === 0 ? stretch : 1)
  return a
}

/**
 * The affine map x ↦ A x + b applied to every row (A is m × d, b has length m; b defaults to zero): rotations,
 * scalings and shears, given as a matrix or as a rotation, shear and stretch of the first two features. When A is
 * square and invertible the truth maps along: p'(x') = p(A⁻¹(x' − b)) / |det A|, so the posterior at x' is the old
 * posterior at A⁻¹(x' − b) and the Bayes error is unchanged; otherwise the truth is dropped.
 */
export function withTransform(d: Dataset, options: TransformOptions): Dataset {
  const [n, d0] = d.x.shape
  const a = transformMatrix(options, d0)
  const b = options.offset
  const m = a.length
  if (a.some((row) => row.length !== d0))
    throw new ShapeError('withTransform', `withTransform: A must have ${d0} columns`)
  const offset = b ?? new Array<number>(m).fill(0)
  if (offset.length !== m) throw new ShapeError('withTransform', `withTransform: b must have length ${m}`)
  const apply = (src: Float64Array) => {
    const out = new Float64Array(n * m)
    for (let i = 0; i < n; i++)
      for (let r = 0; r < m; r++) {
        let v = offset[r]
        for (let c = 0; c < d0; c++) v += a[r][c] * src[i * d0 + c]
        out[i * m + r] = v
      }
    return out
  }
  const x = apply(values(d.x))
  const inv = m === d0 ? invert(a) : undefined
  const back = inv
    ? (p: Row): Float64Array => {
        const out = new Float64Array(d0)
        for (let r = 0; r < d0; r++) {
          let v = 0
          for (let c = 0; c < d0; c++) v += inv.inverse[r][c] * (p[c] - offset[c])
          out[r] = v
        }
        return out
      }
    : undefined
  let truth: Truth | undefined
  const t = classTruth(d)
  const rt = regTruth(d)
  if (t && inv && back) {
    const base = t.model
    const forward = (row: Float64Array) => {
      const out = new Float64Array(m)
      for (let r = 0; r < m; r++) {
        let v = offset[r]
        for (let c = 0; c < d0; c++) v += a[r][c] * row[c]
        out[r] = v
      }
      return out
    }
    const mapRows = (x: Tensor, f: (row: Float64Array) => Float64Array, dOut: number) => {
      const { data, n: rows, d: dIn } = points(x)
      const out = new Float64Array(rows * dOut)
      for (let i = 0; i < rows; i++) out.set(f(data.subarray(i * dIn, (i + 1) * dIn)), i * dOut)
      return pointsFrom(out, rows, dOut)
    }
    truth = remodel(t, {
      logDensity: (x) => {
        const l = values(base.logDensity(mapRows(x, back, d0)))
        return fromData(
          l.map((v) => v - inv.logAbsDet),
          [l.length / base.classes, base.classes],
        )
      },
      reference: lazy((): Reference => {
        const ref = base.reference()
        return { x: mapRows(ref.x, forward, m), weights: ref.weights }
      }),
    })
  } else if (rt && back)
    truth = remodelRegression(rt, { mean: (p) => rt.model.mean(back(p)), sdAt: (p) => rt.model.sdAt(back(p)) })
  const extra: MetaEdit = {
    truth,
    featureNames: m === d0 ? d.meta.featureNames : Array.from({ length: m }, (_, r) => `z${r + 1}`),
  }
  if (d.meta.complete) extra.complete = matrix(apply(values(d.meta.complete)), n, m)
  if (d.meta.missing) extra.missing = labels(Int32Array.from(x, (v) => (Number.isNaN(v) ? 1 : 0)))
  const sentence = inv ? 'Mapped by an invertible linear transform.' : 'Mapped by a linear transform (truth dropped).'
  return {
    ...d,
    x: matrix(x, n, m),
    meta: step(d, 'withTransform', { matrix: a.map((r) => [...r]), offset: [...offset] }, sentence, extra),
  }
}

// ── Missing values ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Missingness mechanisms (Rubin, 1976, "Inference and missing data", Biometrika 63(3)). */
export type MissingMechanism = 'mcar' | 'mar' | 'mnar'

/** Options for `withMissing`. */
export interface MissingOptions {
  /** Expected fraction of entries removed in each affected feature. */
  rate: number
  /**
   * `mcar` (default): every entry independently with probability `rate`. `mar`: feature `observed` stays complete and
   * each other entry goes missing with probability σ(α + strength · z), z the standardised value of the observed
   * feature in that row. `mnar`: each entry goes missing with probability σ(α + strength · z), z its own standardised
   * value, so large values hide themselves. α is set so the mean rate is `rate`.
   */
  mechanism?: MissingMechanism
  /** How strongly missingness depends on a value (MAR and MNAR). Default 2. */
  strength?: number
  /** MAR: the always-observed feature. Default 0. */
  observed?: number
}

/**
 * Remove entries of `x` (set them to NaN). `meta.missing` marks them (n × d, 1 = missing) and `meta.complete` keeps
 * the values before removal, so a figure can show where the missing values were. The truth describes the complete
 * data and is unchanged.
 */
export function withMissing(s: Stream, d: Dataset, options: MissingOptions): Dataset {
  const { rate, mechanism = 'mcar', strength = 2, observed = 0 } = options
  if (!(rate >= 0 && rate <= 1)) throw new DomainError('withMissing', 'withMissing: rate must be in [0, 1]')
  const [n, dim] = d.x.shape
  const x = values(d.x)
  const complete = d.meta.complete ?? d.x
  const full = values(complete)
  const { mean, sd } = columnStats(full, n, dim)
  const z = (i: number, c: number) => (full[i * dim + c] - mean[c]) / (sd[c] || 1)
  const mask = d.meta.missing ? Int32Array.from(values(d.meta.missing)) : new Int32Array(n * dim)
  if (mechanism === 'mar' && dim < 2)
    throw new DomainError('withMissing', 'withMissing: MAR needs at least two features')
  for (let c = 0; c < dim; c++) {
    if (mechanism === 'mar' && c === observed) continue
    const r = child(s, 'feature', c)
    let p: (i: number) => number
    if (mechanism === 'mcar') p = () => rate
    else {
      const source = mechanism === 'mar' ? observed : c
      const logits = Array.from({ length: n }, (_, i) => strength * z(i, source))
      const alpha = calibrate(logits, rate)
      p = (i) => sigmoid(alpha + logits[i])
    }
    for (let i = 0; i < n; i++) if (uniform(r) < p(i)) mask[i * dim + c] = 1
  }
  mask.forEach((v, i) => v && (x[i] = NaN))
  const removed = mask.reduce((a, v) => a + v, 0)
  const names = { mcar: 'completely at random', mar: 'at random', mnar: 'not at random' }
  return {
    ...d,
    x: matrix(x, n, dim),
    meta: step(
      d,
      'withMissing',
      { rate, mechanism, strength, observed },
      `${removed} entries missing ${names[mechanism]} (${mechanism.toUpperCase()}).`,
      { missing: labels(mask), complete },
    ),
  }
}

// ── Covariate shift ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `withCovariateShift`. */
export interface CovariateShiftOptions {
  /** Selection strength along `direction`, per standard deviation. Default 1.5. */
  strength?: number
  /** Direction in feature space (standardised units). Default the first feature. */
  direction?: readonly number[]
  /** Expected fraction of rows kept. Default 0.5. */
  keep?: number
}

/**
 * Covariate shift by biased selection: row i is kept with probability σ(α + strength · uᵀzᵢ), zᵢ its standardised
 * features and u the unit `direction`, with α set so that a fraction `keep` survives (Shimodaira, 2000, Journal of
 * Statistical Planning and Inference 90(2)). Selection depends on x alone, so p(y | x) and the truth's posterior are
 * unchanged, while the marginal of x, the class shares and the Bayes error move.
 */
export function withCovariateShift(s: Stream, d: Dataset, options: CovariateShiftOptions = {}): Dataset {
  const { strength = 1.5, keep = 0.5 } = options
  const [n, dim] = d.x.shape
  const x = values(d.x)
  const { mean, sd } = columnStats(x, n, dim)
  const raw = options.direction ?? [1, ...new Array<number>(dim - 1).fill(0)]
  if (raw.length !== dim)
    throw new ShapeError('withCovariateShift', `withCovariateShift: direction must have ${dim} entries`)
  const norm = Math.hypot(...raw) || 1
  const u = raw.map((v) => v / norm)
  const projection = (p: Row) => {
    let v = 0
    for (let c = 0; c < dim; c++) {
      const zc = (p[c] - mean[c]) / (sd[c] || 1)
      if (!Number.isNaN(zc)) v += u[c] * zc
    }
    return strength * v
  }
  const logits = Array.from({ length: n }, (_, i) => projection(x.subarray(i * dim, (i + 1) * dim)))
  const alpha = calibrate(logits, keep)
  const selection = (p: Row) => sigmoid(alpha + projection(p))
  const index: number[] = []
  logits.forEach((l, i) => uniform(s) < sigmoid(alpha + l) && index.push(i))
  const out = selectRows(d, index)
  const t = classTruth(d)
  let truth: Truth | undefined = d.meta.truth
  if (t) {
    const base = t.model
    const k = base.classes
    const selections = (x: Tensor) => {
      const { data, n: rows } = points(x)
      return Float64Array.from({ length: rows }, (_, i) => selection(data.subarray(i * dim, (i + 1) * dim)))
    }
    truth = remodel(t, {
      // A term shared by every class: the posterior is unchanged, the densities describe the selected population.
      logDensity: (x) => {
        const sel = selections(x)
        const l = values(base.logDensity(x))
        for (let i = 0; i < sel.length; i++) for (let j = 0; j < k; j++) l[i * k + j] += Math.log(sel[i])
        return fromData(l, [sel.length, k])
      },
      reference: lazy((): Reference => {
        const ref = base.reference()
        const sel = selections(ref.x)
        const weights = ref.weights.map((w, i) => w * sel[i])
        const sum = weights.reduce((a, b) => a + b, 0)
        return { x: ref.x, weights: weights.map((w) => w / sum) }
      }),
      shifted: true,
    })
  }
  return {
    ...out,
    meta: step(
      out,
      'withCovariateShift',
      { strength, direction: u, keep, alpha, mean, sd },
      `Covariate shift: ${index.length} of ${n} rows kept with probability rising along the direction (${u.map((v) => +v.toFixed(2)).join(', ')}).`,
      { truth },
    ),
  }
}

// ── Train/test split ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `split`. */
export interface SplitOptions {
  /** Fraction of rows in the test set (rounded). Default 0.25. */
  test?: number
  /** Keep class shares equal in both parts, by exact per-class counts (largest remainder). Default true when labelled. */
  stratify?: boolean
  /** A shift applied to the test part only: covariate shift, or a label shift to new class shares. */
  shift?:
    | { covariate: CovariateShiftOptions }
    | (PrevalenceOptions & ({ prevalence: number } | { weights: readonly number[] }))
}

/**
 * Split a dataset into train and test parts, each keeping the original row order. Stratified by default (integer
 * labels): class j contributes its largest-remainder share of the test rows. With `shift`, the test part is shifted
 * (`withCovariateShift` or `withLabelShift`) so a figure can show a model trained on one distribution and tested on
 * another; the truth follows each part.
 */
export function split(s: Stream, d: Dataset, options: SplitOptions = {}): { train: Dataset; test: Dataset } {
  const { test = 0.25 } = options
  if (!(test >= 0 && test <= 1)) throw new DomainError('split', 'split: test must be in [0, 1]')
  const n = d.x.shape[0]
  const labelled = d.y !== undefined && d.y.dtype === 'int32'
  const stratify = options.stratify ?? labelled
  const total = Math.round(test * n)
  const isTest = new Uint8Array(n)
  if (stratify) {
    const y = intLabels(d, 'split')
    const k = classCount(d, y)
    const byClass: number[][] = Array.from({ length: k }, () => [])
    y.forEach((v, i) => byClass[v].push(i))
    const have = byClass.map((r) => r.length)
    const counts = total > 0 ? classCounts(total, have) : have.map(() => 0)
    byClass.forEach((rows, j) => {
      const order = Array.from(permutation(child(s, 'class', j), rows.length).data)
      order.slice(0, counts[j]).forEach((i) => (isTest[rows[i]] = 1))
    })
  } else
    Array.from(permutation(child(s, 'rows'), n).data)
      .slice(0, total)
      .forEach((i) => (isTest[i] = 1))
  const trainRows: number[] = []
  const testRows: number[] = []
  isTest.forEach((v, i) => (v ? testRows : trainRows).push(i))
  const part = (rows: number[], which: string): Dataset => {
    const out = selectRows(d, rows)
    return {
      ...out,
      meta: step(out, 'split', { part: which, test, stratify }, `The ${which} part (${rows.length} rows).`),
    }
  }
  let testPart = part(testRows, 'test')
  const shift = options.shift
  if (shift && 'covariate' in shift) testPart = withCovariateShift(child(s, 'shift'), testPart, shift.covariate)
  else if (shift) testPart = withLabelShift(child(s, 'shift'), testPart, shift)
  return { train: part(trainRows, 'train'), test: testPart }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const modifier = definer<ModifierInfo>('modifier', 'data/synthetic')

modifier(
  {
    key: 'withLabelNoise',
    name: 'Label noise',
    summary: 'Flip each label to another class at random with probability `rate`.',
    params: space({ rate: real(0, 0.5, { default: 0.1 }) }),
    needs: 'labels',
    random: true,
    notes: ['label-noise-models'],
  },
  withLabelNoise,
)

modifier(
  {
    key: 'withPrevalence',
    name: 'Prevalence',
    summary: 'Resample within classes to a target share of class 1.',
    params: space({ prevalence: real(0, 1, { default: 0.5 }), method: oneOf(['subsample', 'oversample']) }),
    needs: 'labels',
    random: true,
    notes: ['class-imbalance'],
  },
  withPrevalence,
)

modifier(
  {
    key: 'withLabelShift',
    name: 'Label shift',
    summary: 'Resample within classes to new class shares, keeping p(x | y): a label shift.',
    params: space({ prevalence: real(0, 1, { default: 0.5 }), method: oneOf(['subsample', 'oversample']) }),
    needs: 'labels',
    random: true,
    notes: ['dataset-shift'],
  },
  withLabelShift,
)

modifier(
  {
    key: 'withOutliers',
    name: 'Outliers',
    summary: 'Replace a fraction of rows by draws from a broad Gaussian around the data mean.',
    params: space({ fraction: real(0, 0.5, { default: 0.05 }), scale: real(1, 20, { default: 4 }) }),
    random: true,
    notes: ['z-score-and-robust-outlier-detection'],
  },
  withOutliers,
)

modifier(
  {
    key: 'withNuisanceFeatures',
    name: 'Nuisance features',
    summary: 'Append features drawn independently of everything else.',
    params: space({ count: int(0, 50, { default: 2 }), kind: oneOf(['gaussian', 'uniform']) }),
    random: true,
    notes: ['curse-of-dimensionality-for-neighbours'],
  },
  withNuisanceFeatures,
)

modifier(
  {
    key: 'withMissing',
    name: 'Missing values',
    summary: 'Remove entries completely at random, at random given another feature, or not at random.',
    params: space({
      rate: real(0, 0.9, { default: 0.1 }),
      mechanism: oneOf(['mcar', 'mar', 'mnar']),
      strength: real(0, 10, { default: 2 }),
      observed: int(0, 50, { default: 0, when: when('mechanism', 'mar') }),
    }),
    random: true,
    notes: ['missing-data-and-imputation'],
  },
  withMissing,
)

modifier(
  {
    key: 'withCovariateShift',
    name: 'Covariate shift',
    summary: 'Keep rows with a probability that rises along a direction in feature space.',
    params: space({ strength: real(0, 5, { default: 1.5 }), keep: real(0.05, 1, { default: 0.5 }) }),
    random: true,
    notes: ['dataset-shift', 'importance-weighting-for-covariate-shift'],
  },
  withCovariateShift,
)

modifier(
  {
    key: 'withTransform',
    name: 'Linear map',
    summary: 'Rotate, shear and stretch the first two features (or apply any affine map); the truth maps along.',
    params: space({
      rotation: real(-Math.PI, Math.PI, { default: 0, label: 'θ' }),
      shear: real(-3, 3, { default: 0 }),
      stretch: real(0.1, 10, { default: 1, scale: 'log' }),
    }),
    random: false,
  },
  withTransform,
)
