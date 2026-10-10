/**
 * Modifiers: pure functions from a `Dataset` to a `Dataset` that change one thing about a dataset (its labels, class
 * balance, features or rows) and record themselves in `meta.recipe`, plus `split` into a train and a test part. Random
 * ones take a stream first. None modifies its input. Each keeps `meta.truth` consistent where it can: label noise
 * composes the noise matrix into the Bayes posterior, prevalence changes reweight it, covariate shift leaves it
 * unchanged (only the marginal of $\xvec$ moves), and a map of the features maps it along.
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

/**
 * A function that calls `f` once, on first use, and returns the cached value after that.
 *
 * @param f Computes the value.
 * @returns A function returning `f()`, computed once.
 */
function lazy<T>(f: () => T): () => T {
  let cached: { value: T } | undefined
  return () => (cached ??= { value: f() }).value
}

/**
 * The metadata of a modified dataset: the old metadata with `meta` merged in, `sentence` appended to the description,
 * and the step appended to the recipe.
 *
 * @param d The dataset before the modifier; its metadata is copied, not modified.
 * @param op The modifier's registry key, recorded in the recipe step.
 * @param params The modifier's parameters, recorded in the recipe step as plain data.
 * @param sentence One sentence saying what the modifier did, appended to the description.
 * @param meta Metadata fields the modifier sets (labels, masks, the truth), overriding the old ones.
 * @returns The new metadata.
 */
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

/**
 * A copy of a dataset's integer class labels. Throws `TypeError` when `y` is missing or not int32.
 *
 * @param d The dataset.
 * @param what The caller's name, for the error message.
 * @returns The labels, a new array.
 */
function intLabels(d: Dataset, what: string): Int32Array {
  if (!d.y || d.y.dtype !== 'int32') throw new TypeError(`${what}: needs integer class labels y`)
  return Int32Array.from(values(d.y))
}

/**
 * The number of classes: from the truth, the label names or the largest label, in that order of preference.
 *
 * @param d The dataset, whose truth and label names are read.
 * @param y Its integer labels, used when neither says.
 * @returns The number of classes $k$.
 */
function classCount(d: Dataset, y: Int32Array): number {
  const t = d.meta.truth
  if (t?.task === 'classification' && 'classes' in t) return t.classes
  if (d.meta.labelNames) return d.meta.labelNames.length
  return y.reduce((a, b) => Math.max(a, b), -1) + 1
}

/**
 * The number of labels of each class.
 *
 * @param y The integer labels, each in $0, \dots, k - 1$.
 * @param k The number of classes.
 * @returns $k$ counts.
 */
function countsOf(y: Int32Array, k: number): number[] {
  const c = new Array<number>(k).fill(0)
  y.forEach((v) => c[v]++)
  return c
}

/**
 * Column means and standard deviations, ignoring NaN (missing) entries. The standard deviation divides by the count
 * less one (at least 1); a column with no values has mean 0.
 *
 * @param x The matrix, row-major: $n \times d$ values, not modified.
 * @param n The number of rows.
 * @param d The number of columns.
 * @returns `mean` and `sd`, $d$ values each.
 */
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

/**
 * The intercept $\alpha$ with $\frac1n \sum_i \sigma(\alpha + z_i)$ equal to the target, by 100 steps of bisection
 * on $[-60, 60]$ (the mean is increasing in $\alpha$).
 *
 * @param z The logits $z_i$ of every row.
 * @param target The mean probability wanted, in $[0, 1]$.
 * @returns The intercept $\alpha$: $-\infty$ for a target of 0 or less, $\infty$ for 1 or more.
 */
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

/**
 * The first $d$ entries of a point.
 *
 * @param x The point; returned as is when it has exactly $d$ entries.
 * @param d The number of entries kept.
 * @returns The point's first $d$ entries.
 */
function head(x: Row, d: number): Row {
  return x.length === d ? x : Float64Array.from({ length: d }, (_, i) => x[i])
}

/**
 * Columns `from` to `to - 1` of a batch of points.
 *
 * @param x The points, $n \times d$.
 * @param from The first column kept.
 * @param to One past the last column kept.
 * @returns The points restricted to those columns, $n \times (\mathrm{to} - \mathrm{from})$.
 */
function columns(x: Tensor, from: number, to: number): Tensor {
  const { data, n, d } = points(x)
  const out = new Float64Array(n * (to - from))
  for (let i = 0; i < n; i++) out.set(data.subarray(i * d + from, i * d + to), i * (to - from))
  return pointsFrom(out, n, to - from)
}

/**
 * $\log(e^a + e^b)$, computed without overflow.
 *
 * @param a The first log value.
 * @param b The second log value.
 * @returns $\log(e^a + e^b)$, or $-\infty$ when both are $-\infty$.
 */
function logAddExp(a: number, b: number): number {
  const m = Math.max(a, b)
  return m === -Infinity ? -Infinity : m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}

/**
 * The sum of each row of an $n \times m$ tensor.
 *
 * @param t The tensor, $n \times m$.
 * @returns The $n$ row sums.
 */
function rowSums(t: Tensor): Float64Array {
  const [n, m] = t.shape
  const v = values(t)
  return Float64Array.from({ length: n }, (_, i) => v.subarray(i * m, (i + 1) * m).reduce((a, b) => a + b, 0))
}

/**
 * The dataset's classification truth when it has class densities that a modifier can edit.
 *
 * @param d The dataset.
 * @returns The truth, or `undefined` when there is none or it is of another kind (a regime truth has no class
 *   densities, so modifiers leave it alone).
 */
function classTruth(d: Dataset): ClassificationTruth | undefined {
  const t = d.meta.truth
  // A regime truth (`regimeTruth`) is a classification truth without class densities: modifiers leave it alone.
  return t?.task === 'classification' && 'priors' in t ? t : undefined
}

/**
 * A Gaussian regression truth (an additive GAM truth has its own family, which these modifiers do not edit).
 *
 * @param d The dataset.
 * @returns The truth, or `undefined` when the dataset has no Gaussian regression truth.
 */
function regTruth(d: Dataset): RegressionTruth | undefined {
  const t = d.meta.truth
  return t?.task === 'regression' && 'noiseSd' in t ? t : undefined
}

/**
 * A new truth from an edited model.
 *
 * @param t The truth to edit; not modified.
 * @param edit The fields of its class model to replace.
 * @returns The classification truth of the edited model.
 */
function remodel(t: ClassificationTruth, edit: Partial<ClassModel>): ClassificationTruth {
  return classificationTruth({ ...t.model, ...edit })
}

/**
 * A share as a percentage to one decimal place, for descriptions.
 *
 * @param v The share, 1 for all.
 * @returns The text, as `12.5%`.
 */
function percent(v: number): string {
  return `${+(100 * v).toFixed(1)}%`
}

// ── Label noise ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Label noise: `{ rate }` flips each label with probability `rate` to one of the other classes chosen uniformly
 * (symmetric noise); `{ matrix }` gives class-conditional noise, `matrix[i][j]` the probability of observing $j$ for
 * label $i$, rows summing to one.
 */
export type LabelNoiseOptions = { rate: number } | { matrix: readonly (readonly number[])[] }

/**
 * The noise matrix of symmetric label noise at `rate` over `k` classes: $1 - r$ on the diagonal and $r/(k - 1)$
 * elsewhere. Throws `DomainError` unless the rate is in $[0, 1]$.
 *
 * @param rate The probability $r$ that a label is flipped.
 * @param k The number of classes.
 * @returns The $k \times k$ matrix as rows, entry $(i, j)$ the probability of observing $j$ for label $i$.
 *
 * @example Three classes at rate 0.3
 * const m = symmetricNoise(0.3, 3)
 * print('matrix:', m)
 * print('row sums:', m.map((row) => row.reduce((a, v) => a + v, 0)))
 */
export function symmetricNoise(rate: number, k: number): number[][] {
  if (!(rate >= 0 && rate <= 1))
    throw new DomainError('symmetricNoise', `label noise rate must be in [0, 1], got ${rate}`)
  return Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 - rate : rate / (k - 1))))
}

/**
 * Flip labels at random. Each observed label $y_i$ becomes $j$ with probability $M_{y_i j}$ ($\Mmat$ the noise
 * matrix), one uniform draw per row from `s`. `meta.cleanLabels` keeps the labels before the first flip, so
 * `flippedMask` shows which changed. The truth's posterior becomes
 * $P(\tilde y = j \mid \xvec) = \sum_i P(y = i \mid \xvec) M_{ij}$ (Natarajan et al., 2013, "Learning with noisy
 * labels", NeurIPS), and its Bayes error follows. Throws `TypeError` without integer labels, `ShapeError` when the
 * matrix is not $k \times k$, and `DomainError` when a row is not a probability vector.
 *
 * @param s The random stream; one uniform is drawn per row.
 * @param d The labelled dataset; not modified.
 * @param options A flip `rate` for symmetric noise, or the noise `matrix`; see `LabelNoiseOptions`.
 * @returns The dataset with the noisy labels, the clean ones in `meta.cleanLabels` and the truth of the noisy labels.
 *
 * @example Symmetric noise raises the Bayes error
 * const d = blobs(stream(0), { n: 1000, centers: 2, separation: 2 })
 * const noisy = withLabelNoise(stream(1), d, { rate: 0.2 })
 * print('first labels:', toArray(d.y).slice(0, 8), ' noisy:', toArray(noisy.y).slice(0, 8))
 * print('flipped share:', flippedMask(noisy).reduce((a, v) => a + v, 0) / 1000)
 * // With rate r over two classes the Bayes error becomes r + (1 - 2r) e: 0.2 + 0.6 * 0.1587 = 0.2952.
 * print('Bayes error:', d.meta.truth.bayesError, '->', noisy.meta.truth.bayesError)
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

/**
 * 1 where the observed label differs from the clean one (`meta.cleanLabels`); all 0 without label noise. Throws
 * `TypeError` without integer labels.
 *
 * @param d The labelled dataset.
 * @returns $n$ flags (int32), one per row.
 *
 * @example Class-conditional noise flips only class 1
 * const d = blobs(stream(0), { n: 400, centers: 2 })
 * const noisy = withLabelNoise(stream(1), d, { matrix: [[1, 0], [0.4, 0.6]] })
 * const flipped = flippedMask(noisy)
 * const y = toArray(d.y)
 * print('flipped in class 0:', flipped.filter((f, i) => f && y[i] === 0).length, 'of 200')
 * print('flipped in class 1:', flipped.filter((f, i) => f && y[i] === 1).length, 'of 200')
 */
export function flippedMask(d: Dataset): Int32Array {
  const y = intLabels(d, 'flippedMask')
  const clean = d.meta.cleanLabels ? values(d.meta.cleanLabels) : undefined
  return Int32Array.from(y, (v, i) => (clean && clean[i] !== v ? 1 : 0))
}

// ── Prevalence and label shift ───────────────────────────────────────────────────────────────────────────────────────

/** Options for `withPrevalence` and `withLabelShift`. */
export interface PrevalenceOptions {
  /** Two classes: the target share of class 1, in $[0, 1]$. */
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

/**
 * The target class shares from a prevalence or weights, normalised. Throws `DomainError` for a single prevalence with
 * other than two classes, a prevalence outside $[0, 1]$, or negative weights or a non-positive sum, and `ShapeError`
 * for the wrong number of weights.
 *
 * @param target The share of class 1 (two classes), or one non-negative weight per class.
 * @param k The number of classes.
 * @param what The caller's name, for error messages.
 * @returns The $k$ target shares, summing to one.
 */
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

/**
 * Resample the rows within each class to target class shares, with exact counts by largest remainder, and reweight the
 * truth. A class that needs no more rows than it has keeps a random subset of them, in their original order; one that
 * needs more keeps them all and repeats random draws of them. The result lists the rows class by class. Throws
 * `DomainError` when no target is given, when a class with a positive target has no rows, or when subsampling cannot
 * reach the asked `n`.
 *
 * @param s The random stream: class $j$'s subset is drawn from its child `class` $j$, its repeats from `extra` $j$.
 * @param d The labelled dataset; not modified.
 * @param options The target, the method and the number of rows; see `PrevalenceOptions`.
 * @param op The calling modifier's name, for error messages and the recipe step.
 * @returns The resampled dataset, with the truth's priors reweighted.
 */
function resample(s: Stream, d: Dataset, options: PrevalenceOptions, op: string) {
  const { method = 'subsample' } = options
  const target = options.weights ?? options.prevalence ?? (options as any).target
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
  const recordedParams: Record<string, unknown> = {
    method,
    ...(options.prevalence !== undefined
      ? { prevalence: options.prevalence }
      : options.weights !== undefined
        ? { weights: Array.from(options.weights) }
        : pi.length === 2
          ? { prevalence: pi[1] }
          : { weights: pi }),
  }
  return { ...out, meta: step(out, op, recordedParams, sentence, { truth }) }
}

/**
 * Change the class balance to `prevalence` (two classes: the share of class 1) or `weights` (one per class) by
 * resampling rows within each class, with exact counts by largest remainder. The rows come out class by class. The
 * truth becomes the posterior under the new priors: $P'(j \mid \xvec) \propto w_j P(j \mid \xvec)$ with
 * $w_j = \pi'_j/\pi_j$, the target share over the old one (Saerens et al., 2002, Neural Computation 14(1)). Throws
 * `DomainError` when no target is given or it cannot be met, and `TypeError` without integer labels.
 *
 * @param s The random stream the resampling draws from.
 * @param d The labelled dataset; not modified.
 * @param options The target shares, the method and the number of rows; see `PrevalenceOptions`.
 * @returns The resampled dataset with the reweighted truth.
 *
 * @example Down to 20% positives, by subsampling or oversampling
 * const d = blobs(stream(0), { n: 200, centers: 2 })
 * const counts = (e) => [0, 1].map((j) => toArray(e.y).filter((v) => v === j).length)
 * print('before:', counts(d))
 * print('subsample:', counts(withPrevalence(stream(1), d, { prevalence: 0.2 })))
 * const over = withPrevalence(stream(1), d, { prevalence: 0.2, method: 'oversample' })
 * print('oversample:', counts(over), ' truth prevalence:', over.meta.truth.prevalence)
 */
export function withPrevalence(s: Stream, d: Dataset, options: PrevalenceOptions): Dataset {
  return resample(s, d, options, 'withPrevalence')
}

/**
 * Label shift, for a test split: $p(y)$ changes while $p(\xvec \mid y)$ stays fixed. The same resampling as
 * `withPrevalence`, recorded as a shift.
 *
 * @param s The random stream the resampling draws from.
 * @param d The labelled dataset; not modified.
 * @param options The new class shares, the method and the number of rows; see `PrevalenceOptions`.
 * @returns The shifted dataset, whose truth has the new priors.
 *
 * @example The posterior moves with the priors
 * const d = blobs(stream(0), { n: 400, centers: 2, separation: 2 })
 * const shifted = withLabelShift(stream(1), d, { weights: [1, 3] })
 * print('class counts:', [0, 1].map((j) => toArray(shifted.y).filter((v) => v === j).length))
 * // Midway between the blobs the posterior of class 1 goes from 1/2 to 3/4.
 * const mid = tensor([[0, 0]])
 * const p1 = (e) => toArray(e.meta.truth.posterior(mid))[0][1]
 * print('P(class 1 | midpoint):', p1(d), '->', p1(shifted))
 */
export function withLabelShift(s: Stream, d: Dataset, options: PrevalenceOptions): Dataset {
  return resample(s, d, options, 'withLabelShift')
}

// ── Outliers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `withOutliers`. */
export interface OutlierOptions {
  /** Fraction of rows replaced, in $[0, 1]$; the count is rounded. */
  fraction: number
  /** Outlier spread in units of each column's standard deviation. Default 4. */
  scale?: number
  /** Replace features (`x`, the default for labelled data) or targets (`y`, the default for real targets). */
  target?: 'x' | 'y'
}

/**
 * Gross outliers: a random `fraction` of rows get new values from a broad Gaussian centred on the data mean with
 * `scale` times each column's standard deviation (a column of sd 0 counts as 1), keeping their labels.
 * `meta.outliers` marks them (and earlier outliers). For features, the truth's class-conditional densities become the
 * mixture $(1 - \varepsilon) p(\xvec \mid j) + \varepsilon q(\xvec)$, a contamination model (Huber, 1964, Annals of
 * Mathematical Statistics 35(1)); for regression targets the regression function is unchanged and
 * `truth.outlierFraction` records $\varepsilon$. Throws `DomainError` unless the fraction is in $[0, 1]$, and
 * `TypeError` when targets `y` are asked of a dataset without real-valued targets.
 *
 * @param s The random stream: the rows come from its child `rows`, the new values from `points`.
 * @param d The dataset; not modified.
 * @param options The fraction of rows, the spread and what to replace; see `OutlierOptions`.
 * @returns The dataset with the outliers, marked in `meta.outliers` (int32, 1 for an outlier), and the contaminated
 *   truth.
 *
 * @example A tenth of the points replaced
 * const d = blobs(stream(0), { n: 200, centers: [[-2, 0], [2, 0]] })
 * const o = withOutliers(stream(1), d, { fraction: 0.1 })
 * const marked = toArray(o.meta.outliers)
 * const y = toArray(d.y)
 * print('outliers:', marked.reduce((a, v) => a + v, 0), ' labels kept:', toArray(o.y).every((v, i) => v === y[i]))
 * // x2 has sd about 1, so the outliers spread about 4 times as far.
 * const rms = (rows) => Math.sqrt(rows.reduce((a, r) => a + r[1] ** 2, 0) / rows.length)
 * const x = toArray(o.x)
 * print('rms x2 of inliers:', rms(x.filter((_, i) => !marked[i])), ' of outliers:', rms(x.filter((_, i) => marked[i])))
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
  /**
   * `gaussian` (default): $\Gauss(0, \sigma^2)$; `uniform`: uniform on $[-\sqrt3\,\sigma, \sqrt3\,\sigma]$ (the same
   * sd), $\sigma$ the `scale`.
   */
  kind?: 'gaussian' | 'uniform'
  /** Standard deviation of each nuisance feature; by default the mean sd of the existing features. */
  scale?: number
}

/**
 * Append `count` features drawn independently of everything else, named `noise 1`, `noise 2` and so on. They carry no
 * information about $y$, so the truth's posterior depends only on the original features and the Bayes error is
 * unchanged; a fitted model that uses them overfits. The masks `meta.missing` and `meta.complete` are widened to match
 * (the new entries are never missing). Throws `DomainError` unless `count` is a non-negative integer.
 *
 * @param s The random stream the new features are drawn from; its child `reference` widens the truth's reference
 *   sample.
 * @param d The dataset; not modified.
 * @param options The number of features, their law and their spread; see `NuisanceOptions`.
 * @returns The dataset with the new columns after the old ones, and the truth widened to match.
 *
 * @example Three uninformative columns
 * const d = blobs(stream(0), { n: 500, centers: 2, separation: 2 })
 * const wide = withNuisanceFeatures(stream(1), d, { count: 3, scale: 2 })
 * print('x:', d.x.shape, '->', wide.x.shape, ' names:', wide.meta.featureNames)
 * const x = toArray(wide.x)
 * print('sd of the first new column:', Math.sqrt(x.reduce((a, r) => a + r[2] ** 2, 0) / x.length))
 * print('Bayes error:', d.meta.truth.bayesError, '->', wide.meta.truth.bayesError)
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
    extra.missing = fromData(Int32Array.from(widen(mask, () => 0)), [n, d1])
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

/**
 * The $2 \times 2$ rotation by $\theta$ radians anticlockwise,
 * $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.
 *
 * @param theta The angle $\theta$, in radians.
 * @returns The matrix, as two rows.
 *
 * @example A turn by 30 degrees
 * const r = rotation2d(Math.PI / 6)
 * print('matrix:', r, ' cos(pi / 6) = 0.866')
 * // The first column is where (1, 0) goes.
 * print('(1, 0) goes to', [r[0][0], r[1][0]])
 */
export function rotation2d(theta: number): number[][] {
  return [
    [Math.cos(theta), -Math.sin(theta)],
    [Math.sin(theta), Math.cos(theta)],
  ]
}

/**
 * The $2 \times 2$ shear $x_1 \mapsto x_1 + k x_2$, which leaves $x_2$ as it is.
 *
 * @param k The shear factor $k$.
 * @returns The matrix $\begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$, as two rows.
 *
 * @example Shear by one half
 * const m = shear2d(0.5)
 * print('matrix:', m)
 * print('(0, 2) goes to', [m[0][0] * 0 + m[0][1] * 2, m[1][0] * 0 + m[1][1] * 2])
 */
export function shear2d(k: number): number[][] {
  return [
    [1, k],
    [0, 1],
  ]
}

/**
 * Inverse and $\log\lvert\det\Amat\rvert$ of a small square matrix (`aifn-compute/numerics/linalg`); `undefined`
 * when it is singular.
 *
 * @param a The square matrix $\Amat$, as rows.
 * @returns `inverse`, $\Amat^{-1}$ as rows, and `logAbsDet`, or `undefined` when $\Amat$ is singular.
 */
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
  /** The matrix $\Amat$ ($m \times d$), as rows. When given, `rotation`, `shear` and `stretch` are not used. */
  matrix?: readonly (readonly number[])[]
  /** The offset $\bvec$ (length $m$). Default zero. */
  offset?: readonly number[]
  /** Without `matrix`: rotate the first two features by this angle (radians, anticlockwise). Default 0. */
  rotation?: number
  /** Without `matrix`: shear $x_1 \mapsto x_1 + k x_2$ ($k$ this value) before the rotation. Default 0. */
  shear?: number
  /** Without `matrix`: scale $x_1$ by this factor before the shear. Default 1. */
  stretch?: number
}

/**
 * The matrix of a `TransformOptions`: $\Amat$ itself, or $\Rmat\Smat\Dmat$ on the first two of $d$ features (the
 * identity on the rest), $\Rmat$ the rotation, $\Smat$ the shear and $\Dmat = \diag(\text{stretch}, 1)$. With one
 * feature only the stretch applies; a rotation or shear then throws `DomainError`.
 *
 * @param options The matrix, or the rotation, shear and stretch.
 * @param d The number of features $d$.
 * @returns The matrix, as rows: `options.matrix` itself when given, else $d \times d$.
 */
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
 * The affine map $\xvec \mapsto \Amat\xvec + \bvec$ applied to every row ($\Amat$ is $m \times d$, $\bvec$ has
 * length $m$ and defaults to zero): rotations, scalings and shears, given as a matrix or as a rotation, shear and
 * stretch of the first two features. When $\Amat$ is square and invertible the truth maps along:
 * $p'(\xvec') = p(\Amat^{-1}(\xvec' - \bvec))/\lvert\det\Amat\rvert$, so the posterior at $\xvec'$ is the old
 * posterior at $\Amat^{-1}(\xvec' - \bvec)$ and the Bayes error is unchanged; otherwise the truth is dropped. A
 * missing entry makes every entry of its mapped row missing. With $m \ne d$ the features are renamed `z1`, `z2` and so
 * on. Throws `ShapeError` when $\Amat$ or $\bvec$ has the wrong size.
 *
 * @param d The dataset; not modified.
 * @param options The matrix and offset, or the rotation, shear and stretch; see `TransformOptions`.
 * @returns The dataset with the mapped features, and the mapped truth.
 *
 * @example A quarter turn moves the second blob onto the second axis
 * const d = blobs(stream(0), { n: 400, centers: [[0, 0], [3, 0]] })
 * const turned = withTransform(d, { rotation: Math.PI / 2 })
 * const y = toArray(turned.y)
 * const blob = toArray(turned.x).filter((_, i) => y[i] === 1)
 * print('mean of blob 2:', [0, 1].map((c) => blob.reduce((a, r) => a + r[c], 0) / blob.length))
 * print('Bayes error:', d.meta.truth.bayesError, '->', turned.meta.truth.bayesError)
 *
 * @example A matrix and an offset
 * const d = blobs(stream(0), { n: 4, centers: [[0, 0]], sd: 0.1 })
 * const m = withTransform(d, { matrix: [[2, 0], [0, 1]], offset: [10, 0] })
 * print('before:', toArray(d.x))
 * print('after: ', toArray(m.x))
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
  if (d.meta.missing)
    extra.missing = fromData(
      Int32Array.from(x, (v) => (Number.isNaN(v) ? 1 : 0)),
      [n, m],
    )
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
   * each other entry goes missing with probability $\sigma(\alpha + s z)$ ($s$ the `strength`), $z$ the standardised
   * value of the observed feature in that row. `mnar`: each entry goes missing with probability $\sigma(\alpha + s z)$,
   * $z$ its own standardised value, so large values hide themselves. $\alpha$ is set per feature so the mean rate is
   * `rate`.
   */
  mechanism?: MissingMechanism
  /** How strongly missingness depends on a value (MAR and MNAR). Default 2. */
  strength?: number
  /** MAR: the always-observed feature. Default 0. */
  observed?: number
}

/**
 * Remove entries of `x` (set them to NaN). `meta.missing` marks them (int32, 1 = missing; stored as an $n \times d$
 * matrix) and `meta.complete` keeps the values before removal, so a figure can show where the missing values were.
 * Applied again, it adds to the earlier missing entries, and standardises by the complete values. The truth describes
 * the complete data and is unchanged. Throws `DomainError` unless the rate is in $[0, 1]$, or for `mar` with fewer than two features.
 *
 * @param s The random stream; feature $c$'s entries are drawn from its child `feature` $c$.
 * @param d The dataset; not modified.
 * @param options The rate, mechanism, strength and the always-observed feature; see `MissingOptions`.
 * @returns The dataset with NaN for the missing entries, and the masks `meta.missing` and `meta.complete`.
 *
 * @example Missing completely at random, and not at random
 * const d = blobs(stream(0), { n: 1000, centers: [[0, 0]] })
 * const mcar = withMissing(stream(1), d, { rate: 0.2 })
 * print('first rows:', toArray(mcar.x).slice(0, 3))
 * print('missing share:', toArray(mcar.meta.missing).reduce((a, v) => a + v, 0) / 2000)
 * // Under MNAR the large values hide themselves: the missing values of x1 are mostly above its mean of 0.
 * const mnar = withMissing(stream(1), d, { rate: 0.2, mechanism: 'mnar' })
 * const full = toArray(mnar.meta.complete).map((r) => r[0])
 * const gone = toArray(mnar.meta.missing).filter((_, e) => e % 2 === 0)
 * const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length
 * print('mean x1 where missing:', mean(full.filter((_, i) => gone[i])))
 * print('mean x1 where observed:', mean(full.filter((_, i) => !gone[i])))
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
      { missing: fromData(mask, [n, dim]), complete },
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
 * Covariate shift by biased selection: row $i$ is kept with probability
 * $\sigma(\alpha + s\,\uvec^\top\zvec_i)$ ($s$ the `strength`), $\zvec_i$ its standardised features (missing ones
 * count as 0) and $\uvec$ the unit `direction`, with $\alpha$ set so that a fraction `keep` survives on average
 * (Shimodaira, 2000, Journal of Statistical Planning and Inference 90(2)). Selection depends on $\xvec$ alone, so
 * $p(y \mid \xvec)$ and the truth's posterior are unchanged, while the marginal of $\xvec$, the class shares and the
 * Bayes error move. Throws `ShapeError` when `direction` has the wrong length.
 *
 * @param s The random stream; one uniform is drawn per row.
 * @param d The dataset; not modified.
 * @param options The strength, direction and kept fraction; see `CovariateShiftOptions`.
 * @returns The kept rows, in their original order, with the truth of the selected population.
 *
 * @example Keeping more points on the right
 * const d = blobs(stream(0), { n: 1000, centers: 2, separation: 2 })
 * const shifted = withCovariateShift(stream(1), d)
 * const x1 = (e) => toArray(e.x).map((r) => r[0])
 * const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length
 * print('rows kept:', shifted.x.shape[0], 'of 1000')
 * print('mean x1:', mean(x1(d)), '->', mean(x1(shifted)))
 * print('class counts:', [0, 1].map((j) => toArray(shifted.y).filter((v) => v === j).length))
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
  /**
   * Keep class shares equal in both parts, by exact per-class counts (largest remainder). Default true when the labels
   * are integers.
   */
  stratify?: boolean
  /** A shift applied to the test part only: covariate shift, or a label shift to new class shares. */
  shift?:
    | { covariate: CovariateShiftOptions }
    | (PrevalenceOptions & ({ prevalence: number } | { weights: readonly number[] }))
}

/**
 * Split a dataset into train and test parts, each keeping the original row order. Stratified by default (integer
 * labels): class $j$ contributes its largest-remainder share of the test rows. With `shift`, the test part is shifted
 * (`withCovariateShift` or `withLabelShift`) so a figure can show a model trained on one distribution and tested on
 * another; the truth follows each part. Throws `DomainError` unless `test` is in $[0, 1]$.
 *
 * @param s The random stream: class $j$'s test rows come from its child `class` $j$ (stratified) or all of them from
 *   `rows`, and the shift from `shift`.
 * @param d The dataset; not modified.
 * @param options The test fraction, stratification and test shift; see `SplitOptions`.
 * @returns The two parts, `train` and `test`.
 *
 * @example A stratified split, then one with a shifted test part
 * const d = blobs(stream(0), { n: 200, centers: 2, classWeights: [3, 1] })
 * const counts = (e) => [0, 1].map((j) => toArray(e.y).filter((v) => v === j).length)
 * const { train, test } = split(stream(1), d)
 * print('train:', counts(train), ' test:', counts(test))
 * const shifted = split(stream(1), d, { shift: { prevalence: 0.5 } })
 * print('shifted test:', counts(shifted.test))
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
