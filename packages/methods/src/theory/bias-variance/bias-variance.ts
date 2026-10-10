/**
 * The bias–variance decomposition by resampling (Geman, Bienenstock and Doursat, 1992): draw many training sets of $n$
 * points from one regression problem, fit the same model to each, and look at the fits at each $x$. The mean fit's
 * distance from the truth is the bias, the fits' spread around their mean the variance, and the expected squared error
 * at $x$ of a fresh noisy target is $\text{bias}^2 + \text{variance} + \sigma^2$. Averaged over $x$ uniform on
 * $[-1, 1]$, the three terms add up to the expected test error.
 *
 * Models: polynomials of a degree (least squares, optionally ridge) and $k$-nearest-neighbour regression, whose
 * complexity rises with the degree and falls with $k$. The problems and fits are the shared layer of
 * `aifn-methods/theory`: training sets come from `drawTrainingSet`, and the grid from `unitGrid`.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  applyWeights,
  drawTrainingSet,
  leastSquaresWeights,
  targetFunction,
  unitGrid,
  type TargetName,
} from '../regression'

/**
 * A model for the resampling study: a polynomial of `degree` fitted by least squares (ridge-penalised when `ridge`, the
 * penalty $\lambda$, is above its default 0), or $k$-nearest-neighbour regression averaging the `k` nearest training
 * targets.
 */
export type ResampledModel = { kind: 'polynomial'; degree: number; ridge?: number } | { kind: 'knn'; k: number }

/** Options of `biasVariance`. */
export interface BiasVarianceOptions {
  /** The regression function (default `sine`; see `targetFunction`). */
  target?: TargetName
  /** Noise standard deviation $\sigma$ (default 0.3). */
  noise?: number
  /** Training points per set (default 30). */
  n?: number
  /** The model fitted to every training set. */
  model: ResampledModel
  /** Training sets drawn (default 200). */
  repeats?: number
  /** Points of the evaluation grid on $[-1, 1]$ (default 101). */
  grid?: number
  /** Fits kept for drawing (default 20). */
  keep?: number
}

/** The decomposition at each grid point and averaged over the grid. */
export interface BiasVarianceResult {
  /** The evaluation grid on $[-1, 1]$. */
  readonly x: Float64Array
  /** The regression function $f$ on the grid. */
  readonly truth: Float64Array
  /** The mean fit over the training sets. */
  readonly mean: Float64Array
  /** $(\text{mean} - \text{truth})^2$ per grid point. */
  readonly bias2: Float64Array
  /** The fits' variance (divisor the number of sets) per grid point. */
  readonly variance: Float64Array
  /**
   * Grid averages: `bias2`, `variance`, `noise` $\sigma^2$ and `error`, their sum, the expected test error; and
   * `trainError`, the mean squared error of the fits on their own training sets.
   */
  readonly totals: { bias2: number; variance: number; noise: number; error: number; trainError: number }
  /** The first `keep` fits on the grid, row-major $[\text{keep}, \text{grid}]$. */
  readonly fits: Float64Array
  /** The training sets of the kept fits. */
  readonly trainingSets: { x: Float64Array; y: Float64Array }[]
}

/**
 * Polynomial features $1, x, \dots, x^d$ of the points ($x$ is on $[-1, 1]$, where monomials stay well scaled to
 * $d \approx 15$).
 *
 * @param x The points.
 * @param degree The degree $d$.
 * @returns The design, row-major $[n, d + 1]$: row `i` holds the powers $x_i^0, \dots, x_i^d$.
 */
function polynomialDesign(x: ArrayLike<number>, degree: number): Float64Array {
  const p = degree + 1
  const out = new Float64Array(x.length * p)
  for (let i = 0; i < x.length; i++) {
    let v = 1
    for (let j = 0; j < p; j++) {
      out[i * p + j] = v
      v *= x[i]
    }
  }
  return out
}

/**
 * Fit a model to $(x, y)$ and predict at `at`: least squares on polynomial features (minimum-norm when there are more
 * coefficients than points), or the mean target of the $k$ nearest training points ($k$ capped at $n$; ties in
 * distance go to the earlier point).
 *
 * @param model The model (see `ResampledModel`).
 * @param x The training inputs, $n$ values.
 * @param y The training targets, one per input.
 * @param at The points to predict at.
 * @returns The prediction at each point of `at`.
 *
 * @example A straight line and 3-nearest neighbours through five points
 * const x = new Float64Array([-1, -0.5, 0, 0.5, 1])
 * const y = new Float64Array([-1.1, -0.4, 0.1, 0.4, 1.0])
 * print('line:', fitAndPredict({ kind: 'polynomial', degree: 1 }, x, y, [-0.25, 0.75]))
 * print('3-NN:', fitAndPredict({ kind: 'knn', k: 3 }, x, y, [-0.25, 0.75]))
 */
export function fitAndPredict(
  model: ResampledModel,
  x: Float64Array,
  y: Float64Array,
  at: ArrayLike<number>,
): Float64Array {
  if (model.kind === 'polynomial') {
    const p = model.degree + 1
    const w = leastSquaresWeights(polynomialDesign(x, model.degree), x.length, p, y, model.ridge ?? 0)
    return applyWeights(polynomialDesign(at, model.degree), at.length, p, w)
  }
  const k = Math.min(model.k, x.length)
  return Float64Array.from(at, (q) => {
    const order = Array.from(x.keys()).sort((i, j) => Math.abs(x[i] - q) - Math.abs(x[j] - q))
    let s = 0
    for (let j = 0; j < k; j++) s += y[order[j]]
    return s / k
  })
}

/**
 * The bias–variance decomposition of one model by resampling training sets (see the file's introduction). Costs
 * `repeats` fits, each predicted on the grid and on its own training set.
 *
 * @param s The stream; training set $r$ is drawn from `child(s, 'set', r)`, so `s` is not advanced and the same `s`
 *   gives the same sets to every model.
 * @param options The problem, model and study size (see `BiasVarianceOptions`).
 * @returns The decomposition per grid point, its grid averages, and the first `keep` fits (see `BiasVarianceResult`).
 *
 * @example A rigid and a flexible polynomial on the same training sets
 * for (const degree of [1, 3, 9]) {
 *   const { totals } = biasVariance(stream(0), { model: { kind: 'polynomial', degree }, repeats: 50, keep: 0 })
 *   print('degree', degree, ' bias2 =', totals.bias2, ' variance =', totals.variance, ' error =', totals.error)
 * }
 */
export function biasVariance(s: Stream, options: BiasVarianceOptions): BiasVarianceResult {
  const { target = 'sine', noise = 0.3, n = 30, model, repeats = 200, grid = 101, keep = 20 } = options
  const f = targetFunction(target)
  const x = unitGrid(grid)
  const truth = Float64Array.from(x, f)
  const sum = new Float64Array(grid)
  const sumSq = new Float64Array(grid)
  const fits = new Float64Array(Math.min(keep, repeats) * grid)
  const trainingSets: { x: Float64Array; y: Float64Array }[] = []
  let trainError = 0
  for (let r = 0; r < repeats; r++) {
    const train = drawTrainingSet(child(s, 'set', r), target, n, noise)
    const pred = fitAndPredict(model, train.x, train.y, x)
    for (let i = 0; i < grid; i++) {
      sum[i] += pred[i]
      sumSq[i] += pred[i] * pred[i]
    }
    const fitted = fitAndPredict(model, train.x, train.y, train.x)
    let e = 0
    for (let i = 0; i < n; i++) e += (fitted[i] - train.y[i]) ** 2
    trainError += e / n / repeats
    if (r < keep) {
      fits.set(pred, r * grid)
      trainingSets.push(train)
    }
  }
  const mean = Float64Array.from(sum, (v) => v / repeats)
  const variance = Float64Array.from(sumSq, (v, i) => Math.max(0, v / repeats - mean[i] * mean[i]))
  const bias2 = Float64Array.from(mean, (m, i) => (m - truth[i]) ** 2)
  const avg = (a: Float64Array) => a.reduce((p, q) => p + q, 0) / a.length
  const b = avg(bias2)
  const v = avg(variance)
  return {
    x,
    truth,
    mean,
    bias2,
    variance,
    totals: { bias2: b, variance: v, noise: noise * noise, error: b + v + noise * noise, trainError },
    fits,
    trainingSets,
  }
}

/**
 * The decomposition across a range of complexities (polynomial degrees or $k$ values), one `biasVariance` study per
 * value. Every study draws from `child(s, 'sweep')`, so all values see the same training sets.
 *
 * @param s The stream; not advanced.
 * @param options The problem and study size (as `BiasVarianceOptions`, without `model` and `keep`), and the family.
 * @param options.family `polynomial` to sweep the degree, `knn` to sweep $k$.
 * @param options.values The degrees or $k$ values, one study each.
 * @param options.ridge The ridge penalty $\lambda$ of every polynomial fit (default 0).
 * @returns The `values`, and per value the grid averages `bias2`, `variance`, `error` (their sum with $\sigma^2$) and
 *   `trainError`.
 *
 * @example Bias falls with the degree until the variance of the flexible fits swamps it
 * const r = biasVarianceSweep(stream(0), { family: 'polynomial', values: [1, 3, 5, 9], repeats: 40 })
 * print('degree   ', r.values)
 * print('bias2    ', r.bias2)
 * print('variance ', r.variance)
 * print('train MSE', r.trainError)
 */
export function biasVarianceSweep(
  s: Stream,
  options: Omit<BiasVarianceOptions, 'model' | 'keep'> & {
    family: 'polynomial' | 'knn'
    values: readonly number[]
    ridge?: number
  },
): { values: number[]; bias2: Float64Array; variance: Float64Array; error: Float64Array; trainError: Float64Array } {
  const { family, values, ridge = 0, ...rest } = options
  const out = values.map(
    (v) =>
      biasVariance(child(s, 'sweep'), {
        ...rest,
        keep: 0,
        model: family === 'polynomial' ? { kind: 'polynomial', degree: v, ridge } : { kind: 'knn', k: v },
      }).totals,
  )
  return {
    values: [...values],
    bias2: Float64Array.from(out, (t) => t.bias2),
    variance: Float64Array.from(out, (t) => t.variance),
    error: Float64Array.from(out, (t) => t.error),
    trainError: Float64Array.from(out, (t) => t.trainError),
  }
}
