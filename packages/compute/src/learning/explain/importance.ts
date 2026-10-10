/**
 * Model-agnostic global explanations from predictions alone:
 *
 * - `permutationImportance` (Breiman, 2001; Fisher, Rudin and Dominici, 2019): the drop in a score when one feature's
 *   column is shuffled, breaking its link to the target, averaged over repeats; as scikit-learn's
 *   `permutation_importance` (which permutes the already-permuted column on each repeat).
 * - `partialDependence` (Friedman, 2001): the mean prediction with one feature set to each grid value for every row,
 *   and the individual conditional expectation curves (Goldstein et al., 2015) it averages.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { quantile } from 'aifn-compute/probability/stats'
import type { ScalarModel } from './shapley'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A model's outputs on a batch of rows, as a fresh array.
 *
 * @param model The model, called once on the $m \times d$ batch.
 * @param rows The rows, row-major ($m \times d$ values).
 * @param m The number of rows.
 * @param d The number of features.
 * @returns The $m$ outputs.
 */
const outputs = (model: ScalarModel, rows: Float64Array, m: Size, d: Size): Float64Array => {
  const o = model(fromData(rows, [m, d]))
  return 'shape' in o ? Float64Array.from(dense.data(o as Tensor)) : Float64Array.from(o)
}

/** Options of `permutationImportance`. */
export type PermutationImportanceOptions = {
  /** Shuffles per feature (default 5; ignored when `permutations` is given). */
  repeats?: Size
  /** Stream for the shuffles (required unless `permutations` is given). */
  stream?: Stream
  /**
   * Explicit shuffles of row indices, indexed `[feature][repeat]`, each a permutation of the $n$ rows applied to the
   * column as left by the last; the number of repeats is that of feature 0.
   */
  permutations?: readonly (readonly (readonly number[])[])[]
}

/**
 * Permutation importance of each feature for `model` under a score (higher is better): the baseline score minus the
 * score with the feature's column shuffled, per repeat, with the mean and the standard deviation (population, as
 * scikit-learn) over repeats. One model call per feature and repeat, plus one. Throws `DomainError` when neither a
 * `stream` nor `permutations` is given.
 *
 * @param model The model, called on the $n \times d$ data with one column shuffled.
 * @param X The data $\Xmat$ ($n \times d$); not modified.
 * @param y The targets ($n$ values), passed to `score`.
 * @param score The score of predictions against the targets, `score(y, predicted)`, higher being better (a negated
 *   error, an accuracy, an $R^2$).
 * @param options The number of repeats and the stream or explicit permutations.
 * @returns `baseline`, the score on the unshuffled data; `importances`, the drops per feature and repeat ($d$ arrays of
 *   `repeats` values); and their `mean` and `std` per feature ($d$ values each).
 *
 * @example An unused feature has no importance
 * const model = (X) => matmul(X, tensor([3, 0]))
 * const X = [[0, 5], [1, 4], [2, 3], [3, 2], [4, 1], [5, 0]]
 * const y = [0, 3, 6, 9, 12, 15]
 * const negMse = (y, p) => -y.reduce((a, v, i) => a + (v - p[i]) ** 2, 0) / y.length
 * const r = permutationImportance(model, X, y, negMse, { repeats: 3, stream: stream(0) })
 * print('baseline =', r.baseline)
 * print('mean =', r.mean, ' std =', r.std)
 */
export function permutationImportance(
  model: ScalarModel,
  X: MatrixLike,
  y: VectorLike,
  score: (y: Float64Array, predicted: Float64Array) => number,
  options: PermutationImportanceOptions = {},
): { baseline: number; importances: Float64Array[]; mean: Float64Array; std: Float64Array } {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'permutationImportance')
  const yv = Float64Array.from(dense.toF64(y, 'permutationImportance'))
  const repeats = options.permutations ? options.permutations[0].length : (options.repeats ?? 5)
  if (!options.permutations && !options.stream)
    throw new DomainError('permutationImportance', 'permutationImportance: needs a stream')
  const baseline = score(yv, outputs(model, Float64Array.from(data), n, d))
  const importances: Float64Array[] = []
  for (let j = 0; j < d; j++) {
    const rows = Float64Array.from(data)
    const drops = new Float64Array(repeats)
    for (let r = 0; r < repeats; r++) {
      const order = options.permutations
        ? options.permutations[j][r]
        : Array.from(toFlat(permutation(options.stream as Stream, n)))
      const column = Float64Array.from({ length: n }, (_, i) => rows[i * d + j])
      for (let i = 0; i < n; i++) rows[i * d + j] = column[order[i]]
      drops[r] = baseline - score(yv, outputs(model, rows, n, d))
    }
    importances.push(drops)
  }
  const mean = Float64Array.from(importances, (v) => v.reduce((a, b) => a + b, 0) / repeats)
  const std = Float64Array.from(importances, (v, j) =>
    Math.sqrt(v.reduce((a, b) => a + (b - mean[j]) ** 2, 0) / repeats),
  )
  return { baseline, importances, mean, std }
}

/**
 * Grid values for a feature: its distinct values in increasing order when there are at most `resolution` of them,
 * else `resolution` evenly spaced values between two of its percentiles (by default the 5th and 95th, by linear
 * interpolation between order statistics).
 *
 * @param X The data ($n \times d$).
 * @param feature The column whose values are gridded.
 * @param resolution The largest number of grid values.
 * @param percentiles The lower and upper ends of an evenly spaced grid, as fractions in $[0, 1]$.
 * @returns The grid values.
 *
 * @example Few distinct values are kept; many are spread between percentiles
 * const X = [[1, 0], [1, 10], [2, 20], [3, 30], [3, 40], [3, 100]]
 * print('feature 0:', featureGrid(X, 0, 5))
 * print('feature 1:', featureGrid(X, 1, 5))
 * print('feature 1, full range:', featureGrid(X, 1, 5, [0, 1]))
 */
export function featureGrid(
  X: MatrixLike,
  feature: Size,
  resolution = 100,
  percentiles: [number, number] = [0.05, 0.95],
) {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'featureGrid')
  const col = Array.from({ length: n }, (_, i) => data[i * d + feature]).sort((a, b) => a - b)
  const distinct = [...new Set(col)]
  if (distinct.length <= resolution) return Float64Array.from(distinct)
  const [a, b] = [quantile(col, percentiles[0]), quantile(col, percentiles[1])]
  return Float64Array.from({ length: resolution }, (_, k) => a + ((b - a) * k) / (resolution - 1))
}

/**
 * Partial dependence of `model` on `feature` over the rows of $\Xmat$: for each grid value $g$, every row's prediction
 * with $x_{\text{feature}} = g$ (the ICE curves) and their mean (the partial dependence). One model call, on all
 * $n \times \lvert\text{grid}\rvert$ modified rows. Throws `DomainError` when `feature` is not a column of $\Xmat$.
 *
 * @param model The model, called on the batch of modified rows.
 * @param X The data ($n \times d$) the curves are averaged over; not modified.
 * @param feature The column varied.
 * @param options The grid.
 * @param options.grid The values the feature is set to (default `featureGrid(X, feature, resolution)`).
 * @param options.resolution The largest number of grid values when `grid` is not given (default 100).
 * @returns `grid`, the values; `average`, the partial dependence at each; and `individual`, the ICE curves
 *   ($n \times \lvert\text{grid}\rvert$, row $i$ for data row $i$).
 *
 * @example An interaction shows in the ICE curves but averages out
 * // f = x0 * x1: the slope in x0 is x1, which is -1 or 1 across the rows.
 * const model = (X) => toArray(X).map(([a, b]) => a * b)
 * const r = partialDependence(model, [[0, -1], [0, 1]], 0, { grid: [0, 1, 2] })
 * print('average =', r.average)
 * print('individual =', r.individual)
 */
export function partialDependence(
  model: ScalarModel,
  X: MatrixLike,
  feature: Size,
  options: { grid?: VectorLike; resolution?: Size } = {},
): { grid: Float64Array; average: Float64Array; individual: Tensor } {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'partialDependence')
  if (!(feature >= 0 && feature < d))
    throw new DomainError('partialDependence', `partialDependence: feature ${feature} out of range`)
  const grid = options.grid
    ? Float64Array.from(dense.toF64(options.grid, 'partialDependence'))
    : featureGrid(X, feature, options.resolution ?? 100)
  const g = grid.length
  const rows = new Float64Array(n * g * d)
  for (let k = 0; k < g; k++)
    for (let i = 0; i < n; i++) {
      rows.set(data.subarray(i * d, (i + 1) * d), (k * n + i) * d)
      rows[(k * n + i) * d + feature] = grid[k]
    }
  const out = outputs(model, rows, n * g, d)
  const ice = new Float64Array(n * g)
  const average = new Float64Array(g)
  for (let k = 0; k < g; k++)
    for (let i = 0; i < n; i++) {
      ice[i * g + k] = out[k * n + i]
      average[k] += out[k * n + i] / n
    }
  return { grid, average, individual: fromData(ice, [n, g]) }
}
