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

const outputs = (model: ScalarModel, rows: Float64Array, m: Size, d: Size): Float64Array => {
  const o = model(fromData(rows, [m, d]))
  return 'shape' in o ? Float64Array.from(dense.data(o as Tensor)) : Float64Array.from(o)
}

/** Options of `permutationImportance`. */
export type PermutationImportanceOptions = {
  /** Shuffles per feature (default 5). */
  repeats?: Size
  /** Stream for the shuffles (required unless `permutations` is given). */
  stream?: Stream
  /** Explicit shuffles [feature][repeat][n] of row indices, each applied to the column as left by the last. */
  permutations?: readonly (readonly (readonly number[])[])[]
}

/**
 * Permutation importance of each feature of X [n, d] for `model` under `score(y, predicted)` (higher is better): the
 * baseline score minus the score with the feature's column shuffled, per repeat [d][repeats], with mean and standard
 * deviation (population, as scikit-learn).
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
 * Grid values for a feature: its distinct values when there are at most `resolution` of them, else `resolution`
 * evenly spaced values between its 5th and 95th percentiles (linear interpolation between order statistics).
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
 * Partial dependence of `model` on `feature` over the rows of X [n, d]: for each grid value g (default
 * `featureGrid(X, feature)`), every row's prediction with x_feature = g (the ICE curves [n, grid]) and their mean
 * (the partial dependence [grid]).
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
