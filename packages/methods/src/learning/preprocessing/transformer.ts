/**
 * The shape of a preprocessing step: an estimator whose fitted model transforms inputs (and inverts, where an inverse
 * exists), plus shared helpers for numeric matrices.
 *
 * A transformer is unfitted: `fit({ x, y? })` returns a new fitted transform, a plain object with its fitted state as
 * public fields, as scikit-learn's `fit` and `transform` (Pedregosa et al., 2011). Numeric inputs are $n \times d$
 * matrices, rows the examples and columns the features, read as row-major values; a transform given a different number
 * of columns from the one it was fitted on throws `ShapeError`.
 */

import type { Model } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrixShape, type Estimator, type FitOptions, type Transforms } from 'aifn-compute/learning/estimators'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A fitted transform (a model): its fitted state as public fields, `transform`, and `inverseTransform` where one
 * exists.
 */
export interface FittedTransform<X = Tensor, Z = Tensor> extends Model, Transforms<X, Z> {
  /** Which transform this is, e.g. "standard-scaler". */
  readonly name: string
}

/** A fitted transform with an inverse: `inverseTransform(transform(x))` recovers `x` (up to rounding). */
export interface Invertible<X = Tensor, Z = Tensor> {
  /** Map transformed values `z` back to the input space. */
  inverseTransform(z: Z): X
}

/**
 * What a transformer fits on: a dataset's features `x` and optional targets `y` (any `Dataset`, or a plain `{ x }` for
 * inputs that are not a feature matrix, such as a list of categories).
 */
export type TransformerData<X> = { readonly x: X; readonly y?: Tensor }

/** An unfitted transform: `fit({ x, y? }, options)` returns the fitted transform. */
export type Transformer<X, F> = Estimator<TransformerData<X>, F>

/**
 * Fit a transformer and transform its training inputs in one call, as scikit-learn's `fit_transform`.
 *
 * @param transformer The unfitted transform, such as `standardScaler()`.
 * @param data What to fit on: the inputs `x` (and targets `y`, for a transform that uses them); `x` is then
 *   transformed.
 * @param options Passed to `fit`: the random stream of a randomised transform, and tracing.
 * @returns The fitted transform as `model`, and `z`, the training inputs transformed by it.
 *
 * @example Standardise a matrix and keep the fitted scaler
 * const x = tensor([[1, 10], [2, 20], [3, 60]])
 * const { model, z } = fitTransform(standardScaler(), { x })
 * print('mean =', model.mean)
 * print('z =', z)
 */
export function fitTransform<X, F extends FittedTransform<X, unknown>>(
  transformer: Transformer<X, F>,
  data: TransformerData<X>,
  options?: FitOptions,
): { model: F; z: ReturnType<F['transform']> } {
  const model = transformer.fit(data, options)
  return { model, z: model.transform(data.x) as ReturnType<F['transform']> }
}

/**
 * The elements of a tensor in row-major order (`aifn-compute/foundation/tensor`'s `dense.data`: shared with the tensor
 * when it is already dense float64, so not to be written).
 */
export const values = dense.data

/**
 * A numeric $n \times d$ matrix as its size and row-major values; throws `ShapeError` naming the caller for any other
 * shape.
 *
 * @param x The matrix, $n \times d$.
 * @param where The caller's name, for error messages.
 * @returns `n` rows, `d` columns and the row-major values `v` ($n d$ of them, possibly shared with `x`: not to be
 *   written).
 */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/**
 * Check that a transform input has the fitted number of columns, and throw `ShapeError` if it does not.
 *
 * @param d The number of columns of the input given.
 * @param fitted The number of columns the transform was fitted on.
 * @param where The caller's name, for error messages.
 *
 * @example A transform fitted on two columns refuses three
 * checkColumns(2, 2, 'scaler')
 * print('2 columns: ok')
 * try {
 *   checkColumns(3, 2, 'scaler')
 * } catch (e) {
 *   print('3 columns:', e.message)
 * }
 */
export function checkColumns(d: number, fitted: number, where: string): void {
  if (d !== fitted) throw new ShapeError(where, `${where}: fitted on ${fitted} columns, given ${d}`)
}

/**
 * Apply `f(value, column)` to every element of an $n \times d$ matrix, giving a new $n \times d$ matrix. Throws
 * `ShapeError` when `x` is not a matrix of `d` columns.
 *
 * @param x The input matrix, $n \times d$; not modified.
 * @param d The number of columns `x` must have (the fitted number).
 * @param where The caller's name, for error messages.
 * @param f The map of one element, given its value and its column index $j$ ($0 \le j < d$).
 * @returns The mapped matrix, $n \times d$.
 */
export function mapColumns(x: Tensor, d: number, where: string, f: (v: number, j: number) => number): Tensor {
  const m = matrix(x, where)
  checkColumns(m.d, d, where)
  return fromData(
    Float64Array.from(m.v, (v, k) => f(v, k % d)),
    [m.n, d],
  )
}

/**
 * Column $j$ of an $n \times d$ matrix, copied out of its row-major values.
 *
 * @param v The matrix's row-major values, $n d$ of them; not modified.
 * @param n The number of rows.
 * @param d The number of columns.
 * @param j The column to copy, $0 \le j < d$.
 * @returns The $n$ values of column $j$, in a new array.
 */
export function column(v: Float64Array, n: number, d: number, j: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = v[i * d + j]
  return out
}
