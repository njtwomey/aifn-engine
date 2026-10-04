/**
 * The shape of a preprocessing step: an estimator whose fitted model transforms inputs (and inverts, where an inverse
 * exists), plus shared helpers for numeric matrices.
 */

import type { Model } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrixShape, type Estimator, type FitOptions, type Transforms } from 'aifn-compute/learning/estimators'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A fitted transform (a model): its fitted state as public fields, `transform`, and `inverseTransform` where one exists. */
export interface FittedTransform<X = Tensor, Z = Tensor> extends Model, Transforms<X, Z> {
  /** Which transform this is, e.g. "standard-scaler". */
  readonly name: string
}

/** A fitted transform with an inverse: `inverseTransform(transform(x))` recovers x (up to rounding). */
export interface Invertible<X = Tensor, Z = Tensor> {
  inverseTransform(z: Z): X
}

/**
 * What a transformer fits on: a dataset's features `x` and optional targets `y` (any `Dataset`, or a plain `{ x }` for
 * inputs that are not a feature matrix, such as a list of categories).
 */
export type TransformerData<X> = { readonly x: X; readonly y?: Tensor }

/** An unfitted transform: `fit({ x, y? }, options)` returns the fitted transform. */
export type Transformer<X, F> = Estimator<TransformerData<X>, F>

/** Fit a transformer and transform its training inputs in one call. */
export function fitTransform<X, F extends FittedTransform<X, unknown>>(
  transformer: Transformer<X, F>,
  data: TransformerData<X>,
  options?: FitOptions,
): { model: F; z: ReturnType<F['transform']> } {
  const model = transformer.fit(data, options)
  return { model, z: model.transform(data.x) as ReturnType<F['transform']> }
}

/** The elements of `t` in row-major order (`aifn-compute/foundation/tensor`'s `dense.data`: shared when already dense float64). */
export const values = dense.data

/** A numeric matrix [n, d] as its size and row-major values; throws naming the caller otherwise. */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/** Check that a transform input has the fitted number of columns. */
export function checkColumns(d: number, fitted: number, where: string): void {
  if (d !== fitted) throw new ShapeError(where, `${where}: fitted on ${fitted} columns, given ${d}`)
}

/** Apply f(value, column) to every element of an [n, d] matrix. */
export function mapColumns(x: Tensor, d: number, where: string, f: (v: number, j: number) => number): Tensor {
  const m = matrix(x, where)
  checkColumns(m.d, d, where)
  return fromData(
    Float64Array.from(m.v, (v, k) => f(v, k % d)),
    [m.n, d],
  )
}

/** Column j of an [n, d] matrix's values. */
export function column(v: Float64Array, n: number, d: number, j: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = v[i * d + j]
  return out
}
