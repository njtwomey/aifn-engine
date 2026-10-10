/**
 * Internal helpers shared by the learning area's classifiers and regressors: input, label and weight checks, and the
 * probability-model builder that gives a classifier its capabilities (`forward`, `score`, `predictive`, `decide`).
 *
 * Matrices are row-major, $n \times d$ with one row per example. Every check throws `ShapeError` or `DomainError`
 * naming the caller (the `where` argument) instead of returning NaN. The numerics come from compute: `dense.data` for
 * float64 views, `argmax` for decisions, and `softmax` from `aifn-compute/numerics/special`.
 */

import type { AnyUnivariate } from 'aifn-compute/foundation/contracts'
import { argmax, dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { bernoulliPredictive, categoricalPredictive, matrixShape } from 'aifn-compute/learning/estimators'
import { softmax } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The elements of a tensor in row-major order as float64 (`dense.data`: shared when already dense; do not mutate).
 *
 * @param t The tensor to read.
 * @returns Its elements, row-major; the tensor's own buffer when it is already dense float64, so not to be written.
 */
export const values = (t: Tensor): Float64Array => dense.data(t)

/**
 * Rows, columns and values of an $n \times d$ matrix (values shared when already dense float64; do not mutate). Throws
 * `ShapeError` when `x` is not two-dimensional.
 *
 * @param x The matrix, one row per example.
 * @param where The caller's name, for error messages.
 * @returns `n` rows, `d` columns and the $nd$ values `v`, row-major.
 */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/**
 * Integer class labels $0, \dots, K - 1$ from a vector of $n$ labels (or an $n \times 1$ matrix); $K$ is one more
 * than the largest label, and at least `minClasses`. Throws `ShapeError` for any other shape or a count other than
 * `n`, and `DomainError` for a label that is negative or not an integer.
 *
 * @param y The labels, one per row of the data.
 * @param n The number of rows of the data, which `y` must match.
 * @param where The caller's name, for error messages.
 * @param minClasses The smallest $K$ returned, whatever the labels reach (2, so a one-class sample is still binary).
 * @returns The labels `y` as integers and the number of classes `k`.
 */
export function classLabels(y: Tensor, n: number, where: string, minClasses = 2): { y: Int32Array; k: number } {
  if (y.shape.length > 2 || (y.shape.length === 2 && y.shape[1] !== 1)) {
    throw new ShapeError(where, `${where}: expected labels [n], got shape [${y.shape.join(', ')}]`)
  }
  const v = dense.data(y)
  if (v.length !== n) throw new ShapeError(where, `${where}: ${n} rows of x but ${v.length} labels`)
  const out = new Int32Array(n)
  let k = 0
  for (let i = 0; i < n; i++) {
    if (!(Number.isInteger(v[i]) && v[i] >= 0))
      throw new DomainError(where, `${where}: labels must be integers 0 … K−1`)
    out[i] = v[i]
    k = Math.max(k, v[i] + 1)
  }
  return { y: out, k: Math.max(k, minClasses) }
}

/**
 * Real-valued targets as a Float64Array. Only the count is checked (`ShapeError` when it is not `n`), not the shape.
 *
 * @param y The targets, $n$ values.
 * @param n The number of rows of the data, which `y` must match.
 * @param where The caller's name, for error messages.
 * @returns The $n$ targets, shared with `y` when it is dense float64; not to be written.
 */
export function targets(y: Tensor, n: number, where: string): Float64Array {
  const v = dense.data(y)
  if (v.length !== n) throw new ShapeError(where, `${where}: ${n} rows of x but ${v.length} targets`)
  return v
}

/**
 * Sample weights (default all 1), checked to be non-negative and finite (`DomainError` otherwise, NaN included).
 *
 * @param w The weights, one per row, or undefined for unit weights.
 * @param n The number of rows of the data, which `w` must match (`ShapeError` otherwise).
 * @param where The caller's name, for error messages.
 * @returns The $n$ weights, shared with `w` when it is dense float64; not to be written.
 */
export function sampleWeights(w: Tensor | undefined, n: number, where: string): Float64Array {
  if (!w) return new Float64Array(n).fill(1)
  const v = dense.data(w)
  if (v.length !== n) throw new ShapeError(where, `${where}: ${n} rows but ${v.length} sample weights`)
  for (const u of v)
    if (!(u >= 0 && Number.isFinite(u))) throw new DomainError(where, `${where}: sample weights must be ≥ 0`)
  return v
}

/**
 * Checks that a prediction input is a matrix with the fitted number of features (`ShapeError` otherwise).
 *
 * @param x The query matrix, $m \times d$.
 * @param d The number of features the model was fitted on.
 * @param where The caller's name, for error messages.
 * @returns The number of query rows `n` and their values `v`, row-major.
 */
export function inputs(x: Tensor, d: number, where: string): { n: number; v: Float64Array } {
  const m = matrix(x, where)
  if (m.d !== d) throw new ShapeError(where, `${where}: fitted on ${d} features, given ${m.d}`)
  return { n: m.n, v: m.v }
}

/**
 * The predictive law of a classifier from class probabilities: Bernoulli over class 1 for $K = 2$, else Categorical.
 *
 * @param probs The class probabilities, $m \times K$ row-major; each row sums to 1.
 * @param m The number of rows.
 * @param k The number of classes $K$.
 * @returns A batch of $m$ Bernoulli laws (the probability of class 1) when $K = 2$, else of $m$ Categorical laws.
 */
export function classPredictive(probs: Float64Array, m: number, k: number): AnyUnivariate {
  if (k === 2)
    return bernoulliPredictive(
      fromData(
        Float64Array.from({ length: m }, (_, i) => probs[2 * i + 1]),
        [m],
      ),
    )
  return categoricalPredictive(fromData(probs, [m, k]))
}

/**
 * The capabilities of a classifier whose class probabilities are computed from an $m \times K$ head
 * (log-probabilities, scores or probabilities): `forward` and `score` return the head, `predictive` the class law
 * (Bernoulli for two classes, else Categorical), `decide` the most probable class (ties to the lowest label).
 *
 * @param head The model's head on a query matrix: $m \times K$ values, row-major.
 * @param toProbs Turns a head of $m$ rows into class probabilities, $m \times K$ row-major (the identity when the head
 *   already is probabilities).
 * @param k The number of classes $K$.
 * @returns The `forward`, `score`, `predictive` and `decide` methods of the fitted model.
 */
export function probabilityModel(
  head: (x: Tensor) => Float64Array,
  toProbs: (h: Float64Array, m: number) => Float64Array,
  k: number,
) {
  const rows = (x: Tensor) => x.shape[0]
  const forward = (x: Tensor): Tensor => fromData(head(x), [rows(x), k])
  const probabilities = (x: Tensor): Float64Array => toProbs(head(x), rows(x))
  return {
    forward,
    score: forward,
    predictive: (x: Tensor): AnyUnivariate => classPredictive(probabilities(x), rows(x), k),
    decide: (x: Tensor): Tensor => argmax(fromData(probabilities(x), [rows(x), k]), -1),
  }
}

/**
 * Row-wise softmax of scores as class probabilities (`aifn-compute/numerics/special`'s `softmax`).
 *
 * @param s The scores, $m \times K$ row-major.
 * @param m The number of rows.
 * @param k The number of classes $K$.
 * @returns The $m \times K$ probabilities, row-major; each row sums to 1.
 */
export function softmaxRows(s: Float64Array, m: number, k: number): Float64Array {
  return dense.data(softmax(fromData(s, [m, k])))
}

/**
 * A float64 matrix tensor over `v` (shared, not copied).
 *
 * @param v The values, $n \times d$ row-major.
 * @param n The number of rows.
 * @param d The number of columns.
 * @returns The $n \times d$ tensor.
 */
export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])

/**
 * An int32 vector tensor (a copy).
 *
 * @param v The values, converted to 32-bit integers.
 * @returns The vector.
 */
export const ints = (v: ArrayLike<number>): Tensor => fromData(Int32Array.from(v), [v.length])

/**
 * A float64 vector tensor (a copy).
 *
 * @param v The values.
 * @returns The vector.
 */
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
