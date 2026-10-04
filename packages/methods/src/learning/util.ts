/**
 * Private helpers shared by the learning area's classifiers and regressors: input, label and weight checks, and the
 * probability-model builder that gives a classifier its capabilities (`forward`, `score`, `predictive`, `decide`).
 * Numerical helpers come from compute: `dense.data` for float64 views, `argmax` for decisions, `softmax` and `sigmoid`
 * from `aifn-compute/numerics/special`.
 */

import type { AnyUnivariate } from 'aifn-compute/foundation/contracts'
import { argmax, dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { bernoulliPredictive, categoricalPredictive, matrixShape } from 'aifn-compute/learning/estimators'
import { softmax } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The elements of a tensor in row-major order as float64 (`dense.data`: shared when already dense; do not mutate). */
export const values = (t: Tensor): Float64Array => dense.data(t)

/** Rows, columns and values of a matrix [n, d] (values shared when already dense float64; do not mutate). */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/**
 * Integer class labels 0 … K−1 from a vector [n] (or [n, 1]); K is one more than the largest label (at least 2 unless
 * `minClasses` says otherwise).
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

/** Real-valued targets [n] as a Float64Array. */
export function targets(y: Tensor, n: number, where: string): Float64Array {
  const v = dense.data(y)
  if (v.length !== n) throw new ShapeError(where, `${where}: ${n} rows of x but ${v.length} targets`)
  return v
}

/** Sample weights (default all 1), checked to be non-negative and finite. */
export function sampleWeights(w: Tensor | undefined, n: number, where: string): Float64Array {
  if (!w) return new Float64Array(n).fill(1)
  const v = dense.data(w)
  if (v.length !== n) throw new ShapeError(where, `${where}: ${n} rows but ${v.length} sample weights`)
  for (const u of v)
    if (!(u >= 0 && Number.isFinite(u))) throw new DomainError(where, `${where}: sample weights must be ≥ 0`)
  return v
}

/** Checks that a prediction input has the fitted number of features. */
export function inputs(x: Tensor, d: number, where: string): { n: number; v: Float64Array } {
  const m = matrix(x, where)
  if (m.d !== d) throw new ShapeError(where, `${where}: fitted on ${d} features, given ${m.d}`)
  return { n: m.n, v: m.v }
}

/** The predictive law of a classifier from class probabilities [m, K]: Bernoulli over class 1 for K = 2, else Categorical. */
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
 * The capabilities of a classifier whose class probabilities are computed from a head [m, K] (log-probabilities,
 * scores or probabilities): `forward` and `score` return the head, `predictive` the class law (Bernoulli for two
 * classes, else Categorical), `decide` the most probable class (ties to the lowest label).
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

/** Row-wise softmax of scores [m, K] as class probabilities (`aifn-compute/numerics/special`'s `softmax`). */
export function softmaxRows(s: Float64Array, m: number, k: number): Float64Array {
  return dense.data(softmax(fromData(s, [m, k])))
}

/** A float64 matrix tensor over `v`. */
export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])

/** An int32 vector tensor. */
export const ints = (v: ArrayLike<number>): Tensor => fromData(Int32Array.from(v), [v.length])

/** A float64 vector tensor. */
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
