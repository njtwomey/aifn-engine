/**
 * Binary decomposition of ordinal regression (Frank & Hall, 2001, "A simple approach to ordinal classification",
 * ECML): K − 1 independent probabilistic classifiers qₖ(x) ≈ P(y > k | x), k = 0 … K − 2, each trained on the binary
 * target 1[y > k], and class probabilities by differencing: P(y = 0) = 1 − q₀, P(y = k) = q_{k−1} − qₖ,
 * P(y = K − 1) = q_{K−2}. The classifiers are independent, so the qₖ need not decrease in k and a difference can be
 * negative; `coherence` repairs it by clipping negatives to zero and renormalising (`clip`, the default) or by sorting
 * the qₖ into decreasing order first (`sort`). Neither makes the model coherent; `exceedance` returns the raw qₖ.
 *
 * The default classifier is L2-penalised logistic regression fitted by the generalised models' IRLS (binomial family,
 * logit link; one definition with `logisticRegression`); any estimator whose model predicts a Bernoulli law (its mean
 * is qₖ) can be passed as `base`.
 */

import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import {
  categoricalPredictive,
  dataset,
  defineModel,
  matrixShape,
  targetValues,
  withExpectation,
  withSampling,
  type AnyUnivariate,
  type Decides,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Scores,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import { sigmoid } from 'aifn-compute/numerics/special'
import { binomialFamily, link } from 'aifn-compute/probability/likelihoods'
import { irls } from '../irls'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A fitted binary classifier of the decomposition: P(y > k | x) for rows x [m, d]. */
type Exceedance = (x: Tensor) => Float64Array

/** An estimator usable as the base classifier: its model predicts a Bernoulli law whose mean is P(target = 1). */
export type BinaryBase = Estimator<Supervised<Tensor, Tensor>, Fitted<Tensor, Tensor> & Predicts<Tensor, AnyUnivariate>>

/** Hyperparameters of `binaryDecomposition`. */
export type BinaryDecompositionParams = {
  /** The base classifier (default: logistic regression with `l2`). */
  base?: BinaryBase
  /** L2 penalty λ of the default logistic classifiers, on the weights only. Default 1. */
  l2?: number
  /** How negative differences are repaired: `clip` (default) or `sort`. */
  coherence?: 'clip' | 'sort'
  /** Number of classes K (default: the largest label + 1). */
  classes?: number
  /** Most IRLS steps of each default classifier (default 100). */
  maxSteps?: number
}

/** A fitted binary decomposition. */
export interface BinaryDecompositionModel
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Expects<Tensor>,
    Samples<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'binary-decomposition'
  readonly classes: number
  readonly coherence: 'clip' | 'sort'
  /** The raw classifier outputs qₖ(x) ≈ P(y > k | x) [m, K − 1], not repaired. */
  exceedance(x: Tensor): Tensor
  /** Class probabilities P(y = k | x) [m, K], repaired by `coherence`. */
  probabilities(x: Tensor): Tensor
}

/** Penalised logistic regression by IRLS on [x, 1]; returns P(t = 1 | x). */
function logisticExceedance(X: Float64Array, n: number, d: number, t: Float64Array, l2: number, maxSteps: number) {
  const p = d + 1
  const design = new Float64Array(n * p)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) design[i * p + j] = X[i * d + j]
    design[i * p + d] = 1
  }
  const penalty = new Float64Array(p * p)
  for (let j = 0; j < d; j++) penalty[j * p + j] = l2
  const final = run(
    irls({
      design: fromData(design, [n, p]),
      y: fromData(t, [n]),
      family: binomialFamily(),
      link: link('logit'),
      penalty: l2 > 0 ? fromData(penalty, [p, p]) : undefined,
    }),
    {},
    maxSteps,
  )
  const beta = final.coefficients ? Float64Array.from(toFlat(final.coefficients)) : new Float64Array(p)
  return (x: Tensor): Float64Array => {
    const [m] = matrixShape(x, 'binaryDecomposition')
    const v = dense.data(x)
    return Float64Array.from({ length: m }, (_, i) => {
      let eta = beta[d]
      for (let j = 0; j < d; j++) eta += v[i * d + j] * beta[j]
      return sigmoid(eta)
    })
  }
}

/** Class probabilities from exceedance probabilities q [m, K − 1] by differencing, repaired by `coherence`. */
export function differenceExceedance(
  q: ArrayLike<number>,
  m: number,
  K: number,
  coherence: 'clip' | 'sort',
): Float64Array {
  const out = new Float64Array(m * K)
  const qs = new Float64Array(K + 1)
  for (let i = 0; i < m; i++) {
    qs[0] = 1
    qs[K] = 0
    for (let k = 0; k < K - 1; k++) qs[k + 1] = q[i * (K - 1) + k]
    if (coherence === 'sort') qs.subarray(1, K).sort((a, b) => b - a)
    let total = 0
    for (let k = 0; k < K; k++) {
      const pk = Math.max(qs[k] - qs[k + 1], 0)
      out[i * K + k] = pk
      total += pk
    }
    for (let k = 0; k < K; k++) out[i * K + k] = total > 0 ? out[i * K + k] / total : 1 / K
  }
  return out
}

/**
 * Frank and Hall's binary decomposition (see the module comment): labels are class indices 0 … K − 1. Capabilities:
 * `forward` and `score` (E[y], the expected class), `predictive` (categorical over the K classes), `decide` (the most
 * probable class), `expect`, `sample`.
 */
export function binaryDecomposition(
  params: BinaryDecompositionParams = {},
): Estimator<Supervised<Tensor, Tensor>, BinaryDecompositionModel> {
  const { l2 = 1, coherence = 'clip', maxSteps = 100 } = params
  return {
    name: 'binary-decomposition',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      const [n, d] = matrixShape(x, 'binaryDecomposition')
      const t = targetValues(y, 'binaryDecomposition')
      if (t.length !== n)
        throw new ShapeError('binaryDecomposition', `binaryDecomposition: ${n} inputs but ${t.length} labels`)
      if (!t.every((v) => Number.isInteger(v) && v >= 0))
        throw new DomainError('binaryDecomposition', 'binaryDecomposition: labels must be class indices 0, 1, …')
      const K = params.classes ?? Math.max(...t) + 1
      if (K < 2) throw new DomainError('binaryDecomposition', 'binaryDecomposition: needs at least two classes')
      const X = Float64Array.from(dense.data(x))
      const classifiers: Exceedance[] = Array.from({ length: K - 1 }, (_, k) => {
        const target = Float64Array.from(t, (v) => (v > k ? 1 : 0))
        if (params.base) {
          const model = params.base.fit(dataset(x, fromData(Int32Array.from(target), [n])), options)
          return (xs: Tensor) => Float64Array.from(toFlat(model.predictive(xs).mean() as Tensor))
        }
        return logisticExceedance(X, n, d, target, l2, maxSteps)
      })
      const exceedance = (xs: Tensor): Tensor => {
        const cols = classifiers.map((c) => c(xs))
        const m = cols[0].length
        const out = new Float64Array(m * (K - 1))
        cols.forEach((c, k) => c.forEach((v, i) => (out[i * (K - 1) + k] = v)))
        return fromData(out, [m, K - 1])
      }
      const probabilities = (xs: Tensor): Tensor => {
        const q = exceedance(xs)
        const m = q.shape[0]
        return fromData(differenceExceedance(toFlat(q), m, K, coherence), [m, K])
      }
      const expected = (xs: Tensor): Tensor => {
        const P = toFlat(probabilities(xs))
        const m = P.length / K
        return fromData(
          Float64Array.from({ length: m }, (_, i) => {
            let e = 0
            for (let k = 1; k < K; k++) e += k * P[i * K + k]
            return e
          }),
          [m],
        )
      }
      const decide = (xs: Tensor): Tensor => {
        const P = toFlat(probabilities(xs))
        const m = P.length / K
        const out = new Int32Array(m)
        for (let i = 0; i < m; i++) for (let k = 1; k < K; k++) if (P[i * K + k] > P[i * K + out[i]]) out[i] = k
        return fromData(out, [m])
      }
      const base = {
        kind: 'model' as const,
        name: 'binary-decomposition' as const,
        classes: K,
        coherence,
        exceedance,
        probabilities,
        forward: expected,
        score: expected,
        decide,
        predictive: (xs: Tensor): AnyUnivariate => categoricalPredictive(probabilities(xs)),
      }
      return withSampling(withExpectation(base))
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'binaryDecomposition',
    module: 'learning/generalised/ordinal',
    name: 'Binary decomposition (Frank and Hall)',
    summary: 'K − 1 logistic classifiers of P(y > k), differenced into class probabilities.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'score', 'sample'],
    hyper: space({
      l2: real(0, 100, { default: 1, label: 'L2 penalty' }),
      coherence: oneOf(['clip', 'sort']),
      maxSteps: int(1, 1000, { default: 100 }),
    }),
    notes: ['binary-decomposition-for-ordinal-regression', 'ordinal-regression'],
    cite: ['frank2001'],
  },
  binaryDecomposition,
)
