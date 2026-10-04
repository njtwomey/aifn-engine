/**
 * Ordinal regression with a latent linear predictor η = xᵀβ and K − 1 increasing thresholds θ: the cumulative-link
 * (proportional-odds with the logit link), continuation-ratio and adjacent-category models of
 * `aifn-compute/probability/likelihoods`' `ordinalLikelihood` (Agresti, 2010, "Analysis of Ordinal Categorical Data", 2nd ed.,
 * ch. 3–4; McCullagh, 1980). The cumulative model needs increasing thresholds, optimised unconstrained through the
 * ordered bijector of `aifn-compute/probability/bijectors` (θ₁ = u₁, θ_k = θ_{k−1} + e^{u_k}); the continuation-ratio and
 * adjacent-category likelihoods are valid for any θ, so theirs are free (ordering them would bias the fit when the
 * data put two thresholds out of order). The penalised negative log-likelihood
 * −Σ log P(yᵢ | ηᵢ, θ) + ½λ‖β‖² is minimised by L-BFGS with gradients by reverse-mode differentiation of the
 * likelihood (one definition: the model is written once, in the likelihood).
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import {
  add,
  dense,
  fromData,
  matmul,
  mul,
  neg,
  slice,
  square,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace, type Trace } from 'aifn-compute/foundation/trace'
import {
  categoricalPredictive,
  defineModel,
  matrixShape,
  targetValues,
  withExpectation,
  withSampling,
  type Decides,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Scores,
  type Supervised,
  type Trained,
  type AnyUnivariate,
} from 'aifn-compute/learning/estimators'
import { lbfgs, type LbfgsState } from 'aifn-compute/optim/second-order'
import { orderedBijector } from 'aifn-compute/probability/bijectors'
import { ordinalLikelihood, type OrdinalLinkName, type OrdinalModel } from 'aifn-compute/probability/likelihoods'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Hyperparameters of `ordinalRegression`. */
export type OrdinalRegressionParams = {
  /** The ordinal model (default `cumulative`, the proportional-odds model with the logit link). */
  model?: OrdinalModel
  /** The latent cdf (default `logit`; `adjacent-category` takes the logit only). */
  link?: OrdinalLinkName
  /** L2 penalty λ on β (thresholds are not penalised). Default 0, the maximum-likelihood fit. */
  l2?: number
  /** Number of classes K (default: the largest label + 1). */
  classes?: number
  /** Most L-BFGS steps (default 500). */
  maxSteps?: number
  /** Gradient-norm tolerance of L-BFGS (default 1e-6). */
  tolerance?: number
}

/** A fitted ordinal regression. */
export interface OrdinalRegressionModel
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<LbfgsState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'ordinal-regression'
  readonly model: OrdinalModel
  readonly link: OrdinalLinkName
  /** Number of classes K. */
  readonly classes: number
  /** β [d]: higher η = xᵀβ moves mass to higher classes in all three models. */
  readonly coefficients: Tensor
  /** The thresholds θ [K − 1] (increasing for the cumulative model). */
  readonly thresholds: Tensor
  /** The maximised log-likelihood Σ log P(yᵢ | ηᵢ, θ) (without the penalty). */
  readonly logLikelihood: number
  readonly converged: boolean
  /** Class probabilities P(y = k | x) [m, K]. */
  probabilities(x: Tensor): Tensor
}

/** Starting thresholds: the latent cdf's inverse at the cumulative class frequencies (the fit with β = 0 for logit). */
function startingThresholds(labels: Int32Array, K: number): Float64Array {
  const counts = new Float64Array(K)
  for (const c of labels) counts[c]++
  const n = labels.length
  const theta = new Float64Array(K - 1)
  let cum = 0
  for (let k = 0; k < K - 1; k++) {
    cum += counts[k]
    const p = Math.min(1 - 0.5 / n, Math.max(0.5 / n, cum / n))
    theta[k] = Math.log(p / (1 - p))
  }
  for (let k = 1; k < K - 1; k++) theta[k] = Math.max(theta[k], theta[k - 1] + 1e-3)
  return theta
}

/**
 * Ordinal regression (see the module comment): labels are the class indices 0 … K − 1. Capabilities: `forward` and
 * `score` (the latent predictor η, larger for higher classes), `predictive` (categorical over the K classes), `decide`
 * (the most probable class), `expect` (E[y], the expected class index), `sample`. The L-BFGS run is kept in
 * `training`.
 */
export function ordinalRegression(
  params: OrdinalRegressionParams = {},
): Estimator<Supervised<Tensor, Tensor>, OrdinalRegressionModel> {
  const { model = 'cumulative', link = 'logit', l2 = 0, maxSteps = 500, tolerance = 1e-6 } = params
  const likelihood = ordinalLikelihood(model, link)
  const ordered = orderedBijector()
  const toThresholds = (u: Value): Value => (model === 'cumulative' ? ordered.forward(u) : u)
  return {
    name: 'ordinal-regression',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      const [n, d] = matrixShape(x, 'ordinalRegression')
      const t = targetValues(y, 'ordinalRegression')
      if (t.length !== n)
        throw new ShapeError('ordinalRegression', `ordinalRegression: ${n} inputs but ${t.length} labels`)
      const labels = Int32Array.from(t, (v) => {
        if (!(Number.isInteger(v) && v >= 0))
          throw new DomainError('ordinalRegression', 'ordinalRegression: labels must be class indices 0, 1, …')
        return v
      })
      const K = params.classes ?? Math.max(...labels) + 1
      if (K < 2) throw new DomainError('ordinalRegression', 'ordinalRegression: needs at least two classes')
      if (labels.some((c) => c >= K))
        throw new DomainError('ordinalRegression', `ordinalRegression: a label is outside 0 … ${K - 1}`)
      const X = fromData(Float64Array.from(dense.data(x)), [n, d])
      const split = (w: Value) => ({ beta: slice(w, [0, d]), theta: toThresholds(slice(w, [d, d + K - 1])) })
      const negLogLik = (w: Value): Value => {
        const { beta, theta } = split(w)
        const eta = d > 0 ? matmul(X, beta) : fromData(new Float64Array(n), [n])
        let f = neg(sum(likelihood.logLik(labels, eta, theta)))
        if (l2 > 0 && d > 0) f = add(f, mul(0.5 * l2, sum(square(beta))))
        return f
      }
      const vg = valueAndGrad(negLogLik)
      const objective = (w: Tensor) => {
        const { value, grad } = vg(w)
        const v = typeof value === 'number' ? value : toFlat(value as Tensor)[0]
        return { value: v, grad: grad as Tensor }
      }
      const theta0 = startingThresholds(labels, K)
      const u0 = model === 'cumulative' ? toFlat(ordered.inverse(fromData(theta0, [K - 1])) as Tensor) : theta0
      const w0 = new Float64Array(d + K - 1)
      w0.set(u0, d)
      const training: Trace<LbfgsState> = trace(lbfgs(objective, { tolerance }), { x0: w0 }, maxSteps, {
        every: options.trace?.every ?? 1,
        record: { loss: (s) => s.value, gradNorm: (s) => s.gradNorm },
      })
      const final = training.final
      const w = fromData(Float64Array.from(toFlat(final.x)), [d + K - 1])
      const parts = split(w)
      const beta = fromData(Float64Array.from(toFlat(parts.beta as Tensor)), [d])
      const thresholds = fromData(Float64Array.from(toFlat(parts.theta as Tensor)), [K - 1])
      let penalty = 0
      for (const b of toFlat(beta)) penalty += 0.5 * l2 * b * b
      const forward = (input: Tensor): Tensor => {
        const [m, cols] = matrixShape(input, 'ordinalRegression.forward')
        if (cols !== d)
          throw new ShapeError('ordinalRegression', `ordinalRegression: fitted on ${d} features, given ${cols}`)
        return d > 0 ? (matmul(input, beta) as Tensor) : fromData(new Float64Array(m), [m])
      }
      const probabilities = (input: Tensor): Tensor => {
        const eta = forward(input)
        const P = likelihood.probabilities(eta, thresholds) as Tensor
        return fromData(Float64Array.from(toFlat(P)), [eta.shape[0], K])
      }
      const decide = (input: Tensor): Tensor => {
        const P = toFlat(probabilities(input))
        const m = P.length / K
        const out = new Int32Array(m)
        for (let i = 0; i < m; i++) for (let k = 1; k < K; k++) if (P[i * K + k] > P[i * K + out[i]]) out[i] = k
        return fromData(out, [m])
      }
      const base = {
        kind: 'model' as const,
        name: 'ordinal-regression' as const,
        model,
        link,
        classes: K,
        coefficients: beta,
        thresholds,
        logLikelihood: -(final.value - penalty),
        converged: final.converged === true,
        training,
        forward,
        score: forward,
        decide,
        probabilities,
        predictive: (input: Tensor): AnyUnivariate => categoricalPredictive(probabilities(input)),
      }
      return withSampling(withExpectation(base))
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'ordinalRegression',
    module: 'learning/generalised/ordinal',
    name: 'Ordinal regression',
    summary: 'Cumulative-link, continuation-ratio or adjacent-category model of ordered classes, fitted by L-BFGS.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'score', 'sample'],
    hyper: space({
      model: oneOf(['cumulative', 'continuation-ratio', 'adjacent-category']),
      link: oneOf(['logit', 'probit', 'cloglog']),
      l2: real(0, 100, { default: 0, label: 'L2 penalty' }),
      maxSteps: int(1, 5000, { default: 500 }),
      tolerance: real(1e-14, 1e-2, { default: 1e-6, scale: 'log' }),
    }),
    notes: [
      'cumulative-link-model',
      'continuation-ratio-and-adjacent-category-models',
      'proportional-odds-assumption',
      'ordinal-regression',
    ],
  },
  ordinalRegression,
)
