/**
 * Ordinal regression with a latent linear predictor $\eta = \xvec^\top\betavec$ and $K - 1$ thresholds
 * $\thetavec$: the cumulative-link ($\pr(y \le k) = F(\theta_k - \eta)$; proportional odds with the logit link),
 * continuation-ratio and adjacent-category models of `aifn-compute/probability/likelihoods`' `ordinalLikelihood`
 * (Agresti, 2010, "Analysis of Ordinal Categorical Data", 2nd ed., ch. 3–4; McCullagh, 1980). The cumulative model
 * needs increasing thresholds, optimised unconstrained through the ordered bijector of
 * `aifn-compute/probability/bijectors` ($\theta_0 = u_0$, $\theta_k = \theta_{k-1} + e^{u_k}$); the
 * continuation-ratio and adjacent-category likelihoods are valid for any $\thetavec$, so theirs are free (ordering them
 * would bias the fit when the data put two thresholds out of order). The penalised negative log-likelihood
 * $-\sum_i \log \pr(y_i \mid \eta_i, \thetavec) + \tfrac12\lambda\lVert\betavec\rVert^2$ is minimised by L-BFGS
 * with gradients by reverse-mode differentiation of the likelihood (one definition: the model is written once, in the
 * likelihood). There is no intercept: the thresholds play its part. The cumulative logit model is R's `MASS::polr`
 * and statsmodels' `OrderedModel`.
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
  /** The latent cdf $F$ (default `logit`; `adjacent-category` takes the logit only). */
  link?: OrdinalLinkName
  /**
   * L2 penalty $\lambda$, as $\tfrac12\lambda\lVert\betavec\rVert^2$ (thresholds are not penalised). Default 0, the
   * maximum-likelihood fit.
   */
  l2?: number
  /** Number of classes $K$ (default: the largest label + 1). */
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
  /** The brand of a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'ordinal-regression'
  /** The ordinal model it was fitted with. */
  readonly model: OrdinalModel
  /** The latent cdf it was fitted with. */
  readonly link: OrdinalLinkName
  /** Number of classes $K$. */
  readonly classes: number
  /**
   * $\hat\betavec$ ($d$): a higher $\eta = \xvec^\top\betavec$ moves mass to higher classes in all three models.
   */
  readonly coefficients: Tensor
  /** The thresholds $\hat\thetavec$ ($K - 1$; increasing for the cumulative model). */
  readonly thresholds: Tensor
  /** The maximised log-likelihood $\sum_i \log \pr(y_i \mid \eta_i, \thetavec)$ (without the penalty). */
  readonly logLikelihood: number
  /** Whether L-BFGS met its gradient tolerance within `maxSteps`. */
  readonly converged: boolean
  /** Class probabilities $\pr(y = k \mid \xvec)$, $m \times K$, at inputs $m \times d$. */
  probabilities(x: Tensor): Tensor
}

/**
 * Starting thresholds: the logit of the cumulative class frequencies, clamped to $[0.5/n, 1 - 0.5/n]$ and spaced at
 * least $10^{-3}$ apart (for the cumulative logit model, the fit with $\betavec = \zeros$). The logit is used for
 * every link and model.
 *
 * @param labels The class labels, $n$, each in $0, \dots, K - 1$.
 * @param K The number of classes.
 * @returns The $K - 1$ increasing starting thresholds.
 */
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
 * Ordinal regression (see the file comment): labels are the class indices $0, \dots, K - 1$. Capabilities: `forward`
 * and `score` (the latent predictor $\eta$, larger for higher classes), `predictive` (categorical over the $K$
 * classes), `decide` (the most probable class), `expect` ($\expect[y]$, the expected class index), `sample`, and
 * `probabilities`. The L-BFGS run is kept in `training` (series `loss` and `gradNorm`). `fit` throws `DomainError`
 * for labels that are not class indices in range or fewer than two classes, and `ShapeError` when the inputs and
 * labels differ in number.
 *
 * @param params The model, link, ridge penalty, number of classes and L-BFGS controls.
 * @returns An estimator whose `fit` takes `{ x, y }` ($n \times d$ inputs, $n$ class labels) and returns the model.
 *
 * @example Three ordered classes: the slope and the two thresholds of a cumulative logit model
 * // A latent 2x plus logistic noise, cut at -1 and 1: the model's own form, so both are recovered.
 * const s = stream(3)
 * const x = normals(s, [400, 1])
 * const u = uniform(s, 0, 1, { shape: [400] })
 * const z = add(mul(reshape(x, [400]), 2), log(div(u, sub(1, u))))
 * const y = tensor(Array.from(toFlat(z), (v) => (v > -1) + (v > 1)))
 * const model = ordinalRegression().fit({ x, y })
 * print('coefficients (true 2) =', model.coefficients)
 * print('thresholds (true -1, 1) =', model.thresholds)
 * print('P(y = k) at x = 0 =', model.probabilities(tensor([[0]])))
 *
 * @example The three models on the same data
 * const s = stream(3)
 * const x = normals(s, [400, 1])
 * const u = uniform(s, 0, 1, { shape: [400] })
 * const z = add(mul(reshape(x, [400]), 2), log(div(u, sub(1, u))))
 * const y = tensor(Array.from(toFlat(z), (v) => (v > -1) + (v > 1)))
 * for (const model of ['cumulative', 'continuation-ratio', 'adjacent-category']) {
 *   const fit = ordinalRegression({ model }).fit({ x, y })
 *   print(model + ':', 'beta =', fit.coefficients, ' log-likelihood =', fit.logLikelihood)
 * }
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
