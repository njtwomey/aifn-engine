/**
 * Generalised linear models as estimators: IRLS fits with the classical inference of McCullagh and Nelder (1989)
 * (standard errors from the inverse Fisher information, Wald tests, deviance, Pearson dispersion, AIC, residuals), and
 * the negative binomial with its shape estimated by maximum likelihood (Venables and Ripley, 2002, §7.4, `glm.nb`).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { residuals, type ResidualKind } from '../residuals'
import {
  withExpectation,
  withSampling,
  type Decides,
  type Distribution,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { cholesky, inverse, pinv } from 'aifn-compute/numerics/linalg'
import { findRoot } from 'aifn-compute/numerics/roots'
import { digamma, normalCdf, studentTCdf } from 'aifn-compute/numerics/special'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import {
  checkLink,
  negativeBinomialFamily,
  type Family,
  type Link,
  type LinkName,
} from 'aifn-compute/probability/likelihoods'
import { irls, type IrlsState } from '../irls'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, real, space } from 'aifn-compute/foundation/space'
import { NumericalError, ShapeError } from 'aifn-compute/foundation/errors'

/** Data for a GLM fit: inputs x [n, d], responses y [n], optional prior weights and offset [n]. */
export type GlmData = {
  x: Tensor
  y: Tensor
  /** Prior weights (binomial: the number of trials, with y the proportion of successes). */
  weights?: Tensor
  /** Offset added to the linear predictor, e.g. log exposure in a Poisson rate model. */
  offset?: Tensor
}

/** Hyperparameters of `glm`. */
export type GlmParams = {
  family: Family
  /** The link (default: the family's default link). */
  link?: LinkName | Link
  /** Fit an intercept (default true); it is the last coefficient. */
  intercept?: boolean
  /** Ridge penalty λ on ½‖β‖²… added as λI to XᵀWX, intercept unpenalised (default 0). */
  l2?: number
  /** IRLS convergence tolerance on the relative change of the penalised deviance (default 1e-8). */
  tolerance?: number
  /** Most IRLS steps (default 50). */
  maxSteps?: number
}

/** A fitted GLM. */
export interface GlmModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Distribution>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<IrlsState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'glm'
  readonly family: Family
  readonly link: Link
  /** Coefficients β [p]: one per column of x, then the intercept (when fitted). */
  readonly coefficients: Tensor
  /** Names "x0", "x1", …, "intercept". */
  readonly names: string[]
  /** Covariance φ(XᵀWX + λI)⁻¹ of β̂ [p, p] (Bayesian for a penalised fit). */
  readonly covariance: Tensor
  /** √diag(covariance) [p]. */
  readonly standardErrors: Tensor
  /** Wald statistics β̂ⱼ/SEⱼ [p]: z when φ is known, t (n − p degrees of freedom) when it is estimated. */
  readonly statistics: Tensor
  /** Two-sided p-values of the Wald tests [p]. */
  readonly pValues: Tensor
  /** φ: fixed by the family, or the Pearson estimate Σ wᵢ(yᵢ − μᵢ)²/V(μᵢ) / (n − p). */
  readonly dispersion: number
  readonly deviance: number
  /** Deviance of the intercept-only model (with the same offset and weights). */
  readonly nullDeviance: number
  /** Residual degrees of freedom n − p (n − edf when penalised). */
  readonly dfResidual: number
  /** Effective degrees of freedom tr((XᵀWX + λI)⁻¹XᵀWX) (p when unpenalised). */
  readonly edf: number
  /** Log-likelihood at β̂, with φ = deviance/n for families that estimate it (R's convention for AIC). */
  readonly logLikelihood: number
  /** −2 log L + 2k, with k = p (+1 when φ is estimated). */
  readonly aic: number
  /** Fitted means μ̂ [n] and linear predictor η̂ [n] on the training data. */
  readonly fitted: Tensor
  readonly linearPredictor: Tensor
  readonly converged: boolean
  /** IRLS steps taken. */
  readonly steps: number
  /** Training residuals of a kind. */
  residuals(kind?: ResidualKind): Tensor
  /** The linear predictor at new inputs, with an optional offset. */
  forward(x: Tensor, offset?: Tensor): Tensor
}

const vec = (a: Float64Array) => fromData(a, [a.length])
const flat = (t: Tensor) => Float64Array.from(toFlat(t))

function designOf(x: Tensor, intercept: boolean): { X: Float64Array; n: number; d: number; p: number } {
  if (x.shape.length !== 2) throw new ShapeError('glm', `glm: x must be [n, d], got [${x.shape.join(', ')}]`)
  const [n, d] = x.shape
  const p = d + (intercept ? 1 : 0)
  const v = toFlat(x)
  const X = new Float64Array(n * p)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) X[i * p + j] = v[i * d + j]
    if (intercept) X[i * p + d] = 1
  }
  return { X, n, d, p }
}

/**
 * A generalised linear model g(E[y | x]) = xᵀβ + o, fitted by IRLS (`irls`), with the classical Wald inference.
 * Capabilities: `forward` (η), `decide` and `expect` (μ = g⁻¹(η)), `predictive` (the family's distribution at μ with
 * the fitted dispersion), `sample`. The IRLS run is kept in `training`.
 *
 * @example glm({ family: poissonFamily() }).fit({ x, y, offset: logExposure })
 */
export function glm(params: GlmParams): Estimator<GlmData, GlmModel> {
  const { family, intercept = true, l2 = 0, tolerance = 1e-8, maxSteps = 50 } = params
  const link = checkLink(family, params.link, 'glm')
  return {
    name: `glm-${family.name}-${link.name}`,
    params,
    fit(data, options: FitOptions = {}) {
      const { X, n, d, p } = designOf(data.x, intercept)
      const design = fromData(X, [n, p])
      let penalty: Tensor | undefined
      if (l2 > 0) {
        const P = new Float64Array(p * p)
        for (let j = 0; j < d; j++) P[j * p + j] = l2
        if (!intercept) P[(p - 1) * p + p - 1] = l2
        penalty = fromData(P, [p, p])
      }
      const problem = {
        design,
        y: data.y,
        family,
        link,
        weights: data.weights,
        offset: data.offset,
        penalty,
        tolerance,
      }
      const training: Trace<IrlsState> = trace(irls(problem), {}, maxSteps, {
        every: options.trace?.every ?? 1,
        record: { deviance: (s) => s.deviance, penalisedDeviance: (s) => s.penalisedDeviance },
      })
      const final = training.final
      if (!final.coefficients) throw new NumericalError('glm', 'glm: IRLS took no step', 'not-converged')
      return summarise({ family, link, intercept, d, n, p, X, data, final, training, penalty })
    },
  }
}

type SummaryInput = {
  family: Family
  link: Link
  intercept: boolean
  d: number
  n: number
  p: number
  X: Float64Array
  data: GlmData
  final: IrlsState
  training: Trace<IrlsState>
  penalty?: Tensor
}

function summarise(s: SummaryInput): GlmModel {
  const { family, link, intercept, d, n, p, X, data, final, training, penalty } = s
  const beta = flat(final.coefficients!)
  const mu = flat(final.mu)
  const y = flat(data.y)
  const w = data.weights ? flat(data.weights) : new Float64Array(n).fill(1)
  // Working weights at the final β (the state's W is computed at its η).
  const W = flat(final.workingWeights)
  const I = new Float64Array(p * p)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) I[a * p + b] += W[i] * X[i * p + a] * X[i * p + b]
  const H = Float64Array.from(I)
  if (penalty) toFlat(penalty).forEach((v, k) => (H[k] += v))
  const factor = cholesky(fromData(H, [p, p]), { jitter: false })
  const Hinv = flat(factor.failed ? pinv(fromData(H, [p, p])) : (inverse(fromData(H, [p, p])) as Tensor))
  let edf = 0
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) edf += Hinv[a * p + b] * I[b * p + a]
  const dfResidual = n - edf
  const V = flat(family.variance(vec(mu)) as Tensor)
  let pearson = 0
  for (let i = 0; i < n; i++) pearson += (w[i] * (y[i] - mu[i]) ** 2) / V[i]
  const dispersion = family.dispersion ?? pearson / dfResidual
  const covariance = Hinv.map((v) => v * dispersion)
  const se = Float64Array.from({ length: p }, (_, j) => Math.sqrt(covariance[j * p + j]))
  const statistics = Float64Array.from(beta, (b, j) => b / se[j])
  const pValues = Float64Array.from(statistics, (t) =>
    family.dispersion === null
      ? 2 * (studentTCdf(-Math.abs(t), dfResidual) as number)
      : 2 * (normalCdf(-Math.abs(t)) as number),
  )
  // The null model: intercept only (or the offset alone), same family, link, weights and offset.
  let nullDeviance: number
  if (intercept) {
    const nullFit = run(
      irls({
        design: fromData(new Float64Array(n).fill(1), [n, 1]),
        y: data.y,
        family,
        link,
        weights: data.weights,
        offset: data.offset,
      }),
      {},
      50,
    )
    nullDeviance = nullFit.deviance
  } else {
    const o = data.offset ?? fromData(new Float64Array(n), [n])
    const muNull = link.inverse(o) as Tensor
    const u = flat(family.unitDeviance(data.y, muNull) as Tensor)
    nullDeviance = u.reduce((a, v, i) => a + w[i] * v, 0)
  }
  const phiLik = family.dispersion ?? final.deviance / n
  const logLikelihood = family.logLikelihood(data.y, final.mu, phiLik, data.weights ?? vec(w))
  const k = p + (family.dispersion === null ? 1 : 0)
  const names = [...Array.from({ length: d }, (_, j) => `x${j}`), ...(intercept ? ['intercept'] : [])]

  const forward = (x: Tensor, offset?: Tensor) => {
    const { X: Z, n: m, d: cols } = designOf(x, intercept)
    if (cols !== d) throw new ShapeError('glm', `glm: fitted on ${d} features, given ${cols}`)
    const o = offset ? flat(offset) : null
    const out = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      let t = o ? o[i] : 0
      for (let a = 0; a < p; a++) t += Z[i * p + a] * beta[a]
      out[i] = t
    }
    return vec(out)
  }
  const meanAt = (x: Tensor) => link.inverse(forward(x)) as Tensor
  const base = {
    kind: 'model' as const,
    name: 'glm' as const,
    family,
    link,
    coefficients: vec(beta),
    names,
    covariance: fromData(covariance, [p, p]),
    standardErrors: vec(se),
    statistics: vec(statistics),
    pValues: vec(pValues),
    dispersion,
    deviance: final.deviance,
    nullDeviance,
    dfResidual,
    edf,
    logLikelihood,
    aic: -2 * logLikelihood + 2 * k,
    fitted: final.mu,
    linearPredictor: final.eta,
    converged: final.converged,
    steps: final.t,
    training,
    residuals: (kind: ResidualKind = 'deviance') =>
      residuals({ y: data.y, mu: final.mu, eta: final.eta, weights: data.weights, family, link }, kind),
    forward,
    decide: meanAt,
    predictive: (x: Tensor) => family.predictive(meanAt(x), dispersion),
  }
  return withSampling(withExpectation(base)) as GlmModel
}

// ── Negative binomial with θ estimated ───────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `negativeBinomialRegression`. */
export type NegativeBinomialParams = Omit<GlmParams, 'family'> & {
  /** Starting θ (default 1). */
  theta?: number
  /** Most alternations between β and θ (default 25). */
  maxAlternations?: number
}

/** A state of the negative binomial's alternation: the GLM fitted at θ, and the next θ by maximum likelihood. */
export type NegativeBinomialState = Status & {
  /** Alternations done. */
  t: number
  /** The θ the current GLM was fitted with. */
  theta: number
  /** The GLM's coefficients at θ. */
  coefficients: Tensor
  /** Its fitted means μ. */
  fitted: Tensor
  /** Its deviance. */
  deviance: number
  /** |log(θ_new / θ)| of the last alternation (Infinity at the start). */
  change: number
  converged: boolean
}

/** The ML estimate of θ given fitted means (the score equation, solved by Brent's method in log θ). */
export function thetaMaximumLikelihood(y: Tensor, mu: Tensor, weights?: Tensor): { theta: number; converged: boolean } {
  const ys = flat(y)
  const ms = flat(mu)
  const w = weights ? flat(weights) : new Float64Array(ys.length).fill(1)
  const score = (logTheta: number) => {
    const t = Math.exp(logTheta)
    let s = 0
    for (let i = 0; i < ys.length; i++)
      s +=
        w[i] *
        ((digamma(ys[i] + t) as number) -
          (digamma(t) as number) +
          Math.log(t) +
          1 -
          Math.log(t + ms[i]) -
          (ys[i] + t) / (ms[i] + t))
    return s
  }
  const r = findRoot(score, [Math.log(1e-4), Math.log(1e6)])
  return { theta: Math.exp(r.x), converged: r.converged }
}

/**
 * The negative binomial's alternation as a traceable algorithm (Venables and Ripley, 2002, §7.4, `glm.nb`): each step
 * takes the ML estimate of θ at the current fitted means, then refits β by IRLS at that θ; converged when log θ moves
 * by less than 1e-8. No start: θ begins at `params.theta` (default 1).
 */
export function negativeBinomialAlternation(
  data: GlmData,
  params: Omit<NegativeBinomialParams, 'maxAlternations'> = {},
  options?: FitOptions,
): Algorithm<void, NegativeBinomialState> {
  const fitAt = (theta: number) => glm({ ...params, family: negativeBinomialFamily(theta) }).fit(data, options)
  // States are plain data: they keep the GLM's coefficients, means and deviance, not the model.
  const summary = (m: GlmModel) => ({ coefficients: m.coefficients, fitted: m.fitted, deviance: m.deviance })
  return {
    name: 'negative-binomial-alternation',
    init: () => {
      const theta = params.theta ?? 1
      return { t: 0, theta, ...summary(fitAt(theta)), change: Infinity, converged: false }
    },
    step: (s) => {
      const next = thetaMaximumLikelihood(data.y, s.fitted, data.weights).theta
      const change = Math.abs(Math.log(next / s.theta))
      return {
        t: s.t + 1,
        theta: next,
        ...summary(fitAt(next)),
        change,
        converged: change < 1e-8,
        diverged: !Number.isFinite(next),
      }
    },
  }
}

/**
 * Negative binomial (NB2) regression with θ estimated: alternate an IRLS fit of β at fixed θ with the ML estimate of
 * θ at fixed μ until θ settles (`negativeBinomialAlternation`). Returns the final GLM (with `family.params.theta`) and
 * the alternation's trace in `alternation` (record `theta` to plot its path).
 */
export function negativeBinomialRegression(
  params: NegativeBinomialParams = {},
): Estimator<GlmData, GlmModel & { alternation: Trace<NegativeBinomialState> }> {
  const { maxAlternations = 25, ...rest } = params
  return {
    name: 'negative-binomial-regression',
    params,
    fit(data, options) {
      const alternation = trace(negativeBinomialAlternation(data, rest, options), undefined, maxAlternations, {
        record: { theta: (s) => s.theta, deviance: (s) => s.deviance },
      })
      const final = glm({ ...rest, family: negativeBinomialFamily(alternation.final.theta) }).fit(data, options)
      return { ...final, alternation }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'glm',
    module: 'learning/generalised/glm',
    name: 'Generalised linear model',
    summary:
      'An exponential-family response with a link, fitted by iteratively reweighted least squares; the family is a required argument.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({
      intercept: bool({ default: true }),
      l2: real(0, 100, { default: 0, label: 'L2 penalty' }),
      tolerance: real(1e-14, 1e-2, { default: 1e-8, scale: 'log' }),
      maxSteps: int(1, 500, { default: 50 }),
    }),
    notes: ['generalised-linear-model', 'poisson-regression', 'gamma-and-tweedie-regression'],
    cite: ['nelder1972'],
  },
  glm,
)

defineModel(
  {
    key: 'negativeBinomialRegression',
    module: 'learning/generalised/glm',
    name: 'Negative binomial regression',
    summary: 'A log-linear count model with an estimated overdispersion θ, alternating IRLS and a θ update.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({
      intercept: bool({ default: true }),
      l2: real(0, 100, { default: 0, label: 'L2 penalty' }),
      maxAlternations: int(1, 200, { default: 25 }),
    }),
    notes: ['negative-binomial-and-overdispersion'],
  },
  negativeBinomialRegression,
)
