/**
 * Generalised linear models $g(\expect[y \mid \xvec]) = \xvec^\top\betavec + o$ as estimators: IRLS fits with the
 * classical inference of McCullagh and Nelder (1989) (standard errors from the inverse Fisher information, Wald tests,
 * deviance, Pearson dispersion, AIC, residuals), and the negative binomial with its shape $\theta$ estimated by maximum
 * likelihood (Venables and Ripley, 2002, §7.4, `glm.nb`).
 *
 * The design is `x` with a trailing column of ones when an intercept is fitted, so the intercept is the last
 * coefficient. The output is that of R's `glm` and statsmodels' `GLM`: coefficients with standard errors, Wald
 * statistics and p-values, deviance and null deviance, dispersion and AIC.
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

/** Data for a GLM fit: inputs, responses, and optional prior weights and offset, one of each per row. */
export type GlmData = {
  /** Inputs, $n \times d$: one row per observation. */
  x: Tensor
  /** Responses, $n$ (binomial: proportions of successes). */
  y: Tensor
  /** Prior weights, $n$ (binomial: the number of trials, with `y` the proportion of successes); default 1. */
  weights?: Tensor
  /** Offset $o$, $n$, added to the linear predictor, e.g. log exposure in a Poisson rate model; default 0. */
  offset?: Tensor
}

/** Hyperparameters of `glm`. */
export type GlmParams = {
  /** The response family (Poisson, binomial, gamma, ...): its variance function, deviance and predictive law. */
  family: Family
  /** The link $g$, by name or as a `Link` (default: the family's default link); one the family does not take throws. */
  link?: LinkName | Link
  /** Fit an intercept (default true); it is the last coefficient. */
  intercept?: boolean
  /**
   * Ridge penalty $\lambda$: $\lambda\lVert\betavec\rVert^2$ is added to the deviance and $\lambda\Imat$ to
   * $\Xmat^\top\Wmat\Xmat$, the intercept unpenalised (default 0).
   */
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
  /** The brand of a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'glm'
  /** The family it was fitted with. */
  readonly family: Family
  /** The link it was fitted with. */
  readonly link: Link
  /** Coefficients $\hat\betavec$ ($p$): one per column of `x`, then the intercept (when fitted). */
  readonly coefficients: Tensor
  /** Names "x0", "x1", …, "intercept". */
  readonly names: string[]
  /**
   * Covariance $\phi(\Xmat^\top\Wmat\Xmat + \lambda\Imat)^{-1}$ of $\hat\betavec$ ($p \times p$), with the
   * working weights $\Wmat$ at the fit (Bayesian for a penalised fit; the pseudo-inverse when the matrix is singular).
   */
  readonly covariance: Tensor
  /** $\sqrt{\diag(\text{covariance})}$ ($p$). */
  readonly standardErrors: Tensor
  /**
   * Wald statistics $\hat\beta_j / \text{SE}_j$ ($p$): $z$ when $\phi$ is known, $t$ (on `dfResidual` degrees of
   * freedom) when it is estimated.
   */
  readonly statistics: Tensor
  /** Two-sided p-values of the Wald tests ($p$): normal when $\phi$ is known, Student's $t$ when it is estimated. */
  readonly pValues: Tensor
  /**
   * The dispersion $\phi$: fixed by the family, or the Pearson estimate
   * $\sum_i w_i (y_i - \mu_i)^2 / V(\mu_i)$ divided by `dfResidual`.
   */
  readonly dispersion: number
  /** The deviance $\sum_i w_i d(y_i, \hat\mu_i)$ at the fit (without the penalty). */
  readonly deviance: number
  /** Deviance of the intercept-only model (with the same offset and weights). */
  readonly nullDeviance: number
  /** Residual degrees of freedom $n - p$ ($n - \text{edf}$ when penalised). */
  readonly dfResidual: number
  /**
   * Effective degrees of freedom $\trace((\Xmat^\top\Wmat\Xmat + \lambda\Imat)^{-1}\Xmat^\top\Wmat\Xmat)$ ($p$
   * when unpenalised).
   */
  readonly edf: number
  /**
   * Log-likelihood at $\hat\betavec$, with $\phi = \text{deviance}/n$ for families that estimate it (R's convention
   * for AIC).
   */
  readonly logLikelihood: number
  /** $-2\log L + 2k$, with $k = p$ ($p + 1$ when $\phi$ is estimated; $p$ even when penalised). */
  readonly aic: number
  /** Fitted means $\hat\muvec$ ($n$) on the training data. */
  readonly fitted: Tensor
  /** Linear predictor $\hat\etavec$ ($n$) on the training data, offset included. */
  readonly linearPredictor: Tensor
  /** Whether IRLS met its tolerance within `maxSteps`. */
  readonly converged: boolean
  /** IRLS steps taken. */
  readonly steps: number
  /** Training residuals of a kind (default `'deviance'`), as `residuals` computes them. */
  residuals(kind?: ResidualKind): Tensor
  /** The linear predictor $\Xmat\hat\betavec + o$ at new inputs ($m \times d$), with an optional offset ($m$). */
  forward(x: Tensor, offset?: Tensor): Tensor
}

/**
 * A vector tensor over an array, without copying.
 *
 * @param a The values.
 * @returns A tensor of shape $[\text{length}]$.
 */
const vec = (a: Float64Array) => fromData(a, [a.length])
/**
 * A tensor's values as a fresh `Float64Array`, row-major.
 *
 * @param t The tensor.
 * @returns A copy of its values.
 */
const flat = (t: Tensor) => Float64Array.from(toFlat(t))

/**
 * The design matrix $\Xmat$: the inputs with a trailing column of ones when an intercept is fitted. Throws
 * `ShapeError` when `x` is not a matrix.
 *
 * @param x The inputs, $n \times d$.
 * @param intercept Whether to append the column of ones.
 * @returns `X`, the design as a row-major array of $n p$ values; `n`, `d`, and `p`, its columns ($d + 1$ with an
 *   intercept, else $d$).
 */
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
 * A generalised linear model $g(\expect[y \mid \xvec]) = \xvec^\top\betavec + o$, fitted by IRLS (`irls`), with the
 * classical Wald inference (McCullagh and Nelder, 1989; as R's `glm` and statsmodels' `GLM`). Capabilities: `forward`
 * ($\eta$), `decide` and `expect` ($\mu = g^{-1}(\eta)$, without the offset), `predictive` (the family's distribution
 * at $\mu$ with the fitted dispersion), `sample`. The IRLS run is kept in `training` (series `deviance` and
 * `penalisedDeviance`). `fit` throws `NumericalError` when IRLS takes no step, and `ShapeError` when `x` is not a
 * matrix; a run that does not converge is reported in `converged`.
 *
 * @param params The family, and optionally the link, intercept, ridge penalty and IRLS controls.
 * @returns An estimator whose `fit` takes `GlmData` (and `FitOptions`, whose `trace.every` thins the IRLS trace) and
 *   returns a `GlmModel`.
 *
 * @example Counts in two groups: the coefficients are the logs of the ratio of the means and of the baseline mean
 * const x = tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
 * const y = tensor([1, 3, 0, 4, 6, 9, 2, 11])
 * // `glm` takes any family; this one is the negative binomial at the θ that `negativeBinomialRegression` estimates.
 * const { family } = negativeBinomialRegression().fit({ x, y })
 * const model = glm({ family }).fit({ x, y })
 * print('names =', model.names)
 * print('exp(coefficients) =', exp(model.coefficients))
 * print('standard errors =', model.standardErrors)
 * print('p-values =', model.pValues)
 * print('deviance, null deviance =', model.deviance, model.nullDeviance)
 * print('IRLS steps =', model.steps, ' converged =', model.converged)
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

/** What `summarise` needs of a finished `glm` fit. */
type SummaryInput = {
  /** The family fitted. */
  family: Family
  /** The link fitted. */
  link: Link
  /** Whether the design's last column is the intercept. */
  intercept: boolean
  /** Number of input columns $d$. */
  d: number
  /** Number of observations $n$. */
  n: number
  /** Number of coefficients $p$. */
  p: number
  /** The design $\Xmat$, row-major, $n p$ values. */
  X: Float64Array
  /** The data the model was fitted to. */
  data: GlmData
  /** The last IRLS state. */
  final: IrlsState
  /** The IRLS trace, kept on the model. */
  training: Trace<IrlsState>
  /** The ridge penalty matrix $\lambda\Imat$ (intercept entry 0), when penalised. */
  penalty?: Tensor
}

/**
 * The fitted model from a finished IRLS run: the Fisher information $\Xmat^\top\Wmat\Xmat$ at the final working
 * weights, its (penalised) inverse for the covariance and effective degrees of freedom, the Pearson dispersion, Wald
 * tests, the null deviance (by a second IRLS fit of the intercept-only model, or the offset alone without an
 * intercept), the log-likelihood and AIC, and the prediction methods.
 *
 * @param s The fit to summarise.
 * @returns The `GlmModel`.
 */
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
  /** Starting $\theta$ (default 1). */
  theta?: number
  /** Most alternations between $\betavec$ and $\theta$ (default 25). */
  maxAlternations?: number
}

/** A state of the negative binomial's alternation: the GLM fitted at the current $\theta$, as plain data. */
export type NegativeBinomialState = Status & {
  /** Alternations done. */
  t: number
  /** The $\theta$ the current GLM was fitted with. */
  theta: number
  /** The GLM's coefficients at $\theta$. */
  coefficients: Tensor
  /** Its fitted means $\muvec$. */
  fitted: Tensor
  /** Its deviance. */
  deviance: number
  /** $\lvert\log(\theta_{\text{new}} / \theta)\rvert$ of the last alternation (Infinity at the start). */
  change: number
  /** True once `change` is below $10^{-8}$. */
  converged: boolean
}

/**
 * The maximum-likelihood estimate of the negative binomial's shape $\theta$ given fitted means $\mu_i$: the root of
 * the score $\sum_i w_i s_i(\theta)$, with
 * $s_i = \psi(y_i + \theta) - \psi(\theta) + \log\theta + 1 - \log(\theta + \mu_i) - (y_i + \theta)/(\mu_i + \theta)$
 * ($\psi$ the digamma function), found by `findRoot` in $\log\theta$ on $[10^{-4}, 10^6]$ (Venables and Ripley, 2002,
 * §7.4, R's `theta.ml`). When the counts are no more dispersed than Poisson the score has no root there: $\theta$ is
 * then an end of the bracket and `converged` is false.
 *
 * @param y The counts, $n$.
 * @param mu The fitted means $\mu_i$, $n$, held fixed.
 * @param weights Prior weights $w_i$, $n$ (default 1).
 * @returns `theta`, the estimate, and `converged`, whether the root search converged.
 *
 * @example Overdispersed counts give a small shape; counts no more variable than Poisson run to the bound
 * const mu = tensor([4, 4, 4, 4, 4, 4, 4, 4])
 * print('variance 17 about mean 4:', thetaMaximumLikelihood(tensor([0, 2, 5, 1, 9, 3, 0, 12]), mu))
 * print('variance 1.25 about mean 4:', thetaMaximumLikelihood(tensor([3, 5, 4, 4, 2, 6, 4, 4]), mu))
 */
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
 * takes the maximum-likelihood $\theta$ at the current fitted means (`thetaMaximumLikelihood`), then refits
 * $\betavec$ by IRLS (`glm`) at that $\theta$; converged when $\log\theta$ moves by less than $10^{-8}$, diverged when
 * $\theta$ is not finite. No start: $\theta$ begins at `params.theta` (default 1), with a first fit there.
 *
 * @param data The data, as `glm` takes it.
 * @param params The GLM's hyperparameters other than the family, and the starting `theta`.
 * @param options Fit options passed to every IRLS fit.
 * @returns The algorithm; its states hold the coefficients, fitted means and deviance of each fit, not the model.
 *
 * @example The alternation run to convergence
 * const x = tensor([[0], [1], [2], [3], [4], [5], [6], [7]])
 * const y = tensor([1, 0, 3, 1, 6, 2, 12, 5])
 * const final = run(negativeBinomialAlternation({ x, y }), undefined, 25)
 * print('theta =', final.theta, ' alternations =', final.t, ' converged =', final.converged)
 * print('coefficients =', final.coefficients)
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
 * Negative binomial (NB2, $\var(y) = \mu + \mu^2/\theta$) regression with $\theta$ estimated: alternate an IRLS fit
 * of $\betavec$ at fixed $\theta$ with the maximum-likelihood $\theta$ at fixed $\mu$ until $\theta$ settles
 * (`negativeBinomialAlternation`), as R's `MASS::glm.nb`. The model is a final `glm` fit at the last $\theta$ (in
 * `family.params.theta`), with the alternation's trace in `alternation` (series `theta` and `deviance`).
 *
 * @param params The GLM's hyperparameters other than the family, the starting `theta` and `maxAlternations`.
 * @returns An estimator whose `fit` takes `GlmData` and returns the `GlmModel` with its `alternation`.
 *
 * @example Overdispersed counts: the shape, its path, and the log-linear coefficients
 * const x = tensor([[0], [1], [2], [3], [4], [5], [6], [7]])
 * const y = tensor([1, 0, 3, 1, 6, 2, 12, 5])
 * const model = negativeBinomialRegression().fit({ x, y })
 * print('theta =', model.family.params.theta)
 * print('theta by alternation =', model.alternation.series.theta)
 * print('coefficients =', model.coefficients, ' standard errors =', model.standardErrors)
 * print('AIC =', model.aic)
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
