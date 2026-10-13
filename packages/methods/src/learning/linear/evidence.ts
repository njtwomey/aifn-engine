/**
 * The evidence (marginal likelihood) of Bayesian linear regression, with its Occam decomposition, and MacKay's
 * fixed-point updates that maximise it over the two precisions (empirical Bayes, or type-II maximum likelihood).
 *
 * The model is that of Bishop (2006), §3.3 and §3.5: a prior $\wvec \sim \Gauss(\zeros, \alpha^{-1}\Imat)$ on the $M$
 * weights and $\yvec \mid \wvec \sim \Gauss(\Xmat\wvec, \beta^{-1}\Imat)$ for the $N$ targets. With
 * $\Amat = \alpha\Imat + \beta\Xmat^\top\Xmat$, the posterior is $\Gauss(\mvec_N, \Amat^{-1})$ with
 * $\mvec_N = \beta\Amat^{-1}\Xmat^\top\yvec$, and the log evidence is (Bishop's eq. 3.86)
 *
 * $$
 * \ln p(\yvec \mid \alpha, \beta) = \frac{M}{2}\ln\alpha + \frac{N}{2}\ln\beta - E(\mvec_N)
 *   - \frac12\ln\lvert\Amat\rvert - \frac{N}{2}\ln 2\pi,
 * \qquad E(\mvec_N) = \frac{\beta}{2}\norm{\yvec - \Xmat\mvec_N}^2 + \frac{\alpha}{2}\mvec_N^\top\mvec_N.
 * $$
 *
 * It splits into the best-fit log likelihood $\ln p(\yvec \mid \mvec_N, \beta)$ and the log Occam factor
 * $\frac{M}{2}\ln\alpha - \frac{\alpha}{2}\mvec_N^\top\mvec_N - \frac12\ln\lvert\Amat\rvert$, the log of the ratio of
 * the prior density to the posterior density at $\mvec_N$ (MacKay, 1992), which is never positive once the data are
 * informative. Every quantity comes from one eigendecomposition $\Xmat^\top\Xmat = \Vmat\operatorname{diag}(\lambda_i)
 * \Vmat^\top$, so each $(\alpha, \beta)$ costs $M^2 + NM$ operations. MacKay's updates set
 * $\gamma = \sum_i \beta\lambda_i / (\alpha + \beta\lambda_i)$ (the effective number of parameters),
 * $\alpha \leftarrow \gamma / \mvec_N^\top\mvec_N$ and $\beta \leftarrow (N - \gamma) / \norm{\yvec - \Xmat\mvec_N}^2$,
 * as scikit-learn's `BayesianRidge`, whose `lambda_` is $\alpha$ here and whose `alpha_` is $\beta$.
 */

import { eigh } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { DomainError } from 'aifn-compute/foundation/errors'
import { centre, gram, type Centred } from './elasticNet'

/** Options of `bayesianLinearEvidence`. */
export interface EvidenceOptions {
  /** The prior precision $\alpha > 0$ of the weights. */
  alpha: number
  /** The noise precision $\beta > 0$. */
  beta: number
  /**
   * Centre the design and target first, as `BayesianRidge` does with `fit_intercept` (default false: the design is
   * taken as it is, so put a column of ones in it for a bias, as Bishop's basis functions do). The evidence is then
   * that of the centred data, with $N$ still the number of rows.
   */
  intercept?: boolean
}

/** The six terms of the log evidence (Bishop's eq. 3.86), signed so that they sum to it. */
export interface EvidenceTerms {
  /** $\frac{M}{2}\ln\alpha$, the normalisation of the prior. */
  readonly priorPrecision: number
  /** $\frac{N}{2}\ln\beta$, the normalisation of the likelihood. */
  readonly noisePrecision: number
  /** $-\frac{\beta}{2}\norm{\yvec - \Xmat\mvec_N}^2$, the misfit to the data. */
  readonly dataFit: number
  /** $-\frac{\alpha}{2}\mvec_N^\top\mvec_N$, the prior's penalty on the posterior mean. */
  readonly weightPenalty: number
  /** $-\frac12\ln\lvert\Amat\rvert$, the volume of the posterior. */
  readonly posteriorVolume: number
  /** $-\frac{N}{2}\ln 2\pi$. */
  readonly constant: number
}

/** The evidence of a Bayesian linear model at given precisions, with the posterior over the weights. */
export interface LinearEvidence {
  /** The log evidence $\ln p(\yvec \mid \alpha, \beta)$. */
  readonly logEvidence: number
  /** The terms of the log evidence, which sum to it. */
  readonly terms: EvidenceTerms
  /** The best-fit log likelihood $\ln p(\yvec \mid \mvec_N, \beta)$: `noisePrecision + dataFit + constant`. */
  readonly bestFitLogLikelihood: number
  /**
   * The log Occam factor, `priorPrecision + weightPenalty + posteriorVolume`: the log of the prior density over the
   * posterior density at $\mvec_N$, the price of the parameters' fine tuning.
   */
  readonly logOccamFactor: number
  /** The posterior mean $\mvec_N$, $M$ values. */
  readonly mean: Tensor
  /** The posterior covariance $\Amat^{-1}$, $M \times M$. */
  readonly covariance: Tensor
  /** The intercept $\bar y - \bar{\xvec}^\top\mvec_N$ with `intercept`; 0 without. */
  readonly intercept: number
  /** The effective number of parameters $\gamma = \sum_i \beta\lambda_i / (\alpha + \beta\lambda_i)$. */
  readonly gamma: number
  /** The residual sum of squares $\norm{\yvec - \Xmat\mvec_N}^2$. */
  readonly rss: number
  /** The prior precision $\alpha$. */
  readonly alpha: number
  /** The noise precision $\beta$. */
  readonly beta: number
}

/** A centred problem with the eigendecomposition of its Gram matrix: what every evaluation of the evidence needs. */
interface Spectral {
  /** The design and target. */
  readonly c: Centred
  /** The eigenvalues $\lambda_i \ge 0$ of $\Xmat^\top\Xmat$, $M$ values. */
  readonly lambda: Float64Array
  /** The eigenvectors as the columns of $\Vmat$, $M \times M$ row-major. */
  readonly V: Float64Array
  /** $\Vmat^\top\Xmat^\top\yvec$, $M$ values. */
  readonly Vq: Float64Array
}

/**
 * Centre the data (with `intercept`) and decompose $\Xmat^\top\Xmat$. Throws `ShapeError` for mismatched shapes.
 *
 * @param x The design, $N \times M$.
 * @param y The target, $N$ values.
 * @param intercept Whether to centre first.
 * @param where The caller's name, for error messages.
 * @returns The centred problem with the eigenvalues (clipped at 0), eigenvectors and rotated $\Xmat^\top\yvec$.
 */
function spectral(x: Tensor, y: Tensor, intercept: boolean, where: string): Spectral {
  const c = centre(x, y, intercept, where)
  const { d, H, q } = gram(c)
  const { values, vectors } = eigh(fromData(H, [d, d]))
  const lambda = Float64Array.from(dense.data(values), (v) => Math.max(0, v))
  const V = Float64Array.from(dense.data(vectors))
  const Vq = new Float64Array(d)
  for (let a = 0; a < d; a++) for (let j = 0; j < d; j++) Vq[a] += V[j * d + a] * q[j]
  return { c, lambda, V, Vq }
}

/**
 * The posterior mean, residual sum of squares, effective number of parameters and $\ln\lvert\Amat\rvert$ at given
 * precisions.
 *
 * @param s The decomposed problem.
 * @param alpha The prior precision.
 * @param beta The noise precision.
 * @returns `m` ($M$ values), `rss`, `gamma` and `logDet`.
 */
function posterior(
  s: Spectral,
  alpha: number,
  beta: number,
): { m: Float64Array; rss: number; gamma: number; logDet: number } {
  const { c, lambda, V, Vq } = s
  const { n, d, x, y } = c
  const m = new Float64Array(d)
  let gamma = 0
  let logDet = 0
  for (let a = 0; a < d; a++) {
    const shrink = (beta * Vq[a]) / (alpha + beta * lambda[a])
    for (let j = 0; j < d; j++) m[j] += V[j * d + a] * shrink
    gamma += (beta * lambda[a]) / (alpha + beta * lambda[a])
    logDet += Math.log(alpha + beta * lambda[a])
  }
  let rss = 0
  for (let i = 0; i < n; i++) {
    let r = y[i]
    for (let j = 0; j < d; j++) r -= x[i * d + j] * m[j]
    rss += r * r
  }
  return { m, rss, gamma, logDet }
}

/**
 * The evidence and the posterior at given precisions.
 *
 * @param s The decomposed problem.
 * @param alpha The prior precision.
 * @param beta The noise precision.
 * @returns The evidence, its terms, and the posterior mean and covariance.
 */
function evidenceAt(s: Spectral, alpha: number, beta: number): LinearEvidence {
  const { c, lambda, V } = s
  const { n, d } = c
  const { m, rss, gamma, logDet } = posterior(s, alpha, beta)
  let mm = 0
  let b = c.yMean
  for (let j = 0; j < d; j++) {
    mm += m[j] * m[j]
    b -= c.xMean[j] * m[j]
  }
  const terms: EvidenceTerms = {
    priorPrecision: (d / 2) * Math.log(alpha),
    noisePrecision: (n / 2) * Math.log(beta),
    dataFit: -(beta / 2) * rss,
    weightPenalty: -(alpha / 2) * mm,
    posteriorVolume: -logDet / 2,
    constant: -(n / 2) * Math.log(2 * Math.PI),
  }
  const bestFitLogLikelihood = terms.noisePrecision + terms.dataFit + terms.constant
  const logOccamFactor = terms.priorPrecision + terms.weightPenalty + terms.posteriorVolume
  const cov = new Float64Array(d * d)
  for (let a = 0; a < d; a++) {
    const inv = 1 / (alpha + beta * lambda[a])
    for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) cov[j * d + k] += V[j * d + a] * inv * V[k * d + a]
  }
  return {
    logEvidence: bestFitLogLikelihood + logOccamFactor,
    terms,
    bestFitLogLikelihood,
    logOccamFactor,
    mean: fromData(m, [d]),
    covariance: fromData(cov, [d, d]),
    intercept: b,
    gamma,
    rss,
    alpha,
    beta,
  }
}

/**
 * Check that a precision is positive and finite.
 *
 * @param value The precision.
 * @param name Its name, for the message.
 * @param where The caller's name, for error messages.
 */
function checkPrecision(value: number, name: string, where: string): void {
  if (!(value > 0) || !Number.isFinite(value))
    throw new DomainError(where, `${where}: ${name} must be positive and finite`)
}

/**
 * The log evidence $\ln p(\yvec \mid \alpha, \beta)$ of Bayesian linear regression (Bishop, 2006, eq. 3.86), with its
 * decomposition into the best-fit log likelihood and the log Occam factor and the six terms of eq. 3.86, and the
 * posterior $\Gauss(\mvec_N, \Amat^{-1})$ over the weights. Comparing the evidence of nested designs (polynomial
 * degrees, say) at their own best precisions is Bayesian model selection: the fit term rewards complexity and the
 * Occam factor charges for it. The result equals scikit-learn's `BayesianRidge` score at `lambda_` $= \alpha$ and
 * `alpha_` $= \beta$ with its hyperpriors switched off. Throws `DomainError` for a precision that is not positive and
 * finite, and `ShapeError` for mismatched shapes.
 *
 * @param x The design $\Xmat$, $N \times M$ (basis functions evaluated at the inputs, one row per point).
 * @param y The targets, $N$ values.
 * @param options The precisions `alpha` and `beta`, and whether to centre for an intercept, as `EvidenceOptions`.
 * @returns The log evidence, its terms, the posterior mean and covariance, $\gamma$ and the residual sum of squares.
 *
 * @example The evidence picks a middle polynomial degree
 * // Ten noisy points of sin(2 pi t); polynomial designs of degree 0 to 9 at Bishop's alpha = 5e-3, beta = 11.1.
 * const t = Array.from({ length: 10 }, (_, i) => i / 9)
 * const noise = toFlat(normals(stream(0), [10]))
 * const y = tensor(t.map((v, i) => Math.sin(2 * Math.PI * v) + 0.3 * noise[i]))
 * for (const degree of [0, 1, 3, 5, 9]) {
 *   const x = tensor(t.map((v) => Array.from({ length: degree + 1 }, (_, k) => v ** k)))
 *   const e = bayesianLinearEvidence(x, y, { alpha: 5e-3, beta: 11.1 })
 *   print('degree', degree, ' log evidence =', e.logEvidence.toFixed(2), ' fit =',
 *     e.bestFitLogLikelihood.toFixed(2), ' Occam =', e.logOccamFactor.toFixed(2))
 * }
 *
 * @example The terms sum to the log evidence
 * const x = tensor([[1, 0], [1, 1], [1, 2], [1, 3]])
 * const e = bayesianLinearEvidence(x, tensor([0.1, 0.9, 2.2, 2.9]), { alpha: 1, beta: 10 })
 * print('terms =', e.terms)
 * print('sum =', Object.values(e.terms).reduce((a, b) => a + b, 0), ' log evidence =', e.logEvidence)
 * print('posterior mean =', e.mean, ' gamma =', e.gamma)
 */
export function bayesianLinearEvidence(x: Tensor, y: Tensor, options: EvidenceOptions): LinearEvidence {
  const { alpha, beta, intercept = false } = options
  checkPrecision(alpha, 'alpha', 'bayesianLinearEvidence')
  checkPrecision(beta, 'beta', 'bayesianLinearEvidence')
  return evidenceAt(spectral(x, y, intercept, 'bayesianLinearEvidence'), alpha, beta)
}

/** The problem an `evidenceMaximisationSteps` run solves. */
export interface EvidenceProblem {
  /** The design $\Xmat$, $N \times M$. */
  x: Tensor
  /** The targets, $N$ values. */
  y: Tensor
  /** Centre the design and target first, as `BayesianRidge` with `fit_intercept` (default false). */
  intercept?: boolean
  /**
   * The convergence test, scikit-learn's: a state is `converged` when the $\ell_1$ change of the posterior mean from
   * the previous state is below this (default $10^{-3}$, scikit-learn's `tol`; it is in the units of the weights).
   */
  tolerance?: number
}

/** One state of `evidenceMaximisationSteps`: the precisions and the evidence and posterior at them. */
export interface EvidenceState extends Status {
  /** Updates made (the step count). */
  t: number
  /** The prior precision $\alpha$. */
  alpha: number
  /** The noise precision $\beta$. */
  beta: number
  /** $\ln\alpha$, the iterate a trace plots. */
  logAlpha: number
  /** $\ln\beta$. */
  logBeta: number
  /** The effective number of parameters $\gamma$ at these precisions. */
  gamma: number
  /** The log evidence at these precisions. */
  logEvidence: number
  /** The posterior mean $\mvec_N$, $M$ values. */
  mean: Tensor
  /** The residual sum of squares at the posterior mean. */
  rss: number
  /** The $\ell_1$ change of the posterior mean from the previous state (NaN at the start). */
  change: number
}

/**
 * MacKay's (1992) fixed-point maximisation of the evidence over $\alpha$ and $\beta$ as a step-through algorithm,
 * as scikit-learn's `BayesianRidge` with its hyperpriors switched off: each step sets
 * $\alpha \leftarrow \gamma / \mvec_N^\top\mvec_N$ and $\beta \leftarrow (N - \gamma) / \norm{\yvec - \Xmat\mvec_N}^2$
 * from the current state's posterior, then recomputes the posterior and the evidence. The evidence rises at each step
 * in practice, though the updates are not guaranteed to be monotone. `init` takes the starting precisions (default
 * $\alpha = 1$ and $\beta = 1 / \operatorname{var}(\yvec)$, scikit-learn's). The states match `BayesianRidge`'s
 * `scores_` one for one; scikit-learn stops one update after the state flagged `converged` here. A state whose
 * precisions are no longer finite (all weights zero, or a perfect fit) is flagged `diverged`. Throws `DomainError`
 * for a starting precision that is not positive, and `ShapeError` for mismatched shapes.
 *
 * @param problem The data, whether to centre, and the convergence tolerance, as `EvidenceProblem`.
 * @returns The algorithm, to run with `run` or `trace`; its start is `{ alpha?, beta? }`.
 *
 * @example The precisions settle in a few steps
 * const x = normals(stream(0), [50, 3])
 * const noise = toFlat(normals(stream(1), [50]))
 * const y = tensor(toArray(x).map((r, i) => r[0] - 2 * r[1] + 0.5 * noise[i]))
 * const tr = trace(evidenceMaximisationSteps({ x, y }), {}, 20, { keep: 'all' })
 * print('alpha:', tr.steps.map((s) => s.alpha.toFixed(3)))
 * print('beta:', tr.steps.map((s) => s.beta.toFixed(3)), ' (true 4)')
 * print('log evidence:', tr.steps.map((s) => s.logEvidence.toFixed(3)))
 */
export function evidenceMaximisationSteps(
  problem: EvidenceProblem,
): Algorithm<{ alpha?: number; beta?: number }, EvidenceState> {
  const { intercept = false, tolerance = 1e-3 } = problem
  const s = spectral(problem.x, problem.y, intercept, 'evidenceMaximisationSteps')
  const { n } = s.c
  const target = dense.data(problem.y)
  let mean = 0
  for (const v of target) mean += v / target.length
  let variance = 0
  for (const v of target) variance += (v - mean) ** 2 / target.length
  const state = (t: number, alpha: number, beta: number, previous: Float64Array | null): EvidenceState => {
    const finite = Number.isFinite(alpha) && Number.isFinite(beta) && alpha > 0 && beta > 0
    if (!finite) {
      return {
        t,
        alpha,
        beta,
        logAlpha: Math.log(alpha),
        logBeta: Math.log(beta),
        gamma: NaN,
        logEvidence: NaN,
        mean: fromData(previous ?? new Float64Array(s.c.d), [s.c.d]),
        rss: NaN,
        change: NaN,
        diverged: true,
      }
    }
    const e = evidenceAt(s, alpha, beta)
    const m = dense.data(e.mean)
    let change = NaN
    if (previous) {
      change = 0
      for (let j = 0; j < m.length; j++) change += Math.abs(m[j] - previous[j])
    }
    return {
      t,
      alpha,
      beta,
      logAlpha: Math.log(alpha),
      logBeta: Math.log(beta),
      gamma: e.gamma,
      logEvidence: e.logEvidence,
      mean: e.mean,
      rss: e.rss,
      change,
      converged: change < tolerance,
    }
  }
  return {
    name: 'evidence-maximisation',
    init: ({ alpha = 1, beta = 1 / (variance + Number.EPSILON) }) => {
      checkPrecision(alpha, 'alpha', 'evidenceMaximisationSteps')
      checkPrecision(beta, 'beta', 'evidenceMaximisationSteps')
      return state(0, alpha, beta, null)
    },
    step: (current) => {
      const m = dense.data(current.mean)
      let mm = 0
      for (const v of m) mm += v * v
      const alpha = current.gamma / mm
      const beta = (n - current.gamma) / current.rss
      return state(current.t + 1, alpha, beta, m)
    },
  }
}

/** Options of `evidenceMaximisation`. */
export interface EvidenceMaximisationOptions {
  /** The starting prior precision (default 1). */
  alpha0?: number
  /** The starting noise precision (default $1 / \operatorname{var}(\yvec)$). */
  beta0?: number
  /** Centre the design and target first (default false). */
  intercept?: boolean
  /** The $\ell_1$ change of the posterior mean below which the run stops (default $10^{-3}$). */
  tolerance?: number
  /** The most updates (default 300, scikit-learn's `max_iter`). */
  maxSteps?: number
}

/** The evidence at the precisions MacKay's updates reach, with the run that reached them. */
export interface EvidenceMaximisation extends LinearEvidence {
  /** The updates made. */
  readonly steps: number
  /** Whether the run met its convergence test before `maxSteps`. */
  readonly converged: boolean
  /** $\alpha$, $\beta$, $\gamma$ and the log evidence at each state, from the start: `steps + 1` values each. */
  readonly history: {
    readonly alpha: number[]
    readonly beta: number[]
    readonly gamma: number[]
    readonly logEvidence: number[]
  }
}

/**
 * Empirical Bayes for Bayesian linear regression: MacKay's updates (`evidenceMaximisationSteps`) run until the
 * posterior mean stops moving, and the evidence and posterior at the precisions reached, with the history of the
 * run. With `intercept` and the starting values left at their defaults it reproduces scikit-learn's
 * `BayesianRidge(alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0)`: `lambda_` is `alpha`, `alpha_` is `beta`, `coef_`
 * is `mean` and `sigma_` is `covariance` (up to where its looser stop leaves it). Throws as
 * `evidenceMaximisationSteps`, and `DomainError` when the precisions leave the finite positive numbers (a design
 * that fits the data exactly, or carries no signal).
 *
 * @param x The design $\Xmat$, $N \times M$.
 * @param y The targets, $N$ values.
 * @param options The starting precisions, centring, tolerance and the most updates, as
 *   `EvidenceMaximisationOptions`.
 * @returns The evidence and posterior at the final precisions, the updates made and the history of the run.
 *
 * @example The noise precision found against the truth
 * // y = x w + noise of standard deviation 0.5, so beta should come out near 4.
 * const x = normals(stream(0), [200, 4])
 * const noise = toFlat(normals(stream(1), [200]))
 * const y = tensor(toArray(x).map((r, i) => r[0] - r[1] + 0.5 * noise[i]))
 * const fit = evidenceMaximisation(x, y)
 * print('alpha =', fit.alpha, ' beta =', fit.beta, ' gamma =', fit.gamma)
 * print('steps =', fit.steps, ' log evidence by step:', fit.history.logEvidence.map((v) => v.toFixed(2)))
 */
export function evidenceMaximisation(
  x: Tensor,
  y: Tensor,
  options: EvidenceMaximisationOptions = {},
): EvidenceMaximisation {
  const { alpha0, beta0, intercept = false, tolerance = 1e-3, maxSteps = 300 } = options
  const algorithm = evidenceMaximisationSteps({ x, y, intercept, tolerance })
  // The updates draw nothing, so any stream serves.
  const unused = stream(0)
  let state = algorithm.init({ alpha: alpha0, beta: beta0 }, unused)
  const history = { alpha: [state.alpha], beta: [state.beta], gamma: [state.gamma], logEvidence: [state.logEvidence] }
  while (state.t < maxSteps && !state.converged) {
    state = algorithm.step(state, { t: state.t, stream: unused })
    if (state.diverged)
      throw new DomainError('evidenceMaximisation', 'evidenceMaximisation: the precisions are no longer finite')
    history.alpha.push(state.alpha)
    history.beta.push(state.beta)
    history.gamma.push(state.gamma)
    history.logEvidence.push(state.logEvidence)
  }
  const final = bayesianLinearEvidence(x, y, { alpha: state.alpha, beta: state.beta, intercept })
  return { ...final, steps: state.t, converged: state.converged === true, history }
}
