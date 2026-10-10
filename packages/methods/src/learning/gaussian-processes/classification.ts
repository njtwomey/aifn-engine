/**
 * Binary Gaussian-process classification by the Laplace approximation or by expectation propagation
 * (`./classification-ep`): Newton's method for the posterior mode of the latent function (a traceable `Algorithm`),
 * the approximate log marginal likelihood and its gradient in the hyperparameters, type-II maximum likelihood by
 * L-BFGS, and predictive probabilities.
 *
 * Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", Algorithms 3.1 (mode finding, the stable
 * form with $\Bmat = \Imat + \Wmat^{1/2}\Kmat\Wmat^{1/2}$), 3.2 (predictions) and 5.1 (the evidence gradient), and
 * eq. 3.32 (the approximate log marginal likelihood). Labels are 0 and 1 throughout, with $y = 2t - 1 = \pm 1$ for a
 * label $t$.
 */

import { Bernoulli, type Univariate } from 'aifn-compute/probability/distributions'
import type { Status } from 'aifn-compute/foundation/contracts'
import { gaussHermite } from 'aifn-compute/numerics/quadrature'
import {
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
} from 'aifn-compute/learning/estimators'
import {
  asRows,
  gram,
  kernelDiagonal,
  kernelFromLog,
  type Kernel,
  type KernelParams,
} from 'aifn-compute/learning/kernels'
import { lbfgs, type LbfgsState } from 'aifn-compute/optim/second-order'
import { customVjp, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { ravel } from 'aifn-compute/foundation/pytree'
import { epWeights, gpEp, gpEpEvidence, stableFactor, type GpEpOptions } from './classification-ep'
import type { MvEpState } from 'aifn-compute/inference/expectation-propagation'
import { kernelLogVector } from './regression'
import { cholesky, solveTriangular } from 'aifn-compute/numerics/linalg'
import { normalCdf, normalLogCdf, normalPdf, sigmoid, softplus } from 'aifn-compute/numerics/special'
import { fromData, matmul, mul, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { run, trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The link from latent $f$ to $P(y = 1 \mid f)$: logistic $\sigma(f)$ or probit $\Phi(f)$. */
export type ClassificationLikelihood = 'logistic' | 'probit'

/**
 * $\log p(\yvec \mid \fvec)$, its gradient and the negative Hessian diagonal $\Wmat$, for labels $t \in \{0, 1\}$
 * ($y = 2t - 1$). The probit ratio $\phi(z)/\Phi(z)$ is taken through logs, so the tails stay finite.
 *
 * @param likelihood The link: logistic or probit.
 * @param t The labels, 0 or 1, one per latent value.
 * @param f The latent values $\fvec$, `[n]`.
 * @returns `logLik`, the sum $\log p(\yvec \mid \fvec)$; `grad`, $\nabla \log p(\yvec \mid \fvec)$, `[n]`; and `W`,
 *   the diagonal of $-\nabla\nabla \log p(\yvec \mid \fvec)$, `[n]`.
 */
function likelihoodTerms(likelihood: ClassificationLikelihood, t: Float64Array, f: Float64Array) {
  const n = f.length
  const grad = new Float64Array(n)
  const W = new Float64Array(n)
  let logLik = 0
  for (let i = 0; i < n; i++) {
    const y = 2 * t[i] - 1
    if (likelihood === 'logistic') {
      // log σ(yf) = −softplus(−yf); ∇ = t − σ(f); W = σ(f)(1 − σ(f)) (R&W eqs. 3.15).
      logLik -= softplus(-y * f[i])
      const p = sigmoid(f[i])
      grad[i] = t[i] - p
      W[i] = p * (1 - p)
    } else {
      // log Φ(yf); ∇ = yN(f)/Φ(yf); W = r² + yf·r with r = N(f)/Φ(yf) (R&W eq. 3.16), r from logs for the tails.
      const z = y * f[i]
      const logPhi = normalLogCdf(z)
      logLik += logPhi
      const r = Math.exp(Math.log(normalPdf(z)) - logPhi)
      grad[i] = y * r
      W[i] = r * r + z * r
    }
  }
  return { logLik, grad, W }
}

/** A state of the Laplace mode search. */
export type LaplaceState = Status & {
  /** Newton steps taken. */
  t: number
  /** Latent values $\fvec$ at the training inputs, `[n]`. */
  f: Tensor
  /** $\avec$ with $\fvec = \Kmat\avec$ (R&W Algorithm 3.1). */
  a: Tensor
  /**
   * $\Psi(\fvec) = \log p(\yvec \mid \fvec) - \frac{1}{2} \fvec^\top\Kmat^{-1}\fvec$, the objective Newton's method
   * increases.
   */
  objective: number
  /** The approximate log marginal likelihood at $\fvec$ (eq. 3.32). */
  logMarginal: number
  /** True once a step changed $\Psi$ by less than the tolerance. */
  converged: boolean
}

/**
 * A factorised likelihood for the Laplace approximation: at latent values $\fvec$ (`[n]`), $\log p(\yvec \mid \fvec)$,
 * its gradient and the negative Hessian diagonal $\Wmat$ (non-negative for a log-concave likelihood).
 */
export type LaplaceTerms = (f: Float64Array) => { logLik: number; grad: Float64Array; W: Float64Array }

/** The problem a Laplace mode search solves. */
export type LaplaceProblem = {
  /** The prior covariance $\Kmat$ of the latent values, $n \times n$. */
  K: Tensor
  /** Labels, `[n]`: 0 or 1 for the named binary likelihoods; whatever `likelihood` reads when it is a function. */
  labels: Tensor
  /** A binary link by name, or any factorised log-concave likelihood (e.g. the ordinal one of `./ordinal`). */
  likelihood: ClassificationLikelihood | LaplaceTerms
  /** Stop when $\Psi$ changes by less than this in a step (default 1e-10, as scikit-learn). */
  tolerance?: number
}

/**
 * The quantities at $\fvec$ that a Newton step and the evidence need: the likelihood terms, $\Wmat^{1/2}$, and the
 * Cholesky factor of $\Bmat = \Imat + \Wmat^{1/2}\Kmat\Wmat^{1/2}$ with $\frac{1}{2} \log\lvert \Bmat \rvert$.
 *
 * @param problem The Gram matrix, labels and likelihood.
 * @param f The latent values $\fvec$, `[n]`.
 * @returns `logLik`, `grad` and `W` as the likelihood gives them; `sW`, $\sqrt{W_{ii}}$; `K`, the Gram matrix as a
 *   row-major array; `L`, the factor of $\Bmat$ (with `jitter`, any jitter it needed); and `logDetHalf`,
 *   $\frac{1}{2} \log\lvert \Bmat \rvert$.
 */
export function laplaceAt(problem: LaplaceProblem, f: Float64Array) {
  const n = f.length
  const { logLik, grad, W } =
    typeof problem.likelihood === 'function'
      ? problem.likelihood(f)
      : likelihoodTerms(problem.likelihood, Float64Array.from(toFlat(problem.labels)), f)
  const sW = W.map(Math.sqrt)
  const K = Float64Array.from(toFlat(problem.K))
  const B = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) B[i * n + j] = (i === j ? 1 : 0) + sW[i] * K[i * n + j] * sW[j]
  const { L, jitter } = cholesky(fromData(B, [n, n]))
  const Lf = Float64Array.from(toFlat(L))
  let logDetHalf = 0
  for (let i = 0; i < n; i++) logDetHalf += Math.log(Lf[i * n + i])
  return { logLik, grad, W, sW, K, L, jitter, logDetHalf }
}

/**
 * Newton's method for the mode of $p(\fvec \mid \Xmat, \yvec)$ under a GP prior with Gram matrix $\Kmat$ and a
 * logistic or probit likelihood (Rasmussen and Williams, 2006, Algorithm 3.1). Each step is
 * $\bvec = \Wmat\fvec + \nabla \log p(\yvec \mid \fvec)$,
 * $\avec = \bvec - \Wmat^{1/2} \Lmat^{-\top} \Lmat^{-1} \Wmat^{1/2} \Kmat \bvec$ and $\fvec = \Kmat\avec$, with $\Lmat$
 * the Cholesky factor of $\Imat + \Wmat^{1/2} \Kmat \Wmat^{1/2}$. The objective is concave, so plain Newton steps
 * converge; a step that leaves it non-finite marks the state `diverged`. `init` takes an optional starting $\fvec$
 * (default $\zeros$).
 *
 * @param problem The Gram matrix $\Kmat$, the labels, the likelihood and the tolerance.
 * @returns The algorithm; run it with `run` or `trace`.
 *
 * @example Three points: $\Psi$ rises to the mode in a few steps
 * // The RBF Gram matrix at 0, 1 and 2 (lengthscale 1), and the labels 0, 0, 1
 * const K = tensor([[1, 0.61, 0.14], [0.61, 1, 0.61], [0.14, 0.61, 1]])
 * const labels = tensor([0, 0, 1])
 * const mode = laplaceMode({ K, labels, likelihood: 'logistic' })
 * const tr = trace(mode, {}, 20, { record: { objective: (s) => s.objective } })
 * print('objective', tr.series.objective)
 * print('mode', tr.final.f, ' log q(y | X)', tr.final.logMarginal, ' converged', tr.final.converged)
 */
export function laplaceMode(problem: LaplaceProblem): Algorithm<{ f?: Tensor }, LaplaceState> {
  const tol = problem.tolerance ?? 1e-10
  const n = problem.K.shape[0]
  const stateAt = (f: Float64Array, a: Float64Array, t: number, previous: number): LaplaceState => {
    const q = laplaceAt(problem, f)
    let fa = 0
    for (let i = 0; i < n; i++) fa += f[i] * a[i]
    const objective = q.logLik - 0.5 * fa
    return {
      t,
      f: fromData(f, [n]),
      a: fromData(a, [n]),
      objective,
      logMarginal: objective - q.logDetHalf,
      converged: t > 0 && Math.abs(objective - previous) < tol,
      diverged: !Number.isFinite(objective),
    }
  }
  return {
    name: 'gp-laplace-mode',
    init: ({ f } = {}) => {
      const f0 = f ? Float64Array.from(toFlat(f)) : new Float64Array(n)
      // a = K⁻¹f; from 0 it is 0, otherwise solve through the factor of K (+ jitter).
      let a0 = new Float64Array(n)
      if (f) {
        const { L } = cholesky(problem.K)
        a0 = Float64Array.from(toFlat(solveTriangular(L, solveTriangular(L, f), { transpose: true }) as Tensor))
      }
      return stateAt(f0, a0, 0, -Infinity)
    },
    step: (state) => {
      const f = Float64Array.from(toFlat(state.f))
      const q = laplaceAt(problem, f)
      const b = Float64Array.from(f, (fi, i) => q.W[i] * fi + q.grad[i])
      // W^½ K b
      const Kb = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += q.K[i * n + j] * b[j]
        Kb[i] = q.sW[i] * s
      }
      const inner = toFlat(solveTriangular(q.L, solveTriangular(q.L, fromData(Kb, [n])), { transpose: true }) as Tensor)
      const a = Float64Array.from(b, (bi, i) => bi - q.sW[i] * inner[i])
      const fNew = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += q.K[i * n + j] * a[j]
        fNew[i] = s
      }
      return stateAt(fNew, a, state.t + 1, state.objective)
    },
  }
}

/**
 * $\partial^3 \log p(y_i \mid f_i) / \partial f_i^3$ elementwise, for the implicit term of the Laplace evidence
 * gradient.
 *
 * @param likelihood The link: logistic or probit.
 * @param t The labels, 0 or 1, one per latent value.
 * @param f The latent values $\fvec$, `[n]`.
 * @returns The third derivatives, `[n]`.
 */
function thirdDerivative(likelihood: ClassificationLikelihood, t: Float64Array, f: Float64Array): Float64Array {
  return Float64Array.from(f, (fi, i) => {
    const y = 2 * t[i] - 1
    if (likelihood === 'logistic') {
      const p = sigmoid(fi)
      return -p * (1 - p) * (1 - 2 * p)
    }
    // r = N(z)/Φ(z), z = yf: r' = −zr − r², r'' = −r − zr' − 2rr', and ∂³ log Φ(yf)/∂f³ = y r''(z).
    const z = y * fi
    const r = Math.exp(Math.log(normalPdf(z)) - normalLogCdf(z))
    const r1 = -z * r - r * r
    return y * (-r - z * r1 - 2 * r * r1)
  })
}

/** Options of `laplaceEvidence`. */
export type LaplaceEvidenceOptions = {
  /** The link (default logistic). */
  likelihood?: ClassificationLikelihood
  /** Most Newton steps (default 100). */
  maxSteps?: number
  /** Newton tolerance on $\Psi$ (default 1e-10). */
  tolerance?: number
}

/**
 * The Laplace log marginal likelihood (R&W eq. 3.32) as a differentiable function of the Gram matrix $\Kmat$ (labels 0
 * and 1). The mode $\hat\fvec$ depends on $\Kmat$, so the reverse rule adds the implicit term of Rasmussen and Williams
 * (2006), Algorithm 5.1: with $\avec = \nabla \log p(\yvec \mid \hat\fvec)$,
 * $\Rmat = \Wmat^{1/2} \Bmat^{-1} \Wmat^{1/2}$ and
 * $\svec_2 = \frac{1}{2} \diag((\Kmat^{-1} + \Wmat)^{-1}) \nabla^3 \log p(\yvec \mid \hat\fvec)$, the gradient is
 * $\frac{1}{2} \avec\avec^\top - \frac{1}{2} \Rmat + \frac{1}{2}(\uvec\avec^\top + \avec\uvec^\top)$ with
 * $\uvec = (\Imat - \Rmat\Kmat) \svec_2$ (the mode moves by $(\Imat + \Kmat\Wmat)^{-1} \partial\Kmat\, \avec$).
 * Checked against finite differences. Successive calls start Newton from the previous mode, and from $\zeros$ again
 * when that gives a non-finite value; a search that has not converged in `maxSteps` is used where it stopped.
 *
 * @param labels The labels, 0 or 1, `[n]`.
 * @param options The link, the most Newton steps and the tolerance.
 * @returns The function from $\Kmat$ ($n \times n$, possibly traced) to the approximate log marginal likelihood.
 *
 * @example The evidence and its derivative along $c\Kmat$, against a central difference
 * const K = tensor([[1, 0.61, 0.14], [0.61, 1, 0.61], [0.14, 0.61, 1]])
 * const evidence = laplaceEvidence(tensor([0, 0, 1]))
 * print('log q(y | K)', evidence(K))
 * print('d/dc at c = 1', grad((c) => evidence(mul(c, K)))(1))
 * print('central difference', (evidence(mul(1.001, K)) - evidence(mul(0.999, K))) / 0.002)
 */
export function laplaceEvidence(labels: Tensor, options: LaplaceEvidenceOptions = {}): (K: Value) => Value {
  const { likelihood = 'logistic', maxSteps = 100, tolerance = 1e-10 } = options
  const t = Float64Array.from(toFlat(labels))
  let warm: Tensor | null = null
  const solve = (K: Tensor) => {
    const problem: LaplaceProblem = { K, labels, likelihood, tolerance }
    let final = run(laplaceMode(problem), warm ? { f: warm } : {}, maxSteps)
    if (!Number.isFinite(final.logMarginal) && warm) final = run(laplaceMode(problem), {}, maxSteps)
    if (Number.isFinite(final.logMarginal)) warm = final.f
    return { final, problem }
  }
  const gradient = (problem: LaplaceProblem, final: LaplaceState): Tensor => {
    const f = Float64Array.from(toFlat(final.f))
    const n = f.length
    const q = laplaceAt(problem, f)
    const factor = stableFactor(q.K, q.sW)
    const R = factor.R()
    // diag((K⁻¹ + W)⁻¹) = diag(K) − colsum(C²), C = L⁻¹ W^½ K.
    const SK = new Float64Array(n * n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) SK[i * n + j] = q.sW[i] * q.K[i * n + j]
    const C = factor.forward(SK, n)
    const d3 = thirdDerivative(likelihood, t, f)
    const s2 = new Float64Array(n)
    for (let j = 0; j < n; j++) {
      let c2 = 0
      for (let k = 0; k < n; k++) c2 += C[k * n + j] ** 2
      s2[j] = 0.5 * (q.K[j * n + j] - c2) * d3[j]
    }
    // u = s₂ − R K s₂.
    const Ks2 = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Ks2[i] += q.K[i * n + j] * s2[j]
    const u = Float64Array.from(s2)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) u[i] -= R[i * n + j] * Ks2[j]
    const a = q.grad
    const G = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        G[i * n + j] = 0.5 * a[i] * a[j] - 0.5 * R[i * n + j] + 0.5 * (u[i] * a[j] + a[i] * u[j])
    return fromData(G, [n, n])
  }
  return customVjp(
    (K: Value) => solve(K as Tensor).final.logMarginal as Value,
    (K: Value) => {
      const { final, problem } = solve(K as Tensor)
      return { out: final.logMarginal as Value, residuals: gradient(problem, final) }
    },
    (G: Tensor, cot: Value) => [mul(cot, G)],
  )
}

// ── The estimator ────────────────────────────────────────────────────────────────────────────────────────────────

/** How the non-Gaussian posterior is approximated: Laplace (mode and curvature) or EP (moment matching, probit). */
export type GpClassificationMethod = 'laplace' | 'ep'

/** Options of `gpClassifier`. */
export type GpClassifierParams<P extends KernelParams = KernelParams> = {
  /** The kernel $k$ (required), at fixed hyperparameters (fit them with `fitGpClassifier`). */
  kernel: Kernel<P>
  /** The posterior approximation (default 'laplace'). */
  method?: GpClassificationMethod
  /** Default logistic for Laplace; EP supports the probit link only (and defaults to it). */
  likelihood?: ClassificationLikelihood
  /** Most Newton steps for Laplace (default 100), most sweeps for EP (default 100). */
  maxSteps?: number
  /** Newton convergence tolerance on $\Psi$ (default 1e-10), or EP's site-change tolerance (default 1e-8). */
  tolerance?: number
  /** EP damping in $[0, 1)$ (default 0). */
  damping?: number
}

/** A fitted GP classifier. */
export interface GpClassifierModel<P extends KernelParams = KernelParams>
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Univariate<Tensor>>,
    Expects<Tensor>,
    Samples<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gp-classifier'
  /** The kernel $k$. */
  readonly kernel: Kernel<P>
  /** The posterior approximation used. */
  readonly method: GpClassificationMethod
  /** The link used. */
  readonly likelihood: ClassificationLikelihood
  /** The posterior mean of $\fvec$ at the training inputs, `[n]`: the mode $\hat\fvec$ for Laplace, $\muvec$ for EP. */
  readonly mode: Tensor
  /** The approximate $\log p(\yvec \mid \Xmat)$: Laplace's eq. 3.32 or EP's eq. 3.65. */
  readonly logMarginal: number
  /** Whether Newton's method (or EP) converged within `maxSteps`. */
  readonly converged: boolean
  /**
   * The Newton run (Laplace, every step, recording `objective` and `logMarginal`) or the site updates (EP, every
   * sweep, recording `logMarginal` and `change`).
   */
  readonly training: Trace<LaplaceState> | Trace<MvEpState>
  /** Mean and variance of the approximate latent posterior $q(f_* \mid \yvec)$ at `xs`. */
  latent(xs: Tensor): { mean: Tensor; variance: Tensor }
}

/**
 * The 32-point (probabilists') Gauss–Hermite rule with weights normalised to sum to 1, so that
 * $\expect[g(Z)] \approx \sum_k w_k g(z_k)$ for $Z \sim \Gauss(0, 1)$.
 *
 * @returns The nodes $z_k$ and weights $w_k$.
 */
function ruleForExpectation() {
  // 32-point Gauss–Hermite (probabilists') rule, weights normalised to sum to 1: E[g(Z)], Z ~ N(0, 1).
  const rule = gaussHermite(32, { probabilists: true })
  const nodes = toFlat(rule.nodes)
  const raw = toFlat(rule.weights)
  const total = raw.reduce((a, b) => a + b, 0)
  return { nodes, weights: raw.map((w) => w / total) }
}

/**
 * Binary GP classification (labels 0 and 1) by the Laplace approximation (R&W Algorithms 3.1–3.2) or by EP with the
 * probit link (Algorithms 3.5–3.6). Either way $q(\fvec \mid \yvec)$ is Gaussian, and the latent predictive has mean
 * $\kvec_*^\top\alphavec$ and variance $k_{**} - \vvec^\top\vvec$ with $\vvec = \Lmat^{-1} \Smat \kvec_*$,
 * $\Smat = \Wmat^{1/2}$ (Laplace) or $\tilde\Smat^{1/2}$ (EP). Capabilities: `forward` and `score` (the latent mean),
 * `predictive` (Bernoulli with $\pi_* = \int \sigma(f_*) q(f_*) \, df_*$, exactly $\Phi(\mu / \sqrt{1 + v})$ for the
 * probit link and by 32-point Gauss–Hermite quadrature for the logistic one; R&W eq. 3.25), `decide`
 * ($\pi_* > \frac{1}{2}$), `expect`, `sample`. Throws `DomainError` for EP with the logistic link; `fit` throws
 * `ShapeError` when the numbers of inputs and labels differ and `DomainError` for a label other than 0 or 1.
 *
 * @param params The kernel, the method and link, and the iteration limits.
 * @returns The estimator: `fit({ x, y })` on inputs `[n, d]` (or `[n]`) and labels `[n]` returns a
 *   `GpClassifierModel`.
 *
 * @example The probability of class 1 on either side of the boundary, by Laplace and by EP
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[-2], [-1.5], [-1], [-0.5], [0.5], [1], [1.5], [2]])
 * const y = tensor([0, 0, 0, 1, 0, 1, 1, 1])
 * const xs = tensor([[-1.5], [1.5]])
 * const laplace = gpClassifier({ kernel: rbf({ lengthscale: 1, variance: 4 }) }).fit({ x, y })
 * print('Laplace P(y = 1)', laplace.predictive(xs).mean(), ' decision', laplace.decide(xs))
 * const ep = gpClassifier({ kernel: rbf({ lengthscale: 1, variance: 4 }), method: 'ep' }).fit({ x, y })
 * print('EP P(y = 1)', ep.predictive(xs).mean(), ' latent', ep.latent(xs))
 */
export function gpClassifier<P extends KernelParams>(
  params: GpClassifierParams<P>,
): Estimator<Supervised<Tensor, Tensor>, GpClassifierModel<P>> {
  const { kernel, method = 'laplace' } = params
  const likelihood = params.likelihood ?? (method === 'ep' ? 'probit' : 'logistic')
  if (method === 'ep' && likelihood !== 'probit')
    throw new DomainError('gpClassifier', 'gpClassifier: EP supports the probit link only')
  const maxSteps = params.maxSteps ?? 100
  const { nodes, weights } = ruleForExpectation()
  return {
    name: 'gp-classifier',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      const X = asRows(x) as Tensor
      const n = X.shape[0]
      const t = Float64Array.from(toFlat(y))
      if (t.length !== n) throw new ShapeError('gpClassifier', `gpClassifier: ${n} inputs but ${t.length} labels`)
      if (!t.every((v) => v === 0 || v === 1))
        throw new DomainError('gpClassifier', 'gpClassifier: labels must be 0 or 1')
      const K = gram(kernel, X) as Tensor
      const labels = fromData(t, [n])
      let alpha: Float64Array, s: Float64Array, factor: ReturnType<typeof stableFactor>
      let fitted: {
        mode: Tensor
        logMarginal: number
        converged: boolean
        training: Trace<LaplaceState> | Trace<MvEpState>
      }
      if (method === 'laplace') {
        const problem: LaplaceProblem = { K, labels, likelihood, tolerance: params.tolerance ?? 1e-10 }
        const training: Trace<LaplaceState> = trace(laplaceMode(problem), {}, maxSteps, {
          every: options.trace?.every ?? 1,
          record: { objective: (st) => st.objective, logMarginal: (st) => st.logMarginal },
        })
        const final = training.final
        const q = laplaceAt(problem, Float64Array.from(toFlat(final.f)))
        alpha = q.grad
        s = q.sW
        factor = stableFactor(q.K, s)
        fitted = { mode: final.f, logMarginal: final.logMarginal, converged: final.converged, training }
      } else {
        const alg = gpEp({ K, labels, tolerance: params.tolerance ?? 1e-8, damping: params.damping })
        const training: Trace<MvEpState> = trace(alg, {}, maxSteps * n, {
          every: options.trace?.every ?? n,
          keep: 'none',
          record: { logMarginal: (st) => st.logEvidence, change: (st) => st.change },
        })
        const final = training.final
        const w = epWeights(
          Float64Array.from(toFlat(K)),
          Float64Array.from(toFlat(final.sitePrecision)),
          Float64Array.from(toFlat(final.siteShift)),
        )
        alpha = w.alpha
        s = w.s
        factor = w.factor
        fitted = { mode: final.mean, logMarginal: final.logEvidence, converged: final.converged, training }
      }
      const latent = (xs: Tensor) => {
        const S = asRows(xs) as Tensor
        const Ks = gram(kernel, X, S) as Tensor // [n, m]
        const mean = matmul(fromData(alpha, [n]), Ks) as Tensor
        const m = S.shape[0]
        const ks = Float64Array.from(toFlat(Ks))
        const scaled = new Float64Array(n * m)
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) scaled[i * m + j] = s[i] * ks[i * m + j]
        const V = factor.forward(scaled, m)
        const kss = toFlat(kernelDiagonal(kernel, S) as Tensor)
        const variance = Float64Array.from(kss, (k, j) => {
          let acc = 0
          for (let i = 0; i < n; i++) acc += V[i * m + j] ** 2
          return Math.max(k - acc, 0)
        })
        return { mean, variance: fromData(variance, [m]) }
      }
      const probability = (xs: Tensor) => {
        const { mean, variance } = latent(xs)
        const mu = toFlat(mean)
        const v = toFlat(variance)
        return fromData(
          Float64Array.from(mu, (m, j) => {
            if (likelihood === 'probit') return normalCdf(m / Math.sqrt(1 + v[j]))
            let acc = 0
            for (let k = 0; k < nodes.length; k++) acc += weights[k] * sigmoid(m + Math.sqrt(v[j]) * nodes[k])
            return acc
          }),
          [mu.length],
        )
      }
      const forward = (xs: Tensor) => latent(xs).mean
      const base = {
        kind: 'model' as const,
        name: 'gp-classifier' as const,
        kernel,
        method,
        likelihood,
        ...fitted,
        latent,
        forward,
        score: forward,
        decide: (xs: Tensor) =>
          fromData(
            Int32Array.from(toFlat(probability(xs)), (p) => (p > 0.5 ? 1 : 0)),
            [(asRows(xs) as Tensor).shape[0]],
          ),
        predictive: (xs: Tensor) => Bernoulli(probability(xs)) as Univariate<Tensor>,
      }
      return withSampling(withExpectation(base))
    },
  }
}

// ── Hyperparameters ──────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the evidence functions and the fit. */
export type GpClassifierEvidenceOptions = {
  /** The posterior approximation (default 'laplace'). */
  method?: GpClassificationMethod
  /** Default logistic for Laplace, probit for EP (the only link EP supports). */
  likelihood?: ClassificationLikelihood
  /** Most Newton steps (Laplace, default 100) or sweeps (EP, default 100). */
  maxSteps?: number
}

/**
 * The evidence as a differentiable function of $\Kmat$ for the chosen method: `laplaceEvidence`, or `gpEpEvidence`
 * with a site tolerance of 1e-10. Throws `DomainError` for EP with the logistic link.
 *
 * @param labels The labels, 0 or 1, `[n]`.
 * @param options The method, the link and the most Newton steps or EP sweeps.
 * @returns The function from $\Kmat$ to the approximate log marginal likelihood.
 */
function evidenceFor(labels: Tensor, options: GpClassifierEvidenceOptions): (K: Value) => Value {
  const { method = 'laplace', maxSteps = 100 } = options
  const likelihood = options.likelihood ?? (method === 'ep' ? 'probit' : 'logistic')
  if (method === 'ep') {
    if (likelihood !== 'probit') throw new DomainError('gpClassifier', 'gpClassifier: EP supports the probit link only')
    return gpEpEvidence(labels, { maxSweeps: maxSteps, tolerance: 1e-10 })
  }
  return laplaceEvidence(labels, { likelihood, maxSteps })
}

/**
 * The labels as a float vector `[n]`.
 *
 * @param y The labels, any shape holding $n$ values.
 * @param n The number of training inputs.
 * @returns The labels, `[n]`.
 */
const labelsOf = (y: Tensor, n: number) => fromData(Float64Array.from(toFlat(y)), [n])

/**
 * The Laplace approximation to $\log p(\yvec \mid \Xmat, \thetavec)$ for a kernel (R&W eq. 3.32): Newton's method to
 * the mode, then $\Psi(\hat\fvec) - \frac{1}{2} \log\lvert \Bmat \rvert$. Labels 0 and 1. A search that has not
 * converged in `maxSteps` is used where it stopped.
 *
 * @param kernel The kernel $k$.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The labels, 0 or 1, `[n]`.
 * @param options The link and the iteration limit.
 * @param options.likelihood The link.
 * @param options.maxSteps The most Newton steps.
 * @returns The approximate log marginal likelihood.
 *
 * @example The evidence over lengthscales, by Laplace and by EP
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[-2], [-1.5], [-1], [-0.5], [0.5], [1], [1.5], [2]])
 * const y = tensor([0, 0, 0, 1, 0, 1, 1, 1])
 * for (const l of [0.3, 1, 3]) {
 *   const k = rbf({ lengthscale: l, variance: 1 })
 *   print('lengthscale', l, ' Laplace', laplaceLogMarginal(k, x, y), ' EP', gpEpLogMarginal(k, x, y))
 * }
 */
export function laplaceLogMarginal(
  kernel: Kernel,
  x: Tensor,
  y: Tensor,
  { likelihood = 'logistic', maxSteps = 100 }: { likelihood?: ClassificationLikelihood; maxSteps?: number } = {},
): number {
  const X = asRows(x) as Tensor
  const n = X.shape[0]
  const K = gram(kernel, X) as Tensor
  return run(laplaceMode({ K, labels: labelsOf(y, n), likelihood }), {}, maxSteps).logMarginal
}

/**
 * EP's approximation to $\log p(\yvec \mid \Xmat, \thetavec)$ for a kernel with the probit link (R&W eq. 3.65).
 * Labels 0 and 1. NaN when EP has not finished a sweep in which every site was updated.
 *
 * @param kernel The kernel $k$.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The labels, 0 or 1, `[n]`.
 * @param options The most sweeps, the site-change tolerance and the damping.
 * @returns The EP log marginal likelihood.
 *
 * @example EP and Laplace agree closely on the probit link
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[-2], [-1.5], [-1], [-0.5], [0.5], [1], [1.5], [2]])
 * const y = tensor([0, 0, 0, 1, 0, 1, 1, 1])
 * const k = rbf({ lengthscale: 1, variance: 1 })
 * print('EP', gpEpLogMarginal(k, x, y), ' Laplace (probit)', laplaceLogMarginal(k, x, y, { likelihood: 'probit' }))
 */
export function gpEpLogMarginal(kernel: Kernel, x: Tensor, y: Tensor, options: GpEpOptions = {}): number {
  const X = asRows(x) as Tensor
  const n = X.shape[0]
  const K = gram(kernel, X) as Tensor
  return run(
    gpEp({ K, labels: labelsOf(y, n), tolerance: options.tolerance, damping: options.damping }),
    {},
    (options.maxSweeps ?? 100) * n,
  ).logEvidence
}

/** The approximate log evidence and its gradient in the kernel's hyperparameters. */
export type GpClassifierEvidenceGradient<P extends KernelParams = KernelParams> = {
  /** The approximate log evidence. */
  value: number
  /** $\partial / \partial \theta$ for each hyperparameter, shaped like `kernel.params`. */
  kernel: P
  /** The hyperparameters' names, in the order of `logGradient`. */
  names: string[]
  /** $\partial / \partial \log \theta$ for every hyperparameter. */
  logGradient: Float64Array
}

/**
 * The Laplace or EP log evidence and its gradient in the kernel's hyperparameters: reverse-mode differentiation
 * through the Gram matrix, with the evidence's own rule in $\Kmat$ (`laplaceEvidence`, `gpEpEvidence`).
 *
 * @param kernel The kernel $k$; every hyperparameter must be positive.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The labels, 0 or 1, `[n]`.
 * @param options The method, the link and the most Newton steps or EP sweeps.
 * @returns The evidence and its gradient, as a tree and in log space.
 *
 * @example The lengthscale derivative against a central difference
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[-2], [-1.5], [-1], [-0.5], [0.5], [1], [1.5], [2]])
 * const y = tensor([0, 0, 0, 1, 0, 1, 1, 1])
 * const g = gpClassifierEvidenceGradient(rbf({ lengthscale: 1, variance: 1 }), x, y)
 * print('log q(y | X)', g.value, ' d/d lengthscale', g.kernel.lengthscale)
 * const at = (l) => laplaceLogMarginal(rbf({ lengthscale: l, variance: 1 }), x, y)
 * print('central difference', (at(1.001) - at(0.999)) / 0.002)
 */
export function gpClassifierEvidenceGradient<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  options: GpClassifierEvidenceOptions = {},
): GpClassifierEvidenceGradient<P> {
  const X = asRows(x) as Tensor
  const evidence = evidenceFor(labelsOf(y, X.shape[0]), options)
  const lv = kernelLogVector(kernel)
  const { value, grad } = valueAndGrad((tree: P) => evidence(gram(kernelFromLog(kernel, tree), X)))(
    lv.unravel(lv.vector),
  )
  const logGradient = Float64Array.from(ravel(grad as P).vector)
  const params = ravel(kernel.params)
  return {
    value: value as number,
    kernel: params.unravel(Float64Array.from(params.vector, (theta, i) => logGradient[i] / theta)),
    names: lv.names,
    logGradient,
  }
}

/** Options of `fitGpClassifier`. */
export type FitGpClassifierOptions = GpClassifierEvidenceOptions & {
  /** Most L-BFGS steps (default 100). */
  maxIterations?: number
  /** L-BFGS gradient tolerance (default 1e-5). */
  tolerance?: number
  /**
   * The sd of an optional Gaussian hyperprior on each log hyperparameter, centred on its starting value: the fit
   * then maximises $\log q(\yvec \mid \Xmat, \thetavec) + \log p(\log \thetavec)$. Useful on near-separable data,
   * where the evidence keeps rising with the signal variance. Default none.
   */
  logPriorScale?: number
}

/** The result of `fitGpClassifier`. */
export type GpClassifierFit<P extends KernelParams = KernelParams> = {
  /** The kernel at the fitted hyperparameters. */
  kernel: Kernel<P>
  /** The approximate log evidence at the fitted hyperparameters (without the hyperprior). */
  logMarginal: number
  /** The L-BFGS run over log hyperparameters; `value` is the negated objective, with `logMarginal` recorded. */
  training: Trace<LbfgsState>
  /** The hyperparameters' names, in the order of `training`'s `x` (log space). */
  names: string[]
  /** Whether the L-BFGS gradient norm reached `tolerance`. */
  converged: boolean
}

/**
 * Type-II maximum likelihood for GP classification: maximise the Laplace or EP log evidence over the kernel's
 * hyperparameters in log space by L-BFGS (R&W §5.5.1 and §5.5.2), with gradients from `valueAndGrad` through the
 * Gram matrix and the evidence's rule in $\Kmat$ (the implicit dependence of the Laplace mode on $\thetavec$, R&W
 * Algorithm 5.1; the EP fixed point, eq. 5.27). The run is a trace, so the path can be played step by step. A point
 * where the evidence cannot be computed counts as $+\infty$ in the objective.
 *
 * @param kernel The kernel at the starting hyperparameters (all positive).
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The labels, 0 or 1, `[n]`.
 * @param options The method and link, the inner and outer iteration limits, the tolerance and the hyperprior.
 * @returns The fitted kernel, its evidence and the L-BFGS trace.
 *
 * @example From a too-short lengthscale, the fit raises the Laplace evidence
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[-2], [-1.5], [-1], [-0.5], [0.5], [1], [1.5], [2]])
 * const y = tensor([0, 0, 0, 1, 0, 1, 1, 1])
 * const fit = fitGpClassifier(rbf({ lengthscale: 0.3, variance: 1 }), x, y)
 * print('lengthscale', fit.kernel.params.lengthscale, ' variance', fit.kernel.params.variance)
 * print('log q(y | X) from', toFlat(fit.training.series.logMarginal)[0], 'to', fit.logMarginal)
 */
export function fitGpClassifier<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  options: FitGpClassifierOptions = {},
): GpClassifierFit<P> {
  const { maxIterations = 100, tolerance = 1e-5, logPriorScale } = options
  const X = asRows(x) as Tensor
  const evidence = evidenceFor(labelsOf(y, X.shape[0]), options)
  const lv = kernelLogVector(kernel)
  const centre = Float64Array.from(lv.vector)
  const negative = valueAndGrad((tree: P) => mul(-1, evidence(gram(kernelFromLog(kernel, tree), X))))
  const logMarginals = new Map<string, number>()
  const objective = (theta: Tensor) => {
    const th = toFlat(theta)
    const k = th.length
    try {
      const { value, grad } = negative(lv.unravel(th))
      let v = value as number
      if (!Number.isFinite(v)) return { value: Infinity, grad: fromData(new Float64Array(k), [k]) }
      logMarginals.set(Array.from(th).join(','), -v)
      const g = Float64Array.from(ravel(grad as P).vector)
      if (logPriorScale !== undefined) {
        const s2 = logPriorScale * logPriorScale
        for (let i = 0; i < k; i++) {
          v += (0.5 * (th[i] - centre[i]) ** 2) / s2
          g[i] += (th[i] - centre[i]) / s2
        }
      }
      return { value: v, grad: fromData(g, [k]) }
    } catch {
      return { value: Infinity, grad: fromData(new Float64Array(k), [k]) }
    }
  }
  const training = trace(lbfgs(objective, { tolerance }), { x0: fromData(centre, [centre.length]) }, maxIterations, {
    record: { logMarginal: (s: LbfgsState) => logMarginals.get(Array.from(toFlat(s.x)).join(',')) ?? NaN },
  })
  const final = training.final
  return {
    kernel: kernelFromLog(kernel, lv.unravel(toFlat(final.x))),
    logMarginal: logMarginals.get(Array.from(toFlat(final.x)).join(',')) ?? -final.value,
    training,
    names: lv.names,
    converged: final.converged,
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'gpClassifier',
    module: 'learning/gaussian-processes',
    name: 'Gaussian process classifier',
    summary: 'Binary GP classification by the Laplace approximation or EP; the kernel is a required argument.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'score', 'sample'],
    hyper: space({ method: oneOf(['laplace', 'ep']), maxSteps: int(1, 1000, { default: 100 }) }),
    notes: ['gaussian-process-classification', 'expectation-propagation-gaussian-process-classification'],
    cite: ['rasmussen2006'],
  },
  gpClassifier,
)
