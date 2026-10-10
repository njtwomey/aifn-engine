/**
 * The relevance vector machine for regression (Tipping, 2001, "Sparse Bayesian learning and the relevance vector
 * machine", JMLR 1): $y = \sum_i w_i \phi_i(\xvec) + \varepsilon$ with a bias $\phi_0 = 1$ and one kernel basis
 * function $\phi_i(\xvec) = k(\xvec, \xvec_i)$ per training input, $w_i \sim \Gauss(0, \alpha_i^{-1})$ and
 * $\varepsilon \sim \Gauss(0, \beta^{-1})$. For an active set $\Mcal$ of basis functions (those with finite
 * $\alpha_i$), with $\Amat = \diag(\alpha_i)$ and $\Phimat_{\Mcal}$ the design matrix's active columns, the weight
 * posterior has covariance $\Sigmamat = (\Amat + \beta\Phimat_{\Mcal}^\top\Phimat_{\Mcal})^{-1}$ and mean
 * $\muvec = \beta\Sigmamat\Phimat_{\Mcal}^\top\yvec$, and the log marginal likelihood is
 * $\mathcal{L}(\alphavec, \beta) = \frac{1}{2}[N \ln \beta + \sum_{\Mcal} \ln \alpha_i - \ln\abs{\Sigmamat^{-1}} - E]$,
 * with $E = \beta\lVert \yvec - \Phimat_{\Mcal}\muvec \rVert^2 + \muvec^\top\Amat\muvec + N \ln 2\pi$.
 *
 * Two fits, both traceable algorithms whose states carry the active set, $\alphavec$, $\beta$, the log marginal
 * likelihood and what the step did:
 *
 * - `rvmFastSteps`: Tipping and Faul's (2003) fast marginal-likelihood maximisation. From one basis function, each
 *   step takes the single change with the largest exact gain $\Delta\mathcal{L}$: add an inactive $\phi_i$,
 *   re-estimate an active $\alpha_i$, delete an active $\phi_i$, or re-estimate $\beta$. The gains come from the
 *   sparsity and quality factors $s_i = \phivec_i^\top\Cmat_{-i}^{-1}\phivec_i$ and
 *   $q_i = \phivec_i^\top\Cmat_{-i}^{-1}\yvec$, with
 *   $\ell(\alpha_i) = \frac{1}{2}[\ln \alpha_i - \ln(\alpha_i + s_i) + q_i^2/(\alpha_i + s_i)]$ the part of
 *   $\mathcal{L}$ that depends on $\alpha_i$, maximised at $\alpha_i = s_i^2/(q_i^2 - s_i)$ when $q_i^2 > s_i$ and at
 *   $\infty$ otherwise. A $\beta$ update is taken only when it raises $\mathcal{L}$, so $\mathcal{L}$ never decreases.
 * - `rvmReestimationSteps`: the original re-estimation, all basis functions at once:
 *   $\gamma_i = 1 - \alpha_i\Sigma_{ii}$, $\alpha_i \leftarrow \gamma_i/\mu_i^2$,
 *   $\beta \leftarrow (N - \sum_i \gamma_i)/\lVert \yvec - \Phimat\muvec \rVert^2$ (MacKay, 1992), pruning $\phi_i$
 *   when $\alpha_i$ passes a threshold.
 *
 * Linear algebra is aifn's: the Cholesky factor of $\Sigmamat^{-1}$ and triangular solves, with $S$ and $Q$ for every
 * basis function from one solve against the $M \times B$ block of $\Phimat^\top\Phimat$, so a step costs
 * $O(BM^2 + M^3)$ after the $O(NB^2)$ set-up.
 */

import { fromData, matmul, reshape, toFlat, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { asRows, gram, type Kernel } from 'aifn-compute/learning/kernels'
import { cholesky, solveTriangular } from 'aifn-compute/numerics/linalg'
import {
  defineModel,
  withExpectation,
  type Decides,
  type Estimator,
  type Expects,
  type Fitted,
  type Predicts,
  type Supervised,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { Normal, type Univariate } from 'aifn-compute/probability/distributions'
import { bool, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { definer, type AlgorithmInfo } from 'aifn-compute/foundation/registry'
import { ShapeError } from 'aifn-compute/foundation/errors'

const LOG_2PI = Math.log(2 * Math.PI)

// ── The problem ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** The RVM's basis on the training data, with the products every step reuses. */
export interface RvmProblem {
  /** The kernel $k$ of the basis functions. */
  readonly kernel: Kernel
  /** Training inputs, `[N, d]`. */
  readonly x: Tensor
  /** Training targets, `[N]`. */
  readonly y: Tensor
  /** Whether basis function 0 is the constant 1 (then $\phi_i$ is centred on training input $i - 1$). */
  readonly bias: boolean
  /** The number of training points $N$. */
  readonly n: number
  /** The number of candidate basis functions $B = N$ ($+ 1$ with the bias). */
  readonly size: number
  /** $\Phimat^\top\Phimat$, $B \times B$, row-major. */
  readonly phiTphi: Float64Array
  /** $\Phimat^\top\yvec$, $B$ values. */
  readonly phiTy: Float64Array
  /** $\yvec^\top\yvec$. */
  readonly yTy: number
  /** The design matrix at inputs `xs` (`[s, d]` or `[s]`): `[s, B]`. */
  basis(xs: Tensor): Tensor
  /** The training row a basis function is centred on (null for the bias). */
  centre(i: number): number | null
}

/**
 * Build the RVM problem with kernel basis functions: the design matrix's products, computed once. Throws
 * `ShapeError` when the numbers of inputs and targets differ.
 *
 * @param kernel The kernel $k$; basis function $i$ is $k(\cdot, \xvec_i)$.
 * @param x The training inputs, `[N, d]` or `[N]`.
 * @param y The targets, `[N]` or `[N, 1]`.
 * @param options `bias`: add the constant basis function as basis function 0 (default true).
 * @returns The problem.
 *
 * @example Three inputs: the bias and one basis function per input
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const problem = rvmProblem(rbf({ lengthscale: 1, variance: 1 }), tensor([[0], [1], [2]]), tensor([1, 2, 3]))
 * print('candidates', problem.size, ' centres', [0, 1, 2, 3].map(problem.centre))
 * print('design matrix at 0 and 1', problem.basis(tensor([[0], [1]])))
 */
export function rvmProblem(kernel: Kernel, x: Tensor, y: Tensor, options: { bias?: boolean } = {}): RvmProblem {
  const { bias = true } = options
  const X = asRows(x) as Tensor
  const Y = y.shape.length === 2 ? (reshape(y, [y.shape[0]]) as Tensor) : y
  const n = X.shape[0]
  if (Y.shape[0] !== n) throw new ShapeError('rvm', `rvm: ${n} inputs but ${Y.shape[0]} targets`)
  const size = n + (bias ? 1 : 0)
  const basis = (xs: Tensor) => {
    const S = asRows(xs) as Tensor
    const K = toFlat(gram(kernel, S, X))
    const s = S.shape[0]
    const out = new Float64Array(s * size)
    const o = bias ? 1 : 0
    for (let r = 0; r < s; r++) {
      if (bias) out[r * size] = 1
      for (let j = 0; j < n; j++) out[r * size + o + j] = K[r * n + j]
    }
    return fromData(out, [s, size])
  }
  const phi = basis(X)
  const phiTphi = Float64Array.from(toFlat(matmul(transpose(phi), phi) as Tensor))
  const phiTy = Float64Array.from(toFlat(matmul(transpose(phi), Y) as Tensor))
  const yv = toFlat(Y)
  return {
    kernel,
    x: X,
    y: Y,
    bias,
    n,
    size,
    phiTphi,
    phiTy,
    yTy: yv.reduce((a, v) => a + v * v, 0),
    basis,
    centre: (i) => (bias ? (i === 0 ? null : i - 1) : i),
  }
}

/** The weight posterior on an active set, the log marginal likelihood, and S, Q for every basis function. */
export type RvmPosterior = {
  /** The posterior mean $\muvec$ of the active weights (in `active` order). */
  mean: Float64Array
  /** The posterior covariance $\Sigmamat$ of the active weights, $M \times M$. */
  covariance: Tensor
  /** The log marginal likelihood $\mathcal{L}(\alphavec, \beta)$. */
  logMarginal: number
  /** $\gamma_i = 1 - \alpha_i\Sigma_{ii}$ over the active weights. */
  gamma: Float64Array
  /** $\lVert \yvec - \Phimat_{\Mcal}\muvec \rVert^2$. */
  residual: number
  /**
   * $S_i = \phivec_i^\top\Cmat^{-1}\phivec_i$ for every candidate, $B$ values (with $\Cmat$ the current model's
   * covariance of $\yvec$).
   */
  S: Float64Array
  /** $Q_i = \phivec_i^\top\Cmat^{-1}\yvec$ for every candidate, $B$ values. */
  Q: Float64Array
}

/**
 * The posterior of the active weights at precisions $\alphavec$ (the entries at `active`) and noise precision
 * $\beta$, the log marginal likelihood, and $S_i$, $Q_i$ for every candidate basis function.
 *
 * @param problem The problem from `rvmProblem`.
 * @param active The active basis functions (indices into the $B$ candidates).
 * @param alpha The precisions $\alpha_i$ of all $B$ candidates; only the entries at `active` are read.
 * @param beta The noise precision $\beta = 1/\sigma^2$.
 * @returns The posterior.
 *
 * @example The bias and the basis function at 1, on three points
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const problem = rvmProblem(rbf({ lengthscale: 1, variance: 1 }), tensor([[0], [1], [2]]), tensor([1, 2, 3]))
 * const post = rvmPosterior(problem, [0, 2], [1, Infinity, 1, Infinity], 100)
 * print('weights', post.mean, ' gamma', post.gamma, ' log marginal', post.logMarginal)
 * print('S', post.S, ' Q', post.Q)
 */
export function rvmPosterior(
  problem: RvmProblem,
  active: readonly number[],
  alpha: ArrayLike<number>,
  beta: number,
): RvmPosterior {
  const { size: B, phiTphi, phiTy, n } = problem
  const M = active.length
  const H = new Float64Array(M * M)
  for (let a = 0; a < M; a++) {
    for (let b = 0; b < M; b++) H[a * M + b] = beta * phiTphi[active[a] * B + active[b]]
    H[a * M + a] += alpha[active[a]]
  }
  const { L } = cholesky(fromData(H, [M, M]))
  const block = new Float64Array(M * B) // (ΦᵀΦ)[M, :]
  for (let a = 0; a < M; a++) block.set(phiTphi.subarray(active[a] * B, (active[a] + 1) * B), a * B)
  const V = toFlat(solveTriangular(L, fromData(block, [M, B])) as Tensor) // L⁻¹(ΦᵀΦ)[M, :]
  const u = toFlat(
    solveTriangular(
      L,
      fromData(
        Float64Array.from(active, (i) => phiTy[i]),
        [M],
      ),
    ) as Tensor,
  )
  const mean = Float64Array.from(toFlat(solveTriangular(L, fromData(Float64Array.from(u), [M]), { transpose: true })))
  for (let a = 0; a < M; a++) mean[a] *= beta
  const Linv = solveTriangular(L, fromData(identity(M), [M, M])) as Tensor
  const covariance = matmul(transpose(Linv), Linv) as Tensor
  const cov = toFlat(covariance)
  const S = new Float64Array(B)
  const Q = new Float64Array(B)
  for (let i = 0; i < B; i++) {
    let vv = 0
    let vu = 0
    for (let a = 0; a < M; a++) {
      const v = V[a * B + i]
      vv += v * v
      vu += v * u[a]
    }
    S[i] = beta * phiTphi[i * B + i] - beta * beta * vv
    Q[i] = beta * phiTy[i] - beta * beta * vu
  }
  // ‖y − Φμ‖² = yᵀy − 2μᵀΦᵀy + μᵀ(ΦᵀΦ)μ over the active set.
  let residual = problem.yTy
  let prior = 0
  let logAlpha = 0
  let logDetH = 0
  const Lf = toFlat(L)
  const gamma = new Float64Array(M)
  for (let a = 0; a < M; a++) {
    const i = active[a]
    residual -= 2 * mean[a] * phiTy[i]
    for (let b = 0; b < M; b++) residual += mean[a] * phiTphi[i * B + active[b]] * mean[b]
    prior += alpha[i] * mean[a] * mean[a]
    logAlpha += Math.log(alpha[i])
    logDetH += 2 * Math.log(Lf[a * M + a])
    gamma[a] = 1 - alpha[i] * cov[a * M + a]
  }
  residual = Math.max(residual, 0)
  const logMarginal = 0.5 * (n * Math.log(beta) + logAlpha - logDetH - beta * residual - prior - n * LOG_2PI)
  return { mean, covariance, logMarginal, gamma, residual, S, Q }
}

/**
 * The $m \times m$ identity as a row-major array.
 *
 * @param m The number of rows and columns.
 * @returns $\Imat$, $m^2$ values.
 */
function identity(m: number): Float64Array {
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++) out[i * m + i] = 1
  return out
}

// ── States ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What a step did: `add`, `re-estimate` or `delete` one basis function, re-estimate the `noise` precision (fast
 * algorithm), `update` everything at once (re-estimation), or `none` (step 0, or a converged step).
 */
export type RvmAction = 'add' | 're-estimate' | 'delete' | 'noise' | 'update' | 'none'

/** The state of an RVM fit. */
export type RvmState = Status & {
  /** The active basis functions, ascending. */
  active: number[]
  /** $\alpha_i$ for every candidate basis function, $B$ values; Infinity for the inactive (pruned) ones. */
  alpha: Float64Array
  /** Noise precision $\beta = 1/\sigma^2$. */
  beta: number
  /** Posterior mean of the active weights (in `active` order). */
  mean: Float64Array
  /** The log marginal likelihood $\mathcal{L}(\alphavec, \beta)$. */
  logMarginal: number
  /** What this step did. */
  action: RvmAction
  /** The basis function acted on (null for noise, update and none). */
  index: number | null
  /** Basis functions pruned by this step (re-estimation; for the fast algorithm, the deleted one). */
  pruned: number[]
  /** The change in the log marginal likelihood made by this step. */
  gain: number
}

/** Options shared by the RVM fits. */
export type RvmOptions = {
  /** Starting noise variance $\sigma^2 = 1/\beta$ (default 0.1 times the variance of $\yvec$, or 0.1 if that is 0). */
  noiseVariance?: number
  /** Re-estimate $\beta$ (default true). */
  fitNoise?: boolean
  /**
   * Stop when the best gain is at most this (fast; default 1e-6 nats), or when no basis function is pruned and
   * $\max_i \lvert \Delta \ln \alpha_i \rvert$ is at most this (re-estimation; default 1e-3).
   */
  tolerance?: number
}

/**
 * The starting noise precision: $1/\sigma^2$ for the given noise variance, else $1/(0.1 \operatorname{var} \yvec)$.
 *
 * @param problem The problem, whose targets set the default.
 * @param noiseVariance The starting noise variance, when given.
 * @returns $\beta$.
 */
const startBeta = (problem: RvmProblem, noiseVariance?: number) => {
  const yv = toFlat(problem.y)
  const mean = yv.reduce((a, v) => a + v, 0) / yv.length
  const variance = yv.reduce((a, v) => a + (v - mean) ** 2, 0) / yv.length || 1
  return 1 / (noiseVariance ?? 0.1 * variance)
}

/**
 * A fit state at an active set, $\alphavec$ and $\beta$, with the posterior it was computed from.
 *
 * @param problem The problem from `rvmProblem`.
 * @param active The active basis functions, ascending.
 * @param alpha The precisions of all $B$ candidates (Infinity for the inactive).
 * @param beta The noise precision $\beta$.
 * @param rest The step count, the action and its basis function, the pruned ones, the gain and any status flags.
 * @returns The state, and the posterior (for the next step's use).
 */
function stateAt(
  problem: RvmProblem,
  active: number[],
  alpha: Float64Array,
  beta: number,
  rest: Pick<RvmState, 't' | 'action' | 'index' | 'pruned' | 'gain'> & Status,
): { state: RvmState; post: RvmPosterior } {
  const post = rvmPosterior(problem, active, alpha, beta)
  return { state: { ...rest, active, alpha, beta, mean: post.mean, logMarginal: post.logMarginal }, post }
}

// ── Fast marginal-likelihood maximisation (Tipping and Faul, 2003) ──────────────────────────────────────────────────

/**
 * $\frac{1}{2}[\ln a - \ln(a + s) + q^2/(a + s)]$: the part of $\mathcal{L}$ that depends on one $\alpha_i = a$
 * ($\infty$ gives 0).
 *
 * @param a The precision $a = \alpha_i$, or Infinity for an inactive basis function.
 * @param s The sparsity factor $s_i$.
 * @param q The quality factor $q_i$.
 * @returns $\ell(a)$.
 */
const ell = (a: number, s: number, q: number) =>
  a === Infinity ? 0 : 0.5 * (Math.log(a) - Math.log(a + s) + (q * q) / (a + s))

/** One candidate change and its exact gain in L. */
type Move = { action: 'add' | 're-estimate' | 'delete' | 'noise'; index: number | null; gain: number; value: number }

/**
 * The best single change at a state (Tipping and Faul, 2003, §4): its action, basis function, gain and new value. A
 * deletion is never proposed for the last active basis function.
 *
 * @param problem The problem from `rvmProblem`.
 * @param s The state's active set, $\alphavec$ and $\beta$.
 * @param post The posterior at `s`, from `rvmPosterior`.
 * @param fitNoise Whether a $\beta$ re-estimate is a candidate.
 * @returns The move with the largest finite gain (the new $\alpha_i$, or the new $\beta$, in `value`), or null when
 *   there is none.
 */
function bestMove(
  problem: RvmProblem,
  s: Pick<RvmState, 'active' | 'alpha' | 'beta'>,
  post: RvmPosterior,
  fitNoise: boolean,
): Move | null {
  let best: Move | null = null
  const consider = (m: Move) => {
    if (Number.isFinite(m.gain) && (best === null || m.gain > best.gain)) best = m
  }
  for (let i = 0; i < problem.size; i++) {
    const a = s.alpha[i]
    const active = a !== Infinity
    const S = post.S[i]
    const Q = post.Q[i]
    // The factors with φᵢ left out of C: equal to S and Q for an inactive φᵢ.
    const si = active ? (a * S) / (a - S) : S
    const qi = active ? (a * Q) / (a - S) : Q
    const theta = qi * qi - si
    if (theta > 0) {
      const next = (si * si) / theta
      if (active) consider({ action: 're-estimate', index: i, gain: ell(next, si, qi) - ell(a, si, qi), value: next })
      else consider({ action: 'add', index: i, gain: ell(next, si, qi), value: next })
    } else if (active && s.active.length > 1) {
      consider({ action: 'delete', index: i, gain: -ell(a, si, qi), value: Infinity })
    }
  }
  if (fitNoise) {
    const gammaSum = post.gamma.reduce((x, g) => x + g, 0)
    const beta = (problem.n - gammaSum) / Math.max(post.residual, 1e-300)
    if (Number.isFinite(beta) && beta > 0) {
      const value = rvmPosterior(problem, s.active, s.alpha, beta).logMarginal
      consider({ action: 'noise', index: null, gain: value - post.logMarginal, value: beta })
    }
  }
  return best
}

/**
 * Tipping and Faul's (2003) fast sequential RVM fit as a traceable algorithm. Step 0 holds the one basis function
 * with the largest normalised projection on $\yvec$, at its optimal $\alpha_i$ (or 1000 when it has none); each step
 * then takes the add, re-estimate, delete or noise update with the largest exact gain in $\mathcal{L}$, so the log
 * marginal likelihood never decreases. It converges when no change gains more than `tolerance`.
 *
 * @param problem The problem from `rvmProblem`.
 * @param options The starting noise variance, whether to fit it, and the tolerance.
 * @returns The algorithm; run it with `run` or `trace`.
 *
 * @example A noisy sinc: the log marginal likelihood rises with each change, and few basis functions stay
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const v = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
 * const x = tensor(v.map((u) => [u]))
 * const y = add(tensor(v.map((u) => (u === 0 ? 1 : Math.sin(u) / u))), mul(0.05, normals(stream(0), [11])))
 * const problem = rvmProblem(rbf({ lengthscale: 1, variance: 1 }), x, y)
 * const tr = trace(rvmFastSteps(problem), undefined, 50, { record: { logMarginal: (s) => s.logMarginal } })
 * print('log marginal', tr.series.logMarginal)
 * const label = (s) => (s.index === null ? s.action : s.action + ' ' + s.index)
 * print('first actions', tr.steps.slice(1, 8).map(label).join(', '))
 * print('active', tr.final.active, ' noise variance', 1 / tr.final.beta, ' converged', tr.final.converged)
 */
export function rvmFastSteps(problem: RvmProblem, options: RvmOptions = {}): Algorithm<void, RvmState> {
  const { fitNoise = true, tolerance = 1e-6 } = options
  const B = problem.size
  return {
    name: 'rvm-fast',
    init: () => {
      const beta = startBeta(problem, options.noiseVariance)
      // The basis function with the largest (φᵢᵀy)²/‖φᵢ‖², at α = s²/(q² − s) with C = β⁻¹I (or a large α).
      let best = 0
      let bestScore = -Infinity
      for (let i = 0; i < B; i++) {
        const nn = problem.phiTphi[i * B + i]
        const score = nn > 0 ? problem.phiTy[i] ** 2 / nn : -Infinity
        if (score > bestScore) [best, bestScore] = [i, score]
      }
      const s = beta * problem.phiTphi[best * B + best]
      const q = beta * problem.phiTy[best]
      const alpha = new Float64Array(B).fill(Infinity)
      alpha[best] = q * q > s ? (s * s) / (q * q - s) : 1000
      return stateAt(problem, [best], alpha, beta, {
        t: 0,
        action: 'none',
        index: null,
        pruned: [],
        gain: 0,
      }).state
    },
    step: (state) => {
      const post = rvmPosterior(problem, state.active, state.alpha, state.beta)
      const move = bestMove(problem, state, post, fitNoise)
      if (move === null || move.gain <= tolerance) {
        return { ...state, t: state.t + 1, action: 'none', index: null, pruned: [], gain: 0, converged: true }
      }
      const alpha = Float64Array.from(state.alpha)
      let beta = state.beta
      let active = state.active
      if (move.action === 'noise') beta = move.value
      else {
        alpha[move.index!] = move.value
        if (move.action === 'add') active = [...active, move.index!].sort((a, b) => a - b)
        if (move.action === 'delete') active = active.filter((i) => i !== move.index)
      }
      const before = state.logMarginal
      const { state: next } = stateAt(problem, active, alpha, beta, {
        t: state.t + 1,
        action: move.action,
        index: move.index,
        pruned: move.action === 'delete' ? [move.index!] : [],
        gain: 0,
      })
      return { ...next, gain: next.logMarginal - before }
    },
  }
}

// ── Re-estimation (Tipping, 2001) ────────────────────────────────────────────────────────────────────────────────────

/**
 * Tipping's (2001) re-estimation as a traceable algorithm: every basis function starts active at
 * $\alpha_i =$ `initialAlpha` (default 1), and each step applies $\alpha_i \leftarrow \gamma_i/\mu_i^2$ to all of them
 * and $\beta \leftarrow (N - \sum_i \gamma_i)/\lVert \yvec - \Phimat\muvec \rVert^2$ (kept when not positive and
 * finite), then prunes every $\phi_i$ whose $\alpha_i$ exceeds `pruneAt` (default 1e9), never the last one. Each step
 * costs $O(BM^2 + M^3)$ for $M$ active basis functions, so the early steps are the expensive ones. $\mathcal{L}$
 * usually rises but is not guaranteed to at every step.
 *
 * @param problem The problem from `rvmProblem`.
 * @param options The starting noise variance, whether to fit it, the tolerance, `initialAlpha` and `pruneAt`.
 * @returns The algorithm; run it with `run` or `trace`.
 *
 * @example A noisy sinc: basis functions are pruned as their precisions grow
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const v = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
 * const x = tensor(v.map((u) => [u]))
 * const y = add(tensor(v.map((u) => (u === 0 ? 1 : Math.sin(u) / u))), mul(0.05, normals(stream(0), [11])))
 * const problem = rvmProblem(rbf({ lengthscale: 1, variance: 1 }), x, y)
 * const tr = trace(rvmReestimationSteps(problem), undefined, 200, { record: { active: (s) => s.active.length } })
 * print('active basis functions', tr.series.active)
 * print('active', tr.final.active, ' log marginal', tr.final.logMarginal, ' converged', tr.final.converged)
 */
export function rvmReestimationSteps(
  problem: RvmProblem,
  options: RvmOptions & { initialAlpha?: number; pruneAt?: number } = {},
): Algorithm<void, RvmState> {
  const { fitNoise = true, tolerance = 1e-3, initialAlpha = 1, pruneAt = 1e9 } = options
  return {
    name: 'rvm-reestimation',
    init: () => {
      const alpha = new Float64Array(problem.size).fill(initialAlpha)
      const active = Array.from({ length: problem.size }, (_, i) => i)
      return stateAt(problem, active, alpha, startBeta(problem, options.noiseVariance), {
        t: 0,
        action: 'none',
        index: null,
        pruned: [],
        gain: 0,
      }).state
    },
    step: (state) => {
      const post = rvmPosterior(problem, state.active, state.alpha, state.beta)
      const alpha = Float64Array.from(state.alpha)
      let change = 0
      const pruned: number[] = []
      state.active.forEach((i, a) => {
        const next = Math.max(post.gamma[a], 0) / Math.max(post.mean[a] ** 2, 1e-300)
        if (next > pruneAt || !Number.isFinite(next)) {
          alpha[i] = Infinity
          pruned.push(i)
        } else {
          change = Math.max(change, Math.abs(Math.log(next) - Math.log(alpha[i])))
          alpha[i] = next
        }
      })
      let active = state.active.filter((i) => alpha[i] !== Infinity)
      // Never prune the last basis function: keep the one with the smallest α.
      if (active.length === 0) {
        const keep = state.active.reduce((b, i) => (state.alpha[i] < state.alpha[b] ? i : b), state.active[0])
        alpha[keep] = state.alpha[keep]
        active = [keep]
        pruned.splice(pruned.indexOf(keep), 1)
      }
      const gammaSum = post.gamma.reduce((x, g) => x + g, 0)
      const beta = fitNoise ? (problem.n - gammaSum) / Math.max(post.residual, 1e-300) : state.beta
      const { state: next } = stateAt(problem, active, alpha, beta > 0 && Number.isFinite(beta) ? beta : state.beta, {
        t: state.t + 1,
        action: 'update',
        index: null,
        pruned,
        gain: 0,
        converged: pruned.length === 0 && change <= tolerance,
      })
      return { ...next, gain: next.logMarginal - state.logMarginal }
    },
  }
}

// ── The model at a state ─────────────────────────────────────────────────────────────────────────────────────────────

/** An RVM at a state of its fit: the relevance vectors and the predictive distribution. */
export interface RvmModel {
  /** The brand of a model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'relevance-vector-machine'
  /** The problem the model was fitted on. */
  readonly problem: RvmProblem
  /** The active basis functions (indices into the candidates). */
  readonly active: readonly number[]
  /** The training rows of the active kernel basis functions (the relevance vectors). */
  readonly relevanceVectors: number[]
  /** $\alpha_i$ for every candidate (Infinity for the inactive). */
  readonly alpha: Float64Array
  /** The noise precision $\beta$. */
  readonly beta: number
  /** The posterior mean $\muvec$ of the active weights (in `active` order). */
  readonly weights: Float64Array
  /** The log marginal likelihood $\mathcal{L}(\alphavec, \beta)$. */
  readonly logMarginal: number
  /**
   * Predictive mean $\muvec^\top\phivec(\xvec_*)$ and variance $\phivec(\xvec_*)^\top\Sigmamat\phivec(\xvec_*)$
   * (plus $1/\beta$ with `noise`) at `xs` (`[s, d]` or `[s]`).
   */
  predict(xs: Tensor, options?: { noise?: boolean }): { mean: Tensor; variance: Tensor }
}

/**
 * The model at a state of an RVM fit (or any active set, $\alphavec$ and $\beta$): factors $\Sigmamat^{-1}$ once.
 *
 * @param problem The problem from `rvmProblem`.
 * @param state The active set, the precisions $\alphavec$ and the noise precision $\beta$, as a fit state holds them.
 * @returns The model, with `predict` at new inputs.
 *
 * @example The relevance vectors of a fast fit, and its predictions
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const v = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
 * const x = tensor(v.map((u) => [u]))
 * const y = add(tensor(v.map((u) => (u === 0 ? 1 : Math.sin(u) / u))), mul(0.05, normals(stream(0), [11])))
 * const problem = rvmProblem(rbf({ lengthscale: 1, variance: 1 }), x, y)
 * const model = rvmModel(problem, run(rvmFastSteps(problem), undefined, 50))
 * print('relevance vectors (training rows)', model.relevanceVectors, ' weights', model.weights)
 * print('prediction at 0 and 0.5', model.predict(tensor([[0], [0.5]]), { noise: true }))
 */
export function rvmModel(problem: RvmProblem, state: Pick<RvmState, 'active' | 'alpha' | 'beta'>): RvmModel {
  const { active, alpha, beta } = state
  const post = rvmPosterior(problem, active, alpha, beta)
  const M = active.length
  const cov = toFlat(post.covariance)
  return {
    kind: 'model',
    name: 'relevance-vector-machine',
    problem,
    active,
    relevanceVectors: active.map(problem.centre).filter((r): r is number => r !== null),
    alpha,
    beta,
    weights: post.mean,
    logMarginal: post.logMarginal,
    predict: (xs, { noise = false } = {}) => {
      const P = toFlat(problem.basis(xs))
      const s = P.length / problem.size
      const mean = new Float64Array(s)
      const variance = new Float64Array(s)
      const f = new Float64Array(M)
      for (let r = 0; r < s; r++) {
        for (let a = 0; a < M; a++) f[a] = P[r * problem.size + active[a]]
        let m = 0
        let v = 0
        for (let a = 0; a < M; a++) {
          m += f[a] * post.mean[a]
          let row = 0
          for (let b = 0; b < M; b++) row += cov[a * M + b] * f[b]
          v += f[a] * row
        }
        mean[r] = m
        variance[r] = Math.max(v, 0) + (noise ? 1 / beta : 0)
      }
      return { mean: fromData(mean), variance: fromData(variance) }
    },
  }
}

// ── Estimator ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `relevanceVectorMachine`. */
export type RelevanceVectorMachineParams = RvmOptions & {
  /** The kernel $k$ of the basis functions (required). */
  kernel: Kernel
  /** `fast` (Tipping and Faul, 2003; default) or `reestimation` (Tipping, 2001). */
  method?: 'fast' | 'reestimation'
  /** Add the constant basis function (default true). */
  bias?: boolean
  /** Steps of the fit at most (default 500). */
  maxSteps?: number
}

/** A fitted RVM regression model. */
export interface RelevanceVectorMachineModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Univariate<Tensor>>,
    Expects<Tensor>,
    Trained<RvmState> {
  /** The brand of a model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'relevance-vector-machine'
  /** The RVM at the fit's final state. */
  readonly rvm: RvmModel
}

/**
 * RVM regression as an estimator: the fit runs `rvmFastSteps` (or `rvmReestimationSteps`) to convergence or
 * `maxSteps` and keeps its trace in `training`. Capabilities: `forward` and `decide` (the predictive mean),
 * `predictive` (normals $\Gauss(\mu, \phivec^\top\Sigmamat\phivec + 1/\beta)$), `expect`.
 *
 * @param params The kernel, the method, the bias, the step limit and the options of the fit.
 * @returns The estimator: `fit({ x, y })` on inputs `[N, d]` (or `[N]`) and targets `[N]` returns a
 *   `RelevanceVectorMachineModel`.
 *
 * @example A noisy sinc from six relevance vectors
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const v = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
 * const x = tensor(v.map((u) => [u]))
 * const y = add(tensor(v.map((u) => (u === 0 ? 1 : Math.sin(u) / u))), mul(0.05, normals(stream(0), [11])))
 * const model = relevanceVectorMachine({ kernel: rbf({ lengthscale: 1, variance: 1 }) }).fit({ x, y })
 * print('relevance vectors', model.rvm.relevanceVectors, ' noise variance', 1 / model.rvm.beta)
 * print('mean at 0 and 0.5', model.forward(tensor([[0], [0.5]])), ' sinc there', 1, Math.sin(0.5) / 0.5)
 * print('predictive variance', model.predictive(tensor([[0], [0.5]])).variance())
 */
export function relevanceVectorMachine(
  params: RelevanceVectorMachineParams,
): Estimator<Supervised<Tensor, Tensor>, RelevanceVectorMachineModel> {
  const { kernel, method = 'fast', bias = true, maxSteps = 500 } = params
  return {
    name: 'relevance-vector-machine',
    params,
    fit({ x, y }) {
      const problem = rvmProblem(kernel, x, y, { bias })
      const alg = method === 'fast' ? rvmFastSteps(problem, params) : rvmReestimationSteps(problem, params)
      const training: Trace<RvmState> = trace(alg, undefined, maxSteps, {
        keep: 'none',
        record: { logMarginal: (s: RvmState) => s.logMarginal },
      })
      const rvm = rvmModel(problem, training.final)
      const forward = (input: Tensor) => rvm.predict(input).mean
      return withExpectation({
        kind: 'model' as const,
        name: 'relevance-vector-machine' as const,
        rvm,
        training,
        forward,
        decide: forward,
        predictive: (input: Tensor) => {
          const p = rvm.predict(input, { noise: true })
          return Normal(p.mean, fromData(Float64Array.from(toFlat(p.variance), Math.sqrt))) as Univariate<Tensor>
        },
      })
    },
  }
}

defineModel(
  {
    key: 'relevanceVectorMachine',
    module: 'learning/gaussian-processes',
    name: 'Relevance vector machine',
    summary: 'Sparse Bayesian kernel regression with one ARD precision per basis function; the kernel is required.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect'],
    hyper: space({
      method: oneOf(['fast', 'reestimation']),
      bias: bool({ default: true }),
      noiseVariance: real(1e-6, 10, { default: 0.01, scale: 'log' }),
      fitNoise: bool({ default: true }),
      maxSteps: int(1, 5000, { default: 500 }),
    }),
    notes: ['relevance-vector-machine'],
    cite: ['tipping2001', 'tipping2003'],
  },
  relevanceVectorMachine,
)

const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/gaussian-processes')
const rvmRoles = { objective: 'logMarginal', flags: ['converged'] } as const
algorithm(
  {
    key: 'rvmFastSteps',
    name: 'RVM by fast marginal-likelihood maximisation',
    summary:
      'Tipping and Faul: add, re-estimate or delete one basis function (or update β) per step by the largest gain.',
    problem: 'objective',
    state: rvmRoles,
    notes: ['relevance-vector-machine'],
    cite: ['tipping2003'],
  },
  rvmFastSteps,
)
algorithm(
  {
    key: 'rvmReestimationSteps',
    name: 'RVM by re-estimation',
    summary: 'Tipping (2001): αᵢ ← γᵢ/μᵢ² and β ← (N − Σγ)/‖y − Φμ‖² for all basis functions per step, with pruning.',
    problem: 'objective',
    state: rvmRoles,
    notes: ['relevance-vector-machine'],
    cite: ['tipping2001', 'mackay1992'],
  },
  rvmReestimationSteps,
)
