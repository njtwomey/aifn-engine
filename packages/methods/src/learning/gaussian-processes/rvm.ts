/**
 * The relevance vector machine for regression (Tipping, 2001, "Sparse Bayesian learning and the relevance vector
 * machine", JMLR 1): y = Σᵢ wᵢ φᵢ(x) + ε with a bias φ₀ = 1 and one kernel basis function φᵢ(x) = k(x, xᵢ) per training
 * input, wᵢ ~ N(0, αᵢ⁻¹) and ε ~ N(0, β⁻¹). For an active set M of basis functions (those with finite αᵢ),
 *
 *   Σ = (A + βΦ_MᵀΦ_M)⁻¹,   μ = βΣΦ_Mᵀy,
 *   L(α, β) = ½[N ln β + Σ_M ln αᵢ − ln|Σ⁻¹| − β‖y − Φ_Mμ‖² − μᵀAμ − N ln 2π].
 *
 * Two fits, both traceable algorithms whose states carry the active set, α, β, the log marginal likelihood and what
 * the step did:
 *
 * - `rvmFastSteps`: Tipping and Faul's (2003) fast marginal-likelihood maximisation. From one basis function, each step
 *   takes the single change with the largest exact gain ΔL: add an inactive φᵢ, re-estimate an active αᵢ, delete an
 *   active φᵢ, or re-estimate β. The gains come from the sparsity and quality factors sᵢ = φᵢᵀC₋ᵢ⁻¹φᵢ and
 *   qᵢ = φᵢᵀC₋ᵢ⁻¹y, with L(αᵢ) = ½[ln αᵢ − ln(αᵢ + sᵢ) + qᵢ²/(αᵢ + sᵢ)] + const maximised at αᵢ = sᵢ²/(qᵢ² − sᵢ)
 *   when qᵢ² > sᵢ and at ∞ otherwise. A β update is taken only when it raises L, so L never decreases.
 * - `rvmReestimationSteps`: the original re-estimation, all basis functions at once: γᵢ = 1 − αᵢΣᵢᵢ, αᵢ ← γᵢ/μᵢ²,
 *   β ← (N − Σγᵢ)/‖y − Φμ‖² (MacKay, 1992), pruning φᵢ when αᵢ passes a threshold.
 *
 * Linear algebra is aifn's: the Cholesky factor of Σ⁻¹ and triangular solves, with S and Q for every basis function
 * from one solve against the M × B block of ΦᵀΦ, so a step costs O(BM² + M³) after the O(NB²) set-up.
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
  readonly kernel: Kernel
  /** Training inputs [N, d] and targets [N]. */
  readonly x: Tensor
  readonly y: Tensor
  /** Whether basis function 0 is the constant 1 (then φᵢ is centred on training input i − 1). */
  readonly bias: boolean
  /** N and the number of candidate basis functions B = N (+ 1 with the bias). */
  readonly n: number
  readonly size: number
  /** ΦᵀΦ [B·B] row-major, Φᵀy [B], yᵀy. */
  readonly phiTphi: Float64Array
  readonly phiTy: Float64Array
  readonly yTy: number
  /** The design matrix at inputs xs [s, d] (or [s]): [s, B]. */
  basis(xs: Tensor): Tensor
  /** The training row a basis function is centred on (null for the bias). */
  centre(i: number): number | null
}

/** Build the RVM problem for inputs x [N, d] (or [N]) and targets y [N] with kernel basis functions. */
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
  /** μ and Σ over the active weights (in `active` order). */
  mean: Float64Array
  covariance: Tensor
  logMarginal: number
  /** γᵢ = 1 − αᵢΣᵢᵢ over the active weights. */
  gamma: Float64Array
  /** ‖y − Φ_Mμ‖². */
  residual: number
  /** Sᵢ = φᵢᵀC⁻¹φᵢ and Qᵢ = φᵢᵀC⁻¹y for every candidate [B] (with C the current model's covariance). */
  S: Float64Array
  Q: Float64Array
}

/** The posterior of the active weights at precisions α (the entries at `active`) and noise precision β. */
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
  /** αᵢ for every candidate basis function [B]; Infinity for the inactive (pruned) ones. */
  alpha: Float64Array
  /** Noise precision β = 1/σ². */
  beta: number
  /** Posterior mean of the active weights (in `active` order). */
  mean: Float64Array
  logMarginal: number
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
  /** Starting noise variance σ² = 1/β (default 0.1 × the variance of y). */
  noiseVariance?: number
  /** Re-estimate β (default true). */
  fitNoise?: boolean
  /**
   * Stop when the best gain is at most this (fast; default 1e-6 nats), or when no basis function is pruned and
   * max |Δ ln αᵢ| is at most this (re-estimation; default 1e-3).
   */
  tolerance?: number
}

const startBeta = (problem: RvmProblem, noiseVariance?: number) => {
  const yv = toFlat(problem.y)
  const mean = yv.reduce((a, v) => a + v, 0) / yv.length
  const variance = yv.reduce((a, v) => a + (v - mean) ** 2, 0) / yv.length || 1
  return 1 / (noiseVariance ?? 0.1 * variance)
}

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

/** ½[ln a − ln(a + s) + q²/(a + s)]: the part of L that depends on one αᵢ = a (∞ gives 0). */
const ell = (a: number, s: number, q: number) =>
  a === Infinity ? 0 : 0.5 * (Math.log(a) - Math.log(a + s) + (q * q) / (a + s))

/** One candidate change and its exact gain in L. */
type Move = { action: 'add' | 're-estimate' | 'delete' | 'noise'; index: number | null; gain: number; value: number }

/** The best single change at a state (Tipping and Faul, 2003, §4): its action, basis function, gain and new value. */
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
 * with the largest normalised projection on y, at its optimal αᵢ; each step then takes the add, re-estimate, delete or
 * noise update with the largest exact gain in L, so the log marginal likelihood never decreases. It converges when no
 * change gains more than `tolerance`.
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
 * Tipping's (2001) re-estimation as a traceable algorithm: every basis function starts active at αᵢ = `initialAlpha`
 * (default 1), and each step applies αᵢ ← γᵢ/μᵢ² to all of them and β ← (N − Σγᵢ)/‖y − Φμ‖², then prunes every φᵢ whose
 * αᵢ exceeds `pruneAt` (default 1e9). Each step costs O(M³) for M active basis functions, so the early steps are the
 * expensive ones. L usually rises but is not guaranteed to at every step.
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
  readonly kind: 'model'
  readonly name: 'relevance-vector-machine'
  readonly problem: RvmProblem
  readonly active: readonly number[]
  /** The training rows of the active kernel basis functions (the relevance vectors). */
  readonly relevanceVectors: number[]
  readonly alpha: Float64Array
  readonly beta: number
  readonly weights: Float64Array
  readonly logMarginal: number
  /** Predictive mean μᵀφ(x*) and variance φ(x*)ᵀΣφ(x*) (+ 1/β with `noise`) at xs [s, d] (or [s]). */
  predict(xs: Tensor, options?: { noise?: boolean }): { mean: Tensor; variance: Tensor }
}

/** The model at a state of an RVM fit (or any active set, α and β): factors Σ⁻¹ once. */
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
  kernel: Kernel
  /** `fast` (Tipping and Faul, 2003; default) or `reestimation` (Tipping, 2001). */
  method?: 'fast' | 'reestimation'
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
  readonly kind: 'model'
  readonly name: 'relevance-vector-machine'
  readonly rvm: RvmModel
}

/**
 * RVM regression as an estimator: the fit runs `rvmFastSteps` (or `rvmReestimationSteps`) to convergence and keeps its
 * trace in `training`. Capabilities: `forward` and `decide` (the predictive mean), `predictive` (normals N(mean,
 * φᵀΣφ + 1/β)), `expect`.
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
