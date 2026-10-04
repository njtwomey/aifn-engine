/**
 * The Crammer–Singer multiclass SVM (Crammer and Singer, 2001, JMLR 2), part of `aifn-methods/learning/kernel-methods`,
 * solved by LIBLINEAR's sequential dual method (Keerthi et al., 2008, KDD).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import {
  type Decides,
  type Estimator,
  type FitOptions,
  type Fitted,
  type Scores,
  type Supervised,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { argmax, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { classLabels, inputs, matrix, values } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, real, space } from 'aifn-compute/foundation/space'

// ── Crammer–Singer ───────────────────────────────────────────────────────────────────────────────────────────────

/** The Crammer–Singer problem: inputs [n, d], labels 0 … K−1, C, and whether to append a constant feature. */
export interface CrammerSingerProblem {
  x: Tensor
  y: Tensor
  C?: number
  /** Append a constant feature whose weights act as (regularised) biases (default true). */
  intercept?: boolean
  /** Stop when the largest KKT violation of an epoch is at most `tolerance` (default 1e-6). */
  tolerance?: number
}

/** One epoch of the Crammer–Singer dual method. */
export interface CrammerSingerState extends Status {
  /** Epochs done. */
  t: number
  /** Weights [K, d] and biases [K]. */
  weights: Tensor
  bias: Tensor
  /** Dual variables α [n, K]: Σₖ α_ik = 0, α_ik ≤ C·1[k = yᵢ]. */
  alpha: Tensor
  /** ½ Σₖ ‖w̃ₖ‖² + C Σᵢ max_k (1[k ≠ yᵢ] + w̃ₖ·x̃ᵢ − w̃_yᵢ·x̃ᵢ). */
  primalObjective: number
  /** The largest violation max_k G_ik − min_{k: α_ik < C·1[k = yᵢ]} G_ik in the epoch. */
  violation: number
  converged: boolean
}

/**
 * LIBLINEAR's sub-problem: min over β of ½ A ‖β‖² + Bᵀβ s.t. Σβ = 0, β_k ≤ Ĉ_k, solved by sorting (Keerthi et al.,
 * 2008, §3; Crammer and Singer, 2001, Algorithm 2).
 */
function solveSubproblem(A: number, B: Float64Array, Cy: number, y: number): Float64Array {
  const K = B.length
  const D = Float64Array.from(B)
  if (Cy !== 0) D[y] += A * Cy
  const sorted = Array.from(D).sort((a, b) => b - a)
  let beta = sorted[0] - A * Cy
  let r = 1
  while (r < K && beta < r * sorted[r]) {
    beta += sorted[r]
    r++
  }
  beta /= r
  return Float64Array.from(B, (_, k) => {
    const cap = k === y ? Cy : 0
    return Math.min(cap, (beta - B[k]) / A)
  })
}

/**
 * The Crammer–Singer multiclass SVM, min ½ Σₖ ‖wₖ‖² + C Σᵢ ξᵢ s.t. w_yᵢ·xᵢ − wₖ·xᵢ ≥ 1[k ≠ yᵢ] − ξᵢ, by sequential
 * dual coordinate ascent: each step is one epoch over the examples in row order, solving each example's K-variable
 * dual sub-problem exactly.
 */
export function crammerSingerSteps(problem: CrammerSingerProblem): Algorithm<void, CrammerSingerState> {
  const { n, d, v } = matrix(problem.x, 'crammerSingerSteps')
  const { y, k: K } = classLabels(problem.y, n, 'crammerSingerSteps')
  const C = problem.C ?? 1
  const intercept = problem.intercept ?? true
  const tol = problem.tolerance ?? 1e-6
  const D = intercept ? d + 1 : d
  const X = new Float64Array(n * D)
  const sq = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) X[i * D + j] = v[i * d + j]
    if (intercept) X[i * D + d] = 1
    for (let j = 0; j < D; j++) sq[i] += X[i * D + j] ** 2
  }
  const state = (W: Float64Array, alpha: Float64Array, epoch: number, violation: number): CrammerSingerState => {
    let obj = 0
    for (const w of W) obj += 0.5 * w * w
    for (let i = 0; i < n; i++) {
      let fy = 0
      for (let j = 0; j < D; j++) fy += W[y[i] * D + j] * X[i * D + j]
      let worst = 0
      for (let k = 0; k < K; k++) {
        let f = 0
        for (let j = 0; j < D; j++) f += W[k * D + j] * X[i * D + j]
        worst = Math.max(worst, (k === y[i] ? 0 : 1) + f - fy)
      }
      obj += C * worst
    }
    const w = new Float64Array(K * d)
    const b = new Float64Array(K)
    for (let k = 0; k < K; k++) {
      for (let j = 0; j < d; j++) w[k * d + j] = W[k * D + j]
      if (intercept) b[k] = W[k * D + d]
    }
    return {
      weights: fromData(w, [K, d]),
      bias: fromData(b, [K]),
      alpha: fromData(alpha, [n, K]),
      primalObjective: obj,
      violation,
      t: epoch,
      converged: violation <= tol,
      diverged: !Number.isFinite(obj),
    }
  }
  const weightsFrom = (alpha: Float64Array) => {
    const W = new Float64Array(K * D)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) {
        const a = alpha[i * K + k]
        if (a !== 0) for (let j = 0; j < D; j++) W[k * D + j] += a * X[i * D + j]
      }
    return W
  }
  return {
    name: 'crammer-singer',
    init: () => state(new Float64Array(K * D), new Float64Array(n * K), 0, Infinity),
    step: (s) => {
      const alpha = Float64Array.from(values(s.alpha))
      const W = weightsFrom(alpha)
      let violation = 0
      const G = new Float64Array(K)
      for (let i = 0; i < n; i++) {
        if (sq[i] <= 0) continue
        for (let k = 0; k < K; k++) {
          let f = 0
          for (let j = 0; j < D; j++) f += W[k * D + j] * X[i * D + j]
          G[k] = f + (k === y[i] ? 0 : 1)
        }
        let maxG = -Infinity
        let minG = Infinity
        for (let k = 0; k < K; k++) {
          maxG = Math.max(maxG, G[k])
          if (alpha[i * K + k] < (k === y[i] ? C : 0)) minG = Math.min(minG, G[k])
        }
        violation = Math.max(violation, maxG - minG)
        if (maxG - minG <= 1e-12) continue
        // B = G − A α_i, the linear term of the sub-problem in the new α_i.
        const B = Float64Array.from(G, (g, k) => g - sq[i] * alpha[i * K + k])
        const next = solveSubproblem(sq[i], B, C, y[i])
        for (let k = 0; k < K; k++) {
          const delta = next[k] - alpha[i * K + k]
          if (delta === 0) continue
          alpha[i * K + k] = next[k]
          for (let j = 0; j < D; j++) W[k * D + j] += delta * X[i * D + j]
        }
      }
      return state(W, alpha, s.t + 1, violation)
    },
  }
}

/** A fitted Crammer–Singer SVM. */
export interface CrammerSingerModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<CrammerSingerState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'crammer-singer'
  readonly weights: Tensor
  readonly bias: Tensor
  readonly classes: number
  readonly converged: boolean
}

/** The Crammer–Singer multiclass linear SVM (see `crammerSingerSteps`): `score` gives wₖ·x + bₖ [m, K]. */
export function crammerSinger(
  params: { C?: number; intercept?: boolean; tolerance?: number; maxSteps?: number } = {},
): Estimator<Supervised<Tensor, Tensor>, CrammerSingerModel> {
  const { C = 1, intercept = true, tolerance = 1e-6, maxSteps = 1000 } = params
  return {
    name: 'crammer-singer',
    params: { C, intercept, tolerance, maxSteps },
    fit({ x, y }, options: FitOptions = {}) {
      const { d } = matrix(x, 'crammerSinger')
      const training = trace(crammerSingerSteps({ x, y, C, intercept, tolerance }), undefined, maxSteps, {
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        stopOnNonFinite: false,
        record: { primalObjective: (s) => s.primalObjective },
      })
      const final = training.final
      const [K] = final.weights.shape
      const w = values(final.weights)
      const b = values(final.bias)
      const score = (q: Tensor) => {
        const { n: m, v } = inputs(q, d, 'crammerSinger')
        const out = new Float64Array(m * K)
        for (let i = 0; i < m; i++)
          for (let k = 0; k < K; k++) {
            let s = b[k]
            for (let j = 0; j < d; j++) s += w[k * d + j] * v[i * d + j]
            out[i * K + k] = s
          }
        return fromData(out, [m, K])
      }
      return {
        kind: 'model',
        name: 'crammer-singer',
        weights: final.weights,
        bias: final.bias,
        classes: K,
        converged: final.converged,
        training,
        forward: score,
        score,
        decide: (q: Tensor) => argmax(score(q), -1),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'crammerSinger',
    module: 'learning/kernel-methods',
    name: 'Crammer–Singer multiclass SVM',
    summary: 'A single multiclass margin machine solved in the dual.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({
      C: real(1e-3, 1e3, { default: 1, scale: 'log' }),
      intercept: bool({ default: true }),
      tolerance: real(1e-10, 1e-2, { default: 1e-6, scale: 'log' }),
      maxSteps: int(1, 10000, { default: 1000 }),
    }),
    notes: ['multiclass-support-vector-machines'],
    cite: ['crammer2001'],
  },
  crammerSinger,
)
