/**
 * The Crammer–Singer multiclass SVM (Crammer and Singer, 2001, JMLR 2), part of `aifn-methods/learning/kernel-methods`,
 * solved by LIBLINEAR's sequential dual method (Keerthi et al., 2008, KDD).
 *
 * One weight vector $\wvec_k$ per class, trained jointly: the primal is
 * $\min \frac{1}{2} \sum_k \lVert \wvec_k \rVert^2 + C \sum_i \xi_i$ subject to
 * $\wvec_{y_i}^\top\xvec_i - \wvec_k^\top\xvec_i \ge \indicator[k \ne y_i] - \xi_i$ for every $k$, and the dual has
 * $K$ variables $\alpha_{ik}$ per example, with $\wvec_k = \sum_i \alpha_{ik} \xvec_i$. Labels are the integers
 * $0, \dots, K - 1$. With an intercept, $\tilde\xvec_i$ is $\xvec_i$ with a constant 1 appended, whose weights
 * $\tilde\wvec_k$ end in the biases, regularised with the rest.
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

/** The Crammer–Singer problem: inputs, labels, $C$, and whether to append a constant feature. */
export interface CrammerSingerProblem {
  /** Inputs $n \times d$. */
  x: Tensor
  /** Integer class labels $0, \dots, K - 1$, $n$ of them; $K$ is one more than the largest (at least 2). */
  y: Tensor
  /** The penalty $C$ on the slacks (default 1). */
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
  /** Weights $K \times d$, a row per class. */
  weights: Tensor
  /** Biases, $K$ values (zeros without an intercept). */
  bias: Tensor
  /**
   * Dual variables $\alpha_{ik}$, $n \times K$, with $\sum_k \alpha_{ik} = 0$ and
   * $\alpha_{ik} \le C \indicator[k = y_i]$.
   */
  alpha: Tensor
  /**
   * The primal objective $\frac{1}{2} \sum_k \lVert \tilde\wvec_k \rVert^2 + C \sum_i \max_k \ell_{ik}$, with
   * $\ell_{ik} = \indicator[k \ne y_i] + \tilde\wvec_k^\top\tilde\xvec_i - \tilde\wvec_{y_i}^\top\tilde\xvec_i$.
   */
  primalObjective: number
  /**
   * The largest violation $\max_k G_{ik} - \min_{k : \alpha_{ik} < C \indicator[k = y_i]} G_{ik}$ over the epoch's
   * examples, each measured before its update ($\infty$ at the start), with
   * $G_{ik} = \tilde\wvec_k^\top\tilde\xvec_i + \indicator[k \ne y_i]$.
   */
  violation: number
  /** Whether `violation` is at most the tolerance. */
  converged: boolean
}

/**
 * LIBLINEAR's sub-problem for one example,
 * $\min_{\betavec} \frac{1}{2} A \lVert \betavec \rVert^2 + \bvec^\top\betavec$ subject to $\sum_k \beta_k = 0$ and
 * $\beta_k \le \hat C_k$, with $\hat C_k = C_y$ for the example's class and 0 for the others, solved by sorting
 * (Keerthi et al., 2008, §3; Crammer and Singer, 2001, Algorithm 2).
 *
 * @param A The curvature $A = \lVert \tilde\xvec_i \rVert^2$, positive.
 * @param B The linear term $\bvec$, $K$ values (read only).
 * @param Cy The cap $C_y$ of the example's own class.
 * @param y The example's class.
 * @returns The minimiser $\betavec$, $K$ values: the example's new dual variables.
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
 * The Crammer–Singer multiclass SVM, $\min \frac{1}{2} \sum_k \lVert \wvec_k \rVert^2 + C \sum_i \xi_i$ subject to
 * $\wvec_{y_i}^\top\xvec_i - \wvec_k^\top\xvec_i \ge \indicator[k \ne y_i] - \xi_i$, by sequential dual coordinate
 * ascent: each step is one epoch over the examples in row order, solving each example's $K$-variable dual
 * sub-problem exactly (an example within $10^{-12}$ of optimal, or a row of zeros, is skipped). No start:
 * $\alphavec = \zeros$. Throws `ShapeError` or `DomainError` for labels that do not match the rows or are not
 * integers $0, \dots, K - 1$.
 *
 * @param problem The inputs, the labels, $C$, `intercept` and `tolerance`.
 * @returns The algorithm, one epoch per step.
 *
 * @example Three classes, each a pair of points along an axis
 * const x = tensor([[0, 0], [1, 0], [5, 0], [6, 0], [0, 5], [0, 6]])
 * const y = tensor([0, 0, 1, 1, 2, 2])
 * const s = run(crammerSingerSteps({ x, y }), undefined, 100)
 * print('weights =', s.weights)
 * print('biases =', s.bias)
 * print('primal =', s.primalObjective, 'converged:', s.converged, 'after', s.t, 'epochs')
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
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'crammer-singer'
  /** Weights $K \times d$, a row per class. */
  readonly weights: Tensor
  /** Biases, $K$ values (zeros without an intercept). */
  readonly bias: Tensor
  /** The number of classes $K$. */
  readonly classes: number
  /** Whether the dual method reached the tolerance within `maxSteps` epochs. */
  readonly converged: boolean
}

/**
 * The Crammer–Singer multiclass linear SVM (see `crammerSingerSteps`), as scikit-learn's
 * `LinearSVC(multi_class='crammer_singer')`. `score` and `forward` give $\wvec_k^\top\xvec + b_k$, $m \times K$ for
 * $m$ query rows, and `decide` the class of the largest score.
 *
 * @param params The hyperparameters.
 * @param params.C The penalty $C$ on the slacks (default 1).
 * @param params.intercept Learn a bias per class as the weight of a constant feature (default true).
 * @param params.tolerance Stop when an epoch's largest KKT violation is at most this (default 1e-6).
 * @param params.maxSteps The most epochs (default 1000).
 * @returns The estimator; its `fit` takes inputs `x` ($n \times d$) and integer labels `y`.
 *
 * @example Three classes: the decisions at the data and the scores of a new point
 * const x = tensor([[0, 0], [1, 0], [5, 0], [6, 0], [0, 5], [0, 6]])
 * const y = tensor([0, 0, 1, 1, 2, 2])
 * const model = crammerSinger().fit({ x, y })
 * print('decisions at the data:', model.decide(x))
 * print('scores at (0.5, 0):', model.score(tensor([[0.5, 0]])))
 * print('converged:', model.converged, 'after', model.training.final.t, 'epochs')
 */
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
