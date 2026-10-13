/**
 * Weights for averaging the predictive distributions of several models from their leave-one-out log predictive
 * densities: stacking, pseudo-BMA and pseudo-BMA+ (Yao, Vehtari, Simpson and Gelman, 2018, "Using stacking to average
 * Bayesian predictive distributions", Bayesian Analysis 13), as arviz-stats' `compare` (`stacking`, `pseudo-bma`,
 * `bb-pseudo-bma`).
 *
 * Each takes the $n \times J$ matrix $\ell_{ij} = \log p(y_i \mid \yvec_{-i}, M_j)$, a row per observation and a
 * column per model, such as the pointwise elpd of `psisLoo` for each model side by side. Stacking maximises the
 * leave-one-out log score of the mixture over the simplex,
 *
 * $$ \max_{\wvec \in \mathcal{S}_J} \sum_{i=1}^n \log \sum_{j=1}^J w_j \exp \ell_{ij}, $$
 *
 * a concave problem, so it combines models that predict different observations well; pseudo-BMA weighs each model by
 * its total elpd, $w_j \propto \exp \widehat{\mathrm{elpd}}_j$, which puts nearly all the weight on the best one, and
 * pseudo-BMA+ averages those weights over Bayesian-bootstrap replicates of the elpd, which tempers them by its
 * uncertainty.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, exponential, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'

/** Stacking weights, from `stackingWeights`. */
export interface StackingWeights {
  /** The weight of each model, $J$ values on the simplex. */
  weights: Tensor
  /** The maximised log score $\sum_i \log \sum_j w_j \exp \ell_{ij}$ (nats). */
  objective: number
  /** The number of Newton steps taken. */
  iterations: number
  /** Whether the optimality conditions held to the tolerance. */
  converged: boolean
}

/** Pseudo-BMA weights, from `pseudoBmaWeights` and `pseudoBmaPlusWeights`. */
export interface PseudoBmaWeights {
  /** The weight of each model, $J$ values summing to 1. */
  weights: Tensor
  /** Each model's elpd $\sum_i \ell_{ij}$. */
  elpd: Tensor
}

/** Pseudo-BMA+ weights, with the spread of the bootstrapped elpd. */
export interface PseudoBmaPlusWeights extends PseudoBmaWeights {
  /** The standard deviation of each model's bootstrapped elpd over the replicates (divided by $B$). */
  se: Tensor
  /** The number of Bayesian-bootstrap replicates $B$. */
  B: number
}

/**
 * The log densities as a working matrix; throws `DomainError` without rows or models.
 *
 * @param lpd The $n \times J$ log densities.
 * @param where The caller's name, for error messages.
 * @returns The row-major values with $n$ and $J$.
 */
function densities(lpd: MatrixLike, where: string): { data: Float64Array; n: number; J: number } {
  const { data, m, n } = dense.toMatrixF64(lpd, where)
  if (m < 1 || n < 1) throw new DomainError(where, `${where}: needs at least one observation and one model`)
  return { data: Float64Array.from(data), n: m, J: n }
}

/**
 * Stacking weights (Yao et al., 2018, §3): the weights $\wvec$ on the simplex that maximise the leave-one-out log
 * score $\sum_i \log \sum_j w_j \exp \ell_{ij}$ of the mixture of the models' predictive distributions. Solved by an
 * active-set Newton method on the simplex (each step a Newton step in the free weights under $\sum_j w_j = 1$, with a
 * backtracking line search, weights that reach 0 leaving the free set, and a zero weight returning when its gradient
 * exceeds the common value $n$ that the free weights share at the optimum), to the optimality conditions within
 * `tol`. With duplicate models the maximiser is not unique. Throws `DomainError` for an empty matrix.
 *
 * @param looLogDensities The $n \times J$ leave-one-out log predictive densities $\ell_{ij}$: a row per observation, a
 *   column per model.
 * @param options The tolerance and the iteration limit.
 * @param options.tol The relative tolerance on the optimality conditions.
 * @param options.maxIterations The most Newton steps.
 * @returns The weights, the maximised objective, the number of steps and whether they converged.
 *
 * @example Two models that are each good on half the data share the weight
 * // Model A predicts the first three observations well, model B the last three.
 * const lpd = [[-0.5, -3], [-0.6, -2.5], [-0.4, -3.2], [-2.8, -0.5], [-3.1, -0.7], [-2.6, -0.4]]
 * const s = stackingWeights(lpd)
 * print('stacking weights:', s.weights, ' objective:', s.objective)
 * print('pseudo-BMA weights:', pseudoBmaWeights(lpd).weights)
 */
export function stackingWeights(
  looLogDensities: MatrixLike,
  { tol = 1e-12, maxIterations = 500 }: { tol?: number; maxIterations?: number } = {},
): StackingWeights {
  const { data, n, J } = densities(looLogDensities, 'stackingWeights')
  // P_ij = exp(l_ij - max_j l_ij), so that the largest density of each row is 1.
  const shift = new Float64Array(n)
  const P = new Float64Array(n * J)
  for (let i = 0; i < n; i++) {
    let m = -Infinity
    for (let j = 0; j < J; j++) m = Math.max(m, data[i * J + j])
    shift[i] = m
    for (let j = 0; j < J; j++) P[i * J + j] = Math.exp(data[i * J + j] - m)
  }
  const objectiveOf = (w: Float64Array) => {
    let f = 0
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < J; j++) s += w[j] * P[i * J + j]
      f += Math.log(s) + shift[i]
    }
    return f
  }
  let w = new Float64Array(J).fill(1 / J)
  const free = new Uint8Array(J).fill(1)
  let iterations = 0
  let converged = false
  let stalled = false
  while (iterations < maxIterations) {
    // Gradient g_j = sum_i P_ij / s_i and the negated Hessian A_jk = sum_i P_ij P_ik / s_i^2.
    const g = new Float64Array(J)
    const A = new Float64Array(J * J)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < J; j++) s += w[j] * P[i * J + j]
      for (let j = 0; j < J; j++) {
        g[j] += P[i * J + j] / s
        for (let k = 0; k < J; k++) A[j * J + k] += (P[i * J + j] * P[i * J + k]) / (s * s)
      }
    }
    const F = Array.from({ length: J }, (_, j) => j).filter((j) => free[j])
    // The Newton step in the free weights under sum d = 0: [A_FF 1; 1' 0] [d; nu] = [g_F; 0].
    const m = F.length
    const K = new Float64Array((m + 1) * (m + 1))
    const rhs = new Float64Array(m + 1)
    let trace = 0
    for (const j of F) trace += A[j * J + j]
    F.forEach((j, a) => {
      F.forEach((k, b) => (K[a * (m + 1) + b] = A[j * J + k] + (a === b ? 1e-14 * trace : 0)))
      K[a * (m + 1) + m] = K[m * (m + 1) + a] = 1
      rhs[a] = g[j]
    })
    const step = dense.data(solve(fromData(K, [m + 1, m + 1]), fromData(rhs, [m + 1])) as Tensor)
    const d = new Float64Array(J)
    F.forEach((j, a) => (d[j] = step[a]))
    const size = Math.max(...F.map((j) => Math.abs(d[j])))
    if (size <= tol || stalled) {
      stalled = false
      // Stationary on this face: release the zero weight whose gradient most exceeds n, or stop.
      let worst = -1
      for (let j = 0; j < J; j++) if (!free[j] && g[j] > n * (1 + tol) && (worst < 0 || g[j] > g[worst])) worst = j
      if (worst < 0) {
        converged = true
        break
      }
      free[worst] = 1
      iterations++
      continue
    }
    // The largest feasible step, then backtracking on the objective.
    let tMax = Infinity
    let blocking = -1
    for (const j of F)
      if (d[j] < 0 && -w[j] / d[j] < tMax) {
        tMax = -w[j] / d[j]
        blocking = j
      }
    let t = Math.min(1, tMax)
    const f0 = objectiveOf(w)
    let slope = 0
    for (const j of F) slope += g[j] * d[j]
    let next = w
    for (let halvings = 0; halvings < 60; halvings++) {
      next = Float64Array.from(w, (v, j) => Math.max(0, v + t * d[j]))
      if (objectiveOf(next) >= f0 + 1e-4 * t * slope) break
      t /= 2
    }
    // A step that cannot raise the objective (rounding near the optimum) counts as stationary next time.
    stalled = objectiveOf(next) <= f0 && size < 1e-6
    if (t === tMax && blocking >= 0) {
      next[blocking] = 0
      free[blocking] = 0
    }
    const total = next.reduce((acc, v) => acc + v, 0)
    w = Float64Array.from(next, (v) => v / total)
    for (let j = 0; j < J; j++) if (w[j] === 0) free[j] = 0
    iterations++
  }
  return { weights: fromData(w, [J]), objective: objectiveOf(w), iterations, converged }
}

/**
 * Pseudo-BMA weights (Yao et al., 2018, §4.1, without the bootstrap): $w_j \propto \exp \widehat{\mathrm{elpd}}_j$
 * with $\widehat{\mathrm{elpd}}_j = \sum_i \ell_{ij}$, the leave-one-out analogue of Akaike weights. They take no
 * account of the uncertainty in the elpd and concentrate on the best model. Throws `DomainError` for an empty matrix.
 *
 * @param looLogDensities The $n \times J$ leave-one-out log predictive densities $\ell_{ij}$.
 * @returns The weights and each model's elpd.
 *
 * @example An elpd difference of 2 gives odds of about 7 to 1
 * const lpd = [[-1, -1.5], [-1, -1.5], [-1, -1.5], [-1, -1.5]]
 * const p = pseudoBmaWeights(lpd)
 * print('elpd:', p.elpd, ' weights:', p.weights)
 */
export function pseudoBmaWeights(looLogDensities: MatrixLike): PseudoBmaWeights {
  const { data, n, J } = densities(looLogDensities, 'pseudoBmaWeights')
  const elpd = new Float64Array(J)
  for (let i = 0; i < n; i++) for (let j = 0; j < J; j++) elpd[j] += data[i * J + j]
  return { weights: fromData(softmax(elpd), [J]), elpd: fromData(elpd, [J]) }
}

/**
 * The softmax of some values, shifted by their maximum.
 *
 * @param z The values.
 * @returns $\exp(z_j - \max z) / \sum_k \exp(z_k - \max z)$.
 */
function softmax(z: Float64Array): Float64Array {
  const top = Math.max(...z)
  const e = Float64Array.from(z, (v) => Math.exp(v - top))
  const s = e.reduce((acc, v) => acc + v, 0)
  return Float64Array.from(e, (v) => v / s)
}

/**
 * Pseudo-BMA+ weights (Yao et al., 2018, §4.1): pseudo-BMA weights averaged over $B$ Bayesian-bootstrap replicates of
 * the elpd. Replicate $b$ draws $\boldsymbol{\alpha}^{(b)} \sim \mathrm{Dirichlet}(1, \dots, 1)$ over the
 * observations and sets $z^{(b)}_j = n \sum_i \alpha^{(b)}_i \ell_{ij}$; the weights are the mean over replicates of
 * $\operatorname{softmax}(\zvec^{(b)})$, as arviz-stats' `bb-pseudo-bma`. The bootstrap tempers the weights where the
 * elpd differences are uncertain. Throws `DomainError` without a stream or draws, and for draws of the wrong width.
 *
 * @param looLogDensities The $n \times J$ leave-one-out log predictive densities $\ell_{ij}$.
 * @param options The number of replicates and the stream, or the Dirichlet draws themselves.
 * @param options.B The number of replicates $B$; ignored when `draws` are given.
 * @param options.stream Randomness: replicate $b$ draws from `child(stream, 'replicate', b)`.
 * @param options.draws The Dirichlet weights themselves, $B \times n$, each row summing to 1.
 * @returns The weights, each model's elpd, the standard deviation of its bootstrapped elpd and $B$.
 *
 * @example Pseudo-BMA+ tempers pseudo-BMA when the evidence is thin
 * const lpd = [[-1, -1.4], [-0.6, -1.2], [-1.5, -0.9], [-0.8, -1.6], [-1.1, -1.0], [-0.9, -1.3]]
 * print('pseudo-BMA: ', pseudoBmaWeights(lpd).weights)
 * print('pseudo-BMA+:', pseudoBmaPlusWeights(lpd, { B: 500, stream: stream(0) }).weights)
 */
export function pseudoBmaPlusWeights(
  looLogDensities: MatrixLike,
  { B = 1000, stream, draws }: { B?: number; stream?: Stream; draws?: MatrixLike } = {},
): PseudoBmaPlusWeights {
  const where = 'pseudoBmaPlusWeights'
  const { data, n, J } = densities(looLogDensities, where)
  let alpha: Float64Array[]
  if (draws !== undefined) {
    const { data: d, m, n: width } = dense.toMatrixF64(draws, where)
    if (width !== n) throw new DomainError(where, `${where}: ${n} observations but draws of ${width}`)
    alpha = Array.from({ length: m }, (_, b) => Float64Array.from(d.subarray(b * n, b * n + n)))
  } else {
    if (!stream) throw new DomainError(where, `${where}: needs a stream (or draws)`)
    if (!(Number.isInteger(B) && B >= 1)) throw new DomainError(where, `${where}: B must be a positive integer`)
    alpha = Array.from({ length: B }, (_, b) => {
      const e = Float64Array.from(dense.data(exponential(child(stream, 'replicate', b), 1, { shape: [n] }) as Tensor))
      const s = e.reduce((acc, v) => acc + v, 0)
      return Float64Array.from(e, (v) => v / s)
    })
  }
  const R = alpha.length
  const weights = new Float64Array(J)
  const zs = alpha.map((a) => {
    const z = new Float64Array(J)
    for (let i = 0; i < n; i++) for (let j = 0; j < J; j++) z[j] += n * a[i] * data[i * J + j]
    softmax(z).forEach((v, j) => (weights[j] += v / R))
    return z
  })
  const se = Float64Array.from({ length: J }, (_, j) => {
    const mu = zs.reduce((acc, z) => acc + z[j], 0) / R
    return Math.sqrt(zs.reduce((acc, z) => acc + (z[j] - mu) ** 2, 0) / R)
  })
  const elpd = new Float64Array(J)
  for (let i = 0; i < n; i++) for (let j = 0; j < J; j++) elpd[j] += data[i * J + j]
  return { weights: fromData(weights, [J]), elpd: fromData(elpd, [J]), se: fromData(se, [J]), B: R }
}
