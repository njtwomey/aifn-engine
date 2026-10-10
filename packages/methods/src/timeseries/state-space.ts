/**
 * Maximum-likelihood estimation of a linear-Gaussian state-space model
 * $\zvec_t = \Amat\zvec_{t-1} + \wvec_t$, $\yvec_t = \Cmat\zvec_t + \vvec_t$, $\wvec_t \sim \Gauss(\zeros, \Qmat)$,
 * $\vvec_t \sim \Gauss(\zeros, \Rmat)$, $\zvec_0 \sim \Gauss(\mvec_0, \Pmat_0)$, by expectation–maximisation.
 *
 * The model and its conventions are those of `aifn-compute/inference/filtering`, whose Kalman filter and
 * Rauch–Tung–Striebel smoother give the E-step and the log-likelihood; the M-step is closed form. $n$ is the state
 * dimension, $m$ the observation dimension and $T$ the number of observations.
 */
import {
  add,
  div,
  matmul,
  mul,
  outer,
  sub,
  tensor,
  transpose,
  zeros,
  type Matrix,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { luFactor, luSolve } from 'aifn-compute/numerics/linalg'
import { toSeries, type MatrixLike, type VectorLike } from './inputs'
import { filterAll, parseModel, smoothAll, type Model, type StateSpaceModel } from 'aifn-compute/inference/filtering'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The solution $\Xmat$ of $\Amat\Xmat = \Bmat$ by LU factorisation, or null when $\Amat$ is singular to working
 * precision.
 *
 * @param a The square matrix $\Amat$.
 * @param b The right-hand side $\Bmat$, a vector or a matrix with as many rows as $\Amat$.
 * @returns $\Xmat$, with the shape of `b`, or null.
 */
function solveOrNull(a: Tensor, b: Tensor): Tensor | null {
  const f = luFactor(a)
  return f.singular ? null : luSolve(f, b)
}

/**
 * $(\Amat + \Amat^\top)/2$, to remove the asymmetry rounding leaves in a covariance.
 *
 * @param a A square matrix $\Amat$.
 * @returns The symmetric part of $\Amat$.
 */
const symmetrise = (a: Tensor): Tensor => mul(0.5, add(a, transpose(a)))

/**
 * The product $\Amat\Bmat\Amat^\top$.
 *
 * @param a The outer matrix $\Amat$.
 * @param b The inner square matrix $\Bmat$.
 * @returns $\Amat\Bmat\Amat^\top$.
 */
const sandwich = (a: Tensor, b: Tensor): Tensor => matmul(matmul(a, b), transpose(a))

/**
 * Which parts of the model EM re-estimates: `A`, `C`, `Q`, `R`, and `initial` for $\mvec_0$ and $\Pmat_0$ together.
 * Each defaults to true; a part set to false keeps its starting value.
 */
export type EmEstimate = { A?: boolean; C?: boolean; Q?: boolean; R?: boolean; initial?: boolean }

/** The state of `stateSpaceEm`. */
export type StateSpaceEmState = {
  /** EM steps taken. */
  t: number
  /** The current model. */
  model: { A: Matrix; C: Matrix; Q: Matrix; R: Matrix; m0: Tensor; P0: Matrix }
  /** $\log p(\yvec_{1:T})$ under the current model, from the Kalman filter; EM never decreases it. */
  logLikelihood: number
  /** The increase on the last step (NaN at step 0, 0 after a failed step). */
  improvement: number
  /** True when the last increase was at most `tolerance` times $1 + \lvert \log p \rvert$. */
  converged: boolean
  /** True if a covariance solve failed (the step kept the previous model). */
  diverged: boolean
  /** Internal: the model as parsed. */
  current: Model
}

/**
 * Expectation–maximisation for a linear-Gaussian state-space model (Shumway & Stoffer, 1982; Ghahramani & Hinton,
 * 1996) as a traceable algorithm. E-step: the RTS smoother gives $\expect[\zvec_t]$, $\expect[\zvec_t\zvec_t^\top]$
 * and $\expect[\zvec_t\zvec_{t-1}^\top]$. M-step, with sums over $t = 1, \dots, T$ of
 * $\Smat_{11} = \sum_t \expect[\zvec_t\zvec_t^\top]$, $\Smat_{10} = \sum_t \expect[\zvec_t\zvec_{t-1}^\top]$ and
 * $\Smat_{00} = \sum_t \expect[\zvec_{t-1}\zvec_{t-1}^\top]$: $\Amat = \Smat_{10}\Smat_{00}^{-1}$,
 * $\Qmat = (\Smat_{11} - \Amat\Smat_{10}^\top)/T$, $\Cmat = (\sum_t \yvec_t \expect[\zvec_t]^\top) \Smat_{11}^{-1}$,
 * $\Rmat = (\sum_t \yvec_t\yvec_t^\top - \Cmat \sum_t \expect[\zvec_t]\yvec_t^\top)/T$, $\mvec_0 = \muvec_{0 \mid T}$,
 * $\Pmat_0 = \Pmat_{0 \mid T}$. With $\Amat$ (or $\Cmat$) held fixed, $\Qmat$ (or $\Rmat$) is the full average of
 * the expected residual outer products. Each step runs the filter and smoother once and the filter again for the new
 * log-likelihood. A singular $\Smat_{00}$ or $\Smat_{11}$ ends the run with `diverged` and the previous model. Throws
 * `DomainError` for a non-finite observation: no missing values. `init` takes no start; the start is `model`.
 *
 * @param y The observations: a vector of $T$ scalar observations, or a $T \times m$ matrix, one row per time step.
 * @param model The starting model, in the form `aifn-compute/inference/filtering` takes.
 * @param options EM options.
 * @param options.estimate Which parts of the model to re-estimate (default all; see `EmEstimate`).
 * @param options.tolerance Stop when the log-likelihood rises by at most this much relative to
 *   $1 + \lvert \log p \rvert$ (default 1e-8).
 * @returns The algorithm; it is done when it has converged or diverged.
 *
 * @example Estimate the noise variances of a local-level model
 * // A random walk with step sd 0.5 (Q = 0.25) seen through noise of sd 1 (R = 1).
 * const s = stream(1)
 * const y = add(cumsum(normals(s, 60, 0, 0.5)), normals(s, 60, 0, 1))
 * const start = { A: [[1]], C: [[1]], Q: [[1]], R: [[1]], m0: [0], P0: [[10]] }
 * const alg = stateSpaceEm(y, start, { estimate: { A: false, C: false } })
 * print('log L at the start:', run(alg, undefined, 0).logLikelihood)
 * const st = run(alg, undefined, 15)
 * print('after', st.t, 'steps: log L =', st.logLikelihood)
 * print('Q =', st.model.Q, 'R =', st.model.R)
 */
export function stateSpaceEm(
  y: VectorLike | MatrixLike,
  model: StateSpaceModel,
  { estimate = {}, tolerance = 1e-8 }: { estimate?: EmEstimate; tolerance?: number } = {},
): Algorithm<void, StateSpaceEmState> {
  const ys = toSeries(y, 'stateSpaceEm')
  if (ys.some((r) => r.some((v) => !Number.isFinite(v))))
    throw new DomainError('stateSpaceEm', 'stateSpaceEm: observations must be finite (no missing values)')
  const want = { A: true, C: true, Q: true, R: true, initial: true, ...estimate }
  const start = parseModel(model, 'stateSpaceEm')
  const T = ys.length
  const pack = (
    md: Model,
    t: number,
    logLikelihood: number,
    improvement: number,
    diverged: boolean,
  ): StateSpaceEmState => ({
    t,
    model: md,
    logLikelihood,
    improvement,
    converged: t > 0 && Math.abs(improvement) <= tolerance * (1 + Math.abs(logLikelihood)),
    diverged,
    current: md,
  })
  return {
    name: 'state-space-em',
    init: () => pack(start, 0, filterAll(start, ys).logLikelihood, NaN, false),
    step: (s) => {
      const md = s.current
      const [n] = md.A.shape
      const [mDim] = md.C.shape
      const f = filterAll(md, ys)
      const sm = smoothAll(md, f)
      // Second moments E[z zᵀ] = P + m mᵀ.
      const moment = (m: Tensor, P: Tensor) => add(P, outer(m, m))
      const Ezz = (t: number) => moment(sm.mean[t], sm.cov[t])
      const prevM = (t: number) => (t === 0 ? sm.initialMean : sm.mean[t - 1])
      const prevEzz = (t: number) => (t === 0 ? moment(sm.initialMean, sm.initialCov) : Ezz(t - 1))
      let S11: Tensor = zeros([n, n])
      let S10: Tensor = zeros([n, n])
      let S00: Tensor = zeros([n, n])
      let Syz: Tensor = zeros([mDim, n])
      let Syy: Tensor = zeros([mDim, mDim])
      for (let t = 0; t < T; t++) {
        const mt = sm.mean[t]
        const yt = tensor(ys[t])
        S11 = add(S11, Ezz(t))
        S10 = add(S10, add(sm.lag[t], outer(mt, prevM(t))))
        S00 = add(S00, prevEzz(t))
        Syz = add(Syz, outer(yt, mt))
        Syy = add(Syy, outer(yt, yt))
      }
      let A: Tensor = md.A
      let C: Tensor = md.C
      const next: Model = { ...md }
      let failed = false
      if (want.A) {
        // A = S10 S00⁻¹  ⇔  Aᵀ = S00⁻¹ S10ᵀ.
        const sol = solveOrNull(S00, transpose(S10))
        if (sol) A = transpose(sol)
        else failed = true
      }
      if (want.Q && want.A) next.Q = symmetrise(div(sub(S11, matmul(A, transpose(S10))), T)) as Matrix
      if (want.Q && !want.A) {
        // With A fixed the M-step for Q is E[(z_t − A z_{t−1})(…)ᵀ] averaged.
        const AS10t = matmul(A, transpose(S10))
        next.Q = symmetrise(div(add(sub(sub(S11, AS10t), transpose(AS10t)), sandwich(A, S00)), T)) as Matrix
      }
      if (want.C) {
        const sol = solveOrNull(S11, transpose(Syz))
        if (sol) C = transpose(sol)
        else failed = true
      }
      if (want.R) {
        const CSzy = matmul(C, transpose(Syz))
        next.R = (
          want.C
            ? symmetrise(div(sub(Syy, CSzy), T))
            : symmetrise(div(add(sub(sub(Syy, CSzy), transpose(CSzy)), sandwich(C, S11)), T))
        ) as Matrix
      }
      next.A = A as Matrix
      next.C = C as Matrix
      if (want.initial) {
        next.m0 = sm.initialMean
        next.P0 = sm.initialCov
      }
      if (failed) return pack(md, s.t + 1, s.logLikelihood, 0, true)
      const logLikelihood = filterAll(next, ys).logLikelihood
      return pack(next, s.t + 1, logLikelihood, logLikelihood - s.logLikelihood, false)
    },
    done: (s) => s.converged || s.diverged,
  }
}
