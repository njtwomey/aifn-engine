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

/** X with A X = B, or null when A is singular to working precision. */
function solveOrNull(a: Tensor, b: Tensor): Tensor | null {
  const f = luFactor(a)
  return f.singular ? null : luSolve(f, b)
}

/** (A + Aᵀ)/2, to remove the asymmetry rounding leaves in a covariance. */
const symmetrise = (a: Tensor): Tensor => mul(0.5, add(a, transpose(a)))

/** A B Aᵀ. */
const sandwich = (a: Tensor, b: Tensor): Tensor => matmul(matmul(a, b), transpose(a))

/** Which parts of the model EM re-estimates (default all). */
export type EmEstimate = { A?: boolean; C?: boolean; Q?: boolean; R?: boolean; initial?: boolean }

/** The state of `stateSpaceEm`. */
export type StateSpaceEmState = {
  t: number
  /** The current model. */
  model: { A: Matrix; C: Matrix; Q: Matrix; R: Matrix; m0: Tensor; P0: Matrix }
  /** log p(y) under the current model; EM never decreases it. */
  logLikelihood: number
  /** The increase on the last step (NaN at step 0). */
  improvement: number
  converged: boolean
  /** True if a covariance solve failed (the step kept the previous model). */
  diverged: boolean
  /** Internal: the model as parsed. */
  current: Model
}

/**
 * Expectation–maximisation for a linear-Gaussian state-space model (Shumway & Stoffer, 1982; Ghahramani & Hinton,
 * 1996) as a traceable algorithm. E-step: the RTS smoother gives E[z_t], E[z_t z_tᵀ] and E[z_t z_{t−1}ᵀ]. M-step, with
 * sums over t = 1 … T: A = S₁₀ S₀₀⁻¹, Q = (S₁₁ − A S₁₀ᵀ)/T, C = (Σ y_t E[z_t]ᵀ) S₁₁⁻¹,
 * R = (Σ y_t y_tᵀ − C Σ E[z_t] y_tᵀ)/T, m₀ = μ_{0|T}, P₀ = P_{0|T}. Stops when the log-likelihood rises by less than
 * `tolerance` (default 1e-8, relative). No missing observations. `init` takes `{}`; the start is `model`.
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
