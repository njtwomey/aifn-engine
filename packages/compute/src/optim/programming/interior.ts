/**
 * A primal–dual interior-point method for linear programs: Mehrotra's predictor–corrector (Mehrotra, 1992, "On the
 * implementation of a primal-dual interior point method", SIAM J. Optimization 2(4); Nocedal and Wright, 2006,
 * "Numerical Optimization", Algorithm 14.3) on the homogeneous self-dual embedding (Xu, Hung & Ye, 1996, "A simplified
 * homogeneous and self-dual linear programming algorithm and its implementation", Ann. Oper. Res. 62; Andersen &
 * Andersen, 2000), which tells an infeasible problem from an unbounded one and certifies both. It works on the
 * standard form min cᵀz s.t. Az = b, z ≥ 0 and follows the central path z∘s = μ1 towards μ = 0. Every iterate is
 * recorded, and `lpCentralPath` computes points on the exact central path (from Mehrotra's starting point of §14.2).
 */

import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { factorDense, solveFactored } from 'aifn-compute/numerics/linalg'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import type { Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { independentRows, intTensor, matrix, readVector, solve, vector } from './input'
import {
  dualityReport,
  parseLP,
  recoverDuals,
  standardForm,
  toOriginal,
  type LinearProgram,
  type StandardForm,
} from './lp'
import { unsolved, type LinearProgramResult } from './simplex'

/** Options for `linearInteriorPoint`. */
export interface InteriorPointOptions {
  /** Stop when the relative primal and dual residuals and the relative duality gap are all below this (default 1e-9). */
  tolerance?: Scalar
  /** Fraction of the step to the boundary taken (default 0.99). */
  stepFraction?: Scalar
}

/** Where the interior-point method stands: still iterating, or what the embedding has proved. */
export type InteriorPointStatus = 'running' | 'optimal' | 'infeasible' | 'unbounded'

/** One iterate of the interior-point method (on the homogeneous self-dual embedding). */
export interface InteriorPointState extends Status {
  /** The homogeneous primal iterate in standard form, length N (strictly positive); the solution estimate is z/τ. */
  z: Tensor
  /** Homogeneous dual variables of the standard-form rows, length m (estimate y/τ). */
  y: Tensor
  /** Homogeneous dual slacks (reduced costs), length N (strictly positive; estimate s/τ). */
  s: Tensor
  /** The embedding's scale τ > 0: → a positive limit at an optimum, → 0 when the problem has none. */
  tau: Scalar
  /** The embedding's gap variable κ ≥ 0: → 0 at an optimum, → bᵀy − cᵀz > 0 when infeasible or unbounded. */
  kappa: Scalar
  /** The primal estimate z/τ in the original variables, length n: the point drawn on the central path. */
  x: Tensor
  /** The duality measure μ = (zᵀs + τκ) / (N + 1). */
  mu: Scalar
  /** The centring parameter σ of the last step (Mehrotra's (μ_aff/μ)³). */
  sigma: Scalar
  /** Primal and dual step lengths of the last step. */
  alphaPrimal: Scalar
  alphaDual: Scalar
  /** ‖Az − b‖ / (1 + ‖b‖). */
  primalResidual: Scalar
  /** ‖Aᵀy + s − c‖ / (1 + ‖c‖). */
  dualResidual: Scalar
  /** Relative duality gap |cᵀz − bᵀy| / (1 + |cᵀz|). */
  gap: Scalar
  /** cᵀx of the current iterate. */
  objective: Scalar
  /** `optimal` once converged; `infeasible` or `unbounded` once certified (the run then stops, `terminated`). */
  status: InteriorPointStatus
  /**
   * The certificate when there is no optimum: for `unbounded`, a unit ray d in the original variables with cᵀd < 0
   * along which every constraint holds; for `infeasible`, the unit Farkas vector y on the standard-form rows (Aᵀy ≤ 0,
   * bᵀy > 0). Null otherwise.
   */
  certificate: Tensor | null
  converged: boolean
  /** True on numerical failure (singular normal equations, non-finite iterates), never for infeasibility. */
  diverged: boolean
  terminated: boolean
  /** Rows of the standard form kept after removing linearly dependent ones. */
  rowsKept: Tensor
  standard: StandardForm
  /** The starting point's residual norms and μ, for the relative infeasibility tests (internal). */
  initial: { rp: Scalar; rd: Scalar; rg: Scalar; mu: Scalar }
}

/** The reduced standard form: rows made linearly independent. */
type Reduced = { m: number; N: number; A: Float64Array; b: Float64Array; c: Float64Array; kept: number[] }

function reduce(sf: StandardForm): { reduced: Reduced; inconsistent: boolean } {
  const { rows, inconsistent } = independentRows({ m: sf.m, n: sf.N, a: sf.A }, sf.b)
  const A = new Float64Array(rows.length * sf.N)
  rows.forEach((r, i) => A.set(sf.A.subarray(r * sf.N, (r + 1) * sf.N), i * sf.N))
  return {
    reduced: { m: rows.length, N: sf.N, A, b: Float64Array.from(rows, (r) => sf.b[r]), c: sf.c, kept: rows },
    inconsistent,
  }
}

/** A D Aᵀ for a diagonal D. */
function normalMatrix(R: Reduced, d: Float64Array): Float64Array {
  const { m, N, A } = R
  const M = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let k = i; k < m; k++) {
      let s = 0
      for (let j = 0; j < N; j++) s += A[i * N + j] * d[j] * A[k * N + j]
      M[i * m + k] = s
      M[k * m + i] = s
    }
  return M
}

const Av = (R: Reduced, v: ArrayLike<number>) => {
  const out = new Float64Array(R.m)
  for (let i = 0; i < R.m; i++) {
    let s = 0
    for (let j = 0; j < R.N; j++) s += R.A[i * R.N + j] * v[j]
    out[i] = s
  }
  return out
}
const ATv = (R: Reduced, v: ArrayLike<number>) => {
  const out = new Float64Array(R.N)
  for (let i = 0; i < R.m; i++) for (let j = 0; j < R.N; j++) out[j] += R.A[i * R.N + j] * v[i]
  return out
}

/**
 * Solve the Newton system [0 Aᵀ I; A 0 0; S 0 Z] [Δz; Δy; Δs] = [−r_c; −r_b; −r_zs] by the normal equations
 * A D Aᵀ Δy = −r_b + A S⁻¹ r_zs − A D r_c with D = Z S⁻¹.
 */
function newton(
  R: Reduced,
  z: Float64Array,
  s: Float64Array,
  rb: Float64Array,
  rc: Float64Array,
  rzs: Float64Array,
): { dz: Float64Array; dy: Float64Array; ds: Float64Array; singular: boolean } {
  const d = z.map((zi, j) => zi / s[j])
  const t = new Float64Array(R.N)
  for (let j = 0; j < R.N; j++) t[j] = rzs[j] / s[j] - d[j] * rc[j]
  const rhs = Av(R, t)
  for (let i = 0; i < R.m; i++) rhs[i] -= rb[i]
  const { x: dy, singular } = solve(normalMatrix(R, d), R.m, rhs)
  const Aty = ATv(R, dy)
  const ds = new Float64Array(R.N)
  const dz = new Float64Array(R.N)
  for (let j = 0; j < R.N; j++) {
    ds[j] = -rc[j] - Aty[j]
    dz[j] = (-rzs[j] - z[j] * ds[j]) / s[j]
  }
  return { dz, dy, ds, singular }
}

/** The largest α ≤ 1 with v + α dv ≥ 0. */
function maxStep(v: Float64Array, dv: Float64Array): number {
  let a = 1
  for (let j = 0; j < v.length; j++) if (dv[j] < 0) a = Math.min(a, -v[j] / dv[j])
  return a
}

function residuals(R: Reduced, z: Float64Array, y: Float64Array, s: Float64Array) {
  const rb = Av(R, z)
  for (let i = 0; i < R.m; i++) rb[i] -= R.b[i]
  const rc = ATv(R, y)
  for (let j = 0; j < R.N; j++) rc[j] += s[j] - R.c[j]
  return { rb, rc }
}

/** Mehrotra's starting point (Nocedal and Wright, 2006, §14.2): least-norm z and least-squares y, shifted inside. */
function startingPoint(R: Reduced): { z: Float64Array; y: Float64Array; s: Float64Array } {
  const ones = new Float64Array(R.N).fill(1)
  const AAt = normalMatrix(R, ones)
  const { x: w, singular: s1 } = solve(AAt, R.m, R.b)
  const { x: y, singular: s2 } = solve(AAt, R.m, Av(R, R.c))
  if (s1 || s2) return { z: ones.slice(), y: new Float64Array(R.m), s: ones.slice() }
  const z = ATv(R, w)
  const Aty = ATv(R, y)
  const s = R.c.map((cj, j) => cj - Aty[j])
  const dz = Math.max(-1.5 * Math.min(...z, Infinity), 0)
  const ds = Math.max(-1.5 * Math.min(...s, Infinity), 0)
  for (let j = 0; j < R.N; j++) {
    z[j] += dz
    s[j] += ds
  }
  const zs = dense.dot(z, s)
  const sumZ = z.reduce((a, b) => a + b, 0)
  const sumS = s.reduce((a, b) => a + b, 0)
  const dz2 = sumS > 0 ? (0.5 * zs) / sumS : 0
  const ds2 = sumZ > 0 ? (0.5 * zs) / sumZ : 0
  for (let j = 0; j < R.N; j++) {
    z[j] = z[j] + dz2 || 1
    s[j] = s[j] + ds2 || 1
  }
  // Keep strictly positive even for degenerate data (e.g. b = 0 and c = 0).
  for (let j = 0; j < R.N; j++) {
    if (!(z[j] > 0)) z[j] = 1
    if (!(s[j] > 0)) s[j] = 1
  }
  return { z, y, s }
}

/** The residual norms of the starting point and its μ, for the relative infeasibility tests. */
type Initial = { rp: number; rd: number; rg: number; mu: number }

/** Residuals of the homogeneous self-dual model at (z, y, s, τ, κ). */
function hsdResiduals(R: Reduced, z: Float64Array, y: Float64Array, s: Float64Array, tau: number, kappa: number) {
  const rp = Av(R, z)
  for (let i = 0; i < R.m; i++) rp[i] = R.b[i] * tau - rp[i]
  const rd = ATv(R, y)
  for (let j = 0; j < R.N; j++) rd[j] = R.c[j] * tau - rd[j] - s[j]
  const rg = dense.dot(R.c, z) - dense.dot(R.b, y) + kappa
  const mu = (dense.dot(z, s) + tau * kappa) / (R.N + 1)
  return { rp, rd, rg, mu }
}

/** Assemble a state: the solution estimate (z, y, s)/τ, its convergence measures and the infeasibility tests. */
function makeState(
  base: Pick<InteriorPointState, 'standard' | 'rowsKept'>,
  tolerance: Scalar,
  R: Reduced,
  z: Float64Array,
  y: Float64Array,
  s: Float64Array,
  tau: number,
  kappa: number,
  initial: Initial | null,
  extra: Pick<InteriorPointState, 't' | 'sigma' | 'alphaPrimal' | 'alphaDual'> & { singular?: boolean },
): InteriorPointState {
  const r = hsdResiduals(R, z, y, s, tau, kappa)
  const init = initial ?? { rp: dense.norm(r.rp), rd: dense.norm(r.rd), rg: Math.abs(r.rg), mu: r.mu }
  const zt = dense.scale(1 / tau, z)
  const yt = dense.scale(1 / tau, y)
  const primalResidual = dense.norm(r.rp) / tau / (1 + dense.norm(R.b))
  const dualResidual = dense.norm(r.rd) / tau / (1 + dense.norm(R.c))
  const cz = dense.dot(R.c, zt)
  const gap = Math.abs(cz - dense.dot(R.b, yt)) / (1 + Math.abs(cz))
  const x = toOriginal(base.standard, zt)
  const converged = primalResidual < tolerance && dualResidual < tolerance && gap < tolerance
  // Andersen & Andersen's tests (as scipy's former `_linprog_ip`): the model's residuals or μ have shrunk while τ → 0
  // relative to κ, so the solution of the embedding is a certificate of infeasibility rather than an optimum.
  const rhoP = dense.norm(r.rp) / Math.max(1, init.rp)
  const rhoD = dense.norm(r.rd) / Math.max(1, init.rd)
  const rhoG = Math.abs(r.rg) / Math.max(1, init.rg)
  const rhoMu = r.mu / init.mu
  const certified =
    !converged &&
    ((rhoP < tolerance && rhoD < tolerance && rhoG < tolerance && tau < tolerance * Math.max(1, kappa)) ||
      (rhoMu < tolerance && tau < tolerance * Math.min(1, kappa)))
  const by = dense.dot(R.b, y)
  const czRaw = dense.dot(R.c, z)
  // κ = bᵀy − cᵀz > 0 at the limit, so one of the two certificates holds: bᵀy > 0 with Aᵀy ≤ 0 (no z ≥ 0 has Az = b)
  // or cᵀz < 0 with Az = 0, z ≥ 0 (a ray along which the objective decreases without bound).
  const status: InteriorPointStatus = converged
    ? 'optimal'
    : certified
      ? by > 0
        ? 'infeasible'
        : czRaw < 0
          ? 'unbounded'
          : 'running'
      : 'running'
  let certificate: Tensor | null = null
  if (status === 'infeasible') certificate = vector(dense.scale(1 / dense.norm(y), y))
  if (status === 'unbounded') certificate = vector(toOriginal(base.standard, dense.scale(1 / dense.norm(z), z), false))
  const big = Math.max(dense.maxAbs(zt), dense.maxAbs(yt))
  const diverged =
    status === 'running' &&
    (extra.singular === true ||
      !Number.isFinite(big) ||
      [primalResidual, dualResidual, gap, tau, kappa].some((v) => !Number.isFinite(v)))
  let objective = 0
  for (let j = 0; j < x.length; j++) objective += base.standard.lp.c[j] * x[j]
  return {
    ...base,
    sigma: extra.sigma,
    alphaPrimal: extra.alphaPrimal,
    alphaDual: extra.alphaDual,
    t: extra.t,
    z: vector(z),
    y: vector(y),
    s: vector(s),
    tau,
    kappa,
    x: vector(x),
    mu: r.mu,
    primalResidual,
    dualResidual,
    gap,
    objective,
    status,
    certificate,
    converged,
    diverged,
    terminated: status === 'infeasible' || status === 'unbounded',
    initial: init,
  }
}

/**
 * The Newton direction of the homogeneous self-dual model with centring γ, residual reduction η and second-order
 * corrections (cz, cτκ), by the normal equations M = A D Aᵀ, D = Z S⁻¹, solved twice with one factorisation:
 * M p = b + A D c and M q = η r_p + A D (η r_d − Z⁻¹ r_zs); then Δy = q + p Δτ, Δz = u + v Δτ, and Δτ from the gap row.
 */
function hsdDirection(
  R: Reduced,
  z: Float64Array,
  s: Float64Array,
  tau: number,
  kappa: number,
  r: ReturnType<typeof hsdResiduals>,
  gamma: number,
  eta: number,
  corr: { zs: Float64Array | null; tk: number },
) {
  const { m, N } = R
  const d = z.map((zj, j) => zj / s[j])
  // r_zs = γμ1 − Z S 1 − corr, r_tk = γμ − τκ − corr.
  const rzs = new Float64Array(N)
  for (let j = 0; j < N; j++) rzs[j] = gamma * r.mu - z[j] * s[j] - (corr.zs ? corr.zs[j] : 0)
  const rtk = gamma * r.mu - tau * kappa - corr.tk
  const M = normalMatrix(R, d)
  let factor = factorDense(M, m)
  if (factor.singular) {
    // Near the end of an infeasible run D = Z S⁻¹ spans many orders of magnitude and A D Aᵀ loses rank numerically: a
    // tiny diagonal shift (primal regularisation, as practical codes use) keeps the direction defined.
    let big = 0
    for (let i = 0; i < m; i++) big = Math.max(big, M[i * m + i])
    for (let i = 0; i < m; i++) M[i * m + i] += 1e-12 * Math.max(big, 1)
    factor = factorDense(M, m)
  }
  const dc = Float64Array.from(R.c, (c, j) => d[j] * c)
  const rhsP = Av(R, dc)
  for (let i = 0; i < m; i++) rhsP[i] += R.b[i]
  const w = new Float64Array(N)
  for (let j = 0; j < N; j++) w[j] = d[j] * (eta * r.rd[j] - rzs[j] / z[j])
  const rhsQ = Av(R, w)
  for (let i = 0; i < m; i++) rhsQ[i] += eta * r.rp[i]
  const p = solveFactored(factor, rhsP)
  const q = solveFactored(factor, rhsQ)
  if (p === null || q === null) return null
  const Atp = ATv(R, p)
  const Atq = ATv(R, q)
  const u = new Float64Array(N)
  const v = new Float64Array(N)
  for (let j = 0; j < N; j++) {
    u[j] = d[j] * (Atq[j] - eta * r.rd[j] + rzs[j] / z[j])
    v[j] = d[j] * (Atp[j] - R.c[j])
  }
  const denom = -dense.dot(R.c, v) + dense.dot(R.b, p) + kappa / tau
  const dtau = (eta * r.rg + dense.dot(R.c, u) - dense.dot(R.b, q) + rtk / tau) / denom
  const dz = Float64Array.from(u, (uj, j) => uj + v[j] * dtau)
  const dy = Float64Array.from(q, (qi, i) => qi + p[i] * dtau)
  const ds = Float64Array.from(dz, (dzj, j) => (rzs[j] - s[j] * dzj) / z[j])
  const dkappa = (rtk - kappa * dtau) / tau
  if (![dtau, dkappa].every(Number.isFinite) || !dense.allFinite(dz) || !dense.allFinite(ds)) return null
  return { dz, dy, ds, dtau, dkappa }
}

/** The largest α ≤ 1 keeping z, s, τ, κ non-negative along a direction. */
function hsdStep(
  z: Float64Array,
  s: Float64Array,
  tau: number,
  kappa: number,
  dir: { dz: Float64Array; ds: Float64Array; dtau: number; dkappa: number },
): number {
  let a = Math.min(maxStep(z, dir.dz), maxStep(s, dir.ds))
  if (dir.dtau < 0) a = Math.min(a, -tau / dir.dtau)
  if (dir.dkappa < 0) a = Math.min(a, -kappa / dir.dkappa)
  return a
}

/** The reduced problem is held by the state implicitly; rebuild it from the standard form and the kept rows. */
function reducedOf(state: InteriorPointState): Reduced {
  const sf = state.standard
  const kept = Array.from(state.rowsKept.data)
  const A = new Float64Array(kept.length * sf.N)
  kept.forEach((r, i) => A.set(sf.A.subarray(r * sf.N, (r + 1) * sf.N), i * sf.N))
  return { m: kept.length, N: sf.N, A, b: Float64Array.from(kept, (r) => sf.b[r]), c: sf.c, kept }
}

/**
 * The primal–dual interior-point method for the linear program `problem` on the homogeneous self-dual embedding (Ye,
 * Todd & Mizuno, 1994; Xu, Hung & Ye, 1996; Andersen & Andersen, 2000, "The MOSEK interior point optimizer"), with
 * Mehrotra's predictor–corrector, as a traceable algorithm with no start. The embedding adds τ, κ ≥ 0 and solves
 *
 *   Az = bτ,  Aᵀy + s = cτ,  bᵀy − cᵀz = κ,  z∘s = 0,  τκ = 0,
 *
 * which always has a strictly complementary solution, reached from (z, y, s, τ, κ) = (1, 0, 1, 1, 1) without a
 * feasible start. If τ > 0 there, (z, y, s)/τ is an optimal primal–dual pair (`status: 'optimal'`). If τ = 0 then
 * κ = bᵀy − cᵀz > 0, and y or z certifies that there is no optimum: bᵀy > 0 with Aᵀy ≤ 0 proves the primal infeasible
 * (Farkas; `'infeasible'`), and cᵀz < 0 with Az = 0, z ≥ 0 is a ray of unbounded descent (`'unbounded'`, the ray in
 * `certificate`). Each step is one predictor–corrector iteration with a common step length for all variables; `x` is
 * the current estimate z/τ in the original variables. `diverged` is kept for numerical failure only.
 */
export function linearInteriorPoint(
  problem: LinearProgram,
  options: InteriorPointOptions = {},
): Algorithm<object, InteriorPointState> {
  const tolerance = options.tolerance ?? 1e-9
  const stepFraction = options.stepFraction ?? 0.99
  const sf = standardForm(parseLP(problem))
  return {
    name: 'linear-interior-point',
    init: () => {
      const { reduced: R, inconsistent } = reduce(sf)
      const base = { standard: sf, rowsKept: intTensor(R.kept) }
      const ones = () => new Float64Array(R.N).fill(1)
      const state = makeState(base, tolerance, R, ones(), new Float64Array(R.m), ones(), 1, 1, null, {
        t: 0,
        sigma: NaN,
        alphaPrimal: 0,
        alphaDual: 0,
      })
      // Equality rows that contradict each other (found while removing dependent rows) already prove infeasibility.
      return inconsistent ? { ...state, status: 'infeasible', terminated: true } : state
    },
    step: (state) => {
      if (state.status !== 'running' || state.diverged) return state
      const R = reducedOf(state)
      const z = Float64Array.from(state.z.data)
      const y = Float64Array.from(state.y.data)
      const s = Float64Array.from(state.s.data)
      let { tau, kappa } = state
      const N = R.N
      const r = hsdResiduals(R, z, y, s, tau, kappa)
      // Predictor: the affine-scaling direction (γ = 0, η = 1).
      const aff = hsdDirection(R, z, s, tau, kappa, r, 0, 1, { zs: null, tk: 0 })
      const fail = () =>
        makeState(state, tolerance, R, z, y, s, tau, kappa, state.initial, {
          t: state.t + 1,
          sigma: NaN,
          alphaPrimal: 0,
          alphaDual: 0,
          singular: true,
        })
      if (!aff) return fail()
      const aAff = hsdStep(z, s, tau, kappa, aff)
      let muAff = (tau + aAff * aff.dtau) * (kappa + aAff * aff.dkappa)
      for (let j = 0; j < N; j++) muAff += (z[j] + aAff * aff.dz[j]) * (s[j] + aAff * aff.ds[j])
      muAff /= N + 1
      const sigma = r.mu > 0 ? Math.min(1, Math.max(0, muAff / r.mu) ** 3) : 0
      // Corrector: centre towards σμ, reduce the residuals by 1 − σ, and correct for Δz_aff∘Δs_aff, Δτ_aff Δκ_aff.
      const corr = { zs: Float64Array.from(aff.dz, (v, j) => v * aff.ds[j]), tk: aff.dtau * aff.dkappa }
      const dir = hsdDirection(R, z, s, tau, kappa, r, sigma, 1 - sigma, corr)
      if (!dir) return fail()
      const alpha = Math.min(1, stepFraction * hsdStep(z, s, tau, kappa, dir))
      for (let j = 0; j < N; j++) {
        z[j] += alpha * dir.dz[j]
        s[j] += alpha * dir.ds[j]
      }
      for (let i = 0; i < R.m; i++) y[i] += alpha * dir.dy[i]
      tau += alpha * dir.dtau
      kappa += alpha * dir.dkappa
      return makeState(state, tolerance, R, z, y, s, tau, kappa, state.initial, {
        t: state.t + 1,
        sigma,
        alphaPrimal: alpha,
        alphaDual: alpha,
      })
    },
    done: (s) => s.status !== 'running' || s.diverged,
  }
}

/**
 * Solve a linear program by the interior-point method (a `run` of `linearInteriorPoint`); `linprog`'s path. The status
 * is `optimal`, `infeasible` or `unbounded` (certified by the embedding; `ray` holds the unbounded direction),
 * `diverged` on numerical failure, or `limit` after `maxSteps`.
 */
export function interiorPointSolve(
  problem: LinearProgram,
  options: { tolerance?: Scalar; maxSteps?: Size } = {},
): LinearProgramResult {
  const s = run(linearInteriorPoint(problem, { tolerance: options.tolerance }), {}, options.maxSteps ?? 200)
  if (s.status === 'infeasible' || s.status === 'unbounded')
    return unsolved(s.standard.lp.n, s.status, s.t, 'interior-point', s.status === 'unbounded' ? s.certificate : null)
  if (!s.converged) return unsolved(s.standard.lp.n, s.diverged ? 'diverged' : 'limit', s.t, 'interior-point')
  const duals = recoverDuals(s.standard, dense.scale(1 / s.tau, s.y.data), s.rowsKept.data)
  return {
    status: 'optimal',
    x: s.x,
    objective: s.objective,
    steps: s.t,
    method: 'interior-point',
    report: dualityReport(s.standard.lp, s.x, duals),
    ray: null,
  }
}

/** Points on the central path of a linear program. */
export interface CentralPath {
  /** The barrier parameters, as given. */
  mu: Tensor
  /** The central point x(μ) in the original variables, one row per μ: shape [k, n]. */
  x: Tensor
  /** cᵀx(μ). */
  objective: Tensor
  /** True for each μ whose Newton iteration met the tolerance, int32 (0 or 1). */
  converged: Tensor
}

/**
 * The exact central path of a linear program: for each μ, the solution of Az = b, Aᵀy + s = c, z∘s = μ1, z, s > 0
 * (the minimiser of cᵀz − μ Σ log zⱼ subject to Az = b; Nocedal and Wright, 2006, §14.1). Each point is found by
 * damped Newton iterations warm-started from the previous one, so give `mu` in decreasing order. The path exists
 * when the primal and dual problems are both strictly feasible.
 */
export function lpCentralPath(problem: LinearProgram, mu: VectorLike, tolerance = 1e-10): CentralPath {
  const sf = standardForm(parseLP(problem))
  const { reduced: R } = reduce(sf)
  const mus = Array.from(readVector(mu, 'lpCentralPath: mu'))
  let { z, y, s } = startingPoint(R)
  const n = sf.lp.n
  const xs = new Float64Array(mus.length * n)
  const objective = new Float64Array(mus.length)
  const converged = new Int32Array(mus.length)
  mus.forEach((target, k) => {
    for (let it = 0; it < 100; it++) {
      const { rb, rc } = residuals(R, z, y, s)
      const rzs = z.map((zj, j) => zj * s[j] - target)
      const size = Math.max(dense.maxAbs(rb), dense.maxAbs(rc), dense.maxAbs(rzs) / Math.max(target, 1e-300))
      if (size < tolerance) {
        converged[k] = 1
        break
      }
      const dir = newton(R, z, s, rb, rc, rzs)
      if (dir.singular) break
      const a = Math.min(1, 0.995 * Math.min(maxStep(z, dir.dz), maxStep(s, dir.ds)))
      z = z.map((v, j) => v + a * dir.dz[j])
      s = s.map((v, j) => v + a * dir.ds[j])
      y = y.map((v, i) => v + a * dir.dy[i])
    }
    const x = toOriginal(sf, z)
    xs.set(x, k * n)
    objective[k] = dense.dot(sf.lp.c, x)
  })
  return {
    mu: vector(mus),
    x: matrix(xs, mus.length, n),
    objective: vector(objective),
    converged: intTensor(converged),
  }
}
