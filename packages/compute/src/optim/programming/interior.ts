/**
 * A primal–dual interior-point method for linear programs: Mehrotra's predictor–corrector (Mehrotra, 1992, "On the
 * implementation of a primal-dual interior point method", SIAM J. Optimization 2(4); Nocedal and Wright, 2006,
 * "Numerical Optimization", Algorithm 14.3) on the homogeneous self-dual embedding (Xu, Hung & Ye, 1996, "A simplified
 * homogeneous and self-dual linear programming algorithm and its implementation", Ann. Oper. Res. 62; Andersen &
 * Andersen, 2000), which tells an infeasible problem from an unbounded one and certifies both. It works on the
 * standard form of `./lp` (minimise $\cvec^\top\zvec$ subject to $\Amat\zvec = \bvec$, $\zvec \ge \zeros$), with
 * linearly dependent rows removed first, and follows the central path $\zvec \circ \svec = \mu\ones$ towards
 * $\mu = 0$. Every iterate is recorded, and `lpCentralPath` computes points on the exact central path (from
 * Mehrotra's starting point of §14.2). Linear systems are solved by the normal equations
 * $\Amat\Dmat\Amat^\top$, $\Dmat = \Zmat\Smat^{-1}$.
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
  /**
   * Stop when the relative primal and dual residuals and the relative duality gap are all below this (default 1e-9).
   * It is also the threshold of the infeasibility and unboundedness tests.
   */
  tolerance?: Scalar
  /** Fraction of the step to the boundary taken (default 0.99). */
  stepFraction?: Scalar
}

/**
 * Where the interior-point method stands: still iterating (`running`), or what the embedding has proved (`optimal`,
 * `infeasible`, `unbounded`).
 */
export type InteriorPointStatus = 'running' | 'optimal' | 'infeasible' | 'unbounded'

/** One iterate of the interior-point method (on the homogeneous self-dual embedding). */
export interface InteriorPointState extends Status {
  /**
   * The homogeneous primal iterate in standard form, length $N$ (strictly positive); the solution estimate is
   * $\zvec/\tau$.
   */
  z: Tensor
  /** Homogeneous dual variables of the kept standard-form rows, length $m$ (estimate $\yvec/\tau$). */
  y: Tensor
  /** Homogeneous dual slacks (reduced costs), length $N$ (strictly positive; estimate $\svec/\tau$). */
  s: Tensor
  /**
   * The embedding's scale $\tau > 0$: it tends to a positive limit at an optimum, and to 0 when the problem has none.
   */
  tau: Scalar
  /**
   * The embedding's gap variable $\kappa \ge 0$: it tends to 0 at an optimum, and to
   * $\bvec^\top\yvec - \cvec^\top\zvec > 0$ when the problem is infeasible or unbounded.
   */
  kappa: Scalar
  /**
   * The primal estimate $\zvec/\tau$ in the original variables, length $n$: the point drawn on the central path.
   */
  x: Tensor
  /** The duality measure $\mu = (\zvec^\top\svec + \tau\kappa) / (N + 1)$. */
  mu: Scalar
  /**
   * The centring parameter $\sigma$ of the last step (Mehrotra's $(\mu_{\text{aff}}/\mu)^3$, capped at 1); NaN at the
   * start and after a failed step.
   */
  sigma: Scalar
  /** The primal step length of the last step (0 at the start and after a failed step). */
  alphaPrimal: Scalar
  /** The dual step length of the last step; equal to `alphaPrimal`, as one step length is used for all variables. */
  alphaDual: Scalar
  /**
   * $\lVert \Amat\zvec/\tau - \bvec \rVert / (1 + \lVert \bvec \rVert)$, over the kept rows.
   */
  primalResidual: Scalar
  /** $\lVert \Amat^\top\yvec/\tau + \svec/\tau - \cvec \rVert / (1 + \lVert \cvec \rVert)$. */
  dualResidual: Scalar
  /**
   * Relative duality gap $\lvert g - \bvec^\top\tilde\yvec \rvert / (1 + \lvert g \rvert)$ with
   * $g = \cvec^\top\tilde\zvec$, at the estimates $\tilde\zvec = \zvec/\tau$, $\tilde\yvec = \yvec/\tau$.
   */
  gap: Scalar
  /** $\cvec^\top\xvec$ of the current estimate, in the original variables. */
  objective: Scalar
  /** `optimal` once converged; `infeasible` or `unbounded` once certified (the run then stops, `terminated`). */
  status: InteriorPointStatus
  /**
   * The certificate when there is no optimum. For `unbounded`, a ray $\dvec$ in the original variables with
   * $\cvec^\top\dvec < 0$ along which every constraint holds: the image of a unit ray in standard form, so not
   * itself of unit length. For `infeasible`, the unit Farkas vector $\yvec$ on the kept standard-form rows
   * ($\Amat^\top\yvec \le \zeros$, $\bvec^\top\yvec > 0$), or null when the equality rows were found inconsistent at
   * the start. Null otherwise.
   */
  certificate: Tensor | null
  /** True when `status` is `optimal`. */
  converged: boolean
  /** True on numerical failure (singular normal equations, non-finite iterates), never for infeasibility. */
  diverged: boolean
  /** True when `status` is `infeasible` or `unbounded`. */
  terminated: boolean
  /** Rows of the standard form kept after removing linearly dependent ones. */
  rowsKept: Tensor
  /** The standard form the method works on. */
  standard: StandardForm
  /**
   * The starting point's residual norms (primal `rp`, dual `rd`, gap `rg`) and $\mu$, for the relative infeasibility
   * tests (internal).
   */
  initial: { rp: Scalar; rd: Scalar; rg: Scalar; mu: Scalar }
}

/**
 * The reduced standard form: rows made linearly independent. `m` rows and `N` columns; `A` row-major, `b` and `c` as
 * in the standard form; `kept` lists the standard-form row of each row.
 */
type Reduced = { m: number; N: number; A: Float64Array; b: Float64Array; c: Float64Array; kept: number[] }

/**
 * Remove the linearly dependent rows of a standard form (`independentRows`), and report whether a removed row
 * contradicts the others.
 *
 * @param sf The standard form (not modified).
 * @returns The reduced form, and `inconsistent`, true when $\Amat\zvec = \bvec$ has no solution.
 */
function reduce(sf: StandardForm): { reduced: Reduced; inconsistent: boolean } {
  const { rows, inconsistent } = independentRows({ m: sf.m, n: sf.N, a: sf.A }, sf.b)
  const A = new Float64Array(rows.length * sf.N)
  rows.forEach((r, i) => A.set(sf.A.subarray(r * sf.N, (r + 1) * sf.N), i * sf.N))
  return {
    reduced: { m: rows.length, N: sf.N, A, b: Float64Array.from(rows, (r) => sf.b[r]), c: sf.c, kept: rows },
    inconsistent,
  }
}

/**
 * The normal matrix $\Amat\Dmat\Amat^\top$ for a diagonal $\Dmat$.
 *
 * @param R The reduced form, whose $\Amat$ ($m \times N$) is used.
 * @param d The diagonal of $\Dmat$, $N$ values.
 * @returns $\Amat\Dmat\Amat^\top$, row-major, $m \times m$ (symmetric).
 */
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

/**
 * The product $\Amat\vvec$ with the reduced form's matrix.
 *
 * @param R The reduced form ($\Amat$ is $m \times N$).
 * @param v The vector $\vvec$, $N$ values.
 * @returns $\Amat\vvec$, $m$ values.
 */
const Av = (R: Reduced, v: ArrayLike<number>) => {
  const out = new Float64Array(R.m)
  for (let i = 0; i < R.m; i++) {
    let s = 0
    for (let j = 0; j < R.N; j++) s += R.A[i * R.N + j] * v[j]
    out[i] = s
  }
  return out
}
/**
 * The product $\Amat^\top\vvec$ with the reduced form's matrix.
 *
 * @param R The reduced form ($\Amat$ is $m \times N$).
 * @param v The vector $\vvec$, $m$ values.
 * @returns $\Amat^\top\vvec$, $N$ values.
 */
const ATv = (R: Reduced, v: ArrayLike<number>) => {
  const out = new Float64Array(R.N)
  for (let i = 0; i < R.m; i++) for (let j = 0; j < R.N; j++) out[j] += R.A[i * R.N + j] * v[i]
  return out
}

/**
 * Solve the Newton system of the central-path equations,
 * $\Amat^\top\Delta\yvec + \Delta\svec = -\rvec_c$, $\Amat\Delta\zvec = -\rvec_b$,
 * $\Smat\Delta\zvec + \Zmat\Delta\svec = -\rvec_{zs}$, by the normal equations
 * $\Amat\Dmat\Amat^\top\Delta\yvec = -\rvec_b + \Amat\Smat^{-1}\rvec_{zs} - \Amat\Dmat\rvec_c$ with
 * $\Dmat = \Zmat\Smat^{-1}$. A singular normal matrix gives $\Delta\yvec = \zeros$ and `singular: true`.
 *
 * @param R The reduced form.
 * @param z The primal iterate $\zvec$, $N$ positive values.
 * @param s The dual slacks $\svec$, $N$ positive values.
 * @param rb The primal residual $\rvec_b = \Amat\zvec - \bvec$, $m$ values.
 * @param rc The dual residual $\rvec_c = \Amat^\top\yvec + \svec - \cvec$, $N$ values.
 * @param rzs The complementarity residual $\rvec_{zs}$ ($z_j s_j - \mu$ for a target $\mu$), $N$ values.
 * @returns The direction $(\Delta\zvec, \Delta\yvec, \Delta\svec)$, and whether the normal matrix was singular.
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

/**
 * The largest $\alpha \le 1$ with $\vvec + \alpha\,\Delta\vvec \ge \zeros$.
 *
 * @param v The current point $\vvec$, non-negative.
 * @param dv The direction $\Delta\vvec$, of the same length.
 * @returns $\alpha$, 1 when no entry of the direction is negative.
 */
function maxStep(v: Float64Array, dv: Float64Array): number {
  let a = 1
  for (let j = 0; j < v.length; j++) if (dv[j] < 0) a = Math.min(a, -v[j] / dv[j])
  return a
}

/**
 * The residuals of the central-path equations at $(\zvec, \yvec, \svec)$.
 *
 * @param R The reduced form.
 * @param z The primal point $\zvec$, $N$ values.
 * @param y The dual variables $\yvec$, $m$ values.
 * @param s The dual slacks $\svec$, $N$ values.
 * @returns `rb` $= \Amat\zvec - \bvec$ and `rc` $= \Amat^\top\yvec + \svec - \cvec$.
 */
function residuals(R: Reduced, z: Float64Array, y: Float64Array, s: Float64Array) {
  const rb = Av(R, z)
  for (let i = 0; i < R.m; i++) rb[i] -= R.b[i]
  const rc = ATv(R, y)
  for (let j = 0; j < R.N; j++) rc[j] += s[j] - R.c[j]
  return { rb, rc }
}

/**
 * Mehrotra's starting point (Nocedal and Wright, 2006, §14.2): the least-norm $\zvec$ with $\Amat\zvec = \bvec$ and
 * the least-squares $\yvec$ with $\svec = \cvec - \Amat^\top\yvec$, then $\zvec$ and $\svec$ shifted to be strictly
 * positive. When $\Amat\Amat^\top$ is singular it falls back to $\zvec = \svec = \ones$, $\yvec = \zeros$.
 *
 * @param R The reduced form.
 * @returns The starting point $(\zvec, \yvec, \svec)$.
 */
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

/**
 * The residual norms of the starting point (primal `rp`, dual `rd`, gap `rg`) and its $\mu$, for the relative
 * infeasibility tests.
 */
type Initial = { rp: number; rd: number; rg: number; mu: number }

/**
 * Residuals of the homogeneous self-dual model at $(\zvec, \yvec, \svec, \tau, \kappa)$.
 *
 * @param R The reduced form.
 * @param z The primal iterate $\zvec$, $N$ values.
 * @param y The dual iterate $\yvec$, $m$ values.
 * @param s The dual slacks $\svec$, $N$ values.
 * @param tau The scale $\tau$.
 * @param kappa The gap variable $\kappa$.
 * @returns `rp` $= \bvec\tau - \Amat\zvec$, `rd` $= \cvec\tau - \Amat^\top\yvec - \svec$,
 *   `rg` $= \cvec^\top\zvec - \bvec^\top\yvec + \kappa$, and `mu` $= (\zvec^\top\svec + \tau\kappa) / (N + 1)$.
 */
function hsdResiduals(R: Reduced, z: Float64Array, y: Float64Array, s: Float64Array, tau: number, kappa: number) {
  const rp = Av(R, z)
  for (let i = 0; i < R.m; i++) rp[i] = R.b[i] * tau - rp[i]
  const rd = ATv(R, y)
  for (let j = 0; j < R.N; j++) rd[j] = R.c[j] * tau - rd[j] - s[j]
  const rg = dense.dot(R.c, z) - dense.dot(R.b, y) + kappa
  const mu = (dense.dot(z, s) + tau * kappa) / (R.N + 1)
  return { rp, rd, rg, mu }
}

/**
 * Assemble a state: the solution estimate $(\zvec, \yvec, \svec)/\tau$, its convergence measures and the
 * infeasibility tests of Andersen and Andersen (2000), which certify infeasibility or unboundedness when the model's
 * residuals, or $\mu$, have shrunk below the tolerance relative to the start while $\tau$ has become small relative
 * to $\kappa$.
 *
 * @param base The standard form and the kept rows, carried from state to state.
 * @param tolerance The convergence and certification threshold.
 * @param R The reduced form.
 * @param z The primal iterate $\zvec$, $N$ values; copied into the state.
 * @param y The dual iterate $\yvec$, $m$ values; copied into the state.
 * @param s The dual slacks $\svec$, $N$ values; copied into the state.
 * @param tau The scale $\tau$.
 * @param kappa The gap variable $\kappa$.
 * @param initial The starting point's residual norms and $\mu$, or null for the starting point itself, whose own are
 *   then used.
 * @param extra The fields the caller sets: the step count `t`, `sigma` and the step lengths, and `singular`, true when
 *   the step failed (the state is then `diverged` unless it is already decided).
 * @returns The state.
 */
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
 * The Newton direction of the homogeneous self-dual model with centring $\gamma$, residual reduction $\eta$ and
 * second-order corrections, by the normal equations $\Mmat = \Amat\Dmat\Amat^\top$, $\Dmat = \Zmat\Smat^{-1}$,
 * solved twice with one factorisation: $\Mmat\pvec = \bvec + \Amat\Dmat\cvec$ and
 * $\Mmat\qvec = \eta\rvec_p + \Amat\Dmat(\eta\rvec_d - \Zmat^{-1}\rvec_{zs})$; then
 * $\Delta\yvec = \qvec + \pvec\,\Delta\tau$, $\Delta\zvec = \uvec + \vvec\,\Delta\tau$, and $\Delta\tau$ from the gap
 * row. Here $\rvec_{zs} = \gamma\mu\ones - \Zmat\Smat\ones - \cvec_{zs}$. A numerically singular $\Mmat$ gets a
 * diagonal shift of $10^{-12}$ times its largest diagonal entry.
 *
 * @param R The reduced form.
 * @param z The primal iterate $\zvec$, $N$ positive values.
 * @param s The dual slacks $\svec$, $N$ positive values.
 * @param tau The scale $\tau$.
 * @param kappa The gap variable $\kappa$.
 * @param r The model's residuals and $\mu$ at the iterate, as `hsdResiduals` returns them.
 * @param gamma The centring $\gamma$: the target is $\gamma\mu$ (0 for the affine predictor, $\sigma$ for the
 *   corrector).
 * @param eta The fraction $\eta$ of the residuals the step removes (1 for the predictor, $1 - \sigma$ for the
 *   corrector).
 * @param corr The second-order corrections: `zs`, the products $\Delta z_j \Delta s_j$ of the predictor (null for
 *   none), and `tk`, the product $\Delta\tau\,\Delta\kappa$.
 * @returns The direction `dz`, `dy`, `ds`, `dtau`, `dkappa`, or null when the system could not be solved or the
 *   direction is not finite.
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

/**
 * The largest $\alpha \le 1$ keeping $\zvec$, $\svec$, $\tau$ and $\kappa$ non-negative along a direction.
 *
 * @param z The primal iterate $\zvec$.
 * @param s The dual slacks $\svec$.
 * @param tau The scale $\tau$.
 * @param kappa The gap variable $\kappa$.
 * @param dir The direction, as `hsdDirection` returns it (its `dy` is not needed).
 * @returns $\alpha$.
 */
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

/**
 * The reduced problem is held by the state implicitly; rebuild it from the standard form and the kept rows.
 *
 * @param state A state of `linearInteriorPoint`.
 * @returns The reduced form, with fresh copies of the kept rows.
 */
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
 * Mehrotra's predictor–corrector, as a traceable algorithm with no start. The embedding adds $\tau, \kappa \ge 0$ and
 * solves $\Amat\zvec = \bvec\tau$, $\Amat^\top\yvec + \svec = \cvec\tau$,
 * $\bvec^\top\yvec - \cvec^\top\zvec = \kappa$, $\zvec \circ \svec = \zeros$ and $\tau\kappa = 0$, which always has a
 * strictly complementary solution, reached from
 * $(\zvec, \yvec, \svec, \tau, \kappa) = (\ones, \zeros, \ones, 1, 1)$ without a feasible start. If $\tau > 0$ there,
 * $(\zvec, \yvec, \svec)/\tau$ is an optimal primal–dual pair (`status: 'optimal'`). If $\tau = 0$ then
 * $\kappa = \bvec^\top\yvec - \cvec^\top\zvec > 0$, and $\yvec$ or $\zvec$ certifies that there is no optimum:
 * $\bvec^\top\yvec > 0$ with $\Amat^\top\yvec \le \zeros$ proves the primal infeasible (Farkas; `'infeasible'`), and
 * $\cvec^\top\zvec < 0$ with $\Amat\zvec = \zeros$, $\zvec \ge \zeros$ is a ray of unbounded descent (`'unbounded'`,
 * the ray in `certificate`). Each step is one predictor–corrector iteration with a common step length for all
 * variables; `x` is the current estimate $\zvec/\tau$ in the original variables. `diverged` is kept for numerical
 * failure only. Throws as `parseLP` does for an ill-formed problem, when the algorithm is made.
 *
 * @param problem The linear program, in the form of `scipy.optimize.linprog`.
 * @param options The tolerance and the fraction of the step to the boundary taken.
 * @returns The algorithm. Its start is ignored; once the run has ended or diverged, a step returns the state
 *   unchanged.
 *
 * @example Converge to the vertex of a two-variable LP
 * // Maximise x + y subject to x + 2y <= 4 and 3x + y <= 6, x, y >= 0 (so minimise -x - y).
 * const problem = { c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }
 * const tr = trace(linearInteriorPoint(problem), {}, 50)
 * print('mu =', tr.steps.map((s) => s.mu))
 * const s = tr.steps.at(-1)
 * print('status =', s.status, ' steps =', s.t)
 * print('x =', s.x)
 *
 * @example Certify that a program is infeasible
 * // x + y <= -1 has no solution with x, y >= 0: tau shrinks to 0 while kappa stays positive.
 * const s = run(linearInteriorPoint({ c: [1, 1], A_ub: [[1, 1]], b_ub: [-1] }), {}, 50)
 * print('status =', s.status)
 * print('tau =', s.tau, ' kappa =', s.kappa)
 * print('Farkas vector =', s.certificate)
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
 * `diverged` on numerical failure, or `limit` after `maxSteps`. The duals at an optimum are $\yvec/\tau$, mapped back
 * to the original rows and bounds.
 *
 * @param problem The linear program.
 * @param options `tolerance` (default 1e-9) and `maxSteps` (default 200).
 * @returns The result.
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
  /** The barrier parameters, as given ($k$ of them). */
  mu: Tensor
  /** The central point $\xvec(\mu)$ in the original variables, one row per $\mu$: shape `[k, n]`. */
  x: Tensor
  /** $\cvec^\top\xvec(\mu)$, one per $\mu$. */
  objective: Tensor
  /** 1 for each $\mu$ whose Newton iterations met the tolerance, else 0; int32. */
  converged: Tensor
}

/**
 * The exact central path of a linear program: for each $\mu$, the solution of $\Amat\zvec = \bvec$,
 * $\Amat^\top\yvec + \svec = \cvec$, $\zvec \circ \svec = \mu\ones$, $\zvec, \svec > \zeros$ (the minimiser of
 * $\cvec^\top\zvec - \mu \sum_j \log z_j$ subject to $\Amat\zvec = \bvec$; Nocedal and Wright, 2006, §14.1), in the
 * standard form of `./lp`. Each point is found by at most 100 damped Newton iterations, warm-started from the previous
 * one (the first from Mehrotra's starting point), so give `mu` in decreasing order. The path exists when the primal
 * and dual problems are both strictly feasible; a point whose iterations did not meet the tolerance is still returned,
 * flagged in `converged`.
 *
 * @param problem The linear program, in the form of `scipy.optimize.linprog`.
 * @param mu The barrier parameters $\mu > 0$, largest first.
 * @param tolerance The largest residual accepted, the complementarity residual measured relative to $\mu$.
 * @returns The central points, their objectives and convergence flags.
 *
 * @example The central path approaches the optimal vertex as mu shrinks
 * // Maximise x + y subject to x + 2y <= 4 and 3x + y <= 6, x, y >= 0: the optimum is the vertex (1.6, 1.2).
 * const problem = { c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }
 * const path = lpCentralPath(problem, [1, 0.1, 0.01, 0.001])
 * print('x(mu) =', path.x)
 * print('objective =', path.objective)
 * print('converged =', path.converged)
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
