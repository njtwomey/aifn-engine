/**
 * Algebraic Riccati equations. Continuous (CARE) by Kleinman's Newton iteration or the matrix sign function of the
 * Hamiltonian; discrete (DARE) by the Riccati recursion (value iteration) or the structure-preserving doubling
 * algorithm. Every solver is a traceable algorithm whose state carries the current P, the gain K, the residual of the
 * equation and the runner's `Status` flags. Sources are cited at each solver; scipy's `solve_continuous_are` and
 * `solve_discrete_are` are the references.
 */

import {
  abs,
  add,
  concat,
  eye,
  full,
  matmul,
  max,
  mul,
  neg,
  reshape,
  slice,
  sub,
  toFlat,
  transpose,
  zeros,
  complexAbs,
  realPart,
  type Matrix,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { type Algorithm } from 'aifn-compute/foundation/trace'
import { asMatrix } from './dense'
import { eig } from './eig'
import { luFactor, luSolve } from './lu'
import { lyapunov } from './lyapunov'
import { lstsq } from './svd'

/** The data of a Riccati equation: dynamics (A, B) and quadratic costs xᵀQx + uᵀRu (Q ⪰ 0, R ≻ 0). */
export type RiccatiProblem = { A: MatrixLike; B: MatrixLike; Q: MatrixLike; R: MatrixLike }

/** Why a Riccati solver cannot continue: a singular system, a non-finite iterate, or no stabilising start. */
export type RiccatiFailure = 'singular' | 'not finite' | 'not stabilisable'

/** Fields every Riccati solver state carries (with the runner's `Status`). */
export type RiccatiState = Status & {
  /** The current solution estimate P (n×n, symmetric). */
  P: Matrix
  /** The gain it gives: K = R⁻¹BᵀP (continuous) or (R + BᵀPB)⁻¹BᵀPA (discrete), m×n. */
  K: Matrix
  /** max |residual| of the Riccati equation at P, relative to 1 + max |P|. */
  residual: number
  /** max |P_t − P_{t−1}| (Infinity at step 0). */
  change: number
  converged: boolean
  /** True when an iterate is not finite. */
  diverged: boolean
  /** True when the method cannot continue for another reason (`failure` says which). */
  terminated: boolean
  /** Why the method cannot continue, or null. */
  failure: RiccatiFailure | null
}

type Data = { A: Tensor; B: Tensor; Q: Tensor; R: Tensor; n: Size; m: Size; Rinv: Tensor; G: Tensor }

const T = (x: Tensor): Tensor => transpose(x)
const mm = (...xs: Tensor[]): Tensor => xs.reduce((a, b) => matmul(a, b))
const maxAbs = (x: Tensor): number => (x.shape.reduce((a, b) => a * b, 1) === 0 ? 0 : (max(abs(x)) as number))
const allFinite = (x: Tensor): boolean => toFlat(x).every(Number.isFinite)
const symmetrise = (x: Tensor): Tensor => mul(0.5, add(x, T(x)))
const nans = (r: Size, c: Size): Tensor => full([r, c], NaN)

/** X with A X = B, or null when A is singular to working precision or either side is not finite. */
function trySolve(a: Tensor, b: Tensor): Tensor | null {
  if (!allFinite(a) || !allFinite(b)) return null
  const f = luFactor(a)
  return f.singular ? null : luSolve(f, b)
}

function problem({ A, B, Q, R }: RiccatiProblem, where: string): Data {
  const a = asMatrix(A, `${where} A`)
  const b = asMatrix(B, `${where} B`)
  const q = asMatrix(Q, `${where} Q`)
  const r0 = asMatrix(R, `${where} R`)
  const n = a.shape[0]
  const m = b.shape[1]
  if (a.shape[1] !== n || b.shape[0] !== n || q.shape[0] !== n || q.shape[1] !== n)
    throw new ShapeError(where, `${where}: A, B and Q must be n×n, n×m, n×n`)
  if (r0.shape[0] * r0.shape[1] !== m * m) throw new ShapeError(where, `${where}: R must be ${m}×${m}`)
  const r = reshape(r0, [m, m])
  const Rinv = trySolve(r, eye(m))
  if (!Rinv) throw new DomainError(where, `${where}: R is singular (it must be positive definite)`)
  return { A: a, B: b, Q: q, R: r, n, m, Rinv, G: mm(b, Rinv, T(b)) }
}

/** The CARE residual AᵀP + PA − PBR⁻¹BᵀP + Q. */
function careResidual(d: Data, P: Tensor): Tensor {
  return add(sub(add(mm(T(d.A), P), mm(P, d.A)), mm(P, d.G, P)), d.Q)
}

/** The DARE residual AᵀPA − P − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q, and the gain, or null when R + BᵀPB is singular. */
function dareParts(d: Data, P: Tensor): { residual: Tensor; K: Tensor } | null {
  const BtP = mm(T(d.B), P)
  const K = trySolve(add(d.R, mm(BtP, d.B)), mm(BtP, d.A))
  if (!K) return null
  const AtP = mm(T(d.A), P)
  const residual = add(sub(sub(mm(AtP, d.A), P), mm(AtP, d.B, K)), d.Q)
  return { residual, K }
}

const relative = (res: Tensor, P: Tensor) => maxAbs(res) / (1 + maxAbs(P))

/** A state with its flags set from `failure` (non-finite iterates diverge; other failures terminate). */
function flagged(base: Omit<RiccatiState, 'diverged' | 'terminated' | 'converged'>, converged: boolean): RiccatiState {
  return {
    ...base,
    converged: base.failure === null && converged,
    diverged: base.failure === 'not finite',
    terminated: base.failure !== null && base.failure !== 'not finite',
  }
}

function careState(
  d: Data,
  t: Size,
  P: Tensor,
  prev: Tensor | null,
  tolerance: number,
  failure: RiccatiFailure | null,
): RiccatiState {
  const Ps = symmetrise(P)
  const finite = allFinite(Ps)
  const residual = finite ? relative(careResidual(d, Ps), Ps) : NaN
  const change = prev ? maxAbs(sub(Ps, prev)) : Infinity
  return flagged(
    {
      t,
      P: Ps,
      K: mm(d.Rinv, T(d.B), Ps),
      residual,
      change,
      failure: failure ?? (finite ? null : 'not finite'),
    },
    residual <= tolerance || change <= tolerance * (1 + maxAbs(Ps)),
  )
}

const maxReal = (A: Tensor) => Math.max(...toFlat(realPart(eig(A, { vectors: false }).values)))

/**
 * An initial stabilising gain K₀ for Kleinman's iteration by Bass's method (Armstrong, 1975, "An extension of Bass'
 * algorithm for stabilizing linear continuous constant systems", IEEE TAC 20(1)): with β > max Re λ(A), solve
 * (A + βI)Z + Z(A + βI)ᵀ = 2BBᵀ; then K₀ = BᵀZ⁻¹ makes A − BK₀ stable when (A, B) is controllable.
 */
function bassGain(d: Data): Tensor | null {
  const beta = Math.max(0, maxReal(d.A)) + 1
  const Z = lyapunov(add(d.A, mul(beta, eye(d.n))), mul(-2, mm(d.B, T(d.B))))
  if (!Z.X) return null
  const ZiT = trySolve(T(Z.X), d.B)
  return ZiT ? T(ZiT) : null
}

/** Options for the Riccati solvers. */
export type RiccatiOptions = {
  /** Stop when the relative residual or the change in P is below this. Default 1e-12. */
  tolerance?: number
}

/**
 * Kleinman's Newton iteration for the CARE AᵀP + PA − PBR⁻¹BᵀP + Q = 0 (Kleinman, 1968, "On an iterative technique
 * for Riccati equation computations", IEEE TAC 13(1)): given a stabilising gain K_k, solve the Lyapunov equation
 * (A − BK_k)ᵀP + P(A − BK_k) + Q + K_kᵀRK_k = 0 (the cost of the policy u = −K_k x), then set K_{k+1} = R⁻¹BᵀP.
 * Each P is a policy evaluation, P decreases monotonically, and convergence is quadratic near the solution. `init`
 * takes `{ K0 }` (default: Bass's stabilising gain; K₀ = 0 when A is already stable).
 */
export function kleinmanIteration(
  prob: RiccatiProblem,
  options: RiccatiOptions = {},
): Algorithm<{ K0?: MatrixLike } | undefined, RiccatiState> {
  const d = problem(prob, 'kleinmanIteration')
  const tolerance = options.tolerance ?? 1e-12
  const evaluate = (K: Tensor): Tensor | null => {
    const Acl = sub(d.A, mm(d.B, K))
    return lyapunov(T(Acl), add(d.Q, mm(T(K), d.R, K))).X
  }
  const failed = (t: Size, K: Tensor, prev: Tensor | null, failure: RiccatiFailure): RiccatiState =>
    flagged({ t, P: prev ?? nans(d.n, d.n), K, residual: NaN, change: NaN, failure }, false)
  const fromGain = (t: Size, K: Tensor, prev: Tensor | null): RiccatiState => {
    if (!allFinite(K) || maxReal(sub(d.A, mm(d.B, K))) >= 0) return failed(t, K, prev, 'not stabilisable')
    const P = evaluate(K)
    if (!P) return failed(t, K, prev, 'singular')
    return careState(d, t, P, prev, tolerance, null)
  }
  return {
    name: 'kleinman-care',
    init: (start) => {
      let K: Tensor | null
      if (start?.K0 !== undefined) {
        K = asMatrix(start.K0, 'kleinmanIteration K0')
        if (K.shape[0] * K.shape[1] === d.m * d.n) K = reshape(K, [d.m, d.n])
      } else K = maxReal(d.A) < 0 ? zeros([d.m, d.n]) : bassGain(d)
      if (!K) return failed(0, nans(d.m, d.n), null, 'not stabilisable')
      return fromGain(0, K, null)
    },
    step: (s) => fromGain(s.t + 1, s.K, s.P),
  }
}

/** The state of `riccatiMatrixSign`: a Riccati state plus the sign-function iterate. */
export type SignState = RiccatiState & {
  /** The iterate Z_k → sign(H), 2n×2n. */
  Z: Matrix
  /** The determinant scaling c_k = |det Z_k|^{1/2n}. */
  scaling: number
}

/**
 * The CARE by the matrix sign function of the Hamiltonian H = [[A, −BR⁻¹Bᵀ], [−Q, −Aᵀ]] (Roberts, 1971, "Linear
 * model reduction and solution of the algebraic Riccati equation by use of the sign function"; Byers, 1987, "Solving
 * the algebraic Riccati equation with the matrix sign function"). Newton's iteration Z ← (Z/c + cZ⁻¹)/2 with
 * determinant scaling c = |det Z|^{1/2n} converges quadratically to W = sign(H), which is −1 on the stable invariant
 * subspace span[I; P]; hence [W₁₂; W₂₂ + I] P = −[W₁₁ + I; W₂₁], solved by least squares. No eigenvectors are needed.
 * `init` takes no start.
 */
export function riccatiMatrixSign(prob: RiccatiProblem, options: RiccatiOptions = {}): Algorithm<unknown, SignState> {
  const d = problem(prob, 'riccatiMatrixSign')
  const tolerance = options.tolerance ?? 1e-12
  const n = d.n
  const H = concat([concat([d.A, neg(d.G)], 1), concat([neg(d.Q), neg(T(d.A))], 1)], 0)
  const extract = (W: Tensor): Tensor | null => {
    const I = eye(n)
    const top = slice(W, [0, n], [n, 2 * n])
    const bottom = add(slice(W, [n, 2 * n], [n, 2 * n]), I)
    const lhs = concat([top, bottom], 0)
    const rhs = neg(concat([add(slice(W, [0, n], [0, n]), I), slice(W, [n, 2 * n], [0, n])], 0))
    if (!allFinite(lhs) || !allFinite(rhs)) return null
    return lstsq(lhs, rhs).x
  }
  const make = (t: Size, Z: Tensor, scaling: number, prev: Tensor | null, change: number): SignState => {
    const P = extract(Z)
    const base = P ? careState(d, t, P, prev, tolerance, null) : careState(d, t, eye(n), prev, tolerance, 'not finite')
    // Converged when the sign iterate has settled or P already solves the CARE.
    const settled = t > 0 && (change <= 1e-13 * (1 + maxAbs(Z)) || base.residual <= tolerance)
    return { ...base, converged: base.failure === null && settled, Z, scaling }
  }
  return {
    name: 'hamiltonian-sign-care',
    init: () => make(0, H, 1, null, Infinity),
    step: (s) => {
      const Zi = trySolve(s.Z, eye(2 * n))
      if (!Zi) return { ...s, t: s.t + 1, failure: 'singular', terminated: true, converged: false }
      const c = detScaling(s.Z)
      const next = mul(0.5, add(mul(1 / c, s.Z), mul(c, Zi)))
      return make(s.t + 1, next, c, s.P, maxAbs(sub(next, s.Z)))
    },
  }
}

/** |det Z|^{1/N} for an N×N matrix, from the eigenvalue moduli (a product of moduli, taken in log space). */
function detScaling(Z: Tensor): number {
  const moduli = toFlat(complexAbs(eig(Z, { vectors: false }).values))
  let logAbs = 0
  for (let i = 0; i < moduli.length; i++) logAbs += Math.log(moduli[i])
  const c = Math.exp(logAbs / moduli.length)
  return Number.isFinite(c) && c > 0 ? c : 1
}

function dareState(d: Data, t: Size, P: Tensor, prev: Tensor | null, tolerance: number): RiccatiState {
  const Ps = symmetrise(P)
  const finite = allFinite(Ps)
  const parts = finite ? dareParts(d, Ps) : null
  const residual = parts ? relative(parts.residual, Ps) : NaN
  const change = prev ? maxAbs(sub(Ps, prev)) : Infinity
  const failure: RiccatiFailure | null = !finite ? 'not finite' : parts ? null : 'singular'
  return flagged(
    { t, P: Ps, K: parts ? parts.K : nans(d.m, d.n), residual, change, failure },
    residual <= tolerance || change <= tolerance * (1 + maxAbs(Ps)),
  )
}

/**
 * The DARE P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA by the Riccati recursion (value iteration, Bellman's dynamic
 * programming backwards in time; Bertsekas, 2017, "Dynamic Programming and Optimal Control", 4th ed., §3.1): P_t is
 * the optimal cost-to-go of a horizon-t problem, starting from P₀ (default 0). Converges linearly, at a rate set by
 * the slowest closed-loop pole, when (A, B) is stabilisable and (A, Q^{1/2}) detectable. `init` takes `{ P0 }`.
 */
export function riccatiRecursion(
  prob: RiccatiProblem,
  options: RiccatiOptions = {},
): Algorithm<{ P0?: MatrixLike } | undefined, RiccatiState> {
  const d = problem(prob, 'riccatiRecursion')
  const tolerance = options.tolerance ?? 1e-12
  return {
    name: 'riccati-recursion-dare',
    init: (start) =>
      dareState(d, 0, start?.P0 === undefined ? zeros([d.n, d.n]) : asMatrix(start.P0, 'P0'), null, tolerance),
    step: (s) => {
      const AtP = mm(T(d.A), s.P)
      const next = add(d.Q, sub(mm(AtP, d.A), mm(AtP, d.B, s.K)))
      return dareState(d, s.t + 1, next, s.P, tolerance)
    },
  }
}

/** The state of `riccatiDoubling`: a Riccati state plus the doubling iterates. */
export type DoublingState = RiccatiState & { Ak: Matrix; Gk: Matrix }

/**
 * The DARE by the structure-preserving doubling algorithm (Chu, Fan, Lin & Wang, 2004, "Structure-preserving
 * algorithms for periodic discrete-time algebraic Riccati equations", Int. J. Control 77(8)): with G = BR⁻¹Bᵀ and
 * W = (I + G_kH_k)⁻¹, A_{k+1} = A_k W A_k, G_{k+1} = G_k + A_k W G_k A_kᵀ, H_{k+1} = H_k + A_kᵀ H_k W A_k. H_k → P and
 * each step doubles the horizon (H_k is the cost-to-go of horizon 2^k), so convergence is quadratic. `init` takes no
 * start.
 */
export function riccatiDoubling(prob: RiccatiProblem, options: RiccatiOptions = {}): Algorithm<unknown, DoublingState> {
  const d = problem(prob, 'riccatiDoubling')
  const tolerance = options.tolerance ?? 1e-12
  return {
    name: 'doubling-dare',
    init: () => ({ ...dareState(d, 0, d.Q, null, tolerance), Ak: d.A, Gk: d.G }),
    step: (s) => {
      const A = s.Ak
      const G = s.Gk
      const Hm = s.P
      const W = trySolve(add(eye(d.n), mm(G, Hm)), eye(d.n))
      if (!W) return { ...s, t: s.t + 1, failure: 'singular', terminated: true, converged: false }
      const AW = mm(A, W)
      const nextA = mm(AW, A)
      const nextG = add(G, mm(AW, G, T(A)))
      const nextH = add(Hm, mm(T(A), Hm, W, A))
      return { ...dareState(d, s.t + 1, nextH, Hm, tolerance), Ak: nextA, Gk: symmetrise(nextG) }
    },
  }
}
