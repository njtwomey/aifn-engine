/**
 * Algebraic Riccati equations. Continuous (CARE) by Kleinman's Newton iteration or the matrix sign function of the
 * Hamiltonian; discrete (DARE) by the Riccati recursion (value iteration) or the structure-preserving doubling
 * algorithm. Every solver is a traceable algorithm whose state carries the current $\Pmat$, the gain $\Kmat$, the
 * residual of the
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

/**
 * The data of a Riccati equation: dynamics ($\Amat$, $\Bmat$) and quadratic costs
 * $\xvec^\top\Qmat\xvec + \uvec^\top\Rmat\uvec$ ($\Qmat \succeq 0$, $\Rmat \succ 0$).
 */
export type RiccatiProblem = { A: MatrixLike; B: MatrixLike; Q: MatrixLike; R: MatrixLike }

/** Why a Riccati solver cannot continue: a singular system, a non-finite iterate, or no stabilising start. */
export type RiccatiFailure = 'singular' | 'not finite' | 'not stabilisable'

/** Fields every Riccati solver state carries (with the runner's `Status`). */
export type RiccatiState = Status & {
  /** The current solution estimate $\Pmat$ ($n \times n$, symmetric). */
  P: Matrix
  /**
   * The gain it gives: $\Kmat = \Rmat^{-1}\Bmat^\top\Pmat$ (continuous) or
   * $(\Rmat + \Bmat^\top\Pmat\Bmat)^{-1}\Bmat^\top\Pmat\Amat$ (discrete), $m \times n$.
   */
  K: Matrix
  /** $\max |\text{residual}|$ of the Riccati equation at $\Pmat$, relative to $1 + \max |\Pmat|$. */
  residual: number
  /** $\max |\Pmat_t - \Pmat_{t-1}|$ (Infinity at step 0). */
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

/**
 * The transpose.
 *
 * @param x The matrix to transpose ($r \times c$); not modified.
 * @returns $\Xmat^\top$, a $c \times r$ matrix.
 */
const T = (x: Tensor): Tensor => transpose(x)
/**
 * The product of several matrices, left to right.
 *
 * @param xs The factors in order, at least one, with the columns of each matching the rows of the next.
 * @returns The product of all the factors (the factor itself when there is only one).
 */
const mm = (...xs: Tensor[]): Tensor => xs.reduce((a, b) => matmul(a, b))
/**
 * The largest absolute entry (0 for an empty tensor).
 *
 * @param x The tensor whose entries are examined, of any shape.
 * @returns $\max |x_i|$ over every entry, or 0 when `x` has no entries.
 */
const maxAbs = (x: Tensor): number => (x.shape.reduce((a, b) => a * b, 1) === 0 ? 0 : (max(abs(x)) as number))
/**
 * Is every entry finite?
 *
 * @param x The tensor whose entries are examined, of any shape.
 * @returns True when no entry is NaN or infinite (also for an empty tensor).
 */
const allFinite = (x: Tensor): boolean => toFlat(x).every(Number.isFinite)
/**
 * The symmetric part $(\Xmat + \Xmat^\top)/2$.
 *
 * @param x The square matrix $\Xmat$; not modified.
 * @returns A new symmetric matrix of the same size.
 */
const symmetrise = (x: Tensor): Tensor => mul(0.5, add(x, T(x)))
/**
 * An $r \times c$ matrix of NaN, for the state of a failed step.
 *
 * @param r The number of rows.
 * @param c The number of columns.
 * @returns A new $r \times c$ matrix with every entry NaN.
 */
const nans = (r: Size, c: Size): Tensor => full([r, c], NaN)

/**
 * $\Xmat$ with $\Amat\Xmat = \Bmat$, or null when $\Amat$ is singular to working precision or either side is not
 * finite.
 *
 * @param a The square coefficient matrix $\Amat$ ($n \times n$); factored by LU, not modified.
 * @param b The right-hand side $\Bmat$, with $n$ rows; its columns are solved together.
 * @returns The solution $\Xmat$, with the shape of `b`, or null when no reliable solution exists.
 */
function trySolve(a: Tensor, b: Tensor): Tensor | null {
  if (!allFinite(a) || !allFinite(b)) return null
  const f = luFactor(a)
  return f.singular ? null : luSolve(f, b)
}

/**
 * The problem's matrices as tensors, with their shapes checked: $\Amat$ and $\Qmat$ are $n \times n$, $\Bmat$ is
 * $n \times m$ and $\Rmat$ is $m \times m$.
 *
 * @param options The data of the Riccati equation, as the caller gave it.
 * @param options.A The dynamics matrix $\Amat$ ($n \times n$).
 * @param options.B The input matrix $\Bmat$ ($n \times m$).
 * @param options.Q The state cost $\Qmat$ ($n \times n$, positive semi-definite; only its shape is checked).
 * @param options.R The control cost $\Rmat$: $m \times m$, or any matrix of $m^2$ entries, which is reshaped to
 *   $m \times m$. It must be invertible: a singular one throws `DomainError`.
 * @param where The caller's name, used in error messages.
 * @returns The four matrices as tensors with the sizes $n$ and $m$, `Rinv` $= \Rmat^{-1}$ and
 *   `G` $= \Bmat\Rmat^{-1}\Bmat^\top$.
 */
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

/**
 * The CARE residual $\Amat^\top\Pmat + \Pmat\Amat - \Pmat\Bmat\Rmat^{-1}\Bmat^\top\Pmat + \Qmat$.
 *
 * @param d The checked problem data from `problem`: $\Amat$, $\Bmat$, $\Qmat$, $\Rmat$, their sizes $n$ and $m$,
 *   $\Rmat^{-1}$ and $\Gmat = \Bmat\Rmat^{-1}\Bmat^\top$. Read, not modified.
 * @param P The candidate solution $\Pmat$ ($n \times n$) at which the residual is evaluated.
 * @returns The residual as an $n \times n$ matrix: zero when $\Pmat$ solves the CARE.
 */
function careResidual(d: Data, P: Tensor): Tensor {
  return add(sub(add(mm(T(d.A), P), mm(P, d.A)), mm(P, d.G, P)), d.Q)
}

/**
 * The DARE residual
 * $\Amat^\top\Pmat\Amat - \Pmat - \Amat^\top\Pmat\Bmat(\Rmat + \Bmat^\top\Pmat\Bmat)^{-1}\Bmat^\top\Pmat\Amat + \Qmat$,
 * and the gain, or null when $\Rmat + \Bmat^\top\Pmat\Bmat$ is singular.
 *
 * @param d The checked problem data from `problem`: $\Amat$, $\Bmat$, $\Qmat$, $\Rmat$, their sizes $n$ and $m$,
 *   $\Rmat^{-1}$ and $\Gmat = \Bmat\Rmat^{-1}\Bmat^\top$. Read, not modified.
 * @param P The candidate solution $\Pmat$ ($n \times n$) at which the residual and gain are evaluated.
 * @returns `residual` ($n \times n$, zero when $\Pmat$ solves the DARE) and the gain
 *   `K` $= (\Rmat + \Bmat^\top\Pmat\Bmat)^{-1}\Bmat^\top\Pmat\Amat$ ($m \times n$), or null when that system cannot be
 *   solved.
 */
function dareParts(d: Data, P: Tensor): { residual: Tensor; K: Tensor } | null {
  const BtP = mm(T(d.B), P)
  const K = trySolve(add(d.R, mm(BtP, d.B)), mm(BtP, d.A))
  if (!K) return null
  const AtP = mm(T(d.A), P)
  const residual = add(sub(sub(mm(AtP, d.A), P), mm(AtP, d.B, K)), d.Q)
  return { residual, K }
}

/**
 * The residual's largest entry relative to the size of $\Pmat$.
 *
 * @param res The residual matrix of the Riccati equation at $\Pmat$.
 * @param P The solution estimate $\Pmat$ that sets the scale.
 * @returns $\max |\text{res}| / (1 + \max |\Pmat|)$.
 */
const relative = (res: Tensor, P: Tensor) => maxAbs(res) / (1 + maxAbs(P))

/**
 * A state with its flags set from `failure` (non-finite iterates diverge; other failures terminate).
 *
 * @param base The state without its flags: the step count `t`, `P`, `K`, `residual`, `change` and `failure`. Copied
 *   into the result, not modified.
 * @param converged Whether the convergence test passed at this state; it only counts when `failure` is null.
 * @returns The full state, with `converged`, `diverged` (the failure is 'not finite') and `terminated` (any other
 *   failure) filled in.
 */
function flagged(base: Omit<RiccatiState, 'diverged' | 'terminated' | 'converged'>, converged: boolean): RiccatiState {
  return {
    ...base,
    converged: base.failure === null && converged,
    diverged: base.failure === 'not finite',
    terminated: base.failure !== null && base.failure !== 'not finite',
  }
}

/**
 * The state of a continuous-time iteration at $\Pmat$: the gain, the relative residual of the CARE and the change from
 * the previous iterate, with its flags.
 *
 * @param d The checked problem data from `problem`: $\Amat$, $\Bmat$, $\Qmat$, $\Rmat$, their sizes $n$ and $m$,
 *   $\Rmat^{-1}$ and $\Gmat = \Bmat\Rmat^{-1}\Bmat^\top$. Read, not modified.
 * @param t The step count recorded in the state (0 for the initial state).
 * @param P The iterate $\Pmat$ ($n \times n$); it is symmetrised before use, and the symmetric part is what the state
 *   holds.
 * @param prev The previous iterate, from which `change` is measured, or null when there is none (`change` is then
 *   Infinity).
 * @param tolerance The convergence threshold: the state is converged when the relative residual is at most this, or
 *   the change is at most this times $1 + \max |\Pmat|$.
 * @param failure A failure the caller already found, recorded as given; null lets the function report 'not finite'
 *   when $\Pmat$ has a non-finite entry.
 * @returns The state at $\Pmat$, with the gain $\Kmat = \Rmat^{-1}\Bmat^\top\Pmat$.
 */
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

/**
 * The largest real part of the eigenvalues of $\Amat$ (negative when $\Amat$ is stable).
 *
 * @param A The real square matrix $\Amat$ whose eigenvalues are computed (by `eig`, without eigenvectors).
 * @returns $\max_i \operatorname{Re} \lambda_i(\Amat)$.
 */
const maxReal = (A: Tensor) => Math.max(...toFlat(realPart(eig(A, { vectors: false }).values)))

/**
 * An initial stabilising gain $\Kmat_0$ for Kleinman's iteration by Bass's method (Armstrong, 1975, "An extension of
 * Bass' algorithm for stabilizing linear continuous constant systems", IEEE TAC 20(1)): with
 * $\beta > \max \operatorname{Re} \lambda(\Amat)$, solve
 * $(\Amat + \beta\Imat)\Zmat + \Zmat(\Amat + \beta\Imat)^\top = 2\Bmat\Bmat^\top$; then
 * $\Kmat_0 = \Bmat^\top\Zmat^{-1}$ makes $\Amat - \Bmat\Kmat_0$ stable when $(\Amat, \Bmat)$ is controllable.
 *
 * @param d The checked problem data from `problem`: $\Amat$, $\Bmat$, $\Qmat$, $\Rmat$, their sizes $n$ and $m$,
 *   $\Rmat^{-1}$ and $\Gmat = \Bmat\Rmat^{-1}\Bmat^\top$. Read, not modified.
 * @returns The gain $\Kmat_0$ ($m \times n$), or null when the Lyapunov equation or $\Zmat$ is singular (no
 *   stabilising start was found).
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
  /** Stop when the relative residual or the change in $\Pmat$ is below this. Default $10^{-12}$. */
  tolerance?: number
}

/**
 * Kleinman's Newton iteration for the CARE
 * $\Amat^\top\Pmat + \Pmat\Amat - \Pmat\Bmat\Rmat^{-1}\Bmat^\top\Pmat + \Qmat = 0$ (Kleinman, 1968, "On an iterative
 * technique for Riccati equation computations", IEEE TAC 13(1)): given a stabilising gain $\Kmat_k$, solve the Lyapunov
 * equation
 * $(\Amat - \Bmat\Kmat_k)^\top\Pmat + \Pmat(\Amat - \Bmat\Kmat_k) + \Qmat + \Kmat_k^\top\Rmat\Kmat_k = 0$ (the cost of
 * the policy $\uvec = -\Kmat_k\xvec$), then set $\Kmat_{k+1} = \Rmat^{-1}\Bmat^\top\Pmat$. Each $\Pmat$ is a policy
 * evaluation, $\Pmat$ decreases monotonically, and convergence is quadratic near the solution. `init` takes `{ K0 }`
 * (default: Bass's stabilising gain; $\Kmat_0 = 0$ when $\Amat$ is already stable).
 *
 * @param prob The equation's data: the dynamics $\Amat$ ($n \times n$) and $\Bmat$ ($n \times m$) and the cost
 *   weights $\Qmat$ ($n \times n$, positive semi-definite) and $\Rmat$ ($m \times m$, positive definite). Wrong
 *   shapes throw `ShapeError`; a singular $\Rmat$ throws `DomainError`.
 * @param options The stopping tolerance (default $10^{-12}$) on the relative residual and on the change in $\Pmat$.
 * @returns The algorithm: `init` takes an optional starting gain `K0` ($m \times n$, which must stabilise
 *   $\Amat - \Bmat\Kmat_0$, else the state fails with 'not stabilisable'), and each `step` is one Newton iteration.
 *
 * @example LQR for a double integrator, by Kleinman's iteration
 * const problem = { A: [[0, 1], [0, 0]], B: [[0], [1]], Q: [[1, 0], [0, 1]], R: [[1]] }
 * const state = run(kleinmanIteration(problem), undefined, 50)
 * print('P =', state.P)
 * print('K =', state.K)
 * print('converged =', state.converged)
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
  /** The iterate $\Zmat_k \to \operatorname{sign}(\Hmat)$, $2n \times 2n$. */
  Z: Matrix
  /** The determinant scaling $c_k = |\det \Zmat_k|^{1/2n}$. */
  scaling: number
}

/**
 * The CARE by the matrix sign function of the Hamiltonian
 * $\Hmat = \begin{bmatrix} \Amat & -\Bmat\Rmat^{-1}\Bmat^\top \\ -\Qmat & -\Amat^\top \end{bmatrix}$ (Roberts, 1971,
 * "Linear model reduction and solution of the algebraic Riccati equation by use of the sign function"; Byers, 1987,
 * "Solving the algebraic Riccati equation with the matrix sign function"). Newton's iteration
 * $\Zmat \leftarrow (\Zmat/c + c\Zmat^{-1})/2$ with determinant scaling $c = |\det \Zmat|^{1/2n}$ converges
 * quadratically to $\Wmat = \operatorname{sign}(\Hmat)$, which is $-1$ on the stable invariant subspace
 * $\operatorname{span} \begin{bmatrix} \Imat \\ \Pmat \end{bmatrix}$; hence
 * $\begin{bmatrix} \Wmat_{12} \\ \Wmat_{22} + \Imat \end{bmatrix} \Pmat =
 * -\begin{bmatrix} \Wmat_{11} + \Imat \\ \Wmat_{21} \end{bmatrix}$, solved by least squares. No eigenvectors are
 * needed. `init` takes no start.
 *
 * @param prob The equation's data: the dynamics $\Amat$ ($n \times n$) and $\Bmat$ ($n \times m$) and the cost
 *   weights $\Qmat$ ($n \times n$, positive semi-definite) and $\Rmat$ ($m \times m$, positive definite). Wrong
 *   shapes throw `ShapeError`; a singular $\Rmat$ throws `DomainError`.
 * @param options The stopping tolerance (default $10^{-12}$) on the relative residual and on the change in $\Pmat$.
 * @returns The algorithm: `init` ignores its argument and starts from $\Zmat_0 = \Hmat$, and each `step` is one scaled
 *   Newton iteration, whose state also carries `Z` and `scaling`.
 *
 * @example The same CARE by the matrix sign function
 * const problem = { A: [[0, 1], [0, 0]], B: [[0], [1]], Q: [[1, 0], [0, 1]], R: [[1]] }
 * const state = run(riccatiMatrixSign(problem), undefined, 50)
 * print('P =', state.P)
 * print('converged =', state.converged)
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

/**
 * $|\det \Zmat|^{1/N}$ for an $N \times N$ matrix, from the eigenvalue moduli (a product of moduli, taken in log
 * space).
 *
 * @param Z The real square matrix $\Zmat$ ($N \times N$) whose eigenvalues are computed.
 * @returns The scaling $|\det \Zmat|^{1/N}$, or 1 when that is zero or not finite (so the iteration is left unscaled).
 */
function detScaling(Z: Tensor): number {
  const moduli = toFlat(complexAbs(eig(Z, { vectors: false }).values))
  let logAbs = 0
  for (let i = 0; i < moduli.length; i++) logAbs += Math.log(moduli[i])
  const c = Math.exp(logAbs / moduli.length)
  return Number.isFinite(c) && c > 0 ? c : 1
}

/**
 * The state of a discrete-time iteration at $\Pmat$: the gain, the relative residual of the DARE and the change from
 * the previous iterate, with its flags.
 *
 * @param d The checked problem data from `problem`: $\Amat$, $\Bmat$, $\Qmat$, $\Rmat$, their sizes $n$ and $m$,
 *   $\Rmat^{-1}$ and $\Gmat = \Bmat\Rmat^{-1}\Bmat^\top$. Read, not modified.
 * @param t The step count recorded in the state (0 for the initial state).
 * @param P The iterate $\Pmat$ ($n \times n$); it is symmetrised before use, and the symmetric part is what the state
 *   holds.
 * @param prev The previous iterate, from which `change` is measured, or null when there is none (`change` is then
 *   Infinity).
 * @param tolerance The convergence threshold: the state is converged when the relative residual is at most this, or
 *   the change is at most this times $1 + \max |\Pmat|$.
 * @returns The state at $\Pmat$, with the gain $\Kmat = (\Rmat + \Bmat^\top\Pmat\Bmat)^{-1}\Bmat^\top\Pmat\Amat$. Its
 *   failure is 'not finite' for a non-finite $\Pmat$ and 'singular' when the gain cannot be solved for ($\Kmat$ is then
 *   NaN).
 */
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
 * The DARE
 * $\Pmat = \Qmat + \Amat^\top\Pmat\Amat - \Amat^\top\Pmat\Bmat(\Rmat + \Bmat^\top\Pmat\Bmat)^{-1}\Bmat^\top\Pmat\Amat$
 * by the Riccati recursion (value iteration, Bellman's dynamic programming backwards in time; Bertsekas, 2017, "Dynamic
 * Programming and Optimal Control", 4th ed., §3.1): $\Pmat_t$ is the optimal cost-to-go of a horizon-$t$ problem,
 * starting from $\Pmat_0$ (default 0). Converges linearly, at a rate set by the slowest closed-loop pole, when
 * $(\Amat, \Bmat)$ is stabilisable and $(\Amat, \Qmat^{1/2})$ detectable. `init` takes `{ P0 }`.
 *
 * @param prob The equation's data: the dynamics $\Amat$ ($n \times n$) and $\Bmat$ ($n \times m$) and the cost
 *   weights $\Qmat$ ($n \times n$, positive semi-definite) and $\Rmat$ ($m \times m$, positive definite). Wrong
 *   shapes throw `ShapeError`; a singular $\Rmat$ throws `DomainError`.
 * @param options The stopping tolerance (default $10^{-12}$) on the relative residual and on the change in $\Pmat$.
 * @returns The algorithm: `init` takes an optional starting cost `P0` ($n \times n$, default zero), and each `step`
 *   extends the horizon by one.
 *
 * @example The discrete-time Riccati recursion
 * const problem = { A: [[1, 1], [0, 1]], B: [[0], [1]], Q: [[1, 0], [0, 1]], R: [[1]] }
 * const state = run(riccatiRecursion(problem), undefined, 500)
 * print('P =', state.P)
 * print('K =', state.K)
 * print('converged =', state.converged)
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

/**
 * The state of `riccatiDoubling`: a Riccati state plus the doubling iterates $\Amat_k$ (`Ak`) and $\Gmat_k$ (`Gk`),
 * both $n \times n$.
 */
export type DoublingState = RiccatiState & {
  /** The doubling iterate $\Amat_k$ ($n \times n$). */
  Ak: Matrix
  /** The doubling iterate $\Gmat_k$ ($n \times n$). */
  Gk: Matrix
}

/**
 * The DARE by the structure-preserving doubling algorithm (Chu, Fan, Lin & Wang, 2004, "Structure-preserving
 * algorithms for periodic discrete-time algebraic Riccati equations", Int. J. Control 77(8)): with
 * $\Gmat = \Bmat\Rmat^{-1}\Bmat^\top$ and $\Wmat = (\Imat + \Gmat_k\Hmat_k)^{-1}$,
 * $\Amat_{k+1} = \Amat_k\Wmat\Amat_k$, $\Gmat_{k+1} = \Gmat_k + \Amat_k\Wmat\Gmat_k\Amat_k^\top$,
 * $\Hmat_{k+1} = \Hmat_k + \Amat_k^\top\Hmat_k\Wmat\Amat_k$. $\Hmat_k \to \Pmat$ and each step doubles the horizon
 * ($\Hmat_k$ is the cost-to-go of horizon $2^k$), so convergence is quadratic. `init` takes no start.
 *
 * @param prob The equation's data: the dynamics $\Amat$ ($n \times n$) and $\Bmat$ ($n \times m$) and the cost
 *   weights $\Qmat$ ($n \times n$, positive semi-definite) and $\Rmat$ ($m \times m$, positive definite). Wrong
 *   shapes throw `ShapeError`; a singular $\Rmat$ throws `DomainError`.
 * @param options The stopping tolerance (default $10^{-12}$) on the relative residual and on the change in $\Pmat$.
 * @returns The algorithm: `init` ignores its argument and starts from $\Amat_0 = \Amat$, $\Gmat_0 = \Gmat$,
 *   $\Hmat_0 = \Qmat$, and each `step` is one doubling, whose state holds $\Hmat_k$ as `P` with `Ak` and `Gk`.
 *
 * @example Doubling converges in far fewer steps
 * const problem = { A: [[1, 1], [0, 1]], B: [[0], [1]], Q: [[1, 0], [0, 1]], R: [[1]] }
 * const state = run(riccatiDoubling(problem), undefined, 20)
 * print('P =', state.P)
 * print('converged =', state.converged)
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
