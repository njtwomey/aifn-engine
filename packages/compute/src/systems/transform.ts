/**
 * New systems from old: discretisation (zero-order hold, forward Euler, Tustin), state feedback, and interconnection
 * (series, parallel, feedback).
 *
 * Sources: Van Loan (1978), "Computing integrals involving the matrix exponential", IEEE Trans. Automatic Control
 * 23(3) (the zero-order hold by one matrix exponential); Franklin, Powell & Workman (1998), "Digital Control of Dynamic
 * Systems", 3rd ed., §6.2 (Tustin); Ogata (2010), "Modern Control Engineering", §2-3 (block-diagram algebra).
 */

import { expm, LinAlgError, solve } from 'aifn-compute/numerics/linalg'
import {
  add,
  concat,
  dense,
  eye,
  fromData,
  mul,
  shapeOfValue,
  slice,
  sub,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { LtiSystem, MatrixLike, Scalar, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  mulCoefficients,
  rationalOf,
  stateSpace,
  stripLeading,
  toAscending,
  toStateSpace,
  withRepr,
  type LtiOf,
  type Rational,
  type StateSpaceForm,
  type TransferFunctionForm,
} from './system'

const { matMul, identity } = dense

const addM = (a: ArrayLike<number>, b: ArrayLike<number>, beta = 1) => Float64Array.from(a, (v, i) => v + beta * b[i])
const scaleM = (a: ArrayLike<number>, k: Scalar) => Float64Array.from(a, (v) => v * k)

/** Discretisation rules: zero-order hold (exact for inputs held over each step), bilinear (Tustin), forward Euler. */
export type SsmDiscretisation = 'zoh' | 'bilinear' | 'euler'

/** A discrete state-space system x_k = Ā x_{k−1} + B̄ u_k. */
export type DiscreteSsm = { A: Value; B: Value }

/**
 * The discretisation of x′ = A x + B u with step Δ (A [N, N], B [N, M]):
 *
 * - `zoh`: Ā = e^{ΔA}, B̄ = ∫₀^Δ e^{sA} ds B, both read from one exponential exp(Δ[[A, B], [0, 0]]) = [[Ā, B̄], [0, I]]
 *   (Van Loan, 1978), so A need not be invertible;
 * - `bilinear` (Tustin, as S4): Ā = (I − ΔA/2)⁻¹(I + ΔA/2), B̄ = (I − ΔA/2)⁻¹ΔB, which maps the stable half-plane onto
 *   the unit disc;
 * - `euler`: Ā = I + ΔA, B̄ = ΔB.
 *
 * Differentiable in A, B and Δ (through `expm` and `solve`): the one definition, which `discretise` uses for control
 * systems and `aifn-compute/nn/sequence` for learned state-space layers.
 */
export function discretiseSsm(A: Value, B: Value, step: Value, method: SsmDiscretisation = 'zoh'): DiscreteSsm {
  const [n] = shapeOfValue(A)
  const m = shapeOfValue(B)[1]
  const I = eye(n)
  if (method === 'euler') return { A: add(I, mul(step, A)), B: mul(step, B) }
  if (method === 'bilinear') {
    const half = mul(0.5, mul(step, A))
    const left = sub(I, half)
    return { A: solve(left, add(I, half)), B: solve(left, mul(step, B)) }
  }
  const top = concat([A, B], 1)
  const block = mul(step, concat([top, zeros([m, n + m])], 0))
  const E = expm(block as never).value as Value
  return { A: slice(E, [0, n], [0, n]), B: slice(E, [0, n], [n, n + m]) }
}

/** Methods for `discretise`. */
export type DiscretisationMethod = 'zoh' | 'euler' | 'tustin'

/**
 * A continuous system sampled every `dt`, in state-space form (any representation is realised first). A_d and B_d are
 * {@link discretiseSsm}'s (`tustin` is its `bilinear`):
 * - `zoh` (zero-order hold, exact for piecewise-constant inputs): A_d = e^{A dt}, B_d = ∫₀^dt e^{As} ds B; C and D
 *   unchanged.
 * - `euler` (forward difference): A_d = I + A dt, B_d = B dt.
 * - `tustin` (bilinear, s ≈ (2/dt)(z − 1)/(z + 1)): with W = (I − A dt/2)⁻¹, A_d = W(I + A dt/2), B_d = W B dt, and
 *   the output map C_d = C W = C (A_d + I)/2, D_d = D + C B_d/2 (as scipy's `cont2discrete` with `bilinear`). Stability
 *   is preserved.
 * A delay τ (seconds) becomes τ/dt samples.
 */
export function discretise(sys: LtiSystem, dt: Scalar, method: DiscretisationMethod = 'zoh'): LtiOf<StateSpaceForm> {
  if (sys.domain !== 'continuous') throw new DomainError('discretise', 'discretise: the system is already discrete')
  if (!(dt > 0)) throw new DomainError('discretise', 'discretise: dt must be positive')
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  const p = r.C.shape[0]
  const C = dense.data(r.C)
  const D = dense.data(r.D)
  let Ad: Float64Array
  let Bd: Float64Array
  try {
    const d = discretiseSsm(r.A, r.B, dt, method === 'tustin' ? 'bilinear' : method)
    Ad = Float64Array.from(dense.data(d.A as Tensor))
    Bd = Float64Array.from(dense.data(d.B as Tensor))
  } catch (e) {
    if (method === 'tustin' && e instanceof LinAlgError)
      throw new DomainError('discretise', 'discretise: I − A dt/2 is singular (an eigenvalue of A at 2/dt)')
    throw e
  }
  let Cd: ArrayLike<number> = C
  let Dd: ArrayLike<number> = D
  if (method === 'tustin') {
    Cd = scaleM(matMul(C, addM(Ad, identity(n)), p, n, n), 0.5)
    Dd = addM(D, scaleM(matMul(C, Bd, p, n, m), 0.5))
  }
  return stateSpace({
    A: fromData(Ad, [n, n]),
    B: fromData(Bd, [n, m]),
    C: fromData(Float64Array.from(Cd), [p, n]),
    D: fromData(Float64Array.from(Dd), [p, m]),
    dt,
    delay: sys.delay / dt,
  })
}

/** The system with state feedback u = −Kx + v: A ← A − BK, C ← C − DK. K is m×n (a vector for one input). */
export function stateFeedback(sys: LtiSystem, K: MatrixLike | VectorLike): LtiOf<StateSpaceForm> {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  const p = r.C.shape[0]
  const isMatrix =
    (K as { shape?: readonly number[] }).shape?.length === 2 || typeof (K as ArrayLike<unknown>)[0] === 'object'
  const k = isMatrix
    ? dense.toMatrixF64(K as MatrixLike, 'stateFeedback K').data
    : dense.toF64(K as VectorLike, 'stateFeedback K')
  if (k.length !== m * n) throw new ShapeError('stateFeedback', `stateFeedback: K must be ${m}×${n}`)
  const A = addM(dense.data(r.A), matMul(dense.data(r.B), k, n, m, n), -1)
  const C = addM(dense.data(r.C), matMul(dense.data(r.D), k, p, m, n), -1)
  return withRepr(sys, { ...r, A: fromData(A, [n, n]), C: fromData(C, [p, n]) })
}

function sameDomain(a: LtiSystem, b: LtiSystem, where: string) {
  if (a.domain !== b.domain || a.dt !== b.dt)
    throw new DomainError(where, `${where}: the systems must share a domain and sampling interval`)
}

function fromRational(template: LtiSystem, { num, den }: Rational, delay: Scalar): LtiOf<TransferFunctionForm> {
  const vec = (v: number[]): Tensor => fromData(Float64Array.from(v), [v.length])
  if (template.domain === 'continuous')
    return { ...withRepr(template, { form: 'tf', b: vec(stripLeading(num)), a: vec(stripLeading(den)) }), delay }
  const { b, a } = toAscending({ num, den })
  return { ...withRepr(template, { form: 'tf', b: vec(b), a: vec(a) }), delay }
}

const polyAdd = (a: readonly number[], b: readonly number[]) => {
  const n = Math.max(a.length, b.length)
  const pa = [...new Array(n - a.length).fill(0), ...a]
  const pb = [...new Array(n - b.length).fill(0), ...b]
  return pa.map((v, i) => v + pb[i])
}

/** The series connection G₂G₁ of two SISO systems (G₁ first; delays add), as a transfer function. */
export function series(g1: LtiSystem, g2: LtiSystem): LtiOf<TransferFunctionForm> {
  sameDomain(g1, g2, 'series')
  const a = rationalOf(g1)
  const b = rationalOf(g2)
  return fromRational(
    g1,
    { num: mulCoefficients(a.num, b.num), den: mulCoefficients(a.den, b.den) },
    g1.delay + g2.delay,
  )
}

/** The parallel connection G₁ + G₂ of two SISO systems with equal delays, as a transfer function. */
export function parallel(g1: LtiSystem, g2: LtiSystem): LtiOf<TransferFunctionForm> {
  sameDomain(g1, g2, 'parallel')
  if (g1.delay !== g2.delay) throw new DomainError('parallel', 'parallel: the delays must be equal')
  const a = rationalOf(g1)
  const b = rationalOf(g2)
  const num = polyAdd(mulCoefficients(a.num, b.den), mulCoefficients(b.num, a.den))
  return fromRational(g1, { num, den: mulCoefficients(a.den, b.den) }, g1.delay)
}

/**
 * The closed loop of L under negative feedback through H (default unity): L/(1 + LH), i.e.
 * num_L den_H / (den_L den_H + num_L num_H). A delay in the loop has no rational closed form: it throws.
 */
export function feedback(L: LtiSystem, H?: LtiSystem): LtiOf<TransferFunctionForm> {
  const h = H ?? withRepr(L, { form: 'tf', b: fromData(Float64Array.of(1), [1]), a: fromData(Float64Array.of(1), [1]) })
  sameDomain(L, h, 'feedback')
  if (L.delay || h.delay) throw new DomainError('feedback', 'feedback: a loop with delay is not rational; simulate it')
  const l = rationalOf(L)
  const k = rationalOf({ ...h, delay: 0 })
  const num = mulCoefficients(l.num, k.den)
  const den = polyAdd(mulCoefficients(l.den, k.den), mulCoefficients(l.num, k.num))
  return fromRational(L, { num, den }, 0)
}
