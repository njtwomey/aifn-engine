/**
 * New systems from old: discretisation (zero-order hold, forward Euler, Tustin), state feedback, and interconnection
 * (series, parallel, feedback).
 *
 * Discretisation and state feedback work on the state-space realisation and return state space; the interconnections
 * multiply the SISO rational functions and return a transfer function, with nothing cancelled. Systems combined must
 * share their domain and sampling interval.
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

/**
 * The elementwise sum $a + \beta b$ of two equal-length arrays.
 *
 * @param a The first array.
 * @param b The second array, of the same length.
 * @param beta The multiplier $\beta$ of `b` ($-1$ subtracts).
 * @returns A new array.
 */
const addM = (a: ArrayLike<number>, b: ArrayLike<number>, beta = 1) => Float64Array.from(a, (v, i) => v + beta * b[i])
/**
 * An array times a scalar.
 *
 * @param a The array.
 * @param k The scalar.
 * @returns A new array, each entry times `k`.
 */
const scaleM = (a: ArrayLike<number>, k: Scalar) => Float64Array.from(a, (v) => v * k)

/** Discretisation rules: zero-order hold (exact for inputs held over each step), bilinear (Tustin), forward Euler. */
export type SsmDiscretisation = 'zoh' | 'bilinear' | 'euler'

/**
 * A discrete state-space system $\xvec_k = \bar\Amat \xvec_{k-1} + \bar\Bmat \uvec_k$: `A` is $\bar\Amat$ and `B` is
 * $\bar\Bmat$, as values (traced when the inputs are).
 */
export type DiscreteSsm = { A: Value; B: Value }

/**
 * The discretisation of $\dot\xvec = \Amat\xvec + \Bmat\uvec$ with step $\Delta$ ($\Amat$ $N \times N$, $\Bmat$
 * $N \times M$):
 *
 * - `zoh`: $\bar\Amat = e^{\Delta\Amat}$, $\bar\Bmat = \int_0^\Delta e^{s\Amat} \, ds \, \Bmat$, both read from one
 *   exponential: with $\Mmat = \begin{bmatrix} \Amat & \Bmat \\ \zeros & \zeros \end{bmatrix}$,
 *   $e^{\Delta\Mmat} = \begin{bmatrix} \bar\Amat & \bar\Bmat \\ \zeros & \Imat \end{bmatrix}$ (Van Loan, 1978), so
 *   $\Amat$ need not be invertible;
 * - `bilinear` (Tustin, as S4): $\bar\Amat = (\Imat - \Delta\Amat/2)^{-1}(\Imat + \Delta\Amat/2)$,
 *   $\bar\Bmat = (\Imat - \Delta\Amat/2)^{-1}\Delta\Bmat$, which maps the stable half-plane onto the unit disc;
 * - `euler`: $\bar\Amat = \Imat + \Delta\Amat$, $\bar\Bmat = \Delta\Bmat$.
 *
 * Differentiable in $\Amat$, $\Bmat$ and $\Delta$ (through `expm` and `solve`): the one definition, which `discretise`
 * uses for control systems and `aifn-compute/nn/sequence` for learned state-space layers.
 *
 * @param A The state matrix $\Amat$, $N \times N$ (a tensor or a traced value).
 * @param B The input matrix $\Bmat$, $N \times M$ (two-dimensional, even for one input).
 * @param step The step $\Delta$, a scalar (or a traced scalar).
 * @param method The rule: `'zoh'` (zero-order hold), `'bilinear'` or `'euler'`.
 * @returns $\bar\Amat$ ($N \times N$) and $\bar\Bmat$ ($N \times M$).
 *
 * @example The lag $\dot x = -x + u$ sampled at $\Delta = 0.1$ by each rule (exact: $e^{-0.1}$ and $1 - e^{-0.1}$)
 * const A = tensor([[-1]])
 * const B = tensor([[1]])
 * for (const method of ['zoh', 'bilinear', 'euler']) {
 *   const d = discretiseSsm(A, B, 0.1, method)
 *   print(method, 'A =', d.A, ' B =', d.B)
 * }
 *
 * @example The derivative of $e^{-\Delta}$ in the step, through the matrix exponential
 * const dA = grad((step) => sum(discretiseSsm(tensor([[-1]]), tensor([[1]]), step).A))
 * print('dA/dstep =', dA(0.1))
 * print('-exp(-0.1) =', -Math.exp(-0.1))
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

/** Methods for `discretise`: zero-order hold, forward Euler, or Tustin (bilinear). */
export type DiscretisationMethod = 'zoh' | 'euler' | 'tustin'

/**
 * A continuous system sampled every `dt`, in state-space form (any representation is realised first). $\Amat_d$ and
 * $\Bmat_d$ are {@link discretiseSsm}'s (`tustin` is its `bilinear`):
 *
 * - `zoh` (zero-order hold, exact for piecewise-constant inputs): $\Amat_d = e^{\Amat\,dt}$,
 *   $\Bmat_d = \int_0^{dt} e^{\Amat s} \, ds \, \Bmat$; $\Cmat$ and $\Dmat$ unchanged.
 * - `euler` (forward difference): $\Amat_d = \Imat + \Amat\,dt$, $\Bmat_d = \Bmat\,dt$.
 * - `tustin` (bilinear, $s \approx (2/dt)(z - 1)/(z + 1)$): with $\Wmat = (\Imat - \Amat\,dt/2)^{-1}$,
 *   $\Amat_d = \Wmat(\Imat + \Amat\,dt/2)$, $\Bmat_d = \Wmat\Bmat\,dt$, and the output map
 *   $\Cmat_d = \Cmat\Wmat = \Cmat(\Amat_d + \Imat)/2$, $\Dmat_d = \Dmat + \Cmat\Bmat_d/2$ (as scipy's
 *   `cont2discrete` with `bilinear`). Stability is preserved.
 *
 * A delay $\tau$ (seconds) becomes $\tau/dt$ samples. Throws `DomainError` for a discrete system, a non-positive
 * `dt`, or (Tustin) an eigenvalue of $\Amat$ at $2/dt$.
 *
 * @param sys The continuous system.
 * @param dt The sampling interval, in the system's time unit; also the `dt` of the result.
 * @param method The rule: `'zoh'`, `'euler'` or `'tustin'`.
 * @returns The discrete system in state-space form.
 *
 * @example The integrator $1/s$ held over $dt = 0.5$ accumulates $0.5u$ per step
 * const d = discretise(transferFunction([1], [1, 0]), 0.5)
 * print('A =', d.repr.A, ' B =', d.repr.B, ' C =', d.repr.C, ' dt =', d.dt)
 *
 * @example The lag $1/(s + 1)$ by Tustin at $dt = 0.1$: $(1 + z^{-1})/21 \big/ (1 - (19/21) z^{-1})$, as scipy
 * const d = discretise(transferFunction([1], [1, 1]), 0.1, 'tustin')
 * const tf = toTransferFunction(d).repr
 * print('b =', tf.b)
 * print('a =', tf.a)
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

/**
 * The system with state feedback $\uvec = -\Kmat\xvec + \vvec$: $\Amat \leftarrow \Amat - \Bmat\Kmat$,
 * $\Cmat \leftarrow \Cmat - \Dmat\Kmat$, with $\vvec$ the new input ($\Bmat$ and $\Dmat$ unchanged).
 *
 * @param sys The system (realised in state space first; the state is that realisation's).
 * @param K The gain $\Kmat$, $m \times n$, or a vector of $n$ values for one input. Throws `ShapeError` for another
 *   size.
 * @returns The closed loop in state-space form, with the domain, `dt` and delay of `sys`.
 *
 * @example $\Kmat = [2, 3]$ puts the double integrator's poles at $-1$ and $-2$
 * const integrator = stateSpace({ A: [[0, 1], [0, 0]], B: [0, 1], C: [1, 0] })
 * const closed = stateFeedback(integrator, [2, 3])
 * print('A - BK =', closed.repr.A)
 * print('poles =', poles(closed))
 */
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

/**
 * Throws `DomainError` unless two systems share their domain and sampling interval.
 *
 * @param a The first system.
 * @param b The second system.
 * @param where The caller's name, for the error message.
 */
function sameDomain(a: LtiSystem, b: LtiSystem, where: string) {
  if (a.domain !== b.domain || a.dt !== b.dt)
    throw new DomainError(where, `${where}: the systems must share a domain and sampling interval`)
}

/**
 * A transfer function from a rational function in descending powers, in the domain of a template system.
 *
 * @param template The system whose domain and `dt` the result takes.
 * @param options The rational function.
 * @param options.num The numerator, descending powers of $s$ or $z$.
 * @param options.den The denominator, descending powers of $s$ or $z$.
 * @param delay The delay of the result.
 * @returns The transfer function: descending in $s$, or ascending in $z^{-1}$ (`DomainError` when not causal).
 */
function fromRational(template: LtiSystem, { num, den }: Rational, delay: Scalar): LtiOf<TransferFunctionForm> {
  const vec = (v: number[]): Tensor => fromData(Float64Array.from(v), [v.length])
  if (template.domain === 'continuous')
    return { ...withRepr(template, { form: 'tf', b: vec(stripLeading(num)), a: vec(stripLeading(den)) }), delay }
  const { b, a } = toAscending({ num, den })
  return { ...withRepr(template, { form: 'tf', b: vec(b), a: vec(a) }), delay }
}

/**
 * The sum of two polynomials in descending powers, aligned on the constant term.
 *
 * @param a The first polynomial's coefficients, descending.
 * @param b The second polynomial's coefficients, descending.
 * @returns The coefficients of the sum, as long as the longer of the two.
 */
const polyAdd = (a: readonly number[], b: readonly number[]) => {
  const n = Math.max(a.length, b.length)
  const pa = [...new Array(n - a.length).fill(0), ...a]
  const pb = [...new Array(n - b.length).fill(0), ...b]
  return pa.map((v, i) => v + pb[i])
}

/**
 * The series connection $G_2 G_1$ of two SISO systems ($G_1$ first; delays add), as a transfer function. Throws
 * `DomainError` when the domains or sampling intervals differ.
 *
 * @param g1 The first system $G_1$, which the input enters.
 * @param g2 The second system $G_2$, driven by $G_1$'s output.
 * @returns $G_2 G_1$, numerators and denominators multiplied (no cancellation).
 *
 * @example Two lags in a row: $1/((s + 1)(s + 2))$
 * const sys = series(transferFunction([1], [1, 1]), transferFunction([1], [1, 2]))
 * print('b =', sys.repr.b, ' a =', sys.repr.a)
 */
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

/**
 * The parallel connection $G_1 + G_2$ of two SISO systems with equal delays, as a transfer function. Throws
 * `DomainError` when the domains, sampling intervals or delays differ.
 *
 * @param g1 The first system $G_1$.
 * @param g2 The second system $G_2$, driven by the same input; the outputs add.
 * @returns $G_1 + G_2$ over the product of the denominators.
 *
 * @example $1/(s + 1) + 1/(s + 2) = (2s + 3)/(s^2 + 3s + 2)$
 * const sys = parallel(transferFunction([1], [1, 1]), transferFunction([1], [1, 2]))
 * print('b =', sys.repr.b, ' a =', sys.repr.a)
 */
export function parallel(g1: LtiSystem, g2: LtiSystem): LtiOf<TransferFunctionForm> {
  sameDomain(g1, g2, 'parallel')
  if (g1.delay !== g2.delay) throw new DomainError('parallel', 'parallel: the delays must be equal')
  const a = rationalOf(g1)
  const b = rationalOf(g2)
  const num = polyAdd(mulCoefficients(a.num, b.den), mulCoefficients(b.num, a.den))
  return fromRational(g1, { num, den: mulCoefficients(a.den, b.den) }, g1.delay)
}

/**
 * The closed loop of $L$ under negative feedback through $H$ (default unity): $L/(1 + LH)$, i.e.
 * $\mathrm{num}_L \mathrm{den}_H / (\mathrm{den}_L \mathrm{den}_H + \mathrm{num}_L \mathrm{num}_H)$. A delay in the
 * loop has no rational closed form: it throws `DomainError`, as do differing domains or sampling intervals.
 *
 * @param L The forward path $L$.
 * @param H The feedback path $H$; omitted, unity feedback ($H = 1$).
 * @returns The closed loop as a transfer function, with no delay.
 *
 * @example Unity feedback around $1/(s(s + 1))$ gives $1/(s^2 + s + 1)$
 * const loop = transferFunction([1], [1, 1, 0])
 * const closed = feedback(loop)
 * print('b =', closed.repr.b, ' a =', closed.repr.a)
 * print('poles =', poles(closed))
 *
 * @example An integrator with gain 2 in the feedback path: $1/(s + 2)$
 * const closed = feedback(transferFunction([1], [1, 0]), transferFunction([2], [1]))
 * print('b =', closed.repr.b, ' a =', closed.repr.a)
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
