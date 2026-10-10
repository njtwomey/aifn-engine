/**
 * The one linear time-invariant system type, `LtiSystem` (after `scipy.signal`'s `lti`/`dlti` and python-control's
 * `tf`, `zpk`, `ss`): constructors for its four representations, exact conversions between them, and poles, zeros and
 * stability. Continuous systems act on $s$; discrete ones on $z$ with sampling interval `dt`.
 *
 * Conventions (as scipy): a continuous transfer function lists `b` and `a` in descending powers of $s$; a discrete one
 * in ascending powers of $z^{-1}$, $H(z) = (b_0 + b_1 z^{-1} + \dots)/(a_0 + a_1 z^{-1} + \dots)$. Zeros and poles are
 * complex128 vectors. Second-order sections are discrete only, $k \times 6$ rows $b_0, b_1, b_2, a_0, a_1, a_2$.
 * Conversions go through one intermediate, the SISO rational function $\mathrm{num}(x)/\mathrm{den}(x)$ in descending
 * powers of $x = s$ or $z$, and cancel nothing.
 *
 * Sources: Kailath (1980), "Linear Systems", §2.1 (controllable canonical realisation); Oppenheim & Schafer (2010),
 * "Discrete-Time Signal Processing", 3rd ed., §5.3 and §6.3 (pole–zero form, cascades of second-order sections);
 * the pairing of `toSecondOrderSections` follows scipy's `zpk2sos` (`pairing='nearest'`).
 */

import { eig } from 'aifn-compute/numerics/linalg'
import { complexVector, polyFromRoots, polyMul, roots, type ComplexLike } from 'aifn-compute/numerics/polynomial'
import { dense, fromData, toComplexFlat, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type {
  ComplexNumber,
  LtiSystem,
  MatrixLike,
  Representation,
  Scalar,
  Size,
  VectorLike,
} from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// ── Types ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The transfer-function representation, `{ form: 'tf', b, a }`: numerator and denominator coefficient vectors. */
export type TransferFunctionForm = Extract<Representation, { form: 'tf' }>
/** The zeros–poles–gain representation, `{ form: 'zpk', zeros, poles, gain }`: complex128 roots and a real gain. */
export type ZerosPolesGainForm = Extract<Representation, { form: 'zpk' }>
/** The state-space representation, `{ form: 'ss', A, B, C, D }`: the four matrices as tensors. */
export type StateSpaceForm = Extract<Representation, { form: 'ss' }>
/** The second-order-sections representation, `{ form: 'sos', sections }`: a $k \times 6$ tensor of sections. */
export type SecondOrderSectionsForm = Extract<Representation, { form: 'sos' }>

/** An `LtiSystem` known to hold representation `R`, so `sys.repr.b` (etc.) type-checks. */
export type LtiOf<R extends Representation> = LtiSystem & { readonly repr: R }

/** Options shared by the constructors. */
export type SystemOptions = {
  /** Sampling interval of a discrete system; omitted or null for continuous time. */
  dt?: Scalar | null
  /** Input delay: seconds for a continuous system, samples for a discrete one. Default 0. */
  delay?: Scalar
}

/**
 * The `LtiSystem` record around a representation, after checking `dt` and the delay. Throws `DomainError` for a
 * non-positive `dt` or a negative delay.
 *
 * @param repr The representation the system holds, stored as given.
 * @param options The sampling interval and delay of the system.
 * @param options.dt The sampling interval: `null` makes a continuous system, a positive number a discrete one.
 * @param options.delay The input delay (seconds when continuous, samples when discrete).
 * @param where The caller's name, for error messages.
 * @returns The system, continuous when `dt` is `null` and discrete otherwise.
 */
function make<R extends Representation>(repr: R, { dt = null, delay = 0 }: SystemOptions, where: string): LtiOf<R> {
  if (dt !== null && !(dt > 0)) throw new DomainError(where, `${where}: dt must be positive (or null for continuous)`)
  if (!(delay >= 0)) throw new DomainError(where, `${where}: the delay must be non-negative`)
  return { kind: 'lti', domain: dt === null ? 'continuous' : 'discrete', dt, delay, repr }
}

/**
 * A float64 vector tensor of a list of numbers (copied).
 *
 * @param v The values.
 * @returns A tensor of shape $[n]$ holding a copy of `v`.
 */
const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

/**
 * The coefficients without leading zeros (keeping one coefficient, so all zeros give `[0]`).
 *
 * @param c Coefficients in descending powers; not modified.
 * @returns A new list from the first nonzero coefficient on (or the last coefficient alone).
 */
export function stripLeading(c: number[]): number[] {
  let k = 0
  while (k < c.length - 1 && c[k] === 0) k++
  return c.slice(k)
}

// ── Constructors ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A SISO transfer function $H = b/a$. Continuous (no `dt`): coefficients in descending powers of $s$, leading zeros
 * stripped; improper functions are allowed for frequency responses but cannot be realised in state space. Discrete
 * (`dt` given): coefficients in ascending powers of $z^{-1}$, as `scipy.signal.lfilter` and `freqz` read them, kept as
 * given. Throws `DomainError` for a zero denominator, or a discrete one whose first coefficient $a_0$ is 0.
 *
 * @param b The numerator coefficients (an empty list is the zero numerator).
 * @param a The denominator coefficients; not all zero.
 * @param options The sampling interval (`dt`; omitted or `null` for continuous time) and input delay.
 * @returns The system in transfer-function form.
 *
 * @example A first-order lag $1/(s + 1)$: its pole, and the gain $1/\sqrt{2}$ at the corner frequency
 * const lag = transferFunction([1], [1, 1])
 * print('poles =', poles(lag))
 * print('|H(i)| =', bode(lag, [1]).magnitude)
 *
 * @example The two-tap moving average $(1 + z^{-1})/2$, with its zero at the Nyquist frequency $z = -1$
 * const avg = transferFunction([0.5, 0.5], [1], { dt: 1 })
 * print('domain =', avg.domain)
 * print('zeros =', systemZeros(avg))
 */
export function transferFunction(
  b: VectorLike,
  a: VectorLike,
  options: SystemOptions = {},
): LtiOf<TransferFunctionForm> {
  const discrete = options.dt !== undefined && options.dt !== null
  let bs = Array.from(dense.toF64(b, 'transferFunction b'))
  let as = Array.from(dense.toF64(a, 'transferFunction a'))
  if (!discrete) {
    bs = stripLeading(bs)
    as = stripLeading(as)
  }
  if (as.length === 0 || as.every((v) => v === 0))
    throw new DomainError('transferFunction', 'transferFunction: the denominator must be nonzero')
  if (discrete && as[0] === 0) throw new DomainError('transferFunction', 'transferFunction: a[0] must be nonzero')
  if (bs.length === 0) bs = [0]
  return make({ form: 'tf', b: vec(bs), a: vec(as) }, options, 'transferFunction')
}

/**
 * A SISO system from its zeros, poles and gain, $H(x) = k \prod_i (x - z_i) / \prod_j (x - p_j)$ with $x = s$ or $z$.
 * Zeros and poles are `ComplexLike` (real values, `{ re, im }` lists, or real or complex128 tensors), stored as
 * complex128 vectors; complex ones must come in conjugate pairs for the system to be real (conversions to other forms
 * throw otherwise).
 *
 * @param zeros The zeros $z_i$ (an empty list for none).
 * @param poles The poles $p_j$.
 * @param gain The gain $k$, the ratio of the leading coefficients of numerator and denominator.
 * @param options The sampling interval (`dt`; omitted or `null` for continuous time) and input delay.
 * @returns The system in zeros–poles–gain form.
 *
 * @example $2/((s + 1)(s + 2))$ as a transfer function
 * const sys = zerosPolesGain([], [-1, -2], 2)
 * const tf = toTransferFunction(sys).repr
 * print('b =', tf.b)
 * print('a =', tf.a)
 *
 * @example A conjugate pair of poles $-1 \pm 2i$ gives the real denominator $s^2 + 2s + 5$
 * const sys = zerosPolesGain([], [{ re: -1, im: 2 }, { re: -1, im: -2 }], 5)
 * print('a =', toTransferFunction(sys).repr.a)
 */
export function zerosPolesGain(
  zeros: ComplexLike,
  poles: ComplexLike,
  gain: Scalar,
  options: SystemOptions = {},
): LtiOf<ZerosPolesGainForm> {
  const z = complexVector(zeros, 'zerosPolesGain zeros')
  const p = complexVector(poles, 'zerosPolesGain poles')
  return make({ form: 'zpk', zeros: z, poles: p, gain }, options, 'zerosPolesGain')
}

/**
 * The arguments of `stateSpace`: `B` may be a vector (one input), `C` defaults to $\Imat$ (the state is the output),
 * with the `SystemOptions` beside them.
 */
export type StateSpaceInput = {
  /** The state matrix $\Amat$, $n \times n$. */
  A: MatrixLike
  /** The input matrix $\Bmat$, $n \times m$, or a vector of $n$ values for one input. */
  B: MatrixLike | VectorLike
  /** The output matrix $\Cmat$, $p \times n$, or a vector of $n$ values for one output; default $\Imat$. */
  C?: MatrixLike | VectorLike
  /** The feedthrough $\Dmat$, $p \times m$ (a scalar or a row-major vector of $pm$ values also reads); default 0. */
  D?: MatrixLike | VectorLike | Scalar
} & SystemOptions

/**
 * A scalar, vector or matrix argument as row-major data with its shape: a scalar is $1 \times 1$, a vector a column.
 *
 * @param x The value: a number, a vector, or a matrix (nested rows or a 2-d tensor).
 * @param where The caller's name, for error messages.
 * @returns `d`, the row-major data, and its rows `r` and columns `c`.
 */
const asMatrix = (x: MatrixLike | VectorLike | Scalar, where: string): { d: dense.F64; r: Size; c: Size } => {
  if (typeof x === 'number') return { d: Float64Array.of(x), r: 1, c: 1 }
  const shape = (x as { shape?: readonly number[] }).shape
  const nested = !shape && typeof (x as ArrayLike<unknown>)[0] === 'object'
  if ((shape && shape.length === 2) || nested) {
    const { data, m, n } = dense.toMatrixF64(x as MatrixLike, where)
    return { d: data, r: m, c: n }
  }
  const d = dense.toF64(x as VectorLike, where)
  return { d, r: d.length, c: 1 }
}

/**
 * A state-space system $\dot\xvec = \Amat\xvec + \Bmat\uvec$, $\yvec = \Cmat\xvec + \Dmat\uvec$ (continuous) or
 * $\xvec_{k+1} = \Amat\xvec_k + \Bmat\uvec_k$, $\yvec_k = \Cmat\xvec_k + \Dmat\uvec_k$ (with `dt`), with shapes
 * checked (`ShapeError` otherwise). A vector `B` is one input (a column); a vector `C` of $n$ values is one output (a
 * row; when $n = 1$ a vector `C` is a column, one output per entry); `C` defaults to the identity and `D` to zeros.
 *
 * @param input The matrices `A`, `B`, `C`, `D` and the options `dt` and `delay`.
 * @returns The system in state-space form, with $\Amat$ $n \times n$, $\Bmat$ $n \times m$, $\Cmat$ $p \times n$ and
 *   $\Dmat$ $p \times m$.
 *
 * @example The double integrator $\ddot y = u$, position measured
 * const integrator = stateSpace({ A: [[0, 1], [0, 0]], B: [0, 1], C: [1, 0] })
 * print('dimensions =', dimensions(integrator))
 * print('poles =', poles(integrator))
 * print('C =', integrator.repr.C)
 */
export function stateSpace(input: StateSpaceInput): LtiOf<StateSpaceForm> {
  const where = 'stateSpace'
  const a = asMatrix(input.A, `${where} A`)
  const n = a.r
  if (a.c !== n) throw new ShapeError(where, `${where}: A must be square, got ${a.r}×${a.c}`)
  const b = asMatrix(input.B, `${where} B`)
  if (b.r !== n) throw new ShapeError(where, `${where}: B has ${b.r} rows, A is ${n}×${n}`)
  let c = input.C === undefined ? { d: dense.identity(n), r: n, c: n } : asMatrix(input.C, `${where} C`)
  if (c.c === 1 && c.r === n && n !== 1) c = { d: c.d, r: 1, c: n }
  if (c.c !== n) throw new ShapeError(where, `${where}: C has ${c.c} columns, A is ${n}×${n}`)
  const d = input.D === undefined ? { d: new Float64Array(c.r * b.c), r: c.r, c: b.c } : asMatrix(input.D, `${where} D`)
  if (d.r * d.c !== c.r * b.c) throw new ShapeError(where, `${where}: D must be ${c.r}×${b.c}`)
  return make(
    {
      form: 'ss',
      A: fromData(a.d, [n, n]),
      B: fromData(b.d, [n, b.c]),
      C: fromData(c.d, [c.r, n]),
      D: fromData(Float64Array.from(d.d), [c.r, b.c]),
    },
    input,
    where,
  )
}

/**
 * A discrete system as a cascade of second-order sections, `sections` $k \times 6$ with rows
 * $b_0, b_1, b_2, a_0, a_1, a_2$ (each section $(b_0 + b_1 z^{-1} + b_2 z^{-2})/(a_0 + a_1 z^{-1} + a_2 z^{-2})$), as
 * `scipy.signal.sosfilt` reads them. Throws `DomainError` when a section's $a_0$ is 0.
 *
 * @param sections The sections, one row of six coefficients each.
 * @param options The sampling interval (`dt`, default 1: sections are always discrete) and input delay in samples.
 * @returns The system in second-order-sections form.
 *
 * @example One section with poles at $z = 0.5$ and $0.6$
 * const sys = secondOrderSections([[1, 2, 1, 1, -1.1, 0.3]])
 * print('dt =', sys.dt)
 * print('poles =', poles(sys))
 * print('zeros =', systemZeros(sys))
 */
export function secondOrderSections(sections: MatrixLike, options: SystemOptions = {}): LtiOf<SecondOrderSectionsForm> {
  const { data, m } = dense.toMatrixF64(sections, 'secondOrderSections', undefined, 6)
  for (let k = 0; k < m; k++)
    if (data[k * 6 + 3] === 0) throw new DomainError('secondOrderSections', 'secondOrderSections: a₀ must be nonzero')
  return make(
    { form: 'sos', sections: fromData(data, [m, 6]) },
    { ...options, dt: options.dt ?? 1 },
    'secondOrderSections',
  )
}

// ── The shared intermediate: num(x)/den(x) in descending powers of x = s or z ────────────────────────────────────────

/**
 * A SISO rational function $\mathrm{num}(x)/\mathrm{den}(x)$: `num` and `den` hold the coefficients in descending
 * powers of $x$ ($s$ or $z$).
 */
export type Rational = { num: number[]; den: number[] }

/**
 * The product of two real coefficient lists (`aifn-compute/numerics/polynomial`'s `polyMul`, as plain numbers).
 *
 * @param a The first polynomial's coefficients, descending (ascending works the same way).
 * @param b The second polynomial's coefficients, in the same order as `a`.
 * @returns The coefficients of the product, `a.length + b.length - 1` of them.
 */
export const mulCoefficients = (a: readonly number[], b: readonly number[]): number[] =>
  toFlat(polyMul(Float64Array.from(a), Float64Array.from(b)) as Tensor)

/**
 * Discrete ascending $z^{-1}$ coefficients padded with trailing zeros to equal length are descending powers of $z$
 * (both multiplied by the same power of $z$).
 *
 * @param b The numerator, ascending in $z^{-1}$.
 * @param a The denominator, ascending in $z^{-1}$.
 * @returns The same function as a `Rational` in descending powers of $z$.
 */
function padEnd(b: number[], a: number[]): Rational {
  const n = Math.max(b.length, a.length)
  return { num: [...b, ...new Array(n - b.length).fill(0)], den: [...a, ...new Array(n - a.length).fill(0)] }
}

/**
 * A causal rational function of $z$ as ascending $z^{-1}$ coefficients: the numerator is padded in front. Throws
 * `DomainError` when the numerator has the higher degree (not causal).
 *
 * @param options The rational function.
 * @param options.num The numerator in descending powers of $z$.
 * @param options.den The denominator in descending powers of $z$.
 * @returns `b` and `a`, ascending in $z^{-1}$, of equal length (the denominator's, after stripping leading zeros).
 */
export function toAscending({ num, den }: Rational): { b: number[]; a: number[] } {
  const n = stripLeading(num)
  const d = stripLeading(den)
  if (n.length > d.length)
    throw new DomainError('systems', 'systems: a discrete system with more zeros than poles is not causal')
  return { b: [...new Array(d.length - n.length).fill(0), ...n], a: d }
}

/**
 * The real monic polynomial with the given roots (a complex128 vector, or a list); conjugate pairs required.
 *
 * @param rs The roots.
 * @param where The caller's name, for the `DomainError` thrown when a complex root lacks its conjugate.
 * @returns The coefficients in descending powers, leading 1.
 */
function realPoly(rs: Tensor | readonly ComplexNumber[], where: string): number[] {
  try {
    return toFlat(polyFromRoots(rs, { real: true }))
  } catch (e) {
    if (e instanceof DomainError)
      throw new DomainError(where, `${where}: complex zeros and poles must come in conjugate pairs`)
    throw e
  }
}

/**
 * Which input and output of a MIMO state-space system to take as SISO: `input` and `output` are 0-based indices,
 * each default 0. Other representations are SISO and ignore them.
 */
export type ChannelOptions = { input?: Size; output?: Size }

/**
 * The SISO rational function of a system (a channel of a MIMO state-space system), in descending powers. A discrete
 * transfer function is padded (`padEnd`), zpk is expanded with its gain on the numerator, sections are multiplied out.
 *
 * @param sys The system, in any representation.
 * @param options The channel of a state-space system.
 * @param options.input The input (column of $\Bmat$) to take.
 * @param options.output The output (row of $\Cmat$) to take.
 * @returns The numerator and denominator in descending powers of $s$ or $z$.
 */
export function rationalOf(sys: LtiSystem, { input = 0, output = 0 }: ChannelOptions = {}): Rational {
  const r = sys.repr
  if (r.form === 'tf') {
    const b = Array.from(toFlat(r.b))
    const a = Array.from(toFlat(r.a))
    return sys.domain === 'discrete' ? padEnd(b, a) : { num: b, den: a }
  }
  if (r.form === 'zpk') {
    const num = realPoly(r.zeros, 'systems').map((v) => v * r.gain)
    return { num, den: realPoly(r.poles, 'systems') }
  }
  if (r.form === 'sos') {
    const s = dense.data(r.sections)
    let num = [1]
    let den = [1]
    for (let k = 0; k < r.sections.shape[0]; k++) {
      num = mulCoefficients(num, Array.from(s.subarray(6 * k, 6 * k + 3)))
      den = mulCoefficients(den, Array.from(s.subarray(6 * k + 3, 6 * k + 6)))
    }
    return { num, den }
  }
  return stateSpaceRational(r, input, output)
}

/**
 * The characteristic polynomial $\det(x\Imat - \Mmat)$ of a square matrix, from its eigenvalues.
 *
 * @param m The matrix $\Mmat$, row-major, $n^2$ values.
 * @param n Its order (0 gives the polynomial 1).
 * @returns The $n + 1$ coefficients in descending powers, leading 1.
 */
function charPoly(m: dense.F64, n: Size): number[] {
  if (n === 0) return [1]
  return realPoly(eig(fromData(m, [n, n]), { vectors: false }).values, 'systems')
}

/**
 * $G(x) = \cvec(x\Imat - \Amat)^{-1}\bvec + d$ for one input and output: the denominator is $\det(x\Imat - \Amat)$ and
 * the numerator $\det(x\Imat - \Amat + \bvec\cvec) + (d - 1)\det(x\Imat - \Amat)$, the determinant identity scipy's
 * `ss2tf` uses. No pole–zero cancellation; coefficients below $10^{-13}$ of the largest are set to 0.
 *
 * @param r The state-space representation.
 * @param input The input, the column $\bvec$ of $\Bmat$. Throws `ShapeError` when out of range.
 * @param output The output, the row $\cvec$ of $\Cmat$ (and $d$ the matching entry of $\Dmat$).
 * @returns The numerator and denominator, each $n + 1$ coefficients in descending powers.
 */
function stateSpaceRational(r: StateSpaceForm, input: Size, output: Size): Rational {
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  const p = r.C.shape[0]
  if (input >= m || output >= p) throw new ShapeError('systems', `systems: no channel ${input} → ${output}`)
  const A = dense.data(r.A)
  const B = dense.data(r.B)
  const C = dense.data(r.C)
  const d = dense.data(r.D)[output * m + input]
  const Abc = Float64Array.from(A)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Abc[i * n + j] -= B[i * m + input] * C[output * n + j]
  const den = charPoly(A, n)
  const other = charPoly(Abc, n)
  const num = other.map((v, i) => v + (d - 1) * den[i])
  // Round away the ~1e-16 noise that the eigenvalue route leaves in structurally zero coefficients.
  const scale = Math.max(...den.map(Math.abs), ...num.map(Math.abs))
  const clean = (v: number) => (Math.abs(v) < 1e-13 * scale ? 0 : v)
  return { num: num.map(clean), den: den.map(clean) }
}

/**
 * The system with its representation replaced (domain, `dt` and delay kept).
 *
 * @param sys The system whose domain, `dt` and delay are kept.
 * @param repr The new representation (of the same system, for the result to be correct).
 * @returns A new system record.
 */
export const withRepr = <R extends Representation>(sys: LtiSystem, repr: R): LtiOf<R> => ({ ...sys, repr })

// ── Conversions ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The system as a transfer function. A MIMO state-space system gives the channel from `input` to `output` (default 0
 * to 0), through $\Cmat(x\Imat - \Amat)^{-1}\Bmat + \Dmat$; nothing cancels, so an uncontrollable or unobservable mode
 * stays in both polynomials. A system already in transfer-function form is returned as it is. Throws `DomainError` for
 * a discrete system with more zeros than poles.
 *
 * @param sys The system.
 * @param channel The input and output of a MIMO state-space system.
 * @returns The transfer function: descending powers of $s$ (leading zeros stripped), or ascending powers of $z^{-1}$.
 *
 * @example The mass–spring–damper $\ddot y + 3\dot y + 2y = u$ from state space
 * const sys = stateSpace({ A: [[0, 1], [-2, -3]], B: [0, 1], C: [1, 0] })
 * const tf = toTransferFunction(sys).repr
 * print('b =', tf.b)
 * print('a =', tf.a)
 */
export function toTransferFunction(sys: LtiSystem, channel: ChannelOptions = {}): LtiOf<TransferFunctionForm> {
  if (sys.repr.form === 'tf') return sys as LtiOf<TransferFunctionForm>
  const rational = rationalOf(sys, channel)
  if (sys.domain === 'continuous') {
    const num = stripLeading(rational.num)
    return withRepr(sys, { form: 'tf', b: vec(num), a: vec(stripLeading(rational.den)) })
  }
  const { b, a } = toAscending(rational)
  return withRepr(sys, { form: 'tf', b: vec(b), a: vec(a) })
}

/**
 * The system in zeros–poles–gain form: the roots of the numerator and denominator, and the ratio of their leading
 * coefficients as the gain (0, with no zeros, for a zero numerator). A system already in zpk form is returned as it
 * is.
 *
 * @param sys The system.
 * @param channel The input and output of a MIMO state-space system.
 * @returns The system in zeros–poles–gain form.
 *
 * @example $(s + 3)/((s + 1)(s + 2))$
 * const sys = transferFunction([1, 3], [1, 3, 2])
 * const zpk = toZerosPolesGain(sys).repr
 * print('zeros =', zpk.zeros)
 * print('poles =', zpk.poles)
 * print('gain =', zpk.gain)
 */
export function toZerosPolesGain(sys: LtiSystem, channel: ChannelOptions = {}): LtiOf<ZerosPolesGainForm> {
  if (sys.repr.form === 'zpk') return sys as LtiOf<ZerosPolesGainForm>
  const { num, den } = rationalOf(sys, channel)
  const n = stripLeading(num)
  const d = stripLeading(den)
  const gain = n.length === 1 && n[0] === 0 ? 0 : n[0] / d[0]
  const none = fromData(new Float64Array(0), [0], 'complex128')
  return withRepr(sys, { form: 'zpk', zeros: gain === 0 ? none : roots(n), poles: roots(d), gain })
}

/**
 * The system in state-space form. A transfer function (or zpk, sos) is realised in controllable canonical form, as
 * scipy's `tf2ss`: with the denominator made monic and the numerator padded to its length, $\Amat$ has first row
 * $-[a_1, \dots, a_n]$ and ones on the subdiagonal, $\Bmat = \evec_1$, $\Dmat = b_0$ and
 * $\Cmat = [b_1, \dots, b_n] - b_0 [a_1, \dots, a_n]$. Improper transfer functions cannot be realised (`DomainError`).
 * The delay is kept on the system but is not part of $\Amat$, $\Bmat$, $\Cmat$, $\Dmat$. A state-space system is
 * returned as it is.
 *
 * @param sys The system (SISO unless already in state space).
 * @returns The system in state-space form, one input and one output when realised here.
 *
 * @example The controllable canonical form of $1/(s^2 + 3s + 2)$
 * const ss = toStateSpace(transferFunction([1], [1, 3, 2])).repr
 * print('A =', ss.A)
 * print('B =', ss.B)
 * print('C =', ss.C)
 * print('D =', ss.D)
 */
export function toStateSpace(sys: LtiSystem): LtiOf<StateSpaceForm> {
  if (sys.repr.form === 'ss') return sys as LtiOf<StateSpaceForm>
  const rational = rationalOf(sys)
  const den = stripLeading(rational.den)
  let num = stripLeading(rational.num)
  const n = den.length - 1
  if (num.length - 1 > n) throw new DomainError('toStateSpace', 'toStateSpace: the transfer function is improper')
  num = [...new Array(n + 1 - num.length).fill(0), ...num]
  const a = den.map((v) => v / den[0])
  const b = num.map((v) => v / den[0])
  const A = new Float64Array(n * n)
  for (let j = 0; j < n; j++) A[j] = -a[j + 1]
  for (let i = 1; i < n; i++) A[i * n + i - 1] = 1
  const B = new Float64Array(n)
  if (n > 0) B[0] = 1
  const C = Float64Array.from({ length: n }, (_, j) => b[j + 1] - b[0] * a[j + 1])
  return withRepr(sys, {
    form: 'ss',
    A: fromData(A, [n, n]),
    B: fromData(B, [n, 1]),
    C: fromData(C, [1, n]),
    D: fromData(Float64Array.of(b[0]), [1, 1]),
  })
}

/** A complex number `{ re, im }`. */
type Complex = ComplexNumber
/**
 * The complex conjugate.
 *
 * @param z The number.
 * @returns $\bar z$.
 */
const conj = (z: Complex): Complex => ({ re: z.re, im: -z.im })
/**
 * The modulus.
 *
 * @param z The number.
 * @returns $\lvert z \rvert$.
 */
const modulus = (z: Complex) => Math.hypot(z.re, z.im)
/**
 * The distance between two complex numbers.
 *
 * @param u The first number.
 * @param v The second number.
 * @returns $\lvert u - v \rvert$.
 */
const distance = (u: Complex, v: Complex) => Math.hypot(u.re - v.re, u.im - v.im)
/**
 * Whether a complex number is real to a relative tolerance.
 *
 * @param z The number.
 * @param tol The tolerance on $\lvert \operatorname{Im} z \rvert$, relative to $\max(1, \lvert z \rvert)$.
 * @returns True when the imaginary part is within the tolerance.
 */
const isReal = (z: Complex, tol: number) => Math.abs(z.im) <= tol * Math.max(1, modulus(z))

/**
 * Splits roots into real ones and one representative ($\operatorname{Im} > 0$) of each conjugate pair. Roots within
 * a relative $10^{-9}$ of the real axis count as real (their imaginary part set to 0).
 *
 * @param rs The roots.
 * @param where The caller's name, for the `DomainError` thrown when the upper and lower roots do not pair up in number.
 * @returns `real`, the real roots, and `pairs`, the roots in the upper half-plane.
 */
function splitRoots(rs: Complex[], where: string): { real: Complex[]; pairs: Complex[] } {
  const tol = 1e-9
  const real = rs.filter((z) => isReal(z, tol)).map((z) => ({ re: z.re, im: 0 }))
  const upper = rs.filter((z) => !isReal(z, tol) && z.im > 0)
  const lower = rs.filter((z) => !isReal(z, tol) && z.im < 0)
  if (upper.length !== lower.length)
    throw new DomainError(where, `${where}: complex zeros and poles must come in conjugate pairs`)
  return { real, pairs: upper }
}

/**
 * A discrete system as second-order sections, as scipy's `zpk2sos` with `pairing='nearest'`: poles are taken in
 * pairs (conjugate pairs, then real poles two at a time in ascending order), the pair nearest the unit circle last,
 * and each is matched with the nearest remaining zeros; the gain goes into the first section. Sections with fewer
 * zeros than poles carry the difference as a delay ($b_0 = 0$), so the cascade equals the system exactly. Throws
 * `DomainError` for a continuous system or more zeros than poles; sections are returned as they are.
 *
 * @param sys The discrete system.
 * @returns The system in second-order-sections form, $\lceil n/2 \rceil$ sections for $n$ poles (one for none).
 *
 * @example Four real poles become two sections, the poles nearest the unit circle last
 * const sys = zerosPolesGain([-1, -1, -1, -1], [0.5, 0.6, 0.7, 0.8], 0.01, { dt: 1 })
 * print('sections =', toSecondOrderSections(sys).repr.sections)
 */
export function toSecondOrderSections(sys: LtiSystem): LtiOf<SecondOrderSectionsForm> {
  const where = 'toSecondOrderSections'
  if (sys.domain !== 'discrete') throw new DomainError(where, `${where}: second-order sections are discrete-time only`)
  if (sys.repr.form === 'sos') return sys as LtiOf<SecondOrderSectionsForm>
  const zpk = toZerosPolesGain(sys).repr
  const zeros = toComplexFlat(zpk.zeros)
  const poles = toComplexFlat(zpk.poles)
  if (zeros.length > poles.length) throw new DomainError(where, `${where}: more zeros than poles (not causal)`)
  const zs = splitRoots(zeros, where)
  const ps = splitRoots(poles, where)
  // Pole groups: each conjugate pair, then the real poles two by two (sorted so close ones are grouped).
  const groups: Complex[][] = ps.pairs.map((p) => [p, conj(p)])
  const reals = [...ps.real].sort((u, v) => u.re - v.re)
  for (let k = 0; k < reals.length; k += 2) groups.push(reals.slice(k, k + 2))
  // Process the group nearest the unit circle first (it becomes the last section).
  const toCircle = (g: Complex[]) => Math.min(...g.map((p) => Math.abs(1 - modulus(p))))
  groups.sort((u, v) => toCircle(u) - toCircle(v))
  let realZeros = [...zs.real]
  let pairZeros = [...zs.pairs]
  const sections: number[][] = []
  for (const group of groups) {
    const target = group[0]
    const assigned: Complex[] = []
    const nearestPair = () => {
      let best = -1
      pairZeros.forEach((z, k) => {
        if (best < 0 || distance(z, target) < distance(pairZeros[best], target)) best = k
      })
      return best
    }
    const nearestReal = () => {
      let best = -1
      realZeros.forEach((z, k) => {
        if (best < 0 || distance(z, target) < distance(realZeros[best], target)) best = k
      })
      return best
    }
    if (group.length === 2 && pairZeros.length > 0 && (realZeros.length < 2 || !isReal(target, 1e-9))) {
      const k = nearestPair()
      assigned.push(pairZeros[k], conj(pairZeros[k]))
      pairZeros = pairZeros.filter((_, j) => j !== k)
    }
    while (assigned.length < group.length && realZeros.length > 0) {
      const k = nearestReal()
      assigned.push(realZeros[k])
      realZeros = realZeros.filter((_, j) => j !== k)
    }
    if (assigned.length < group.length && pairZeros.length > 0 && group.length === 2 && assigned.length === 0) {
      const k = nearestPair()
      assigned.push(pairZeros[k], conj(pairZeros[k]))
      pairZeros = pairZeros.filter((_, j) => j !== k)
    }
    const a = realPoly(group, where)
    const b = [...new Array(group.length - assigned.length).fill(0), ...realPoly(assigned, where)]
    const pad = (c: number[]) => [...c, ...new Array(3 - c.length).fill(0)]
    sections.push([...pad(b), ...pad(a)])
  }
  if (realZeros.length + pairZeros.length > 0)
    throw new DomainError(where, `${where}: zeros left after pairing (more zeros than poles)`)
  if (sections.length === 0) sections.push([1, 0, 0, 1, 0, 0])
  sections.reverse()
  for (let j = 0; j < 3; j++) sections[0][j] *= zpk.gain
  return withRepr(sys, { form: 'sos', sections: fromData(Float64Array.from(sections.flat()), [sections.length, 6]) })
}

/**
 * The system in representation `form`, by `toTransferFunction`, `toZerosPolesGain`, `toStateSpace` or
 * `toSecondOrderSections` (channel 0 to 0 of a MIMO system).
 *
 * @param sys The system.
 * @param form The representation wanted: `'tf'`, `'zpk'`, `'ss'` or `'sos'` (discrete systems only).
 * @returns The system in that representation.
 *
 * @example One system in two other forms
 * const sys = transferFunction([1], [1, 3, 2])
 * print('zpk =', convert(sys, 'zpk').repr)
 * print('ss A =', convert(sys, 'ss').repr.A)
 */
export function convert(sys: LtiSystem, form: Representation['form']): LtiSystem {
  if (form === 'tf') return toTransferFunction(sys)
  if (form === 'zpk') return toZerosPolesGain(sys)
  if (form === 'ss') return toStateSpace(sys)
  return toSecondOrderSections(sys)
}

// ── Poles, zeros, stability ──────────────────────────────────────────────────────────────────────────────────────────

/**
 * The poles of a system as a complex128 vector: eigenvalues of $\Amat$ (ss), or roots of the denominator.
 *
 * @param sys The system.
 * @returns The poles, complex128 of shape $[n]$.
 *
 * @example The lightly damped pair of $1/(s^2 + 2s + 5)$
 * print('poles =', poles(transferFunction([1], [1, 2, 5])))
 */
export function poles(sys: LtiSystem): Tensor {
  const r = sys.repr
  if (r.form === 'zpk') return r.poles
  if (r.form === 'ss') return eig(r.A, { vectors: false }).values
  return toZerosPolesGain(sys).repr.poles
}

/**
 * The zeros of a SISO system (or of a channel of a MIMO state-space system) as a complex128 vector; named apart from
 * the tensor constructor `zeros`.
 *
 * @param sys The system.
 * @param channel The input and output of a MIMO state-space system.
 * @returns The zeros, complex128 (empty when the numerator is constant).
 *
 * @example The zero of $(s + 3)/(s^2 + 3s + 2)$
 * print('zeros =', systemZeros(transferFunction([1, 3], [1, 3, 2])))
 */
export function systemZeros(sys: LtiSystem, channel: ChannelOptions = {}): Tensor {
  return toZerosPolesGain(sys, channel).repr.zeros
}

/** The stability verdict of a system or of a bare state matrix. */
export type Stability = {
  /** The poles, complex128 of shape $[k]$. */
  poles: Tensor
  /**
   * Continuous: every $\operatorname{Re} \lambda < 0$. Discrete: every $\lvert \lambda \rvert < 1$. Poles within
   * $10^{-12}$ of the boundary count as not stable (marginal).
   */
  stable: boolean
  /**
   * The largest $\operatorname{Re} \lambda$ (continuous) or $\lvert \lambda \rvert$ (discrete): the margin to the
   * boundary 0 or 1. With no poles, $-\infty$ (continuous) or 0 (discrete).
   */
  abscissa: Scalar
}

/**
 * Whether a system is asymptotically stable: poles in the open left half-plane (continuous) or inside the unit
 * circle (discrete). Pass a square matrix with `{ discrete }` to test a bare $\Amat$ by its eigenvalues.
 *
 * @param sys The system, or a square state matrix.
 * @param options The domain of the test.
 * @param options.discrete Whether to test against the unit circle; default the system's own domain (continuous for
 *   a bare matrix).
 * @returns The poles, the verdict and the abscissa.
 *
 * @example A lag, an integrator (marginal, so not stable) and a discrete state matrix
 * print('lag =', stability(transferFunction([1], [1, 1])))
 * print('integrator =', stability(transferFunction([1], [1, 0])).stable)
 * print('discrete A =', stability([[0.5, 1], [0, 0.9]], { discrete: true }))
 */
export function stability(sys: LtiSystem | MatrixLike, { discrete }: { discrete?: boolean } = {}): Stability {
  const isSystem = (sys as { kind?: unknown }).kind === 'lti'
  let p: Tensor
  if (isSystem) p = poles(sys as LtiSystem)
  else p = eig(sys as MatrixLike, { vectors: false }).values
  const disc = discrete ?? (isSystem ? (sys as LtiSystem).domain === 'discrete' : false)
  const xs = toComplexFlat(p)
  const abscissa = xs.length ? Math.max(...xs.map((z) => (disc ? modulus(z) : z.re))) : disc ? 0 : -Infinity
  return { poles: p, stable: abscissa < (disc ? 1 : 0) - 1e-12, abscissa }
}

/**
 * The state, input and output dimensions of a system (a SISO transfer function has one input and one output, and as
 * many states as its denominator's degree).
 *
 * @param sys The system.
 * @returns The numbers of states $n$, inputs $m$ and outputs $p$.
 *
 * @example A two-input, one-output system, and a second-order transfer function
 * const sys = stateSpace({ A: [[-1, 0], [0, -2]], B: [[1, 0], [0, 1]], C: [[1, 1]] })
 * print('dimensions =', dimensions(sys))
 * print('tf dimensions =', dimensions(transferFunction([1], [1, 3, 2])))
 */
export function dimensions(sys: LtiSystem): { states: Size; inputs: Size; outputs: Size } {
  const r = toStateSpace(sys).repr
  return { states: r.A.shape[0], inputs: r.B.shape[1], outputs: r.C.shape[0] }
}
