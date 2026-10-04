/**
 * The one linear time-invariant system type, `LtiSystem` (after `scipy.signal`'s `lti`/`dlti` and python-control's
 * `tf`, `zpk`, `ss`): constructors for its four representations, exact conversions between them, and poles, zeros and
 * stability. Continuous systems act on s; discrete ones on z with sampling interval `dt`.
 *
 * Conventions (as scipy): a continuous transfer function lists b and a in descending powers of s; a discrete one in
 * ascending powers of z⁻¹, H(z) = (b₀ + b₁z⁻¹ + …)/(a₀ + a₁z⁻¹ + …). Zeros and poles are complex128 vectors.
 * Second-order sections are discrete only, [k, 6] rows b₀ b₁ b₂ a₀ a₁ a₂.
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

/** The transfer-function representation, `{ form: 'tf', b, a }`. */
export type TransferFunctionForm = Extract<Representation, { form: 'tf' }>
/** The zeros–poles–gain representation, `{ form: 'zpk', zeros, poles, gain }`. */
export type ZerosPolesGainForm = Extract<Representation, { form: 'zpk' }>
/** The state-space representation, `{ form: 'ss', A, B, C, D }`. */
export type StateSpaceForm = Extract<Representation, { form: 'ss' }>
/** The second-order-sections representation, `{ form: 'sos', sections }`. */
export type SecondOrderSectionsForm = Extract<Representation, { form: 'sos' }>

/** An `LtiSystem` known to hold representation R, so `sys.repr.b` (etc.) type-checks. */
export type LtiOf<R extends Representation> = LtiSystem & { readonly repr: R }

/** Options shared by the constructors. */
export type SystemOptions = {
  /** Sampling interval of a discrete system; omitted or null for continuous time. */
  dt?: Scalar | null
  /** Input delay: seconds for a continuous system, samples for a discrete one. Default 0. */
  delay?: Scalar
}

function make<R extends Representation>(repr: R, { dt = null, delay = 0 }: SystemOptions, where: string): LtiOf<R> {
  if (dt !== null && !(dt > 0)) throw new DomainError(where, `${where}: dt must be positive (or null for continuous)`)
  if (!(delay >= 0)) throw new DomainError(where, `${where}: the delay must be non-negative`)
  return { kind: 'lti', domain: dt === null ? 'continuous' : 'discrete', dt, delay, repr }
}

const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

/** The coefficients without leading zeros (keeping one coefficient). */
export function stripLeading(c: number[]): number[] {
  let k = 0
  while (k < c.length - 1 && c[k] === 0) k++
  return c.slice(k)
}

// ── Constructors ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A SISO transfer function b/a. Continuous (no `dt`): coefficients in descending powers of s, leading zeros stripped;
 * improper functions are allowed for frequency responses but cannot be realised in state space. Discrete (`dt`
 * given): coefficients in ascending powers of z⁻¹, as `scipy.signal.lfilter` and `freqz` read them.
 *
 * @example transferFunction([1], [1, 2, 1]) // 1/(s + 1)²
 * @example transferFunction([0.5, 0.5], [1], { dt: 1 }) // the two-tap moving average
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
 * A SISO system from its zeros, poles and gain, H = k Π(x − zᵢ)/Π(x − pⱼ) with x = s or z. Zeros and poles are
 * `ComplexLike` (real values, `{ re, im }` lists, or real or complex128 tensors), stored as complex128 vectors; complex
 * ones must come in conjugate pairs for the system to be real.
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

/** The arguments of `stateSpace`: `B` may be a vector (one input), `C` defaults to I (the state is the output). */
export type StateSpaceInput = {
  A: MatrixLike
  B: MatrixLike | VectorLike
  C?: MatrixLike | VectorLike
  D?: MatrixLike | VectorLike | Scalar
} & SystemOptions

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
 * A state-space system x′ = Ax + Bu, y = Cx + Du (continuous) or x_{k+1} = Ax_k + Bu_k, y_k = Cx_k + Du_k (with
 * `dt`), with shapes checked. A vector `B` is one input (a column); a vector `C` is one output (a row); `C` defaults to
 * the identity and `D` to zeros.
 *
 * @example stateSpace({ A: [[0, 1], [0, 0]], B: [0, 1], C: [1, 0] }) // the double integrator, position measured
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
 * A discrete system as a cascade of second-order sections, `sections` [k, 6] with rows b₀ b₁ b₂ a₀ a₁ a₂ (each
 * section (b₀ + b₁z⁻¹ + b₂z⁻²)/(a₀ + a₁z⁻¹ + a₂z⁻²)), as `scipy.signal.sosfilt` reads them. `dt` defaults to 1.
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

/** A SISO rational function num(x)/den(x), coefficients in descending powers of x (s, or z). */
export type Rational = { num: number[]; den: number[] }

/** The product of two real coefficient lists (`aifn-compute/numerics/polynomial`'s `polyMul`, as plain numbers). */
export const mulCoefficients = (a: readonly number[], b: readonly number[]): number[] =>
  toFlat(polyMul(Float64Array.from(a), Float64Array.from(b)) as Tensor)

/** Discrete ascending z⁻¹ coefficients padded to equal length are descending powers of z. */
function padEnd(b: number[], a: number[]): Rational {
  const n = Math.max(b.length, a.length)
  return { num: [...b, ...new Array(n - b.length).fill(0)], den: [...a, ...new Array(n - a.length).fill(0)] }
}

/** A causal rational function of z as ascending z⁻¹ coefficients: the numerator is padded in front. */
export function toAscending({ num, den }: Rational): { b: number[]; a: number[] } {
  const n = stripLeading(num)
  const d = stripLeading(den)
  if (n.length > d.length)
    throw new DomainError('systems', 'systems: a discrete system with more zeros than poles is not causal')
  return { b: [...new Array(d.length - n.length).fill(0), ...n], a: d }
}

/** The real monic polynomial with the given roots (a complex128 vector, or a list); conjugate pairs required. */
function realPoly(rs: Tensor | readonly ComplexNumber[], where: string): number[] {
  try {
    return toFlat(polyFromRoots(rs, { real: true }))
  } catch (e) {
    if (e instanceof DomainError)
      throw new DomainError(where, `${where}: complex zeros and poles must come in conjugate pairs`)
    throw e
  }
}

/** Which input and output of a MIMO state-space system to take as SISO. */
export type ChannelOptions = { input?: Size; output?: Size }

/** The SISO rational function of a system (a channel of a MIMO state-space system), in descending powers. */
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

/** The characteristic polynomial det(xI − M) of a square matrix, from its eigenvalues. */
function charPoly(m: dense.F64, n: Size): number[] {
  if (n === 0) return [1]
  return realPoly(eig(fromData(m, [n, n]), { vectors: false }).values, 'systems')
}

/**
 * G(x) = C(xI − A)⁻¹B + D for one input and output: the denominator is det(xI − A) and the numerator
 * det(xI − A + bc) + (d − 1)det(xI − A), the determinant identity scipy's `ss2tf` uses. No pole–zero cancellation.
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

/** The system with its representation replaced (domain, dt and delay kept). */
export const withRepr = <R extends Representation>(sys: LtiSystem, repr: R): LtiOf<R> => ({ ...sys, repr })

// ── Conversions ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The system as a transfer function. A MIMO state-space system gives the channel `input` → `output` (default 0 → 0),
 * through C(xI − A)⁻¹B + D; nothing cancels, so an uncontrollable or unobservable mode stays in both polynomials.
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

/** The system in zeros–poles–gain form (roots of the numerator and denominator; a SISO channel of a MIMO system). */
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
 * scipy's `tf2ss`: with the denominator made monic, A has first row −[a₁ … aₙ] and ones on the subdiagonal, B = e₁,
 * D = b₀ and C = [b₁ … bₙ] − b₀[a₁ … aₙ]. Improper transfer functions cannot be realised. The delay is kept on the
 * system but is not part of A, B, C, D.
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

type Complex = ComplexNumber
const conj = (z: Complex): Complex => ({ re: z.re, im: -z.im })
const modulus = (z: Complex) => Math.hypot(z.re, z.im)
const distance = (u: Complex, v: Complex) => Math.hypot(u.re - v.re, u.im - v.im)
const isReal = (z: Complex, tol: number) => Math.abs(z.im) <= tol * Math.max(1, modulus(z))

/** Splits roots into real ones and one representative (im > 0) of each conjugate pair. */
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
 * pairs (conjugate pairs, then real poles two at a time), the pair nearest the unit circle last, and each is matched
 * with the nearest remaining zeros; the gain goes into the first section. Sections with fewer zeros than poles carry
 * the difference as a delay (b₀ = 0), so the cascade equals the system exactly.
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

/** The system in representation `form`. */
export function convert(sys: LtiSystem, form: Representation['form']): LtiSystem {
  if (form === 'tf') return toTransferFunction(sys)
  if (form === 'zpk') return toZerosPolesGain(sys)
  if (form === 'ss') return toStateSpace(sys)
  return toSecondOrderSections(sys)
}

// ── Poles, zeros, stability ──────────────────────────────────────────────────────────────────────────────────────────

/** The poles of a system as a complex128 vector: eigenvalues of A (ss), or roots of the denominator. */
export function poles(sys: LtiSystem): Tensor {
  const r = sys.repr
  if (r.form === 'zpk') return r.poles
  if (r.form === 'ss') return eig(r.A, { vectors: false }).values
  return toZerosPolesGain(sys).repr.poles
}

/**
 * The zeros of a SISO system (or of a channel of a MIMO state-space system) as a complex128 vector; named apart from
 * the tensor constructor `zeros`.
 */
export function systemZeros(sys: LtiSystem, channel: ChannelOptions = {}): Tensor {
  return toZerosPolesGain(sys, channel).repr.zeros
}

/** The stability verdict of a system or of a bare state matrix. */
export type Stability = {
  /** The poles, complex128 [k]. */
  poles: Tensor
  /**
   * Continuous: every Re λ < 0. Discrete: every |λ| < 1. Poles within 1e-12 of the boundary count as not stable
   * (marginal).
   */
  stable: boolean
  /** The largest Re λ (continuous) or |λ| (discrete): the margin to the boundary 0 or 1. */
  abscissa: Scalar
}

/**
 * Whether a system is asymptotically stable: poles in the open left half-plane (continuous) or inside the unit
 * circle (discrete). Pass a square matrix with `{ discrete }` to test a bare A.
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

/** The state, input and output dimensions of a system (a SISO transfer function has one input and one output). */
export function dimensions(sys: LtiSystem): { states: Size; inputs: Size; outputs: Size } {
  const r = toStateSpace(sys).repr
  return { states: r.A.shape[0], inputs: r.B.shape[1], outputs: r.C.shape[0] }
}
