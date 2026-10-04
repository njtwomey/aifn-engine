/**
 * Complex numbers as a dtype (design K §8.1): complex128, stored interleaved (re, im) in a Float64Array, so every
 * structural primitive works on it unchanged. This file holds the primitives that exist because of complex values:
 * `complex(re, im)`, `conj`, `realPart`, `imagPart` (zero-copy float64 views), `angle`, the modulus behind `abs`, and
 * `expj(θ) = e^{iθ}`. The arithmetic primitives (add, mul, exp, log, sqrt, pow, …) carry their complex rules in
 * elementwise.ts.
 *
 * **Derivatives: complex values are pairs of reals.** The cotangent of z = x + iy is x̄ + iȳ, the gradient of the
 * real loss with respect to (x, y) packed as one complex number, so gradient descent is z − η·z̄ exactly as for reals.
 * A holomorphic f with derivative f′ then has vjp ḡ·conj(f′(z)) and jvp ż·f′(z) (`elementwise` with `holomorphic`).
 * A real-valued function of z (|z|, arg z) has the ℝ² gradient w = ∂f/∂x + i ∂f/∂y: vjp ḡ·w, jvp Re(conj(w)·ż).
 * `conj`, `realPart`, `imagPart` and `complex` are ℝ-linear, and their transposes are their ℝ² adjoints. The Wirtinger
 * derivative ∂L/∂z̄ is ½(x̄ + iȳ).
 */

import { DTypeError } from 'aifn-compute/foundation/errors'
import { complexPartView, type DType, type Tensor } from './core'
import { astype, scalar, zeros } from './create'
import { resultType } from './dtype'
import { complexKernel } from './kernels'
import { definePrimitive, elementwise, sumLike, type Binary, type Op, type Raw, type Unary } from './primitive'
import { avalOf, type Aval, type Value } from './trace'
import { add, div, equalTo, mul, square, where } from './elementwise'

/** The imaginary unit i as a rank-0 complex128 tensor (a constant for rules). */
export const I: Tensor = scalar({ re: 0, im: 1 })

const isComplexValue = (x: Value): boolean => avalOf(x).dtype === 'complex128'

/** The abstract value of a unary elementwise result with dtype `dtype`. */
const unaryShape = (dtype: (x: DType) => DType) => (avals: readonly Aval[]) => ({
  shape: avals[0].shape,
  dtype: dtype(avals[0].dtype),
  number: avals[0].number,
})

// ── complex(re, im) and expj ─────────────────────────────────────────────────────────────────────────────────────────

/**
 * re + i·im, broadcast: a complex128 tensor from real and imaginary parts (numbers give a rank-0 tensor). Linear, and
 * holomorphic in each argument (∂/∂re = 1, ∂/∂im = i), so its vjp is (Re ḡ, Im ḡ) for real parts.
 */
export const complex: Binary = elementwise({
  id: 'foundation/tensor/complex',
  f: (a, b) => a + b, // never used: the result is always complex, so the complex rule runs
  complex: (o, z) => {
    o[0] = z[0] - z[3]
    o[1] = z[1] + z[2]
  },
  holomorphic: true,
  derivative: [() => 1, () => I],
  dtype: 'complex',
  doc: { summary: 'The complex number re + i·im.', formula: 'z = x + i y' },
})

/** e^{iθ} = cos θ + i sin θ (for complex θ = x + iy, e^{−y}(cos x + i sin x)). Holomorphic: d/dθ = i·e^{iθ}. */
export const expj: Unary = elementwise({
  id: 'foundation/tensor/expj',
  f: (x) => x, // never used (complex result)
  complex: (o, z) => {
    const m = Math.exp(-z[1])
    o[0] = m * Math.cos(z[0])
    o[1] = m * Math.sin(z[0])
  },
  holomorphic: true,
  derivative: [(_x, y) => mul(y, I)],
  dtype: 'complex',
  doc: { summary: 'The unit phasor e^{iθ}.', formula: 'e^{i\\theta} = \\cos\\theta + i\\sin\\theta' },
})

// ── conj, realPart, imagPart ─────────────────────────────────────────────────────────────────────────────────────────

const conjOp: Op<undefined> = definePrimitive<undefined>({
  id: 'foundation/tensor/conj',
  arity: 1,
  kind: 'elementwise',
  dtype: 'same',
  impl: ([z]) => {
    if (typeof z === 'number') return z
    if (z.dtype === 'complex128') return complexKernel([z], conjugate)
    return z.dtype === 'bool' ? astype(z, 'int32') : z
  },
  linear: 'linear',
  // (x, y) ↦ (x, −y) is its own adjoint.
  transpose: (ct) => conj(ct),
  shape: unaryShape((d) => resultType('same', d)),
  doc: { summary: 'The complex conjugate x − iy.' },
  test: { complex: true, secondOrder: true },
})

function conjugate(o: Float64Array, z: Float64Array): void {
  o[0] = z[0]
  o[1] = -z[1]
}

/** The complex conjugate x − iy; the identity on real values (returned as they are, recording nothing). */
export function conj<X extends Value>(z: X): X {
  return (isComplexValue(z) ? conjOp([z], undefined) : z) as X
}

/** The real or imaginary part of a raw value: a zero-copy float64 view of a complex tensor. */
function part(z: Raw, which: 0 | 1): Raw {
  if (typeof z === 'number') return which === 0 ? z : 0
  if (z.dtype === 'complex128') return complexPartView(z, which)
  const dtype = resultType('real', z.dtype)
  return which === 1 ? zeros(z.shape, dtype) : dtype === z.dtype ? z : astype(z, dtype)
}

const realPartOp: Op<undefined> = definePrimitive<undefined>({
  id: 'foundation/tensor/realPart',
  arity: 1,
  kind: 'elementwise',
  dtype: 'real',
  impl: ([z]) => part(z, 0),
  linear: 'linear',
  // Re: (x, y) ↦ x; its adjoint puts the cotangent in the real part (fitted back to a real input as it is).
  transpose: (ct, [z]) => (isComplexValue(z) ? complex(ct, 0) : ct),
  shape: unaryShape((d) => resultType('real', d)),
  doc: { summary: 'The real part x of z = x + iy, as a zero-copy float64 view.' },
  test: { complex: true, secondOrder: true },
})

const imagPartOp: Op<undefined> = definePrimitive<undefined>({
  id: 'foundation/tensor/imagPart',
  arity: 1,
  kind: 'elementwise',
  dtype: 'real',
  impl: ([z]) => part(z, 1),
  linear: 'linear',
  // Im: (x, y) ↦ y; its adjoint puts the cotangent in the imaginary part (a real input has no imaginary part).
  transpose: (ct, [z]) => (isComplexValue(z) ? complex(0, ct) : null),
  shape: unaryShape((d) => resultType('real', d)),
  doc: { summary: 'The imaginary part y of z = x + iy, as a zero-copy float64 view.' },
  test: { complex: true, secondOrder: true },
})

/**
 * The real part x of z = x + iy: a zero-copy float64 view of a complex tensor's storage (strides doubled, offset 2o).
 * Linear. A real value is returned as it is. (`real` is taken by `aifn-compute/foundation/space`.)
 */
export function realPart<X extends Value>(z: X): X {
  return (isComplexValue(z) ? realPartOp([z], undefined) : z) as X
}

/** The imaginary part y of z = x + iy: a zero-copy float64 view (offset 2o + 1). Linear. Zero for a real value. */
export function imagPart<X extends Value>(z: X): X {
  if (isComplexValue(z)) return imagPartOp([z], undefined) as X
  const aval = avalOf(z)
  return (aval.number ? 0 : zeros(aval.shape)) as X
}

// ── Real-valued functions of a complex argument: |z| and arg z ───────────────────────────────────────────────────────

/**
 * A real-valued elementwise primitive of one complex argument, with its ℝ² gradient w(z, y) = ∂f/∂x + i ∂f/∂y
 * written with primitives: vjp ḡ·w (the real part for a real argument), jvp Re(conj(w)·ż).
 */
function realValued(
  id: string,
  f: (re: number, im: number) => number,
  gradient: (z: Value, y: Value) => Value,
  summary: string,
): Op<undefined> {
  const rule = (o: Float64Array, z: Float64Array) => {
    o[0] = f(z[0], z[1])
  }
  return definePrimitive<undefined>({
    id,
    arity: 1,
    kind: 'elementwise',
    dtype: 'realFloat',
    impl: ([z]) => {
      if (typeof z === 'number') return f(z, 0)
      const y = complexKernel([z], rule, true)
      return z.dtype === 'float32' ? astype(y, 'float32') : y
    },
    vjp: (g, [z], y) => [sumLike(mul(g, gradient(z, y)), z)],
    jvp: ([t], [z], y) => (t === null ? null : realPart(mul(conj(gradient(z, y)), t))),
    shape: unaryShape((d) => resultType('realFloat', d)),
    doc: { summary },
    test: { secondOrder: true, complex: true, domain: { lo: 0.5, hi: 2 } },
  })
}

/** |z|² = x² + y², real. */
const abs2 = (z: Value): Value => add(square(realPart(z)), square(imagPart(z)))

const modulusOp = realValued(
  'foundation/tensor/complexAbs',
  Math.hypot,
  // ∇|z| = z/|z|, taken as 0 at z = 0 (the minimum-norm subgradient).
  (z, y) => div(z, where(equalTo(y, 0), 1, y)),
  'The modulus |z| = √(x² + y²) of a complex number.',
)

/** |z| of a complex value (float64); `abs` dispatches here for complex128. */
export function complexAbs<X extends Value>(z: X): X {
  return modulusOp([z], undefined) as X
}

const angleOp = realValued(
  'foundation/tensor/angle',
  (re, im) => Math.atan2(im, re),
  // ∇ arg z = (−y, x)/|z|² = i·z/|z|², taken as 0 at z = 0.
  (z) => {
    const r2 = abs2(z)
    return div(mul(I, z), where(equalTo(r2, 0), 1, r2))
  },
  'The argument (phase) arg z = atan2(y, x) in (−π, π].',
)

/** The argument (phase) of z = x + iy, atan2(y, x) in (−π, π], as float64 (π for negative reals, 0 otherwise). */
export function angle<X extends Value>(z: X): X {
  return angleOp([z], undefined) as X
}

/** Throw a `DTypeError` naming `op` when a value is complex: for operations that need an ordering. */
export function refuseComplex(op: string, x: Value, why = 'needs an ordering, which complex values lack'): void {
  if (isComplexValue(x)) throw new DTypeError(op, `${op}: ${why} (got complex128)`, ['complex128'])
}
