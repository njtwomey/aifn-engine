/**
 * Complex numbers as a dtype (design K §8.1): complex128, stored interleaved (re, im) in a Float64Array, so every
 * structural primitive works on it unchanged. This file holds the primitives that exist because of complex values:
 * `complex(re, im)`, `conj`, `realPart`, `imagPart` (zero-copy float64 views), `angle`, the modulus behind `abs`, and
 * `expj`, $\theta \mapsto e^{i\theta}$. The arithmetic primitives (`add`, `mul`, `exp`, `log`, `sqrt`, `pow`, …) carry
 * their complex rules in elementwise.ts.
 *
 * **Derivatives: complex values are pairs of reals.** The cotangent of $z = x + iy$ is $\bar{x} + i\bar{y}$, the
 * gradient of the real loss with respect to $(x, y)$ packed as one complex number, so gradient descent is
 * $z - \eta\bar{z}$ ($\bar{z}$ the cotangent) exactly as for reals. A holomorphic $f$ with derivative $f'$ then has vjp
 * $\bar{g} \operatorname{conj}(f'(z))$ and jvp $\dot{z} f'(z)$ (`elementwise` with `holomorphic`). A real-valued
 * function of $z$ ($\lvert z \rvert$, $\arg z$) has the $\reals^2$ gradient
 * $w = \partial f/\partial x + i\, \partial f/\partial y$: vjp $\bar{g} w$, jvp
 * $\operatorname{Re}(\operatorname{conj}(w) \dot{z})$. `conj`, `realPart`, `imagPart` and `complex` are
 * $\reals$-linear, and their transposes are their $\reals^2$ adjoints. The Wirtinger derivative
 * $\partial L/\partial \bar{z}$ is $\tfrac{1}{2}(\bar{x} + i\bar{y})$.
 */

import { DTypeError } from 'aifn-compute/foundation/errors'
import { complexPartView, type DType, type Tensor } from './core'
import { astype, scalar, zeros } from './create'
import { resultType } from './dtype'
import { complexKernel } from './kernels'
import { definePrimitive, elementwise, sumLike, type Binary, type Op, type Raw, type Unary } from './primitive'
import { avalOf, type Aval, type Value } from './trace'
import { add, div, equalTo, mul, square, where } from './elementwise'

/** The imaginary unit $i$ as a rank-0 complex128 tensor (a constant for rules). */
export const I: Tensor = scalar({ re: 0, im: 1 })

/**
 * Whether a value has dtype complex128.
 *
 * @param x The value: a number (always real), a tensor or a traced value.
 * @returns True for a complex128 tensor or traced value.
 */
const isComplexValue = (x: Value): boolean => avalOf(x).dtype === 'complex128'

/**
 * The shape rule of a unary elementwise primitive whose result dtype follows from its argument's by `dtype`.
 *
 * @param dtype The map from the argument's dtype to the result's.
 * @returns The shape rule: the argument's shape and kind (number or tensor), with the mapped dtype.
 */
const unaryShape = (dtype: (x: DType) => DType) => (avals: readonly Aval[]) => ({
  shape: avals[0].shape,
  dtype: dtype(avals[0].dtype),
  number: avals[0].number,
})

// ── complex(re, im) and expj ─────────────────────────────────────────────────────────────────────────────────────────

/**
 * $\mathrm{re} + i \cdot \mathrm{im}$, elementwise with broadcasting: a complex128 tensor from real and imaginary parts
 * (numbers give a rank-0 tensor). Linear, and holomorphic in each argument (derivative 1 in $\mathrm{re}$ and $i$ in
 * $\mathrm{im}$), so its vjp is $(\operatorname{Re} \bar{g}, \operatorname{Im} \bar{g})$ for real parts.
 *
 * @example A complex number, and a vector sharing one imaginary part
 * print('3 + 4i =', complexItem(complex(3, 4)))
 * print('[1, 2] + i as (re, im) pairs:', complex(tensor([1, 2]), 1))
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

/**
 * The unit phasor $e^{i\theta} = \cos\theta + i \sin\theta$, elementwise (for complex $\theta = x + iy$,
 * $e^{-y}(\cos x + i \sin x)$). Holomorphic, with derivative $i e^{i\theta}$.
 *
 * @example Half turns around the unit circle
 * print('expj([0, π/2, π]) as (re, im) pairs:', expj(tensor([0, Math.PI / 2, Math.PI])))
 */
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

/**
 * The conjugation primitive behind `conj`: $(x, y) \mapsto (x, -y)$, linear and its own adjoint. Real values pass
 * through (bool becomes int32).
 */
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

/**
 * The complex rule of `conj`: $(x, y) \mapsto (x, -y)$.
 *
 * @param o Where the conjugate is written: its real part to `o[0]` and its imaginary part to `o[1]`.
 * @param z The argument as its real part in `z[0]` and its imaginary part in `z[1]`.
 */
function conjugate(o: Float64Array, z: Float64Array): void {
  o[0] = z[0]
  o[1] = -z[1]
}

/**
 * The complex conjugate $\bar{z} = x - iy$, elementwise; the identity on real values (returned as they are, recording
 * nothing). Linear over the reals, so differentiable.
 *
 * @param z The value to conjugate: complex, or real (returned unchanged).
 * @returns $\bar{z}$, of the kind, shape and dtype of `z`.
 *
 * @example The conjugate, and $z\bar{z} = \lvert z \rvert^2$
 * const z = complex(3, 4)
 * print('conj(3 + 4i) =', complexItem(conj(z)))
 * print('z conj(z) =', complexItem(mul(z, conj(z))))
 */
export function conj<X extends Value>(z: X): X {
  return (isComplexValue(z) ? conjOp([z], undefined) : z) as X
}

/**
 * The real or imaginary part of a raw value: a zero-copy float64 view of a complex tensor. A real tensor is its own
 * real part (bool converted to int32) and has an imaginary part of zeros.
 *
 * @param z The raw value: a number or a tensor of any dtype.
 * @param which 0 for the real part, 1 for the imaginary part.
 * @returns The part: a number for a number, otherwise a tensor of the shape of `z`.
 */
function part(z: Raw, which: 0 | 1): Raw {
  if (typeof z === 'number') return which === 0 ? z : 0
  if (z.dtype === 'complex128') return complexPartView(z, which)
  const dtype = resultType('real', z.dtype)
  return which === 1 ? zeros(z.shape, dtype) : dtype === z.dtype ? z : astype(z, dtype)
}

/** The primitive behind `realPart`: $(x, y) \mapsto x$, whose adjoint puts the cotangent in the real part. */
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

/** The primitive behind `imagPart`: $(x, y) \mapsto y$, whose adjoint puts the cotangent in the imaginary part. */
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
 * The real part $x$ of $z = x + iy$, elementwise: a zero-copy float64 view of a complex tensor's storage (strides
 * doubled, offset $2o$). Linear. A real value is returned as it is. (`real` is taken by
 * `aifn-compute/foundation/space`.)
 *
 * @param z The value: complex, or real (returned unchanged).
 * @returns The real parts, of the shape of `z`: float64 for a complex `z`.
 *
 * @example The two parts of a complex vector
 * const z = complex(tensor([1, 2]), tensor([3, 4]))
 * print('realPart =', realPart(z))
 * print('imagPart =', imagPart(z))
 */
export function realPart<X extends Value>(z: X): X {
  return (isComplexValue(z) ? realPartOp([z], undefined) : z) as X
}

/**
 * The imaginary part $y$ of $z = x + iy$, elementwise: a zero-copy float64 view (offset $2o + 1$). Linear. Zero for a
 * real value (0 for a number, float64 zeros of its shape otherwise), which is a constant.
 *
 * @param z The value: complex, or real.
 * @returns The imaginary parts, float64, of the shape of `z`.
 *
 * @example A complex vector, and a real one
 * print('imagPart(1 + 2i, 3 - 4i) =', imagPart(complex(tensor([1, 3]), tensor([2, -4]))))
 * print('imagPart([5, 6]) =', imagPart(tensor([5, 6])))
 */
export function imagPart<X extends Value>(z: X): X {
  if (isComplexValue(z)) return imagPartOp([z], undefined) as X
  const aval = avalOf(z)
  return (aval.number ? 0 : zeros(aval.shape)) as X
}

// ── Real-valued functions of a complex argument: |z| and arg z ───────────────────────────────────────────────────────

/**
 * A real-valued elementwise primitive of one complex argument, with its $\reals^2$ gradient
 * $w(z, y) = \partial f/\partial x + i\, \partial f/\partial y$ written with primitives: vjp $\bar{g} w$ (the real
 * part for a real argument), jvp $\operatorname{Re}(\operatorname{conj}(w) \dot{z})$. The result is float64 (float32
 * for a float32 argument).
 *
 * @param id The primitive's registered id (`module/name`).
 * @param f The scalar rule, of one element's real and imaginary parts (a real element has imaginary part 0).
 * @param gradient The $\reals^2$ gradient $w$, given the argument $z$ and the output $y$.
 * @param summary The one-line summary of the primitive in the registry.
 * @returns The primitive, applied as `op([z], undefined)`.
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

/**
 * $\lvert z \rvert^2 = x^2 + y^2$, elementwise and real, written with primitives (so differentiable).
 *
 * @param z The value: complex, or real.
 * @returns $\lvert z \rvert^2$, of the shape of `z`.
 */
const abs2 = (z: Value): Value => add(square(realPart(z)), square(imagPart(z)))

/**
 * The modulus primitive behind `complexAbs`: $\lvert z \rvert = \sqrt{x^2 + y^2}$, with $\reals^2$ gradient
 * $z / \lvert z \rvert$, taken as 0 at $z = 0$ (the minimum-norm subgradient).
 */
const modulusOp = realValued(
  'foundation/tensor/complexAbs',
  Math.hypot,
  // ∇|z| = z/|z|, taken as 0 at z = 0 (the minimum-norm subgradient).
  (z, y) => div(z, where(equalTo(y, 0), 1, y)),
  'The modulus |z| = √(x² + y²) of a complex number.',
)

/**
 * The modulus $\lvert z \rvert = \sqrt{x^2 + y^2}$, elementwise, as float64 (computed by `Math.hypot`, so without
 * overflow in $x^2 + y^2$); `abs` dispatches here for complex128. Differentiable, with $\reals^2$ gradient
 * $z / \lvert z \rvert$ (0 at $z = 0$).
 *
 * @param z The value: complex, or real (its modulus is then its absolute value).
 * @returns $\lvert z \rvert$, of the shape of `z`: float64 (float32 for a float32 argument).
 *
 * @example The modulus of $3 + 4i$ and its gradient $z / \lvert z \rvert$
 * const z = complex(3, 4)
 * print('|3 + 4i| =', complexAbs(z))
 * print('gradient =', complexItem(grad(complexAbs)(z)))
 */
export function complexAbs<X extends Value>(z: X): X {
  return modulusOp([z], undefined) as X
}

/**
 * The phase primitive behind `angle`: $\arg z = \operatorname{atan2}(y, x)$, with $\reals^2$ gradient
 * $(-y, x)/\lvert z \rvert^2 = iz/\lvert z \rvert^2$, taken as 0 at $z = 0$.
 */
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

/**
 * The argument (phase) of $z = x + iy$, $\operatorname{atan2}(y, x)$ in $(-\pi, \pi]$, elementwise, as float64 (for a
 * real argument, $\pi$ where it is negative and 0 elsewhere). Differentiable, with $\reals^2$ gradient
 * $iz/\lvert z \rvert^2$ (0 at $z = 0$).
 *
 * @param z The value: complex, or real.
 * @returns $\arg z$ in radians, of the shape of `z`: float64 (float32 for a float32 argument).
 *
 * @example Phases of $i$, $-1 - i$ and two reals
 * print('angle(i) =', angle(complex(0, 1)))
 * print('angle(-1 - i) =', angle(complex(-1, -1)))
 * print('angle([2, -2]) =', angle(tensor([2, -2])))
 */
export function angle<X extends Value>(z: X): X {
  return angleOp([z], undefined) as X
}

/**
 * Throw a `DTypeError` naming `op` when a value is complex: for operations that need an ordering.
 *
 * @param op The caller's name, for the error message.
 * @param x The value to check: a number, tensor or traced value.
 * @param why What the operation needs that complex values lack, for the error message.
 *
 * @example A real value passes; a complex one throws
 * refuseComplex('sort', tensor([3, 1]))
 * print('a real tensor passes')
 * try { refuseComplex('sort', complex(1, 2)) } catch (e) { print(e.name, e.message) }
 */
export function refuseComplex(op: string, x: Value, why = 'needs an ordering, which complex values lack'): void {
  if (isComplexValue(x)) throw new DTypeError(op, `${op}: ${why} (got complex128)`, ['complex128'])
}
