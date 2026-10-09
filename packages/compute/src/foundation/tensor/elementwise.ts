/**
 * Elementwise primitives with NumPy broadcasting. Each is defined once with its derivative rule; the rules are written
 * with primitives so that derivatives of derivatives work for these built-ins.
 *
 * Complex values: `neg`, `square`, `exp`, `log`, `sqrt`, `add`, `sub`, `mul`, `div` and `pow` carry a complex rule
 * (principal branches) and are holomorphic, so their derivatives are complex derivatives (see complex.ts for the
 * $\reals^2$ convention); `abs` gives the float64 modulus, and `where` selects complex values as they are. The others
 * are defined on the real line and raise `DTypeError` for complex arguments, as do the ordering comparisons and the
 * scalar maps `map` and `map2`; `equalTo` and `notEqualTo` compare complex values.
 */

import { DTypeError } from 'aifn-compute/foundation/errors'
import { allocate, flatData, fromData, sizeOf, type Tensor } from './core'
import { binaryKernel, complexKernel, unaryKernel, type ComplexRule } from './kernels'
import { complexAbs } from './complex'
import { definePrimitive, elementwise, elementwiseDType, type Binary, type Raw, type Unary } from './primitive'
import { avalOf, type Traced, type Value } from './trace'
import { broadcastShapes, broadcastView } from './views'

// ── Unary ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Negation $-x$, elementwise, for real and complex values; the dtype is kept (int32 stays int32).
 *
 * @example Negate a vector
 * print('neg([1, -2, 0.5]) =', neg(tensor([1, -2, 0.5])))
 */
export const neg: Unary = elementwise({
  id: 'foundation/tensor/neg',
  f: (x) => -x,
  complex: (o, z) => {
    o[0] = -z[0]
    o[1] = -z[1]
  },
  holomorphic: true,
  derivative: [() => -1],
  dtype: 'same',
  doc: { summary: 'Negation −x.' },
})

/**
 * The sign of $x$, elementwise: $-1$, $0$ or $1$ (NaN for NaN). Piecewise constant: its derivative is taken as zero
 * everywhere, which it is wherever it exists. Real values only.
 *
 * @example The sign of each entry
 * print('sign([-3, 0, 2]) =', sign(tensor([-3, 0, 2])))
 */
export const sign: Unary = elementwise({
  id: 'foundation/tensor/sign',
  f: Math.sign,
  derivative: ['zero'],
  dtype: 'same',
  doc: { summary: 'The sign of x: −1, 0 or 1.' },
})

/** The real absolute value $\lvert x \rvert$ as a primitive, with derivative $\sgn x$ (so 0 at 0); `abs` calls it. */
const absReal: Unary = elementwise({
  id: 'foundation/tensor/abs',
  f: Math.abs,
  derivative: [(x) => sign(x)],
  dtype: 'same',
  doc: { summary: 'The absolute value |x|.' },
})

/**
 * The absolute value $\lvert x \rvert$, elementwise; the derivative at 0 is taken as 0 (the subgradient $\sgn 0$). For
 * complex values, the modulus $\lvert z \rvert$ as float64 (`complexAbs`, whose $\reals^2$ gradient is
 * $z / \lvert z \rvert$).
 *
 * @example Real and complex values
 * print('abs([-2, 0, 3]) =', abs(tensor([-2, 0, 3])))
 * print('abs(3 + 4i) =', abs(complex(3, 4)))
 */
export const abs: Unary = ((x: Value) =>
  typeof x !== 'number' && avalOf(x).dtype === 'complex128' ? complexAbs(x) : absReal(x)) as Unary

/**
 * The square $x^2$, elementwise, for real and complex values; the dtype is kept.
 *
 * @example Square a vector
 * print('square([-3, 0.5, 2]) =', square(tensor([-3, 0.5, 2])))
 */
export const square: Unary = elementwise({
  id: 'foundation/tensor/square',
  f: (x) => x * x,
  complex: (o, z) => {
    o[0] = z[0] * z[0] - z[1] * z[1]
    o[1] = 2 * z[0] * z[1]
  },
  holomorphic: true,
  derivative: [(x) => mul(2, x)],
  dtype: 'same',
  doc: { summary: 'The square x².' },
})

/**
 * The exponential $e^x$, elementwise; for complex $z = x + iy$, $e^z = e^x(\cos y + i \sin y)$. Its derivative is the
 * output itself.
 *
 * @example Real values, and Euler's identity
 * print('exp([0, 1]) =', exp(tensor([0, 1])))
 * print('exp(iπ) =', complexItem(exp(complex(0, Math.PI))))
 */
export const exp: Unary = elementwise({
  id: 'foundation/tensor/exp',
  f: Math.exp,
  complex: (o, z) => {
    const m = Math.exp(z[0])
    o[0] = m * Math.cos(z[1])
    o[1] = m * Math.sin(z[1])
  },
  holomorphic: true,
  derivative: [(_x, y) => y],
  doc: { note: 'exponential-and-logarithm', summary: 'The exponential eˣ.' },
})

/**
 * $e^x - 1$, elementwise, accurate for small $x$, where computing $e^x$ first loses the digits that matter. Real
 * values only.
 *
 * @example Accurate where the plain formula is not
 * print('expm1(1e-15) =', expm1(1e-15))
 * print('exp(1e-15) - 1 =', exp(1e-15) - 1)
 */
export const expm1: Unary = elementwise({
  id: 'foundation/tensor/expm1',
  f: Math.expm1,
  derivative: [(_x, y) => add(y, 1)],
  doc: { summary: 'eˣ − 1, accurate for small x.' },
})

/**
 * The natural logarithm $\log x$, elementwise (NaN below 0, $-\infty$ at 0). For complex $z$, the principal branch
 * $\log \lvert z \rvert + i \arg z$.
 *
 * @example Real values, and the logarithm of $-1$ as a complex number
 * print('log([1, e, 0]) =', log(tensor([1, Math.E, 0])))
 * print('log(-1) =', log(-1))
 * print('log(-1 + 0i) =', complexItem(log(complex(-1, 0))))
 */
export const log: Unary = elementwise({
  id: 'foundation/tensor/log',
  f: Math.log,
  complex: (o, z) => {
    o[0] = Math.log(Math.hypot(z[0], z[1]))
    o[1] = Math.atan2(z[1], z[0])
  },
  holomorphic: true,
  derivative: [(x) => div(1, x)],
  doc: { note: 'exponential-and-logarithm', summary: 'The natural logarithm.' },
  test: { domain: { lo: 0.1, hi: 3 } },
})

/**
 * $\log(1 + x)$, elementwise, accurate for small $x$, where $1 + x$ would round. Real values only.
 *
 * @example Accurate where the plain formula is not
 * print('log1p(1e-17) =', log1p(1e-17))
 * print('log(1 + 1e-17) =', log(1 + 1e-17))
 */
export const log1p: Unary = elementwise({
  id: 'foundation/tensor/log1p',
  f: Math.log1p,
  derivative: [(x) => div(1, add(x, 1))],
  doc: { summary: 'log(1 + x), accurate for small x.' },
  test: { domain: { lo: -0.5, hi: 2 } },
})

/**
 * The square root $\sqrt{x}$, elementwise (NaN below 0). For complex $z$, the principal root (real part $\ge 0$;
 * branch cut on the negative real axis).
 *
 * @example Real values, and the root of $-1$ as a complex number
 * print('sqrt([4, 2, -1]) =', sqrt(tensor([4, 2, -1])))
 * print('sqrt(-1 + 0i) =', complexItem(sqrt(complex(-1, 0))))
 */
export const sqrt: Unary = elementwise({
  id: 'foundation/tensor/sqrt',
  f: Math.sqrt,
  complex: complexSqrt,
  holomorphic: true,
  derivative: [(_x, y) => div(0.5, y)],
  doc: { summary: 'The square root.' },
  test: { domain: { lo: 0.1, hi: 3 } },
})

/**
 * The sine $\sin x$, elementwise, of $x$ in radians. Real values only.
 *
 * @example At $0$, $\pi/2$ and $\pi$
 * print('sin([0, π/2, π]) =', sin(tensor([0, Math.PI / 2, Math.PI])))
 */
export const sin: Unary = elementwise({
  id: 'foundation/tensor/sin',
  f: Math.sin,
  derivative: [(x) => cos(x)],
  doc: { summary: 'The sine (radians).' },
})

/**
 * The cosine $\cos x$, elementwise, of $x$ in radians. Real values only.
 *
 * @example At $0$, $\pi/2$ and $\pi$
 * print('cos([0, π/2, π]) =', cos(tensor([0, Math.PI / 2, Math.PI])))
 */
export const cos: Unary = elementwise({
  id: 'foundation/tensor/cos',
  f: Math.cos,
  derivative: [(x) => neg(sin(x))],
  doc: { summary: 'The cosine (radians).' },
})

/**
 * The hyperbolic tangent $\tanh x$, elementwise, with derivative $1 - \tanh^2 x$ (computed from the output). Real
 * values only.
 *
 * @example Values and slopes
 * print('tanh([-1, 0, 1]) =', tanh(tensor([-1, 0, 1])))
 * print('slope at 0 =', grad(tanh)(0))
 */
export const tanh: Unary = elementwise({
  id: 'foundation/tensor/tanh',
  f: Math.tanh,
  derivative: [(_x, y) => sub(1, square(y))],
  doc: { summary: 'The hyperbolic tangent.' },
})

// ── Binary ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The sum $a + b$, elementwise with broadcasting, for real and complex values; the result has the promoted dtype.
 *
 * @example A row added to every row of a matrix
 * print('add =', add(tensor([[1, 2], [3, 4]]), tensor([10, 20])))
 */
export const add: Binary = elementwise({
  id: 'foundation/tensor/add',
  f: (a, b) => a + b,
  complex: (o, z) => {
    o[0] = z[0] + z[2]
    o[1] = z[1] + z[3]
  },
  holomorphic: true,
  derivative: [() => 1, () => 1],
  dtype: 'same',
  kernel: 'add',
  doc: { summary: 'The sum a + b.' },
})

/**
 * The difference $a - b$, elementwise with broadcasting, for real and complex values; the result has the promoted
 * dtype.
 *
 * @example Subtract a number from every entry
 * print('sub([5, 7], 2) =', sub(tensor([5, 7]), 2))
 */
export const sub: Binary = elementwise({
  id: 'foundation/tensor/sub',
  f: (a, b) => a - b,
  complex: (o, z) => {
    o[0] = z[0] - z[2]
    o[1] = z[1] - z[3]
  },
  holomorphic: true,
  derivative: [() => 1, () => -1],
  dtype: 'same',
  kernel: 'sub',
  doc: { summary: 'The difference a − b.' },
})

/**
 * The product $a \cdot b$, elementwise with broadcasting (not the matrix product: see `matmul`), for real and complex
 * values; the result has the promoted dtype.
 *
 * @example Scale a vector, and broadcast a column against a row
 * print('mul([1, 2, 3], 2) =', mul(tensor([1, 2, 3]), 2))
 * print('column × row =', mul(tensor([[1], [2]]), tensor([10, 20, 30])))
 */
export const mul: Binary = elementwise({
  id: 'foundation/tensor/mul',
  f: (a, b) => a * b,
  complex: (o, z) => {
    o[0] = z[0] * z[2] - z[1] * z[3]
    o[1] = z[0] * z[3] + z[1] * z[2]
  },
  holomorphic: true,
  derivative: [(_a, b) => b, (a) => a],
  dtype: 'same',
  kernel: 'mul',
  doc: { summary: 'The product a·b.' },
})

/**
 * The quotient $a / b$, elementwise with broadcasting, always floating point (int32 operands give float64). Division
 * by zero gives $\pm\infty$ or NaN, as in IEEE arithmetic. Complex quotients use Smith's algorithm.
 *
 * @example Halve a vector, and divide by zero
 * print('div([1, 2, 3], 2) =', div(tensor([1, 2, 3]), 2))
 * print('div([1, -1, 0], 0) =', div(tensor([1, -1, 0]), 0))
 */
export const div: Binary = elementwise({
  id: 'foundation/tensor/div',
  f: (a, b) => a / b,
  complex: complexDivide,
  holomorphic: true,
  derivative: [(_a, b) => div(1, b), (_a, b, y) => neg(div(y, b))],
  kernel: 'div',
  doc: { summary: 'The quotient a / b.' },
  test: {
    domain: [
      { lo: -2, hi: 2 },
      { lo: 0.5, hi: 2 },
    ],
  },
})

/**
 * The power $a^b$, elementwise with broadcasting (always floating point; complex values on the principal branch). The
 * derivative in $a$ is $b a^{b-1}$, taken as 0 at $a = b = 0$ ($a^0 = 1$ for every $a$, where $b a^{b-1}$ would be
 * $0 \cdot \infty$, e.g. the constant term of polynomial features at $x = 0$). The derivative in $b$ is $y \log a$,
 * taken as 0 where $a = 0$ (the limit for $b > 0$) and NaN for $a < 0$, where $a^b$ is not differentiable in $b$. It is
 * computed only when $b$ is differentiated, so a constant exponent never evaluates $\log a$. Both guards also replace
 * the operand inside the unused branch (the "double where"), so that branch stays finite and second derivatives are not
 * NaN there.
 *
 * @example Squares, and a square root as a power
 * print('pow([2, 3, 4], 2) =', pow(tensor([2, 3, 4]), 2))
 * print('pow(2, 0.5) =', pow(2, 0.5))
 *
 * @example The derivative of $x^0$ at $x = 0$ is 0, not NaN
 * print('d/dx x^0 at 0 =', grad((x) => pow(x, 0))(0))
 * print('d/dx x^3 at 2 =', grad((x) => pow(x, 3))(2))
 */
export const pow: Binary = elementwise({
  id: 'foundation/tensor/pow',
  f: Math.pow,
  complex: complexPower,
  holomorphic: true,
  derivative: [
    (a, b) => {
      const origin = mul(equalTo(a, 0), equalTo(b, 0))
      return where(origin, 0, mul(b, pow(a, sub(where(origin, 1, b), 1))))
    },
    (a, _b, y) => {
      const zero = equalTo(a, 0)
      return where(zero, 0, mul(y, log(where(zero, 1, a))))
    },
  ],
  doc: { summary: 'The power aᵇ.' },
  test: {
    domain: [
      { lo: 0.2, hi: 2 },
      { lo: -2, hi: 2 },
    ],
  },
})

/**
 * The larger of $a$ and $b$, elementwise with broadcasting. Where they tie the derivative goes to $a$. NaN propagates.
 * Real values only.
 *
 * @example A floor of 3, and NaN propagating
 * print('maximum([1, 5, NaN], 3) =', maximum(tensor([1, 5, NaN]), 3))
 */
export const maximum: Binary = elementwise({
  id: 'foundation/tensor/maximum',
  f: (a, b) => (a !== a || b !== b ? NaN : a >= b ? a : b),
  derivative: [(a, b) => greaterEqual(a, b), (a, b) => less(a, b)],
  dtype: 'same',
  doc: { summary: 'The larger of a and b.' },
})

/**
 * The smaller of $a$ and $b$, elementwise with broadcasting. Where they tie the derivative goes to $a$. NaN
 * propagates. Real values only.
 *
 * @example A ceiling of 3, and NaN propagating
 * print('minimum([1, 5, NaN], 3) =', minimum(tensor([1, 5, NaN]), 3))
 */
export const minimum: Binary = elementwise({
  id: 'foundation/tensor/minimum',
  f: (a, b) => (a !== a || b !== b ? NaN : a <= b ? a : b),
  derivative: [(a, b) => lessEqual(a, b), (a, b) => greater(a, b)],
  dtype: 'same',
  doc: { summary: 'The smaller of a and b.' },
})

/**
 * $x$ limited to $[\ell, h]$ elementwise, $\min(\max(x, \ell), h)$, with broadcasting (an explicit operation: nothing
 * in aifn clips silently). Differentiable, through `maximum` and `minimum`: the derivative goes to $x$ inside the range
 * (and at its ends), and to the bound outside it. Real values only.
 *
 * @param x The values to limit.
 * @param lo The lower bound $\ell$, broadcast against `x`.
 * @param hi The upper bound $h$, broadcast against `x`. It is applied last, so it wins where it is below `lo`.
 * @returns `x` with entries below `lo` raised to it and entries above `hi` lowered to it, in the broadcast shape.
 *
 * @example Clip to the unit interval
 * print('clip([-2, 0.5, 3], 0, 1) =', clip(tensor([-2, 0.5, 3]), 0, 1))
 */
export function clip(x: Value, lo: Value, hi: Value): Value {
  return minimum(maximum(x, lo), hi)
}

// ── Comparisons (bool results; piecewise constant, so their derivatives are zero) ────────────────────────────────────

/** A comparison: numbers give 1 or 0, tensors give a bool mask, traced values a traced mask. */
export interface Comparison {
  (a: number, b: number): number
  (a: Tensor, b: Value): Tensor
  (a: Value, b: Tensor): Tensor
  (a: Raw, b: Raw): Raw
  (a: Value, b: Value): Value
}

// A comparison is a primitive (so it batches under vmap) whose derivative is zero: the derivative transforms compute
// the mask on the primal values and treat it as a constant.
// Ordering comparisons refuse complex values; equalTo and notEqualTo compare (re, im) pairs.
/**
 * Define the comparison primitive `foundation/tensor/<name>`: 1 where `test` holds and 0 elsewhere, a number for two
 * numbers and a bool tensor otherwise. Its derivative is zero (the mask is a constant to the transforms), and it
 * batches under `vmap`. Without a complex rule it refuses complex values with `DTypeError`.
 *
 * @param name The comparison's name: its primitive is registered as `foundation/tensor/<name>`, and errors name it.
 * @param test The comparison of two real numbers.
 * @param summary The one-line summary of the primitive in the registry.
 * @param complex The rule for complex arguments, given (re, im) pairs, which writes 1 or 0 to the real part of its
 *   result. Left out, complex arguments throw `DTypeError`.
 * @returns The comparison, callable on numbers, tensors and traced values.
 */
function comparison(
  name: string,
  test: (a: number, b: number) => boolean,
  summary: string,
  complex?: ComplexRule,
): Comparison {
  const f = (a: number, b: number) => (test(a, b) ? 1 : 0)
  return elementwise({
    id: `foundation/tensor/${name}`,
    f,
    derivative: ['zero', 'zero'],
    dtype: 'bool',
    impl: ([a, b]) => {
      if (typeof a === 'number' && typeof b === 'number') return f(a, b)
      const cx = [a, b].some((v) => typeof v !== 'number' && v.dtype === 'complex128')
      if (!cx) return binaryKernel(a, b, f, 'bool')
      if (complex === undefined) throw new DTypeError(name, `${name}: complex values have no ordering`, ['complex128'])
      const t = complexKernel([a, b], complex, true)
      return fromData(flatData(t, 'bool'), t.shape, 'bool')
    },
    doc: { summary },
  }) as Comparison
}

/**
 * $a < b$, elementwise with broadcasting: 1 where it holds, 0 elsewhere (a bool tensor for tensor arguments). Its
 * derivative is zero. Complex values throw `DTypeError`.
 *
 * @example Which entries are below 2
 * print('less([1, 2, 3], 2) =', less(tensor([1, 2, 3]), 2))
 */
export const less = comparison('less', (a, b) => a < b, 'a < b, as 1 or 0.')
/**
 * $a \le b$, elementwise with broadcasting: 1 where it holds, 0 elsewhere (a bool tensor for tensor arguments). Its
 * derivative is zero. Complex values throw `DTypeError`.
 *
 * @example Which entries are at most 2
 * print('lessEqual([1, 2, 3], 2) =', lessEqual(tensor([1, 2, 3]), 2))
 */
export const lessEqual = comparison('lessEqual', (a, b) => a <= b, 'a ≤ b, as 1 or 0.')
/**
 * $a > b$, elementwise with broadcasting: 1 where it holds, 0 elsewhere (a bool tensor for tensor arguments). Its
 * derivative is zero. Complex values throw `DTypeError`.
 *
 * @example Which entries are above 2
 * print('greater([1, 2, 3], 2) =', greater(tensor([1, 2, 3]), 2))
 */
export const greater = comparison('greater', (a, b) => a > b, 'a > b, as 1 or 0.')
/**
 * $a \ge b$, elementwise with broadcasting: 1 where it holds, 0 elsewhere (a bool tensor for tensor arguments). Its
 * derivative is zero. Complex values throw `DTypeError`.
 *
 * @example Which entries are at least 2
 * print('greaterEqual([1, 2, 3], 2) =', greaterEqual(tensor([1, 2, 3]), 2))
 */
export const greaterEqual = comparison('greaterEqual', (a, b) => a >= b, 'a ≥ b, as 1 or 0.')
/**
 * $a = b$, elementwise with broadcasting: 1 where it holds, 0 elsewhere (a bool tensor for tensor arguments). NaN
 * equals nothing, itself included. Complex values are equal when both parts are. Its derivative is zero.
 *
 * @example NaN equals nothing; complex values compare both parts
 * print('equalTo([1, NaN, 3], [1, NaN, 0]) =', equalTo(tensor([1, NaN, 3]), tensor([1, NaN, 0])))
 * print('1 + 2i = 1 + 2i:', equalTo(complex(1, 2), complex(1, 2)))
 */
export const equalTo = comparison(
  'equalTo',
  (a, b) => a === b,
  'a = b, as 1 or 0.',
  (o, z) => {
    o[0] = z[0] === z[2] && z[1] === z[3] ? 1 : 0
  },
)
/**
 * $a \ne b$, elementwise with broadcasting: 1 where it holds, 0 elsewhere (a bool tensor for tensor arguments). NaN
 * differs from everything, itself included. Complex values differ when either part does. Its derivative is zero.
 *
 * @example Which entries differ from 1
 * print('notEqualTo([1, 2, NaN], 1) =', notEqualTo(tensor([1, 2, NaN]), 1))
 */
export const notEqualTo = comparison(
  'notEqualTo',
  (a, b) => a !== b,
  'a ≠ b, as 1 or 0.',
  (o, z) => {
    o[0] = z[0] !== z[2] || z[1] !== z[3] ? 1 : 0
  },
)

// ── Selection and arbitrary maps ─────────────────────────────────────────────────────────────────────────────────────

/**
 * The selection primitive behind `where`: its derivative is zero in the condition $c$, $[c \ne 0]$ in $a$ and
 * $[c = 0]$ in $b$.
 */
const whereOp = elementwise({
  id: 'foundation/tensor/where',
  f: (c, a, b) => (c !== 0 ? a : b),
  // The condition is piecewise constant; the choice is linear in a and b.
  derivative: ['zero', (c) => notEqualTo(c, 0), (c) => equalTo(c, 0)],
  impl: ([c, a, b]) => {
    if (typeof c === 'number' && typeof a === 'number' && typeof b === 'number') return c !== 0 ? a : b
    if ([a, b].some((v) => typeof v !== 'number' && v.dtype === 'complex128'))
      return complexKernel([c, a, b], (o, z) => {
        const pick = z[0] !== 0 || z[1] !== 0 ? 2 : 4
        o[0] = z[pick]
        o[1] = z[pick + 1]
      })
    const asTensor = (v: Raw): Tensor => (typeof v === 'number' ? fromData(new Float64Array([v]), []) : v)
    const [tc, ta, tb] = [asTensor(c), asTensor(a), asTensor(b)]
    const shape = broadcastShapes(tc.shape, ta.shape, tb.shape)
    const [fc, fa, fb] = [tc, ta, tb].map((t) => flatData(broadcastView(t, shape)))
    const out = allocate(elementwiseDType([c, a, b], 'same'), sizeOf(shape))
    for (let k = 0; k < out.length; k++) out[k] = fc[k] !== 0 ? fa[k] : fb[k]
    return fromData(out, shape)
  },
  // The result has the promoted dtype of its arguments (a bool condition promotes nothing), as the shape rule says.
  dtype: 'same',
  doc: { summary: 'Elementwise choice: a where the condition is non-zero, b elsewhere.' },
  test: {
    domain: [
      { lo: 0, hi: 1, integer: true },
      { lo: -2, hi: 2 },
      { lo: -2, hi: 2 },
    ],
  },
})

/**
 * Elementwise choice: $a$ where `condition` is non-zero, $b$ elsewhere, all three broadcast. Differentiable in $a$ and
 * $b$ (each receives the cotangent where it was chosen); the condition is treated as a constant. Complex `a` and `b`
 * are selected as they are.
 *
 * @param condition The mask: true where it is non-zero (a bool mask from a comparison, or any numbers).
 * @param a The values taken where `condition` is non-zero.
 * @param b The values taken where `condition` is zero.
 * @returns The chosen values, in the broadcast shape of the three and the promoted dtype of the three (a bool
 *   condition promotes nothing).
 *
 * @example Replace negative entries with zero
 * const x = tensor([-1, 2, -3, 4])
 * print('where(x > 0, x, 0) =', where(greater(x, 0), x, 0))
 *
 * @example The gradient flows to the branch that was chosen
 * // A leaky ReLU: slope 1 above 0, 0.1 below.
 * const leaky = (x) => sum(where(greater(x, 0), x, mul(0.1, x)))
 * print('gradient at [-2, 3] =', grad(leaky)(tensor([-2, 3])))
 */
export function where(condition: Value, a: number, b: number): number
export function where(condition: Value, a: Traced, b: Value): Traced
export function where(condition: Value, a: Value, b: Traced): Traced
export function where(condition: Value, a: Tensor, b: Tensor | number): Tensor
export function where(condition: Value, a: number, b: Tensor): Tensor
export function where(condition: Value, a: Value, b: Value): Value
export function where(condition: Value, a: Value, b: Value): Value {
  return whereOp(condition, a, b)
}

/**
 * Apply an arbitrary scalar function elementwise. It has no derivative (differentiating through it throws): to make a
 * function differentiable, define it with `elementwise` and give its derivative. It batches under `vmap`.
 *
 * @param x The values: a number, or a real tensor (complex128 throws `DTypeError`).
 * @param f The scalar function, called once per element.
 * @returns `f` of each element: a number for a number, otherwise a float64 tensor of the shape of `x`.
 *
 * @example Any JavaScript function of one number
 * print('odd entries:', map(tensor([1, 2, 3, 4]), (v) => v % 2))
 */
export function map(x: number, f: (v: number) => number): number
export function map(x: Tensor, f: (v: number) => number): Tensor
export function map(x: Value, f: (v: number) => number): Value
export function map(x: Value, f: (v: number) => number): Value {
  return mapOp([x], f)
}

/**
 * A local primitive for `map` and `map2`: no derivative at all (`vjp: null`, so jvp is missing by design: the scalar
 * function is opaque), batched by broadcasting like every elementwise primitive, with the broadcasting shape rule
 * (float64 results). Built once each; the scalar function is the primitive's parameter.
 *
 * @param name The primitive's id: a bare name (`map`, `map2`), so the primitive is local and not registered.
 * @param impl The forward rule on raw arguments, given the scalar function (the primitive's parameter).
 * @returns The primitive, applied as `op(args, f)`.
 */
function escapeHatch<P>(name: string, impl: (args: Raw[], f: P) => Raw) {
  return definePrimitive<P>({
    id: name,
    kind: 'elementwise',
    impl,
    vjp: null,
    shape: (avals) => ({
      shape: broadcastShapes(...avals.map((a) => a.shape)),
      dtype: 'float64',
      number: avals.every((a) => a.number),
    }),
  })
}

/**
 * Apply an arbitrary scalar function of two broadcast arguments elementwise. It has no derivative (see `map`).
 *
 * @param a The first arguments: a number, or a real tensor (complex128 throws `DTypeError`), broadcast against `b`.
 * @param b The second arguments, broadcast against `a`.
 * @param f The scalar function, called once per pair of broadcast elements.
 * @returns `f` of each pair: a number when both are numbers, otherwise a float64 tensor of the broadcast shape.
 *
 * @example Any JavaScript function of two numbers
 * print('[7, 8, 9] mod 4 =', map2(tensor([7, 8, 9]), 4, (x, y) => x % y))
 */
export function map2(a: number, b: number, f: (x: number, y: number) => number): number
export function map2(a: Tensor, b: Tensor | number, f: (x: number, y: number) => number): Tensor
export function map2(a: number, b: Tensor, f: (x: number, y: number) => number): Tensor
export function map2(a: Value, b: Value, f: (x: number, y: number) => number): Value
export function map2(a: Value, b: Value, f: (x: number, y: number) => number): Value {
  return map2Op([a, b], f)
}

/** The primitive behind `map`: the scalar function applied to each element of a real value. */
const mapOp = escapeHatch<(v: number) => number>('map', ([v], f) =>
  typeof v === 'number' ? f(v) : unaryKernel(realOnly('map', v), f, 'float64'),
)

/** The primitive behind `map2`: the scalar function applied to each pair of broadcast elements of two real values. */
const map2Op = escapeHatch<(x: number, y: number) => number>('map2', ([x, y], f) =>
  typeof x === 'number' && typeof y === 'number'
    ? f(x, y)
    : binaryKernel(realOnly('map2', x), realOnly('map2', y), f, 'float64'),
)

/**
 * A raw value that must not be complex (the scalar maps take one real number per element).
 *
 * @param where The caller's name, for the error message.
 * @param x The raw value to check.
 * @returns `x` itself; a complex128 tensor throws `DTypeError` instead.
 */
function realOnly<R extends Raw>(where: string, x: R): R {
  if (typeof x !== 'number' && x.dtype === 'complex128')
    throw new DTypeError(where, `${where}: the scalar function takes real numbers, not complex128`, ['complex128'])
  return x
}

// ── Complex scalar rules (principal branches) ────────────────────────────────────────────────────────────────────────

/**
 * $(a + ib)/(c + id)$ by Smith's algorithm (1962), which avoids overflow in $c^2 + d^2$. A zero divisor divides each
 * part by 0 (infinite or NaN parts).
 *
 * @param o Where the quotient is written: its real part to `o[0]` and its imaginary part to `o[1]`.
 * @param z The operands as (re, im) pairs: $a$ and $b$ in `z[0]` and `z[1]`, $c$ and $d$ in `z[2]` and `z[3]`.
 */
function complexDivide(o: Float64Array, z: Float64Array): void {
  const [a, b, c, d] = [z[0], z[1], z[2], z[3]]
  if (Math.abs(c) >= Math.abs(d)) {
    if (c === 0 && d === 0) {
      o[0] = a / 0
      o[1] = b / 0
      return
    }
    const r = d / c
    const den = c + d * r
    o[0] = (a + b * r) / den
    o[1] = (b - a * r) / den
  } else {
    const r = c / d
    const den = c * r + d
    o[0] = (a * r + b) / den
    o[1] = (b * r - a) / den
  }
}

/**
 * The principal square root of $z = x + iy$: real part $\ge 0$, and the imaginary part takes the sign of $y$ (so $-0$
 * below the cut): $t = \sqrt{(\lvert z \rvert + \lvert x \rvert)/2}$, then $(t, y/2t)$ for $x \ge 0$ and
 * $(\lvert y \rvert/2t, \pm t)$ for $x < 0$. The root of 0 is 0.
 *
 * @param o Where the root is written: its real part to `o[0]` and its imaginary part to `o[1]`.
 * @param z The argument as $x$ in `z[0]` and $y$ in `z[1]`.
 */
function complexSqrt(o: Float64Array, z: Float64Array): void {
  const [x, y] = [z[0], z[1]]
  if (x === 0 && y === 0) {
    o[0] = 0
    o[1] = y
    return
  }
  const t = Math.sqrt((Math.hypot(x, y) + Math.abs(x)) / 2)
  if (x >= 0) {
    o[0] = t
    o[1] = y / (2 * t)
  } else {
    o[0] = Math.abs(y) / (2 * t)
    o[1] = y < 0 || Object.is(y, -0) ? -t : t
  }
}

/**
 * $a^b = \exp(b \log a)$ on the principal branch; $0^b$ is 1 for $b = 0$, 0 for $\operatorname{Re} b > 0$ and NaN
 * otherwise.
 *
 * @param o Where the power is written: its real part to `o[0]` and its imaginary part to `o[1]`.
 * @param z The operands as (re, im) pairs: the base $a$ in `z[0]` and `z[1]`, the exponent $b$ in `z[2]` and `z[3]`.
 */
function complexPower(o: Float64Array, z: Float64Array): void {
  const [ar, ai, br, bi] = [z[0], z[1], z[2], z[3]]
  if (ar === 0 && ai === 0) {
    const one = br === 0 && bi === 0
    o[0] = one ? 1 : br > 0 ? 0 : NaN
    o[1] = one || br > 0 ? 0 : NaN
    return
  }
  const lr = Math.log(Math.hypot(ar, ai))
  const li = Math.atan2(ai, ar)
  const er = br * lr - bi * li
  const ei = br * li + bi * lr
  const m = Math.exp(er)
  o[0] = m * Math.cos(ei)
  o[1] = m * Math.sin(ei)
}
