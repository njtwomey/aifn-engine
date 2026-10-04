/**
 * Elementwise primitives with NumPy broadcasting. Each is defined once with its derivative rule; the rules are written
 * with primitives so that derivatives of derivatives work for these built-ins.
 *
 * Complex values: neg, square, exp, log, sqrt, add, sub, mul, div and pow carry a complex rule (principal branches)
 * and are holomorphic, so their derivatives are complex derivatives (see complex.ts for the ℝ² convention); `abs`
 * gives the float64 modulus. The others are defined on the real line and raise `DTypeError` for complex arguments,
 * as do the ordering comparisons; `equalTo` and `notEqualTo` compare complex values.
 */

import { DTypeError } from 'aifn-compute/foundation/errors'
import { allocate, flatData, fromData, sizeOf, type Tensor } from './core'
import { binaryKernel, complexKernel, unaryKernel, type ComplexRule } from './kernels'
import { complexAbs } from './complex'
import { definePrimitive, elementwise, elementwiseDType, type Binary, type Raw, type Unary } from './primitive'
import { avalOf, type Traced, type Value } from './trace'
import { broadcastShapes, broadcastView } from './views'

// ── Unary ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** −x. */
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

/** The sign of x: −1, 0 or 1 (NaN for NaN). Piecewise constant: its derivative is zero wherever it exists. */
export const sign: Unary = elementwise({
  id: 'foundation/tensor/sign',
  f: Math.sign,
  derivative: ['zero'],
  dtype: 'same',
  doc: { summary: 'The sign of x: −1, 0 or 1.' },
})

const absReal: Unary = elementwise({
  id: 'foundation/tensor/abs',
  f: Math.abs,
  derivative: [(x) => sign(x)],
  dtype: 'same',
  doc: { summary: 'The absolute value |x|.' },
})

/**
 * |x|; the derivative at 0 is taken as 0 (the subgradient sign(0)). For complex values, the modulus |z| as float64
 * (`complexAbs`, whose ℝ² gradient is z/|z|).
 */
export const abs: Unary = ((x: Value) =>
  typeof x !== 'number' && avalOf(x).dtype === 'complex128' ? complexAbs(x) : absReal(x)) as Unary

/** x². */
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

/** eˣ. */
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

/** eˣ − 1, accurate for small x. */
export const expm1: Unary = elementwise({
  id: 'foundation/tensor/expm1',
  f: Math.expm1,
  derivative: [(_x, y) => add(y, 1)],
  doc: { summary: 'eˣ − 1, accurate for small x.' },
})

/** Natural logarithm (NaN below 0, −∞ at 0). For complex z, the principal branch log|z| + i·arg z. */
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

/** log(1 + x), accurate for small x. */
export const log1p: Unary = elementwise({
  id: 'foundation/tensor/log1p',
  f: Math.log1p,
  derivative: [(x) => div(1, add(x, 1))],
  doc: { summary: 'log(1 + x), accurate for small x.' },
  test: { domain: { lo: -0.5, hi: 2 } },
})

/** √x (NaN below 0). For complex z, the principal root (real part ≥ 0; branch cut on the negative real axis). */
export const sqrt: Unary = elementwise({
  id: 'foundation/tensor/sqrt',
  f: Math.sqrt,
  complex: complexSqrt,
  holomorphic: true,
  derivative: [(_x, y) => div(0.5, y)],
  doc: { summary: 'The square root.' },
  test: { domain: { lo: 0.1, hi: 3 } },
})

/** sin x (radians). */
export const sin: Unary = elementwise({
  id: 'foundation/tensor/sin',
  f: Math.sin,
  derivative: [(x) => cos(x)],
  doc: { summary: 'The sine (radians).' },
})

/** cos x (radians). */
export const cos: Unary = elementwise({
  id: 'foundation/tensor/cos',
  f: Math.cos,
  derivative: [(x) => neg(sin(x))],
  doc: { summary: 'The cosine (radians).' },
})

/** tanh x. */
export const tanh: Unary = elementwise({
  id: 'foundation/tensor/tanh',
  f: Math.tanh,
  derivative: [(_x, y) => sub(1, square(y))],
  doc: { summary: 'The hyperbolic tangent.' },
})

// ── Binary ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** a + b. */
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

/** a − b. */
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

/** a · b (elementwise). */
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

/** a / b (always floating point). */
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
 * aᵇ (always floating point). The derivative in a is b·aᵇ⁻¹, taken as 0 at a = b = 0 (a⁰ = 1 for every a, where
 * b·aᵇ⁻¹ would be 0·∞, e.g. the constant term of polynomial features at x = 0). The derivative in b is y·log a, taken
 * as 0 where a = 0 (the limit for b > 0) and NaN for a < 0, where aᵇ is not differentiable in b. It is computed only
 * when b is differentiated, so a constant exponent never evaluates log a. Both guards also replace the operand inside
 * the unused branch (the "double where"), so that branch stays finite and second derivatives are not NaN there.
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

/** The larger of a and b. Where they tie the derivative goes to a. NaN propagates. */
export const maximum: Binary = elementwise({
  id: 'foundation/tensor/maximum',
  f: (a, b) => (a !== a || b !== b ? NaN : a >= b ? a : b),
  derivative: [(a, b) => greaterEqual(a, b), (a, b) => less(a, b)],
  dtype: 'same',
  doc: { summary: 'The larger of a and b.' },
})

/** The smaller of a and b. Where they tie the derivative goes to a. NaN propagates. */
export const minimum: Binary = elementwise({
  id: 'foundation/tensor/minimum',
  f: (a, b) => (a !== a || b !== b ? NaN : a <= b ? a : b),
  derivative: [(a, b) => lessEqual(a, b), (a, b) => greater(a, b)],
  dtype: 'same',
  doc: { summary: 'The smaller of a and b.' },
})

/** x limited to [lo, hi] elementwise (an explicit operation: nothing in aifn clips silently). */
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

/** a < b. */
export const less = comparison('less', (a, b) => a < b, 'a < b, as 1 or 0.')
/** a ≤ b. */
export const lessEqual = comparison('lessEqual', (a, b) => a <= b, 'a ≤ b, as 1 or 0.')
/** a > b. */
export const greater = comparison('greater', (a, b) => a > b, 'a > b, as 1 or 0.')
/** a ≥ b. */
export const greaterEqual = comparison('greaterEqual', (a, b) => a >= b, 'a ≥ b, as 1 or 0.')
/** a = b (NaN equals nothing). */
export const equalTo = comparison(
  'equalTo',
  (a, b) => a === b,
  'a = b, as 1 or 0.',
  (o, z) => {
    o[0] = z[0] === z[2] && z[1] === z[3] ? 1 : 0
  },
)
/** a ≠ b. */
export const notEqualTo = comparison(
  'notEqualTo',
  (a, b) => a !== b,
  'a ≠ b, as 1 or 0.',
  (o, z) => {
    o[0] = z[0] !== z[2] || z[1] !== z[3] ? 1 : 0
  },
)

// ── Selection and arbitrary maps ─────────────────────────────────────────────────────────────────────────────────────

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
 * Elementwise choice: a where `condition` is non-zero, b elsewhere, all three broadcast. Differentiable in a and b;
 * the condition is treated as a constant.
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
 * Apply an arbitrary scalar function elementwise. It has no derivative: to make a function differentiable, define it
 * with `elementwise` and give its derivative.
 */
export function map(x: number, f: (v: number) => number): number
export function map(x: Tensor, f: (v: number) => number): Tensor
export function map(x: Value, f: (v: number) => number): Value
export function map(x: Value, f: (v: number) => number): Value {
  return mapOp([x], f)
}

/**
 * A local primitive for `map` and `map2`: no derivative at all (`vjp: null`, so jvp is missing by design: the scalar
 * function is opaque), batched by broadcasting like every elementwise primitive, with the broadcasting shape rule.
 * Built once each; the scalar function is the primitive's parameter.
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

/** Apply an arbitrary scalar function of two broadcast arguments elementwise. It has no derivative (see `map`). */
export function map2(a: number, b: number, f: (x: number, y: number) => number): number
export function map2(a: Tensor, b: Tensor | number, f: (x: number, y: number) => number): Tensor
export function map2(a: number, b: Tensor, f: (x: number, y: number) => number): Tensor
export function map2(a: Value, b: Value, f: (x: number, y: number) => number): Value
export function map2(a: Value, b: Value, f: (x: number, y: number) => number): Value {
  return map2Op([a, b], f)
}

const mapOp = escapeHatch<(v: number) => number>('map', ([v], f) =>
  typeof v === 'number' ? f(v) : unaryKernel(realOnly('map', v), f, 'float64'),
)

const map2Op = escapeHatch<(x: number, y: number) => number>('map2', ([x, y], f) =>
  typeof x === 'number' && typeof y === 'number'
    ? f(x, y)
    : binaryKernel(realOnly('map2', x), realOnly('map2', y), f, 'float64'),
)

/** A raw value that must not be complex (the scalar maps take one real number per element). */
function realOnly<R extends Raw>(where: string, x: R): R {
  if (typeof x !== 'number' && x.dtype === 'complex128')
    throw new DTypeError(where, `${where}: the scalar function takes real numbers, not complex128`, ['complex128'])
  return x
}

// ── Complex scalar rules (principal branches) ────────────────────────────────────────────────────────────────────────

/** (a + ib)/(c + id) by Smith's algorithm (1962), which avoids overflow in c² + d². */
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
 * The principal square root: real part ≥ 0, and the imaginary part takes the sign of im z (so −0 below the cut):
 * t = √((|z| + |x|)/2), then (t, y/2t) for x ≥ 0 and (|y|/2t, ±t) for x < 0.
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

/** aᵇ = exp(b·log a) on the principal branch; 0ᵇ is 1 for b = 0, 0 for Re b > 0 and NaN otherwise. */
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
