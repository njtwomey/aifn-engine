/**
 * Complex scalars and coefficient lists for the polynomial algorithms that run element by element (division, partial
 * fractions). Private to `aifn-compute/numerics/polynomial`; everything public takes and returns complex128 tensors. Division
 * uses Smith's (1962) scaling ("Algorithm 116: Complex division", CACM 5(8)).
 */

import { astype, fromData, isTensor, isTraced, tensor, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { ComplexNumber, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, NotDifferentiableError } from 'aifn-compute/foundation/errors'

export type C = ComplexNumber

/**
 * Construct a complex number $z = \text{re} + i\,\text{im}$.
 *
 * @param re The real part of the complex number.
 * @param im The imaginary part (default 0).
 * @returns The complex number representation `{ re, im }`.
 */
export const of = (re: number, im = 0): C => ({ re, im })

/**
 * Add two complex numbers $a + b$.
 *
 * @param a The first complex term.
 * @param b The second complex term.
 * @returns The sum $a + b$.
 */
export const add = (a: C, b: C): C => of(a.re + b.re, a.im + b.im)

/**
 * Subtract two complex numbers $a - b$.
 *
 * @param a The minuend.
 * @param b The subtrahend.
 * @returns The difference $a - b$.
 */
export const sub = (a: C, b: C): C => of(a.re - b.re, a.im - b.im)

/**
 * Multiply two complex numbers $a \cdot b$.
 *
 * @param a The first complex factor.
 * @param b The second complex factor.
 * @returns The complex product $a \cdot b$.
 */
export const mul = (a: C, b: C): C => of(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)

/**
 * Scale a complex number by a real scalar $k \cdot a$.
 *
 * @param a The complex number to scale.
 * @param k The real scaling factor.
 * @returns The scaled complex number.
 */
export const scale = (a: C, k: number): C => of(a.re * k, a.im * k)

/**
 * Compute the absolute value (modulus) $|a| = \sqrt{a_{\text{re}}^2 + a_{\text{im}}^2}$.
 *
 * @param a The complex number.
 * @returns The Euclidean modulus $|a|$.
 */
export const abs = (a: C): number => Math.hypot(a.re, a.im)

/**
 * Divide two complex numbers $a / b$ using Smith's (1962) scaled algorithm to avoid intermediate overflow.
 *
 * @param a The complex numerator.
 * @param b The complex denominator.
 * @returns The quotient $a / b$.
 */
export function div(a: C, b: C): C {
  if (Math.abs(b.re) >= Math.abs(b.im)) {
    const r = b.im / b.re
    const d = b.re + b.im * r
    return of((a.re + a.im * r) / d, (a.im - a.re * r) / d)
  }
  const r = b.re / b.im
  const d = b.re * r + b.im
  return of((a.re * r + a.im) / d, (a.im * r - a.re) / d)
}

/**
 * Evaluate a polynomial $p(x)$ by Horner's rule, given coefficients in descending degree order.
 *
 * @param p The polynomial coefficients $[c_d, \dots, c_0]$ representing $\sum_{k=0}^d c_{d-k} x^k$.
 * @param x The complex evaluation point.
 * @returns The value of $p(x)$.
 */
export function horner(p: readonly C[], x: C): C {
  let v = of(0)
  for (const c of p) v = add(mul(v, x), c)
  return v
}

/**
 * Compute the polynomial product $a(x) \cdot b(x)$ via discrete coefficient convolution.
 *
 * @param a The first polynomial's coefficients in descending degree order.
 * @param b The second polynomial's coefficients in descending degree order.
 * @returns The product polynomial's coefficients in descending degree order.
 */
export function product(a: readonly C[], b: readonly C[]): C[] {
  const out = Array.from({ length: a.length + b.length - 1 }, () => of(0))
  a.forEach((x, i) => b.forEach((y, j) => (out[i + j] = add(out[i + j], mul(x, y)))))
  return out
}

/**
 * Polynomial synthetic division (numpy `polydiv`): computes quotient $q(x)$ and remainder $r(x)$ such that
 * $u(x) = q(x) v(x) + r(x)$, trimming relative leading zeros of the remainder.
 *
 * @param u The dividend polynomial's coefficients in descending degree order.
 * @param v The divisor polynomial's coefficients in descending degree order.
 * @returns An object containing quotient coefficients `q` and remainder coefficients `r`.
 */
export function divide(u: readonly C[], v: readonly C[]): { q: C[]; r: C[] } {
  const m = u.length - 1
  const n = v.length - 1
  if (abs(v[0]) === 0) throw new DomainError('polyDivide', 'polyDivide: the divisor has a zero leading coefficient')
  const q = Array.from({ length: Math.max(m - n + 1, 1) }, () => of(0))
  const r = [...u]
  for (let k = 0; k <= m - n; k++) {
    const d = div(r[k], v[0])
    q[k] = d
    for (let j = 0; j <= n; j++) r[k + j] = sub(r[k + j], mul(d, v[j]))
  }
  const big = Math.max(...u.map(abs), 1e-300)
  let start = 0
  while (start < r.length - 1 && abs(r[start]) <= 1e-14 * big) start++
  return { q, r: r.slice(start) }
}

/**
 * Parse an untraced input into a list of complex scalar coefficients and indicate whether any entry was complex.
 * Throws `NotDifferentiableError` if given traced values.
 *
 * @param x The input coefficient values: a number, tensor, or array of numbers or complex objects.
 * @param where The caller name for error messages.
 * @returns The parsed list of coefficients and a boolean flag indicating if any coefficient was complex.
 */
export function readList(x: Value | VectorLike | readonly C[], where: string): { list: C[]; complex: boolean } {
  if (isTraced(x as unknown))
    throw new NotDifferentiableError(where, `${where}: runs element by element and has no derivative rule`)
  if (typeof x === 'number') return { list: [of(x)], complex: false }
  if (isTensor(x)) {
    const d = astype(x, 'complex128').data as Float64Array
    return {
      list: Array.from({ length: d.length / 2 }, (_, k) => of(d[2 * k], d[2 * k + 1])),
      complex: x.dtype === 'complex128',
    }
  }
  // A list may mix numbers and `{ re, im }` objects.
  const arr = Array.from(x as ArrayLike<number | C>)
  const complex = arr.some((v) => typeof v === 'object')
  return { list: arr.map((v) => (typeof v === 'number' ? of(v) : of(v.re, v.im))), complex }
}

/**
 * Convert a list of complex numbers to a 1D tensor: `complex128` when `complex` is true, otherwise `float64` of the
 * real parts.
 *
 * @param xs The list of complex numbers.
 * @param complex Whether to construct a complex tensor (`complex128`) or real tensor (`float64`).
 * @returns A rank-1 tensor holding the coefficient values.
 */
export function toTensor(xs: readonly C[], complex: boolean): Tensor {
  if (!complex)
    return tensor(
      xs.map((z) => z.re),
      [xs.length],
    )
  const d = new Float64Array(2 * xs.length)
  xs.forEach((z, k) => {
    d[2 * k] = z.re
    d[2 * k + 1] = z.im
  })
  return fromData(d, [xs.length], 'complex128')
}
