/**
 * Complex scalars and coefficient lists for the polynomial algorithms that run element by element (division, partial
 * fractions). Private to `aifn-compute/numerics/polynomial`; everything public takes and returns complex128 tensors. Division
 * uses Smith's (1962) scaling ("Algorithm 116: Complex division", CACM 5(8)).
 */

import { astype, fromData, isTensor, isTraced, tensor, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { ComplexNumber, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, NotDifferentiableError } from 'aifn-compute/foundation/errors'

export type C = ComplexNumber

export const of = (re: number, im = 0): C => ({ re, im })
export const add = (a: C, b: C): C => of(a.re + b.re, a.im + b.im)
export const sub = (a: C, b: C): C => of(a.re - b.re, a.im - b.im)
export const mul = (a: C, b: C): C => of(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
export const scale = (a: C, k: number): C => of(a.re * k, a.im * k)
export const abs = (a: C): number => Math.hypot(a.re, a.im)
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

/** p(x) by Horner's rule, coefficients highest power first. */
export function horner(p: readonly C[], x: C): C {
  let v = of(0)
  for (const c of p) v = add(mul(v, x), c)
  return v
}

/** Polynomial product (coefficient convolution). */
export function product(a: readonly C[], b: readonly C[]): C[] {
  const out = Array.from({ length: a.length + b.length - 1 }, () => of(0))
  a.forEach((x, i) => b.forEach((y, j) => (out[i + j] = add(out[i + j], mul(x, y)))))
  return out
}

/** numpy's `polydiv`: quotient and remainder, the remainder's leading (relative) zeros trimmed to one coefficient. */
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
 * Coefficients of an untraced input as complex scalars, and whether any was complex. Traced input throws: the caller
 * runs element by element.
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

/** A list as a rank-1 tensor: complex128 when `complex`, else float64 of the real parts. */
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
