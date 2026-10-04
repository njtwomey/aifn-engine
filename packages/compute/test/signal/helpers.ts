/** Comparison helpers for the signal tests: real and complex128 values against the scipy fixture `signal`. */
import { expect } from 'vitest'
import { imagPart, realPart, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { fixture } from '../fixtures'

export type Cx = { re: number[] | number[][]; im: number[] | number[][] }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const F = fixture<any>('signal')

/** max |actual − expected| / max(1, max |expected|) < tol. */
export function close(actual: Value | readonly number[], expected: number[] | number[][], tol = 1e-10): void {
  const a = Array.isArray(actual) ? (actual as number[]) : toFlat(actual as Tensor)
  const e = (expected as number[]).flat() as number[]
  expect(a.length).toBe(e.length)
  const scale = Math.max(1, ...e.map(Math.abs))
  let worst = 0
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - e[i]))
  expect(worst / scale).toBeLessThan(tol)
}

/** A complex128 value against `{ re, im }`. */
export function closeC(actual: Value, expected: Cx, tol = 1e-10): void {
  expect((actual as Tensor).dtype).toBe('complex128')
  close(realPart(actual), expected.re, tol)
  close(imagPart(actual), expected.im, tol)
}
