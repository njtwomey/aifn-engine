/**
 * Floating-point constants and default tolerances, defined once for all of aifn (design K §9). Modules import these
 * rather than writing `2 ** -52` or `1e-12` of their own.
 *
 * Tolerances are `{ rtol, atol }` pairs: two values a and b agree when |a − b| ≤ atol + rtol·|b|, and an iteration has
 * converged when ‖Δ‖ ≤ atol + rtol·‖x‖.
 */

import type { DType } from './core'

/** Machine epsilon of float64: the gap between 1 and the next representable number, 2⁻⁵² ≈ 2.22e-16. */
export const EPS = 2 ** -52

/** √EPS = 2⁻²⁶ ≈ 1.49e-8: the usual relative step of a forward finite difference, and a loose tolerance. */
export const SQRT_EPS = 2 ** -26

/** The smallest positive normal float64, 2⁻¹⁰²² ≈ 2.23e-308. Values below it are subnormal and lose precision. */
export const TINY = 2 ** -1022

/** Machine epsilon of float32, 2⁻²³ ≈ 1.19e-7. */
export const EPS32 = 2 ** -23

/** A relative and an absolute tolerance. */
export type Tolerance = { readonly rtol: number; readonly atol: number }

/**
 * Default tolerances for comparing results of each dtype, as `torch.testing.assert_close` sets them: float64
 * rtol 1e-7, atol 1e-7; float32 rtol 1.3e-6, atol 1e-5; int32 and bool exact; complex128 as float64 (applied to
 * the modulus of the difference).
 */
export const DEFAULT_TOLERANCE: Readonly<Record<DType, Tolerance>> = Object.freeze({
  float64: Object.freeze({ rtol: 1e-7, atol: 1e-7 }),
  float32: Object.freeze({ rtol: 1.3e-6, atol: 1e-5 }),
  int32: Object.freeze({ rtol: 0, atol: 0 }),
  bool: Object.freeze({ rtol: 0, atol: 0 }),
  complex128: Object.freeze({ rtol: 1e-7, atol: 1e-7 }),
})

/** The default tolerance for a dtype (float64 when omitted). */
export function tolerance(dtype: DType = 'float64'): Tolerance {
  return DEFAULT_TOLERANCE[dtype]
}
