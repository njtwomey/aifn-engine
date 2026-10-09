/**
 * Floating-point constants and default tolerances, defined once for all of aifn (design K §9). Modules import these
 * rather than writing `2 ** -52` or `1e-12` of their own.
 *
 * Tolerances are `{ rtol, atol }` pairs: two values $a$ and $b$ agree when
 * $\lvert a - b \rvert \le \text{atol} + \text{rtol} \cdot \lvert b \rvert$, and an iteration has converged when
 * $\lVert \Delta \rVert \le \text{atol} + \text{rtol} \cdot \lVert \xvec \rVert$.
 */

import type { DType } from './core'

/**
 * Machine epsilon of float64: the gap between 1 and the next representable number,
 * $2^{-52} \approx 2.22 \times 10^{-16}$.
 */
export const EPS = 2 ** -52

/**
 * $\sqrt{\varepsilon} = 2^{-26} \approx 1.49 \times 10^{-8}$ for $\varepsilon$ = `EPS`: the usual relative step of a
 * forward finite difference, and a loose tolerance.
 */
export const SQRT_EPS = 2 ** -26

/**
 * The smallest positive normal float64, $2^{-1022} \approx 2.23 \times 10^{-308}$. Values below it are subnormal and
 * lose precision.
 */
export const TINY = 2 ** -1022

/** Machine epsilon of float32, $2^{-23} \approx 1.19 \times 10^{-7}$. */
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

/**
 * The default tolerance for a dtype (float64 when omitted), from `DEFAULT_TOLERANCE`.
 *
 * @param dtype The dtype of the values being compared.
 * @returns Its `{ rtol, atol }` pair (frozen).
 *
 * @example Tolerances by dtype
 * print('float64:', tolerance())
 * print('float32:', tolerance('float32'))
 * print('int32:', tolerance('int32'))
 */
export function tolerance(dtype: DType = 'float64'): Tolerance {
  return DEFAULT_TOLERANCE[dtype]
}
