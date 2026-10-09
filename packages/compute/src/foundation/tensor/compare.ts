/** Whole-tensor comparisons for tests and assertions (not primitives: they return booleans). */

import { flatData, type Tensor } from './core'
import { binaryKernel, complexKernel } from './kernels'
import { unwrap, type Value } from './trace'

/** Options for `allclose`. */
export type CloseOptions = {
  /** Relative tolerance (default 1e-5, as NumPy). */
  rtol?: number
  /** Absolute tolerance (default 1e-8, as NumPy). */
  atol?: number
  /** Count NaN as equal to NaN (default false). */
  equalNan?: boolean
}

/**
 * True when $\lvert a - b \rvert \le \text{atol} + \text{rtol} \cdot \lvert b \rvert$ for every broadcast pair, as
 * `np.allclose` (note the asymmetry: $b$ is the reference). Infinities must match exactly. Incompatible shapes are an
 * error. Complex values compare by the modulus of the difference (a real value against a complex one has imaginary
 * part 0).
 *
 * @param a The values to test: a number, a tensor or a traced value (its value is compared).
 * @param b The reference values, broadcast against `a`; the relative tolerance scales with them.
 * @param options The tolerances.
 * @param options.rtol The relative tolerance, a multiple of $\lvert b \rvert$.
 * @param options.atol The absolute tolerance, added to the relative one.
 * @param options.equalNan Whether NaN against NaN counts as close (a complex value is NaN when either part is).
 * @returns Whether every pair is close.
 *
 * @example Close within the default tolerances
 * print('1 vs 1 + 1e-9:', allclose(tensor([1, 2]), tensor([1 + 1e-9, 2])))
 * print('1 vs 1.001:', allclose(tensor([1, 2]), tensor([1.001, 2])))
 * print('1 vs 1.001, rtol 1e-2:', allclose(tensor([1, 2]), tensor([1.001, 2]), { rtol: 1e-2 }))
 *
 * @example NaN, and broadcasting a number
 * print('NaN vs NaN:', allclose(tensor([NaN]), tensor([NaN])))
 * print('with equalNan:', allclose(tensor([NaN]), tensor([NaN]), { equalNan: true }))
 * print('all near 3:', allclose(tensor([3, 3.0000001]), 3))
 */
export function allclose(
  a: Value,
  b: Value,
  { rtol = 1e-5, atol = 1e-8, equalNan = false }: CloseOptions = {},
): boolean {
  const close = (x: number, y: number) => {
    if (x !== x || y !== y) return equalNan && x !== x && y !== y ? 1 : 0
    if (x === y) return 1
    if (!Number.isFinite(x) || !Number.isFinite(y)) return 0
    return Math.abs(x - y) <= atol + rtol * Math.abs(y) ? 1 : 0
  }
  const ra = unwrap(a)
  const rb = unwrap(b)
  if (typeof ra === 'number' && typeof rb === 'number') return close(ra, rb) === 1
  if ([ra, rb].some((v) => typeof v !== 'number' && v.dtype === 'complex128')) {
    const t = complexKernel(
      [ra, rb],
      (o, z) => {
        const [xr, xi, yr, yi] = [z[0], z[1], z[2], z[3]]
        if (xr !== xr || xi !== xi || yr !== yr || yi !== yi)
          o[0] = equalNan && (xr !== xr || xi !== xi) && (yr !== yr || yi !== yi) ? 1 : 0
        else if (xr === yr && xi === yi) o[0] = 1
        else if (![xr, xi, yr, yi].every(Number.isFinite)) o[0] = 0
        else o[0] = Math.hypot(xr - yr, xi - yi) <= atol + rtol * Math.hypot(yr, yi) ? 1 : 0
      },
      true,
    )
    return t.data.every((v) => v === 1)
  }
  return binaryKernel(ra, rb, close, 'int32').data.every((v) => v === 1)
}

/**
 * True when $a$ and $b$ have the same shape and equal elements (NaN equals nothing), as `np.array_equal`. The dtypes
 * may differ. Nothing is broadcast: a number equals only a rank-0 tensor holding it.
 *
 * @param a A number, a tensor or a traced value (its value is compared).
 * @param b The same; a real value against a complex one compares with imaginary parts 0.
 * @returns Whether the shapes and every element are equal.
 *
 * @example Exact equality, across dtypes
 * print('same values:', equal(tensor([1, 2]), tensor([1, 2], undefined, 'int32')))
 * print('different shape:', equal(tensor([1, 2]), tensor([[1, 2]])))
 * print('NaN:', equal(tensor([NaN]), tensor([NaN])))
 */
export function equal(a: Value, b: Value): boolean {
  const ra = unwrap(a)
  const rb = unwrap(b)
  if (typeof ra === 'number' || typeof rb === 'number') {
    if (typeof ra === 'number' && typeof rb === 'number') return ra === rb
    const t = (typeof ra === 'number' ? rb : ra) as Tensor
    const x = (typeof ra === 'number' ? ra : rb) as number
    if (t.shape.length !== 0) return false
    const v = flatData(t)
    return v[0] === x && (t.dtype !== 'complex128' || v[1] === 0)
  }
  if (ra.shape.length !== rb.shape.length || ra.shape.some((d, k) => d !== rb.shape[k])) return false
  // A real tensor against a complex one compares as complex (imaginary parts 0), so both read (re, im) pairs.
  const field = ra.dtype === 'complex128' || rb.dtype === 'complex128' ? 'complex128' : undefined
  const fa = flatData(ra, field ?? ra.dtype)
  const fb = flatData(rb, field ?? rb.dtype)
  for (let k = 0; k < fa.length; k++) if (fa[k] !== fb[k]) return false
  return true
}
