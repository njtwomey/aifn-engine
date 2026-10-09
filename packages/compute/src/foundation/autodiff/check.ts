/**
 * Gradient checking: compare reverse-mode gradients with central finite differences, element by element.
 *
 * Each element $x_i$ is perturbed in turn by a step $h = \epsilon \max(1, \lvert x_i \rvert)$, and the derivative
 * estimated as $(f(\xvec + h\evec_i) - f(\xvec - h\evec_i)) / 2h$. A mismatch is reported, never thrown.
 */

import { fromData, isTensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { grad } from './transforms'
import { treeFlatten, treeUnflatten } from 'aifn-compute/foundation/pytree'

/** Options of `gradCheck`. */
export type GradCheckOptions = {
  /** Finite-difference step $\epsilon$, scaled by $\max(1, \lvert x_i \rvert)$ per element (default 1e-6). */
  eps?: number
  /** Relative tolerance $r$ (default 1e-5): see `GradCheckEntry.ok`. */
  rtol?: number
  /** Absolute tolerance $a$ (default 1e-7): see `GradCheckEntry.ok`. */
  atol?: number
}

/** One checked element of x. */
export type GradCheckEntry = {
  /** The leaf's path in x (`x`, `x[1]`, `w.bias`). */
  path: string
  /** Flat (row-major) index within the leaf. */
  index: number
  /** The derivative from `grad`. */
  analytic: number
  /** The central-difference estimate. */
  numeric: number
  /** $\lvert g_a - g_n \rvert$, with $g_a$ the analytic and $g_n$ the numeric derivative. */
  absError: number
  /** $\lvert g_a - g_n \rvert / \max(\lvert g_a \rvert, \lvert g_n \rvert)$, or 0 when both are 0. */
  relError: number
  /** Whether $\lvert g_a - g_n \rvert \le a + r \lvert g_n \rvert$, with $a$ = `atol` and $r$ = `rtol`. */
  ok: boolean
}

/** The report of `gradCheck`. */
export type GradCheckReport = {
  /** True when every entry is `ok`. */
  ok: boolean
  /** The largest `absError` over the entries (0 when there are none). */
  maxAbsError: number
  /** The largest `relError` over the entries (0 when there are none). */
  maxRelError: number
  /** One entry per element of x, leaf by leaf in flattening order and row-major within a leaf. */
  entries: GradCheckEntry[]
}

/**
 * Check `grad(f)` at x against central differences $(f(\xvec + h\evec_i) - f(\xvec - h\evec_i)) / 2h$ for every
 * element of x (a number, tensor or pytree). Central differences have truncation error $O(h^2)$ and rounding error
 * $O(\epsilon_{\text{mach}} / h)$, so $h \approx 10^{-6}$ balances them for smooth f of values of order one. The
 * report lists every element and never throws for a mismatch. It costs $2n$ evaluations of f for $n$ elements.
 *
 * @param f A scalar function of x, returning a number or a rank-0 tensor. It is differentiated once by `grad`, then
 *   called on raw values for the differences.
 * @param x The point to check at: a number, a tensor or a pytree of them. Traced leaves are read through their values.
 * @param options The step and the tolerances; see `GradCheckOptions`.
 * @returns Whether every element agreed, the largest errors, and an entry per element.
 *
 * @example A correct gradient
 * const report = gradCheck((x) => sum(mul(x, sin(x))), tensor([0.5, 1, 2]))
 * print('ok =', report.ok)
 * print('max abs error =', report.maxAbsError)
 * print('first entry =', report.entries[0])
 *
 * @example A wrong custom rule is caught
 * // The backward rule forgets the factor 2 of d(x^2)/dx.
 * const wrong = customVjp((x) => mul(x, x), (x) => ({ out: mul(x, x), residuals: x }), (x, g) => [mul(g, x)])
 * const report = gradCheck(wrong, 3)
 * print('ok =', report.ok)
 * print('analytic', report.entries[0].analytic, 'numeric', report.entries[0].numeric)
 */
export function gradCheck<T>(f: (x: T) => Value, x: T, options: GradCheckOptions = {}): GradCheckReport {
  const { eps = 1e-6, rtol = 1e-5, atol = 1e-7 } = options
  const flat = treeFlatten(x, 'x')
  const analytic = treeFlatten(grad(f)(x)).leaves.map((g) => flatNumbers(unwrap(g)))
  const base = flat.leaves.map((leaf) => flatNumbers(unwrap(leaf)))
  const entries: GradCheckEntry[] = []
  const evaluate = (values: number[][]): number => {
    const leaves = flat.leaves.map((leaf, i) => {
      const raw = unwrap(leaf)
      return typeof raw === 'number' ? values[i][0] : fromData(new Float64Array(values[i]), raw.shape)
    })
    const y = unwrap(f(treeUnflatten(flat.treedef, leaves) as T))
    return typeof y === 'number' ? y : toFlat(y)[0]
  }
  base.forEach((values, i) => {
    values.forEach((v, k) => {
      const h = eps * Math.max(1, Math.abs(v))
      const at = (d: number) => evaluate(base.map((b, j) => (j === i ? b.map((u, q) => (q === k ? v + d : u)) : b)))
      const numeric = (at(h) - at(-h)) / (2 * h)
      const a = analytic[i][k]
      const absError = Math.abs(a - numeric)
      const scale = Math.max(Math.abs(a), Math.abs(numeric))
      const relError = scale === 0 ? 0 : absError / scale
      entries.push({
        path: flat.paths[i],
        index: k,
        analytic: a,
        numeric,
        absError,
        relError,
        ok: absError <= atol + rtol * Math.abs(numeric),
      })
    })
  })
  return {
    ok: entries.every((e) => e.ok),
    maxAbsError: Math.max(0, ...entries.map((e) => e.absError)),
    maxRelError: Math.max(0, ...entries.map((e) => e.relError)),
    entries,
  }
}

/**
 * The entries of a raw leaf as a plain array: the tensor's row-major entries, or the number alone.
 *
 * @param x A number or a tensor (not traced).
 * @returns Its entries, in row-major order.
 */
function flatNumbers(x: number | Tensor): number[] {
  return isTensor(x) ? toFlat(x) : [x]
}
