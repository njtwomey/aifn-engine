/**
 * Gradient checking: compare reverse-mode gradients with central finite differences, element by element.
 */

import { fromData, isTensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { grad } from './transforms'
import { treeFlatten, treeUnflatten } from 'aifn-compute/foundation/pytree'

/** Options of `gradCheck`. */
export type GradCheckOptions = {
  /** Finite-difference step, scaled by max(1, |xᵢ|) per element (default 1e-6). */
  eps?: number
  /** Relative tolerance (default 1e-5). */
  rtol?: number
  /** Absolute tolerance (default 1e-7). */
  atol?: number
}

/** One checked element of x. */
export type GradCheckEntry = {
  /** The leaf's path in x (`x`, `x[1]`, `w.bias`). */
  path: string
  /** Flat (row-major) index within the leaf. */
  index: number
  analytic: number
  numeric: number
  absError: number
  /** |analytic − numeric| / max(|analytic|, |numeric|), or 0 when both are 0. */
  relError: number
  /** |analytic − numeric| ≤ atol + rtol·|numeric|. */
  ok: boolean
}

/** The report of `gradCheck`. */
export type GradCheckReport = {
  ok: boolean
  maxAbsError: number
  maxRelError: number
  entries: GradCheckEntry[]
}

/**
 * Check `grad(f)` at x against central differences (f(x + hεᵢ) − f(x − hεᵢ)) / 2h for every element of x (a number,
 * tensor or pytree). Central differences have truncation error O(h²) and rounding error O(ε_machine/h), so h ≈ 1e-6
 * balances them for smooth f of values of order one. The report lists every element and never throws for a mismatch.
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

function flatNumbers(x: number | Tensor): number[] {
  return isTensor(x) ? toFlat(x) : [x]
}
