/**
 * Gradient checks for the tensor primitives: `checkGradient(f, inputs)` differentiates Σ w·f(inputs) with `grad`
 * (reverse mode through every primitive's vjp) and compares each input's gradient with central finite differences.
 */

import { expect } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import {
  mul,
  size,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

/** Fixed, distinct weights that turn any output into a scalar test function Σ w·f. */
function weights(shape: readonly number[]): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  return tensor(
    Array.from({ length: n }, (_, k) => 0.3 + 0.7 * Math.sin(1.7 * k + 0.4)),
    shape,
  )
}

function scalarise(y: Value): Value {
  const raw = unwrap(y)
  if (typeof raw === 'number') return mul(y, 0.9)
  return sum(mul(y, weights(raw.shape)))
}

function flat(v: Value): number[] {
  const raw = unwrap(v)
  return typeof raw === 'number' ? [raw] : toFlat(raw)
}

/** Replace element k of a raw value. */
function bump(x: Raw, k: number, h: number): Raw {
  if (typeof x === 'number') return x + h
  const values = toFlat(x)
  values[k] += h
  return tensor(values, x.shape)
}

/**
 * Check the vjps of `f` against central finite differences of Σ w·f(inputs), with respect to every input (or those
 * listed in `wrt`). Returns the analytic gradients.
 */
export function checkGradient(
  f: (...xs: Value[]) => Value,
  inputs: Raw[],
  { wrt, eps = 1e-6, tol = 1e-6 }: { wrt?: number[]; eps?: number; tol?: number } = {},
): number[][] {
  const which = wrt ?? inputs.map((_, i) => i)
  const grads = grad((...xs: Value[]) => scalarise(f(...xs)), { argnums: which })(...inputs) as Value[]
  const analytic = grads.map(flat)
  which.forEach((i, j) => {
    const x = inputs[i]
    const n = typeof x === 'number' ? 1 : size(x)
    expect(analytic[j].length, `gradient ${i} size`).toBe(n)
    for (let k = 0; k < n; k++) {
      const at = (h: number) => {
        const xs = inputs.map((v, q) => (q === i ? bump(v, k, h) : v))
        return unwrap(scalarise(f(...xs))) as number
      }
      const numeric = (at(eps) - at(-eps)) / (2 * eps)
      const scale = Math.max(1, Math.abs(numeric))
      expect(
        Math.abs(analytic[j][k] - numeric) / scale,
        `input ${i}, element ${k}: ${analytic[j][k]} vs ${numeric}`,
      ).toBeLessThan(tol)
    }
  })
  return analytic
}
