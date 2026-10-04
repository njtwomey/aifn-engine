/**
 * Shapley values (Shapley, 1953) of a set function v over d players, and KernelSHAP (Lundberg and Lee, 2017), which
 * explains a model's output f(x) by the Shapley values of the interventional value function
 * v(S) = mean over background rows b of f(x_S, b_S̄) (features in S from x, the rest from b).
 *
 * The Shapley value of player i averages its marginal contribution over coalitions S ∌ i with weight
 * |S|!(d − |S| − 1)!/d!. KernelSHAP finds the same values as the solution of a weighted least-squares problem: fit
 * v(S) − v(∅) ≈ Σ_{i∈S} φᵢ with Σφᵢ = v(all) − v(∅), weighting S by the Shapley kernel
 * π(S) = (d − 1)/(C(d, |S|) |S| (d − |S|)). Over all 2^d − 2 proper coalitions the solution is exact; with sampled
 * coalitions (sizes drawn ∝ the kernel's mass per size, each with its complement) it converges to it.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { lstsq } from 'aifn-compute/numerics/linalg'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A model to explain: a batch of rows [m, d] to one output per row [m]. */
export type ScalarModel = (x: Tensor) => Tensor | ArrayLike<number>

/** Shapley values and the decomposition they give. */
export type Attribution = {
  /** One value per feature [d]. */
  values: Float64Array
  /** v(∅): the expected output (over the background) with no feature known. */
  base: number
  /** v(all) = f(x). base + Σ values = output (exactly for exact methods). */
  output: number
}

const MAX_EXACT = 20

function binomial(n: Size, k: Size): number {
  let r = 1
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i
  return r
}

/** The Shapley kernel weight (d − 1)/(C(d, s) s (d − s)) of a coalition of size s, 0 < s < d (∞ at 0 and d). */
export function shapleyKernelWeight(d: Size, s: Size): number {
  if (s <= 0 || s >= d) return Infinity
  return (d - 1) / (binomial(d, s) * s * (d - s))
}

/**
 * Exact Shapley values of a set function over d ≤ 20 players by enumerating all 2^d coalitions. `value` receives a
 * coalition as a membership mask (`mask[i]` true when player i is in it).
 */
export function exactShapley(value: (mask: readonly boolean[]) => number, d: Size): Attribution {
  if (!(Number.isInteger(d) && d >= 1 && d <= MAX_EXACT))
    throw new DomainError('exactShapley', `exactShapley: d must lie in 1 … ${MAX_EXACT}, got ${d}`)
  const total = 1 << d
  const v = new Float64Array(total)
  for (let c = 0; c < total; c++) v[c] = value(Array.from({ length: d }, (_, i) => ((c >> i) & 1) === 1))
  // Weights by coalition size: s!(d − s − 1)!/d!.
  const weight = Float64Array.from({ length: d }, (_, s) => 1 / (d * binomial(d - 1, s)))
  const phi = new Float64Array(d)
  for (let c = 0; c < total; c++) {
    let size = 0
    for (let i = 0; i < d; i++) size += (c >> i) & 1
    for (let i = 0; i < d; i++) if (!((c >> i) & 1)) phi[i] += weight[size] * (v[c | (1 << i)] - v[c])
  }
  return { values: phi, base: v[0], output: v[total - 1] }
}

/**
 * The interventional value function of `model` at x [d] over `background` rows [b, d]: the mean output when the
 * features in the coalition come from x and the others from each background row. Evaluates b rows per call.
 */
export function interventionalValue(
  model: ScalarModel,
  x: VectorLike,
  background: MatrixLike,
): (mask: readonly boolean[]) => number {
  const xv = dense.toF64(x, 'interventionalValue')
  const bg = dense.toMatrixF64(background, 'interventionalValue')
  const d = xv.length
  if (bg.n !== d)
    throw new ShapeError('interventionalValue', `interventionalValue: x has ${d} features, the background ${bg.n}`)
  return (mask) => {
    const rows = Float64Array.from(bg.data)
    for (let r = 0; r < bg.m; r++) for (let i = 0; i < d; i++) if (mask[i]) rows[r * d + i] = xv[i]
    const out = model(fromData(rows, [bg.m, d]))
    const o = 'shape' in out ? dense.data(out as Tensor) : out
    let s = 0
    for (let r = 0; r < bg.m; r++) s += o[r]
    return s / bg.m
  }
}

/** Options of `kernelShap`. */
export type KernelShapOptions = {
  /**
   * Coalitions to evaluate (default 2d + 2048). When it reaches 2^d − 2 every proper coalition is enumerated with its
   * exact kernel weight and the result is the exact Shapley value; otherwise coalitions are sampled from `stream`.
   */
  samples?: Size
  stream?: Stream
}

/** KernelSHAP attribution of `model` at x [d] against `background` rows [b, d] (see the module comment). */
export function kernelShap(
  model: ScalarModel,
  x: VectorLike,
  background: MatrixLike,
  options: KernelShapOptions = {},
): Attribution & { coalitions: Size; exact: boolean } {
  const v = interventionalValue(model, x, background)
  const d = dense.toF64(x, 'kernelShap').length
  const base = v(new Array<boolean>(d).fill(false))
  const output = v(new Array<boolean>(d).fill(true))
  if (d === 1) return { values: Float64Array.of(output - base), base, output, coalitions: 0, exact: true }
  const budget = options.samples ?? 2 * d + 2048
  const exact = d <= MAX_EXACT && budget >= 2 ** d - 2
  const masks: boolean[][] = []
  const weights: number[] = []
  if (exact) {
    for (let c = 1; c < (1 << d) - 1; c++) {
      const mask = Array.from({ length: d }, (_, i) => ((c >> i) & 1) === 1)
      masks.push(mask)
      weights.push(shapleyKernelWeight(d, mask.filter(Boolean).length))
    }
  } else {
    if (!options.stream) throw new DomainError('kernelShap', 'kernelShap: sampling coalitions needs a stream')
    const s = options.stream
    // Kernel mass per size s: C(d, s) π(s) ∝ 1/(s(d − s)).
    const mass = Float64Array.from({ length: d - 1 }, (_, k) => 1 / ((k + 1) * (d - k - 1)))
    const total = mass.reduce((a, b) => a + b, 0)
    for (let k = 0; k + 1 < budget; k += 2) {
      let u = (uniform(s) as number) * total
      let size = 1
      while (size < d - 1 && u > mass[size - 1]) u -= mass[size++ - 1]
      const order = Array.from({ length: d }, (_, i) => i)
      for (let i = 0; i < size; i++) {
        const j = i + integers(s, d - i)
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      const mask = new Array<boolean>(d).fill(false)
      for (let i = 0; i < size; i++) mask[order[i]] = true
      masks.push(
        mask,
        mask.map((b) => !b),
      )
      weights.push(1, 1)
    }
  }
  // Weighted least squares with Σφ = output − base, eliminating φ_{d−1} (minimum-norm when under-determined).
  const delta = output - base
  const q = d - 1
  const A = new Float64Array(q * q)
  const rhs = new Float64Array(q)
  masks.forEach((mask, k) => {
    const w = weights[k]
    const zl = mask[q] ? 1 : 0
    const y = v(mask) - base - zl * delta
    const row = Float64Array.from({ length: q }, (_, i) => (mask[i] ? 1 : 0) - zl)
    for (let i = 0; i < q; i++) {
      rhs[i] += w * row[i] * y
      for (let j = 0; j < q; j++) A[i * q + j] += w * row[i] * row[j]
    }
  })
  // The pseudo-inverse solution: with few sampled coalitions the normal equations can be singular.
  const head = dense.data(lstsq(fromData(A, [q, q]), fromData(rhs, [q])).x)
  const values = new Float64Array(d)
  values.set(head)
  values[q] = delta - head.reduce((a, b) => a + b, 0)
  return { values, base, output, coalitions: masks.length, exact }
}
