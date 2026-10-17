/**
 * Permutation-invariant pooling over padded batches of sets, for Deep Sets and PointNet. `padSets` stacks sets of
 * different sizes into a tensor `[B, K, d]` with a mask `[B, K]` (1 for an element, 0 for padding). `setPool` reduces
 * the element axis of per-element features `[B, K, m]` by a masked sum, mean or maximum, giving `[B, m]`; padding never
 * contributes, so the result equals the pool of each set alone, and any permutation of a set's elements gives the same
 * result. A Deep Sets network is $\rho\big(\sum_k \phi(\xvec_k)\big)$ (Zaheer et al. 2017, "Deep Sets", NeurIPS,
 * theorem 2 and §3); PointNet max-pools instead, and `criticalSet` returns the elements that win the maximum in at
 * least one channel, the critical point set that alone determines a max-pooled output (Qi et al. 2017, "PointNet: deep
 * learning on point sets for 3D classification and segmentation", CVPR, §4.3, theorem 2: at most one per channel).
 *
 * The padded layout is what per-element layers want: an `Mlp` applies to `[B, K, d]` row by row. The masked maximum
 * puts $-\infty$ at padded positions (as the masks of `aifn-compute/nn/attention` do). An empty set has no pool and is
 * an error, not a silent 0 or NaN.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  div,
  fromData,
  max,
  mul,
  reshape,
  shapeOfValue,
  sum,
  toFlat,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

/** How `setPool` reduces a set: the sum, the mean or the elementwise maximum of its elements. */
export type SetPooling = 'sum' | 'mean' | 'max'

/** Sets padded to a common size. */
export type PaddedSets = {
  /** The elements, `[B, K, d]`: set $b$'s element $k$ at `[b, k]`, zeros past the set's size. */
  x: Tensor
  /** `[B, K]`: 1 for an element, 0 for padding. */
  mask: Tensor
  /** The size of each set. */
  sizes: number[]
}

/**
 * Pad sets of $d$-dimensional elements to the largest size (or `size`) with zeros, with a mask of the real elements.
 * Throws `ShapeError` when the elements do not all have one length, or a set is larger than `size`.
 *
 * @param sets The sets: `sets[b][k]` is element $k$ of set $b$, a vector of length $d$.
 * @param size The padded size $K$ (default the largest set, at least 1).
 * @returns The padded elements, the mask and the sizes.
 *
 * @example Two sets of 2-d points, of sizes 3 and 1
 * const { x, mask, sizes } = padSets([[[1, 2], [2, 3], [4, 5]], [[-3, -1]]])
 * print('x:', x.shape, ' mask:', mask, ' sizes:', sizes)
 */
export function padSets(sets: readonly (readonly (readonly number[])[])[], size?: number): PaddedSets {
  const d = sets.find((s) => s.length > 0)?.[0].length ?? 1
  const K = size ?? Math.max(1, ...sets.map((s) => s.length))
  const B = sets.length
  const x = new Float64Array(B * K * d)
  const mask = new Float64Array(B * K)
  sets.forEach((s, b) => {
    if (s.length > K) throw new ShapeError('padSets', `padSets: set ${b} has ${s.length} elements, more than ${K}`)
    s.forEach((e, k) => {
      if (e.length !== d) throw new ShapeError('padSets', `padSets: element ${k} of set ${b} has length ${e.length}`)
      x.set(e, (b * K + k) * d)
      mask[b * K + k] = 1
    })
  })
  return { x: fromData(x, [B, K, d]), mask: fromData(mask, [B, K]), sizes: sets.map((s) => s.length) }
}

/**
 * Pool per-element features over each set: the sum $\sum_k \hvec_{bk}$, its mean over the set's elements, or the
 * elementwise maximum, padded elements left out. Differentiable in `h` (the maximum passes the gradient to the
 * winner). Throws `ShapeError` unless `h` is `[B, K, m]` for a `[B, K]` mask, and `DomainError` for an empty set.
 *
 * @param h The per-element features, `[B, K, m]`.
 * @param mask `[B, K]`: 1 for an element and 0 for padding, as `padSets` returns it.
 * @param pooling `sum`, `mean` or `max`.
 * @returns The pooled features, `[B, m]`.
 *
 * @example The three pools of two padded sets
 * const { x, mask } = padSets([[[1, 2], [2, 3], [4, 4]], [[-3, -1]]])
 * print('sum:', setPool(x, mask, 'sum'))
 * print('mean:', setPool(x, mask, 'mean'))
 * print('max:', setPool(x, mask, 'max'))
 */
export function setPool(h: Value, mask: Tensor, pooling: SetPooling): Value {
  const [B, K] = mask.shape
  const s = shapeOfValue(h)
  if (mask.shape.length !== 2 || s.length !== 3 || s[0] !== B || s[1] !== K)
    throw new ShapeError('setPool', `setPool: features must be [${B}, ${K}, m], got [${s.join(', ')}]`, [s, mask.shape])
  const m = toFlat(mask)
  const counts = new Float64Array(B)
  for (let b = 0; b < B; b++) for (let k = 0; k < K; k++) counts[b] += m[b * K + k]
  const empty = counts.findIndex((c) => c === 0)
  if (empty >= 0) throw new DomainError('setPool', `setPool: set ${empty} is empty, so it has no pool`)
  const m3 = reshape(mask, [B, K, 1])
  if (pooling === 'max') return max(where(m3, h, -Infinity), 1)
  const total = sum(mul(h, m3), 1)
  if (pooling === 'sum') return total
  return div(total, fromData(counts, [B, 1]))
}

/**
 * The critical set of one max-pooled set: the indices of the elements that attain the maximum in at least one of the
 * $m$ channels (the first such element on ties), in increasing order. A max-pooled output depends on these elements
 * only, so removing any other element leaves it unchanged; there are at most $m$ of them. Throws `ShapeError` unless
 * `h` is a matrix.
 *
 * @param h The per-element features of one set, `[n, m]`.
 * @returns The critical elements' indices (int32).
 *
 * @example Element 2 wins no channel, so it is not critical
 * print(criticalSet(tensor([[1, 5], [4, 2], [2, 2]])))
 */
export function criticalSet(h: Tensor): Tensor {
  if (h.shape.length !== 2) throw new ShapeError('criticalSet', `criticalSet: expected [n, m], got [${h.shape}]`)
  const [n, m] = h.shape
  const x = toFlat(h)
  const winners = new Set<number>()
  if (n > 0)
    for (let c = 0; c < m; c++) {
      let best = 0
      for (let k = 1; k < n; k++) if (x[k * m + c] > x[best * m + c]) best = k
      winners.add(best)
    }
  return fromData(Int32Array.from([...winners].sort((a, b) => a - b)))
}
