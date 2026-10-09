/**
 * The private aggregation step of DP-SGD (Abadi et al., 2016): clip each example's gradient to $L_2$ norm at most $C$
 * (the global norm over every parameter leaf), sum, add $\Gauss(0, \sigma^2 C^2)$ to every coordinate and divide by
 * the expected batch size. The sum of clipped gradients has $L_2$ sensitivity $C$ to adding or removing one example, so
 * the noisy sum is the Gaussian mechanism with noise multiplier $\sigma$. The privacy spent is counted by
 * `dpSgdEpsilon`.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { treeFlatten, treeUnflatten } from 'aifn-compute/foundation/pytree'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The result of `clipAndNoise`. */
export type PrivateGradient<P> = {
  /** The noisy mean gradient, shaped as one example's gradient. */
  gradient: P
  /** Each example's gradient norm before clipping, $B$ values. */
  norms: Float64Array
  /** Share of examples whose gradient was clipped (norm above $C$). */
  clippedShare: number
}

/**
 * Clip, sum and noise per-example gradients: each example's gradient $\gvec_b$ is scaled by
 * $\min(1, C / \lVert \gvec_b \rVert)$, the scaled gradients are summed, and $\Gauss(0, \sigma^2 C^2)$ noise is added
 * to every coordinate before dividing by the denominator. Not differentiable (it reads the leaves' values). Throws a
 * `DomainError` for a clipping norm that is not positive, a negative noise multiplier or a pytree without leaves,
 * and a `ShapeError` when the leaves' batch axes differ.
 *
 * @param perExample A pytree of tensors whose leaves carry a leading batch axis of length $B$, one gradient per
 *   example (as `vmap(grad(loss))` returns them).
 * @param clipNorm The clipping norm $C > 0$, on the global $L_2$ norm of an example's gradient over all leaves.
 * @param noiseMultiplier The noise multiplier $\sigma \ge 0$: the noise's standard deviation is $\sigma C$ (0 adds
 *   none).
 * @param stream The random stream the noise is drawn from.
 * @param denominator What the noisy sum is divided by: the expected batch size under Poisson sampling. Defaults to
 *   $B$.
 * @returns The noisy mean `gradient`, shaped as one example's gradient, with the norms before clipping and the share
 *   clipped.
 *
 * @example Two examples, one clipped, without noise
 * const g = { w: tensor([[3, 4], [0.6, 0.8]]) }
 * const r = clipAndNoise(g, 1, 0, stream(0))
 * print('gradient:', r.gradient.w)
 * print('norms:', r.norms, 'clipped share:', r.clippedShare)
 *
 * @example With noise multiplier 1
 * const g = { w: tensor([[3, 4], [0.6, 0.8]]) }
 * print('gradient:', clipAndNoise(g, 1, 1, stream(0)).gradient.w)
 */
export function clipAndNoise<P>(
  perExample: P,
  clipNorm: number,
  noiseMultiplier: number,
  stream: Stream,
  denominator?: Size,
): PrivateGradient<P> {
  if (!(clipNorm > 0)) throw new DomainError('clipAndNoise', 'clipAndNoise: the clipping norm must be positive')
  if (!(noiseMultiplier >= 0))
    throw new DomainError('clipAndNoise', 'clipAndNoise: the noise multiplier must be non-negative')
  const flat = treeFlatten(perExample)
  const leaves = flat.leaves.map((l) => l as Tensor)
  if (leaves.length === 0) throw new DomainError('clipAndNoise', 'clipAndNoise: no gradient leaves')
  const B = leaves[0].shape[0]
  const data = leaves.map((l) => {
    if (l.shape[0] !== B) throw new ShapeError('clipAndNoise', 'clipAndNoise: every leaf needs the same batch axis')
    return dense.data(l)
  })
  const sizes = data.map((v) => v.length / B)
  const norms = new Float64Array(B)
  data.forEach((v, k) => {
    for (let b = 0; b < B; b++) for (let j = 0; j < sizes[k]; j++) norms[b] += v[b * sizes[k] + j] ** 2
  })
  let clipped = 0
  const factor = Float64Array.from(norms, (s, b) => {
    norms[b] = Math.sqrt(s)
    if (norms[b] > clipNorm) clipped++
    return Math.min(1, clipNorm / Math.max(norms[b], Number.MIN_VALUE))
  })
  const scale = 1 / (denominator ?? B)
  const out = data.map((v, k) => {
    const m = sizes[k]
    const sum = new Float64Array(m)
    for (let b = 0; b < B; b++) for (let j = 0; j < m; j++) sum[j] += factor[b] * v[b * m + j]
    const noise = toFlat(normal(stream, 0, noiseMultiplier * clipNorm, { shape: [m] }))
    for (let j = 0; j < m; j++) sum[j] = (sum[j] + noise[j]) * scale
    return fromData(sum, leaves[k].shape.slice(1))
  })
  return { gradient: treeUnflatten<P>(flat.treedef, out), norms, clippedShare: clipped / B }
}
