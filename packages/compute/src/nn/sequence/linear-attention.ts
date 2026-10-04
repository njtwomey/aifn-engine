/**
 * Linear attention (Katharopoulos et al., 2020): replace the softmax kernel exp(q·k) by a feature map φ with
 * sim(q, k) = φ(q)·φ(k). Then the causal output
 *
 *   o_t = φ(q_t)ᵀ S_t / φ(q_t)ᵀ z_t,   S_t = Σ_{s≤t} φ(k_s) v_sᵀ,   z_t = Σ_{s≤t} φ(k_s),
 *
 * is a recurrent network whose state is the matrix S_t [d_k, d_v]: constant memory per token, as an RNN, yet trainable
 * in parallel as attention. A decay γ, S_t = γ S_{t−1} + φ(k_t) v_tᵀ, gives retention (Sun et al., 2023), whose
 * parallel form weights score (t, s) by γ^{t−s}.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import {
  add,
  div,
  expandDims,
  fromData,
  matmul,
  mul,
  shapeOfValue,
  squeeze,
  sum,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { elu } from 'aifn-compute/nn/functional'
import { linearRecurrence } from './ssm'

/** The feature map of Katharopoulos et al. (2020): φ(x) = elu(x) + 1, positive everywhere. */
export function eluFeatureMap(x: Value): Value {
  return add(elu(x), 1)
}

/** Options of linear attention. */
export type LinearAttentionOptions = {
  /** φ (default elu + 1). */
  featureMap?: (x: Value) => Value
  /** Causal (each query sees keys up to its own position; default true). */
  causal?: boolean
  /** A decay γ ∈ (0, 1] per step on older keys (retention; default 1, none). */
  decay?: number
  /** Divide by φ(q)ᵀz (default true; retention leaves it out). */
  normalise?: boolean
}

/** The [T, T] weights γ^{t−s} for s ≤ t and 0 above (causal), or γ^{|t−s|}·1 (not causal, γ = 1 only). */
function decayMask(T: Size, decay: number, causal: boolean): Tensor {
  const out = new Float64Array(T * T)
  for (let t = 0; t < T; t++)
    for (let s = 0; s < T; s++) out[t * T + s] = !causal || s <= t ? decay ** Math.abs(t - s) : 0
  return fromData(out, [T, T])
}

/**
 * Linear attention in its parallel (attention) form for q, k [..., T, d_k] and v [..., T, d_v]: the weights
 * φ(Q)φ(K)ᵀ, masked (and decayed) causally, normalised by their row sums, times V. O(T²) like softmax attention, but
 * equal to `linearAttentionRecurrent`, the O(T) form.
 */
export function linearAttention(q: Value, k: Value, v: Value, options: LinearAttentionOptions = {}): Value {
  const { featureMap = eluFeatureMap, causal = true, decay = 1, normalise = true } = options
  const fq = featureMap(q)
  const fk = featureMap(k)
  const r = shapeOfValue(fk).length
  const axes = Array.from({ length: r }, (_, i) => i)
  ;[axes[r - 2], axes[r - 1]] = [axes[r - 1], axes[r - 2]]
  let scores: Value = matmul(fq, transpose(fk, axes))
  const T = shapeOfValue(q).at(-2)!
  if (causal || decay !== 1) scores = mul(scores, decayMask(T, decay, causal))
  const out = matmul(scores, v)
  return normalise ? div(out, sum(scores, -1, true)) : out
}

/** The outputs of `linearAttentionRecurrent`: o [T, d_v], the states S_t [T, d_k, d_v] and normalisers z_t [T, d_k]. */
export type LinearAttentionStates = { output: Value; states: Value; normalisers: Value }

/**
 * Causal linear attention as a recurrent network over q, k [T, d_k] and v [T, d_v]: S_t = γ S_{t−1} + φ(k_t) v_tᵀ and
 * z_t = γ z_{t−1} + φ(k_t) by `linearRecurrence` (an associative scan), then o_t = φ(q_t)ᵀS_t / φ(q_t)ᵀz_t. Every
 * state is returned, so a figure can show the memory the network carries.
 */
export function linearAttentionRecurrent(
  q: Value,
  k: Value,
  v: Value,
  options: Omit<LinearAttentionOptions, 'causal'> = {},
): LinearAttentionStates {
  const { featureMap = eluFeatureMap, decay = 1, normalise = true } = options
  const fq = featureMap(q)
  const fk = featureMap(k)
  const updates = mul(expandDims(fk, 2), expandDims(v, 1))
  const states = linearRecurrence(decay, updates)
  const normalisers = linearRecurrence(decay, fk)
  const num = squeeze(matmul(expandDims(fq, 1), states), 1)
  const output = normalise ? div(num, sum(mul(fq, normalisers), -1, true)) : num
  return { output, states, normalisers }
}
