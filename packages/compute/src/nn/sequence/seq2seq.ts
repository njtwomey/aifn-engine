/**
 * Attention for recurrent encoder–decoders (sequence to sequence): at each decoder step a query (the decoder state)
 * scores every encoder state, the scores are normalised by a softmax over the source positions, and the context is
 * the weighted sum of the encoder states. Bahdanau et al. (2015) score with a one-hidden-layer network on the previous
 * decoder state (additive attention); Luong, Pham and Manning (2015) score with a dot product, a bilinear form
 * (`general`) or a network on the concatenation (`concat`), from the current decoder state.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import { softmax } from 'aifn-compute/numerics/special'
import {
  add,
  broadcastTo,
  concat,
  expandDims,
  matmul,
  shapeOfValue,
  squeeze,
  tanh,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Linear, linear, type LinearParams } from 'aifn-compute/nn/layers'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The context vector [..., d_v] and the attention weights over the source positions [..., T]. */
export type AlignmentResult = { context: Value; weights: Value; scores: Value }

/**
 * An attention mechanism for a recurrent decoder: parameters from `init`, and `attend(params, query [..., d_q],
 * keys [..., T, d_k], values = keys, mask [..., T])` → context and weights.
 */
export interface SequenceAttention<P> {
  readonly kind: string
  readonly label: string
  init(s: Stream): P
  attend(params: P, query: Value, keys: Value, values?: Value, mask?: Value): AlignmentResult
}

function align(scores: Value, values: Value, mask?: Value): AlignmentResult {
  const masked = mask === undefined ? scores : where(mask, scores, -Infinity)
  const weights = softmax(masked, { axis: -1 })
  return { context: squeeze(matmul(expandDims(weights, -2), values), -2), weights, scores }
}

/** Parameters of `BahdanauAttention`: W_q [d_q, n], W_k [d_k, n] (with bias) and v [n, 1]. */
export type BahdanauParams = { query: LinearParams; key: LinearParams; v: Tensor }

/**
 * Additive attention (Bahdanau, Cho and Bengio, 2015, §3.1 and A.1.2): e_j = vᵀ tanh(W_q s + W_k h_j) for the decoder
 * state s and each encoder state h_j, α = softmax(e), context = Σ_j α_j h_j. `hidden` is the width n of the scoring
 * network.
 */
export function BahdanauAttention(queryDim: Size, keyDim: Size, hidden: Size): SequenceAttention<BahdanauParams> {
  const q = Linear(queryDim, hidden, { bias: false, init: xavierUniform() })
  const k = Linear(keyDim, hidden, { init: xavierUniform() })
  const v = Linear(hidden, 1, { bias: false, init: xavierUniform() })
  return {
    kind: 'BahdanauAttention',
    label: `BahdanauAttention(${queryDim}, ${keyDim} → ${hidden})`,
    init: (s) => ({
      query: q.init(child(s, 'query')),
      key: k.init(child(s, 'key')),
      v: v.init(child(s, 'v')).weight,
    }),
    attend: (p, query, keys, values = keys, mask) => {
      const hidden = tanh(add(expandDims(linear(query, p.query.weight), -2), linear(keys, p.key.weight, p.key.bias)))
      const scores = squeeze(matmul(hidden, p.v), -1)
      return align(scores, values, mask)
    },
  }
}

/** Luong's scores: `dot` sᵀh, `general` sᵀW h, `concat` vᵀ tanh(W[s; h]). */
export type LuongScore = 'dot' | 'general' | 'concat'

/** Parameters of `LuongAttention`: W for `general` [d_k, d_q], or W [d_q + d_k, n] and v [n, 1] for `concat`. */
export type LuongParams = { weight?: Tensor; concat?: LinearParams; v?: Tensor }

/**
 * Multiplicative (global) attention of Luong, Pham and Manning (2015, §3.1, eq. 8): scores of the current decoder
 * state s against each encoder state h_j by `dot` (d_q = d_k), `general` or `concat`; α = softmax(scores), context =
 * Σ_j α_j h_j.
 */
export function LuongAttention(
  queryDim: Size,
  keyDim: Size,
  score: LuongScore = 'general',
  hidden: Size = queryDim,
): SequenceAttention<LuongParams> {
  if (score === 'dot' && queryDim !== keyDim)
    throw new ShapeError(
      'LuongAttention',
      `LuongAttention: dot scores need equal widths, got ${queryDim} and ${keyDim}`,
    )
  const general = Linear(keyDim, queryDim, { bias: false, init: xavierUniform() })
  const cat = Linear(queryDim + keyDim, hidden, { bias: false, init: xavierUniform() })
  const v = Linear(hidden, 1, { bias: false, init: xavierUniform() })
  return {
    kind: 'LuongAttention',
    label: `LuongAttention(${score})`,
    init: (s) =>
      score === 'dot'
        ? {}
        : score === 'general'
          ? { weight: general.init(child(s, 'weight')).weight }
          : { concat: cat.init(child(s, 'concat')), v: v.init(child(s, 'v')).weight },
    attend: (p, query, keys, values = keys, mask) => {
      let scores: Value
      if (score === 'dot') scores = squeeze(matmul(keys, expandDims(query, -1)), -1)
      else if (score === 'general') scores = squeeze(matmul(linear(keys, p.weight!), expandDims(query, -1)), -1)
      else {
        const ks = shapeOfValue(keys)
        const qs = broadcastTo(expandDims(query, -2), [...ks.slice(0, -1), shapeOfValue(query).at(-1)!])
        scores = squeeze(matmul(tanh(linear(concat([qs, keys], -1), p.concat!.weight)), p.v!), -1)
      }
      return align(scores, values, mask)
    },
  }
}
