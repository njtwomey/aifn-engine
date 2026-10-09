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

/**
 * One attention step: the `context` vector $[\dots, d_v]$, the attention `weights` $\alpha_j$ over the $T$ source
 * positions $[\dots, T]$ (summing to one), and the raw `scores` $e_j$ before masking and the softmax $[\dots, T]$.
 */
export type AlignmentResult = { context: Value; weights: Value; scores: Value }

/**
 * An attention mechanism for a recurrent decoder. `init` draws its parameters from a stream; `attend` scores a query
 * $[\dots, d_q]$ (the decoder state) against the keys $[\dots, T, d_k]$ (the encoder states) and returns the weighted
 * sum of the values (default the keys), with the positions where `mask` is zero left out.
 */
export interface SequenceAttention<P> {
  readonly kind: string
  readonly label: string
  init(s: Stream): P
  attend(params: P, query: Value, keys: Value, values?: Value, mask?: Value): AlignmentResult
}

/**
 * The shared end of every score: mask, softmax over the source positions, and the weighted sum of the values.
 *
 * @param scores The scores $e_j$, $[\dots, T]$.
 * @param values The values to average, $[\dots, T, d_v]$.
 * @param mask Which positions may be attended, $[\dots, T]$: where it is zero (false) the score is set to
 *   $-\infty$, so the weight is 0. Left out, every position is attended.
 * @returns The context $\sum_j \alpha_j \vvec_j$, the weights $\alpha = \mathrm{softmax}(e)$ and the unmasked scores.
 */
function align(scores: Value, values: Value, mask?: Value): AlignmentResult {
  const masked = mask === undefined ? scores : where(mask, scores, -Infinity)
  const weights = softmax(masked, { axis: -1 })
  return { context: squeeze(matmul(expandDims(weights, -2), values), -2), weights, scores }
}

/**
 * Parameters of `BahdanauAttention`: `query` $\Wmat_q$ ($d_q \times n$, no bias), `key` $\Wmat_k$ ($d_k \times n$,
 * with bias) and `v` ($n \times 1$).
 */
export type BahdanauParams = { query: LinearParams; key: LinearParams; v: Tensor }

/**
 * Additive attention (Bahdanau, Cho and Bengio, 2015, §3.1 and A.1.2):
 * $e_j = \vvec^\top \tanh(\Wmat_q \svec + \Wmat_k \hvec_j + \bvec)$ for the decoder state $\svec$ and each encoder
 * state $\hvec_j$, $\alpha = \mathrm{softmax}(e)$, context $= \sum_j \alpha_j \hvec_j$. Every weight is
 * Glorot-uniform initialised and the bias zero. Differentiable in the parameters, the query and the keys.
 *
 * @param queryDim The width $d_q$ of the decoder state.
 * @param keyDim The width $d_k$ of the encoder states.
 * @param hidden The width $n$ of the scoring network.
 * @returns The mechanism: `init` draws `BahdanauParams`, and `attend` gives the context and weights.
 *
 * @example Attend over three encoder states
 * const att = BahdanauAttention(2, 3, 4)
 * const p = att.init(stream(0))
 * const keys = normals(stream(1), [3, 3])
 * const { weights, context } = att.attend(p, tensor([1, -1]), keys)
 * print('weights =', weights)
 * print('sum =', sum(weights))
 * print('context =', context)
 * // A mask of zeros leaves positions out.
 * print('masked weights =', att.attend(p, tensor([1, -1]), keys, keys, tensor([1, 1, 0])).weights)
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

/**
 * Luong's scores: `dot` $\svec^\top \hvec$, `general` $\svec^\top \Wmat \hvec$ and `concat`
 * $\vvec^\top \tanh(\Wmat [\svec; \hvec])$.
 */
export type LuongScore = 'dot' | 'general' | 'concat'

/**
 * Parameters of `LuongAttention`: none for `dot`; `weight` $\Wmat$ ($d_k \times d_q$, scoring
 * $\hvec_j^\top \Wmat \svec$) for `general`; `concat` (a weight of $(d_q + d_k) \times n$, no bias) and `v`
 * ($n \times 1$) for `concat`.
 */
export type LuongParams = { weight?: Tensor; concat?: LinearParams; v?: Tensor }

/**
 * Multiplicative (global) attention of Luong, Pham and Manning (2015, §3.1, eq. 8): scores of the current decoder
 * state $\svec$ against each encoder state $\hvec_j$ by `dot` ($d_q = d_k$), `general` or `concat`;
 * $\alpha = \mathrm{softmax}(e)$, context $= \sum_j \alpha_j \hvec_j$. Weights are Glorot-uniform initialised.
 * Throws `ShapeError` for `dot` scores with $d_q \ne d_k$.
 *
 * @param queryDim The width $d_q$ of the decoder state.
 * @param keyDim The width $d_k$ of the encoder states.
 * @param score The score function: `dot`, `general` (a bilinear form) or `concat` (a one-hidden-layer network).
 * @param hidden The width $n$ of the `concat` scoring network (default $d_q$); unused by the other scores.
 * @returns The mechanism: `init` draws `LuongParams`, and `attend` gives the context and weights.
 *
 * @example Dot scores: the query picks out the keys most like it
 * const att = LuongAttention(2, 2, 'dot')
 * const keys = tensor([[1, 0], [0, 1], [1, 1]])
 * const { scores, weights, context } = att.attend(att.init(stream(0)), tensor([2, 0]), keys)
 * print('scores =', scores)
 * print('weights =', weights)
 * print('context =', context)
 *
 * @example General scores between different widths
 * const att = LuongAttention(2, 3, 'general')
 * const p = att.init(stream(0))
 * print('W =', p.weight)
 * print('weights =', att.attend(p, tensor([1, 0]), normals(stream(1), [3, 3])).weights)
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
