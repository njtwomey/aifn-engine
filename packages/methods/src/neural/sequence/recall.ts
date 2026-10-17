/**
 * Multi-query associative recall (MQAR) and a probe of what fixed-size memories can hold.
 *
 * `multiQueryAssociativeRecall` draws $N$ key–value pairs, then every key again as a query; the answer to a query is
 * the value that followed the key (Arora et al., 2023, "Zoology", §3). Here all queries come after all pairs (Zoology
 * interleaves them in some variants). The generator lives here, beside the probe that calls it, and
 * `aifn-methods/data/synthetic` re-exports it with the other synthetic sequence tasks (`selectiveCopy`,
 * `parityStrings`).
 *
 * `associativeRecallByDimension` measures the MQAR accuracy of three memories with idealised projections. Keys and
 * values are random unit vectors in $\reals^d$; the queries are the keys themselves. Softmax attention reads the cache
 * of all $N$ pairs; linear attention (identity feature map, no normaliser) reads $\Smat = \sum_i \vvec_i\kvec_i^\top$;
 * the delta rule (strength $\beta$, unit keys) reads its final state. A read is decoded to the value token whose
 * embedding has the largest dot product with it. Nothing is trained: the probe measures what each memory can hold when
 * the projections are ideal, not what training finds. It is a plain function of a stream and options, so a figure can
 * run it in the compute worker (about 0.2 s for 16 pairs, 11 dimensions and 8 trials).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, choice, integers, normals, permutation, type Stream } from 'aifn-compute/foundation/random'
import { argmax, div, einsum, norm, slice, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { scaledDotProductAttention } from 'aifn-compute/nn/attention'
import { deltaRule } from 'aifn-compute/nn/sequence'

/** One MQAR example: $N$ distinct keys, their values, and the queries (the keys in a random order) with answers. */
export type MqarExample = {
  /** The key tokens $k_1, \dots, k_N$, distinct. */
  keys: number[]
  /** The value token after each key (repeats allowed). */
  values: number[]
  /** The keys again, in a random order. */
  queries: number[]
  /** The value of each query's key. */
  answers: number[]
}

/** Options of `multiQueryAssociativeRecall`. */
export type MqarOptions = {
  /** The number of key–value pairs $N$. */
  pairs: number
  /** The number of distinct key tokens to draw from (at least $N$). */
  keyVocab: number
  /** The number of distinct value tokens to draw from. */
  valueVocab: number
}

/**
 * One multi-query associative recall example: $N$ distinct keys from `keyVocab` (drawn without replacement from the
 * child stream `keys`), a value for each from `valueVocab` (repeats allowed; child `values`), and every key once more
 * as a query, in a random order (child `queries`). Throws `DomainError` when `pairs` exceeds `keyVocab`.
 *
 * @param s The stream; only its children `keys`, `values` and `queries` are drawn from.
 * @param options The number of pairs and the two vocabulary sizes.
 * @returns The keys, values, queries and answers, as token indices.
 *
 * @example Eight pairs over 26 keys and 10 values
 * const m = multiQueryAssociativeRecall(stream(1), { pairs: 8, keyVocab: 26, valueVocab: 10 })
 * print('keys:', m.keys)
 * print('values:', m.values)
 * print('queries:', m.queries)
 * print('answers:', m.answers)
 */
export function multiQueryAssociativeRecall(s: Stream, options: MqarOptions): MqarExample {
  const { pairs, keyVocab, valueVocab } = options
  if (pairs > keyVocab)
    throw new DomainError('multiQueryAssociativeRecall', 'multiQueryAssociativeRecall: pairs must not exceed keyVocab')
  const keys = toFlat(choice(child(s, 'keys'), keyVocab, pairs, { replace: false }))
  const values = toFlat(integers(child(s, 'values'), valueVocab, { shape: [pairs] }) as Tensor)
  const order = toFlat(permutation(child(s, 'queries'), pairs))
  return { keys, values, queries: order.map((i) => keys[i]), answers: order.map((i) => values[i]) }
}

/** Options of `associativeRecallByDimension`. */
export type RecallProbeOptions = {
  /** The number of key–value pairs $N$ written before the queries. */
  pairs: number
  /** The key and value dimensions $d$ to try. */
  dims: readonly number[]
  /** The number of random sequences per dimension. */
  trials: number
  /** The number of value tokens the reads are decoded against. */
  valueVocab: number
  /** The scale of the attention scores (a sharp softmax, as trained recall heads have). */
  scale: number
  /** The writing strength $\beta$ of the delta rule. */
  beta: number
}

/** MQAR accuracy (the fraction of queries answered correctly) per dimension $d$, for each of the three memories. */
export type RecallProbeResult = {
  /** The dimensions tried, in order. */
  dims: number[]
  /** Softmax attention's accuracy at each dimension. */
  attention: number[]
  /** Linear attention's accuracy at each dimension. */
  linear: number[]
  /** The delta rule's accuracy at each dimension. */
  delta: number[]
}

/**
 * Each row scaled to unit length.
 *
 * @param x A matrix, one vector per row.
 * @returns The rows divided by their norms.
 */
function unitRows(x: Tensor): Tensor {
  return div(x, norm(x, -1, true)) as Tensor
}

/**
 * The MQAR accuracy of softmax attention, linear attention and the delta rule at each key dimension $d$. Trial $r$ at
 * dimension $d$ draws from the child stream $(d, r)$ of `s`: its task, keys and value embeddings from children of
 * that, so a trial does not depend on the dimensions or trials run before it.
 *
 * @param s The stream the trials are drawn from.
 * @param options The number of pairs, the dimensions, the trials, the value vocabulary, the attention scale and the
 *   delta rule's strength.
 * @returns The accuracy of each memory at each dimension.
 *
 * @example Attention recalls at every width; the fixed-size memories need d well above N
 * const r = associativeRecallByDimension(stream(7), {
 *   pairs: 8, dims: [4, 16, 64], trials: 4, valueVocab: 16, scale: 30, beta: 1,
 * })
 * print('attention:', r.attention)
 * print('linear:', r.linear)
 * print('delta:', r.delta)
 */
export function associativeRecallByDimension(s: Stream, options: RecallProbeOptions): RecallProbeResult {
  const { pairs: N, dims, trials, valueVocab, scale, beta } = options
  const out: RecallProbeResult = { dims: [...dims], attention: [], linear: [], delta: [] }
  for (const d of dims) {
    let att = 0
    let lin = 0
    let del = 0
    for (let r = 0; r < trials; r++) {
      const sr = child(s, d, r)
      const example = multiQueryAssociativeRecall(child(sr, 'task'), { pairs: N, keyVocab: N, valueVocab })
      const K = unitRows(normals(child(sr, 'keys'), [N, d]))
      const E = unitRows(normals(child(sr, 'values'), [valueVocab, d]))
      const V = einsum('nv,vd->nd', tensor(oneHot(example.values, valueVocab)), E) as Tensor
      // Every key is read once; the order of the queries does not change any read, since all writes come first.
      const readsAttention = scaledDotProductAttention(K, K, V, { scale }).output as Tensor
      const readsLinear = einsum('ij,je->ie', einsum('id,jd->ij', K, K), V) as Tensor
      const states = deltaRule(K, K, V, beta).states as Tensor
      const S = slice(states, N - 1) as Tensor
      const readsDelta = einsum('ed,id->ie', S, K) as Tensor
      att += correct(readsAttention, E, example.values)
      lin += correct(readsLinear, E, example.values)
      del += correct(readsDelta, E, example.values)
    }
    out.attention.push(att / (trials * N))
    out.linear.push(lin / (trials * N))
    out.delta.push(del / (trials * N))
  }
  return out
}

/**
 * One-hot rows of tokens.
 *
 * @param tokens The token indices.
 * @param vocab The vocabulary size (the row length).
 * @returns One row per token, 1 at the token's index and 0 elsewhere.
 */
function oneHot(tokens: readonly number[], vocab: number): number[][] {
  return tokens.map((t) => Array.from({ length: vocab }, (_, j) => (j === t ? 1 : 0)))
}

/**
 * The number of reads whose nearest value embedding (largest dot product) is the right value token.
 *
 * @param reads The reads, one row per query.
 * @param embeddings The value embeddings, one row per value token.
 * @param values The right value token of each read.
 * @returns How many reads decode to their right value.
 */
function correct(reads: Tensor, embeddings: Tensor, values: readonly number[]): number {
  const guesses = toFlat(argmax(einsum('id,vd->iv', reads, embeddings) as Tensor, -1) as Tensor)
  return guesses.reduce((a, g, i) => a + (g === values[i] ? 1 : 0), 0)
}
