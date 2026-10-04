/**
 * Multiple-instance learning with attention pooling (Ilse, Tomczak and Welling, 2018): a bag of instances is labelled
 * positive when at least one instance is; the model embeds each instance, h = tanh(W x + b), weighs the instances of a
 * bag by attention a_i = softmax_bag(wᵀ tanh(V h_i)), pools z = Σ a_i h_i and classifies the bag by σ(cᵀz + c₀). The
 * attention weights point at the instances that made the bag positive.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { child, normal, stream as makeStream } from 'aifn-compute/foundation/random'
import {
  add,
  dense,
  fromData,
  matmul,
  reshape,
  tanh,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { methodTraining, type TrainingMethod } from 'aifn-compute/nn/training'
import { sigmoid, softmax } from 'aifn-compute/numerics/special'

/** Parameters of the attention-MIL model. */
export type AttentionMilParams = { W: Tensor; b: Tensor; V: Tensor; w: Tensor; c: Tensor; c0: Tensor }

/** Options of `attentionMil`. */
export type AttentionMilOptions = {
  /** Embedding width H (default 8) and attention width (default 8). */
  hidden?: Size
  attention?: Size
  /** Training steps (default 200) and method (default full-batch L-BFGS). */
  steps?: Size
  method?: TrainingMethod
  seed?: number | string
}

/** A trained attention-MIL model. */
export type AttentionMil = {
  params: AttentionMilParams
  /** P(bag positive) for bags of new instances. */
  predict: (x: MatrixLike, bags: ArrayLike<number>, count: Size) => { bags: Float64Array; attention: Float64Array }
  losses: number[]
}

/** The averaging pattern of bags as an additive mask [B, N]: 0 where instance i is in bag b, −10⁹ elsewhere. */
function bagMask(bags: ArrayLike<number>, count: Size): Tensor {
  const N = bags.length
  const mask = new Float64Array(count * N).fill(-1e9)
  for (let i = 0; i < N; i++) mask[bags[i] * N + i] = 0
  return fromData(mask, [count, N])
}

function forward(p: AttentionMilParams, X: Value, mask: Tensor) {
  const h = tanh(add(matmul(X, p.W), p.b)) // [N, H]
  const [B, N] = mask.shape
  const scores = reshape(matmul(tanh(matmul(h, p.V)), p.w), [1, N])
  const attention = softmax(add(mask, scores)) // [B, N], rows sum to 1 over each bag's instances
  const z = matmul(attention, h) // [B, H]
  const logits = reshape(add(matmul(z, p.c), p.c0), [B])
  return { logits, attention }
}

/**
 * Train attention-MIL on instances x [N, d] grouped into bags (`bags`: each instance's bag index in 0 … B − 1) with
 * bag labels in {0, 1}, by binary cross-entropy on the bag predictions.
 */
export function attentionMil(
  x: MatrixLike,
  bags: ArrayLike<number>,
  bagLabels: ArrayLike<number>,
  options: AttentionMilOptions = {},
): AttentionMil {
  const { hidden: H = 8, attention: A = 8, steps = 200, method = { method: 'lbfgs' }, seed = 0 } = options
  const m = dense.toMatrixF64(x, 'attentionMil')
  const X = fromData(Float64Array.from(m.data), [m.m, m.n])
  const B = bagLabels.length
  const mask = bagMask(bags, B)
  const y = fromData(Float64Array.from(bagLabels), [B])
  const root = makeStream(seed)
  const init = (s = child(root, 'init')): AttentionMilParams => ({
    W: normal(child(s, 'W'), 0, 1 / Math.sqrt(m.n), { shape: [m.n, H] }) as Tensor,
    b: fromData(new Float64Array(H), [H]),
    V: normal(child(s, 'V'), 0, 1 / Math.sqrt(H), { shape: [H, A] }) as Tensor,
    w: normal(child(s, 'w'), 0, 1 / Math.sqrt(A), { shape: [A, 1] }) as Tensor,
    c: normal(child(s, 'c'), 0, 1 / Math.sqrt(H), { shape: [H, 1] }) as Tensor,
    c0: fromData(Float64Array.of(0), [1]),
  })
  const alg = methodTraining<AttentionMilParams, Record<string, Tensor>>(
    (p) => binaryCrossEntropyWithLogits(forward(p, X, mask).logits, y),
    { x: X },
    method,
  )
  let s = alg.init({ params: init() }, child(root, 'train'))
  const losses = [s.loss]
  for (let t = 0; t < steps && !s.stopped; t++) {
    s = alg.step(s, { t, stream: child(root, 'step', t) })
    losses.push(s.loss)
  }
  const params = s.params
  return {
    params,
    losses,
    predict: (q, qb, count) => {
      const qm = dense.toMatrixF64(q, 'attentionMil.predict')
      const r = forward(params, fromData(Float64Array.from(qm.data), [qm.m, qm.n]), bagMask(qb, count))
      // Each instance's weight within its own bag.
      const att = toFlat(transpose(r.attention as Tensor))
      const weights = Float64Array.from({ length: qm.m }, (_, i) => att[i * count + qb[i]])
      return { bags: Float64Array.from(toFlat(sigmoid(r.logits) as Tensor)), attention: weights }
    },
  }
}
