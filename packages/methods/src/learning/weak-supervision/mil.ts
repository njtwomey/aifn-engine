/**
 * Multiple-instance learning with attention pooling (Ilse, Tomczak and Welling, 2018): a bag of instances is labelled
 * positive when at least one instance is; the model embeds each instance, $\hvec = \tanh(\Wmat^\top \xvec + \bvec)$,
 * weighs the instances of a bag by attention $a_i = \operatorname{softmax}_{\text{bag}}(\wvec^\top
 * \tanh(\Vmat^\top \hvec_i))$, pools $\zvec = \sum_i a_i \hvec_i$ and classifies the bag by
 * $\sigma(\cvec^\top \zvec + c_0)$. The attention weights point at the instances that made the bag positive.
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

/**
 * Parameters of the attention-MIL model: the embedding $\Wmat$ ($d \times H$) and $\bvec$ ($H$), the attention
 * $\Vmat$ ($H \times A$) and $\wvec$ ($A \times 1$), and the bag classifier $\cvec$ ($H \times 1$) and $c_0$ (one
 * value).
 */
export type AttentionMilParams = { W: Tensor; b: Tensor; V: Tensor; w: Tensor; c: Tensor; c0: Tensor }

/** Options of `attentionMil`. */
export type AttentionMilOptions = {
  /** Embedding width $H$ (default 8). */
  hidden?: Size
  /** Attention width $A$ (default 8). */
  attention?: Size
  /** The most training steps (default 200); training ends sooner when the method stops. */
  steps?: Size
  /** The training method (default full-batch L-BFGS). */
  method?: TrainingMethod
  /** Seed of the initial weights and of the training streams (default 0). */
  seed?: number | string
}

/** A trained attention-MIL model. */
export type AttentionMil = {
  /** The trained parameters. */
  params: AttentionMilParams
  /**
   * $p(\text{bag positive})$ for bags of new instances: given the instances (the rows of `x`), each one's bag index and
   * the number of bags, it returns `bags`, one probability per bag, and `attention`, each instance's weight within its
   * own bag.
   */
  predict: (x: MatrixLike, bags: ArrayLike<number>, count: Size) => { bags: Float64Array; attention: Float64Array }
  /** The training loss at the start and after each step. */
  losses: number[]
}

/**
 * The averaging pattern of bags as an additive mask $B \times N$: 0 where instance $i$ is in bag $b$, $-10^9$
 * elsewhere, so that a softmax over a row after adding it spreads over that bag's instances only.
 *
 * @param bags Each instance's bag index in $0, \dots, B - 1$.
 * @param count The number of bags $B$.
 * @returns The mask, $B \times N$.
 */
function bagMask(bags: ArrayLike<number>, count: Size): Tensor {
  const N = bags.length
  const mask = new Float64Array(count * N).fill(-1e9)
  for (let i = 0; i < N; i++) mask[bags[i] * N + i] = 0
  return fromData(mask, [count, N])
}

/**
 * The model's forward pass on every bag at once.
 *
 * @param p The parameters.
 * @param X The instances, $N \times d$.
 * @param mask The bag mask of `bagMask`, $B \times N$.
 * @returns `logits`, one per bag ($B$ values), and `attention`, $B \times N$, each row the weights of one bag's
 *   instances (0 elsewhere).
 */
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
 * Train attention-MIL on instances grouped into bags with bag labels in $\{0, 1\}$, by binary cross-entropy on the bag
 * predictions. The weights start from a seeded normal draw scaled by the inverse square root of each layer's input
 * width. Deterministic for a given `seed`.
 *
 * @param x The instances, $N \times d$.
 * @param bags Each instance's bag index in $0, \dots, B - 1$.
 * @param bagLabels Each bag's label, 0 or 1; their number is the number of bags $B$.
 * @param options The widths, the training and the seed.
 * @returns The trained model with its losses and a `predict` for new bags.
 *
 * @example Positive bags hold one instance far from the rest; attention finds it
 * const s = stream(11)
 * const [x, bags, labels] = [[], [], []]
 * for (let b = 0; b < 12; b++) {
 *   labels.push(b % 2)
 *   for (let i = 0; i < 4; i++) {
 *     const key = b % 2 === 1 && i === b % 4
 *     x.push([normal(s, key ? 3 : 0, 0.5), normal(s, key ? 3 : 0, 0.5)])
 *     bags.push(b)
 *   }
 * }
 * const mil = attentionMil(x, bags, labels, { steps: 100 })
 * const r = mil.predict([[0, 0], [3, 3], [0.2, -0.1], [-0.3, 0.1], [0.1, 0.2], [-0.1, 0]], [0, 0, 0, 1, 1, 1], 2)
 * print('loss at the start and the end:', mil.losses[0], mil.losses.at(-1))
 * print('p(positive) of a bag with the key instance and one without:', r.bags)
 * print('attention in the first bag:', r.attention.slice(0, 3))
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
