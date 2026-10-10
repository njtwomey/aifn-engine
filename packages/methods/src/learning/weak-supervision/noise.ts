/**
 * Estimating label noise by confident learning (Northcutt, Jiang and Chuang, 2021): from noisy labels $\tilde{y}$ and
 * out-of-sample predicted probabilities $\hat{p}(y \mid \xvec)$, count the examples whose label disagrees with a class
 * the model is confident in (the confident joint), calibrate the counts to the label totals, and read off the joint
 * distribution of noisy and true labels, the noise rates $p(\tilde{y} = i \mid y = j)$ and the examples likely
 * mislabelled. Classes are $0, \dots, K - 1$.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The estimates of `confidentLearning`. */
export type ConfidentLearning = {
  /** $C_{ij}$: the number of examples labelled $i$ whose confident class is $j$ ($K \times K$ counts). */
  confidentJoint: Tensor
  /**
   * $Q_{ij} \approx p(\tilde{y} = i, y = j)$, $K \times K$, calibrated so each row sums to the share of examples
   * labelled $i$ (when every label has at least one confident example; a row with none is 0) and the whole to 1.
   */
  joint: Tensor
  /**
   * $p(\tilde{y} = i \mid y = j)$: the noise transition matrix, $K \times K$ (columns sum to 1; a column with no
   * estimated mass is 0).
   */
  noise: Tensor
  /** $p(y = j)$, the estimated distribution of the true labels ($K$ values, the column sums of $\Qmat$). */
  prior: Tensor
  /**
   * Per-class thresholds $t_j$: the mean predicted probability of class $j$ over the examples labelled $j$ (1 for a
   * class with no examples).
   */
  thresholds: Float64Array
  /**
   * Indices of the examples whose confident class differs from their label (likely label errors), in ascending order.
   */
  issues: number[]
}

/**
 * Confident learning (Northcutt, Jiang and Chuang, 2021): thresholds $t_j$ are the mean $\hat{p}_j$ over the examples
 * labelled $j$; an example labelled $i$ counts towards $C_{ij}$ for the class
 * $j = \argmax \{\hat{p}_j : \hat{p}_j \ge t_j\}$ when such a class exists; rows of $\Cmat$ are rescaled to the
 * number of examples labelled $i$ and the whole normalised, giving $\Qmat$. Throws `ShapeError` unless there is one
 * label per row of probabilities.
 *
 * @param labels The noisy labels $\tilde{y}$, one per example, each a class in $0, \dots, K - 1$.
 * @param probabilities The out-of-sample predictions of any classifier, $n \times K$: row $i$ holds
 *   $\hat{p}(y = j \mid \xvec_i)$ for each class $j$.
 * @returns The confident joint, the calibrated joint, the noise matrix, the prior, the thresholds and the flagged
 *   examples.
 *
 * @example A fifth of class 0 relabelled as class 1, found from the predictions
 * const s = stream(10)
 * const truth = Array.from({ length: 300 }, (_, i) => i % 3)
 * const labels = truth.map((y) => (y === 0 && uniform(s) < 0.2 ? 1 : y))
 * const probabilities = truth.map((y) => {
 *   const p = 0.6 + 0.3 * uniform(s)
 *   return [0, 1, 2].map((k) => (k === y ? p : (1 - p) / 2))
 * })
 * const cl = confidentLearning(labels, probabilities)
 * print('p(noisy label i | true class j):', cl.noise)
 * print('prior of the true classes:', cl.prior)
 * const flipped = labels.filter((l, i) => l !== truth[i]).length
 * const caught = cl.issues.filter((i) => labels[i] !== truth[i]).length
 * print('flipped:', flipped, ' flagged:', cl.issues.length, ' of which flipped:', caught)
 */
export function confidentLearning(labels: ArrayLike<number>, probabilities: MatrixLike): ConfidentLearning {
  const P = dense.toMatrixF64(probabilities, 'confidentLearning')
  const { m: n, n: K } = P
  if (labels.length !== n)
    throw new ShapeError('confidentLearning', 'confidentLearning: one label per row of probabilities')
  const thresholds = new Float64Array(K)
  const counts = new Float64Array(K)
  for (let i = 0; i < n; i++) {
    counts[labels[i]]++
    for (let j = 0; j < K; j++) if (labels[i] === j) thresholds[j] += P.data[i * K + j]
  }
  thresholds.forEach((t, j) => (thresholds[j] = counts[j] > 0 ? t / counts[j] : 1))
  const C = new Float64Array(K * K)
  const issues: number[] = []
  for (let i = 0; i < n; i++) {
    let best = -1
    for (let j = 0; j < K; j++) {
      const p = P.data[i * K + j]
      if (p >= thresholds[j] && (best < 0 || p > P.data[i * K + best])) best = j
    }
    if (best < 0) continue
    C[labels[i] * K + best]++
    if (best !== labels[i]) issues.push(i)
  }
  const Q = new Float64Array(K * K)
  for (let a = 0; a < K; a++) {
    let row = 0
    for (let b = 0; b < K; b++) row += C[a * K + b]
    for (let b = 0; b < K; b++) Q[a * K + b] = row > 0 ? (C[a * K + b] / row) * counts[a] : 0
  }
  const total = Q.reduce((s, v) => s + v, 0)
  Q.forEach((v, i) => (Q[i] = v / total))
  const prior = new Float64Array(K)
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) prior[b] += Q[a * K + b]
  const noise = Float64Array.from(Q, (v, i) => (prior[i % K] > 0 ? v / prior[i % K] : 0))
  return {
    confidentJoint: fromData(C, [K, K]),
    joint: fromData(Q, [K, K]),
    noise: fromData(noise, [K, K]),
    prior: fromData(prior, [K]),
    thresholds,
    issues,
  }
}

/**
 * The size of a class-count vector, for callers that pass labels without the number of classes: one more than the
 * largest label (0 for no labels).
 *
 * @param labels The labels, classes counted from 0.
 * @returns The number of classes $K$ the labels imply.
 *
 * @example Labels up to 3 imply four classes
 * print('K =', classCount([0, 3, 1, 1]))
 */
export function classCount(labels: ArrayLike<number>): Size {
  let k = 0
  for (let i = 0; i < labels.length; i++) k = Math.max(k, labels[i] + 1)
  return k
}
