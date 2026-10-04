/**
 * Estimating label noise by confident learning (Northcutt, Jiang and Chuang, 2021): from noisy labels ỹ and
 * out-of-sample predicted probabilities p̂(y | x), count the examples whose label disagrees with a class the model is
 * confident in (the confident joint), calibrate the counts to the label totals, and read off the joint distribution of
 * noisy and true labels, the noise rates P(ỹ = i | y = j) and the examples most likely mislabelled.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The estimates of `confidentLearning`. */
export type ConfidentLearning = {
  /** C[i][j]: examples labelled i whose confident class is j (K × K counts). */
  confidentJoint: Tensor
  /** Q[i][j] ≈ P(ỹ = i, y = j), calibrated so each row sums to the share of examples labelled i. */
  joint: Tensor
  /** P(ỹ = i | y = j): the noise transition matrix (columns sum to 1). */
  noise: Tensor
  /** P(y = j), the estimated distribution of the true labels. */
  prior: Tensor
  /** Per-class thresholds t_j: the mean predicted probability of class j over the examples labelled j. */
  thresholds: Float64Array
  /** Examples whose confident class differs from their label (likely label errors). */
  issues: number[]
}

/**
 * Confident learning: thresholds t_j are the mean p̂_j over the examples labelled j; an example labelled i counts
 * towards C[i][j] for the class j = argmax{p̂_j : p̂_j ≥ t_j} when such a class exists; rows of C are rescaled to the
 * number of examples labelled i and the whole normalised, giving Q. `labels` are the noisy labels ỹ and
 * `probabilities` the [n, K] out-of-sample predictions of any classifier.
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

/** The size of a class-count vector, for callers that pass labels without the number of classes. */
export function classCount(labels: ArrayLike<number>): Size {
  let k = 0
  for (let i = 0; i < labels.length; i++) k = Math.max(k, labels[i] + 1)
  return k
}
