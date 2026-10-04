/**
 * Concept-based explanations: testing with concept activation vectors (TCAV; Kim et al., 2018).
 *
 * A concept (stripes, a corner dot) is given by examples. A linear probe (L2-regularised logistic regression) separates
 * their activations at a hidden layer from those of random examples; the unit normal of its boundary, pointing to the
 * concept, is the concept activation vector v_C. The conceptual sensitivity of an input is the directional derivative
 * S_C(x) = ∇_h f(h(x)) · v_C of the class logit f with respect to the activations h; the TCAV score of a class is the
 * fraction of its examples with S_C > 0. A score near 1/2 can arise by chance, so CAVs are trained against several
 * random sets, random-against-random CAVs give a null distribution of scores, and a two-sided t-test (equal variances,
 * as the authors' code) compares the two.
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import {
  add,
  dense,
  fromData,
  matmul,
  mean,
  mul,
  slice,
  sum,
  square,
  toFlat,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { softplus } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import { pooledTTest } from 'aifn-compute/probability/tests'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Differentiable } from './gradients'

/** A linear probe: weights [h], bias, and training accuracy. */
export type LinearProbe = { weights: Float64Array; bias: number; accuracy: number }

/**
 * Fit an L2-regularised logistic regression separating `positive` rows [p, h] (label 1) from `negative` rows [q, h]:
 * minimise mean softplus(−s(wᵀa + b)) + (λ/2)‖w‖² by L-BFGS (λ = `l2`, default 0.01).
 */
export function linearProbe(positive: MatrixLike, negative: MatrixLike, options: { l2?: number } = {}): LinearProbe {
  const P = dense.toMatrixF64(positive, 'linearProbe')
  const Q = dense.toMatrixF64(negative, 'linearProbe')
  if (P.n !== Q.n) throw new ShapeError('linearProbe', 'linearProbe: both sets need the same width')
  const h = P.n
  const n = P.m + Q.m
  const A = new Float64Array(n * h)
  A.set(P.data, 0)
  A.set(Q.data, P.m * h)
  const s = Float64Array.from({ length: n }, (_, i) => (i < P.m ? 1 : -1))
  const At = fromData(A, [n, h])
  const st = fromData(s, [n])
  const { l2 = 0.01 } = options
  const objective = {
    kind: 'objective' as const,
    name: 'linear probe',
    dim: h + 1,
    value: (theta: Tensor) => {
      const w = slice(theta, [0, h])
      const b = slice(theta, h)
      const margin = mul(st, add(matmul(At, w), b))
      return add(mean(softplus(mul(-1, margin))), mul(l2 / 2, sum(square(w))))
    },
  }
  const res = minimize(objective, new Float64Array(h + 1), { method: 'lbfgs', maxSteps: 500 })
  const theta = toFlat(res.x)
  const weights = Float64Array.from(theta.slice(0, h))
  const bias = theta[h]
  let right = 0
  for (let i = 0; i < n; i++) {
    let z = bias
    for (let j = 0; j < h; j++) z += weights[j] * A[i * h + j]
    if ((z > 0 ? 1 : -1) === s[i]) right++
  }
  return { weights, bias, accuracy: right / n }
}

/** The concept activation vector: the unit normal of a linear probe of `concept` [p, h] against `random` [q, h]. */
export function conceptActivationVector(
  concept: MatrixLike,
  random: MatrixLike,
  options: { l2?: number } = {},
): { vector: Float64Array; accuracy: number } {
  const probe = linearProbe(concept, random, options)
  const norm = Math.hypot(...probe.weights) || 1
  return { vector: Float64Array.from(probe.weights, (w) => w / norm), accuracy: probe.accuracy }
}

/** Gradients [m, h] of a differentiable head (activations → class logit) at each row of `activations` [m, h]. */
export function activationGradients(head: Differentiable, activations: MatrixLike): Tensor {
  const { data, m, n } = dense.toMatrixF64(activations, 'activationGradients')
  return vmap(grad(head))(fromData(Float64Array.from(data), [m, n])) as Tensor
}

/** The directional derivatives S = ∇h · v [m] of gradients [m, h] along a CAV v [h]. */
export function conceptSensitivity(gradients: MatrixLike, cav: ArrayLike<number>): Float64Array {
  const { data, m, n } = dense.toMatrixF64(gradients, 'conceptSensitivity')
  if (cav.length !== n) throw new ShapeError('conceptSensitivity', 'conceptSensitivity: the CAV must match the width')
  return Float64Array.from({ length: m }, (_, i) => {
    let s = 0
    for (let j = 0; j < n; j++) s += data[i * n + j] * cav[j]
    return s
  })
}

/** The TCAV score: the fraction of rows of `gradients` [m, h] with a positive derivative along the CAV. */
export function tcavScore(gradients: MatrixLike, cav: ArrayLike<number>): number {
  const s = conceptSensitivity(gradients, cav)
  return s.reduce((a, v) => a + (v > 0 ? 1 : 0), 0) / s.length
}

/** The result of `tcav`. */
export type TcavResult = {
  /** TCAV scores of the concept against each random set, and their CAVs' probe accuracies. */
  scores: Float64Array
  accuracies: Float64Array
  /** TCAV scores of random-against-random CAVs: the null distribution. */
  randomScores: Float64Array
  mean: number
  sd: number
  /** The two-sided pooled t-test p-value of concept against random scores, and whether it is below α. */
  pValue: number
  significant: boolean
  /** The CAV against the first random set, for display. */
  cav: Float64Array
  /**
   * The mean directional derivative S_C over the class's examples, averaged over the random sets: the score only counts
   * signs, so a concept the logit barely responds to can still score 0 or 1; this says how much it responds.
   */
  sensitivity: number
}

/**
 * TCAV of a class: `head` maps activations [h] to the class logit, `activations` [m, h] are the class's examples at
 * the layer, `concept` [p, h] the concept's examples and `randoms` (at least two sets, each [q, h]) random examples.
 * Trains a CAV against each random set and between consecutive random sets (r₀ vs r₁, r₁ vs r₂, …), scores each, and
 * tests the difference at level α (default 0.05).
 */
export function tcav(
  head: Differentiable,
  activations: MatrixLike,
  concept: MatrixLike,
  randoms: readonly MatrixLike[],
  options: { l2?: number; alpha?: number } = {},
): TcavResult {
  if (randoms.length < 2) throw new DomainError('tcav', 'tcav: needs at least two random sets')
  const { alpha = 0.05 } = options
  const grads = activationGradients(head, activations)
  const cavs = randoms.map((r) => conceptActivationVector(concept, r, options))
  let sensitivity = 0
  for (const c of cavs) {
    const sc = conceptSensitivity(grads, c.vector)
    sensitivity += sc.reduce((a, b) => a + b, 0) / sc.length / cavs.length
  }
  const scores = Float64Array.from(cavs, (c) => tcavScore(grads, c.vector))
  const accuracies = Float64Array.from(cavs, (c) => c.accuracy)
  const randomScores = Float64Array.from({ length: randoms.length }, (_, i) =>
    tcavScore(grads, conceptActivationVector(randoms[i], randoms[(i + 1) % randoms.length], options).vector),
  )
  const m = scores.reduce((a, b) => a + b, 0) / scores.length
  const sd = Math.sqrt(scores.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, scores.length - 1))
  const same = (a: Float64Array) => a.every((v) => v === a[0])
  // Identical constant samples have no variance: the test is decided by whether the constants differ.
  const pValue =
    same(scores) && same(randomScores)
      ? scores[0] === randomScores[0]
        ? 1
        : 0
      : pooledTTest(scores, randomScores, { alternative: 'two-sided' } as never).pValue
  return {
    scores,
    accuracies,
    randomScores,
    mean: m,
    sd,
    pValue,
    significant: pValue < alpha,
    cav: cavs[0].vector,
    sensitivity,
  }
}
