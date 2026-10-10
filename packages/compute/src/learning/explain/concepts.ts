/**
 * Concept-based explanations: testing with concept activation vectors (TCAV; Kim et al., 2018).
 *
 * A concept (stripes, a corner dot) is given by examples. A linear probe (L2-regularised logistic regression) separates
 * their activations at a hidden layer from those of random examples; the unit normal of its boundary, pointing to the
 * concept, is the concept activation vector $\vvec_C$. The conceptual sensitivity of an input is the directional
 * derivative $S_C(\xvec) = \nabla_{\hvec} f(\hvec(\xvec)) \cdot \vvec_C$ of the class logit $f$ with respect to the
 * activations $\hvec$; the TCAV score of a class is the fraction of its examples with $S_C > 0$. A score near $1/2$ can
 * arise by chance, so CAVs are trained against several random sets, random-against-random CAVs give a null
 * distribution of scores, and a two-sided t-test (equal variances, as the authors' code) compares the two.
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

/**
 * A linear probe $\wvec^\top\avec + b$: its `weights` $\wvec$ ($h$ values), its `bias` $b$, and its `accuracy` on
 * the rows it was trained on.
 */
export type LinearProbe = { weights: Float64Array; bias: number; accuracy: number }

/**
 * Fit an L2-regularised logistic regression separating `positive` rows (label $s = 1$) from `negative` rows
 * ($s = -1$): minimise the mean of $\operatorname{softplus}(-s(\wvec^\top\avec + b))$ plus
 * $(\lambda/2)\lVert \wvec \rVert^2$ by L-BFGS (at most 500 steps, from zero). Throws `ShapeError` when the two sets'
 * widths differ.
 *
 * @param positive The positive rows ($p \times h$).
 * @param negative The negative rows ($q \times h$).
 * @param options The penalty.
 * @param options.l2 The L2 penalty $\lambda$ on the weights, not the bias (default 0.01).
 * @returns The probe, with its training accuracy (a row on the boundary counts as negative).
 *
 * @example Two clouds split along the first axis
 * const probe = linearProbe([[2, 0], [3, 1], [2.5, -1]], [[-2, 0], [-3, 1], [-2, -1]])
 * print(probe)
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

/**
 * The concept activation vector: the unit normal of a `linearProbe` of the concept's activations against random ones,
 * pointing towards the concept.
 *
 * @param concept The concept's examples' activations ($p \times h$), the positive class.
 * @param random The random examples' activations ($q \times h$), the negative class.
 * @param options The penalty.
 * @param options.l2 The probe's L2 penalty (default 0.01).
 * @returns `vector`, the CAV ($h$ values, unit length unless the probe's weights are all zero), and `accuracy`, the
 *   probe's training accuracy.
 *
 * @example The concept lies along the first axis
 * print(conceptActivationVector([[2, 0], [3, 1], [2.5, -1]], [[-2, 0], [-3, 1], [-2, -1]]))
 */
export function conceptActivationVector(
  concept: MatrixLike,
  random: MatrixLike,
  options: { l2?: number } = {},
): { vector: Float64Array; accuracy: number } {
  const probe = linearProbe(concept, random, options)
  const norm = Math.hypot(...probe.weights) || 1
  return { vector: Float64Array.from(probe.weights, (w) => w / norm), accuracy: probe.accuracy }
}

/**
 * Gradients of a differentiable head (from a layer's activations to the class logit) at each row of activations, in
 * one `vmap(grad(head))` call.
 *
 * @param head The rest of the network: a function of one activation vector ($h$ values) returning the class logit.
 * @param activations The activations of the class's examples ($m \times h$).
 * @returns The gradients $\nabla_{\hvec} f$ ($m \times h$), one row per example.
 *
 * @example The gradient of a product
 * const head = (a) => mul(get(a, 0), get(a, 1))
 * print(activationGradients(head, [[1, 2], [3, -1]]))
 */
export function activationGradients(head: Differentiable, activations: MatrixLike): Tensor {
  const { data, m, n } = dense.toMatrixF64(activations, 'activationGradients')
  return vmap(grad(head))(fromData(Float64Array.from(data), [m, n])) as Tensor
}

/**
 * The directional derivatives $S = \nabla_{\hvec} f \cdot \vvec$ of each row of gradients along a CAV $\vvec$. Throws
 * `ShapeError` when the CAV's length is not the gradients' width.
 *
 * @param gradients The gradients ($m \times h$), as `activationGradients` returns them.
 * @param cav The CAV $\vvec$ ($h$ values).
 * @returns One sensitivity per row ($m$ values).
 *
 * @example Along the first axis
 * print(conceptSensitivity([[1, 0], [0.5, 2], [-1, 1]], [1, 0]))
 */
export function conceptSensitivity(gradients: MatrixLike, cav: ArrayLike<number>): Float64Array {
  const { data, m, n } = dense.toMatrixF64(gradients, 'conceptSensitivity')
  if (cav.length !== n) throw new ShapeError('conceptSensitivity', 'conceptSensitivity: the CAV must match the width')
  return Float64Array.from({ length: m }, (_, i) => {
    let s = 0
    for (let j = 0; j < n; j++) s += data[i * n + j] * cav[j]
    return s
  })
}

/**
 * The TCAV score: the fraction of rows of gradients with a positive derivative along the CAV.
 *
 * @param gradients The gradients ($m \times h$), as `activationGradients` returns them.
 * @param cav The CAV ($h$ values).
 * @returns The score, in $[0, 1]$.
 *
 * @example Two of three examples respond positively
 * print(tcavScore([[1, 0], [0.5, 2], [-1, 1]], [1, 0]))
 */
export function tcavScore(gradients: MatrixLike, cav: ArrayLike<number>): number {
  const s = conceptSensitivity(gradients, cav)
  return s.reduce((a, v) => a + (v > 0 ? 1 : 0), 0) / s.length
}

/** The result of `tcav`. */
export type TcavResult = {
  /** TCAV scores of the concept against each random set. */
  scores: Float64Array
  /** The training accuracy of each of those CAVs' probes. */
  accuracies: Float64Array
  /** TCAV scores of random-against-random CAVs: the null distribution. */
  randomScores: Float64Array
  /** The mean of `scores`. */
  mean: number
  /** The standard deviation of `scores` (denominator $n - 1$). */
  sd: number
  /**
   * The two-sided pooled t-test p-value of concept against random scores (1 or 0 when both are constant: whether the
   * constants agree).
   */
  pValue: number
  /** Whether `pValue` is below $\alpha$. */
  significant: boolean
  /** The CAV against the first random set, for display. */
  cav: Float64Array
  /**
   * The mean directional derivative $S_C$ over the class's examples, averaged over the random sets: the score only
   * counts signs, so a concept the logit barely responds to can still score 0 or 1; this says how much it responds.
   */
  sensitivity: number
}

/**
 * TCAV of a class: trains a CAV of the concept against each random set and one between each random set and the next
 * (cyclically: $r_0$ against $r_1$, ..., $r_{n-1}$ against $r_0$), scores each, and tests the difference between the
 * two groups of scores. Throws `DomainError` for fewer than two random sets.
 *
 * @param head The rest of the network: maps one activation vector ($h$ values) to the class logit.
 * @param activations The class's examples' activations at the layer ($m \times h$).
 * @param concept The concept's examples' activations ($p \times h$).
 * @param randoms At least two sets of random examples' activations (each $q \times h$).
 * @param options The probes' penalty and the test's level.
 * @param options.l2 The probes' L2 penalty (default 0.01).
 * @param options.alpha The significance level $\alpha$ (default 0.05).
 * @returns The scores, the null scores, the test and the CAV (see `TcavResult`).
 *
 * @example The logit rises along the concept's direction
 * // The logit a0 * a1 rises with a0 where a1 > 0, as it is for the class; the concept's examples sit far along a0.
 * const head = (a) => mul(get(a, 0), get(a, 1))
 * const examples = add(normals(stream(0), [10, 2]), tensor([0, 2]))
 * const concept = add(normals(stream(1), [10, 2]), tensor([3, 0]))
 * const randoms = [2, 3, 4, 5, 6, 7, 8, 9].map((k) => normals(stream(k), [10, 2]))
 * const r = tcav(head, examples, concept, randoms)
 * print('scores =', r.scores, ' random =', r.randomScores)
 * print('p =', r.pValue, ' significant =', r.significant, ' sensitivity =', r.sensitivity)
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
