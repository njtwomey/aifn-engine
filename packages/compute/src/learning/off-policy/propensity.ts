/**
 * Propensity estimation: when a log records the actions but not the probabilities the logging policy gave them, the
 * propensities π̂₀(a | x) are estimated by a model of the action given the context: a multinomial logistic regression
 * (fitted by L-BFGS on the softmax cross-entropy with an L2 penalty), or, for discrete contexts, the smoothed share of
 * each action within each context.
 */

import type { MatrixLike, Objective, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  add,
  concat,
  dense,
  matmul,
  mul,
  ones,
  reshape,
  sum,
  square,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { softmax } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'

/** A fitted propensity model. */
export interface PropensityModel {
  /** The coefficients [d + 1, K]: one column per action, the last row the intercepts. */
  readonly coefficients: Tensor
  /** π̂₀(a | xᵢ) for the training contexts [n, K]. */
  readonly probabilities: Tensor
  /** π̂₀(aᵢ | xᵢ), the estimated propensity of each logged action [n]. */
  readonly propensities: Tensor
  /** The mean cross-entropy at the fit (without the penalty). */
  readonly logLoss: number
  readonly converged: boolean
  /** π̂₀(· | x) for new contexts [m, d] → [m, K]. */
  predict(contexts: MatrixLike): Tensor
}

/**
 * Estimate the logging propensities by multinomial logistic regression of the logged action on the context: the
 * coefficients minimise the mean softmax cross-entropy plus (λ/2)‖W‖² (the intercepts unpenalised). When the logging
 * policy is a softmax of a linear score of x, the model is well specified and π̂₀ → π₀.
 */
export function estimatePropensities(
  contexts: MatrixLike,
  actions: VectorLike,
  options: { actions?: number; l2?: number; maxSteps?: number } = {},
): PropensityModel {
  const where = 'estimatePropensities'
  const { data: x, m: n, n: d } = dense.toMatrixF64(contexts, where)
  const a = dense.toF64(actions, where)
  if (a.length !== n) throw new DomainError(where, `${where}: ${n} contexts and ${a.length} actions`)
  let K = options.actions ?? 0
  for (const v of a) {
    if (!Number.isInteger(v) || v < 0) throw new DomainError(where, `${where}: action ${v} is not an index`)
    K = Math.max(K, v + 1)
  }
  const l2 = options.l2 ?? 1e-3
  const design = (rows: dense.F64, m: number) => concat([dense.mat(rows, m, d), ones([m, 1])], 1)
  const X = design(x, n)
  const labels = Int32Array.from(a)
  const penalty = Float64Array.from({ length: (d + 1) * K }, (_, k) => (k < d * K ? 1 : 0))
  const objective: Objective = {
    kind: 'objective',
    name: 'propensity log-loss',
    dim: (d + 1) * K,
    value: (w) => {
      const W = reshape(w, [d + 1, K])
      const ce = softmaxCrossEntropy(matmul(X, W), labels, { reduction: 'mean' })
      return add(ce, mul(l2 / 2, sum(mul(dense.vec(penalty), square(w)))))
    },
  }
  const fit = minimize(objective, new Float64Array((d + 1) * K), { maxSteps: options.maxSteps ?? 500 })
  const W = dense.mat(dense.data(fit.x as Tensor), d + 1, K)
  const predict = (c: MatrixLike) => {
    const { data, m, n: width } = dense.toMatrixF64(c, where)
    if (width !== d) throw new DomainError(where, `${where}: contexts have ${width} features, the model ${d}`)
    return softmax(matmul(design(data, m), W) as Tensor)
  }
  const probabilities = predict(contexts)
  const P = dense.data(probabilities)
  const propensities = Float64Array.from({ length: n }, (_, i) => P[i * K + labels[i]])
  const logLoss = -propensities.reduce((s, p) => s + Math.log(p), 0) / n
  return {
    coefficients: W,
    probabilities,
    propensities: dense.vec(propensities),
    logLoss,
    converged: fit.converged,
    predict,
  }
}

/**
 * Propensities for discrete contexts: π̂₀(a | g) = (count(g, a) + α)/(count(g) + Kα), the share of action a among the
 * rounds of context group g, with additive smoothing α (default 0). Returns π̂₀(aᵢ | gᵢ) for each round [n] and the
 * table [G, K].
 */
export function empiricalPropensities(
  groups: VectorLike,
  actions: VectorLike,
  options: { actions?: number; smoothing?: number } = {},
): { propensities: Tensor; table: Tensor } {
  const where = 'empiricalPropensities'
  const g = dense.toF64(groups, where)
  const a = dense.toF64(actions, where)
  if (g.length !== a.length) throw new DomainError(where, `${where}: ${g.length} groups and ${a.length} actions`)
  const alpha = options.smoothing ?? 0
  let G = 0
  let K = options.actions ?? 0
  for (let i = 0; i < g.length; i++) {
    if (!Number.isInteger(g[i]) || g[i] < 0 || !Number.isInteger(a[i]) || a[i] < 0)
      throw new DomainError(where, `${where}: groups and actions must be indices`)
    G = Math.max(G, g[i] + 1)
    K = Math.max(K, a[i] + 1)
  }
  const counts = new Float64Array(G * K)
  for (let i = 0; i < g.length; i++) counts[g[i] * K + a[i]] += 1
  const table = new Float64Array(G * K)
  for (let r = 0; r < G; r++) {
    let total = 0
    for (let k = 0; k < K; k++) total += counts[r * K + k]
    for (let k = 0; k < K; k++)
      table[r * K + k] = total + K * alpha > 0 ? (counts[r * K + k] + alpha) / (total + K * alpha) : 1 / K
  }
  return {
    propensities: dense.vec(Float64Array.from({ length: g.length }, (_, i) => table[g[i] * K + a[i]])),
    table: dense.mat(table, G, K),
  }
}
