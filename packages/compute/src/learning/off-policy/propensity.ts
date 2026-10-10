/**
 * Propensity estimation: when a log records the actions but not the probabilities the logging policy gave them, the
 * propensities $\hat\pi_0(a \mid x)$ are estimated by a model of the action given the context: a multinomial logistic
 * regression (fitted by `minimize` of `aifn-compute/optim/minimize` on the softmax cross-entropy with an L2 penalty),
 * or, for discrete contexts, the smoothed share of each action within each context.
 *
 * Actions are indices $0, \dots, K - 1$ and contexts are rows of an $n \times d$ matrix; the estimated propensities of
 * the logged actions are what the off-policy estimators take as `propensities`. Malformed input (lengths that
 * disagree, an action or group that is not an index) throws `DomainError`.
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
  /** The coefficients, $(d + 1) \times K$: one column per action, the last row the intercepts. */
  readonly coefficients: Tensor
  /** $\hat\pi_0(a \mid x_i)$ for the training contexts, $n \times K$. */
  readonly probabilities: Tensor
  /** $\hat\pi_0(a_i \mid x_i)$, the estimated propensity of each logged action, $n$ values. */
  readonly propensities: Tensor
  /** The mean cross-entropy at the fit (without the penalty). */
  readonly logLoss: number
  /** Whether the optimiser reported convergence within `maxSteps`. */
  readonly converged: boolean
  /**
   * $\hat\pi_0(\cdot \mid x)$ for new contexts: an $m \times d$ matrix in, an $m \times K$ matrix out. Throws
   * `DomainError` for a different number of features.
   */
  predict(contexts: MatrixLike): Tensor
}

/**
 * Estimate the logging propensities by multinomial logistic regression of the logged action on the context: the
 * coefficients $\Wmat$ minimise the mean softmax cross-entropy plus $\frac{\lambda}{2} \norm{\Wmat}^2$ (the intercepts
 * unpenalised), starting from zero. When the logging policy is a softmax of a linear score of $x$, the model is well
 * specified and $\hat\pi_0 \to \pi_0$. Throws `DomainError` when the counts disagree or an action is not an index.
 *
 * @param contexts The contexts, $n \times d$, one row per logged round.
 * @param actions The logged actions, $n$ indices.
 * @param options The number of actions, the penalty and the optimiser's budget.
 * @param options.actions The number of actions $K$, when some were never logged (default: the largest logged action
 *   plus 1).
 * @param options.l2 The penalty $\lambda$ on the non-intercept coefficients (default $10^{-3}$).
 * @param options.maxSteps The most optimiser steps (default 500).
 * @returns The fitted model, with the estimated propensities of the logged actions.
 *
 * @example A binary context; the logger took action 1 a third of the time at 0 and two thirds at 1
 * const contexts = [[0], [0], [0], [1], [1], [1]]
 * const actions = [0, 0, 1, 1, 1, 0]
 * const model = estimatePropensities(contexts, actions)
 * print('converged:', model.converged)
 * print('at x = 0 and x = 1:', model.predict([[0], [1]]))
 * print('propensities of the logged actions:', model.propensities)
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
 * Propensities for discrete contexts:
 * $\hat\pi_0(a \mid g) = (\mathrm{count}(g, a) + \alpha)/(\mathrm{count}(g) + K\alpha)$, the share of action $a$ among
 * the rounds of context group $g$, with additive smoothing $\alpha$ (default 0). A group with no rounds and no
 * smoothing gets $1/K$ for every action. Throws `DomainError` when the counts disagree or a group or action is not an
 * index.
 *
 * @param groups The context group $g_i$ of each round, $n$ indices $0, \dots, G - 1$.
 * @param actions The logged action $a_i$ of each round, $n$ indices.
 * @param options The number of actions and the smoothing.
 * @param options.actions The number of actions $K$, when some were never logged (default: the largest logged action
 *   plus 1).
 * @param options.smoothing The pseudo-count $\alpha$ added to every (group, action) count.
 * @returns `propensities`, $\hat\pi_0(a_i \mid g_i)$ for each round ($n$ values), and `table`, $\hat\pi_0(a \mid g)$
 *   as a $G \times K$ matrix.
 *
 * @example Shares of actions within two groups, with and without smoothing
 * const groups = [0, 0, 0, 1, 1]
 * const actions = [0, 1, 1, 0, 0]
 * const raw = empiricalPropensities(groups, actions)
 * print('table:', raw.table)
 * print('propensities:', raw.propensities)
 * print('smoothed table:', empiricalPropensities(groups, actions, { smoothing: 1 }).table)
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
