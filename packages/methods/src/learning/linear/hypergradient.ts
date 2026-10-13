/**
 * The hypergradient of ridge regression: the derivative of a validation loss with respect to the ridge penalty, by
 * the implicit function theorem (Bengio, 2000; Pedregosa, 2016), and gradient descent on $\ln\lambda$ that uses it.
 *
 * The weights $\wvec(\lambda)$ minimise $\norm{\yvec - \Xmat\wvec}^2 + \lambda\norm{\wvec}^2$ on the training data
 * (centred when an intercept is fitted, as `linearRegression({ l2 })` and scikit-learn's `Ridge`), so they solve
 * $(\Xmat^\top\Xmat + \lambda\Imat)\wvec = \Xmat^\top\yvec$. Differentiating that identity in $\lambda$ gives
 *
 * $$ \frac{d\wvec}{d\lambda} = -(\Xmat^\top\Xmat + \lambda\Imat)^{-1}\wvec, $$
 *
 * without differentiating through any solver. With $L(\lambda) = \frac1m\norm{\yvec_v - \hat{\yvec}_v}^2$ the
 * validation mean squared error and $\gvec = \partial L / \partial\wvec$, the chain rule gives
 * $dL/d\lambda = \gvec^\top d\wvec/d\lambda$, which costs one more solve with the same Cholesky factor.
 */

import { cholesky, choleskySolve } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { matrixShape, targetValues } from 'aifn-compute/learning/estimators'
import { centre, gram } from './elasticNet'

/** The validation loss of ridge at one penalty, with its derivative. */
export interface RidgeHypergradient {
  /** The penalty $\lambda$. */
  readonly lambda: number
  /** The validation mean squared error $L(\lambda)$. */
  readonly loss: number
  /** $dL/d\lambda$. */
  readonly gradient: number
  /** $dL/d\ln\lambda = \lambda\, dL/d\lambda$, the gradient on the log scale on which penalties are searched. */
  readonly logGradient: number
  /** The ridge weights $\wvec(\lambda)$, $d$ values. */
  readonly weights: Tensor
  /** The intercept $\bar y - \bar{\xvec}^\top\wvec$ (0 without an intercept). */
  readonly intercept: number
  /** $d\wvec/d\lambda = -(\Xmat^\top\Xmat + \lambda\Imat)^{-1}\wvec$, $d$ values. */
  readonly weightsGradient: Tensor
}

/** The evaluator of the validation loss and its hypergradient, built once for a train-validation split. */
type Evaluate = (lambda: number) => RidgeHypergradient

/**
 * Centre the training data, centre the validation data by the training means, and return the evaluator. Throws
 * `ShapeError` for mismatched shapes.
 *
 * @param xTrain The training design, $n \times d$.
 * @param yTrain The training targets, $n$ values.
 * @param xVal The validation design, $m \times d$.
 * @param yVal The validation targets, $m$ values.
 * @param intercept Whether to centre, for an unpenalised intercept.
 * @param where The caller's name, for error messages.
 * @returns The evaluator of the loss and hypergradient at a penalty.
 */
function evaluator(
  xTrain: Tensor,
  yTrain: Tensor,
  xVal: Tensor,
  yVal: Tensor,
  intercept: boolean,
  where: string,
): Evaluate {
  const c = centre(xTrain, yTrain, intercept, where)
  const { d, H, q } = gram(c)
  const [m, cols] = matrixShape(xVal, where)
  if (cols !== d) throw new ShapeError(where, `${where}: ${d} training columns but ${cols} validation columns`)
  const target = targetValues(yVal, where)
  if (target.length !== m) throw new ShapeError(where, `${where}: ${m} validation rows but ${target.length} targets`)
  const V = Float64Array.from(dense.data(xVal), (v, k) => v - c.xMean[k % d])
  const yv = Float64Array.from(target, (v) => v - c.yMean)
  return (lambda) => {
    const A = Float64Array.from(H)
    for (let j = 0; j < d; j++) A[j * d + j] += lambda
    const { L } = cholesky(fromData(A, [d, d]), { jitter: false })
    const w = dense.data(choleskySolve(L, fromData(q, [d])))
    // The residuals on the validation set and the gradient of the loss in the weights, g = -(2/m) Vᵀ r.
    const g = new Float64Array(d)
    let loss = 0
    for (let i = 0; i < m; i++) {
      let r = yv[i]
      for (let j = 0; j < d; j++) r -= V[i * d + j] * w[j]
      loss += (r * r) / m
      for (let j = 0; j < d; j++) g[j] -= (2 / m) * V[i * d + j] * r
    }
    const dw = Float64Array.from(dense.data(choleskySolve(L, fromData(Float64Array.from(w), [d]))), (v) => -v)
    let gradient = 0
    for (let j = 0; j < d; j++) gradient += g[j] * dw[j]
    let b = c.yMean
    for (let j = 0; j < d; j++) b -= c.xMean[j] * w[j]
    return {
      lambda,
      loss,
      gradient,
      logGradient: lambda * gradient,
      weights: fromData(Float64Array.from(w), [d]),
      intercept: b,
      weightsGradient: fromData(dw, [d]),
    }
  }
}

/**
 * Check a penalty.
 *
 * @param lambda The penalty; must be positive and finite.
 * @param where The caller's name, for error messages.
 */
function checkLambda(lambda: number, where: string): void {
  if (!(lambda > 0) || !Number.isFinite(lambda))
    throw new DomainError(where, `${where}: lambda must be positive and finite`)
}

/**
 * The validation mean squared error of ridge regression and its derivative in the penalty, by the implicit function
 * theorem: the weights fitted on the training data at $\lambda$ satisfy
 * $(\Xmat^\top\Xmat + \lambda\Imat)\wvec = \Xmat^\top\yvec$, so
 * $d\wvec/d\lambda = -(\Xmat^\top\Xmat + \lambda\Imat)^{-1}\wvec$, and $dL/d\lambda$ follows by the chain rule.
 * The penalty is in the scaling of `linearRegression({ l2 })` and scikit-learn's `Ridge(alpha)`; the intercept, when
 * fitted, is unpenalised and moves with the weights. Throws `DomainError` for $\lambda \le 0$, and `ShapeError` for
 * mismatched shapes.
 *
 * @param xTrain The training design, $n \times d$.
 * @param yTrain The training targets, $n$ values.
 * @param xVal The validation design, $m \times d$.
 * @param yVal The validation targets, $m$ values.
 * @param lambda The penalty $\lambda > 0$.
 * @param options Whether to fit an unpenalised intercept by centring (default true).
 * @param options.intercept Centre the training data first; the validation data are centred by the training means.
 * @returns The validation loss, its derivative in $\lambda$ and in $\ln\lambda$, and the weights and their derivative.
 *
 * @example The hypergradient against a finite difference
 * const z = toArray(normals(stream(0), [60, 4]))
 * const noise = toFlat(normals(stream(1), [60]))
 * const y = z.map((r, i) => r[0] - r[1] + 0.5 * r[2] + noise[i])
 * const [xt, yt, xv, yv] = [tensor(z.slice(0, 40)), tensor(y.slice(0, 40)), tensor(z.slice(40)), tensor(y.slice(40))]
 * const h = ridgeHypergradient(xt, yt, xv, yv, 5)
 * const step = 1e-5
 * const up = ridgeHypergradient(xt, yt, xv, yv, 5 + step).loss
 * const down = ridgeHypergradient(xt, yt, xv, yv, 5 - step).loss
 * print('loss =', h.loss, ' dL/dlambda =', h.gradient)
 * print('finite difference =', (up - down) / (2 * step))
 */
export function ridgeHypergradient(
  xTrain: Tensor,
  yTrain: Tensor,
  xVal: Tensor,
  yVal: Tensor,
  lambda: number,
  { intercept = true }: { intercept?: boolean } = {},
): RidgeHypergradient {
  checkLambda(lambda, 'ridgeHypergradient')
  return evaluator(xTrain, yTrain, xVal, yVal, intercept, 'ridgeHypergradient')(lambda)
}

/** The problem a `ridgeHyperDescentSteps` run solves, and how it steps. */
export interface RidgeHyperDescentProblem {
  /** The training design, $n \times d$. */
  xTrain: Tensor
  /** The training targets, $n$ values. */
  yTrain: Tensor
  /** The validation design, $m \times d$. */
  xVal: Tensor
  /** The validation targets, $m$ values. */
  yVal: Tensor
  /**
   * The step size $\eta$ on $\ln\lambda$ (default 1). The gradient is in the units of the loss, so scale $\eta$ by
   * the inverse of the validation loss's size (standardise the target, or divide by its variance).
   */
  eta?: number
  /** A state is `converged` when $\lvert dL/d\ln\lambda \rvert$ is below this (default $10^{-8}$). */
  tolerance?: number
  /** Fit an unpenalised intercept by centring (default true). */
  intercept?: boolean
}

/** One state of `ridgeHyperDescentSteps`. */
export interface RidgeHyperDescentState extends Status {
  /** Steps taken. */
  t: number
  /** The penalty $\lambda$. */
  lambda: number
  /** $\ln\lambda$, the iterate. */
  logLambda: number
  /** The validation mean squared error at $\lambda$. */
  loss: number
  /** $dL/d\ln\lambda$ at $\lambda$. */
  gradient: number
  /** The ridge weights at $\lambda$, $d$ values. */
  weights: Tensor
}

/**
 * Gradient descent on $\ln\lambda$ for the validation loss of ridge regression, with the hypergradient of
 * `ridgeHypergradient`: each step sets $\ln\lambda \leftarrow \ln\lambda - \eta\, dL/d\ln\lambda$. Searching on the
 * log scale keeps $\lambda$ positive and matches how penalties are gridded. The loss is not convex in $\lambda$ in
 * general, so the run finds a local minimum, the nearest one downhill from the start. A state whose loss or penalty
 * is no longer finite is flagged `diverged`. `init` takes the starting penalty (default 1). Throws `DomainError` for a
 * step size that is not positive or a starting penalty that is not positive, and `ShapeError` for mismatched shapes.
 *
 * @param problem The training and validation data, the step size, tolerance and centring, as
 *   `RidgeHyperDescentProblem`.
 * @returns The algorithm, to run with `run` or `trace`; its start is `{ lambda? }`.
 *
 * @example Descending to the validation-optimal penalty
 * // Thirty training points and fifty validation points, eight columns, a weak signal in four of them.
 * const z = toArray(normals(stream(0), [80, 8]))
 * const noise = toFlat(normals(stream(1), [80]))
 * const y = z.map((r, i) => r[0] - 0.5 * r[1] + 0.3 * r[2] + 0.2 * r[3] + 1.5 * noise[i])
 * const alg = ridgeHyperDescentSteps({
 *   xTrain: tensor(z.slice(0, 30)), yTrain: tensor(y.slice(0, 30)),
 *   xVal: tensor(z.slice(30)), yVal: tensor(y.slice(30)), eta: 5,
 * })
 * const tr = trace(alg, { lambda: 1 }, 20, { keep: 'all' })
 * print('lambda:', tr.steps.map((s) => s.lambda.toFixed(2)))
 * print('validation loss:', tr.steps.map((s) => s.loss.toFixed(4)))
 */
export function ridgeHyperDescentSteps(
  problem: RidgeHyperDescentProblem,
): Algorithm<{ lambda?: number }, RidgeHyperDescentState> {
  const { eta = 1, tolerance = 1e-8, intercept = true } = problem
  if (!(eta > 0)) throw new DomainError('ridgeHyperDescentSteps', 'ridgeHyperDescentSteps: eta must be positive')
  const evaluate = evaluator(
    problem.xTrain,
    problem.yTrain,
    problem.xVal,
    problem.yVal,
    intercept,
    'ridgeHyperDescentSteps',
  )
  const state = (t: number, lambda: number, previous?: Tensor): RidgeHyperDescentState => {
    if (!(lambda > 0) || !Number.isFinite(lambda)) {
      const d = previous?.shape[0] ?? 0
      return {
        t,
        lambda,
        logLambda: Math.log(lambda),
        loss: NaN,
        gradient: NaN,
        weights: previous ?? fromData(new Float64Array(d), [d]),
        diverged: true,
      }
    }
    const h = evaluate(lambda)
    return {
      t,
      lambda,
      logLambda: Math.log(lambda),
      loss: h.loss,
      gradient: h.logGradient,
      weights: h.weights,
      converged: Math.abs(h.logGradient) < tolerance,
      diverged: !Number.isFinite(h.loss),
    }
  }
  return {
    name: 'ridge-hyper-descent',
    init: ({ lambda = 1 }) => {
      checkLambda(lambda, 'ridgeHyperDescentSteps')
      return state(0, lambda)
    },
    step: (s) => state(s.t + 1, Math.exp(s.logLambda - eta * s.gradient), s.weights),
  }
}
