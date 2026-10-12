/**
 * Stochastic gradient descent for penalised linear regression, as scikit-learn's `SGDRegressor`: one shuffled pass
 * over the data per step, with the cumulative $\ell_1$ penalty of Tsuruoka, Tsujii and Ananiadou (2009) that leaves
 * exact zeros, or the plain subgradient step that never reaches them.
 *
 * The objective is
 *
 * $$ \frac1n\sum_i \frac12(y_i - \xvec_i^\top\wvec - b)^2 + \alpha R(\wvec) $$
 *
 * with $R(\wvec) = \frac12\norm{\wvec}^2$ (`'l2'`), $\norm{\wvec}_1$ (`'l1'`) or
 * $\rho\norm{\wvec}_1 + \frac{1 - \rho}{2}\norm{\wvec}^2$ (`'elasticnet'`), the scaling of `elasticNet`, so a long run
 * approaches `elasticNet({ alpha, l1Ratio })` or, for `'l2'`, `linearRegression({ l2: n * alpha })`. Each example
 * visited, with $\eta$ the step size and $p = \xvec_i^\top\wvec + b$, shrinks the weights by
 * $1 - (1 - \rho)\eta\alpha$, adds $\eta(y_i - p)\xvec_i$, moves $b$ by $\eta(y_i - p)$, and then applies the
 * $\ell_1$ part, in scikit-learn's order.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { permutation, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { matrix, values } from '../util'

/** The problem a `linearSgdSteps` run solves, and how it steps. */
export interface LinearSgdProblem {
  /** The inputs, $n \times d$ (standardise them first: one step size serves every feature). */
  x: Tensor
  /** The targets, $n$ values. */
  y: Tensor
  /** The penalty $R$: `'l2'` (default), `'l1'` or `'elasticnet'`. */
  penalty?: 'l2' | 'l1' | 'elasticnet'
  /** The penalty strength $\alpha \ge 0$ (default $10^{-4}$, as scikit-learn). */
  alpha?: number
  /**
   * The mixing $\rho$ of `'elasticnet'` (default 0.15, as scikit-learn); `'l1'` is $\rho = 1$ and `'l2'`
   * $\rho = 0$.
   */
  l1Ratio?: number
  /**
   * The step size: `'invscaling'` (default), $\eta_t = \eta_0 / t^{p}$ with $t$ the examples visited so far plus one;
   * `'constant'`, $\eta_0$.
   */
  schedule?: 'invscaling' | 'constant'
  /** The initial step size $\eta_0$ (default 0.01). */
  eta0?: number
  /** The exponent $p$ of `'invscaling'` (default 0.25). */
  powerT?: number
  /**
   * How the $\ell_1$ part is applied: `'truncated'` (default), the cumulative penalty of Tsuruoka et al. (2009) that
   * `SGDRegressor` uses, which clips a weight at zero rather than letting it cross and so leaves exact zeros;
   * `'subgradient'`, a step of $-\eta\alpha\rho\operatorname{sign}(w_j)$, which jitters around zero and never reaches
   * it.
   */
  l1Handling?: 'truncated' | 'subgradient'
  /** Fit an intercept (default true). */
  intercept?: boolean
  /** Visit the rows in a fresh random order each epoch (default true); otherwise in row order. */
  shuffle?: boolean
}

/** One state of `linearSgdSteps`: the fit after a whole number of epochs. */
export interface LinearSgdState extends Status {
  /** Epochs completed (the step count). */
  t: number
  /** Epochs completed, as `t`. */
  epoch: number
  /** Examples visited so far. */
  samples: number
  /** The weights $\wvec$, $d$ values. */
  weights: Tensor
  /** The intercept $b$. */
  intercept: number
  /** The objective on the whole data at these weights. */
  objective: number
  /** The number of weights that are exactly zero. */
  zeros: number
  /** The step size of the last update (that of the first update at the start). */
  eta: number
  /** The total $\ell_1$ shrinkage $u$ each weight could have had so far ($\sum \eta\alpha\rho$; truncated only). */
  u: number
  /** The $\ell_1$ shrinkage each weight has actually had, $d$ values (truncated only). */
  q: Tensor
}

/**
 * Stochastic gradient descent for penalised least squares as a step-through algorithm, one epoch per step: each step
 * visits every example once, in a fresh random order from the step's stream (with `shuffle`), and updates after
 * each, as scikit-learn's `SGDRegressor` (`loss='squared_error'`). The states record the weights, intercept,
 * objective and count of exact zeros after each epoch, which is what a trajectory plot needs. The run's order of
 * examples is not scikit-learn's, so a trajectory matches `SGDRegressor` only in distribution; the final weights
 * approach the penalised optimum, within the noise that the step size leaves. `init` takes optional starting weights
 * and intercept (default zero). A state whose weights are no longer finite is flagged `diverged`. Throws
 * `DomainError` for a negative $\alpha$, a $\rho$ outside $[0, 1]$ or a non-positive $\eta_0$, and `ShapeError` for
 * mismatched shapes.
 *
 * @param problem The data, the penalty and its strength, the step-size schedule and how the $\ell_1$ part is applied,
 *   as `LinearSgdProblem`.
 * @returns The algorithm, to run with `run` or `trace`; its start is `{ weights?, intercept? }`.
 *
 * @example The lasso by SGD: exact zeros with the truncated penalty, none with the subgradient
 * // y depends on the first two of five columns.
 * const x = normals(stream(0), [200, 5])
 * const y = tensor(toArray(x).map((r) => 3 * r[0] - 2 * r[1]))
 * for (const l1Handling of ['truncated', 'subgradient']) {
 *   const alg = linearSgdSteps({ x, y, penalty: 'l1', alpha: 0.2, l1Handling, schedule: 'constant', eta0: 0.01 })
 *   const s = run(alg, {}, 20, { stream: stream(1) })
 *   print(l1Handling, ' weights =', s.weights, ' zeros =', s.zeros)
 * }
 *
 * @example The objective falls epoch by epoch
 * const x = normals(stream(0), [200, 3])
 * const y = tensor(toArray(x).map((r) => r[0] - r[1] + 0.5 * r[2]))
 * const tr = trace(linearSgdSteps({ x, y, alpha: 0.01 }), {}, 10, { stream: stream(2), keep: 'all' })
 * print('objective by epoch:', tr.steps.map((s) => s.objective.toFixed(4)))
 */
export function linearSgdSteps(
  problem: LinearSgdProblem,
): Algorithm<{ weights?: Tensor; intercept?: number }, LinearSgdState> {
  const { n, d, v } = matrix(problem.x, 'linearSgdSteps')
  const y = values(problem.y)
  if (y.length !== n) throw new ShapeError('linearSgdSteps', `linearSgdSteps: ${n} rows of x but ${y.length} targets`)
  const {
    penalty = 'l2',
    alpha = 1e-4,
    schedule = 'invscaling',
    eta0 = 0.01,
    powerT = 0.25,
    l1Handling = 'truncated',
    intercept = true,
    shuffle = true,
  } = problem
  const rho = penalty === 'l1' ? 1 : penalty === 'l2' ? 0 : (problem.l1Ratio ?? 0.15)
  if (!(alpha >= 0)) throw new DomainError('linearSgdSteps', 'linearSgdSteps: alpha must be non-negative')
  if (!(rho >= 0 && rho <= 1)) throw new DomainError('linearSgdSteps', 'linearSgdSteps: l1Ratio must lie in [0, 1]')
  if (!(eta0 > 0)) throw new DomainError('linearSgdSteps', 'linearSgdSteps: eta0 must be positive')
  const stepSize = (t: number): number => (schedule === 'constant' ? eta0 : eta0 / t ** powerT)
  const objective = (w: Float64Array, b: number): number => {
    let loss = 0
    for (let i = 0; i < n; i++) {
      let r = y[i] - b
      for (let j = 0; j < d; j++) r -= v[i * d + j] * w[j]
      loss += (r * r) / 2
    }
    let l1 = 0
    let l2 = 0
    for (const wj of w) {
      l1 += Math.abs(wj)
      l2 += wj * wj
    }
    return loss / n + alpha * (rho * l1 + ((1 - rho) / 2) * l2)
  }
  const zerosOf = (w: Float64Array): number => w.reduce((k, wj) => k + (wj === 0 ? 1 : 0), 0)
  const order = (s: Stream): Int32Array =>
    shuffle ? Int32Array.from(permutation(s, n).data as Int32Array) : Int32Array.from({ length: n }, (_, i) => i)
  return {
    name: 'linear-sgd',
    init: ({ weights, intercept: b = 0 }) => {
      const w = new Float64Array(d)
      if (weights) {
        const start = values(weights)
        if (start.length !== d)
          throw new ShapeError('linearSgdSteps', `linearSgdSteps: ${start.length} starting weights for ${d} features`)
        w.set(start)
      }
      return {
        t: 0,
        epoch: 0,
        samples: 0,
        weights: fromData(w, [d]),
        intercept: b,
        objective: objective(w, b),
        zeros: zerosOf(w),
        eta: stepSize(1),
        u: 0,
        q: fromData(new Float64Array(d), [d]),
      }
    },
    step: (state, ctx) => {
      const w = Float64Array.from(values(state.weights))
      const q = Float64Array.from(values(state.q))
      let b = state.intercept
      let u = state.u
      let samples = state.samples
      let eta = state.eta
      for (const i of order(ctx.stream)) {
        // scikit-learn counts from t = 1 for the first example.
        eta = stepSize(samples + 1)
        let p = b
        for (let j = 0; j < d; j++) p += w[j] * v[i * d + j]
        const update = -eta * Math.max(-1e12, Math.min(1e12, p - y[i]))
        const decay = rho < 1 ? Math.max(0, 1 - (1 - rho) * eta * alpha) : 1
        const l1Step = eta * alpha * rho
        for (let j = 0; j < d; j++) {
          const sign = Math.sign(w[j])
          w[j] = decay * w[j] + update * v[i * d + j]
          if (rho > 0 && l1Handling === 'subgradient') w[j] -= l1Step * sign
        }
        if (intercept) b += update
        if (rho > 0 && l1Handling === 'truncated') {
          // The cumulative penalty: pull each weight towards zero by what it is owed, never past zero.
          u += l1Step
          for (let j = 0; j < d; j++) {
            const z = w[j]
            if (z > 0) w[j] = Math.max(0, z - (u + q[j]))
            else if (z < 0) w[j] = Math.min(0, z + (u - q[j]))
            q[j] += w[j] - z
          }
        }
        samples++
      }
      const finite = Number.isFinite(b) && w.every(Number.isFinite)
      return {
        t: state.t + 1,
        epoch: state.epoch + 1,
        samples,
        weights: fromData(w, [d]),
        intercept: b,
        objective: finite ? objective(w, b) : NaN,
        zeros: zerosOf(w),
        eta,
        u,
        q: fromData(q, [d]),
        diverged: !finite,
      }
    },
  }
}
