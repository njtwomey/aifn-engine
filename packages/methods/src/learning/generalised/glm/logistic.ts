/**
 * Logistic regression (binary and multinomial) fitted by Newton's method: the binary model is the binomial GLM with
 * the logit link, fitted by the generalised models' penalised IRLS (`irls`, one definition); the softmax model by
 * Newton's method on the multinomial log-likelihood (`softmaxNewton`), whose Hessian couples the classes.
 *
 * Both minimise the penalised negative log-likelihood
 * $\sum_i -\log p(y_i \mid \xvec_i) + \tfrac12\lambda\lVert\Wmat\rVert^2$ with the intercepts unpenalised,
 * scikit-learn's `LogisticRegression` objective with $C = 1/\lambda$. The softmax weights are a $p \times K$ matrix,
 * row $j$ for design column $j$ (the intercept row last).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { lstsq } from 'aifn-compute/numerics/linalg'
import { sigmoid, softmax } from 'aifn-compute/numerics/special'
import { dense, fromData, logsumexp, matmul, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import type { Decides, Expects, Fitted, Predicts, Samples, Scores, Trained } from 'aifn-compute/learning/estimators'
import type { Estimator, FitOptions, Supervised } from 'aifn-compute/learning/estimators'
import { bernoulliPredictive, categoricalPredictive, type AnyUnivariate } from 'aifn-compute/learning/estimators'
import { withExpectation, withSampling } from 'aifn-compute/learning/estimators'
import { matrixShape, targetValues } from 'aifn-compute/learning/estimators'
import { binomialFamily, link } from 'aifn-compute/probability/likelihoods'
import { irls, type IrlsState } from '../irls'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const values = dense.data

/** Hyperparameters of `logisticRegression`. */
export interface LogisticRegressionParams {
  /**
   * L2 penalty $\lambda$ in $\sum_i -\log p(y_i \mid \xvec_i) + \tfrac12\lambda\lVert\Wmat\rVert^2$ (intercepts are not
   * penalised). scikit-learn's `C` is $1/\lambda$. Default 1, scikit-learn's default. A negative value throws.
   */
  l2?: number
  /** Fit intercepts (default true). */
  intercept?: boolean
  /** Use the softmax model even for two classes (default: only for three or more). */
  multinomial?: boolean
  /**
   * Convergence tolerance: for the softmax model, on half the squared Newton decrement (an estimate of $f - f^*$);
   * for the binary model, on the relative change of the penalised deviance (default 1e-12).
   */
  tolerance?: number
  /** Most Newton steps (default 100). */
  maxSteps?: number
}

/** The problem the softmax Newton method solves: the design with a trailing column of ones when fitting intercepts. */
export interface SoftmaxProblem {
  /** Design matrix $\Xmat$, $n \times p$ ($p = d + 1$ with an intercept column last). */
  design: Tensor
  /** Class labels $0, \dots, K - 1$, length $n$. */
  labels: Tensor
  /** Number of classes $K$ (one weight column each). */
  columns: number
  /** Penalty $\lambda$ on every weight but the intercept row's. */
  l2: number
  /** True when the last design column is the (unpenalised) intercept. */
  intercept: boolean
  /** Converged when half the squared Newton decrement, $\lambda(\Wmat)^2/2$, is at most this. */
  tolerance: number
}

/** One Newton state of the softmax model: the weights and everything computed at them. */
export interface SoftmaxNewtonState extends Status {
  /** Newton steps taken. */
  t: number
  /** Weights, $p \times K$: row $j$ for design column $j$ (the intercept last), column $k$ for class $k$. */
  weights: Tensor
  /** Penalised negative log-likelihood at `weights`. */
  loss: number
  /** Gradient $\gvec$ of the loss, $p \times K$. */
  grad: Tensor
  /**
   * Newton direction $\Hmat^{-1}\gvec$, $p \times K$ (minimum norm when the Hessian $\Hmat$ is singular, as the
   * model's intercepts make it).
   */
  direction: Tensor
  /** Newton decrement $\lambda(\Wmat) = \sqrt{\gvec^\top\Hmat^{-1}\gvec}$ (Boyd and Vandenberghe, 2004, §9.5.1). */
  decrement: number
  /** Step length taken to reach this state by backtracking (1 is the full Newton step; 0 at the start). */
  stepSize: number
  /** $\lambda(\Wmat)^2/2 \le$ `tolerance`. */
  converged: boolean
  /** The loss is not finite. */
  diverged: boolean
  /** The line search could not decrease the loss (rounding level reached). */
  stalled: boolean
}

/**
 * The penalised negative log-likelihood at a point (`loss`), its gradient (`gradient`, $p K$ values, row-major
 * $p \times K$) and its Hessian (`hessian`, $pK \times pK$ row-major, or null when not asked for).
 */
type Evaluated = { loss: number; gradient: Float64Array; hessian: Float64Array | null }

/**
 * Loss, gradient and (optionally) Hessian of the penalised negative log-likelihood at $\Wmat$. The Hessian entry for
 * weights $(a, k)$ and $(b, l)$ is $\sum_i x_{ia} x_{ib} \pi_{ik}(\delta_{kl} - \pi_{il})$, plus $\lambda$ on the
 * diagonal of the penalised rows.
 *
 * @param problem The design, labels, class count and penalty.
 * @param W The weights, row-major $p \times K$ ($pK$ values); read only.
 * @param withHessian Whether to form the Hessian (the line search needs only the loss).
 * @returns The loss, gradient and Hessian.
 */
function evaluate(problem: SoftmaxProblem, W: Float64Array, withHessian: boolean): Evaluated {
  const X = values(problem.design)
  const y = values(problem.labels)
  const [n, p] = problem.design.shape
  const C = problem.columns
  const P = p * C
  const gradient = new Float64Array(P)
  const hessian = withHessian ? new Float64Array(P * P) : null
  let loss = 0
  // η = X W [n, C]; log Σₖ e^{ηₖ} and the class probabilities softmax(η) per row.
  const etaAll = matmul(problem.design, fromData(W, [p, C]))
  const ETA = values(etaAll)
  const LSE = values(logsumexp(etaAll, 1))
  const PROB = values(softmax(etaAll))
  for (let i = 0; i < n; i++) {
    const yi = y[i]
    const prob = PROB.subarray(i * C, (i + 1) * C)
    {
      loss += LSE[i] - ETA[i * C + yi]
      for (let a = 0; a < p; a++) {
        const xa = X[i * p + a]
        for (let k = 0; k < C; k++) gradient[a * C + k] += xa * (prob[k] - (k === yi ? 1 : 0))
      }
      if (hessian) {
        // H[(a,k),(b,l)] = Σᵢ x_ia x_ib p_k (δ_kl − p_l).
        for (let a = 0; a < p; a++) {
          for (let k = 0; k < C; k++) {
            const row = (a * C + k) * P
            for (let b = 0; b <= a; b++) {
              const xx = X[i * p + a] * X[i * p + b]
              for (let l = 0; l < C; l++) {
                if (b * C + l > a * C + k) break
                hessian[row + b * C + l] += xx * prob[k] * ((k === l ? 1 : 0) - prob[l])
              }
            }
          }
        }
      }
    }
  }
  const penalised = problem.intercept ? p - 1 : p
  for (let a = 0; a < penalised; a++) {
    for (let k = 0; k < C; k++) {
      const j = a * C + k
      loss += 0.5 * problem.l2 * W[j] * W[j]
      gradient[j] += problem.l2 * W[j]
      if (hessian) hessian[j * P + j] += problem.l2
    }
  }
  if (hessian) for (let r = 0; r < P; r++) for (let c = r + 1; c < P; c++) hessian[r * P + c] = hessian[c * P + r]
  return { loss, gradient, hessian }
}

/**
 * The state at $\Wmat$: loss, gradient, the Newton direction (the minimum-norm least-squares solution of
 * $\Hmat\dvec = \gvec$) and the decrement. With a non-finite loss the direction is zero and the decrement NaN.
 *
 * @param problem The design, labels, class count, penalty and tolerance.
 * @param W The weights, row-major $p \times K$ ($pK$ values).
 * @param t The step count to record.
 * @param stepSize The step length that reached $\Wmat$, to record (0 at the start).
 * @param stalled Whether the line search failed to reach a lower loss.
 * @returns The state.
 */
function stateAt(
  problem: SoftmaxProblem,
  W: Float64Array,
  t: number,
  stepSize: number,
  stalled = false,
): SoftmaxNewtonState {
  const [, p] = problem.design.shape
  const C = problem.columns
  const P = p * C
  const { loss, gradient, hessian } = evaluate(problem, W, true)
  const shape = [p, C]
  let direction = new Float64Array(P)
  let decrement = NaN
  if (Number.isFinite(loss)) {
    direction = values(lstsq(fromData(hessian!, [P, P]), fromData(gradient, [P])).x).slice()
    let gd = 0
    for (let j = 0; j < P; j++) gd += gradient[j] * direction[j]
    decrement = Math.sqrt(Math.max(gd, 0))
  }
  return {
    t,
    weights: fromData(W, shape),
    loss,
    grad: fromData(gradient, shape),
    direction: fromData(direction, shape),
    decrement,
    stepSize,
    converged: (decrement * decrement) / 2 <= problem.tolerance,
    diverged: !Number.isFinite(loss),
    stalled,
  }
}

/**
 * Newton's method for (penalised) softmax regression as a traceable algorithm (Hastie, Tibshirani and Friedman, 2009,
 * §4.4.1). Each step backtracks from the full Newton step along the direction $\Deltamat$, halving $t$, until the
 * Armijo condition $f(\Wmat - t\Deltamat) \le f(\Wmat) - 10^{-4} t \lambda^2$ holds (Boyd and Vandenberghe, 2004,
 * Algorithm 9.5), which makes the method globally convergent; when no $t$ above $10^{-12}$ does, the state is marked
 * `stalled` and the run stops. `init` takes starting weights $p \times K$ (default zeros).
 *
 * @param problem The design (with its intercept column, if any), labels, number of classes, penalty and tolerance.
 * @returns The algorithm, for `run` or `trace`.
 *
 * @example Three classes: the loss and Newton decrement fall quadratically
 * const s = stream(1)
 * const x = normals(s, [200, 1])
 * const labels = categorical(s, exp(matmul(x, tensor([[0, 1.5, -1.5]]))))
 * const design = concat([x, ones([200, 1])], 1)
 * const problem = { design, labels, columns: 3, l2: 0, intercept: true, tolerance: 1e-12 }
 * const record = { loss: (state) => state.loss, decrement: (state) => state.decrement }
 * const t = trace(softmaxNewton(problem), {}, 20, { record })
 * print('loss by step =', t.series.loss)
 * print('decrement by step =', t.series.decrement)
 * print('weights (rows x, intercept) =', t.final.weights)
 */
export function softmaxNewton(problem: SoftmaxProblem): Algorithm<{ weights?: Tensor }, SoftmaxNewtonState> {
  return {
    name: 'softmax-newton',
    init: ({ weights } = {}) => {
      const [, p] = problem.design.shape
      const W = weights ? values(weights).slice() : new Float64Array(p * problem.columns)
      return stateAt(problem, W, 0, 0)
    },
    step: (state) => {
      const W = values(state.weights)
      const d = values(state.direction)
      const slope = state.decrement * state.decrement
      let t = 1
      while (t > 1e-12) {
        const next = Float64Array.from(W, (w, j) => w - t * d[j])
        const { loss } = evaluate(problem, next, false)
        if (loss <= state.loss - 1e-4 * t * slope) return stateAt(problem, next, state.t + 1, t)
        t /= 2
      }
      return { ...state, t: state.t + 1, stepSize: 0, stalled: true }
    },
    done: (state) => state.stalled === true,
  }
}

/** A fitted logistic regression. */
export interface LogisticRegressionModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Scores<Tensor>,
    Trained<IrlsState | SoftmaxNewtonState> {
  /** The brand of a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'logistic-regression'
  /** Number of classes $K$ (labels $0, \dots, K - 1$; at least 2). */
  readonly classes: number
  /** True for the softmax model with one weight column per class. */
  readonly multinomial: boolean
  /** Coefficients: $d$ for the binary model, $d \times K$ for the multinomial one. */
  readonly weights: Tensor
  /**
   * Intercepts: a scalar tensor for the binary model, $K$ for the multinomial one (all 0 without intercepts).
   */
  readonly intercept: Tensor
  /** The final penalised negative log-likelihood. */
  readonly loss: number
  /** Whether the Newton (IRLS) run met its tolerance within `maxSteps`. */
  readonly converged: boolean
  /** Newton (IRLS) steps taken. */
  readonly steps: number
  /** The penalty $\lambda$ the model was fitted with. */
  readonly l2: number
}

/**
 * Logistic regression for labels $0, \dots, K - 1$: Bernoulli with
 * $P(y = 1 \mid \xvec) = \sigma(\xvec^\top\wvec + b)$ for two classes, softmax over $K$ weight columns otherwise,
 * with an L2 penalty on the weights. The binary model is fitted by `irls` (binomial family, logit link, penalty
 * $\lambda$ on the weights), the softmax model by `softmaxNewton`; the run is kept in `training` (series `loss`, the
 * penalised negative log-likelihood, and for the softmax model `decrement` and `stepSize`). The multinomial Hessian is
 * singular along "add a constant to every intercept"; the minimum-norm Newton step keeps $\sum_k b_k = 0$, as
 * scikit-learn's solution has.
 *
 * Capabilities: `forward` and `score` (logits: $m$ binary, $m \times K$ multinomial), `decide` (the most probable
 * class; logit $> 0$ when binary), `predictive` (Bernoulli or categorical), `expect` ($P(y = 1)$, or $\expect[k]$),
 * `sample`. `fit` throws `DomainError` for labels that are not non-negative integers and `ShapeError` when the rows of
 * `x` and the labels differ in number.
 *
 * @param params The penalty, intercept, model choice and Newton controls; a negative `l2` throws `DomainError` here.
 * @returns An estimator whose `fit` takes `{ x, y }` ($n \times d$ inputs, $n$ integer labels) and returns the model.
 *
 * @example Recovering known coefficients from simulated labels
 * const s = stream(1)
 * const x = normals(s, [500, 2])
 * const y = bernoulli(s, div(1, add(1, exp(neg(add(matmul(x, tensor([2, -1])), 0.5))))))
 * const model = logisticRegression({ l2: 0 }).fit({ x, y })
 * print('weights (true 2, -1) =', model.weights)
 * print('intercept (true 0.5) =', model.intercept)
 * print('Newton steps =', model.steps, ' converged =', model.converged)
 *
 * @example The penalty shrinks the weights towards zero
 * const s = stream(1)
 * const x = normals(s, [40, 2])
 * const y = bernoulli(s, div(1, add(1, exp(neg(matmul(x, tensor([2, -1])))))))
 * for (const l2 of [0, 1, 10]) print('l2 =', l2, ' weights =', logisticRegression({ l2 }).fit({ x, y }).weights)
 */
export function logisticRegression(
  params: LogisticRegressionParams = {},
): Estimator<Supervised<Tensor, Tensor>, LogisticRegressionModel> {
  const { l2 = 1, intercept = true, multinomial = false, tolerance = 1e-12, maxSteps = 100 } = params
  if (!(l2 >= 0)) throw new DomainError('logisticRegression', 'logisticRegression: l2 must be non-negative')
  return {
    name: 'logistic-regression',
    params: { l2, intercept, multinomial, tolerance, maxSteps },
    fit({ x, y }, options: FitOptions = {}) {
      const [n, d] = matrixShape(x, 'logisticRegression')
      const target = targetValues(y, 'logisticRegression')
      if (target.length !== n)
        throw new ShapeError('logisticRegression', `logisticRegression: ${n} rows of x but ${target.length} labels`)
      let K = 0
      for (const v of target) {
        if (!(Number.isInteger(v) && v >= 0))
          throw new DomainError('logisticRegression', 'logisticRegression: labels must be integers 0 … K−1')
        K = Math.max(K, v + 1)
      }
      K = Math.max(K, 2)
      const softmax = multinomial || K > 2
      const C = softmax ? K : 1
      const p = intercept ? d + 1 : d
      const X = values(x)
      const design = new Float64Array(n * p)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) design[i * p + j] = X[i * d + j]
        if (intercept) design[i * p + d] = 1
      }
      const traceOptions = {
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
      }
      let training: Trace<IrlsState> | Trace<SoftmaxNewtonState>
      let W: Float64Array
      let loss: number
      let converged: boolean
      let steps: number
      if (C === 1) {
        // The binary model is the binomial GLM with the logit link: IRLS on the penalised deviance D + λ‖w‖², which
        // is twice the penalised negative log-likelihood Σ −log p(yᵢ | xᵢ) + ½λ‖w‖².
        const penalty = new Float64Array(p * p)
        for (let j = 0; j < d; j++) penalty[j * p + j] = l2
        const run = trace(
          irls({
            design: fromData(design, [n, p]),
            y: fromData(Float64Array.from(target), [n]),
            family: binomialFamily(),
            link: link('logit'),
            penalty: l2 > 0 ? fromData(penalty, [p, p]) : undefined,
            tolerance,
          }),
          {},
          maxSteps,
          {
            ...traceOptions,
            record: {
              loss: (s) => s.penalisedDeviance / 2,
              ...(options.trace?.record as Record<string, (s: IrlsState, t: number) => number> | undefined),
            },
          },
        )
        const final = run.final
        W = final.coefficients ? Float64Array.from(values(final.coefficients)) : new Float64Array(p)
        loss = final.penalisedDeviance / 2
        converged = final.converged === true
        steps = final.t
        training = run
      } else {
        const problem: SoftmaxProblem = {
          design: fromData(design, [n, p]),
          labels: fromData(Float64Array.from(target), [n]),
          columns: C,
          l2,
          intercept,
          tolerance,
        }
        const run = trace(softmaxNewton(problem), {}, maxSteps, {
          ...traceOptions,
          record: {
            loss: (s) => s.loss,
            decrement: (s) => s.decrement,
            stepSize: (s) => s.stepSize,
            ...(options.trace?.record as Record<string, (s: SoftmaxNewtonState, t: number) => number> | undefined),
          },
        })
        const final = run.final
        W = values(final.weights)
        loss = final.loss
        converged = final.converged
        steps = final.t
        training = run
      }
      const coef = new Float64Array(d * C)
      const bias = new Float64Array(C)
      for (let j = 0; j < d; j++) for (let k = 0; k < C; k++) coef[j * C + k] = W[j * C + k]
      if (intercept) for (let k = 0; k < C; k++) bias[k] = W[d * C + k]
      const forward = (input: Tensor): Tensor => {
        const [m, cols] = matrixShape(input, 'logisticRegression.forward')
        if (cols !== d)
          throw new ShapeError('logisticRegression', `logisticRegression: fitted on ${d} features, given ${cols}`)
        const Z = values(input)
        const out = new Float64Array(m * C)
        for (let i = 0; i < m; i++) {
          for (let k = 0; k < C; k++) {
            let s = bias[k]
            for (let j = 0; j < d; j++) s += Z[i * d + j] * coef[j * C + k]
            out[i * C + k] = s
          }
        }
        return fromData(out, C === 1 ? [m] : [m, C])
      }
      const predictive = (input: Tensor): AnyUnivariate => {
        const eta = values(forward(input))
        if (C === 1) {
          return bernoulliPredictive(
            fromData(
              Float64Array.from(eta, (e) => sigmoid(e)),
              [eta.length],
            ),
          )
        }
        const m = eta.length / C
        const probs = new Float64Array(eta.length)
        for (let i = 0; i < m; i++) {
          let mx = -Infinity
          for (let k = 0; k < C; k++) mx = Math.max(mx, eta[i * C + k])
          let z = 0
          for (let k = 0; k < C; k++) z += probs[i * C + k] = Math.exp(eta[i * C + k] - mx)
          for (let k = 0; k < C; k++) probs[i * C + k] /= z
        }
        return categoricalPredictive(fromData(probs, [m, C]))
      }
      const decide = (input: Tensor): Tensor => {
        const eta = values(forward(input))
        const m = eta.length / C
        const out = new Int32Array(m)
        for (let i = 0; i < m; i++) {
          if (C === 1) out[i] = eta[i] > 0 ? 1 : 0
          else for (let k = 1; k < C; k++) if (eta[i * C + k] > eta[i * C + out[i]]) out[i] = k
        }
        return fromData(out, [m])
      }
      const base = {
        kind: 'model' as const,
        name: 'logistic-regression' as const,
        classes: K,
        multinomial: softmax,
        weights: fromData(coef, C === 1 ? [d] : [d, C]),
        intercept: fromData(bias, C === 1 ? [] : [C]),
        loss,
        converged,
        steps,
        l2,
        training,
        forward,
        score: forward,
        decide,
        predictive,
      }
      return withSampling(withExpectation(base))
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'logisticRegression',
    module: 'learning/generalised/glm',
    name: 'Logistic regression',
    summary:
      'Binary or softmax logistic regression fitted by Newton steps, with a Bernoulli or categorical predictive.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'score', 'sample'],
    hyper: space({
      l2: real(0, 100, { default: 1, label: 'L2 penalty' }),
      intercept: bool({ default: true }),
      multinomial: bool(),
      tolerance: real(1e-14, 1e-2, { default: 1e-12, scale: 'log' }),
      maxSteps: int(1, 1000, { default: 100 }),
    }),
    notes: ['logistic-regression'],
    cite: ['hastie2009'],
  },
  logisticRegression,
)
