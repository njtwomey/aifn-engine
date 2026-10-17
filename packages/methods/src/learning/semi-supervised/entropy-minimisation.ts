/**
 * Entropy minimisation for a multinomial logistic model on fixed features (Grandvalet and Bengio, 2005, eq. 6).
 *
 * The model is $p(k \mid \xvec) = \mathrm{softmax}_k(\Wmat^\top\phivec(\xvec) + \bvec)$ with $\Wmat$ of shape
 * $d \times K$. The objective is the mean cross-entropy of the $\ell$ labelled points, plus $\lambda$ times the mean
 * Shannon entropy (nats) of the predictions on the $u$ unlabelled points, plus $\frac{\gamma}{2}\lVert \Wmat
 * \rVert^2$ (the bias is not penalised):
 *
 * $$
 * J = -\frac{1}{\ell}\sum_{i} \log p(y_i \mid \xvec_i) + \frac{\lambda}{u}\sum_j \operatorname{H}(p(\cdot \mid
 * \xvec_j)) + \frac{\gamma}{2}\lVert \Wmat \rVert^2 .
 * $$
 *
 * The gradient is closed form: with respect to the logits, a labelled row contributes $(p_{ik} - [k = y_i]) / \ell$
 * and an unlabelled row $-(\lambda/u)\,p_{jk}(\log p_{jk} + \operatorname{H}_j)$. The objective is minimised by
 * L-BFGS. `entropyMinimisationPath` warm-starts each weight $\lambda$ of an increasing list from the previous
 * solution, the slow increase of $\lambda$ that Grandvalet and Bengio (§3) compare with deterministic annealing.
 *
 * Differences from the paper: the paper sums over points and maximises the penalised likelihood; this file minimises
 * means, so $\lambda$ is the weight of one unlabelled point's entropy relative to one labelled point's log loss when
 * $\ell = u$. The paper's smoothness penalty is the weight decay $\gamma$. Pure functions of plain arrays, so a
 * worker can run them.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { minimize } from 'aifn-compute/optim/minimize'
import { packRows, type FeatureRows } from './features'

const OP = 'entropyMinimisation'

/** A semi-supervised classification problem on fixed features. */
export interface EntropyMinimisationProblem {
  /** Features of the labelled points, $\ell$ rows of length $d$. */
  labelled: FeatureRows
  /** The class of each labelled point, an integer in $0, \dots, K - 1$. */
  labels: ArrayLike<number>
  /** Features of the unlabelled points, $u$ rows of length $d$. */
  unlabelled: FeatureRows
  /** The number of classes $K \ge 2$. */
  classes: number
}

/** The weights of the objective. */
export interface EntropyMinimisationWeights {
  /** The weight $\lambda \ge 0$ of the mean unlabelled entropy (0 gives L2-penalised logistic regression). */
  lambda: number
  /** The weight decay $\gamma \ge 0$ on $\Wmat$. */
  l2: number
}

/** A fitted model and the parts of its objective. */
export interface EntropyMinimisationFit {
  /** The parameters: $\Wmat$ row-major ($d \times K$), then $\bvec$ ($K$ values). */
  params: Float64Array
  /** The feature dimension $d$. */
  dim: number
  /** The number of classes $K$. */
  classes: number
  /** The weight $\lambda$ the model was fitted with. */
  lambda: number
  /** The objective $J$ at `params`. */
  objective: number
  /** The mean labelled cross-entropy (nats). */
  crossEntropy: number
  /** The mean entropy of the unlabelled predictions (nats). */
  entropy: number
  /** L-BFGS steps taken. */
  steps: number
  /** Whether L-BFGS reached its gradient tolerance. */
  converged: boolean
}

/**
 * Softmax probabilities and log-probabilities of packed rows under parameters (stable: logits minus their maximum).
 *
 * @param x The rows, packed row-major $n \times d$.
 * @param n The number of rows.
 * @param dim The number of features $d$.
 * @param K The number of classes.
 * @param params $\Wmat$ row-major ($d \times K$), then $\bvec$ ($K$ values).
 * @returns `p` and `logp`, row-major $n \times K$.
 */
function forward(x: Float64Array, n: number, dim: number, K: number, params: ArrayLike<number>) {
  const p = new Float64Array(n * K)
  const logp = new Float64Array(n * K)
  const bias = dim * K
  for (let i = 0; i < n; i++) {
    let max = -Infinity
    for (let k = 0; k < K; k++) {
      let z = params[bias + k]
      for (let c = 0; c < dim; c++) z += x[i * dim + c] * params[c * K + k]
      logp[i * K + k] = z
      if (z > max) max = z
    }
    let total = 0
    for (let k = 0; k < K; k++) total += Math.exp(logp[i * K + k] - max)
    const lse = max + Math.log(total)
    for (let k = 0; k < K; k++) {
      logp[i * K + k] -= lse
      p[i * K + k] = Math.exp(logp[i * K + k])
    }
  }
  return { p, logp }
}

/**
 * Checks a problem: an integer $K \ge 2$, at least one labelled point, one label per labelled row, each a class.
 * Throws `DomainError` or `ShapeError` otherwise.
 *
 * @param problem The problem.
 * @returns The number of classes `K` and the feature dimension `dim` (of the first labelled row).
 */
function checkProblem(problem: EntropyMinimisationProblem) {
  const K = problem.classes
  if (!(K >= 2) || !Number.isInteger(K)) throw new DomainError(OP, `${OP}: classes must be an integer ≥ 2`)
  if (problem.labelled.length === 0) throw new DomainError(OP, `${OP}: no labelled points`)
  if (problem.labels.length !== problem.labelled.length)
    throw new ShapeError(OP, `${OP}: labels do not match the labelled rows`)
  for (let i = 0; i < problem.labels.length; i++) {
    const y = problem.labels[i]
    if (!Number.isInteger(y) || y < 0 || y >= K) throw new DomainError(OP, `${OP}: label ${y} is not a class`)
  }
  return { K, dim: problem.labelled[0].length }
}

/**
 * The objective $J$ and its closed-form gradient as a function of the flat parameter vector ($\Wmat$ row-major,
 * then $\bvec$), in the form the `aifn-compute/optim/minimize` methods take, plus a function that splits $J$ into its
 * parts. Throws `DomainError` for a bad problem (see `entropyMinimisationFit`) or a negative weight.
 *
 * @param problem The labelled and unlabelled features, the labels and the number of classes.
 * @param weights $\lambda$ and the weight decay $\gamma$.
 * @returns `size` ($dK + K$), `dim`, `classes`, `value(params)` (the objective with `grad`) and `parts(params)`
 *   (objective, mean cross-entropy, mean entropy).
 *
 * @example At zero parameters every prediction is uniform
 * const problem = { labelled: [[0], [1]], labels: [0, 1], unlabelled: [[0.5]], classes: 2 }
 * const f = entropyMinimisationObjective(problem, { lambda: 1, l2: 0.1 })
 * print('parts at 0:', f.parts([0, 0, 0, 0]), ' log 2 =', Math.log(2))
 */
export function entropyMinimisationObjective(problem: EntropyMinimisationProblem, weights: EntropyMinimisationWeights) {
  const { K, dim } = checkProblem(problem)
  if (!(weights.lambda >= 0) || !(weights.l2 >= 0)) throw new DomainError(OP, `${OP}: weights must be ≥ 0`)
  const nl = problem.labelled.length
  const nu = problem.unlabelled.length
  const xl = packRows(problem.labelled, dim, OP)
  const xu = packRows(problem.unlabelled, dim, OP)
  const y = Array.from(problem.labels)
  const size = dim * K + K

  const evaluate = (params: ArrayLike<number>, withGrad: boolean) => {
    const grad = withGrad ? new Float64Array(size) : null
    const L = forward(xl, nl, dim, K, params)
    let crossEntropy = 0
    for (let i = 0; i < nl; i++) {
      crossEntropy -= L.logp[i * K + y[i]] / nl
      if (grad)
        for (let k = 0; k < K; k++) {
          const g = (L.p[i * K + k] - (k === y[i] ? 1 : 0)) / nl
          for (let c = 0; c < dim; c++) grad[c * K + k] += xl[i * dim + c] * g
          grad[dim * K + k] += g
        }
    }
    let entropy = 0
    if (nu > 0) {
      const U = forward(xu, nu, dim, K, params)
      for (let j = 0; j < nu; j++) {
        let h = 0
        for (let k = 0; k < K; k++) h -= U.p[j * K + k] * U.logp[j * K + k]
        entropy += h / nu
        if (grad && weights.lambda > 0)
          for (let k = 0; k < K; k++) {
            const g = (-weights.lambda / nu) * U.p[j * K + k] * (U.logp[j * K + k] + h)
            for (let c = 0; c < dim; c++) grad[c * K + k] += xu[j * dim + c] * g
            grad[dim * K + k] += g
          }
      }
    }
    let decay = 0
    for (let q = 0; q < dim * K; q++) {
      decay += 0.5 * weights.l2 * params[q] * params[q]
      if (grad) grad[q] += weights.l2 * params[q]
    }
    return { objective: crossEntropy + weights.lambda * entropy + decay, crossEntropy, entropy, grad }
  }

  return {
    size,
    dim,
    classes: K,
    value: (params: ArrayLike<number>) => {
      const e = evaluate(params, true)
      return { value: e.objective, grad: e.grad as Float64Array }
    },
    parts: (params: ArrayLike<number>) => {
      const e = evaluate(params, false)
      return { objective: e.objective, crossEntropy: e.crossEntropy, entropy: e.entropy }
    },
  }
}

/**
 * Fits the entropy-regularised logistic model by L-BFGS (`minimize` of `aifn-compute/optim/minimize`) from `start`
 * (default all zeros). With $\lambda = 0$ it is L2-penalised multinomial logistic regression. Throws `DomainError`
 * for fewer than two classes, no labelled points, a label that is not a class or a negative weight, and `ShapeError`
 * for mismatched labels, rows of the wrong length or a start of the wrong length.
 *
 * @param problem The labelled and unlabelled features, the labels and the number of classes.
 * @param options $\lambda$, the weight decay, the step budget `maxSteps` (default 300), the gradient `tolerance`
 *   (default $10^{-6}$) and an optional `start`.
 * @returns The fitted parameters with the objective's parts.
 *
 * @example The unlabelled entropy pushes the threshold into the gap
 * const line = [0.4, 0.8, 1.2, 1.6, 2, 2.4, 3.4, 3.8]
 * const problem = { labelled: [[0], [4.2]], labels: [0, 1], unlabelled: line.map((x) => [x]), classes: 2 }
 * for (const lambda of [0, 1]) {
 *   const fit = entropyMinimisationFit(problem, { lambda, l2: 0.2 })
 *   const [w0, w1, b0, b1] = fit.params
 *   print('lambda', lambda, ' threshold:', -(b1 - b0) / (w1 - w0), ' mean entropy:', fit.entropy)
 * }
 */
export function entropyMinimisationFit(
  problem: EntropyMinimisationProblem,
  options: EntropyMinimisationWeights & { maxSteps?: number; tolerance?: number; start?: ArrayLike<number> },
): EntropyMinimisationFit {
  const f = entropyMinimisationObjective(problem, options)
  const x0 = options.start ? Array.from(options.start) : new Array(f.size).fill(0)
  if (x0.length !== f.size) throw new ShapeError(OP, `${OP}: start has ${x0.length} values, expected ${f.size}`)
  const result = minimize((x: Tensor) => f.value(toFlat(x)), x0, {
    method: 'lbfgs',
    maxSteps: options.maxSteps ?? 300,
    tolerance: options.tolerance ?? 1e-6,
  })
  const params = Float64Array.from(toFlat(result.x))
  const parts = f.parts(params)
  return {
    params,
    dim: f.dim,
    classes: f.classes,
    lambda: options.lambda,
    ...parts,
    steps: result.steps,
    converged: result.converged,
  }
}

/**
 * The solution path over an increasing list of weights $\lambda$, each fit started from the previous one (the first
 * from zeros), as in Grandvalet and Bengio's suggestion of increasing $\lambda$ slowly. Plain arrays in and out, so a
 * worker can run it (31 weights on 200 points take about 90 ms).
 *
 * @param problem The labelled and unlabelled features, the labels and the number of classes.
 * @param options The weights `lambdas` in the order to visit them, the weight decay `l2`, and the step budget
 *   `maxSteps` and gradient `tolerance` of each fit.
 * @returns One fit per weight, in the order given.
 *
 * @example Mean unlabelled entropy along the path
 * const line = [0.4, 0.8, 1.2, 1.6, 2, 2.4, 3.4, 3.8]
 * const problem = { labelled: [[0], [4.2]], labels: [0, 1], unlabelled: line.map((x) => [x]), classes: 2 }
 * const path = entropyMinimisationPath(problem, { lambdas: [0, 0.5, 1, 2], l2: 0.2 })
 * print('entropy:', path.map((f) => f.entropy))
 */
export function entropyMinimisationPath(
  problem: EntropyMinimisationProblem,
  options: { lambdas: readonly number[]; l2: number; maxSteps?: number; tolerance?: number },
): EntropyMinimisationFit[] {
  const fits: EntropyMinimisationFit[] = []
  let start: Float64Array | undefined
  for (const lambda of options.lambdas) {
    const fit = entropyMinimisationFit(problem, {
      lambda,
      l2: options.l2,
      maxSteps: options.maxSteps,
      tolerance: options.tolerance,
      start,
    })
    fits.push(fit)
    start = fit.params
  }
  return fits
}

/**
 * Class probabilities of feature rows under fitted parameters.
 *
 * @param rows Feature rows, each of length `fit.dim`.
 * @param fit A fit (or anything with `params`, `dim` and `classes`).
 * @returns Row-major probabilities, $n \times K$.
 *
 * @example Probabilities of a hand-made two-class model
 * const fit = { params: Float64Array.from([-1, 1, 1, -1]), dim: 1, classes: 2 }
 * print(entropyMinimisationProbabilities([[0], [1], [2]], fit))
 */
export function entropyMinimisationProbabilities(
  rows: FeatureRows,
  fit: Pick<EntropyMinimisationFit, 'params' | 'dim' | 'classes'>,
): Float64Array {
  return forward(packRows(rows, fit.dim, OP), rows.length, fit.dim, fit.classes, fit.params).p
}
