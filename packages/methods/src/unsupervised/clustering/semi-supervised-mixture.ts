/**
 * Semi-supervised expectation-maximisation for a Gaussian class-conditional model
 * $p(\xvec, y) = \pi_y \Gauss(\xvec \mid \muvec_y, \Sigmamat_y)$, one component per class, fitted to labelled and
 * unlabelled points together (Nigam, McCallum, Thrun and Mitchell, 2000, "Text classification from labeled and
 * unlabeled documents using EM", Machine Learning 39, §4 and §5.3.1, there for naive Bayes; Cozman, Cohen and Cirelo,
 * 2003, for Gaussian classes and their failure under misspecification; Zhu and Goldberg, 2009, §3.2).
 *
 * The responsibilities of a labelled point are clamped to the one-hot vector of its class; those of an unlabelled
 * point are its posterior class probabilities at the current parameters. The M-step (the `maximisation` of
 * `gaussianMixtureSteps`, with row weights) weights every unlabelled point by $\lambda \in [0, 1]$ (EM-$\lambda$), so
 * the objective
 *
 * $$
 *   \sum_{i \in L} \log \pi_{y_i} \Gauss(\xvec_i \mid \muvec_{y_i}, \Sigmamat_{y_i})
 *   + \lambda \sum_{i \in U} \log \sum_k \pi_k \Gauss(\xvec_i \mid \muvec_k, \Sigmamat_k)
 * $$
 *
 * never decreases. $\lambda = 0$ gives the labelled-only fit and $\lambda = 1$ plain semi-supervised EM. Covariances
 * are `'tied'` (one shared full matrix, the model behind linear discriminant analysis), `'full'` (one per class, behind
 * quadratic discriminant analysis), `'diagonal'` or `'spherical'`; `regularisation` is added to every variance in each
 * M-step, as scikit-learn's `reg_covar`, and `varianceFloor` raises every variance to at least that value, so a class
 * with one or two labels keeps a proper density. State 0 is the labelled-only fit (`labelledOnlyFit`): either the
 * pooled within-class covariance of the labelled points plus `ridge` (`start: 'pooled'`, the default) or an M-step of
 * the chosen covariance on the labelled points alone (`start: 'per-class'`, Nigam's start). The objective is reported
 * per unit of weight, divided by $n_L + \lambda n_U$, or per row, divided by $n$ (`objective: 'per-row'`). The
 * diagonal form with the per-class start, a floor, no regularisation and the per-row objective is the EM of the
 * semi-supervised-in-practice notes; the tied and full forms with the pooled start are those of the generative notes.
 * Parameters and states are flat `Float64Array`s, so a worker can post them.
 */

import type { MatrixLike, Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { cholesky, choleskyLogDet, choleskySolve } from 'aifn-compute/numerics/linalg'
import { maximisation, type CovarianceType } from './mixture'

/** The data of a fit. */
export interface SemiSupervisedMixtureProblem {
  /** The inputs, $n \times d$, labelled and unlabelled together. */
  x: MatrixLike
  /** One entry per row: its class in $0, \dots, K - 1$, or $-1$ when unlabelled. */
  labels: ArrayLike<number>
  /** The number of classes $K \ge 2$. */
  classes: number
}

/** Options of `semiSupervisedMixtureSteps` and `labelledOnlyFit`. */
export interface SemiSupervisedMixtureOptions {
  /** How the class covariances are shaped and shared (default `'tied'`). */
  covariance?: CovarianceType
  /** The weight $\lambda \in [0, 1]$ of each unlabelled point (default 1: plain EM; 0: the labelled-only fit). */
  unlabelledWeight?: number
  /** Added to every variance in each M-step (default $10^{-6}$). */
  regularisation?: number
  /** Every variance is raised to at least this after each M-step and at the start (default 0: none). */
  varianceFloor?: number
  /**
   * The labelled-only start: `'pooled'` (default), the tied pooled covariance of the labelled points plus `ridge`;
   * `'per-class'`, an M-step of `covariance` on the labelled points alone (with `regularisation` and the floor).
   */
  start?: 'pooled' | 'per-class'
  /** Added to every variance of the pooled start (default 0.1), so that a few labels give an invertible matrix. */
  ridge?: number
  /** `'weighted'` (default): the objective divided by $n_L + \lambda n_U$; `'per-row'`: divided by $n$. */
  objective?: 'weighted' | 'per-row'
  /** Converged when the objective changes by less than this (default $10^{-8}$). */
  tolerance?: number
}

/** Class-conditional Gaussian parameters, plain data. */
export interface ClassGaussians {
  /** $\pi_k$, $K$ values summing to 1. */
  weights: Float64Array
  /** $\muvec_k$, row-major $K \times d$: class $k$'s mean is entries $kd$ to $kd + d - 1$. */
  means: Float64Array
  /** $\Sigmamat_k$, row-major $K \times d \times d$ (all equal when tied). */
  covariances: Float64Array
  /** The input dimension $d$. */
  dimension: number
}

/** One EM state. */
export interface SemiSupervisedMixtureState extends ClassGaussians, Status {
  /** EM iterations done; state 0 is the labelled-only fit (or the start given). */
  t: number
  /** Responsibilities at these parameters, row-major $n \times K$: one-hot for labelled rows. */
  responsibilities: Float64Array
  /** The weighted objective, labelled plus $\lambda$ times unlabelled, per unit of weight or per row. */
  objective: number
  /** Mean $\log p(\xvec_i, y_i)$ over the labelled rows (NaN when there are none). */
  labelledLogLikelihood: number
  /** Mean $\log p(\xvec_i)$ over the unlabelled rows (NaN when there are none). */
  unlabelledLogLikelihood: number
  /** Change in `objective` from the previous state (NaN at the start). */
  change: number
  /** The objective changed by less than the tolerance; at the start, $\lambda = 0$ with the per-class start. */
  converged: boolean
  /** The objective is not finite. */
  diverged: boolean
  /** The diagonal jitter needed to factor each class covariance, $K$ values (0 when positive definite). */
  jitter: Float64Array
}

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * $\log \pi_k + \log \Gauss(\xvec_i \mid \muvec_k, \Sigmamat_k)$ for every row and class, by one Cholesky factor per
 * class, with the jitter each factor needed.
 *
 * @param params The class parameters.
 * @param X The rows, row-major $n \times d$.
 * @param n The number of rows.
 * @returns `joint`, row-major $n \times K$, and `jitter`, $K$ values.
 */
function logJoint(params: ClassGaussians, X: Float64Array, n: number): { joint: Float64Array; jitter: Float64Array } {
  const d = params.dimension
  const K = params.weights.length
  const joint = new Float64Array(n * K)
  const jitter = new Float64Array(K)
  for (let k = 0; k < K; k++) {
    const factor = cholesky(fromData(params.covariances.slice(k * d * d, (k + 1) * d * d), [d, d]))
    jitter[k] = factor.jitter
    const logDet = choleskyLogDet(factor.L) as number
    // Deviations as columns, d × n, so that one solve gives Σ⁻¹ δ for every row.
    const D = new Float64Array(d * n)
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) D[j * n + i] = X[i * d + j] - params.means[k * d + j]
    const sol = dense.data(choleskySolve(factor.L, fromData(D, [d, n])) as Tensor)
    const base = Math.log(params.weights[k]) - 0.5 * (d * LOG_2PI + logDet)
    for (let i = 0; i < n; i++) {
      let m = 0
      for (let j = 0; j < d; j++) m += D[j * n + i] * sol[j * n + i]
      joint[i * K + k] = base - 0.5 * m
    }
  }
  return { joint, jitter }
}

/**
 * The rows of a matrix argument, checked against the parameters' dimension.
 *
 * @param params The class parameters.
 * @param x The rows, $n \times d$.
 * @param where The caller's name, for error messages.
 * @returns The rows, row-major, and their number $n$.
 */
function rowsFor(params: ClassGaussians, x: MatrixLike, where: string): { X: Float64Array; n: number } {
  const { data, m, n: d } = dense.toMatrixF64(x, where)
  if (d !== params.dimension)
    throw new ShapeError(where, `${where}: x has ${d} columns, the parameters ${params.dimension}`)
  return { X: data, n: m }
}

/**
 * $\log \pi_k + \log \Gauss(\xvec_i \mid \muvec_k, \Sigmamat_k)$ for every row of `x` and every class. A covariance
 * that is not positive definite is factored with jitter. Throws `ShapeError` when `x` does not have $d$ columns.
 *
 * @param params The class parameters, as `labelledOnlyFit` or a state of `semiSupervisedMixtureSteps` holds them.
 * @param x The rows, $n \times d$.
 * @returns Row-major $n \times K$ log joint densities.
 *
 * @example Two classes on a line
 * const params = { weights: Float64Array.from([0.5, 0.5]), means: Float64Array.from([-1, 1]),
 *   covariances: Float64Array.from([1, 1]), dimension: 1 }
 * print(classLogJoint(params, [[0], [1]]))
 */
export function classLogJoint(params: ClassGaussians, x: MatrixLike): Float64Array {
  const { X, n } = rowsFor(params, x, 'classLogJoint')
  return logJoint(params, X, n).joint
}

/**
 * Posterior class probabilities $p(y = k \mid \xvec_i)$ by Bayes' rule, row-major $n \times K$. Throws `ShapeError`
 * when `x` does not have $d$ columns.
 *
 * @param params The class parameters.
 * @param x The rows, $n \times d$.
 * @returns Row-major $n \times K$ probabilities, each row summing to 1.
 *
 * @example Halfway between equal classes the posterior is even
 * const params = { weights: Float64Array.from([0.5, 0.5]), means: Float64Array.from([-1, 1]),
 *   covariances: Float64Array.from([1, 1]), dimension: 1 }
 * print(classPosterior(params, [[0], [1]]))
 */
export function classPosterior(params: ClassGaussians, x: MatrixLike): Float64Array {
  const lj = classLogJoint(params, x)
  const K = params.weights.length
  const out = new Float64Array(lj.length)
  for (let i = 0; i < lj.length / K; i++) {
    let top = -Infinity
    for (let k = 0; k < K; k++) top = Math.max(top, lj[i * K + k])
    let z = 0
    for (let k = 0; k < K; k++) z += Math.exp(lj[i * K + k] - top)
    for (let k = 0; k < K; k++) out[i * K + k] = Math.exp(lj[i * K + k] - top) / z
  }
  return out
}

/**
 * The inputs and labels of a problem, checked: the rows row-major, their number and width, and the labels. Throws
 * `ShapeError` when `x` and `labels` differ in rows and `DomainError` for a label that is neither $-1$ nor a class.
 *
 * @param problem The problem.
 * @param where The caller's name, for error messages.
 * @returns `X`, `n`, `d`, `K` and the labels as int32 ($-1$ for unlabelled).
 */
function problemData(problem: SemiSupervisedMixtureProblem, where: string) {
  const { data: X, m: n, n: d } = dense.toMatrixF64(problem.x, where)
  const K = problem.classes
  if (problem.labels.length !== n) throw new ShapeError(where, `${where}: x and labels differ in rows`)
  const labels = Int32Array.from(problem.labels, (c) => {
    if (c >= K || (c >= 0 && !Number.isInteger(c)))
      throw new DomainError(where, `${where}: labels must be -1 or 0, ..., ${K - 1}`)
    return c >= 0 ? c : -1
  })
  return { X, n, d, K, labels }
}

/**
 * The labelled-only fit that semi-supervised EM starts from: with `start: 'pooled'` (default) the labelled class
 * frequencies and means and the pooled within-class covariance of the labelled points plus `ridge` times the identity,
 * shared by every class; with `start: 'per-class'` an M-step of `covariance` on the labelled points alone (with
 * `regularisation`). The variance floor applies to both. Throws `DomainError` when a class has no labelled point.
 *
 * @param problem The inputs, their labels ($-1$ for unlabelled) and the number of classes.
 * @param options The start, the covariance shape, `ridge`, `regularisation` and `varianceFloor`; the rest of
 *   `SemiSupervisedMixtureOptions` is ignored.
 * @returns The class parameters.
 *
 * @example Pooled and per-class starts from four labels on a line
 * const problem = { x: [[0], [2], [10], [14], [5]], labels: [0, 0, 1, 1, -1], classes: 2 }
 * print('pooled covariances:', labelledOnlyFit(problem).covariances)
 * const perClass = labelledOnlyFit(problem, { start: 'per-class', covariance: 'diagonal', regularisation: 0 })
 * print('per class:', perClass.covariances)
 */
export function labelledOnlyFit(
  problem: SemiSupervisedMixtureProblem,
  options: SemiSupervisedMixtureOptions = {},
): ClassGaussians {
  const where = 'labelledOnlyFit'
  const { X, n, d, K, labels } = problemData(problem, where)
  const {
    start = 'pooled',
    covariance = 'tied',
    ridge = 0.1,
    regularisation = 1e-6,
    varianceFloor: floor = 0,
  } = options
  const r = new Float64Array(n * K)
  const w = new Float64Array(n)
  const count = new Array<number>(K).fill(0)
  for (let i = 0; i < n; i++)
    if (labels[i] >= 0) {
      r[i * K + labels[i]] = 1
      w[i] = 1
      count[labels[i]]++
    }
  count.forEach((c, k) => {
    if (c === 0) throw new DomainError(where, `${where}: class ${k} has no labelled point`)
  })
  const p =
    start === 'pooled'
      ? maximisation(X, n, d, K, r, 'tied', ridge, { rowWeights: w, pad: 0, floor })
      : maximisation(X, n, d, K, r, covariance, regularisation, { rowWeights: w, pad: 0, floor })
  return { weights: p.weights, means: p.means, covariances: p.covs, dimension: d }
}

/**
 * Semi-supervised EM for Gaussian classes as a traceable algorithm; see the file comment. `init` takes starting
 * parameters, or `undefined` for `labelledOnlyFit`; each step is an M-step on the current responsibilities followed by
 * an E-step. A covariance that is not positive definite is factored with jitter, reported in `jitter`. Throws
 * `DomainError` for $\lambda$ outside $[0, 1]$ or a bad label, and `ShapeError` when `x` and `labels` differ in rows.
 * Pure and worker-safe (plain arrays in and out).
 *
 * @param problem The inputs, their labels ($-1$ for unlabelled) and the number of classes.
 * @param options The covariance shape, $\lambda$, the regularisation and floor, the start and its ridge, how the
 *   objective is normalised and the tolerance.
 * @returns The algorithm; its state holds the parameters, the responsibilities and the objective.
 *
 * @example Two labels per class and four unlabelled points on a line
 * const x = [[-1.2], [-0.8], [0.8], [1.2], [-2], [-1.5], [1.5], [2]]
 * const problem = { x, labels: [0, 0, 1, 1, -1, -1, -1, -1], classes: 2 }
 * const final = run(semiSupervisedMixtureSteps(problem, { covariance: 'full' }), undefined, 50)
 * print('means:', final.means, ' weights:', final.weights)
 * print('steps:', final.t, ' converged:', final.converged, ' objective:', final.objective)
 *
 * @example The diagonal form with a per-class start, a floor and the per-row objective
 * const problem = { x: [[0], [2], [10], [14], [5], [6]], labels: [0, 0, 1, 1, -1, -1], classes: 2 }
 * const options = { covariance: 'diagonal', start: 'per-class', regularisation: 0, varianceFloor: 1e-3 }
 * const s0 = run(semiSupervisedMixtureSteps(problem, options), undefined, 0)
 * print('state 0, the labelled fit: means', s0.means, ' variances', s0.covariances)
 */
export function semiSupervisedMixtureSteps(
  problem: SemiSupervisedMixtureProblem,
  options: SemiSupervisedMixtureOptions = {},
): Algorithm<ClassGaussians | undefined, SemiSupervisedMixtureState> {
  const where = 'semiSupervisedMixtureSteps'
  const {
    covariance = 'tied',
    unlabelledWeight: lambda = 1,
    regularisation: reg = 1e-6,
    varianceFloor: floor = 0,
    start = 'pooled',
    objective: normalise = 'weighted',
    tolerance: tol = 1e-8,
  } = options
  if (!(lambda >= 0 && lambda <= 1)) throw new DomainError(where, `${where}: unlabelledWeight must be in [0, 1]`)
  const { X, n, d, K, labels } = problemData(problem, where)
  const w = Float64Array.from(labels, (c) => (c >= 0 ? 1 : lambda))
  let nL = 0
  for (let i = 0; i < n; i++) if (labels[i] >= 0) nL++
  const nU = n - nL
  const scale = normalise === 'per-row' ? n : nL + lambda * nU

  // The E-step and the objective at given parameters.
  const evaluate = (params: ClassGaussians, t: number, previous: number): SemiSupervisedMixtureState => {
    const { joint: lj, jitter } = logJoint(params, X, n)
    const r = new Float64Array(n * K)
    let sumL = 0
    let sumU = 0
    for (let i = 0; i < n; i++) {
      const c = labels[i]
      if (c >= 0) {
        r[i * K + c] = 1
        sumL += lj[i * K + c]
        continue
      }
      let top = -Infinity
      for (let k = 0; k < K; k++) top = Math.max(top, lj[i * K + k])
      let z = 0
      for (let k = 0; k < K; k++) z += Math.exp(lj[i * K + k] - top)
      for (let k = 0; k < K; k++) r[i * K + k] = Math.exp(lj[i * K + k] - top) / z
      sumU += top + Math.log(z)
    }
    const objective = (sumL + lambda * sumU) / scale
    const change = Number.isFinite(previous) ? objective - previous : NaN
    return {
      t,
      ...params,
      responsibilities: r,
      objective,
      labelledLogLikelihood: nL ? sumL / nL : NaN,
      unlabelledLogLikelihood: nU ? sumU / nU : NaN,
      change,
      converged: t === 0 ? lambda === 0 && start === 'per-class' : Math.abs(change) < tol,
      diverged: !Number.isFinite(objective),
      jitter,
    }
  }

  return {
    name: `semiSupervisedMixtureSteps(${covariance}, λ = ${lambda})`,
    init(given) {
      return evaluate(given ?? labelledOnlyFit(problem, options), 0, NaN)
    },
    step(state) {
      const p = maximisation(X, n, d, K, state.responsibilities, covariance, reg, { rowWeights: w, pad: 0, floor })
      const params = { weights: p.weights, means: p.means, covariances: p.covs, dimension: d }
      return evaluate(params, state.t + 1, state.objective)
    },
  }
}
