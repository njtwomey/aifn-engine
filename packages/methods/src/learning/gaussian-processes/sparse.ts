/**
 * Sparse Gaussian-process regression through $m$ inducing inputs $\Zmat$: the Nyström approximation
 * $\Qmat = \Kmat_{nm} \Kmat_{mm}^{-1} \Kmat_{mn}$ of the training covariance, used three ways (Quiñonero-Candela and
 * Rasmussen, 2005, "A unifying view of sparse approximate Gaussian process regression"):
 *
 * - `sor` (subset of regressors) and `dtc` (deterministic training conditional) share the marginal likelihood
 *   $\Gauss(\yvec \mid m\ones, \Qmat + \sigma^2\Imat)$; they differ only in the predictive variance (SoR's collapses
 *   away from $\Zmat$, DTC's does not).
 * - `fitc` (fully independent training conditional; Snelson and Ghahramani, 2006) corrects the diagonal:
 *   $\Gauss(\yvec \mid m\ones, \Qmat + \diag(\Kmat - \Qmat) + \sigma^2\Imat)$.
 * - `vfe` (Titsias, 2009, "Variational learning of inducing variables in sparse Gaussian processes") keeps the DTC
 *   likelihood term and subtracts $\trace(\Kmat - \Qmat)/(2\sigma^2)$, so the bound never exceeds the exact log
 *   marginal likelihood and inducing inputs can be optimised without overfitting.
 *
 * The algebra is the numerically stable form of GPflow's `SGPR` and `GPRFITC` (Matthews et al., 2017): with $\Lmat$
 * the Cholesky factor of $\Kmat_{mm}$, $\Amat = \Lmat^{-1}\Kmat_{mn} \Lambdamat^{-1/2}$ and $\Lmat_B$ that of
 * $\Imat + \Amat\Amat^\top$, everything costs $O(nm^2)$. Every quantity is built from tensor primitives, so the bound
 * is differentiable in $\Zmat$, the kernel's hyperparameters and $\sigma^2$.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import {
  asRows,
  gram,
  kernelDiagonal,
  kernelFromLog,
  type Kernel,
  type KernelParams,
} from 'aifn-compute/learning/kernels'
import { ravel } from 'aifn-compute/foundation/pytree'
import { permutation, type Stream } from 'aifn-compute/foundation/random'
import {
  withExpectation,
  type Decides,
  type Estimator,
  type Expects,
  type Fitted,
  type Predicts,
  type Supervised,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { Normal, type Univariate } from 'aifn-compute/probability/distributions'
import { kernelLogVector, logMarginalLikelihood } from './regression'
import { cholesky, solveTriangular } from 'aifn-compute/numerics/linalg'
import { lbfgs, type LbfgsState } from 'aifn-compute/optim/second-order'
import {
  add,
  concat,
  diagonal,
  div,
  exp,
  fromData,
  log,
  matmul,
  mul,
  reshape,
  shapeOfValue,
  sqrt,
  square,
  sub,
  sum,
  take,
  tensor,
  toFlat,
  transpose,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Status, type Trace } from 'aifn-compute/foundation/trace'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { definer, type AlgorithmInfo } from 'aifn-compute/foundation/registry'
import { DomainError } from 'aifn-compute/foundation/errors'

const LOG_2PI = Math.log(2 * Math.PI)

/** The sparse approximation: `vfe` (Titsias), `fitc`, `dtc` or `sor`. */
export type SparseMethod = 'vfe' | 'fitc' | 'dtc' | 'sor'

/** Options of `sparseGp`. */
export type SparseGpOptions = {
  /** The approximation (default `vfe`). */
  method?: SparseMethod
  /** Observation-noise variance $\sigma^2 > 0$. */
  noiseVariance: Value
  /** The constant prior mean $m$ (default 0). */
  mean?: number
  /**
   * Diagonal jitter on $\Kmat_{mm}$ relative to its mean diagonal (default 1e-8), always added and reported (with
   * any further jitter its factorisation needed).
   */
  relativeJitter?: number
}

/** The pieces of a sparse GP, possibly traced. */
type Pieces = {
  /** The Cholesky factor $\Lmat$ of $\Kmat_{mm}$ plus jitter, $m \times m$. */
  L: Value
  /** The Cholesky factor $\Lmat_B$ of $\Imat + \Amat\Amat^\top$, $m \times m$. */
  LB: Value
  /** $\cvec = \Lmat_B^{-1}\Amat\Lambdamat^{-1/2}(\yvec - m\ones)$, `[m]`: the predictive mean's weights. */
  c: Value
  /** The approximate log marginal likelihood (the ELBO for `vfe`), the sum of `terms`. */
  bound: Value
  /** The bound's terms. */
  terms: { fit: Value; complexity: Value; trace: Value; constant: number }
  /** The total jitter added to $\Kmat_{mm}$. */
  jitter: number
}

/**
 * The $n \times n$ identity.
 *
 * @param n The number of rows and columns.
 * @returns $\Imat$.
 */
function eye(n: number): Tensor {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) out[i * n + i] = 1
  return fromData(out, [n, n])
}

/**
 * The factors and the bound of a sparse GP, as tensor primitives (traced when the arguments are). The factor of
 * $\Imat + \Amat\Amat^\top$ takes no jitter, so its failure is not reported.
 *
 * @param kernel The kernel $k$.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The targets, `[n]`.
 * @param z The inducing inputs $\Zmat$, `[m, d]` or `[m]`.
 * @param o The method, noise variance, prior mean and relative jitter.
 * @returns The factors, the bound and its terms, and the jitter added.
 */
function pieces(kernel: Kernel, x: Value, y: Tensor, z: Value, o: SparseGpOptions): Pieces {
  const { method = 'vfe', noiseVariance, mean = 0, relativeJitter = 1e-8 } = o
  const X = asRows(x)
  const Z = asRows(z)
  const n = shapeOfValue(X)[0]
  const m = shapeOfValue(Z)[0]
  const Kmm = gram(kernel, Z)
  const diagMean = (toFlat(unwrap(kernelDiagonal(kernel, Z)) as Tensor) as number[]).reduce((a, b) => a + b, 0) / m
  const jitter = relativeJitter * (diagMean > 0 ? diagMean : 1)
  const factor = cholesky(add(Kmm, mul(jitter, eye(m))))
  const L = factor.L
  const A0 = solveTriangular(L, gram(kernel, Z, X)) // L⁻¹K_mn, [m, n]
  const qnn = sum(square(A0), 0) // diag Q, [n]
  const knn = kernelDiagonal(kernel, X)
  const lambda =
    method === 'fitc'
      ? add(sub(knn, qnn), noiseVariance)
      : mul(noiseVariance, fromData(new Float64Array(n).fill(1), [n]))
  const rootLambda = sqrt(lambda)
  const A = div(A0, reshape(rootLambda, [1, n]))
  const { L: LB } = cholesky(add(eye(m), matmul(A, transpose(A))), { jitter: false })
  const yw = div(sub(y, mean), rootLambda)
  const c = solveTriangular(LB, matmul(A, yw))
  const constant = -0.5 * n * LOG_2PI
  const complexity = add(mul(-1, sum(log(diagonal(LB)))), mul(-0.5, sum(log(lambda))))
  const fit = add(mul(-0.5, sum(square(yw))), mul(0.5, sum(square(c))))
  const traceTerm = method === 'vfe' ? mul(-0.5, div(sum(sub(knn, qnn)), noiseVariance)) : 0
  return {
    L,
    LB,
    c,
    bound: add(add(add(fit, complexity), traceTerm), constant),
    terms: { fit, complexity, trace: traceTerm, constant },
    jitter: jitter + factor.jitter,
  }
}

/** A fitted sparse GP. */
export interface SparseGp<P extends KernelParams = KernelParams> {
  /** The approximation. */
  readonly method: SparseMethod
  /** The kernel $k$. */
  readonly kernel: Kernel<P>
  /** Inducing inputs $\Zmat$, `[m, d]`. */
  readonly inducing: Tensor
  /** The observation-noise variance $\sigma^2$. */
  readonly noiseVariance: number
  /** The constant prior mean $m$. */
  readonly mean: number
  /** The approximate log marginal likelihood (for `vfe`, the evidence lower bound): the sum of `terms`. */
  readonly logMarginal: number
  /**
   * The terms of `logMarginal`: data fit, complexity ($-\log\lvert \Lmat_B \rvert - \frac{1}{2} \sum \log \Lambda$),
   * the VFE trace penalty $-\trace(\Kmat - \Qmat)/(2\sigma^2)$ (0 for the others) and the constant.
   */
  readonly terms: { fit: number; complexity: number; trace: number; constant: number }
  /** Jitter added to $\Kmat_{mm}$. */
  readonly jitter: number
  /** Predictive mean and variance of $f$ at `xs` (`[s, d]` or `[s]`); `noise` adds $\sigma^2$. */
  predict(xs: Tensor, options?: { noise?: boolean }): { mean: Tensor; variance: Tensor }
}

/**
 * A sparse GP regression through inducing inputs; see the file comment for the methods. With $\Zmat = \Xmat$, `fitc`
 * and `vfe` reproduce the exact GP. Throws `DomainError` unless the noise variance is positive.
 *
 * @param kernel The kernel $k$.
 * @param x The training inputs $\Xmat$, `[n, d]` or `[n]`.
 * @param y The targets, `[n]` or `[n, 1]`.
 * @param z The inducing inputs $\Zmat$, `[m, d]` or `[m]`.
 * @param options The method (default `vfe`), the noise variance (required), the prior mean and the jitter on
 *   $\Kmat_{mm}$.
 * @returns The sparse GP: its bound and terms, and `predict` at new inputs.
 *
 * @example With every training input as an inducing input VFE is exact; two cost a lot
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const k = rbf({ lengthscale: 1, variance: 1 })
 * print('exact', logMarginalLikelihood(k, x, y, { noiseVariance: 0.01 }).value)
 * print('VFE with Z = X', sparseGp(k, x, y, x, { noiseVariance: 0.01 }).logMarginal)
 * const two = sparseGp(k, x, y, tensor([[0.5], [3]]), { noiseVariance: 0.01 })
 * print('VFE with two inducing inputs', two.logMarginal, two.terms)
 * print('prediction at pi/2', two.predict(tensor([[Math.PI / 2]])))
 */
export function sparseGp<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  z: Tensor,
  options: SparseGpOptions & { noiseVariance: number },
): SparseGp<P> {
  const { method = 'vfe', noiseVariance, mean = 0 } = options
  if (!(noiseVariance > 0)) throw new DomainError('sparseGp', 'sparseGp: noiseVariance must be positive')
  const Y = y.shape.length === 2 ? (reshape(y, [y.shape[0]]) as Tensor) : y
  const Z = asRows(z) as Tensor
  const p = pieces(kernel, x, Y, Z, { ...options, method })
  const L = unwrap(p.L) as Tensor
  const LB = unwrap(p.LB) as Tensor
  const c = unwrap(p.c) as Tensor
  const num = (v: Value) => unwrap(v) as number
  return {
    method,
    kernel,
    inducing: Z,
    noiseVariance,
    mean,
    logMarginal: num(p.bound),
    terms: {
      fit: num(p.terms.fit),
      complexity: num(p.terms.complexity),
      trace: num(p.terms.trace),
      constant: p.terms.constant,
    },
    jitter: p.jitter,
    predict: (xs, { noise = false } = {}) => {
      const S = asRows(xs) as Tensor
      const t1 = solveTriangular(L, gram(kernel, Z, S)) as Tensor // [m, s]
      const t2 = solveTriangular(LB, t1) as Tensor
      const mu = add(matmul(c, t2), mean) as Tensor
      const q = toFlat(sum(square(t1), 0) as Tensor)
      const r = toFlat(sum(square(t2), 0) as Tensor)
      const k = toFlat(kernelDiagonal(kernel, S) as Tensor)
      const extra = noise ? noiseVariance : 0
      // SoR's predictive covariance is Q_** − Q_*m(…)Q_m* = t2ᵀt2; the others keep K_** − Q_** as well.
      const variance = Float64Array.from(k, (kk, j) => (method === 'sor' ? r[j] : kk - q[j] + r[j]) + extra)
      return { mean: mu, variance: fromData(variance, [variance.length]) }
    },
  }
}

/**
 * The approximate log marginal likelihood (the ELBO for `vfe`), a traced value when its arguments are traced: the
 * objective that `sparseGpFitSteps` differentiates. Unlike `sparseGp` it neither checks the noise variance nor
 * reshapes the targets.
 *
 * @param kernel The kernel $k$, possibly with traced hyperparameters.
 * @param x The training inputs $\Xmat$, `[n, d]` or `[n]`.
 * @param y The targets, `[n]`.
 * @param z The inducing inputs $\Zmat$, `[m, d]` or `[m]`, possibly traced.
 * @param options The method (default `vfe`), the noise variance $\sigma^2 > 0$ (possibly traced), the prior mean and
 *   the jitter on $\Kmat_{mm}$.
 * @returns The bound.
 *
 * @example The four approximations at the same inducing inputs: VFE lowest, SoR and DTC equal, all below exact
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const k = rbf({ lengthscale: 1, variance: 1 })
 * const z = tensor([[0.5], [2], [3.5]])
 * for (const method of ['vfe', 'dtc', 'sor', 'fitc'])
 *   print(method, sparseLogMarginal(k, x, y, z, { method, noiseVariance: 0.01 }))
 * print('exact', logMarginalLikelihood(k, x, y, { noiseVariance: 0.01 }).value)
 */
export function sparseLogMarginal(kernel: Kernel, x: Value, y: Tensor, z: Value, options: SparseGpOptions): Value {
  return pieces(kernel, x, y, z, options).bound
}

/** Options of `fitSparseGp` and `sparseGpFitSteps`. */
export type FitSparseGpOptions = {
  /** The approximation whose bound is maximised (default `vfe`). */
  method?: SparseMethod
  /** Starting noise variance (default 0.1). */
  noiseVariance?: number
  /** Fit the noise variance too (default true); otherwise it stays at `noiseVariance`. */
  fitNoise?: boolean
  /** Move the inducing inputs too (default true). */
  fitInducing?: boolean
  /** Fit the kernel's hyperparameters (default true); otherwise they stay as given. */
  fitKernel?: boolean
  /** The constant prior mean $m$, held fixed (default 0). */
  mean?: number
  /** Most L-BFGS steps, for `fitSparseGp` (default 200). */
  maxSteps?: number
  /** Converged when the gradient norm in the fitted parameters is at most this (default 1e-6). */
  tolerance?: number
  /** Also record the exact GP's log marginal likelihood at every state's hyperparameters ($O(n^3)$ a state). */
  exact?: boolean
}

/** What every state of a sparse-GP fit reports: the inducing inputs, hyperparameters and objective it has reached. */
export type SparseGpFitFields = {
  /** Inducing inputs $\Zmat$, `[m, d]`. */
  inducing: Tensor
  /** The kernel's log hyperparameters, in `kernelLogVector` order (rebuild the kernel with `sparseGpAt`). */
  logKernel: Float64Array
  /** The kernel's hyperparameters by name (pytree path). */
  hyper: Record<string, number>
  /** The noise variance $\sigma^2$. */
  noiseVariance: number
  /** The approximate log marginal likelihood (for `vfe`, the ELBO). */
  logMarginal: number
  /** With `exact`: the exact GP's log marginal likelihood at the same hyperparameters ($\ge$ the VFE bound). */
  exact?: number
}

/** The state of `sparseGpFitSteps`: an L-BFGS state with the fit's fields. */
export type SparseGpFitState = LbfgsState & SparseGpFitFields

/**
 * The fitting problem: the objective over $\thetavec = [\log \thetavec_k, \log \sigma^2, \Zmat]$ (each part present
 * when fitted, $\Zmat$ flattened row-major) and its layout. A point where the bound cannot be computed counts as
 * $+\infty$.
 *
 * @param kernel The kernel at the starting hyperparameters.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The targets, `[n]` or `[n, 1]`.
 * @param z The starting inducing inputs, `[m, d]` or `[m]`.
 * @param o What is fitted, the starting noise, the method and the prior mean.
 * @returns The inputs as rows and the targets as a vector, the objective (minus the bound, with its gradient), the
 *   starting $\thetavec$, `fields` (a state's reported fields at $\thetavec$), the method and the mean.
 */
function fitProblem<P extends KernelParams>(kernel: Kernel<P>, x: Tensor, y: Tensor, z: Tensor, o: FitSparseGpOptions) {
  const { method = 'vfe', noiseVariance = 0.1, fitNoise = true, fitInducing = true, fitKernel = true, mean = 0 } = o
  const X = asRows(x) as Tensor
  const Y = y.shape.length === 2 ? (reshape(y, [y.shape[0]]) as Tensor) : y
  const Z0 = asRows(z) as Tensor
  const [m, d] = Z0.shape
  // The gradient is taken with respect to the parts (the kernel's as a pytree) and concatenated in θ's order.
  const lv = kernelLogVector(kernel)
  const k = fitKernel ? lv.vector.length : 0
  const zStart = k + (fitNoise ? 1 : 0)
  const parts = (theta: ArrayLike<number>) => {
    const th = Array.from(theta)
    const logKernel = fitKernel ? Float64Array.from(th.slice(0, k)) : lv.vector
    return {
      logKernel,
      tree: lv.unravel(logKernel),
      logNoise: fitNoise ? th[k] : Math.log(noiseVariance),
      z: fitInducing ? fromData(Float64Array.from(th.slice(zStart, zStart + m * d)), [m, d]) : Z0,
    }
  }
  const negative = valueAndGrad(
    (tree: P, logNoise: Value, zz: Value) =>
      mul(-1, sparseLogMarginal(kernelFromLog(kernel, tree), X, Y, zz, { method, noiseVariance: exp(logNoise), mean })),
    { argnums: [0, 1, 2] },
  )
  const objective = (theta: Tensor) => {
    try {
      const u = parts(toFlat(theta))
      const { value, grad } = negative(u.tree, u.logNoise, u.z)
      if (!Number.isFinite(value as number)) return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
      const [gTree, gNoise, gZ] = grad as [P, Value, Value]
      const g = [
        ...(fitKernel ? ravel(gTree).vector : []),
        ...(fitNoise ? [typeof gNoise === 'number' ? gNoise : toFlat(gNoise as Tensor)[0]] : []),
        ...(fitInducing ? toFlat(gZ as Tensor) : []),
      ]
      return { value: value as number, grad: fromData(Float64Array.from(g), [g.length]) }
    } catch {
      return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
    }
  }
  const start = [
    ...(fitKernel ? lv.vector : []),
    ...(fitNoise ? [Math.log(noiseVariance)] : []),
    ...(fitInducing ? (toFlat(Z0) as number[]) : []),
  ]
  /** The fields of the state at θ whose objective is `value`. */
  const fields = (theta: Tensor, value: number): SparseGpFitFields => {
    const u = parts(toFlat(theta))
    const noise = Math.exp(u.logNoise)
    const hyper = Object.fromEntries(lv.names.map((name, i) => [name, Math.exp(u.logKernel[i])]))
    return {
      inducing: u.z,
      logKernel: u.logKernel,
      hyper,
      noiseVariance: noise,
      logMarginal: -value,
      ...(o.exact ? { exact: exactLogMarginal(kernelFromLog(kernel, u.tree), X, Y, noise, mean) } : {}),
    }
  }
  return { X, Y, objective, start, fields, method, mean }
}

/**
 * The exact GP's log marginal likelihood, as a number.
 *
 * @param kernel The kernel $k$.
 * @param x The training inputs, `[n, d]`.
 * @param y The targets, `[n]`.
 * @param noiseVariance The noise variance $\sigma^2$.
 * @param mean The constant prior mean $m$.
 * @returns $\log p(\yvec \mid \Xmat, \thetavec)$.
 */
function exactLogMarginal(kernel: Kernel, x: Tensor, y: Tensor, noiseVariance: number, mean: number): number {
  return logMarginalLikelihood(kernel, x, y, { noiseVariance, mean }).value as number
}

/**
 * The sparse GP's fit as a traceable algorithm: L-BFGS on minus the approximate log marginal likelihood (the ELBO for
 * `vfe`) over the kernel's log hyperparameters, $\log \sigma^2$ and the inducing inputs $\Zmat$, with gradients from
 * `aifn-compute/foundation/autodiff`. Every state carries $\Zmat$, the hyperparameters and the objective, so a figure
 * can play the optimisation; `init` takes nothing. The objective never decreases from one state to the next (a Wolfe
 * line search).
 *
 * @param kernel The kernel at the starting hyperparameters.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The targets, `[n]` or `[n, 1]`.
 * @param z The starting inducing inputs, `[m, d]` or `[m]`.
 * @param options What is fitted, the starting noise, the method and the tolerance (`maxSteps` is not read here).
 * @returns The algorithm; run it with `run` or `trace`.
 *
 * @example The ELBO rises step by step from three inducing inputs
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const steps = sparseGpFitSteps(rbf({ lengthscale: 1, variance: 1 }), x, y, tensor([[0], [0.5], [1]]))
 * const tr = trace(steps, undefined, 30, { record: { logMarginal: (s) => s.logMarginal } })
 * print('ELBO', tr.series.logMarginal)
 * print('inducing', tr.final.inducing, ' hyperparameters', tr.final.hyper, ' noise', tr.final.noiseVariance)
 */
export function sparseGpFitSteps<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  z: Tensor,
  options: FitSparseGpOptions = {},
): Algorithm<void, SparseGpFitState> {
  const problem = fitProblem(kernel, x, y, z, options)
  const opt = lbfgs(problem.objective, { tolerance: options.tolerance ?? 1e-6 })
  const wrap = (s: LbfgsState): SparseGpFitState => ({ ...s, ...problem.fields(s.x, s.value) })
  return {
    name: 'sparse-gp-fit',
    init: (_start, stream) => wrap(opt.init({ x0: tensor(problem.start) }, stream)),
    step: (s, ctx) => wrap(opt.step(s, ctx)),
  }
}

/**
 * The sparse GP at a fit state's inducing inputs and hyperparameters (`kernel` is the template the fit started from).
 *
 * @param kernel The kernel the fit started from; only its shape and names are used.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The targets, `[n]` or `[n, 1]`.
 * @param state A state of `sparseGpFitSteps` or `sparseGpGrowSteps` (its inducing inputs, log hyperparameters and
 *   noise variance).
 * @param options `method` (default `vfe`) and `mean` (default 0), as the fit used them.
 * @returns The sparse GP at the state.
 *
 * @example The model at the end of a run
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const k = rbf({ lengthscale: 1, variance: 1 })
 * const state = run(sparseGpFitSteps(k, x, y, tensor([[0], [0.5], [1]])), undefined, 30)
 * const model = sparseGpAt(k, x, y, state)
 * print('ELBO', model.logMarginal, ' as the state reports it', state.logMarginal)
 * print('mean at pi/2', model.predict(tensor([[Math.PI / 2]])).mean, ' sin(pi/2) = 1')
 */
export function sparseGpAt<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  state: Pick<SparseGpFitFields, 'inducing' | 'logKernel' | 'noiseVariance'>,
  options: { method?: SparseMethod; mean?: number } = {},
): SparseGp<P> {
  const fitted = kernelFromLog(kernel, kernelLogVector(kernel).unravel(state.logKernel))
  return sparseGp(fitted, x, y, state.inducing, { ...options, noiseVariance: state.noiseVariance })
}

/** The result of `fitSparseGp`. */
export type SparseGpFit<P extends KernelParams = KernelParams> = {
  /** The sparse GP at the final state. */
  model: SparseGp<P>
  /** The trace of `sparseGpFitSteps`, with the objective recorded as `logMarginal`. */
  training: Trace<SparseGpFitState>
  /** Whether the final gradient norm reached `tolerance`. */
  converged: boolean
}

/**
 * Maximise the sparse approximation's log marginal likelihood (the ELBO for `vfe`) over the kernel's log
 * hyperparameters, $\log \sigma^2$ and the inducing inputs, by L-BFGS (`sparseGpFitSteps` run to `maxSteps`, default
 * 200).
 *
 * @param kernel The kernel at the starting hyperparameters.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The targets, `[n]` or `[n, 1]`.
 * @param z The starting inducing inputs, `[m, d]` or `[m]`.
 * @param options What is fitted, the starting noise, the method, the step limit and the tolerance.
 * @returns The fitted sparse GP, the trace and whether it converged.
 *
 * @example Three inducing inputs on a noisy sine, with the exact evidence above the bound
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const z = tensor([[0], [0.5], [1]])
 * const fit = fitSparseGp(rbf({ lengthscale: 1, variance: 1 }), x, y, z, { exact: true })
 * print('ELBO', fit.model.logMarginal, ' exact', fit.training.final.exact, ' converged', fit.converged)
 * print('inducing', fit.model.inducing, ' noise variance', fit.model.noiseVariance)
 * print('mean at pi/2', fit.model.predict(tensor([[Math.PI / 2]])).mean, ' sin(pi/2) = 1')
 */
export function fitSparseGp<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  z: Tensor,
  options: FitSparseGpOptions = {},
): SparseGpFit<P> {
  const training = trace(sparseGpFitSteps(kernel, x, y, z, options), undefined, options.maxSteps ?? 200, {
    record: { logMarginal: (s: SparseGpFitState) => s.logMarginal },
  })
  const final = training.final
  const model = sparseGpAt(kernel, x, y, final, { method: options.method, mean: options.mean })
  return { model, training, converged: final.converged ?? false }
}

// ── Greedy growth of the inducing set ────────────────────────────────────────────────────────────────────────────────

/** Options of `sparseGpGrowSteps`. */
export type SparseGpGrowOptions = FitSparseGpOptions & {
  /** Inducing inputs at step 0, themselves chosen greedily (default 1). */
  initial?: number
  /** Stop at this many inducing inputs (default 20). */
  maxInducing?: number
  /** Candidates: this many training rows drawn without replacement from the stream (default $\min(n, 50)$). */
  candidates?: number
  /**
   * L-BFGS steps on the hyperparameters and $\Zmat$ after each addition (default 0: hyperparameters held, $\Zmat$ only
   * grows).
   */
  reoptimise?: number
}

/** The state of `sparseGpGrowSteps`. */
export type SparseGpGrowState = Status &
  SparseGpFitFields & {
    /** Candidate training rows not yet added. */
    candidates: number[]
    /** The training row added at this step (null at step 0). */
    added: number | null
    /** The objective just after the addition, before any re-optimisation (NaN at step 0). */
    afterAdd: number
  }

/**
 * Greedy selection of inducing inputs (in the spirit of Seeger et al., 2003, and Titsias, 2009, §3): each step adds
 * the candidate training input whose addition most increases the objective (the ELBO for `vfe`, the approximate log
 * marginal likelihood otherwise), then optionally re-optimises everything by `reoptimise` L-BFGS steps. Each candidate
 * is scored exactly, at $O(nm^2)$ for $m$ inducing inputs, so a step costs $O(cnm^2)$ for $c$ candidates. With the
 * hyperparameters held, the VFE bound never decreases as $\Zmat$ grows. `init` draws the candidates from its stream
 * and adds the first `initial` inducing inputs; the run terminates when the candidates run out or `maxInducing` is
 * reached.
 *
 * @param kernel The kernel, held at its hyperparameters unless `reoptimise` is positive.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param y The targets, `[n]` or `[n, 1]`.
 * @param options The method, the noise variance (default 0.1), the candidates, the sizes and the re-optimisation.
 * @returns The algorithm; run it with `run` or `trace`.
 *
 * @example Four inducing inputs, one added per step
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const grow = sparseGpGrowSteps(rbf({ lengthscale: 1, variance: 1 }), x, y, { noiseVariance: 0.01, maxInducing: 4 })
 * const tr = trace(grow, undefined, 10, { record: { logMarginal: (s) => s.logMarginal, added: (s) => s.added ?? -1 } })
 * print('row added (-1 at step 0)', tr.series.added)
 * print('ELBO', tr.series.logMarginal)
 * print('inducing', tr.final.inducing)
 */
export function sparseGpGrowSteps<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  options: SparseGpGrowOptions = {},
): Algorithm<void, SparseGpGrowState> {
  const { method = 'vfe', mean = 0, initial = 1, maxInducing = 20, reoptimise = 0 } = options
  const X = asRows(x) as Tensor
  const Y = y.shape.length === 2 ? (reshape(y, [y.shape[0]]) as Tensor) : y
  const n = X.shape[0]
  const count = Math.min(n, options.candidates ?? 50)
  const lv = kernelLogVector(kernel)
  const kernelOf = (logKernel: ArrayLike<number>) => kernelFromLog(kernel, lv.unravel(logKernel))
  const score = (k: Kernel, z: Tensor, noise: number) => {
    try {
      const v = sparseLogMarginal(k, X, Y, z, { method, noiseVariance: noise, mean }) as number
      return Number.isFinite(v) ? v : -Infinity
    } catch {
      return -Infinity
    }
  }
  const rowsOf = (z: Tensor | null, row: number) => {
    const r = take(X, [row]) as Tensor
    return z ? (concat([z, r], 0) as Tensor) : r
  }
  /** Add the best candidate to Z; returns the new Z, the row added and its objective. */
  const grow = (z: Tensor | null, candidates: number[], logKernel: Float64Array, noise: number) => {
    const k = kernelOf(logKernel)
    let best = -1
    let bestValue = -Infinity
    for (const c of candidates) {
      const v = score(k, rowsOf(z, c), noise)
      if (v > bestValue || best < 0) [best, bestValue] = [c, v]
    }
    return { z: rowsOf(z, best), added: best, value: bestValue, rest: candidates.filter((c) => c !== best) }
  }
  const fieldsAt = (z: Tensor, logKernel: Float64Array, noise: number, value: number): SparseGpFitFields => ({
    inducing: z,
    logKernel,
    hyper: Object.fromEntries(lv.names.map((name, i) => [name, Math.exp(logKernel[i])])),
    noiseVariance: noise,
    logMarginal: value,
    ...(options.exact ? { exact: exactLogMarginal(kernelOf(logKernel), X, Y, noise, mean) } : {}),
  })
  const noise0 = options.noiseVariance ?? 0.1
  return {
    name: 'sparse-gp-grow',
    init: (_start, stream) => {
      let candidates = Array.from(toFlat(permutation(stream, n))).slice(0, count)
      let z: Tensor | null = null
      let value = -Infinity
      for (let i = 0; i < Math.max(1, Math.min(initial, count)); i++) {
        const g = grow(z, candidates, lv.vector, noise0)
        ;[z, value, candidates] = [g.z, g.value, g.rest]
      }
      return {
        t: 0,
        ...fieldsAt(z!, lv.vector, noise0, value),
        candidates,
        added: null,
        afterAdd: NaN,
        terminated: candidates.length === 0 || z!.shape[0] >= maxInducing,
      }
    },
    step: (s, ctx) => {
      if (s.candidates.length === 0 || s.inducing.shape[0] >= maxInducing) return { ...s, t: s.t + 1, terminated: true }
      const g = grow(s.inducing, s.candidates, s.logKernel, s.noiseVariance)
      let fields = fieldsAt(g.z, s.logKernel, s.noiseVariance, g.value)
      if (reoptimise > 0) {
        const tr = trace(
          sparseGpFitSteps(kernelOf(s.logKernel) as Kernel<P>, X, Y, g.z, {
            ...options,
            noiseVariance: s.noiseVariance,
          }),
          undefined,
          reoptimise,
          { keep: 'none', stream: ctx.stream },
        )
        // A failed line search leaves the state where it was, so the objective after re-optimisation is never lower.
        const { inducing, logKernel, noiseVariance, logMarginal } = tr.final
        fields = fieldsAt(inducing, logKernel, noiseVariance, logMarginal)
      }
      return {
        t: s.t + 1,
        ...fields,
        candidates: g.rest,
        added: g.added,
        afterAdd: g.value,
        terminated: g.rest.length === 0 || g.z.shape[0] >= maxInducing,
      }
    },
  }
}

// ── Estimator ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `sparseGaussianProcessRegressor`. */
export type SparseGaussianProcessRegressorParams<P extends KernelParams = KernelParams> = FitSparseGpOptions & {
  /** The kernel $k$ (required); with `optimise`, the start of the fit. */
  kernel: Kernel<P>
  /** Number of inducing inputs $m$, chosen from the training inputs without replacement (default $\min(n, 20)$). */
  inducing?: number
  /** Fit the kernel, noise and inducing inputs by maximising the bound (default true). */
  optimise?: boolean
}

/** A fitted sparse GP regression model. */
export interface SparseGaussianProcessRegressionModel<P extends KernelParams = KernelParams>
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Univariate<Tensor>>,
    Expects<Tensor>,
    Partial<Trained<LbfgsState>> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'sparse-gaussian-process-regression'
  /** The fitted sparse GP. */
  readonly sparse: SparseGp<P>
}

/**
 * Sparse GP regression as an estimator (the C1 conformance of `sparseGp`/`fitSparseGp`): the inducing inputs start
 * at `inducing` training inputs drawn from the `stream` given to `fit` (the first ones without a stream), and with
 * `optimise` the kernel, noise and inducing inputs are fitted by `fitSparseGp`, whose L-BFGS run is kept in
 * `training`. Capabilities: `forward` and `decide` (the predictive mean), `predictive` (normals
 * $\Gauss(\mu, v + \sigma^2)$), `expect`.
 *
 * @param params The kernel, the number of inducing inputs, whether to optimise, and the options of `fitSparseGp`.
 * @returns The estimator: `fit({ x, y })` on inputs `[n, d]` (or `[n]`) and targets `[n]` returns a
 *   `SparseGaussianProcessRegressionModel`.
 *
 * @example Three inducing inputs, fitted with the kernel and the noise
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const gp = sparseGaussianProcessRegressor({ kernel: rbf({ lengthscale: 1, variance: 1 }), inducing: 3 })
 * const model = gp.fit({ x, y })
 * print('inducing', model.sparse.inducing, ' ELBO', model.sparse.logMarginal)
 * const xs = tensor([[Math.PI / 2]])
 * print('mean at pi/2', model.forward(xs), ' predictive variance', model.predictive(xs).variance())
 */
export function sparseGaussianProcessRegressor<P extends KernelParams>(
  params: SparseGaussianProcessRegressorParams<P>,
): Estimator<Supervised<Tensor, Tensor>, SparseGaussianProcessRegressionModel<P>> {
  const { kernel, optimise = true, method = 'vfe', noiseVariance = 0.1, mean = 0 } = params
  return {
    name: 'sparse-gaussian-process-regression',
    params,
    fit({ x, y }, options: { stream?: Stream } = {}) {
      const X = asRows(x) as Tensor
      const n = X.shape[0]
      const m = Math.min(n, params.inducing ?? 20)
      const rows = options.stream
        ? Array.from(toFlat(permutation(options.stream, n))).slice(0, m)
        : [...Array(m).keys()]
      const Z = take(X, rows) as Tensor
      let sparse: SparseGp<P>
      let training: Trace<LbfgsState> | undefined
      if (optimise) {
        const fitted = fitSparseGp(kernel, X, y, Z, params)
        sparse = fitted.model
        training = fitted.training
      } else sparse = sparseGp(kernel, X, y, Z, { method, noiseVariance, mean })
      const forward = (input: Tensor) => sparse.predict(input).mean
      const base = {
        kind: 'model' as const,
        name: 'sparse-gaussian-process-regression' as const,
        sparse,
        ...(training ? { training } : {}),
        forward,
        decide: forward,
        predictive: (input: Tensor) => {
          const p = sparse.predict(input, { noise: true })
          return Normal(
            p.mean,
            fromData(Float64Array.from(toFlat(p.variance), Math.sqrt), p.variance.shape),
          ) as Univariate<Tensor>
        },
      }
      return withExpectation(base)
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'sparseGaussianProcessRegressor',
    module: 'learning/gaussian-processes',
    name: 'Sparse Gaussian process regression',
    summary: 'Inducing-point GP regression (VFE, FITC or DTC); the kernel is a required argument.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect'],
    hyper: space({
      inducing: int(1, 500, { default: 20 }),
      method: oneOf(['vfe', 'fitc', 'dtc', 'sor']),
      noiseVariance: real(1e-6, 10, { default: 0.1, scale: 'log' }),
      optimise: bool({ default: true }),
      mean: real(-10, 10, { default: 0 }),
    }),
    notes: ['sparse-gaussian-processes'],
    cite: ['titsias2009'],
  },
  sparseGaussianProcessRegressor,
)

const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/gaussian-processes')
algorithm(
  {
    key: 'sparseGpFitSteps',
    name: 'Sparse GP fit by L-BFGS',
    summary: 'L-BFGS on the sparse objective (the VFE bound, or FITC/DTC/SoR evidence) over log θ, log σ² and Z.',
    problem: 'objective',
    state: {
      iterate: 'inducing',
      objective: 'logMarginal',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'stalled'],
    },
    notes: ['sparse-gaussian-processes'],
    cite: ['titsias2009'],
  },
  sparseGpFitSteps,
)
algorithm(
  {
    key: 'sparseGpGrowSteps',
    name: 'Sparse GP by greedy inducing-point selection',
    summary:
      'Add the candidate training input that most raises the sparse objective, one per step, optionally re-optimising.',
    problem: 'objective',
    state: { iterate: 'inducing', objective: 'logMarginal', flags: ['terminated'] },
    random: true,
    notes: ['sparse-gaussian-processes'],
    cite: ['titsias2009'],
  },
  sparseGpGrowSteps,
)
