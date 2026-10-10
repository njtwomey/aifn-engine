/**
 * Gaussian-process regression with Gaussian noise: the exact posterior by Cholesky factorisation, predictive means and
 * variances, draws from prior and posterior, the log marginal likelihood and its gradient, and hyperparameter fitting.
 *
 * The computations follow Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", Algorithm 2.1
 * (posterior and log marginal likelihood from the Cholesky factor $\Lmat$ of $\Kmat + \sigma^2\Imat$) and eq. 5.9 (the
 * gradient), with jitter reported rather than hidden whenever $\Kmat + \sigma^2\Imat$ is not numerically positive
 * definite. Inputs are an $n \times d$ matrix of rows (or a vector of $n$ one-dimensional inputs), targets a vector of
 * $n$, and the prior mean is a constant $m$.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { MultivariateNormal, Normal, type Multivariate, type Univariate } from 'aifn-compute/probability/distributions'
import type {
  Decides,
  Estimator,
  Expects,
  FitOptions,
  Fitted,
  Predicts,
  Samples,
  Supervised,
  Trained,
} from 'aifn-compute/learning/estimators'
import { withExpectation } from 'aifn-compute/learning/estimators'
import {
  asRows,
  gram,
  kernelDiagonal,
  kernelFromLog,
  logParams,
  type Kernel,
  type KernelParams,
} from 'aifn-compute/learning/kernels'
import { ravel, treeFlatten } from 'aifn-compute/foundation/pytree'
import { cholesky, choleskyLogDet, solveTriangular } from 'aifn-compute/numerics/linalg'
import { lbfgs, type LbfgsState } from 'aifn-compute/optim/second-order'
import { normals, uniform, type Stream, child } from 'aifn-compute/foundation/random'
import {
  add,
  exp,
  fromData,
  matmul,
  mul,
  reshape,
  shapeOfValue,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace, type Trace } from 'aifn-compute/foundation/trace'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * A kernel's hyperparameters in log space as one vector (for L-BFGS), with their names (pytree paths) and the map
 * back to a log-parameter tree (`aifn-compute/foundation/pytree`'s `ravel`). Gradients are taken with respect to the
 * tree (`valueAndGrad` over pytrees) and ravelled in the same leaf order.
 *
 * @param kernel The kernel whose hyperparameters `kernel.params` (all positive) are taken to log space; not modified.
 * @returns `vector`, the $\log \theta$ of every leaf in pytree order; `unravel`, from such a vector back to a tree of
 *   log-hyperparameters shaped like `kernel.params` (for `kernelFromLog`); and `names`, the leaves' paths.
 *
 * @example An RBF kernel's two log-hyperparameters
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const { vector, names, unravel } = kernelLogVector(rbf({ lengthscale: 2, variance: 1 }))
 * print('names', names, ' log values', vector)
 * print('as a tree', unravel(vector))
 */
export function kernelLogVector<P extends KernelParams>(
  kernel: Kernel<P>,
): { vector: Float64Array; unravel: (v: ArrayLike<number>) => P; names: string[] } {
  const tree = logParams(kernel)
  const { vector, unravel } = ravel(tree)
  return { vector, unravel, names: treeFlatten(tree).paths }
}

/** Options shared by the exact GP computations. */
export type GpOptions = {
  /**
   * Observation-noise variance $\sigma^2 \ge 0$ (default 0: interpolation, with jitter added if needed and reported).
   */
  noiseVariance?: Value
  /** Constant prior mean $m$ (default 0). */
  mean?: number
}

/**
 * Float64 contents of a raw tensor, row-major.
 *
 * @param t The tensor; not modified.
 * @returns A new array of its values.
 */
function flat(t: Tensor): Float64Array {
  return Float64Array.from(toFlat(t))
}

/**
 * The targets as a vector `[n]`: an `[n, 1]` column is reshaped, and any other shape but `[n]` throws `ShapeError`.
 *
 * @param y The targets, `[n]` or `[n, 1]`.
 * @returns The targets, `[n]`.
 */
function targets(y: Tensor): Tensor {
  if (y.shape.length === 2 && y.shape[1] === 1) return reshape(y, [y.shape[0]])
  if (y.shape.length !== 1) throw new ShapeError('gp', `gp: targets must be [n], got shape [${y.shape.join(', ')}]`)
  return y
}

/**
 * $\Kmat + \sigma^2\Imat$ for the training inputs, differentiable in the kernel's hyperparameters, the inputs and
 * $\sigma^2$ when they are traced.
 *
 * @param kernel The kernel $k$.
 * @param x The training inputs, `[n, d]` or `[n]`.
 * @param noiseVariance $\sigma^2$, added to every diagonal entry.
 * @returns The $n \times n$ matrix.
 */
function noisyGram(kernel: Kernel, x: Value, noiseVariance: Value): Value {
  const K = gram(kernel, x)
  const n = shapeOfValue(K)[0]
  const eye = new Float64Array(n * n)
  for (let i = 0; i < n; i++) eye[i * n + i] = 1
  return add(K, mul(noiseVariance, fromData(eye, [n, n])))
}

// ── Prior ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Draws from a zero-mean (or constant-mean) Gaussian with covariance $\Cmat$, and the jitter its factor needed. */
export type Draws = {
  /** Draws `[n, m]`: row $r$ is one function evaluated at the $m$ inputs. */
  draws: Tensor
  /** Diagonal jitter added to the covariance to factor it (0 when none was needed). */
  jitter: number
  /** True when even the largest jitter tried did not make it factor. */
  failed: boolean
}

/**
 * $\muvec + \Lmat\zvec$ for each row $\zvec$ of standard normals `[n, m]`, with $\Lmat$ the lower Cholesky factor of
 * the covariance (jitter added as `cholesky`'s `'auto'` chooses, and reported).
 *
 * @param s The stream the standard normals are drawn from.
 * @param mean The mean $\muvec$, `[m]`.
 * @param cov The covariance, $m \times m$.
 * @param n The number of draws.
 * @returns The `[n, m]` draws, the jitter added to factor `cov`, and whether even the largest jitter failed.
 */
function drawsFrom(s: Stream, mean: Tensor, cov: Tensor, n: number): Draws {
  const m = mean.shape[0]
  const { L, jitter, failed } = cholesky(cov)
  const z = normals(s, [n, m])
  const draws = add(matmul(z, transpose(L)), reshape(mean, [1, m])) as Tensor
  return { draws, jitter, failed }
}

/**
 * The GP prior at inputs `xs`: mean $m\ones$ and covariance $k(\Xmat_*, \Xmat_*)$.
 *
 * @param kernel The kernel $k$.
 * @param xs The inputs $\Xmat_*$, `[m, d]` or `[m]`.
 * @param options The prior's options.
 * @param options.mean The constant prior mean $m$.
 * @returns `mean`, `[m]`, and `covariance`, $m \times m$.
 *
 * @example The prior at three points one apart
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const { mean, covariance } = gpPrior(rbf({ lengthscale: 1, variance: 1 }), tensor([[0], [1], [2]]))
 * print('mean', mean)
 * print('covariance', covariance)
 * print('exp(-1/2), exp(-2):', Math.exp(-0.5), Math.exp(-2))
 */
export function gpPrior(kernel: Kernel, xs: Tensor, { mean = 0 }: { mean?: number } = {}) {
  const m = asRows(xs) as Tensor
  const covariance = gram(kernel, m) as Tensor
  return { mean: fromData(new Float64Array(m.shape[0]).fill(mean), [m.shape[0]]), covariance }
}

/**
 * `n` functions drawn from the GP prior at inputs `xs`. The draws are $m\ones + \Lmat\zvec$ with standard normals
 * $\zvec$ from the stream, so draws from the same stream move continuously as the kernel's hyperparameters change.
 *
 * @param s The stream the standard normals are drawn from.
 * @param kernel The kernel $k$.
 * @param xs The inputs, `[m, d]` or `[m]`.
 * @param n The number of functions to draw.
 * @param options `mean`, the constant prior mean $m$ (default 0).
 * @returns The draws `[n, m]`, with the jitter the prior covariance needed to factor.
 *
 * @example Two draws: nearby inputs get nearly equal values
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const { draws, jitter } = samplePrior(stream(0), rbf({ lengthscale: 1, variance: 1 }), tensor([[0], [0.1], [3]]), 2)
 * print('draws', draws, ' jitter', jitter)
 */
export function samplePrior(s: Stream, kernel: Kernel, xs: Tensor, n: number, options: { mean?: number } = {}): Draws {
  const { mean, covariance } = gpPrior(kernel, xs, options)
  return drawsFrom(s, mean, covariance, n)
}

// ── Log marginal likelihood ─────────────────────────────────────────────────────────────────────────────────────

/**
 * The log marginal likelihood $\log p(\yvec \mid \Xmat, \thetavec)$ and its three terms (Rasmussen and Williams, 2006,
 * eq. 5.8).
 */
export type LogMarginal<V = number> = {
  /** $\log p(\yvec \mid \Xmat, \thetavec)$, the sum of the three terms. */
  value: V
  /** $-\frac{1}{2} (\yvec - m\ones)^\top(\Kmat + \sigma^2\Imat)^{-1}(\yvec - m\ones)$. */
  dataFit: V
  /** $-\frac{1}{2} \log \lvert \Kmat + \sigma^2\Imat \rvert$. */
  complexity: V
  /** $-\frac{n}{2} \log 2\pi$. */
  constant: number
  /** Jitter added to $\Kmat + \sigma^2\Imat$ to factor it. */
  jitter: number
  /** True when no jitter made $\Kmat + \sigma^2\Imat$ factor; the terms then come from a partial factor. */
  failed: boolean
}

/**
 * The log marginal likelihood $\log p(\yvec \mid \Xmat, \thetavec)$ (Rasmussen and Williams, 2006, Algorithm 2.1):
 * the sum of the data fit $-\frac{1}{2} \rvec^\top(\Kmat + \sigma^2\Imat)^{-1}\rvec$ with $\rvec = \yvec - m\ones$,
 * the complexity penalty $-\frac{1}{2} \log\lvert \Kmat + \sigma^2\Imat \rvert$ and the constant
 * $-\frac{n}{2} \log 2\pi$. Differentiable when the kernel's hyperparameters, the inputs or the noise variance are
 * traced.
 *
 * @param kernel The kernel $k$, possibly with traced hyperparameters.
 * @param x The training inputs $\Xmat$, `[n, d]` or `[n]`.
 * @param y The targets $\yvec$, `[n]` or `[n, 1]`.
 * @param options The noise variance $\sigma^2$ (default 0, with jitter added if $\Kmat$ alone does not factor) and
 *   the constant prior mean $m$ (default 0).
 * @returns The value, its three terms, and the jitter the factorisation needed.
 *
 * @example Two points, against the value by hand
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [1]])
 * const y = tensor([1, -1])
 * const lml = logMarginalLikelihood(rbf({ lengthscale: 1, variance: 1 }), x, y, { noiseVariance: 0.1 })
 * print('log p(y | X)', lml.value, ' data fit', lml.dataFit, ' complexity', lml.complexity)
 * // K + 0.1 I = [[a, c], [c, a]] with a = 1.1 and c = exp(-1/2)
 * const a = 1.1, c = Math.exp(-0.5), det = a * a - c * c
 * print('by hand', -(2 * a + 2 * c) / det / 2 - Math.log(det) / 2 - Math.log(2 * Math.PI))
 */
export function logMarginalLikelihood(
  kernel: Kernel,
  x: Value,
  y: Tensor,
  options: GpOptions = {},
): LogMarginal<Value> {
  const { noiseVariance = 0, mean = 0 } = options
  const r = sub(targets(y), mean)
  const n = r.shape[0]
  const { L, jitter, failed } = cholesky(noisyGram(kernel, x, noiseVariance))
  const z = solveTriangular(L, r)
  const dataFit = mul(-0.5, sum(square(z)))
  const complexity = mul(-0.5, choleskyLogDet(L))
  const constant = -0.5 * n * LOG_2PI
  return { value: add(add(dataFit, complexity), constant), dataFit, complexity, constant, jitter, failed }
}

/** The gradient of the log marginal likelihood in the kernel's hyperparameters and the noise variance. */
export type LogMarginalGradient<P extends KernelParams = KernelParams> = {
  /** The log marginal likelihood $\log p(\yvec \mid \Xmat, \thetavec)$. */
  value: number
  /** $\partial / \partial \theta$ for each kernel hyperparameter, as a tree shaped like `kernel.params`. */
  kernel: P
  /** $\partial / \partial \sigma^2$ (NaN when $\sigma^2 = 0$, as it is recovered from the log-space gradient). */
  noiseVariance: number
  /** Names of the log-parameters in `logGradient`: the kernel's hyperparameter paths, then "noiseVariance". */
  names: string[]
  /**
   * $\partial / \partial \log \theta$ for every kernel hyperparameter, then $\partial / \partial \log \sigma^2$ (the
   * gradient scikit-learn reports).
   */
  logGradient: Float64Array
}

/**
 * The log marginal likelihood and its gradient, whose entry for $\theta_j$ is
 * $\frac{1}{2} \trace((\alphavec\alphavec^\top - \Kmat^{-1}) \partial \Kmat / \partial \theta_j)$ (Rasmussen and
 * Williams, 2006, eq. 5.9), computed by reverse-mode differentiation through the Cholesky factor
 * (`aifn-compute/foundation/autodiff`) in log space, and divided by $\theta$ for the gradient in the hyperparameters
 * themselves.
 *
 * @param kernel The kernel $k$; every hyperparameter must be positive.
 * @param x The training inputs $\Xmat$, `[n, d]` or `[n]`.
 * @param y The targets $\yvec$, `[n]` or `[n, 1]`.
 * @param options `noiseVariance`, $\sigma^2$ (default 0; give a positive value for a finite derivative in
 *   $\sigma^2$), and `mean`, the constant prior mean $m$ (default 0).
 * @returns The value, the gradient as a tree and in log space, and the names of the log-parameters.
 *
 * @example The lengthscale derivative against a central difference
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [1], [2]])
 * const y = tensor([0, 1, 0])
 * const g = logMarginalLikelihoodGradient(rbf({ lengthscale: 1, variance: 1 }), x, y, { noiseVariance: 0.1 })
 * print('d/d lengthscale', g.kernel.lengthscale, ' d/d noise variance', g.noiseVariance)
 * const at = (l) => logMarginalLikelihood(rbf({ lengthscale: l, variance: 1 }), x, y, { noiseVariance: 0.1 }).value
 * print('central difference', (at(1.001) - at(0.999)) / 0.002)
 */
export function logMarginalLikelihoodGradient<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  options: { noiseVariance?: number; mean?: number } = {},
): LogMarginalGradient<P> {
  const { noiseVariance = 0, mean = 0 } = options
  const lv = kernelLogVector(kernel)
  const f = (tree: P, noise: Value) =>
    logMarginalLikelihood(kernelFromLog(kernel, tree), x, y, { noiseVariance: noise as number, mean }).value
  const { value, grad } = valueAndGrad(f, { argnums: [0, 1] })(lv.unravel(lv.vector), noiseVariance)
  const [kernelGrad, noiseGrad] = grad as [P, Value]
  const kernelLogGrad = ravel(kernelGrad).vector
  const directNoise = typeof noiseGrad === 'number' ? noiseGrad : flat(noiseGrad as Tensor)[0]
  const logNoiseGrad = directNoise * noiseVariance
  const logGradient = Float64Array.from([...kernelLogGrad, logNoiseGrad])
  // ∂/∂θ = (∂/∂ log θ) / θ.
  const params = ravel(kernel.params)
  const direct = Float64Array.from(params.vector, (theta, i) => kernelLogGrad[i] / theta)
  return {
    value: value as number,
    kernel: params.unravel(direct),
    noiseVariance: directNoise,
    names: [...lv.names, 'noiseVariance'],
    logGradient,
  }
}

// ── Posterior ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Predictive moments at test inputs. */
export type Prediction = {
  /** Posterior mean of $f$ (or $y$) at each test input, `[m]`. */
  mean: Tensor
  /**
   * Posterior variance, `[m]`; entries computed below 0 by rounding are set to 0 (before any noise is added) and
   * counted in `clipped`.
   */
  variance: Tensor
  /** Full covariance $m \times m$ when asked for (`full: true`). */
  covariance?: Tensor
  /** How many variances were negative by rounding and set to 0. */
  clipped: number
}

/** The exact GP posterior given noisy observations. */
export interface GpPosterior<P extends KernelParams = KernelParams> {
  /** The kernel $k$. */
  readonly kernel: Kernel<P>
  /** Training inputs `[n, d]`. */
  readonly x: Tensor
  /** Training targets `[n]`. */
  readonly y: Tensor
  /** The observation-noise variance $\sigma^2$. */
  readonly noiseVariance: number
  /** The constant prior mean $m$. */
  readonly mean: number
  /** Lower Cholesky factor of $\Kmat + \sigma^2\Imat$ (plus $j\Imat$ for the jitter $j$). */
  readonly L: Tensor
  /** $\alphavec = (\Kmat + \sigma^2\Imat)^{-1}(\yvec - m\ones)$, `[n]`. */
  readonly alpha: Tensor
  /** Jitter added to $\Kmat + \sigma^2\Imat$ to factor it (0 when none was needed). */
  readonly jitter: number
  /** True when no jitter made $\Kmat + \sigma^2\Imat$ factor. */
  readonly failed: boolean
  /** The log marginal likelihood of the training targets and its terms (all 0 with no observations). */
  readonly logMarginal: LogMarginal
  /**
   * The posterior of $f$ at `xs` (`[m, d]` or `[m]`): mean $m + \kvec_*^\top\alphavec$ and variance
   * $k_{**} - \norm{\Lmat^{-1}\kvec_*}^2$; with `noise`, the variance of a new observation $y_*$ (plus $\sigma^2$);
   * with `full`, the covariance matrix too.
   */
  predict(xs: Tensor, options?: { full?: boolean; noise?: boolean }): Prediction
  /** The posterior of $f$ at `xs` as a multivariate normal (jitter added to its covariance if needed). */
  latent(xs: Tensor): Multivariate<Tensor>
  /**
   * `n` posterior draws of $f$ at `xs`, `[n, m]`, from the stream's normals (so draws from the same stream move
   * smoothly with the data).
   */
  sample(s: Stream, xs: Tensor, n: number): Draws
}

/**
 * The exact posterior of a GP with kernel `kernel`, constant mean and Gaussian noise (Rasmussen and Williams, 2006,
 * Algorithm 2.1). With no observations it is the prior. Throws `DomainError` for a negative noise variance and
 * `ShapeError` when the numbers of inputs and targets differ; jitter is added and reported when
 * $\Kmat + \sigma^2\Imat$ does not factor.
 *
 * @param kernel The kernel $k$.
 * @param x The training inputs $\Xmat$, `[n, d]` or `[n]`.
 * @param y The targets $\yvec$, `[n]` or `[n, 1]`.
 * @param options `noiseVariance`, $\sigma^2 \ge 0$ (default 0: interpolation), and `mean`, the constant prior mean
 *   $m$ (default 0).
 * @returns The posterior, with `predict`, `latent` and `sample` at new inputs.
 *
 * @example At the training points the mean is near the targets and the variance near the noise; far away, the prior
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [1], [2]])
 * const post = gpPosterior(rbf({ lengthscale: 1, variance: 1 }), x, tensor([0, 1, 0]), { noiseVariance: 0.01 })
 * const p = post.predict(tensor([[0], [1], [2], [10]]))
 * print('mean', p.mean)
 * print('variance', p.variance)
 */
export function gpPosterior<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  options: { noiseVariance?: number; mean?: number } = {},
): GpPosterior<P> {
  const { noiseVariance = 0, mean = 0 } = options
  if (!(noiseVariance >= 0)) throw new DomainError('gpPosterior', 'gpPosterior: noiseVariance must be ≥ 0')
  const X = asRows(x) as Tensor
  const Y = targets(y)
  const n = X.shape[0]
  if (Y.shape[0] !== n) throw new ShapeError('gpPosterior', `gpPosterior: ${n} inputs but ${Y.shape[0]} targets`)
  const r = sub(Y, mean) as Tensor
  let L: Tensor
  let jitter = 0
  let failed = false
  let alpha: Tensor
  let logMarginal: LogMarginal
  if (n === 0) {
    L = fromData(new Float64Array(0), [0, 0])
    alpha = fromData(new Float64Array(0), [0])
    logMarginal = { value: 0, dataFit: 0, complexity: 0, constant: 0, jitter: 0, failed: false }
  } else {
    const c = cholesky(noisyGram(kernel, X, noiseVariance) as Tensor)
    ;({ L, jitter, failed } = c)
    const z = solveTriangular(L, r) as Tensor
    alpha = solveTriangular(L, z, { transpose: true }) as Tensor
    const dataFit = -0.5 * (sum(square(z)) as number)
    const complexity = -0.5 * (choleskyLogDet(L) as number)
    const constant = -0.5 * n * LOG_2PI
    logMarginal = { value: dataFit + complexity + constant, dataFit, complexity, constant, jitter, failed }
  }

  const predict = (
    xs: Tensor,
    { full = false, noise = false }: { full?: boolean; noise?: boolean } = {},
  ): Prediction => {
    const S = asRows(xs) as Tensor
    const m = S.shape[0]
    const extra = noise ? noiseVariance : 0
    if (n === 0) {
      const covariance = full ? (gram(kernel, S) as Tensor) : undefined
      const v = flat(kernelDiagonal(kernel, S) as Tensor).map((u) => u + extra)
      return {
        mean: fromData(new Float64Array(m).fill(mean), [m]),
        variance: fromData(v, [m]),
        covariance: covariance && (add(covariance, diagonalMatrix(m, extra)) as Tensor),
        clipped: 0,
      }
    }
    const Ks = gram(kernel, X, S) as Tensor // [n, m]
    const mu = add(matmul(alpha, Ks), mean) as Tensor
    const V = solveTriangular(L, Ks) as Tensor // [n, m]
    const kss = flat(kernelDiagonal(kernel, S) as Tensor)
    const vv = flat(sum(square(V), 0) as Tensor)
    let clipped = 0
    const variance = Float64Array.from(kss, (k, j) => {
      const v = k - vv[j]
      if (v < 0) {
        clipped++
        return extra
      }
      return v + extra
    })
    let covariance: Tensor | undefined
    if (full) covariance = add(sub(gram(kernel, S), matmul(transpose(V), V)), diagonalMatrix(m, extra)) as Tensor
    return { mean: mu, variance: fromData(variance, [m]), covariance, clipped }
  }

  return {
    kernel,
    x: X,
    y: Y,
    noiseVariance,
    mean,
    L,
    alpha,
    jitter,
    failed,
    logMarginal,
    predict,
    latent: (xs) => {
      const p = predict(xs, { full: true })
      return MultivariateNormal(p.mean, { covariance: p.covariance! }, { jitter: 'auto' }) as Multivariate<Tensor>
    },
    sample: (s, xs, count) => {
      const p = predict(xs, { full: true })
      return drawsFrom(s, p.mean, p.covariance!, count)
    },
  }
}

/**
 * The $m \times m$ matrix $v\Imat$.
 *
 * @param m The number of rows and columns.
 * @param v The value on the diagonal.
 * @returns The matrix.
 */
function diagonalMatrix(m: number, v: number): Tensor {
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++) out[i * m + i] = v
  return fromData(out, [m, m])
}

// ── Hyperparameter fitting ───────────────────────────────────────────────────────────────────────────────────────

/** Options of `fitGp`. */
export type FitGpOptions = {
  /** Starting noise variance $\sigma^2 > 0$ (default 0.1). */
  noiseVariance?: number
  /** Fit the noise variance too (default true); otherwise it stays at `noiseVariance`. */
  fitNoise?: boolean
  /** The constant prior mean $m$, held fixed (default 0). */
  mean?: number
  /** Most L-BFGS steps per start (default 200). */
  maxSteps?: number
  /**
   * Extra starts (default 0), each shifting every log-parameter of the first start by a uniform draw on $[-2, 2]$
   * from `stream`.
   */
  restarts?: number
  /** The stream the restarts are drawn from; required when `restarts` is positive. */
  stream?: Stream
  /** Stop when $\lVert \nabla \rVert$ in log-parameters is at most this (default 1e-6). */
  tolerance?: number
}

/** The result of `fitGp`. */
export type GpFit<P extends KernelParams = KernelParams> = {
  /** The kernel at the fitted hyperparameters. */
  kernel: Kernel<P>
  /** The fitted noise variance $\sigma^2$ (the starting one when `fitNoise` is false). */
  noiseVariance: number
  /** The maximised log marginal likelihood. */
  logMarginal: number
  /**
   * The L-BFGS trace of the best start, over log-parameters (the kernel's hyperparameter paths, then noise), with the
   * log marginal likelihood recorded as `logMarginal`.
   */
  training: Trace<LbfgsState>
  /** The names of the log-parameters: the kernel's hyperparameter paths, then "noiseVariance" when it is fitted. */
  names: string[]
  /** Whether the best start's gradient norm reached `tolerance`. */
  converged: boolean
  /** The log marginal likelihood reached from each start. */
  starts: number[]
}

/**
 * Type-II maximum likelihood: maximise $\log p(\yvec \mid \Xmat, \thetavec)$ over the kernel's hyperparameters (and
 * the noise variance) in log space by L-BFGS, with gradients by `aifn-compute/foundation/autodiff` (Rasmussen and
 * Williams, 2006, §5.4.1). Several starts guard against local optima; the best is kept. A point where the
 * likelihood cannot be computed counts as $+\infty$ in the objective. Throws `DomainError` when `restarts` is
 * positive and no `stream` is given.
 *
 * @param kernel The kernel at the starting hyperparameters (all positive).
 * @param x The training inputs $\Xmat$, `[n, d]` or `[n]`.
 * @param y The targets $\yvec$, `[n]` or `[n, 1]`.
 * @param options The starting noise, what is fitted, the restarts and the stopping rule.
 * @returns The fitted kernel and noise variance, the log marginal likelihood they reach, and the L-BFGS trace.
 *
 * @example A noisy sine: the fit raises the log marginal likelihood from a too-short lengthscale
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const fit = fitGp(rbf({ lengthscale: 0.3, variance: 1 }), x, y)
 * print('lengthscale', fit.kernel.params.lengthscale, ' variance', fit.kernel.params.variance)
 * print('noise variance', fit.noiseVariance, ' converged', fit.converged)
 * print('log p(y | X) from', toFlat(fit.training.series.logMarginal)[0], 'to', fit.logMarginal)
 */
export function fitGp<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  options: FitGpOptions = {},
): GpFit<P> {
  const { noiseVariance = 0.1, fitNoise = true, mean = 0, maxSteps = 200, restarts = 0, tolerance = 1e-6 } = options
  const X = asRows(x) as Tensor
  const Y = targets(y)
  const lv = kernelLogVector(kernel)
  const k = lv.vector.length
  const logNoise = Math.log(noiseVariance)
  const negative = valueAndGrad(
    (tree: P, logNoiseValue: Value) =>
      mul(
        -1,
        logMarginalLikelihood(kernelFromLog(kernel, tree), X, Y, {
          noiseVariance: fitNoise ? exp(logNoiseValue) : noiseVariance,
          mean,
        }).value,
      ),
    { argnums: [0, 1] },
  )
  const objective = (theta: Tensor) => {
    const th = toFlat(theta)
    try {
      const { value, grad } = negative(lv.unravel(th.slice(0, k)), fitNoise ? th[k] : logNoise)
      const v = value as number
      if (!Number.isFinite(v)) return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
      const [kernelGrad, noiseGrad] = grad as [P, Value]
      const g = [
        ...ravel(kernelGrad).vector,
        ...(fitNoise ? [typeof noiseGrad === 'number' ? noiseGrad : flat(noiseGrad as Tensor)[0]] : []),
      ]
      return { value: v, grad: fromData(Float64Array.from(g), [g.length]) }
    } catch {
      return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
    }
  }
  const start0 = [...lv.vector, ...(fitNoise ? [logNoise] : [])]
  const starts: number[][] = [start0]
  if (restarts > 0) {
    const s = options.stream
    if (!s) throw new DomainError('fitGp', 'fitGp: restarts need a stream')
    for (let r = 0; r < restarts; r++) {
      const shift = toFlat(uniform(child(s, 'restart', r), -2, 2, { shape: [start0.length] }) as Tensor)
      starts.push(start0.map((v, i) => v + shift[i]))
    }
  }
  let best: Trace<LbfgsState> | null = null
  const reached: number[] = []
  for (const start of starts) {
    const run = trace(lbfgs(objective, { tolerance }), { x0: tensor(start) }, maxSteps, {
      record: { logMarginal: (s: LbfgsState) => -s.value },
    })
    const final = run.final
    reached.push(-final.value)
    if (!best || final.value < best.final.value) best = run
  }
  const final = best!.final
  const theta = toFlat(final.x)
  return {
    kernel: kernelFromLog(kernel, lv.unravel(theta.slice(0, k))),
    noiseVariance: fitNoise ? Math.exp(theta[k]) : noiseVariance,
    logMarginal: -final.value,
    training: best!,
    names: [...lv.names, ...(fitNoise ? ['noiseVariance'] : [])],
    converged: final.converged,
    starts: reached,
  }
}

// ── Estimator ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `gaussianProcessRegressor`. */
export type GaussianProcessRegressorParams<P extends KernelParams = KernelParams> = {
  /** The kernel $k$ (required); with `optimise`, the start of the fit. */
  kernel: Kernel<P>
  /**
   * Observation-noise variance $\sigma^2$ (default 0.1 when optimised, else 0); with `optimise`, the start of the fit.
   */
  noiseVariance?: number
  /**
   * Fit the kernel's hyperparameters (and the noise, with `fitNoise`) by type-II maximum likelihood (default false).
   */
  optimise?: boolean
  /** With `optimise`, fit the noise variance too (default true). */
  fitNoise?: boolean
  /** The constant prior mean $m$ (default 0). */
  mean?: number
  /** With `optimise`, the extra L-BFGS starts (default 0); they draw from the `stream` given to `fit`. */
  restarts?: number
}

/** A fitted GP regression model. */
export interface GaussianProcessRegressionModel<P extends KernelParams = KernelParams>
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Univariate<Tensor>>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Partial<Trained<LbfgsState>> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gaussian-process-regression'
  /** The exact posterior given the training data. */
  readonly posterior: GpPosterior<P>
  /** The kernel used (the fitted one with `optimise`). */
  readonly kernel: Kernel<P>
  /** The noise variance $\sigma^2$ used (the fitted one with `optimise`). */
  readonly noiseVariance: number
  /** The log marginal likelihood of the training targets. */
  readonly logMarginal: number
}

/**
 * GP regression as an estimator. Capabilities: `forward` and `decide` (the posterior mean), `predictive` (a batch of
 * normals $\Gauss(\mu, v + \sigma^2)$ for new observations), `expect`, and `sample` (joint draws of $y_*$ at the
 * inputs, from the full posterior covariance plus noise; one draw `[m]` when no count is given, else `[count, m]`).
 * With `optimise`, hyperparameters are fitted first; the L-BFGS run is kept in `training`.
 *
 * @param params The kernel, the noise variance and whether to fit them.
 * @returns The estimator: `fit({ x, y })` on inputs `[n, d]` (or `[n]`) and targets `[n]` returns a
 *   `GaussianProcessRegressionModel`.
 *
 * @example A fixed kernel: the mean, the predictive variance and a draw
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [1], [2], [3]])
 * const gp = gaussianProcessRegressor({ kernel: rbf({ lengthscale: 1, variance: 1 }), noiseVariance: 0.01 })
 * const model = gp.fit({ x, y: tensor([0, 1, 0, -1]) })
 * const xs = tensor([[1], [1.5]])
 * print('posterior mean', model.forward(xs))
 * print('predictive variance', model.predictive(xs).variance())
 * print('a draw', model.sample(stream(0), xs))
 *
 * @example Hyperparameters by type-II maximum likelihood on a noisy sine
 * // An RBF kernel on 1-d inputs, as `rbf` of aifn-compute/learning/kernels makes it
 * const rbf = (p) => ({ kind: 'kernel', name: 'rbf', params: p, stationary: true, withParams: rbf,
 *   evaluate: (a, b) => mul(p.variance, exp(div(square(sub(a, transpose(b ?? a))), mul(-2, square(p.lengthscale))))),
 *   diagonal: (a) => mul(p.variance, ones([shapeOfValue(a)[0]])) })
 * const x = tensor([[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5]])
 * const y = add(sin(reshape(x, [8])), mul(0.1, normals(stream(0), [8])))
 * const gp = gaussianProcessRegressor({ kernel: rbf({ lengthscale: 0.3, variance: 1 }), optimise: true })
 * const model = gp.fit({ x, y })
 * print('fitted lengthscale', model.kernel.params.lengthscale, ' noise variance', model.noiseVariance)
 * print('mean at pi/2', model.forward(tensor([[Math.PI / 2]])), ' sin(pi/2) = 1')
 */
export function gaussianProcessRegressor<P extends KernelParams>(
  params: GaussianProcessRegressorParams<P>,
): Estimator<Supervised<Tensor, Tensor>, GaussianProcessRegressionModel<P>> {
  const { kernel, optimise = false, fitNoise = true, mean = 0, restarts = 0 } = params
  return {
    name: 'gaussian-process-regression',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      let k = kernel
      let noise = params.noiseVariance ?? (optimise ? 0.1 : 0)
      let training: Trace<LbfgsState> | undefined
      if (optimise) {
        const fitted = fitGp(kernel, x, y, { noiseVariance: noise, fitNoise, mean, restarts, stream: options.stream })
        k = fitted.kernel
        noise = fitted.noiseVariance
        training = fitted.training
      }
      const posterior = gpPosterior(k, x, y, { noiseVariance: noise, mean })
      const forward = (input: Tensor) => posterior.predict(input).mean
      const base = {
        kind: 'model' as const,
        name: 'gaussian-process-regression' as const,
        posterior,
        kernel: k,
        noiseVariance: noise,
        logMarginal: posterior.logMarginal.value,
        ...(training ? { training } : {}),
        forward,
        decide: forward,
        predictive: (input: Tensor) => {
          const p = posterior.predict(input, { noise: true })
          return Normal(p.mean, fromData(flat(p.variance).map(Math.sqrt), p.variance.shape)) as Univariate<Tensor>
        },
        sample: (s: Stream, input: Tensor, count?: number) => {
          const p = posterior.predict(input, { full: true, noise: true })
          const { draws } = drawsFrom(s, p.mean, p.covariance!, count ?? 1)
          return count === undefined ? (reshape(draws, [p.mean.shape[0]]) as Tensor) : draws
        },
      }
      return withExpectation(base)
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'gaussianProcessRegressor',
    module: 'learning/gaussian-processes',
    name: 'Gaussian process regression',
    summary:
      'Exact GP regression with optional type-II maximum-likelihood hyperparameters; the kernel is a required argument.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({
      noiseVariance: real(0, 10, { default: 0 }),
      optimise: bool(),
      fitNoise: bool({ default: true }),
      mean: real(-10, 10, { default: 0 }),
      restarts: int(0, 20, { default: 0 }),
    }),
    notes: ['gaussian-process', 'gaussian-process-hyperparameter-learning'],
    cite: ['rasmussen2006'],
  },
  gaussianProcessRegressor,
)
