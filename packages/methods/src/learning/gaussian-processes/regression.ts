/**
 * Gaussian-process regression with Gaussian noise: the exact posterior by Cholesky factorisation, predictive means and
 * variances, draws from prior and posterior, the log marginal likelihood and its gradient, and hyperparameter fitting.
 *
 * The computations follow Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", Algorithm 2.1
 * (posterior and log marginal likelihood from L = chol(K + σ²I)) and eq. 5.9 (the gradient), with jitter reported
 * rather than hidden whenever K + σ²I is not numerically positive definite.
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
 * back to a log-parameter tree (`aifn-compute/foundation/pytree`'s `ravel`). Gradients are taken with respect to the tree
 * (`valueAndGrad` over pytrees) and ravelled in the same leaf order.
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
  /** Observation-noise variance σ² ≥ 0 (default 0: interpolation, with jitter added if needed and reported). */
  noiseVariance?: Value
  /** Constant prior mean m (default 0). */
  mean?: number
}

/** Float64 contents of a raw tensor, row-major. */
function flat(t: Tensor): Float64Array {
  return Float64Array.from(toFlat(t))
}

/** y as a vector [n]. */
function targets(y: Tensor): Tensor {
  if (y.shape.length === 2 && y.shape[1] === 1) return reshape(y, [y.shape[0]])
  if (y.shape.length !== 1) throw new ShapeError('gp', `gp: targets must be [n], got shape [${y.shape.join(', ')}]`)
  return y
}

/** K + σ²I for the training inputs, possibly traced. */
function noisyGram(kernel: Kernel, x: Value, noiseVariance: Value): Value {
  const K = gram(kernel, x)
  const n = shapeOfValue(K)[0]
  const eye = new Float64Array(n * n)
  for (let i = 0; i < n; i++) eye[i * n + i] = 1
  return add(K, mul(noiseVariance, fromData(eye, [n, n])))
}

// ── Prior ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Draws from a zero-mean (or constant-mean) Gaussian with covariance C, and the jitter its factor needed. */
export type Draws = {
  /** Draws [n, m]: row r is one function evaluated at the m inputs. */
  draws: Tensor
  /** Diagonal jitter added to the covariance to factor it (0 when none was needed). */
  jitter: number
  /** True when even the largest jitter tried did not make it factor. */
  failed: boolean
}

/** mean + Lz for each row z of standard normals [n, m], with L the lower Cholesky factor of cov (jitter reported). */
function drawsFrom(s: Stream, mean: Tensor, cov: Tensor, n: number): Draws {
  const m = mean.shape[0]
  const { L, jitter, failed } = cholesky(cov)
  const z = normals(s, [n, m])
  const draws = add(matmul(z, transpose(L)), reshape(mean, [1, m])) as Tensor
  return { draws, jitter, failed }
}

/**
 * The GP prior at inputs xs [m, d] (or [m]): mean m·1 and covariance k(xs, xs).
 */
export function gpPrior(kernel: Kernel, xs: Tensor, { mean = 0 }: { mean?: number } = {}) {
  const m = asRows(xs) as Tensor
  const covariance = gram(kernel, m) as Tensor
  return { mean: fromData(new Float64Array(m.shape[0]).fill(mean), [m.shape[0]]), covariance }
}

/**
 * `n` functions drawn from the GP prior at inputs xs [m, d] (or [m]), [n, m]. The draws are mean + Lz with fixed
 * standard normals z from the stream, so they move continuously as the kernel's hyperparameters change.
 */
export function samplePrior(s: Stream, kernel: Kernel, xs: Tensor, n: number, options: { mean?: number } = {}): Draws {
  const { mean, covariance } = gpPrior(kernel, xs, options)
  return drawsFrom(s, mean, covariance, n)
}

// ── Log marginal likelihood ─────────────────────────────────────────────────────────────────────────────────────

/** The log marginal likelihood log p(y | X, θ) and its three terms (Rasmussen and Williams, 2006, eq. 5.8). */
export type LogMarginal<V = number> = {
  value: V
  /** −½ (y − m)ᵀ(K + σ²I)⁻¹(y − m). */
  dataFit: V
  /** −½ log |K + σ²I|. */
  complexity: V
  /** −(n/2) log 2π. */
  constant: number
  /** Jitter added to K + σ²I to factor it. */
  jitter: number
  failed: boolean
}

/**
 * log p(y | X, θ) = −½ (y − m)ᵀ(K + σ²I)⁻¹(y − m) − ½ log|K + σ²I| − (n/2) log 2π for inputs x [n, d] (or [n]) and
 * targets y [n] (Rasmussen and Williams, 2006, Algorithm 2.1). Differentiable when the kernel's hyperparameters or the
 * noise variance are traced.
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
  value: number
  /** ∂/∂θ for each kernel hyperparameter, as a tree shaped like `kernel.params`. */
  kernel: P
  /** ∂/∂σ². */
  noiseVariance: number
  /** Names of the log-parameters in `logGradient`: the kernel's hyperparameter paths, then "noiseVariance". */
  names: string[]
  /** ∂/∂ log θ for every kernel hyperparameter, then ∂/∂ log σ² (the gradient scikit-learn reports). */
  logGradient: Float64Array
}

/**
 * The log marginal likelihood and its gradient ∂/∂θⱼ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θⱼ) (Rasmussen and Williams, 2006,
 * eq. 5.9), computed by reverse-mode differentiation through the Cholesky factor (`aifn-compute/foundation/autodiff`).
 */
export function logMarginalLikelihoodGradient<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  options: { noiseVariance?: number; mean?: number } = {},
): LogMarginalGradient<P> {
  const { noiseVariance = 0, mean = 0 } = options
  const lv = kernelLogVector(kernel)
  const f = (tree: P, logNoise: Value) =>
    logMarginalLikelihood(kernelFromLog(kernel, tree), x, y, { noiseVariance: exp(logNoise), mean }).value
  const { value, grad } = valueAndGrad(f, { argnums: [0, 1] })(lv.unravel(lv.vector), Math.log(noiseVariance))
  const [kernelGrad, noiseGrad] = grad as [P, Value]
  const logGradient = Float64Array.from([
    ...ravel(kernelGrad).vector,
    typeof noiseGrad === 'number' ? noiseGrad : flat(noiseGrad as Tensor)[0],
  ])
  // ∂/∂θ = (∂/∂ log θ) / θ.
  const params = ravel(kernel.params)
  const direct = Float64Array.from(params.vector, (theta, i) => logGradient[i] / theta)
  return {
    value: value as number,
    kernel: params.unravel(direct),
    noiseVariance: logGradient[lv.vector.length] / noiseVariance,
    names: [...lv.names, 'noiseVariance'],
    logGradient,
  }
}

// ── Posterior ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Predictive moments at test inputs. */
export type Prediction = {
  /** Posterior mean of f (or y) at each test input, [m]. */
  mean: Tensor
  /** Posterior variance, [m]; entries computed below 0 by rounding are set to 0 and counted in `clipped`. */
  variance: Tensor
  /** Full covariance [m, m] when asked for (`full: true`). */
  covariance?: Tensor
  /** How many variances were negative by rounding and set to 0. */
  clipped: number
}

/** The exact GP posterior given noisy observations. */
export interface GpPosterior<P extends KernelParams = KernelParams> {
  readonly kernel: Kernel<P>
  /** Training inputs [n, d]. */
  readonly x: Tensor
  /** Training targets [n]. */
  readonly y: Tensor
  readonly noiseVariance: number
  readonly mean: number
  /** Lower Cholesky factor of K + σ²I (+ jitter·I). */
  readonly L: Tensor
  /** α = (K + σ²I)⁻¹(y − m), [n]. */
  readonly alpha: Tensor
  /** Jitter added to K + σ²I to factor it (0 when none was needed). */
  readonly jitter: number
  readonly failed: boolean
  readonly logMarginal: LogMarginal
  /**
   * The posterior of f at xs [m, d] (or [m]): mean m + k*ᵀα and variance k** − ‖L⁻¹k*‖²; with `noise`, the variance
   * of a new observation y* (plus σ²); with `full`, the covariance matrix too.
   */
  predict(xs: Tensor, options?: { full?: boolean; noise?: boolean }): Prediction
  /** The posterior of f at xs as a multivariate normal (jitter added to its covariance if needed). */
  latent(xs: Tensor): Multivariate<Tensor>
  /** `n` posterior draws of f at xs, [n, m], from fixed normals of the stream (so they move smoothly with the data). */
  sample(s: Stream, xs: Tensor, n: number): Draws
}

/**
 * The exact posterior of a GP with kernel `kernel`, constant mean and Gaussian noise, given inputs x [n, d] (or [n])
 * and targets y [n] (Rasmussen and Williams, 2006, Algorithm 2.1). With no observations it is the prior.
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

function diagonalMatrix(m: number, v: number): Tensor {
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++) out[i * m + i] = v
  return fromData(out, [m, m])
}

// ── Hyperparameter fitting ───────────────────────────────────────────────────────────────────────────────────────

/** Options of `fitGp`. */
export type FitGpOptions = {
  /** Starting noise variance σ² > 0 (default 0.1). */
  noiseVariance?: number
  /** Fit the noise variance too (default true); otherwise it stays at `noiseVariance`. */
  fitNoise?: boolean
  mean?: number
  /** Most L-BFGS steps per start (default 200). */
  maxSteps?: number
  /** Extra starts from log-uniform perturbations of the initial hyperparameters (default 0), drawn from `stream`. */
  restarts?: number
  stream?: Stream
  /** Stop when ‖∇‖ in log-parameters is at most this (default 1e-6). */
  tolerance?: number
}

/** The result of `fitGp`. */
export type GpFit<P extends KernelParams = KernelParams> = {
  kernel: Kernel<P>
  noiseVariance: number
  /** The maximised log marginal likelihood. */
  logMarginal: number
  /** The L-BFGS trace of the best start, over log-parameters (the kernel's hyperparameter paths, then noise). */
  training: Trace<LbfgsState>
  names: string[]
  converged: boolean
  /** The log marginal likelihood reached from each start. */
  starts: number[]
}

/**
 * Type-II maximum likelihood: maximise log p(y | X, θ) over the kernel's hyperparameters (and the noise variance) in
 * log space by L-BFGS, with gradients by `aifn-compute/foundation/autodiff` (Rasmussen and Williams, 2006, §5.4.1). Several starts guard
 * against local optima.
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
  kernel: Kernel<P>
  /** Observation-noise variance σ² (default 0.1 when fitted, else 0). */
  noiseVariance?: number
  /** Fit the kernel's hyperparameters (and the noise, with `fitNoise`) by type-II maximum likelihood (default false). */
  optimise?: boolean
  fitNoise?: boolean
  mean?: number
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
  readonly posterior: GpPosterior<P>
  readonly kernel: Kernel<P>
  readonly noiseVariance: number
  readonly logMarginal: number
}

/**
 * GP regression as an estimator. Capabilities: `forward` and `decide` (the posterior mean), `predictive` (a batch of
 * normals N(mean, var + σ²) for new observations), `expect`, and `sample` (joint draws of y* at the inputs, from the
 * full posterior covariance plus noise). With `optimise`, hyperparameters are fitted first; the L-BFGS run is kept in
 * `training`.
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
