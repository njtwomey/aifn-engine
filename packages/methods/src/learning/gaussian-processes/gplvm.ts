/**
 * The Gaussian process latent variable model (Lawrence, 2005, "Probabilistic Non-linear Principal Component Analysis
 * with Gaussian Process Latent Variable Models", JMLR 6): each of the $D$ output dimensions of the data $\Ymat$
 * ($N \times D$) is an independent GP over latent points $\Xmat$ ($N \times Q$) with a shared kernel, so, with
 * $\Kmat = k(\Xmat, \Xmat) + \sigma^2 \Imat$, the log likelihood $\log p(\Ymat \mid \Xmat, \thetavec)$ is
 * $-\frac{D}{2} \log\lvert \Kmat \rvert - \frac{1}{2} \trace(\Kmat^{-1} \Ymat\Ymat^\top) - \frac{ND}{2} \log 2\pi$.
 *
 * The MAP fit maximises this plus the prior
 * $\log p(\Xmat) = -\frac{1}{2} \lVert \Xmat \rVert_F^2 - \frac{NQ}{2} \log 2\pi$ (standard normal latent
 * points) jointly over $\Xmat$ and the log hyperparameters by L-BFGS, with gradients by reverse-mode differentiation
 * (`valueAndGrad`) through the Cholesky factor. The data enter only through the $N \times N$ matrix
 * $\Ymat\Ymat^\top = \Mmat\Mmat^\top$, factored once, so an evaluation costs $O(N^3)$ whatever $D$ is:
 * $\trace(\Kmat^{-1}\Ymat\Ymat^\top) = \lVert \Lmat^{-1}\Mmat \rVert_F^2$ with $\Kmat = \Lmat\Lmat^\top$. $\Xmat$
 * starts at the principal-component scores, obtained from the same eigendecomposition of $\Ymat\Ymat^\top$ (dual PCA,
 * which the GPLVM with a linear kernel reproduces; Lawrence, §3). The data are centred, and by default scaled to unit
 * overall sd, before the fit.
 *
 * `project` gives the posterior mean of the outputs at any latent point, the "drag a point, see the data" map of a
 * fitted GPLVM: $\fvec(\xvec_*) = k(\xvec_*, \Xmat) \Kmat^{-1} \Ymat$, plus the data mean.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import {
  add,
  exp,
  fromData,
  matmul,
  mul,
  reshape,
  slice,
  square,
  sum,
  tensor,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { gram, rbf, type Kernel, type StationaryParams } from 'aifn-compute/learning/kernels'
import { cholesky, choleskyLogDet, eigh, solveTriangular } from 'aifn-compute/numerics/linalg'
import { lbfgs, type LbfgsState } from 'aifn-compute/optim/second-order'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * The $n \times n$ identity.
 *
 * @param n The number of rows and columns.
 * @returns $\Imat$.
 */
function identity(n: number): Tensor {
  return fromData(
    Float64Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? 1 : 0)),
    [n, n],
  )
}

/** Options of a GPLVM fit. */
export type GplvmOptions = {
  /** Latent dimensions $Q$ (default 2), from 1 to $N - 1$. */
  latentDim?: number
  /**
   * The stationary kernel, built from its lengthscale and variance (default `rbf`; e.g. `matern52`, `matern32`). Its
   * value at zero distance must be the variance.
   */
  kernel?: (params: StationaryParams) => Kernel
  /** Starting lengthscale $\ell$ of the kernel (default 1). */
  lengthscale?: number
  /** Starting signal variance $\sigma_f^2$ (default 1). */
  signalVariance?: number
  /** Starting noise variance $\sigma^2$ (default 0.1); it must exceed `noiseFloor`. */
  noiseVariance?: number
  /**
   * Least noise variance, in the (standardised) data's units (default 1e-4). With $D \gg N$ the likelihood alone
   * drives $\sigma^2$ to 0 and the latent points to an interpolating configuration; the noise is parameterised as
   * $\sigma^2 = \text{floor} + e^u$.
   */
  noiseFloor?: number
  /**
   * Divide the centred data by their overall sd, so that the starting variances are on the data's scale (default
   * true).
   */
  standardise?: boolean
  /** Add the standard-normal prior on the latent points (default true: MAP; false: maximum likelihood in $\Xmat$). */
  prior?: boolean
  /** Stop when $\lVert \nabla \rVert$ is at most this (default 1e-5). */
  tolerance?: number
}

/**
 * The fitting problem: the data summarised by $\Ymat\Ymat^\top$, the parameter vector's layout, and the negative log
 * posterior with its gradient. The parameter vector is
 * $\thetavec = (\operatorname{vec} \Xmat, \log \ell, \log \sigma_f^2, \log(\sigma^2 - \text{floor}))$, with
 * $\Xmat$ row-major.
 */
export interface GplvmProblem {
  /** The number of rows (data points) $N$. */
  readonly n: number
  /** The number of output dimensions $D$. */
  readonly d: number
  /** The number of latent dimensions $Q$. */
  readonly q: number
  /** The column means subtracted from $\Ymat$, `[D]`. */
  readonly mean: Tensor
  /** The scale the centred data were divided by (1 without `standardise`). */
  readonly scale: number
  /** The centred, scaled data, $N \times D$. */
  readonly y: Tensor
  /** $\Mmat$ with $\Mmat\Mmat^\top = \Ymat\Ymat^\top$, $N \times r$, $r$ the numerical rank (at least $Q$). */
  readonly m: Tensor
  /** The starting parameter vector (PCA scores and the starting hyperparameters). */
  readonly theta0: Tensor
  /** Fraction of the centred data's variance along each of the first $Q$ principal directions. */
  readonly explained: readonly number[]
  /**
   * The negative log posterior (or likelihood) and its gradient at $\thetavec$. Non-finite values come back as
   * $+\infty$.
   */
  objective(theta: Tensor): { value: number; grad: Tensor }
  /** The same objective as a differentiable function of $\thetavec$ (for checks and other transforms). */
  negLogPosterior(theta: Value): Value
  /** The kernel family (from `options.kernel`). */
  readonly kernel: (params: StationaryParams) => Kernel
  /** $\thetavec$ split into its parts: the latent points $N \times Q$ and the three hyperparameters. */
  unpack(theta: Tensor): { latent: Tensor; lengthscale: number; signalVariance: number; noiseVariance: number }
}

/**
 * Build the GPLVM fitting problem: centre (and by default standardise) the data, factor $\Ymat\Ymat^\top$, and start
 * the latent points at the PCA scores scaled so that the first has unit sd. Throws `ShapeError` unless `y` is a
 * matrix, and `DomainError` for a `latentDim` outside $1, \dots, N - 1$ or a starting noise variance not above the
 * floor.
 *
 * @param y The data $\Ymat$, $N \times D$.
 * @param options The latent dimension, the kernel and its starting hyperparameters, the noise floor, the scaling and
 *   the prior.
 * @returns The problem: the objective and its layout, the PCA start and the variance each latent direction explains.
 *
 * @example Six points on a curve in three dimensions, with one latent dimension
 * const y = tensor([0, 0.5, 1, 1.5, 2, 2.5].map((s) => [Math.cos(s), Math.sin(s), s / 2]))
 * const problem = gplvmProblem(y, { latentDim: 1 })
 * print('explained by the first principal direction', problem.explained, ' M', problem.m.shape)
 * print('PCA start', problem.unpack(problem.theta0).latent)
 * print('-log posterior there', problem.objective(problem.theta0).value)
 */
export function gplvmProblem(y: Tensor, options: GplvmOptions = {}): GplvmProblem {
  const {
    latentDim: q = 2,
    lengthscale = 1,
    signalVariance = 1,
    noiseVariance = 0.1,
    noiseFloor = 1e-4,
    standardise = true,
    prior = true,
    kernel: makeKernel = rbf,
  } = options
  if (y.shape.length !== 2) throw new ShapeError('gplvm', `gplvm: y must be [N, D], got [${y.shape.join(', ')}]`)
  const [n, d] = y.shape
  if (!(Number.isInteger(q) && q >= 1 && q < n))
    throw new DomainError('gplvm', `gplvm: latentDim must be in 1 … N − 1, got ${q}`)
  const raw = toFlat(y)
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += raw[i * d + j] / n
  const centred = Float64Array.from(raw, (v, k) => v - mean[k % d])
  const scale = standardise ? Math.sqrt(centred.reduce((a, v) => a + v * v, 0) / (n * d)) || 1 : 1
  const Y = fromData(
    centred.map((v) => v / scale),
    [n, d],
  )
  // YYᵀ = V Λ Vᵀ; M = V Λ^½ over the non-negligible eigenvalues.
  const S = matmul(Y, transpose(Y)) as Tensor
  const { values, vectors } = eigh(S)
  const lam = toFlat(values)
  const V = toFlat(vectors)
  const total = lam.reduce((a, v) => a + Math.max(v, 0), 0)
  const r = Math.max(q, lam.filter((v) => v > 1e-12 * lam[0]).length)
  const M = new Float64Array(n * r)
  for (let i = 0; i < n; i++) for (let j = 0; j < r; j++) M[i * r + j] = V[i * n + j] * Math.sqrt(Math.max(lam[j], 0))
  const m = fromData(M, [n, r])
  // PCA scores v_j √λ_j, all divided by the sd of the first, so the latent points start at the prior's scale.
  const sd1 = Math.sqrt(Math.max(lam[0], 0) / n) || 1
  const theta0 = new Float64Array(n * q + 3)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < q; j++) theta0[i * q + j] = (V[i * n + j] * Math.sqrt(Math.max(lam[j], 0))) / sd1
  theta0[n * q] = Math.log(lengthscale)
  theta0[n * q + 1] = Math.log(signalVariance)
  if (!(noiseVariance > noiseFloor))
    throw new DomainError('gplvm', 'gplvm: the starting noise variance must exceed the floor')
  theta0[n * q + 2] = Math.log(noiseVariance - noiseFloor)

  const eye = identity(n)
  const scalarAt = (theta: Value, i: number) => sum(slice(theta, [i, i + 1]))
  const negLogPosterior = (theta: Value): Value => {
    const X = reshape(slice(theta, [0, n * q]), [n, q])
    const kernel = makeKernel({ lengthscale: exp(scalarAt(theta, n * q)), variance: exp(scalarAt(theta, n * q + 1)) })
    const noise = add(noiseFloor, exp(scalarAt(theta, n * q + 2)))
    const K = add(gram(kernel, X), mul(noise, eye))
    const { L } = cholesky(K)
    const Z = solveTriangular(L, m)
    let value: Value = add(mul(0.5 * d, choleskyLogDet(L)), mul(0.5, sum(square(Z))))
    value = add(value, 0.5 * n * d * LOG_2PI)
    if (prior) value = add(value, add(mul(0.5, sum(square(X))), 0.5 * n * q * LOG_2PI))
    return value
  }
  const vg = valueAndGrad(negLogPosterior)
  const objective = (theta: Tensor) => {
    try {
      const { value, grad } = vg(theta)
      const v = typeof value === 'number' ? value : toFlat(value as Tensor)[0]
      if (!Number.isFinite(v)) return { value: Infinity, grad: fromData(new Float64Array(theta.shape[0])) }
      return { value: v, grad: grad as Tensor }
    } catch {
      return { value: Infinity, grad: fromData(new Float64Array(theta.shape[0])) }
    }
  }
  const unpack = (theta: Tensor) => {
    const th = toFlat(theta)
    return {
      latent: fromData(Float64Array.from(th.slice(0, n * q)), [n, q]),
      lengthscale: Math.exp(th[n * q]),
      signalVariance: Math.exp(th[n * q + 1]),
      noiseVariance: noiseFloor + Math.exp(th[n * q + 2]),
    }
  }
  return {
    n,
    d,
    q,
    mean: fromData(mean),
    scale,
    y: Y,
    m,
    theta0: fromData(theta0),
    explained: lam.slice(0, q).map((v) => Math.max(v, 0) / total),
    objective,
    negLogPosterior,
    kernel: makeKernel,
    unpack,
  }
}

/** The state of a GPLVM fit after t L-BFGS steps. */
export type GplvmState = LbfgsState & {
  /** Latent points, $N \times Q$. */
  latent: Tensor
  /** The kernel's lengthscale $\ell$. */
  lengthscale: number
  /** The signal variance $\sigma_f^2$. */
  signalVariance: number
  /** The noise variance $\sigma^2$. */
  noiseVariance: number
  /** $\log p(\Ymat \mid \Xmat, \thetavec)$ (plus $\log p(\Xmat)$ with the prior): minus the optimiser's value. */
  logPosterior: number
}

/**
 * The MAP GPLVM fit as a traceable algorithm: L-BFGS on the problem's negative log posterior, from PCA. Every state
 * carries the latent points and hyperparameters, so a figure can play the optimisation. `init` takes nothing.
 *
 * @param problem The problem from `gplvmProblem`.
 * @param options `tolerance`: converged when the gradient norm is at most this (default 1e-5).
 * @returns The algorithm; run it with `run` or `trace`.
 *
 * @example The log likelihood rises step by step (without the latent prior)
 * const y = tensor([0, 0.5, 1, 1.5, 2, 2.5].map((s) => [Math.cos(s), Math.sin(s), s / 2]))
 * const problem = gplvmProblem(y, { latentDim: 1, prior: false, noiseFloor: 0.01 })
 * const tr = trace(gplvmFitSteps(problem), undefined, 20, { record: { logLik: (s) => s.logPosterior } })
 * print('log likelihood', tr.series.logLik)
 * print('latent points', tr.final.latent)
 */
export function gplvmFitSteps(
  problem: GplvmProblem,
  options: { tolerance?: number } = {},
): Algorithm<void, GplvmState> {
  const { tolerance = 1e-5 } = options
  const opt = lbfgs(problem.objective, { tolerance })
  const wrap = (s: LbfgsState): GplvmState => ({ ...s, ...problem.unpack(s.x), logPosterior: -s.value })
  return {
    name: 'gplvm',
    init: (_start, stream) => wrap(opt.init({ x0: problem.theta0 }, stream)),
    step: (s, ctx) => wrap(opt.step(s, ctx)),
  }
}

/** A fitted GPLVM: the latent points, hyperparameters and the map from latent space to data space. */
export interface GplvmModel {
  /** The brand of a model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gplvm'
  /** Latent points, $N \times Q$. */
  readonly latent: Tensor
  /** The kernel's lengthscale $\ell$. */
  readonly lengthscale: number
  /** The signal variance $\sigma_f^2$, in the standardised data's units. */
  readonly signalVariance: number
  /** The noise variance $\sigma^2$, in the standardised data's units. */
  readonly noiseVariance: number
  /** The log posterior (or log likelihood without the prior) at these parameters. */
  readonly logPosterior: number
  /**
   * The posterior mean of the data at latent points `x` (`[m, Q]`, or one point `[Q]`): `[m, D]` (or `[D]`), in the
   * data's original units.
   */
  project(x: Tensor | readonly number[]): Tensor
  /**
   * The posterior variance of each output dimension at `x` (`[m]`; the same for every dimension), in the original
   * units squared; with `noise`, of a new observation.
   */
  variance(x: Tensor | readonly number[], options?: { noise?: boolean }): Tensor
}

/**
 * The model at a state of the fit (or at the parameters of any $\thetavec$): factors $\Kmat$ once so each `project`
 * costs $O(mN^2 + mND)$ (no solve against all $D$ outputs up front).
 *
 * @param problem The problem from `gplvmProblem`.
 * @param state The parameter vector `x` ($\thetavec$) and the objective `value` there, as a state of `gplvmFitSteps`
 *   holds them.
 * @returns The model, with `project` and `variance` at new latent points.
 *
 * @example A latent point maps back near its data point; far from the data, to the data mean
 * const y = tensor([0, 0.5, 1, 1.5, 2, 2.5].map((s) => [Math.cos(s), Math.sin(s), s / 2]))
 * const problem = gplvmProblem(y, { latentDim: 1, prior: false, noiseFloor: 0.01 })
 * const model = gplvmModel(problem, run(gplvmFitSteps(problem), undefined, 50))
 * const x0 = toFlat(model.latent)[0]
 * print('the first data point', toFlat(y).slice(0, 3))
 * print('projected from its latent point', model.project([x0]), ' variance', model.variance([x0]))
 * print('far away', model.project([10]), ' variance', model.variance([10]))
 */
export function gplvmModel(problem: GplvmProblem, state: Pick<GplvmState, 'x' | 'value'>): GplvmModel {
  const { latent, lengthscale, signalVariance, noiseVariance } = problem.unpack(state.x)
  const { n, d, q, scale } = problem
  const kernel = problem.kernel({ lengthscale, variance: signalVariance })
  const K = add(gram(kernel, latent), mul(noiseVariance, identity(n))) as Tensor
  const { L } = cholesky(K)
  const mean = toFlat(problem.mean)
  const points = (x: Tensor | readonly number[]) => {
    const t = Array.isArray(x) ? tensor([x as number[]]) : (x as Tensor)
    return {
      t: t.shape.length === 1 ? reshape(t, [1, q]) : t,
      single: Array.isArray(x) || (x as Tensor).shape.length === 1,
    }
  }
  return {
    kind: 'model',
    name: 'gplvm',
    latent,
    lengthscale,
    signalVariance,
    noiseVariance,
    logPosterior: -state.value,
    project: (x) => {
      const { t, single } = points(x)
      // f = k(x*, X) K⁻¹ Y = (K⁻¹ k(X, x*))ᵀ Y: two triangular solves against the m query columns, so building the
      // model costs only the Cholesky factor, not a solve against all D outputs.
      const w = solveTriangular(L, solveTriangular(L, gram(kernel, latent, t)) as Tensor, { transpose: true }) as Tensor
      const f = toFlat(matmul(transpose(w), problem.y) as Tensor)
      const rows = f.length / d
      const out = Float64Array.from(f, (v, k) => v * scale + mean[k % d])
      return single ? fromData(out) : fromData(out, [rows, d])
    },
    variance: (x, options = {}) => {
      const { t } = points(x)
      const ks = gram(kernel, latent, t) as Tensor
      const v = toFlat(solveTriangular(L, ks) as Tensor)
      const rows = t.shape[0]
      const out = new Float64Array(rows)
      for (let j = 0; j < rows; j++) {
        let s = 0
        for (let i = 0; i < n; i++) s += v[i * rows + j] ** 2
        out[j] = (Math.max(signalVariance - s, 0) + (options.noise ? noiseVariance : 0)) * scale * scale
      }
      return fromData(out)
    },
  }
}

/** The result of `fitGplvm`. */
export type GplvmFit = {
  /** The model at the final state. */
  model: GplvmModel
  /** The problem the fit solved. */
  problem: GplvmProblem
  /** The L-BFGS trace, with the log posterior recorded at every step. */
  training: Trace<GplvmState>
  /** Whether the final gradient norm reached `tolerance`. */
  converged: boolean
}

/**
 * Fit a MAP GPLVM (at most `maxSteps` L-BFGS steps, default 300) and return the model and its trace.
 *
 * @param y The data $\Ymat$, $N \times D$.
 * @param options The options of `gplvmProblem`, the tolerance, and `maxSteps`.
 * @returns The model, the problem, the trace and whether it converged.
 *
 * @example Six points on a curve: one latent dimension orders them along it (maximum likelihood in $\Xmat$)
 * const y = tensor([0, 0.5, 1, 1.5, 2, 2.5].map((s) => [Math.cos(s), Math.sin(s), s / 2]))
 * const fit = fitGplvm(y, { latentDim: 1, prior: false, noiseFloor: 0.01 })
 * print('latent points', fit.model.latent, ' converged', fit.converged)
 * print('log likelihood from', toFlat(fit.training.series.logPosterior)[0], 'to', fit.model.logPosterior)
 * print('lengthscale', fit.model.lengthscale, ' noise variance', fit.model.noiseVariance)
 */
export function fitGplvm(y: Tensor, options: GplvmOptions & { maxSteps?: number } = {}): GplvmFit {
  const problem = gplvmProblem(y, options)
  const training = trace(gplvmFitSteps(problem, options), undefined, options.maxSteps ?? 300, {
    record: { logPosterior: (s: GplvmState) => s.logPosterior },
  })
  return { model: gplvmModel(problem, training.final), problem, training, converged: training.final.converged }
}
