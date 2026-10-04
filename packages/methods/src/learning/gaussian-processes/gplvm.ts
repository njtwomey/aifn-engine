/**
 * The Gaussian process latent variable model (Lawrence, 2005, "Probabilistic Non-linear Principal Component Analysis
 * with Gaussian Process Latent Variable Models", JMLR 6): each of the D output dimensions of the data Y [N, D] is an
 * independent GP over latent points X [N, Q] with a shared kernel, so
 *
 *   log p(Y | X, θ) = −(D/2) log|K| − ½ tr(K⁻¹ Y Yᵀ) − (ND/2) log 2π,   K = k(X, X) + σ² I.
 *
 * The MAP fit maximises this plus the prior log p(X) = −½ ‖X‖²_F (standard normal latent points) jointly over X and
 * the log hyperparameters by L-BFGS, with gradients by reverse-mode differentiation (`valueAndGrad`) through the
 * Cholesky factor. The data enter only through the N × N matrix YYᵀ = M Mᵀ, factored once, so an evaluation costs
 * O(N³) whatever D is: tr(K⁻¹YYᵀ) = ‖L⁻¹M‖²_F with K = L Lᵀ. X starts at the principal-component scores, obtained
 * from the same eigendecomposition of YYᵀ (dual PCA, which the GPLVM with a linear kernel reproduces; Lawrence, §3).
 *
 * `project` gives the posterior mean of the outputs at any latent point, the "drag a point, see the data" map of a
 * fitted GPLVM: f(x*) = k(x*, X) K⁻¹ Y, plus the data mean.
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

function identity(n: number): Tensor {
  return fromData(
    Float64Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? 1 : 0)),
    [n, n],
  )
}

/** Options of a GPLVM fit. */
export type GplvmOptions = {
  /** Latent dimensions Q (default 2). */
  latentDim?: number
  /**
   * The stationary kernel, built from its lengthscale and variance (default `rbf`; e.g. `matern52`, `matern32`). Its
   * value at zero distance must be the variance.
   */
  kernel?: (params: StationaryParams) => Kernel
  /** Starting RBF lengthscale ℓ (default 1), signal variance σ_f² (default 1) and noise variance σ² (default 0.1). */
  lengthscale?: number
  signalVariance?: number
  noiseVariance?: number
  /**
   * Least noise variance, in the (standardised) data's units (default 1e-4). With D ≫ N the likelihood alone drives σ²
   * to 0 and the latent points to an interpolating configuration; the noise is parameterised as floor + exp(u).
   */
  noiseFloor?: number
  /** Divide the centred data by their overall sd, so that the starting variances are on the data's scale (default true). */
  standardise?: boolean
  /** Add the standard-normal prior on the latent points (default true: MAP; false: maximum likelihood in X). */
  prior?: boolean
  /** Stop when ‖∇‖ is at most this (default 1e-5). */
  tolerance?: number
}

/**
 * The fitting problem: the data summarised by YYᵀ, the parameter vector's layout, and the negative log posterior with
 * its gradient. θ = (vec X row-major, log ℓ, log σ_f², log(σ² − floor)).
 */
export interface GplvmProblem {
  /** Rows N, output dimensions D, latent dimensions Q. */
  readonly n: number
  readonly d: number
  readonly q: number
  /** The column means subtracted from Y, [D]. */
  readonly mean: Tensor
  /** The scale the centred data were divided by (1 without `standardise`). */
  readonly scale: number
  /** The centred, scaled data [N, D]. */
  readonly y: Tensor
  /** M with M Mᵀ = YYᵀ [N, r], r the numerical rank. */
  readonly m: Tensor
  /** The starting parameter vector (PCA scores and the starting hyperparameters). */
  readonly theta0: Tensor
  /** Fraction of the centred data's variance along each of the first Q principal directions. */
  readonly explained: readonly number[]
  /** The negative log posterior (or likelihood) and its gradient at θ. Non-finite values come back as +∞. */
  objective(theta: Tensor): { value: number; grad: Tensor }
  /** The same objective as a differentiable function of θ (for checks and other transforms). */
  negLogPosterior(theta: Value): Value
  /** The kernel family (from `options.kernel`). */
  readonly kernel: (params: StationaryParams) => Kernel
  /** θ split into its parts. */
  unpack(theta: Tensor): { latent: Tensor; lengthscale: number; signalVariance: number; noiseVariance: number }
}

/** Build the GPLVM fitting problem for data y [N, D]. */
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
  /** Latent points [N, Q]. */
  latent: Tensor
  lengthscale: number
  signalVariance: number
  noiseVariance: number
  /** log p(Y | X, θ) (+ log p(X) with the prior): minus the optimiser's value. */
  logPosterior: number
}

/**
 * The MAP GPLVM fit as a traceable algorithm: L-BFGS on the problem's negative log posterior, from PCA. Every state
 * carries the latent points and hyperparameters, so a figure can play the optimisation. `init` takes nothing.
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
  readonly kind: 'model'
  readonly name: 'gplvm'
  /** Latent points [N, Q]. */
  readonly latent: Tensor
  readonly lengthscale: number
  readonly signalVariance: number
  readonly noiseVariance: number
  readonly logPosterior: number
  /**
   * The posterior mean of the data at latent points x ([m, Q], or one point [Q]): [m, D] (or [D]), in the data's
   * original units.
   */
  project(x: Tensor | readonly number[]): Tensor
  /**
   * The posterior variance of each output dimension at x ([m]; the same for every dimension), in the original units
   * squared; with `noise`, of a new observation.
   */
  variance(x: Tensor | readonly number[], options?: { noise?: boolean }): Tensor
}

/**
 * The model at a state of the fit (or at the parameters of any θ): factors K once so each `project` costs O(mN² + mND)
 * (no solve against all D outputs up front).
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
  model: GplvmModel
  problem: GplvmProblem
  /** The L-BFGS trace, with the log posterior recorded at every step. */
  training: Trace<GplvmState>
  converged: boolean
}

/** Fit a MAP GPLVM to y [N, D] (at most `maxSteps` L-BFGS steps, default 300) and return the model and its trace. */
export function fitGplvm(y: Tensor, options: GplvmOptions & { maxSteps?: number } = {}): GplvmFit {
  const problem = gplvmProblem(y, options)
  const training = trace(gplvmFitSteps(problem, options), undefined, options.maxSteps ?? 300, {
    record: { logPosterior: (s: GplvmState) => s.logPosterior },
  })
  return { model: gplvmModel(problem, training.final), problem, training, converged: training.final.converged }
}
