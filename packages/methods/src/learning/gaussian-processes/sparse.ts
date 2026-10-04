/**
 * Sparse Gaussian-process regression through m inducing inputs Z: the Nyström approximation Q = K_nm K_mm⁻¹ K_mn of
 * the training covariance, used three ways (Quiñonero-Candela and Rasmussen, 2005, "A unifying view of sparse
 * approximate Gaussian process regression"):
 *
 * - `sor` (subset of regressors) and `dtc` (deterministic training conditional) share the marginal likelihood
 *   N(y | m, Q + σ²I); they differ only in the predictive variance (SoR's collapses away from Z, DTC's does not).
 * - `fitc` (fully independent training conditional; Snelson and Ghahramani, 2006) corrects the diagonal:
 *   N(y | m, Q + diag(K − Q) + σ²I).
 * - `vfe` (Titsias, 2009, "Variational learning of inducing variables in sparse Gaussian processes") keeps the DTC
 *   likelihood term and subtracts tr(K − Q)/(2σ²), so the bound never exceeds the exact log marginal likelihood and
 *   inducing inputs can be optimised without overfitting.
 *
 * The algebra is the numerically stable form of GPflow's `SGPR` and `GPRFITC` (Matthews et al., 2017): with
 * L = chol(K_mm), A = L⁻¹K_mn Λ^{−½} and L_B = chol(I + AAᵀ), everything costs O(nm²). Every quantity is built from
 * tensor primitives, so the bound is differentiable in Z, the kernel's hyperparameters and σ².
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
  method?: SparseMethod
  /** Observation-noise variance σ² > 0. */
  noiseVariance: Value
  mean?: number
  /** Diagonal jitter on K_mm relative to its mean diagonal (default 1e-8), always added and reported. */
  relativeJitter?: number
}

/** The pieces of a sparse GP, possibly traced. */
type Pieces = {
  L: Value
  LB: Value
  c: Value
  bound: Value
  /** The bound's terms. */
  terms: { fit: Value; complexity: Value; trace: Value; constant: number }
  jitter: number
}

function eye(n: number): Tensor {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) out[i * n + i] = 1
  return fromData(out, [n, n])
}

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
  readonly method: SparseMethod
  readonly kernel: Kernel<P>
  /** Inducing inputs Z [m, d]. */
  readonly inducing: Tensor
  readonly noiseVariance: number
  readonly mean: number
  /**
   * The approximate log marginal likelihood (for `vfe`, the evidence lower bound), and its terms: data fit, complexity
   * (−log|L_B| − ½ Σ log Λ), the VFE trace penalty −tr(K − Q)/(2σ²) (0 for the others) and the constant.
   */
  readonly logMarginal: number
  readonly terms: { fit: number; complexity: number; trace: number; constant: number }
  /** Jitter added to K_mm. */
  readonly jitter: number
  /** Predictive mean and variance of f at xs [s, d] (or [s]); `noise` adds σ². */
  predict(xs: Tensor, options?: { noise?: boolean }): { mean: Tensor; variance: Tensor }
}

/**
 * A sparse GP regression with inducing inputs z [m, d] (or [m]) for inputs x [n, d] and targets y [n]. See the module
 * comment for the methods. With z = x, `fitc` and `vfe` reproduce the exact GP.
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

/** The approximate log marginal likelihood (a traced value when its arguments are traced). */
export function sparseLogMarginal(kernel: Kernel, x: Value, y: Tensor, z: Value, options: SparseGpOptions): Value {
  return pieces(kernel, x, y, z, options).bound
}

/** Options of `fitSparseGp` and `sparseGpFitSteps`. */
export type FitSparseGpOptions = {
  method?: SparseMethod
  /** Starting noise variance (default 0.1). */
  noiseVariance?: number
  fitNoise?: boolean
  /** Move the inducing inputs too (default true). */
  fitInducing?: boolean
  /** Fit the kernel's hyperparameters (default true); otherwise they stay as given. */
  fitKernel?: boolean
  mean?: number
  maxSteps?: number
  tolerance?: number
  /** Also record the exact GP's log marginal likelihood at every state's hyperparameters (O(n³) a state). */
  exact?: boolean
}

/** What every state of a sparse-GP fit reports: the inducing inputs, hyperparameters and objective it has reached. */
export type SparseGpFitFields = {
  /** Inducing inputs Z [m, d]. */
  inducing: Tensor
  /** The kernel's log hyperparameters, in `kernelLogVector` order (rebuild the kernel with `sparseGpAt`). */
  logKernel: Float64Array
  /** The kernel's hyperparameters by name (pytree path). */
  hyper: Record<string, number>
  noiseVariance: number
  /** The approximate log marginal likelihood (for `vfe`, the ELBO). */
  logMarginal: number
  /** With `exact`: the exact GP's log marginal likelihood at the same hyperparameters (≥ the VFE bound). */
  exact?: number
}

/** The state of `sparseGpFitSteps`: an L-BFGS state with the fit's fields. */
export type SparseGpFitState = LbfgsState & SparseGpFitFields

/** The fitting problem: the objective over θ = [log θ_k, log σ², Z] (each part present when fitted) and its layout. */
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

function exactLogMarginal(kernel: Kernel, x: Tensor, y: Tensor, noiseVariance: number, mean: number): number {
  return logMarginalLikelihood(kernel, x, y, { noiseVariance, mean }).value as number
}

/**
 * The sparse GP's fit as a traceable algorithm: L-BFGS on minus the approximate log marginal likelihood (the ELBO for
 * `vfe`) over the kernel's log hyperparameters, log σ² and the inducing inputs Z, with gradients from
 * `aifn-compute/foundation/autodiff`. Every state carries Z, the hyperparameters and the objective, so a figure can play the
 * optimisation; `init` takes nothing. The objective never decreases from one state to the next (a Wolfe line search).
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

/** The sparse GP at a fit state's inducing inputs and hyperparameters (`kernel` is the template the fit started from). */
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
  model: SparseGp<P>
  /** The trace of `sparseGpFitSteps`, with the objective recorded as `logMarginal`. */
  training: Trace<SparseGpFitState>
  converged: boolean
}

/**
 * Maximise the sparse approximation's log marginal likelihood (the ELBO for `vfe`) over the kernel's log
 * hyperparameters, log σ² and the inducing inputs, by L-BFGS (`sparseGpFitSteps` run to `maxSteps`, default 200).
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
  /** Candidates: this many training rows drawn without replacement from the stream (default min(n, 50)). */
  candidates?: number
  /** L-BFGS steps on the hyperparameters and Z after each addition (default 0: hyperparameters held, Z only grows). */
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
 * Greedy selection of inducing inputs (in the spirit of Seeger et al., 2003, and Titsias, 2009, §3): each step adds the
 * candidate training input whose addition most increases the objective (the ELBO for `vfe`, the approximate log
 * marginal likelihood otherwise), then optionally re-optimises everything by `reoptimise` L-BFGS steps. Each candidate is
 * scored exactly, at O(nm²) for m inducing inputs, so a step costs O(cnm²) for c candidates. With the hyperparameters
 * held, the VFE bound never decreases as Z grows. `init` draws the candidates from its stream.
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
  kernel: Kernel<P>
  /** Number of inducing inputs m, chosen from the training inputs without replacement (default min(n, 20)). */
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
  readonly sparse: SparseGp<P>
}

/**
 * Sparse GP regression as an estimator (the C1 conformance of `sparseGp`/`fitSparseGp`): the inducing inputs start
 * at `inducing` training inputs drawn from `options.stream` (the first ones without a stream), and with `optimise` the
 * kernel, noise and inducing inputs are fitted by `fitSparseGp`, whose L-BFGS run is kept in `training`.
 * Capabilities: `forward` and `decide` (the predictive mean), `predictive` (normals N(mean, var + σ²)), `expect`.
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
