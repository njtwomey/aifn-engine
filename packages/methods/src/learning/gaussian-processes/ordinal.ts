/**
 * Gaussian-process ordinal regression (Chu & Ghahramani, 2005, "Gaussian processes for ordinal regression", JMLR 6):
 * a latent function f ~ GP(0, k) and the cumulative probit likelihood with noise σ,
 * P(y = k | f) = Φ((θ_k − f)/σ) − Φ((θ_{k−1} − f)/σ), which is `aifn-compute/probability/likelihoods`' cumulative-probit
 * `ordinalLikelihood` at η = f/σ and thresholds θ/σ. The posterior is approximated by Laplace's method: the Newton
 * mode search of `./classification` (`laplaceMode`) with the likelihood's gradient and negative Hessian diagonal W
 * taken by automatic differentiation of the compute likelihood (a reverse pass and one Hessian–vector product with the
 * ones vector, exact because the likelihood factorises). The evidence is log p(y | f̂) − ½ f̂ᵀK⁻¹f̂ − ½ log|B|,
 * B = I + W^½KW^½ (their eq. 14). With `optimise`, the thresholds, σ and the kernel's hyperparameters maximise it,
 * by Nelder–Mead over (ordered-bijector coordinates of θ, log σ, log kernel hyperparameters).
 *
 * Predictions (their eq. 16–17): f* given the data is approximately N(μ*, s*²) with μ* = k*ᵀ∇log p(y | f̂) and
 * s*² = k** − k*ᵀ(K + W⁻¹)⁻¹k*, so P(y* = k) = Φ((θ_k − μ*)/√(σ² + s*²)) − Φ((θ_{k−1} − μ*)/√(σ² + s*²)).
 */

import { hvp, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { int, space } from 'aifn-compute/foundation/space'
import { fromData, sum, toFlat, div, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { run, trace, type Trace } from 'aifn-compute/foundation/trace'
import {
  categoricalPredictive,
  defineModel,
  withExpectation,
  withSampling,
  type AnyUnivariate,
  type Decides,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Scores,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import {
  asRows,
  gram,
  kernelDiagonal,
  kernelFromLog,
  type Kernel,
  type KernelParams,
} from 'aifn-compute/learning/kernels'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import { orderedBijector } from 'aifn-compute/probability/bijectors'
import { ordinalLikelihood } from 'aifn-compute/probability/likelihoods'
import { laplaceAt, laplaceMode, type LaplaceProblem, type LaplaceState, type LaplaceTerms } from './classification'
import { stableFactor } from './classification-ep'
import { kernelLogVector } from './regression'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const LIKELIHOOD = ordinalLikelihood('cumulative', 'probit')

/**
 * The cumulative-probit ordinal likelihood of classes `labels` with thresholds θ and noise σ as a Laplace likelihood:
 * log p(y | f), its gradient and W = −diag ∇² log p(y | f), by reverse mode and one forward-over-reverse pass.
 */
export function ordinalLaplaceTerms(labels: Int32Array, thresholds: ArrayLike<number>, noise: number): LaplaceTerms {
  const n = labels.length
  const theta = fromData(
    Float64Array.from(thresholds, (t) => t / noise),
    [thresholds.length],
  )
  const logLik = (f: Value): Value => sum(LIKELIHOOD.logLik(labels, div(f, noise), theta))
  const vg = valueAndGrad(logLik)
  const ones = fromData(new Float64Array(n).fill(1), [n])
  return (f) => {
    const F = fromData(Float64Array.from(f), [n])
    const { value, grad } = vg(F)
    const curvature = toFlat(hvp(logLik, F, ones) as Tensor)
    return {
      logLik: typeof value === 'number' ? value : toFlat(value as Tensor)[0],
      grad: Float64Array.from(toFlat(grad as Tensor)),
      W: Float64Array.from(curvature, (h) => Math.max(-h, 0)),
    }
  }
}

/** Hyperparameters of `gpOrdinalRegression`. */
export type GpOrdinalRegressionParams<P extends KernelParams = KernelParams> = {
  kernel: Kernel<P>
  /** Number of classes K (default: the largest label + 1). */
  classes?: number
  /** Starting (or, without `optimise`, fixed) increasing thresholds θ [K − 1]; default from the class frequencies. */
  thresholds?: ArrayLike<number>
  /** Starting (or fixed) noise σ > 0 of the latent cdf. Default 1. */
  noise?: number
  /** Maximise the Laplace evidence over θ, σ and the kernel's hyperparameters. Default true. */
  optimise?: boolean
  /** Most Nelder–Mead steps of the evidence maximisation (default 300). */
  hyperSteps?: number
  /** Most Newton steps of each mode search (default 100). */
  maxSteps?: number
  /** Newton tolerance on Ψ (default 1e-10). */
  tolerance?: number
}

/** A fitted GP ordinal regression. */
export interface GpOrdinalRegressionModel<P extends KernelParams = KernelParams>
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Expects<Tensor>,
    Samples<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gp-ordinal-regression'
  readonly kernel: Kernel<P>
  readonly classes: number
  /** The thresholds θ [K − 1], increasing. */
  readonly thresholds: Tensor
  /** The noise σ of the latent cdf. */
  readonly noise: number
  /** The posterior mode f̂ at the training inputs, [n]. */
  readonly mode: Tensor
  /** The Laplace approximation to log p(y | X, θ, σ). */
  readonly logMarginal: number
  readonly converged: boolean
  /** The Newton run at the final hyperparameters. */
  readonly training: Trace<LaplaceState>
  /** Mean and variance of the approximate latent posterior q(f* | y) at xs. */
  latent(xs: Tensor): { mean: Tensor; variance: Tensor }
  /** Class probabilities P(y* = k | x*) [m, K]. */
  probabilities(xs: Tensor): Tensor
}

/** Starting thresholds: the probit of the cumulative class frequencies, scaled by √(σ² + 1). */
function startingThresholds(labels: Int32Array, K: number, noise: number): Float64Array {
  const counts = new Float64Array(K)
  for (const c of labels) counts[c]++
  const n = labels.length
  const theta = new Float64Array(K - 1)
  let cum = 0
  for (let k = 0; k < K - 1; k++) {
    cum += counts[k]
    theta[k] = normalQuantile(Math.min(1 - 0.5 / n, Math.max(0.5 / n, cum / n))) * Math.sqrt(noise * noise + 1)
    if (k > 0) theta[k] = Math.max(theta[k], theta[k - 1] + 1e-3)
  }
  return theta
}

/** The Laplace mode and evidence for a Gram matrix and likelihood (Newton from `warm` when given). */
function laplaceFit(problem: LaplaceProblem, maxSteps: number, warm?: Tensor): LaplaceState {
  return run(laplaceMode(problem), warm ? { f: warm } : {}, maxSteps)
}

/**
 * Gaussian-process ordinal regression (see the module comment): labels are the class indices 0 … K − 1.
 * Capabilities: `forward` and `score` (the latent posterior mean, larger for higher classes), `predictive`
 * (categorical over the K classes), `decide` (the most probable class), `expect` (E[y]), `sample`. The kernel is a
 * required argument.
 */
export function gpOrdinalRegression<P extends KernelParams>(
  params: GpOrdinalRegressionParams<P>,
): Estimator<Supervised<Tensor, Tensor>, GpOrdinalRegressionModel<P>> {
  const { optimise = true, hyperSteps = 300, maxSteps = 100, tolerance = 1e-10 } = params
  return {
    name: 'gp-ordinal-regression',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      const X = asRows(x) as Tensor
      const n = X.shape[0]
      const t = toFlat(y)
      if (t.length !== n)
        throw new ShapeError('gpOrdinalRegression', `gpOrdinalRegression: ${n} inputs but ${t.length} labels`)
      const labels = Int32Array.from(t, (v) => {
        if (!(Number.isInteger(v) && v >= 0))
          throw new DomainError('gpOrdinalRegression', 'gpOrdinalRegression: labels must be class indices 0, 1, …')
        return v
      })
      const K = params.classes ?? Math.max(...labels) + 1
      if (K < 2) throw new DomainError('gpOrdinalRegression', 'gpOrdinalRegression: needs at least two classes')
      if (labels.some((c) => c >= K))
        throw new DomainError('gpOrdinalRegression', `gpOrdinalRegression: a label is outside 0 … ${K - 1}`)
      const labelTensor = fromData(labels, [n])
      let kernel: Kernel<P> = params.kernel
      let noise = params.noise ?? 1
      let theta = params.thresholds ? Float64Array.from(params.thresholds) : startingThresholds(labels, K, noise)
      if (theta.length !== K - 1)
        throw new ShapeError('gpOrdinalRegression', `gpOrdinalRegression: ${K} classes need ${K - 1} thresholds`)
      for (let k = 1; k < theta.length; k++)
        if (!(theta[k] > theta[k - 1]))
          throw new DomainError('gpOrdinalRegression', 'gpOrdinalRegression: thresholds must be increasing')
      const problemFor = (kern: Kernel<P>, th: ArrayLike<number>, sigma: number): LaplaceProblem => ({
        K: gram(kern, X) as Tensor,
        labels: labelTensor,
        likelihood: ordinalLaplaceTerms(labels, th, sigma),
        tolerance,
      })

      if (optimise) {
        const ordered = orderedBijector()
        const lv = kernelLogVector(kernel)
        const m = K - 1
        const unpack = (w: ArrayLike<number>) => ({
          th: toFlat(
            ordered.forward(
              fromData(
                Float64Array.from({ length: m }, (_, i) => w[i]),
                [m],
              ),
            ) as Tensor,
          ),
          sigma: Math.exp(w[m]),
          kern: kernelFromLog(kernel, lv.unravel(Array.from({ length: lv.vector.length }, (_, i) => w[m + 1 + i]))),
        })
        const w0 = new Float64Array(m + 1 + lv.vector.length)
        w0.set(toFlat(ordered.inverse(fromData(theta, [m])) as Tensor), 0)
        w0[m] = Math.log(noise)
        w0.set(lv.vector, m + 1)
        let warm: Tensor | undefined
        const negativeEvidence = (w: Tensor): number => {
          try {
            const { th, sigma, kern } = unpack(toFlat(w))
            const final = laplaceFit(problemFor(kern, th, sigma), maxSteps, warm)
            if (!Number.isFinite(final.logMarginal)) return Infinity
            warm = final.f
            return -final.logMarginal
          } catch {
            return Infinity
          }
        }
        const best = minimize(negativeEvidence, w0, {
          method: 'nelder-mead',
          maxSteps: hyperSteps,
          xTolerance: 1e-6,
          fTolerance: 1e-9,
        })
        const fitted = unpack(toFlat(best.x))
        theta = Float64Array.from(fitted.th)
        noise = fitted.sigma
        kernel = fitted.kern
      }

      const problem = problemFor(kernel, theta, noise)
      const training: Trace<LaplaceState> = trace(laplaceMode(problem), {}, maxSteps, {
        every: options.trace?.every ?? 1,
        record: { objective: (st) => st.objective, logMarginal: (st) => st.logMarginal },
      })
      const final = training.final
      const q = laplaceAt(problem, Float64Array.from(toFlat(final.f)))
      const alpha = q.grad
      const s = q.sW
      const factor = stableFactor(q.K, s)
      const thresholds = fromData(theta, [K - 1])
      const latent = (xs: Tensor) => {
        const S = asRows(xs) as Tensor
        const Ks = gram(kernel, X, S) as Tensor // [n, m]
        const m = S.shape[0]
        const ks = Float64Array.from(toFlat(Ks))
        const mean = new Float64Array(m)
        const scaled = new Float64Array(n * m)
        for (let i = 0; i < n; i++)
          for (let j = 0; j < m; j++) {
            mean[j] += alpha[i] * ks[i * m + j]
            scaled[i * m + j] = s[i] * ks[i * m + j]
          }
        const V = factor.forward(scaled, m)
        const kss = toFlat(kernelDiagonal(kernel, S) as Tensor)
        const variance = Float64Array.from(kss, (k, j) => {
          let acc = 0
          for (let i = 0; i < n; i++) acc += V[i * m + j] ** 2
          return Math.max(k - acc, 0)
        })
        return { mean: fromData(mean, [m]), variance: fromData(variance, [m]) }
      }
      const probabilities = (xs: Tensor): Tensor => {
        const { mean, variance } = latent(xs)
        const mu = toFlat(mean)
        const v = toFlat(variance)
        const out = new Float64Array(mu.length * K)
        mu.forEach((mj, j) => {
          const scale = Math.sqrt(noise * noise + v[j])
          const th = fromData(
            Float64Array.from(theta, (tk) => tk / scale),
            [K - 1],
          )
          out.set(toFlat(LIKELIHOOD.probabilities(mj / scale, th) as Tensor), j * K)
        })
        return fromData(out, [mu.length, K])
      }
      const decide = (xs: Tensor): Tensor => {
        const P = toFlat(probabilities(xs))
        const m = P.length / K
        const out = new Int32Array(m)
        for (let i = 0; i < m; i++) for (let k = 1; k < K; k++) if (P[i * K + k] > P[i * K + out[i]]) out[i] = k
        return fromData(out, [m])
      }
      const forward = (xs: Tensor) => latent(xs).mean
      const base = {
        kind: 'model' as const,
        name: 'gp-ordinal-regression' as const,
        kernel,
        classes: K,
        thresholds,
        noise,
        mode: final.f,
        logMarginal: final.logMarginal,
        converged: final.converged,
        training,
        latent,
        probabilities,
        forward,
        score: forward,
        decide,
        predictive: (xs: Tensor): AnyUnivariate => categoricalPredictive(probabilities(xs)),
      }
      return withSampling(withExpectation(base))
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'gpOrdinalRegression',
    module: 'learning/gaussian-processes',
    name: 'Gaussian process ordinal regression',
    summary:
      'A GP latent function with the cumulative probit likelihood, by the Laplace approximation; thresholds, noise ' +
      'and kernel fitted to the evidence. The kernel is a required argument.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'score', 'sample'],
    hyper: space({ hyperSteps: int(0, 2000, { default: 300 }), maxSteps: int(1, 1000, { default: 100 }) }),
    notes: ['gaussian-process-ordinal-regression', 'ordinal-regression'],
    cite: ['chu2005gp'],
  },
  gpOrdinalRegression,
)
