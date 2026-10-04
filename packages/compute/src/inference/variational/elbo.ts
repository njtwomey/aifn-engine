/**
 * The evidence lower bound and its Monte Carlo gradient estimators.
 *
 * For an unnormalised target p̃(x) = Z p(x | data) and q_λ, log Z = ELBO(λ) + KL(q_λ ‖ p), with
 * ELBO(λ) = E_q[log p̃(x)] + H[q_λ]. Variational inference here minimises the reverse divergence KL(q ‖ p), which is
 * zero-forcing: q avoids regions where p is small, so it under-covers and picks one mode of a multimodal p.
 *
 * - Reparameterisation (pathwise; Kingma & Welling, 2014; Titsias & Lázaro-Gredilla, 2014): x = μ + Lε,
 *   ∇ELBO ≈ (1/S) Σₛ (∂x/∂λ)ᵀ ∇ log p̃(xₛ) + ∇H, with the entropy term exact.
 * - Score function (REINFORCE; Williams, 1992; Ranganath, Gerrish & Blei, 2014):
 *   ∇ELBO = E_q[∇_λ log q(x) (log p̃(x) − log q(x) − b)] for any constant b, since E_q[∇ log q] = 0. Baselines:
 *   `leave-one-out` (b for sample s is the mean of the other samples' f, which keeps the estimator unbiased) and
 *   `control-variate` (the per-coordinate optimal scale a*ᵢ = Cov(hᵢf, hᵢ)/Var(hᵢ) with h = ∇ log q; Ranganath et
 *   al., 2014, eq. 9), estimated for each sample from the other samples so the estimator stays unbiased.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { LogDensity } from 'aifn-compute/foundation/contracts'
import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import { fromData, item, toFlat, unwrap, type Tensor, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import { toF64, vec, type F64, type GaussianFamily, type VectorLike } from './family'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Which gradient estimator. */
export type GradientEstimator = 'reparameterisation' | 'score'
/** Baseline for the score-function estimator. */
export type Baseline = 'none' | 'leave-one-out' | 'control-variate'

const toNumber = (v: Value) => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : item(raw)
}

/** log p̃(x) as a number. */
export function logTarget(target: LogDensity, x: F64): number {
  return toNumber(target.logDensity(vec(x)) as Value)
}

/** log p̃(x) and ∇ log p̃(x), from `target.grad` when given, else by reverse-mode autodiff. */
export function logTargetAndGrad(target: LogDensity, x: F64): { value: number; grad: F64 } {
  if (target.grad) return { value: logTarget(target, x), grad: toF64(target.grad(vec(x)) as Vector) }
  const r = valueAndGrad((t: Tensor) => target.logDensity(t) as Value)(vec(x))
  return { value: toNumber(r.value as Value), grad: Float64Array.from(toFlat(r.grad as Tensor)) }
}

/** An ELBO estimate from S draws of q. */
export type ElboEstimate = {
  /** (1/S) Σ (log p̃(xₛ) − log q(xₛ)). */
  value: number
  /** Its Monte Carlo standard error. */
  standardError: number
  /** The draws xₛ (S×d) and log p̃, log q at each. */
  draws: Tensor
  logTarget: Vector
  logQ: Vector
}

/**
 * Estimate ELBO(λ) = E_q[log p̃(x) − log q(x)] with `samples` draws from q_λ (default 1000), drawn from `s`. When p̃ is
 * normalised (Z = 1), −ELBO is KL(q ‖ p).
 */
export function elbo(
  s: Stream,
  target: LogDensity,
  family: GaussianFamily,
  lambda: VectorLike,
  options: { samples?: number } = {},
): ElboEstimate {
  const S = options.samples ?? 1000
  const l = toF64(lambda)
  const d = family.dim
  const eps = toFlat(normals(s, [S, d]))
  const X = new Float64Array(S * d)
  const lp = new Float64Array(S)
  const lq = new Float64Array(S)
  for (let k = 0; k < S; k++) {
    const e = Float64Array.from(eps.slice(k * d, (k + 1) * d))
    const x = family.kernels.transform(l, e)
    X.set(x, k * d)
    lp[k] = logTarget(target, x)
    lq[k] = family.kernels.logDensity(l, x)
  }
  const f = lp.map((v, k) => v - lq[k])
  const mean = f.reduce((a, b) => a + b, 0) / S
  const variance = S > 1 ? f.reduce((a, b) => a + (b - mean) ** 2, 0) / (S - 1) : NaN
  return {
    value: mean,
    standardError: Math.sqrt(variance / S),
    draws: fromData(X, [S, d]),
    logTarget: vec(lp),
    logQ: vec(lq),
  }
}

/** Options for `elboGradient`. */
export type ElboGradientOptions = {
  estimator?: GradientEstimator
  /** Draws S per estimate. Default 1 (reparameterisation) or 10 (score). */
  samples?: number
  /** Score estimator only. Default `leave-one-out` when S ≥ 2, else `none`. */
  baseline?: Baseline
}

/** A stochastic ELBO gradient. */
export type ElboGradient = {
  /** The estimate of ∇_λ ELBO. */
  grad: Vector
  /** The ELBO estimate from the same draws. */
  elbo: number
  /** The draws (S×d). */
  draws: Tensor
}

/**
 * One Monte Carlo estimate of ∇_λ ELBO(λ) by the chosen estimator (see the module comment), with draws from `s`.
 * Both estimators are unbiased with every baseline.
 */
export function elboGradient(
  s: Stream,
  target: LogDensity,
  family: GaussianFamily,
  lambda: VectorLike,
  options: ElboGradientOptions = {},
): ElboGradient {
  const estimator = options.estimator ?? 'reparameterisation'
  const S = options.samples ?? (estimator === 'score' ? 10 : 1)
  const l = toF64(lambda)
  const d = family.dim
  const P = family.size
  const k = family.kernels
  const eps = toFlat(normals(s, [S, d]))
  const X = new Float64Array(S * d)
  const grad = new Float64Array(P)
  let elboSum = 0
  if (estimator === 'reparameterisation') {
    for (let r = 0; r < S; r++) {
      const e = Float64Array.from(eps.slice(r * d, (r + 1) * d))
      const x = k.transform(l, e)
      X.set(x, r * d)
      const { value, grad: g } = logTargetAndGrad(target, x)
      elboSum += value - k.logDensity(l, x)
      const pg = k.pathGrad(l, e, g)
      for (let i = 0; i < P; i++) grad[i] += pg[i] / S
    }
    const hg = k.entropyGrad(l)
    for (let i = 0; i < P; i++) grad[i] += hg[i]
    return { grad: vec(grad), elbo: elboSum / S, draws: fromData(X, [S, d]) }
  }
  const baseline = options.baseline ?? (S >= 2 ? 'leave-one-out' : 'none')
  if (baseline !== 'none' && S < 2)
    throw new DomainError('elboGradient', `elboGradient: the ${baseline} baseline needs at least 2 samples`)
  const f = new Float64Array(S)
  const h: F64[] = []
  for (let r = 0; r < S; r++) {
    const e = Float64Array.from(eps.slice(r * d, (r + 1) * d))
    const x = k.transform(l, e)
    X.set(x, r * d)
    f[r] = logTarget(target, x) - k.logDensity(l, x)
    h.push(k.score(l, e))
  }
  const total = f.reduce((a, b) => a + b, 0)
  if (baseline === 'control-variate') {
    if (S < 3)
      throw new DomainError('elboGradient', 'elboGradient: the control-variate baseline needs at least 3 samples')
    for (let i = 0; i < P; i++) {
      // a*ᵢ = Cov(hᵢf, hᵢ)/Var(hᵢ), estimated for each sample from the other S − 1 so the estimate stays unbiased.
      let sh = 0
      let sh2 = 0
      let shf = 0
      let shfh = 0
      for (let r = 0; r < S; r++) {
        const hr = h[r][i]
        sh += hr
        sh2 += hr * hr
        shf += hr * f[r]
        shfh += hr * f[r] * hr
      }
      const n = S - 1
      for (let r = 0; r < S; r++) {
        const hr = h[r][i]
        const mh = (sh - hr) / n
        const mhf = (shf - hr * f[r]) / n
        const cov = (shfh - hr * f[r] * hr) / n - mhf * mh
        const v = (sh2 - hr * hr) / n - mh * mh
        const a = v > 0 ? cov / v : 0
        grad[i] += (hr * (f[r] - a)) / S
      }
    }
  } else {
    for (let r = 0; r < S; r++) {
      const b = baseline === 'leave-one-out' ? (total - f[r]) / (S - 1) : 0
      for (let i = 0; i < P; i++) grad[i] += (h[r][i] * (f[r] - b)) / S
    }
  }
  return { grad: vec(grad), elbo: total / S, draws: fromData(X, [S, d]) }
}

/** The spread of an estimator over repeated estimates. */
export type GradientVariance = {
  /** Mean and variance of each coordinate of the estimate over the repeats. */
  mean: Vector
  variance: Vector
  /** Σᵢ Var ĝᵢ, the trace of the estimator's covariance. */
  totalVariance: number
  repeats: number
}

/**
 * The variance of an ELBO gradient estimator at λ, from `repeats` independent estimates (default 200; repeat r draws
 * from `child(s, r)`). The reparameterisation estimator typically has a much lower variance than the score
 * function's, which is why it is preferred whenever log p̃ is differentiable.
 */
export function gradientVariance(
  s: Stream,
  target: LogDensity,
  family: GaussianFamily,
  lambda: VectorLike,
  options: ElboGradientOptions & { repeats?: number } = {},
): GradientVariance {
  const R = options.repeats ?? 200
  const P = family.size
  // Welford's update: no cancellation when the mean is large against the spread.
  const mean = new Float64Array(P)
  const m2 = new Float64Array(P)
  for (let r = 0; r < R; r++) {
    const g = toFlat(elboGradient(child(s, r), target, family, lambda, options).grad)
    for (let i = 0; i < P; i++) {
      const d = g[i] - mean[i]
      mean[i] += d / (r + 1)
      m2[i] += d * (g[i] - mean[i])
    }
  }
  const variance = m2.map((v) => v / (R - 1))
  return { mean: vec(mean), variance: vec(variance), totalVariance: variance.reduce((a, b) => a + b, 0), repeats: R }
}
