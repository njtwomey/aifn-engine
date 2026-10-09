/**
 * The evidence lower bound and its Monte Carlo gradient estimators.
 *
 * For an unnormalised target $\tilde p(\xvec) = Z\, p(\xvec \mid \text{data})$ and $q_{\lambdavec}$,
 * $\log Z = \operatorname{ELBO}(\lambdavec) + \KL(q_{\lambdavec} \,\|\, p)$, with
 * $\operatorname{ELBO}(\lambdavec) = \expect_q[\log \tilde p(\xvec)] + \entropy[q_{\lambdavec}]$. Variational
 * inference here minimises the reverse divergence $\KL(q \,\|\, p)$, which is zero-forcing: $q$ avoids regions where
 * $p$ is small, so it under-covers and picks one mode of a multimodal $p$.
 *
 * - Reparameterisation (pathwise; Kingma & Welling, 2014; Titsias & Lázaro-Gredilla, 2014):
 *   $\xvec_s = \muvec + \Lmat\epsilonvec_s$ and, with $\Jmat_s = \partial \xvec_s / \partial \lambdavec$ and
 *   $\gvec_s = \nabla \log \tilde p(\xvec_s)$,
 *   $\nabla \operatorname{ELBO} \approx \frac{1}{S} \sum_s \Jmat_s^\top \gvec_s + \nabla \entropy$, with the
 *   entropy term exact.
 * - Score function (REINFORCE; Williams, 1992; Ranganath, Gerrish & Blei, 2014): with the score
 *   $\hvec = \nabla_{\lambdavec} \log q(\xvec)$ and $f = \log \tilde p(\xvec) - \log q(\xvec)$,
 *   $\nabla \operatorname{ELBO} = \expect_q[\hvec\,(f - b)]$ for any constant $b$, since $\expect_q[\hvec] = \zeros$.
 *   Baselines: `leave-one-out` ($b$ for sample $s$ is the mean of the other samples' $f$, which keeps the estimator
 *   unbiased) and `control-variate` (the per-coordinate optimal scale $a^*_i = \cov(h_i f, h_i) / \var(h_i)$;
 *   Ranganath et al., 2014, eq. 9), estimated for each sample from the other samples so the estimator stays unbiased.
 *
 * The functions take the target as a `LogDensity` and $\lambdavec$ as a flat vector of the family's parameters, and
 * draw from the stream they are given, so a seeded stream gives the same estimate every time.
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

/**
 * A scalar value as a number.
 *
 * @param v A number or a scalar tensor (traced values are unwrapped).
 * @returns Its value.
 */
const toNumber = (v: Value) => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : item(raw)
}

/**
 * $\log \tilde p(\xvec)$ as a number.
 *
 * @param target The unnormalised target; its `logDensity` is called on $\xvec$ as a rank-1 tensor.
 * @param x The point $\xvec$, a working array of `target.dim` values (not modified).
 * @returns The log density at $\xvec$.
 */
export function logTarget(target: LogDensity, x: F64): number {
  return toNumber(target.logDensity(vec(x)) as Value)
}

/**
 * $\log \tilde p(\xvec)$ and $\nabla \log \tilde p(\xvec)$, from `target.grad` when given, else by reverse-mode
 * autodiff of `target.logDensity`.
 *
 * @param target The unnormalised target.
 * @param x The point $\xvec$, a working array of `target.dim` values (not modified).
 * @returns `value`, the log density, and `grad`, its gradient as a new working array.
 */
export function logTargetAndGrad(target: LogDensity, x: F64): { value: number; grad: F64 } {
  if (target.grad) return { value: logTarget(target, x), grad: toF64(target.grad(vec(x)) as Vector) }
  const r = valueAndGrad((t: Tensor) => target.logDensity(t) as Value)(vec(x))
  return { value: toNumber(r.value as Value), grad: Float64Array.from(toFlat(r.grad as Tensor)) }
}

/** An ELBO estimate from $S$ draws of $q$. */
export type ElboEstimate = {
  /** $\frac{1}{S} \sum_s (\log \tilde p(\xvec_s) - \log q(\xvec_s))$. */
  value: number
  /** Its Monte Carlo standard error (NaN when $S = 1$). */
  standardError: number
  /** The draws $\xvec_s$, one per row ($S \times d$). */
  draws: Tensor
  /** $\log \tilde p(\xvec_s)$ at each draw ($S$ values). */
  logTarget: Vector
  /** $\log q(\xvec_s)$ at each draw ($S$ values). */
  logQ: Vector
}

/**
 * Estimate $\operatorname{ELBO}(\lambdavec) = \expect_q[\log \tilde p(\xvec) - \log q(\xvec)]$ with `samples` draws
 * from $q_{\lambdavec}$ (default 1000), drawn from `s`. When $\tilde p$ is normalised ($Z = 1$),
 * $-\operatorname{ELBO}$ is $\KL(q \,\|\, p)$; when $q$ is the exact posterior, every draw gives $\log Z$.
 *
 * @param s The random stream the draws come from; advanced.
 * @param target The unnormalised target $\tilde p$, of dimension `family.dim`.
 * @param family The variational family $q$.
 * @param lambda The family's parameters $\lambdavec$ (`family.size` values).
 * @param options `samples`, the number of draws $S$ (default 1000).
 * @returns The estimate, its standard error, and the draws with $\log \tilde p$ and $\log q$ at each.
 *
 * @example At the exact posterior the ELBO is the log evidence, with no noise
 * // log p(m) = 3m - 2m^2 + c: the posterior of a mean m with prior N(0, 1) after three unit-noise observations
 * // summing to 3, which is N(0.75, 0.5^2), with log Z = 1.125 + log(pi / 2) / 2.
 * const target = { kind: 'log-density', dim: 1, logDensity: (m) => sum(sub(mul(3, m), mul(2, mul(m, m)))) }
 * const q = meanFieldGaussian(1)
 * const exact = elbo(stream(0), target, q, q.parameters([0.75], 0.5), { samples: 100 })
 * print('ELBO =', exact.value, 'standard error', exact.standardError)
 * print('log Z =', 1.125 + Math.log(Math.PI / 2) / 2)
 *
 * @example Elsewhere it falls short of log Z by the KL divergence
 * // log p(m) = 3m - 2m^2 + c: the posterior of a mean m with prior N(0, 1) after three unit-noise observations
 * // summing to 3, which is N(0.75, 0.5^2), with log Z = 1.125 + log(pi / 2) / 2.
 * const target = { kind: 'log-density', dim: 1, logDensity: (m) => sum(sub(mul(3, m), mul(2, mul(m, m)))) }
 * const q = meanFieldGaussian(1)
 * const prior = elbo(stream(0), target, q, q.parameters([0], 1))
 * print('ELBO at q = N(0, 1):', prior.value, '±', prior.standardError)
 * // KL(N(0, 1) || N(0.75, 0.25)) = log 0.5 + (1 + 0.75^2) / 0.5 - 0.5
 * print('exact', 1.125 + Math.log(Math.PI / 2) / 2 - (Math.log(0.5) + (1 + 0.75 ** 2) / 0.5 - 0.5))
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
  /** `'reparameterisation'` (default; needs a differentiable target) or `'score'`. */
  estimator?: GradientEstimator
  /** Draws $S$ per estimate. Default 1 (reparameterisation) or 10 (score). */
  samples?: number
  /**
   * Score estimator only. Default `leave-one-out` when $S \ge 2$, else `none`; `leave-one-out` needs $S \ge 2$ and
   * `control-variate` $S \ge 3$ (else `DomainError`).
   */
  baseline?: Baseline
}

/** A stochastic ELBO gradient. */
export type ElboGradient = {
  /** The estimate of $\nabla_{\lambdavec} \operatorname{ELBO}$ (`family.size` values). */
  grad: Vector
  /** The ELBO estimate from the same draws. */
  elbo: number
  /** The draws ($S \times d$). */
  draws: Tensor
}

/**
 * One Monte Carlo estimate of $\nabla_{\lambdavec} \operatorname{ELBO}(\lambdavec)$ by the chosen estimator (see the
 * file comment), with draws from `s`. Both estimators are unbiased with every baseline. A baseline that needs more
 * samples than given throws `DomainError`.
 *
 * @param s The random stream the draws come from; advanced.
 * @param target The unnormalised target $\tilde p$. The reparameterisation estimator uses its `grad` when given, else
 *   differentiates `logDensity`; the score estimator only evaluates it.
 * @param family The variational family $q$.
 * @param lambda The family's parameters $\lambdavec$ (`family.size` values).
 * @param options The estimator, the number of draws and the baseline (defaults: reparameterisation with one draw).
 * @returns The gradient estimate, the ELBO estimate from the same draws, and the draws.
 *
 * @example Both estimators agree on average
 * // log p(m) = 3m - 2m^2 + c: the posterior of a mean m with prior N(0, 1) after three unit-noise observations
 * // summing to 3, which is N(0.75, 0.5^2), with log Z = 1.125 + log(pi / 2) / 2.
 * const target = { kind: 'log-density', dim: 1, logDensity: (m) => sum(sub(mul(3, m), mul(2, mul(m, m)))) }
 * // At q = N(0, 1) the exact gradient over (mean, log sd) is (3 - 4 * 0, 1 - 4 * 1) = (3, -3).
 * const q = meanFieldGaussian(1)
 * const lambda = q.parameters([0], 1)
 * print('reparameterisation:', elboGradient(stream(0), target, q, lambda, { samples: 1000 }).grad)
 * print('score:', elboGradient(stream(0), target, q, lambda, { estimator: 'score', samples: 1000 }).grad)
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
  /** The mean of each coordinate of the estimate over the repeats. */
  mean: Vector
  /** The variance of each coordinate of the estimate over the repeats (divisor `repeats` $- 1$). */
  variance: Vector
  /** $\sum_i \var \hat g_i$, the trace of the estimator's covariance. */
  totalVariance: number
  /** The number of estimates the spread was measured over. */
  repeats: number
}

/**
 * The variance of an ELBO gradient estimator at $\lambdavec$, from `repeats` independent estimates (default 200; repeat
 * $r$ draws from `child(s, r)`, so `s` itself is not advanced). The reparameterisation estimator typically has a much
 * lower variance than the score function's, which is why it is preferred whenever $\log \tilde p$ is differentiable.
 *
 * @param s The random stream the repeats' streams are derived from.
 * @param target The unnormalised target $\tilde p$.
 * @param family The variational family $q$.
 * @param lambda The family's parameters $\lambdavec$ at which the estimator is measured.
 * @param options The estimator's options, as `elboGradient` takes them, and `repeats`, the number of estimates
 *   (default 200; at least 2 for a finite variance).
 * @returns The mean and variance of each coordinate over the repeats, and their total.
 *
 * @example The reparameterisation estimator has the lower variance
 * // log p(m) = 3m - 2m^2 + c: the posterior of a mean m with prior N(0, 1) after three unit-noise observations
 * // summing to 3, which is N(0.75, 0.5^2), with log Z = 1.125 + log(pi / 2) / 2.
 * const target = { kind: 'log-density', dim: 1, logDensity: (m) => sum(sub(mul(3, m), mul(2, mul(m, m)))) }
 * const q = meanFieldGaussian(1)
 * const lambda = q.parameters([0], 1)
 * const options = { samples: 10, repeats: 100 }
 * print('reparameterisation:', gradientVariance(stream(0), target, q, lambda, options).totalVariance)
 * print('score:', gradientVariance(stream(0), target, q, lambda, { ...options, estimator: 'score' }).totalVariance)
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
