/**
 * GARCH(1,1) (Bollerslev, 1986): r_t = μ + σ_t ε_t, ε_t ~ N(0, 1), σ_t² = ω + α(r_{t−1} − μ)² + β σ_{t−1}².
 */

import { normals, type Stream } from 'aifn-compute/foundation/random'
import { mean as meanOf, tensor, toFlat, type Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { logit, sigmoid } from 'aifn-compute/numerics/special'
import { simplexFit, type FitState } from './fit'
import { toVec, type VectorLike } from './inputs'

/** GARCH(1,1) parameters: ω > 0, α ≥ 0, β ≥ 0, and the mean μ (default 0). */
export type GarchSpec = { omega: number; alpha: number; beta: number; mean?: number }

/** Summary quantities of a GARCH(1,1): persistence α + β, the unconditional variance and kurtosis where they exist. */
export function garchProperties({ omega, alpha, beta }: GarchSpec): {
  persistence: number
  stationary: boolean
  unconditionalVariance: number
  kurtosis: number
} {
  const p = alpha + beta
  const denom = 1 - p * p - 2 * alpha * alpha
  return {
    persistence: p,
    stationary: p < 1,
    unconditionalVariance: p < 1 ? omega / (1 - p) : Infinity,
    // E[r⁴]/E[r²]² = 3(1 − p²)/(1 − p² − 2α²) when the fourth moment exists (Bollerslev, 1986, Theorem 2).
    kurtosis: p < 1 && denom > 0 ? (3 * (1 - p * p)) / denom : Infinity,
  }
}

/**
 * Simulate n returns and their conditional variances, starting from the unconditional variance (or ω when the model
 * is not stationary) and discarding `burn` values (default 200).
 */
export function simulateGarch(
  s: Stream,
  spec: GarchSpec,
  n: number,
  { burn = 200 }: { burn?: number } = {},
): { returns: Vector; variance: Vector; stationary: boolean } {
  const { omega, alpha, beta, mean = 0 } = spec
  const props = garchProperties(spec)
  const e = toFlat(normals(s, n + burn))
  let v = props.stationary ? props.unconditionalVariance : omega
  let prev2 = v
  const r: number[] = []
  const vs: number[] = []
  for (let t = 0; t < n + burn; t++) {
    v = omega + alpha * prev2 + beta * v
    const x = Math.sqrt(v) * e[t]
    prev2 = x * x
    if (t >= burn) {
      r.push(mean + x)
      vs.push(v)
    }
  }
  return { returns: tensor(r), variance: tensor(vs), stationary: props.stationary }
}

/**
 * The conditional variances σ_t² of a return series under a GARCH(1,1), started (backcast) at the sample variance of
 * the demeaned returns, and the Gaussian log-likelihood −½ Σ (log 2π + log σ_t² + (r_t − μ)²/σ_t²).
 */
export function garchLogLikelihood(r: VectorLike, spec: GarchSpec): { logLikelihood: number; variance: Vector } {
  const xs = toVec(r, 'garchLogLikelihood')
  const { omega, alpha, beta, mean = 0 } = spec
  const d = xs.map((x) => x - mean)
  const s0 = d.reduce((a, v) => a + v * v, 0) / d.length
  let v = s0
  let prev2 = s0
  let ll = 0
  const vs: number[] = []
  for (const x of d) {
    v = omega + alpha * prev2 + beta * v
    vs.push(v)
    ll += -0.5 * (Math.log(2 * Math.PI) + Math.log(v) + (x * x) / v)
    prev2 = x * x
  }
  return { logLikelihood: ll, variance: tensor(vs) }
}

/**
 * A maximum-likelihood GARCH(1,1) fitter as a traceable algorithm: Nelder–Mead over ω = exp(u₀), persistence
 * α + β = logistic(u₁) < 1 and α = (α + β)·logistic(u₂), so every iterate is stationary. μ is the sample mean.
 */
export function garchFitSteps(r: VectorLike): Algorithm<void, FitState<GarchSpec & { logLikelihood: number }>> {
  const xs = toVec(r, 'garchFitSteps')
  const mean = meanOf(tensor(xs))
  const variance = xs.reduce((a, x) => a + (x - mean) ** 2, 0) / xs.length
  const decode = (u: number[]) => {
    const p = sigmoid(u[1])
    const alpha = p * sigmoid(u[2])
    const spec = { omega: Math.exp(u[0]), alpha, beta: p - alpha, mean }
    return { ...spec, logLikelihood: garchLogLikelihood(xs, spec).logLikelihood }
  }
  // Start at persistence 0.9 with α = 0.1, and ω matching the sample variance.
  const u0 = [Math.log(variance * 0.1), logit(0.9), logit(0.1 / 0.9)]
  return simplexFit('garch', (u) => -decode(u).logLikelihood, decode, u0, 1e-10)
}

/** Fit a GARCH(1,1) model by maximum likelihood (see `garchFitSteps`), running at most `maxSteps` steps (default 4000). */
export function fitGarch(
  r: VectorLike,
  { maxSteps = 4000 }: { maxSteps?: number } = {},
): GarchSpec & { logLikelihood: number; converged: boolean; steps: number } {
  const s = run(garchFitSteps(r), undefined, maxSteps)
  return { ...s.params, converged: s.converged, steps: s.t }
}

/**
 * Variance forecasts σ²_{T+h}, h = 1 … horizon, from the next-step variance `next` = σ²_{T+1}: since
 * E[σ²_{t+1} | F_T] = ω + (α + β) E[σ²_t | F_T], σ²_{T+h} = σ̄² + (α + β)^{h−1}(σ²_{T+1} − σ̄²) when α + β < 1.
 */
export function garchForecast(spec: GarchSpec, next: number, horizon: number): Vector {
  const p = spec.alpha + spec.beta
  const out = [next]
  for (let h = 1; h < horizon; h++) out.push(spec.omega + p * out[h - 1])
  return tensor(out.slice(0, horizon))
}
