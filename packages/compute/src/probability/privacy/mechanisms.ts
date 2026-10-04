/**
 * Differentially private mechanisms (Dwork, McSherry, Nissim and Smith, 2006; Dwork and Roth, 2014). A mechanism M is
 * (ε, δ)-differentially private when P(M(D) ∈ S) ≤ e^ε P(M(D′) ∈ S) + δ for all neighbouring datasets D, D′ and sets S.
 *
 * - `laplaceMechanism`: f(D) + Lap(Δ₁/ε) per coordinate, ε-DP for a query of L1 sensitivity Δ₁.
 * - `gaussianMechanism`: f(D) + N(0, σ²) per coordinate. `classicGaussianSigma` is σ = Δ₂√(2 ln(1.25/δ))/ε (valid for
 *   ε < 1); `analyticGaussianSigma` (Balle and Wang, 2018) is the smallest σ whose exact privacy profile
 *   `gaussianDelta(σ, ε)` = Φ(Δ/2σ − εσ/Δ) − e^ε Φ(−Δ/2σ − εσ/Δ) is at most δ, for any ε > 0.
 * - `exponentialMechanism` (McSherry and Talwar, 2007): picks candidate r with probability ∝ exp(ε u(r)/(2Δu)).
 * - `randomisedResponse` (Warner, 1965): each bit kept with probability e^ε/(1 + e^ε), else flipped; ε-DP per bit,
 *   with the unbiased estimate of the true share of ones from the reports.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalLogCdf } from 'aifn-compute/numerics/special'
import { DomainError } from 'aifn-compute/foundation/errors'

const num = (v: unknown) => v as number

const positive = (where: string, name: string, v: number) => {
  if (!(v > 0 && Number.isFinite(v)))
    throw new DomainError(where, `${where}: ${name} must be positive and finite, got ${v}`)
}

/** f(D) + Lap(0, Δ₁/ε) noise on each coordinate: ε-DP for a query of L1 sensitivity Δ₁ (`sensitivity`). */
export function laplaceMechanism(
  value: VectorLike,
  sensitivity: number,
  epsilon: number,
  stream: Stream,
): Float64Array {
  positive('laplaceMechanism', 'sensitivity', sensitivity)
  positive('laplaceMechanism', 'epsilon', epsilon)
  const v = dense.toF64(value, 'laplaceMechanism')
  const b = sensitivity / epsilon
  // Inversion: Lap(0, b) = −b sign(u − ½) log(1 − 2|u − ½|) for u uniform on (0, 1).
  const u = toFlat(uniform(stream, 0, 1, { shape: [v.length] }))
  return Float64Array.from(v, (x, i) => {
    const c = u[i] - 0.5
    return x - b * Math.sign(c) * Math.log1p(-2 * Math.abs(c))
  })
}

/** f(D) + N(0, σ²) noise on each coordinate (σ from `classicGaussianSigma` or `analyticGaussianSigma`). */
export function gaussianMechanism(value: VectorLike, sigma: number, stream: Stream): Float64Array {
  positive('gaussianMechanism', 'sigma', sigma)
  const v = dense.toF64(value, 'gaussianMechanism')
  const noise = toFlat(normal(stream, 0, sigma, { shape: [v.length] }))
  return Float64Array.from(v, (u, i) => u + noise[i])
}

/** The classic calibration σ = Δ₂√(2 ln(1.25/δ))/ε (Dwork and Roth, 2014, Thm. A.1), (ε, δ)-DP for ε < 1. */
export function classicGaussianSigma(sensitivity: number, epsilon: number, delta: number): number {
  positive('classicGaussianSigma', 'sensitivity', sensitivity)
  if (!(epsilon > 0 && epsilon < 1))
    throw new DomainError('classicGaussianSigma', 'classicGaussianSigma: needs 0 < ε < 1')
  if (!(delta > 0 && delta < 1)) throw new DomainError('classicGaussianSigma', 'classicGaussianSigma: needs 0 < δ < 1')
  return (sensitivity * Math.sqrt(2 * Math.log(1.25 / delta))) / epsilon
}

/**
 * The exact δ(ε) of the Gaussian mechanism with noise σ and L2 sensitivity Δ (Balle and Wang, 2018, Thm. 8):
 * Φ(Δ/2σ − εσ/Δ) − e^ε Φ(−Δ/2σ − εσ/Δ), the smallest δ for which it is (ε, δ)-DP.
 */
export function gaussianDelta(sensitivity: number, sigma: number, epsilon: number): number {
  const a = sensitivity / (2 * sigma)
  const b = (epsilon * sigma) / sensitivity
  const first = num(normalCdf(a - b))
  const second = Math.exp(epsilon + num(normalLogCdf(-a - b)))
  return Math.max(first - second, 0)
}

/**
 * The smallest ε at which the Gaussian mechanism with noise σ and L2 sensitivity Δ is (ε, δ)-DP: the inverse of
 * `gaussianDelta` in ε (δ(ε) decreases in ε), by bisection. 0 when δ(0) ≤ δ already.
 */
export function gaussianEpsilon(sensitivity: number, sigma: number, delta: number): number {
  positive('gaussianEpsilon', 'sensitivity', sensitivity)
  positive('gaussianEpsilon', 'sigma', sigma)
  if (!(delta > 0 && delta < 1)) throw new DomainError('gaussianEpsilon', 'gaussianEpsilon: needs 0 < δ < 1')
  if (gaussianDelta(sensitivity, sigma, 0) <= delta) return 0
  let lo = 0
  let hi = 1
  while (gaussianDelta(sensitivity, sigma, hi) > delta) hi *= 2
  for (let k = 0; k < 200 && hi - lo > 1e-13 * hi; k++) {
    const mid = (lo + hi) / 2
    if (gaussianDelta(sensitivity, sigma, mid) > delta) lo = mid
    else hi = mid
  }
  return hi
}

/**
 * The analytic Gaussian mechanism's σ (Balle and Wang, 2018): the smallest σ with `gaussianDelta(Δ, σ, ε)` ≤ δ, by
 * bisection on log σ (δ(σ) decreases in σ). Valid for every ε > 0, and never larger than the classic σ.
 */
export function analyticGaussianSigma(sensitivity: number, epsilon: number, delta: number): number {
  positive('analyticGaussianSigma', 'sensitivity', sensitivity)
  positive('analyticGaussianSigma', 'epsilon', epsilon)
  if (!(delta > 0 && delta < 1))
    throw new DomainError('analyticGaussianSigma', 'analyticGaussianSigma: needs 0 < δ < 1')
  let lo = Math.log(sensitivity) - 20
  let hi = Math.log(sensitivity) + 20
  for (let k = 0; k < 200; k++) {
    const mid = (lo + hi) / 2
    if (gaussianDelta(sensitivity, Math.exp(mid), epsilon) > delta) lo = mid
    else hi = mid
    if (hi - lo < 1e-13) break
  }
  return Math.exp(hi)
}

/** The exponential mechanism's selection probabilities ∝ exp(ε u_r / (2Δu)) over candidates with utilities u. */
export function exponentialMechanismProbabilities(
  utilities: VectorLike,
  sensitivity: number,
  epsilon: number,
): Float64Array {
  positive('exponentialMechanism', 'sensitivity', sensitivity)
  positive('exponentialMechanism', 'epsilon', epsilon)
  const u = dense.toF64(utilities, 'exponentialMechanism')
  const logits = Float64Array.from(u, (x) => (epsilon * x) / (2 * sensitivity))
  // A loop, not Math.max(...logits): spreading a long candidate list overflows the stack (review G2).
  let top = -Infinity
  for (const l of logits) if (l > top) top = l
  if (!(top > -Infinity) || Number.isNaN(top))
    throw new DomainError(
      'exponentialMechanism',
      'exponentialMechanism: needs at least one candidate with a finite utility',
    )
  const w = Float64Array.from(logits, (l) => Math.exp(l - top))
  const z = w.reduce((a, b) => a + b, 0)
  return w.map((x) => x / z)
}

/** Pick a candidate by the exponential mechanism (ε-DP for utilities of sensitivity Δu). */
export function exponentialMechanism(
  utilities: VectorLike,
  sensitivity: number,
  epsilon: number,
  stream: Stream,
): Size {
  const p = exponentialMechanismProbabilities(utilities, sensitivity, epsilon)
  let r = num(uniform(stream))
  for (let i = 0; i < p.length; i++) {
    r -= p[i]
    if (r < 0) return i
  }
  return p.length - 1
}

/** The probability e^ε/(1 + e^ε) that randomised response reports the true bit. */
export const randomisedResponseKeep = (epsilon: number): number => 1 / (1 + Math.exp(-epsilon))

/** Randomised response: each bit (0 or 1) reported truthfully with probability e^ε/(1 + e^ε), else flipped. */
export function randomisedResponse(bits: VectorLike, epsilon: number, stream: Stream): Float64Array {
  positive('randomisedResponse', 'epsilon', epsilon)
  const b = dense.toF64(bits, 'randomisedResponse')
  const keep = randomisedResponseKeep(epsilon)
  const u = toFlat(uniform(stream, 0, 1, { shape: [b.length] }))
  return Float64Array.from(b, (x, i) => (u[i] < keep ? x : 1 - x))
}

/** The unbiased estimate of the share of ones from randomised-response reports: (mean − (1 − p))/(2p − 1). */
export function randomisedResponseEstimate(reports: VectorLike, epsilon: number): number {
  const r = dense.toF64(reports, 'randomisedResponseEstimate')
  const p = randomisedResponseKeep(epsilon)
  const mean = r.reduce((a, c) => a + c, 0) / r.length
  return (mean - (1 - p)) / (2 * p - 1)
}
