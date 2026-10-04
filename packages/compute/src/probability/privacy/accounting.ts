/**
 * Privacy accounting: the total privacy loss of a sequence of mechanisms.
 *
 * - `sequentialComposition`: k mechanisms (εᵢ, δᵢ)-DP compose to (Σεᵢ, Σδᵢ)-DP (Dwork et al., 2006).
 * - `advancedComposition` (Dwork, Rothblum and Vadhan, 2010): k-fold composition of an (ε, δ)-DP mechanism is
 *   (ε√(2k ln(1/δ′)) + kε(e^ε − 1), kδ + δ′)-DP for any δ′ > 0.
 * - Rényi DP (Mironov, 2017): RDP of order α composes by adding. `rdpSubsampledGaussian` is the RDP of the sampled
 *   Gaussian mechanism (Poisson sampling rate q, noise multiplier σ) of Mironov, Talwar and Zhang (2019), computed as
 *   Opacus and TensorFlow Privacy do (exact sums for integer α, the two-sided series for fractional α);
 *   `rdpToEpsilon` converts to (ε, δ) by Balle et al. (2020, Thm. 21), minimised over the orders.
 * - Zero-concentrated DP (Bun and Steinke, 2016): the Gaussian mechanism is ρ = Δ²/2σ² zCDP, ρ adds under composition,
 *   and ρ-zCDP implies (ρ + 2√(ρ ln(1/δ)), δ)-DP.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { normalLogCdf } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** (Σεᵢ, Σδᵢ) for mechanisms (εᵢ, δᵢ) applied in sequence. */
export function sequentialComposition(mechanisms: readonly { epsilon: number; delta?: number }[]): {
  epsilon: number
  delta: number
} {
  let epsilon = 0
  let delta = 0
  for (const m of mechanisms) {
    epsilon += m.epsilon
    delta += m.delta ?? 0
  }
  return { epsilon, delta }
}

/** The advanced composition bound for k uses of an (ε, δ)-DP mechanism, at slack δ′. */
export function advancedComposition(
  epsilon: number,
  delta: number,
  k: Size,
  slack: number,
): { epsilon: number; delta: number } {
  if (!(slack > 0 && slack < 1)) throw new DomainError('advancedComposition', 'advancedComposition: needs 0 < δ′ < 1')
  return {
    epsilon: epsilon * Math.sqrt(2 * k * Math.log(1 / slack)) + k * epsilon * Math.expm1(epsilon),
    delta: k * delta + slack,
  }
}

/** Opacus's default RDP orders: 1.1, 1.2, …, 10.9 and 12, 13, …, 63. */
export const DEFAULT_ORDERS: readonly number[] = [
  ...Array.from({ length: 99 }, (_, x) => 1 + (x + 1) / 10),
  ...Array.from({ length: 52 }, (_, x) => 12 + x),
]

function logAdd(a: number, b: number): number {
  const lo = Math.min(a, b)
  const hi = Math.max(a, b)
  if (lo === -Infinity) return hi
  return Math.log1p(Math.exp(lo - hi)) + hi
}

function logSub(a: number, b: number): number {
  if (a < b) throw new DomainError('logSub', 'logSub: negative result')
  if (b === -Infinity) return a
  if (a === b) return -Infinity
  const d = Math.expm1(a - b)
  return Number.isFinite(d) ? Math.log(d) + b : a
}

/** log C(α, i) for integer i ≥ 0 and real α, with the binomial's sign. */
function binomial(alpha: number, i: number): number {
  let c = 1
  for (let k = 1; k <= i; k++) c *= (alpha - k + 1) / k
  return c
}

const logErfc = (x: number) => Math.log(2) + (normalLogCdf(-x * Math.SQRT2) as number)

function logAInt(q: number, sigma: number, alpha: number): number {
  let logA = -Infinity
  for (let i = 0; i <= alpha; i++) {
    const coef = Math.log(binomial(alpha, i)) + i * Math.log(q) + (alpha - i) * Math.log(1 - q)
    logA = logAdd(logA, coef + (i * i - i) / (2 * sigma * sigma))
  }
  return logA
}

function logAFrac(q: number, sigma: number, alpha: number): number {
  let a0 = -Infinity
  let a1 = -Infinity
  const z0 = sigma * sigma * Math.log(1 / q - 1) + 0.5
  for (let i = 0; ; i++) {
    const coef = binomial(alpha, i)
    const logCoef = Math.log(Math.abs(coef))
    const j = alpha - i
    const t0 = logCoef + i * Math.log(q) + j * Math.log(1 - q)
    const t1 = logCoef + j * Math.log(q) + i * Math.log(1 - q)
    const e0 = Math.log(0.5) + logErfc((i - z0) / (Math.SQRT2 * sigma))
    const e1 = Math.log(0.5) + logErfc((z0 - j) / (Math.SQRT2 * sigma))
    const s0 = t0 + (i * i - i) / (2 * sigma * sigma) + e0
    const s1 = t1 + (j * j - j) / (2 * sigma * sigma) + e1
    if (coef > 0) {
      a0 = logAdd(a0, s0)
      a1 = logAdd(a1, s1)
    } else {
      a0 = logSub(a0, s0)
      a1 = logSub(a1, s1)
    }
    if (Math.max(s0, s1) < -30) break
  }
  return logAdd(a0, a1)
}

/**
 * RDP ε(α) of `steps` compositions of the sampled Gaussian mechanism with sampling rate q and noise multiplier σ (the
 * noise's standard deviation over the L2 sensitivity), at each order α > 1.
 */
export function rdpSubsampledGaussian(
  q: number,
  noiseMultiplier: number,
  steps: Size,
  orders: readonly number[] = DEFAULT_ORDERS,
): Float64Array {
  if (!(q >= 0 && q <= 1)) throw new DomainError('rdpSubsampledGaussian', 'rdpSubsampledGaussian: needs 0 ≤ q ≤ 1')
  return Float64Array.from(orders, (alpha) => {
    if (!(alpha > 1)) throw new DomainError('rdpSubsampledGaussian', 'rdpSubsampledGaussian: orders must exceed 1')
    let rdp: number
    if (q === 0) rdp = 0
    else if (noiseMultiplier === 0) rdp = Infinity
    else if (q === 1) rdp = alpha / (2 * noiseMultiplier ** 2)
    else if (!Number.isFinite(alpha)) rdp = Infinity
    else {
      const logA = Number.isInteger(alpha) ? logAInt(q, noiseMultiplier, alpha) : logAFrac(q, noiseMultiplier, alpha)
      rdp = logA / (alpha - 1)
    }
    return rdp * steps
  })
}

/**
 * The (ε, δ) guarantee implied by RDP values at several orders (Balle et al., 2020, Thm. 21):
 * ε = min_α ε(α) − (log δ + log α)/(α − 1) + log((α − 1)/α), with the order that attains it.
 */
export function rdpToEpsilon(
  rdp: ArrayLike<number>,
  delta: number,
  orders: readonly number[] = DEFAULT_ORDERS,
): { epsilon: number; order: number } {
  if (rdp.length !== orders.length) throw new ShapeError('rdpToEpsilon', 'rdpToEpsilon: one RDP value per order')
  if (!(delta > 0 && delta < 1)) throw new DomainError('rdpToEpsilon', 'rdpToEpsilon: needs 0 < δ < 1')
  let best = Infinity
  let order = NaN
  for (let k = 0; k < orders.length; k++) {
    const a = orders[k]
    const e = rdp[k] - (Math.log(delta) + Math.log(a)) / (a - 1) + Math.log((a - 1) / a)
    if (e < best) {
      best = e
      order = a
    }
  }
  return { epsilon: best, order }
}

/**
 * ε at δ after `steps` of DP-SGD with Poisson sampling rate q and noise multiplier σ, by RDP accounting
 * (`rdpSubsampledGaussian` then `rdpToEpsilon`), as Opacus's `RDPAccountant`.
 */
export function dpSgdEpsilon(
  q: number,
  noiseMultiplier: number,
  steps: Size,
  delta: number,
  orders: readonly number[] = DEFAULT_ORDERS,
): number {
  return rdpToEpsilon(rdpSubsampledGaussian(q, noiseMultiplier, steps, orders), delta, orders).epsilon
}

/** The zCDP ρ = Δ²/(2σ²) of the Gaussian mechanism with L2 sensitivity Δ and noise σ. */
export const gaussianZcdp = (sensitivity: number, sigma: number): number => sensitivity ** 2 / (2 * sigma ** 2)

/** The (ε, δ)-DP implied by ρ-zCDP: ε = ρ + 2√(ρ ln(1/δ)) (Bun and Steinke, 2016, Prop. 1.3). */
export function zcdpToEpsilon(rho: number, delta: number): number {
  if (!(delta > 0 && delta < 1)) throw new DomainError('zcdpToEpsilon', 'zcdpToEpsilon: needs 0 < δ < 1')
  return rho + 2 * Math.sqrt(rho * Math.log(1 / delta))
}
