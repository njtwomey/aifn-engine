/**
 * Coordinate-ascent VI for the univariate Gaussian with unknown mean μ and precision τ (Bishop, 2006, §10.1.3):
 * the model xₙ ~ N(μ, τ⁻¹), μ | τ ~ N(μ₀, (λ₀τ)⁻¹), τ ~ Gamma(a₀, b₀) (rate b₀), and the mean-field family
 * q(μ)q(τ) = N(μ | μ_N, λ_N⁻¹) Gamma(τ | a_N, b_N). The exact posterior is normal-gamma, so the result can be compared
 * with the truth: the factorised q gets the marginal means right and under-states the spread of μ.
 */

import { digamma, logGamma } from 'aifn-compute/numerics/special'
import type { Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { dense } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'

const toF64 = (x: VectorLike) => dense.toF64(x, 'caviNormalGamma')

const LOG_2PI = Math.log(2 * Math.PI)
const dg = (x: number) => digamma(x) as number
const lg = (x: number) => logGamma(x) as number

/** The normal-gamma prior: μ | τ ~ N(mean, (precisionScale τ)⁻¹), τ ~ Gamma(shape, rate). */
export type NormalGammaPrior = { mean: number; precisionScale: number; shape: number; rate: number }

/** Default prior μ₀ = 0, λ₀ = 1, a₀ = 1, b₀ = 1. */
export const defaultNormalGammaPrior: NormalGammaPrior = { mean: 0, precisionScale: 1, shape: 1, rate: 1 }

/** The exact posterior (normal-gamma) and the log evidence log p(x). */
export type NormalGammaPosterior = NormalGammaPrior & {
  logEvidence: number
  /** Marginal posterior mean and variance of μ (Student t with 2a degrees of freedom) and of τ. */
  meanOfMu: number
  varianceOfMu: number
  meanOfTau: number
  varianceOfTau: number
}

/**
 * The exact normal-gamma posterior (Murphy, 2007, §3): λ_N = λ₀ + N, μ_N = (λ₀μ₀ + Nx̄)/λ_N, a_N = a₀ + N/2,
 * b_N = b₀ + ½Σ(xₙ − x̄)² + λ₀N(x̄ − μ₀)²/(2λ_N), and log p(x) = log Γ(a_N) − log Γ(a₀) + a₀ log b₀ − a_N log b_N +
 * ½ log(λ₀/λ_N) − (N/2) log 2π.
 */
export function normalGammaPosterior(
  x: VectorLike,
  prior: NormalGammaPrior = defaultNormalGammaPrior,
): NormalGammaPosterior {
  const xs = toF64(x)
  const N = xs.length
  const xbar = xs.reduce((a, b) => a + b, 0) / N
  const ss = xs.reduce((a, b) => a + (b - xbar) ** 2, 0)
  const precisionScale = prior.precisionScale + N
  const mean = (prior.precisionScale * prior.mean + N * xbar) / precisionScale
  const shape = prior.shape + N / 2
  const rate = prior.rate + ss / 2 + (prior.precisionScale * N * (xbar - prior.mean) ** 2) / (2 * precisionScale)
  const logEvidence =
    lg(shape) -
    lg(prior.shape) +
    prior.shape * Math.log(prior.rate) -
    shape * Math.log(rate) +
    0.5 * Math.log(prior.precisionScale / precisionScale) -
    (N / 2) * LOG_2PI
  return {
    mean,
    precisionScale,
    shape,
    rate,
    logEvidence,
    meanOfMu: mean,
    varianceOfMu: shape > 1 ? rate / (precisionScale * (shape - 1)) : Infinity,
    meanOfTau: shape / rate,
    varianceOfTau: shape / (rate * rate),
  }
}

/** The state of `caviNormalGamma`. */
export type CaviNormalGammaState = Status & {
  /** Coordinate sweeps done. */
  t: number
  /** q(μ) = N(muMean, 1/muPrecision). */
  muMean: number
  muPrecision: number
  /** q(τ) = Gamma(tauShape, tauRate); E[τ] = tauShape/tauRate. */
  tauShape: number
  tauRate: number
  expectedTau: number
  /** The half step after updating q(μ) and before q(τ): (E[μ], E[τ]), for drawing the coordinate zig-zag. */
  halfStep: { muMean: number; expectedTau: number }
  elbo: number
  /** ELBO(t) − ELBO(t − 1) (never negative for coordinate ascent; NaN at t = 0). */
  elboChange: number
  /** log p(x) − ELBO = KL(q ‖ p(μ, τ | x)), exact here. */
  kl: number
  converged: boolean
  diverged: boolean
}

/**
 * CAVI for the Gaussian with unknown mean and precision (Bishop, 2006, eqs. 10.26–10.30): alternately
 * q(μ) ← N(μ_N, (λ₀ + N)E[τ]) with μ_N = (λ₀μ₀ + Nx̄)/(λ₀ + N), and q(τ) ← Gamma(a₀ + (N + 1)/2,
 * b₀ + ½E_μ[Σ(xₙ − μ)² + λ₀(μ − μ₀)²]). Each update maximises the ELBO in its factor, so the ELBO never decreases.
 * `init` takes a starting E[τ] (default 1) and E[μ] (default the prior mean); converged when the ELBO changes by less
 * than `tolerance` (default 1e-10).
 */
export function caviNormalGamma(
  x: VectorLike,
  prior: NormalGammaPrior = defaultNormalGammaPrior,
  options: { tolerance?: number } = {},
): Algorithm<{ expectedTau0?: number; muMean0?: number }, CaviNormalGammaState> {
  const xs = toF64(x)
  const N = xs.length
  const xbar = xs.reduce((a, b) => a + b, 0) / N
  const sumSq = xs.reduce((a, b) => a + b * b, 0)
  const { mean: m0, precisionScale: l0, shape: a0, rate: b0 } = prior
  const tolerance = options.tolerance ?? 1e-10
  const logEvidence = normalGammaPosterior(xs, prior).logEvidence
  const aN = a0 + (N + 1) / 2
  const muMeanOpt = (l0 * m0 + N * xbar) / (l0 + N)

  // ELBO = E[log p(x, μ, τ)] + H[q(μ)] + H[q(τ)] (Bishop, 2006, Exercise 10.7).
  const elboOf = (mN: number, lN: number, a: number, b: number) => {
    const eTau = a / b
    const eLogTau = dg(a) - Math.log(b)
    const eMu2 = mN * mN + 1 / lN
    const sumSqDev = sumSq - 2 * N * xbar * mN + N * eMu2
    const lik = (N / 2) * (eLogTau - LOG_2PI) - (eTau / 2) * sumSqDev
    const priorMu = 0.5 * (Math.log(l0) + eLogTau - LOG_2PI) - ((l0 * eTau) / 2) * ((mN - m0) ** 2 + 1 / lN)
    const priorTau = a0 * Math.log(b0) - lg(a0) + (a0 - 1) * eLogTau - b0 * eTau
    const hMu = 0.5 * (1 + LOG_2PI - Math.log(lN))
    const hTau = a - Math.log(b) + lg(a) + (1 - a) * dg(a)
    return lik + priorMu + priorTau + hMu + hTau
  }
  const rateFor = (mN: number, lN: number) => {
    const eMu2 = mN * mN + 1 / lN
    return b0 + 0.5 * (sumSq - 2 * N * xbar * mN + N * eMu2 + l0 * (eMu2 - 2 * m0 * mN + m0 * m0))
  }
  const state = (
    t: number,
    mN: number,
    lN: number,
    b: number,
    half: CaviNormalGammaState['halfStep'],
    previous: number,
  ) => {
    const value = elboOf(mN, lN, aN, b)
    const change = value - previous
    return {
      t,
      muMean: mN,
      muPrecision: lN,
      tauShape: aN,
      tauRate: b,
      expectedTau: aN / b,
      halfStep: half,
      elbo: value,
      elboChange: change,
      kl: logEvidence - value,
      converged: Math.abs(change) < tolerance,
      diverged: !Number.isFinite(value),
    }
  }
  return {
    name: 'cavi-normal-gamma',
    init: ({ expectedTau0 = 1, muMean0 = m0 } = {}) =>
      state(
        0,
        muMean0,
        (l0 + N) * expectedTau0,
        aN / expectedTau0,
        { muMean: muMean0, expectedTau: expectedTau0 },
        NaN,
      ),
    step: (s) => {
      const lN = (l0 + N) * s.expectedTau
      const b = rateFor(muMeanOpt, lN)
      return state(s.t + 1, muMeanOpt, lN, b, { muMean: muMeanOpt, expectedTau: s.expectedTau }, s.elbo)
    },
  }
}
