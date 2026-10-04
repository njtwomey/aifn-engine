/**
 * Step-size adaptation for the Hamiltonian samplers by dual averaging (Nesterov, 2009; Hoffman & Gelman, 2014, §3.2,
 * Algorithms 5 and 6). During the first `warmup` steps the step size ε is tuned so that the mean acceptance statistic
 * approaches a target δ; after warmup ε is frozen at the weighted average ε̄ of the iterates, as in Stan.
 *
 * With aₘ the acceptance statistic of step m (m = 1, 2, …), μ = log(10ε₀), and H̄₀ = 0, log ε̄₀ = 0:
 *
 *   H̄ₘ = (1 − 1/(m + t₀)) H̄ₘ₋₁ + (δ − aₘ)/(m + t₀)
 *   log εₘ = μ − (√m/γ) H̄ₘ
 *   log ε̄ₘ = m^(−κ) log εₘ + (1 − m^(−κ)) log ε̄ₘ₋₁
 *
 * H̄ is the running error of the acceptance statistic; t₀ damps the first iterations, γ sets how far log ε moves from μ,
 * and κ ∈ (0.5, 1] makes the average forget the early iterates.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of dual-averaging step-size adaptation. Defaults are Stan's (and Hoffman & Gelman's, except δ). */
export type DualAveragingOptions = {
  /** Steps that adapt ε; ε is frozen at ε̄ from step `warmup + 1` on. */
  warmup: Size
  /** The target mean acceptance statistic δ in (0, 1). Default 0.8 (Stan; Hoffman & Gelman use 0.65). */
  targetAcceptance?: number
  /** γ > 0, the scale of the shrinkage of log ε towards μ. Default 0.05. */
  gamma?: number
  /** t₀ ≥ 0, which damps the first iterations. Default 10. */
  t0?: number
  /** κ in (0.5, 1], the forgetting exponent of the average. Default 0.75. */
  kappa?: number
}

/** The adaptation fields of a Hamiltonian sampler's state. */
export type DualAveragingState = {
  /** The step size the next step will use: εₘ during warmup, ε̄ after it (the fixed ε without adaptation). */
  nextStepSize: number
  /** ε̄ₘ, the weighted average of the adapted step sizes (the final step size after warmup). */
  stepSizeBar: number
  /** H̄ₘ, the running error δ − aₘ of the acceptance statistic. */
  hBar: number
  /** True while the next step still adapts. */
  adapting: boolean
}

type Resolved = { warmup: Size; delta: number; gamma: number; t0: number; kappa: number; mu: number }

/** Check the options and fix μ = log(10ε₀). */
export function resolveDualAveraging(
  options: DualAveragingOptions | undefined,
  stepSize: number,
  name: string,
): Resolved | null {
  if (!(stepSize > 0 && Number.isFinite(stepSize)))
    throw new DomainError(name, `${name}: the step size must be positive and finite, got ${stepSize}`)
  if (!options) return null
  const { warmup, targetAcceptance: delta = 0.8, gamma = 0.05, t0 = 10, kappa = 0.75 } = options
  if (!(Number.isInteger(warmup) && warmup >= 0))
    throw new DomainError(name, `${name}: adapt.warmup must be a non-negative integer, got ${warmup}`)
  if (!(delta > 0 && delta < 1))
    throw new DomainError(name, `${name}: adapt.targetAcceptance must lie in (0, 1), got ${delta}`)
  if (!(gamma > 0)) throw new DomainError(name, `${name}: adapt.gamma must be positive, got ${gamma}`)
  if (!(t0 >= 0)) throw new DomainError(name, `${name}: adapt.t0 must be non-negative, got ${t0}`)
  if (!(kappa > 0.5 && kappa <= 1))
    throw new DomainError(name, `${name}: adapt.kappa must lie in (0.5, 1], got ${kappa}`)
  return { warmup, delta, gamma, t0, kappa, mu: Math.log(10 * stepSize) }
}

/** The adaptation state before the first step. */
export function dualAveragingStart(stepSize: number, adapt: Resolved | null): DualAveragingState {
  return { nextStepSize: stepSize, stepSizeBar: adapt ? 1 : stepSize, hBar: 0, adapting: !!adapt && adapt.warmup > 0 }
}

/**
 * The adaptation fields after step m (1-based) with acceptance statistic `acceptStat` in [0, 1], read from `state`
 * (a sampler's whole state may be passed; only these fields are returned). Steps after warmup (and every step without
 * adaptation) leave them unchanged.
 */
export function dualAveragingUpdate(
  state: DualAveragingState,
  m: Size,
  acceptStat: number,
  adapt: Resolved | null,
): DualAveragingState {
  if (!adapt || m > adapt.warmup) {
    const { nextStepSize, stepSizeBar, hBar, adapting } = state
    return { nextStepSize, stepSizeBar, hBar, adapting }
  }
  const a = Number.isFinite(acceptStat) ? Math.min(1, Math.max(0, acceptStat)) : 0
  const w = 1 / (m + adapt.t0)
  const hBar = (1 - w) * state.hBar + w * (adapt.delta - a)
  const logStep = adapt.mu - (Math.sqrt(m) / adapt.gamma) * hBar
  const eta = m ** -adapt.kappa
  const stepSizeBar = Math.exp(eta * logStep + (1 - eta) * Math.log(state.stepSizeBar))
  const adapting = m < adapt.warmup
  return { nextStepSize: adapting ? Math.exp(logStep) : stepSizeBar, stepSizeBar, hBar, adapting }
}
