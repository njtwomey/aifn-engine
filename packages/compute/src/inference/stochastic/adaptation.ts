/**
 * Step-size adaptation for the Hamiltonian samplers by dual averaging (Nesterov, 2009; Hoffman & Gelman, 2014, §3.2,
 * Algorithms 5 and 6). During the first `warmup` steps the step size $\varepsilon$ is tuned so that the mean acceptance
 * statistic approaches a target $\delta$; after warmup $\varepsilon$ is frozen at the weighted average
 * $\bar\varepsilon$ of the iterates, as in Stan.
 *
 * With $a_m$ the acceptance statistic of step $m$ ($m = 1, 2, \dots$), clamped to $[0, 1]$,
 * $\mu = \log(10\varepsilon_0)$, and $\bar H_0 = 0$, $\log \bar\varepsilon_0 = 0$:
 *
 * - $\bar H_m = \big(1 - \frac{1}{m + t_0}\big) \bar H_{m-1} + \frac{\delta - a_m}{m + t_0}$
 * - $\log \varepsilon_m = \mu - \frac{\sqrt{m}}{\gamma} \bar H_m$
 * - $\log \bar\varepsilon_m = m^{-\kappa} \log \varepsilon_m + (1 - m^{-\kappa}) \log \bar\varepsilon_{m-1}$
 *
 * $\bar H$ is the running error of the acceptance statistic; $t_0$ damps the first iterations, $\gamma$ sets how far
 * $\log \varepsilon$ moves from $\mu$, and $\kappa \in (0.5, 1]$ makes the average forget the early iterates.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Options of dual-averaging step-size adaptation. Defaults are Stan's (and Hoffman & Gelman's, except $\delta$).
 */
export type DualAveragingOptions = {
  /**
   * Steps that adapt $\varepsilon$ (a non-negative integer; 0 adapts nothing); $\varepsilon$ is frozen at
   * $\bar\varepsilon$ from step `warmup + 1` on.
   */
  warmup: Size
  /** The target mean acceptance statistic $\delta \in (0, 1)$. Default 0.8 (Stan; Hoffman & Gelman use 0.65). */
  targetAcceptance?: number
  /** $\gamma > 0$, the scale of the shrinkage of $\log \varepsilon$ towards $\mu$. Default 0.05. */
  gamma?: number
  /** $t_0 \ge 0$, which damps the first iterations. Default 10. */
  t0?: number
  /** $\kappa \in (0.5, 1]$, the forgetting exponent of the average. Default 0.75. */
  kappa?: number
}

/** The adaptation fields of a Hamiltonian sampler's state. */
export type DualAveragingState = {
  /**
   * The step size the next step will use: $\varepsilon_m$ during warmup, $\bar\varepsilon$ after it (the fixed
   * $\varepsilon$ without adaptation).
   */
  nextStepSize: number
  /** $\bar\varepsilon_m$, the weighted average of the adapted step sizes (the final step size after warmup). */
  stepSizeBar: number
  /** $\bar H_m$, the running average of the error $\delta - a_m$ of the acceptance statistic. */
  hBar: number
  /** True while the next step still adapts. */
  adapting: boolean
}

/**
 * Checked adaptation options: `warmup`, `delta` ($\delta$), `gamma`, `t0`, `kappa` with their defaults filled in, and
 * `mu` ($\mu = \log(10\varepsilon_0)$).
 */
type Resolved = { warmup: Size; delta: number; gamma: number; t0: number; kappa: number; mu: number }

/**
 * Check the options and fix $\mu = \log(10\varepsilon_0)$. Throws `DomainError` for a step size that is not positive
 * and finite, or an option out of its range.
 *
 * @param options The adaptation options, or `undefined` for no adaptation.
 * @param stepSize The initial step size $\varepsilon_0$; checked even without adaptation.
 * @param name The caller's name for error messages.
 * @returns The options with defaults filled in and $\mu$, or `null` when `options` is left out.
 */
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

/**
 * The adaptation state before the first step.
 *
 * @param stepSize The initial step size $\varepsilon_0$, used by the first step.
 * @param adapt The checked options from `resolveDualAveraging`, or `null` for a fixed step size.
 * @returns The state with $\bar H_0 = 0$ and $\bar\varepsilon_0 = 1$ (or $\varepsilon_0$ without adaptation),
 *   adapting when `warmup` is positive.
 */
export function dualAveragingStart(stepSize: number, adapt: Resolved | null): DualAveragingState {
  return { nextStepSize: stepSize, stepSizeBar: adapt ? 1 : stepSize, hBar: 0, adapting: !!adapt && adapt.warmup > 0 }
}

/**
 * The adaptation fields after step $m$ (1-based) with acceptance statistic `acceptStat` in $[0, 1]$, read from `state`
 * (a sampler's whole state may be passed; only these fields are returned). Steps after warmup (and every step without
 * adaptation) leave them unchanged.
 *
 * @param state The adaptation fields after step $m - 1$; not modified.
 * @param m The 1-based index of the step just taken.
 * @param acceptStat The acceptance statistic $a_m$ of that step; clamped to $[0, 1]$, and taken as 0 when not finite
 *   (a divergent step).
 * @param adapt The checked options from `resolveDualAveraging`, or `null` for a fixed step size.
 * @returns New adaptation fields: `nextStepSize` is $\varepsilon_m$ while adapting and $\bar\varepsilon_m$ once
 *   step `warmup` is reached.
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
