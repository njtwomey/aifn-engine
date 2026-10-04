/**
 * Step-size schedules t ↦ η_t (the family's shared layer): for the first-order methods and update rules, and for
 * anything else that decays a step size over steps (stochastic-gradient Langevin dynamics, annealing). Robbins &
 * Monro (1951), "A stochastic approximation method": Σ η_t = ∞ and Σ η_t² < ∞ hold for `inverseTimeDecay`.
 */

import type { Scalar, Schedule } from 'aifn-compute/foundation/contracts'
import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'

/** The schedule η_t = η₀ / (1 + kt). */
export const inverseTimeDecay =
  (initial: Scalar, k: Scalar): Schedule =>
  (t) =>
    initial / (1 + k * t)

/** The schedule η_t = η₀γᵗ. */
export const exponentialDecay =
  (initial: Scalar, gamma: Scalar): Schedule =>
  (t) =>
    initial * gamma ** t

/** The schedule η_t = η₀ / √(t + 1). */
export const inverseSqrtDecay =
  (initial: Scalar): Schedule =>
  (t) =>
    initial / Math.sqrt(t + 1)

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const schedule = definer<FunctionInfo>('function', 'optim')
const LR = ['learning-rate-schedules']
schedule(
  { key: 'inverseTimeDecay', name: 'Inverse-time decay', tex: '\\eta_0 / (1 + k t)', role: 'construction', notes: LR },
  inverseTimeDecay,
)
schedule(
  { key: 'exponentialDecay', name: 'Exponential decay', tex: '\\eta_0 \\gamma^t', role: 'construction', notes: LR },
  exponentialDecay,
)
schedule(
  {
    key: 'inverseSqrtDecay',
    name: 'Inverse square-root decay',
    tex: '\\eta_0 / \\sqrt{1 + t}',
    role: 'construction',
    notes: LR,
  },
  inverseSqrtDecay,
)

/** The step-size schedules, keyed by name. */
export const scheduleFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', { inverseTimeDecay, exponentialDecay, inverseSqrtDecay }) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
