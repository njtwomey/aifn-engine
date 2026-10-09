/**
 * Step-size schedules $t \mapsto \eta_t$ (the family's shared layer): for the first-order methods and update rules, and
 * for anything else that decays a step size over steps (stochastic-gradient Langevin dynamics, annealing). Each is a
 * `Schedule`, read at $t = 0, 1, 2, \dots$ and equal to $\eta_0$ at $t = 0$. Robbins & Monro (1951), "A stochastic
 * approximation method": $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ hold for `inverseTimeDecay`.
 */

import type { Scalar, Schedule } from 'aifn-compute/foundation/contracts'
import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'

/**
 * The schedule $\eta_t = \eta_0 / (1 + kt)$.
 *
 * @param initial The step size $\eta_0$ at $t = 0$.
 * @param k The decay rate $k$: the step size has halved at $t = 1/k$.
 * @returns The schedule.
 *
 * @example The step size halves after two steps when k is 0.5
 * const eta = inverseTimeDecay(0.1, 0.5)
 * print('steps 0 to 4:', [0, 1, 2, 3, 4].map(eta))
 */
export const inverseTimeDecay =
  (initial: Scalar, k: Scalar): Schedule =>
  (t) =>
    initial / (1 + k * t)

/**
 * The schedule $\eta_t = \eta_0\gamma^t$.
 *
 * @param initial The step size $\eta_0$ at $t = 0$.
 * @param gamma The factor $\gamma$ applied at each step, in $(0, 1)$ for a decay.
 * @returns The schedule.
 *
 * @example A factor of 0.9 a step
 * const eta = exponentialDecay(0.1, 0.9)
 * print('steps 0 to 4:', [0, 1, 2, 3, 4].map(eta))
 * print('step 50:', eta(50))
 */
export const exponentialDecay =
  (initial: Scalar, gamma: Scalar): Schedule =>
  (t) =>
    initial * gamma ** t

/**
 * The schedule $\eta_t = \eta_0 / \sqrt{t + 1}$.
 *
 * @param initial The step size $\eta_0$ at $t = 0$.
 * @returns The schedule.
 *
 * @example A quarter of the initial step after fifteen steps
 * const eta = inverseSqrtDecay(0.1)
 * print('steps 0, 3, 15, 99:', [0, 3, 15, 99].map(eta))
 */
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
