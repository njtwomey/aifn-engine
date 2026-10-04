/**
 * Per-step series for `GymTrainer`: what to plot against step under the Player for the chosen episode. The state
 * series are the environment's own (`render.series`); the action series follow from the action domain (a discrete
 * action is a strip of its names, a box action a line per element).
 */
import type { Environment, StateSeries } from 'aifn-compute/foundation/contracts'

const NONE: readonly StateSeries<unknown>[] = []

/** The state series an environment declares on its render spec (none without one). */
export const stateSeries = (env: Environment<unknown, unknown, unknown>): readonly StateSeries<unknown>[] =>
  env.render?.series ?? NONE

/** The action series of an environment: a strip for a discrete action, else one line per element of a box. */
export type ActionSeries =
  { kind: 'strip'; name: string; names: readonly string[] } | { kind: 'line'; name: string; index: number }

export function actionSeries(env: Environment<unknown, unknown, unknown>): ActionSeries[] {
  const a = env.action
  if (a.kind === 'discrete')
    return [{ kind: 'strip', name: 'action', names: a.names ?? Array.from({ length: a.n }, (_, i) => `${i}`) }]
  return a.low.map((_, i) => ({ kind: 'line', name: a.names?.[i] ?? `action ${i}`, index: i }))
}

/** Whether `GymTrainer` draws per-step series for this environment: it declares state series and episodes last. */
export const hasStepSeries = (env: Environment<unknown, unknown, unknown>): boolean =>
  stateSeries(env).length > 0 && env.horizon > 1
