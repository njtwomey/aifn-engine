/**
 * The training-run row every `GymTrainer` page shares: the budget (in episodes or in environment steps), its size and
 * the seed. A page spreads `run: trainingRun(defaults)` into its `useFigureState`, so the fields live in the figure's
 * state and URL like any other; `GymTrainer` reads them from `state.run`. Pages declare no budget or seed fields.
 */
import type { ReactNode } from 'react'
import { choice, int, row, when } from '@render/state'

/** A page's default budget: a number of episodes or of environment steps, and the seed (default 1). */
export type TrainingRunDefaults = ({ episodes: number } | { steps: number }) & {
  seed?: number
  /** What an episode is called on this page, e.g. 'rounds' for a bandit (default 'episodes'). */
  unit?: string
  /** The row's label (default 'training run'). */
  label?: ReactNode
}

/** The values of the training-run row. */
export type TrainingRunValues = { budget: 'episodes' | 'steps'; episodes: number; steps: number; seed: number }

/** The training-run row, with the page's defaults (module docs). */
export function trainingRun(defaults: TrainingRunDefaults) {
  const unit = defaults.unit ?? 'episodes'
  const byEpisodes = 'episodes' in defaults
  return row(defaults.label ?? 'training run', {
    budget: choice(
      [
        { value: 'episodes', label: unit },
        { value: 'steps', label: 'environment steps' },
      ],
      byEpisodes ? 'episodes' : 'steps',
      { label: 'budget in' },
    ),
    episodes: int(byEpisodes ? defaults.episodes : 300, {
      label: unit,
      ge: 1,
      le: 100_000,
      suggestions: [100, 300, 1000, 3000],
      when: when('budget', 'episodes'),
    }),
    steps: int(byEpisodes ? 50_000 : defaults.steps, {
      label: 'steps',
      ge: 1,
      le: 10_000_000,
      suggestions: [10_000, 25_000, 50_000, 100_000],
      when: when('budget', 'steps'),
    }),
    seed: int(defaults.seed ?? 1, { label: 'seed', ge: 0, le: 9999 }),
  })
}

/** The training run in `state.run`, as `train`'s budget and seed. Throws when the page did not add the row. */
export function trainingRunOf(state: unknown): { episodes?: number; steps?: number; seed: number } {
  const run = (state as { run?: Partial<TrainingRunValues> } | undefined)?.run
  if (!run || run.seed === undefined || !run.budget)
    throw new Error('GymTrainer: the figure state needs `run: trainingRun(defaults)` (aifn-render/gym trainingRun)')
  return run.budget === 'steps' ? { steps: run.steps, seed: run.seed } : { episodes: run.episodes, seed: run.seed }
}
