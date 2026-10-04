/**
 * Headless training (docs/aifn-gym.md §4b): `train` runs an agent for a number of episodes with no display and returns
 * plain data, per-episode summaries (return, length, terminated, success or failure, pseudo-regret, first action and
 * the agent's own `scalars`) and agent-state checkpoints; `training` is the same run as a generator that yields partial
 * results, so a worker can stream progress. Episode e (1-based) runs on `child(stream(seed), 'step', e − 1)` from an
 * agent initialised on `child(stream(seed), 'init')`, exactly as a trace of `episodes(env, agent)` on `stream(seed)`
 * would, so any training episode can be re-run from the nearest checkpoint: `replay` gives episode e as it happened
 * (with exploration), `agentAfter` the agent's state after e episodes, and `evaluateEpisode` a fresh greedy episode of
 * that agent on an evaluation seed.
 */

import type { Agent, Checkpoint, Environment, Training, Trajectory } from 'aifn-compute/foundation/contracts'
import { child, stream, type Stream } from 'aifn-compute/foundation/random'
import { runEpisode } from './rollout'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Options for `train` and `training`. The budget is `episodes`, or `steps`: run episodes until at least this many
 * environment steps are done. The episode that crosses the budget runs to its end (it is not cut short), so a run
 * may overshoot by up to one episode and every episode stays replayable as it happened.
 */
export interface TrainOptions {
  /** Episodes to run. */
  episodes?: number
  /** Environment steps to run instead (see above). */
  steps?: number
  /** The root seed. Default 0. */
  seed?: number | string
  /** Least spacing of the agent-state checkpoints, in episodes. Default 1 (widened to keep `maxCheckpoints`). */
  checkpointEvery?: number
  /** At most this many checkpoints besides the initial state. Default 200. */
  maxCheckpoints?: number
  /**
   * `training` yields after every this many episodes. Default about a twentieth of the run (with a step budget, after
   * every twentieth of the steps).
   */
  chunk?: number
}

/** The checkpoint spacing for a run: at least `checkpointEvery`, wide enough for at most `maxCheckpoints`. */
export function checkpointSpacing(episodes: number, checkpointEvery = 1, maxCheckpoints = 200): number {
  return Math.max(1, Math.round(checkpointEvery), Math.ceil(episodes / Math.max(1, maxCheckpoints)))
}

const root = (seed: number | string): Stream => stream(seed)

/**
 * Training as a generator: yields the run so far after every `chunk` episodes and at the end (`done: true`). Each
 * yielded value is a fresh snapshot (its arrays are copies), safe to post from a worker.
 */
export function* training<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  options: TrainOptions,
): Generator<Training<G>, Training<G>> {
  const { seed = 0, checkpointEvery, maxCheckpoints = 200 } = options
  const budget = options.steps ?? NaN
  const total = Number.isNaN(budget) ? (options.episodes ?? NaN) : NaN
  if (!(total >= 0) && !(budget >= 0)) throw new DomainError('training', 'training: give a budget, episodes or steps')
  // Under an episode budget the spacing is fixed; under a step budget it doubles as the checkpoints fill up.
  let every = checkpointSpacing(Number.isNaN(total) ? 0 : total, checkpointEvery, maxCheckpoints)
  const chunk = Math.max(1, Math.round(options.chunk ?? Math.ceil((Number.isNaN(total) ? 0 : total) / 20)))
  const stepChunk = Math.max(1, Math.ceil(budget / 20))
  const r = root(seed)
  let g = agent.init(env, child(child(r, 'init'), 'agent'))
  const returns: number[] = []
  const lengths: number[] = []
  const terminated: number[] = []
  const hasRegret = !!(env.oracle?.expectedReward && env.oracle.bestExpectedReward)
  const regret: number[] = []
  const outcome: number[] = []
  const firstAction: number[] = []
  const scalars: Record<string, number[]> = {}
  let checkpoints: Checkpoint<G>[] = [{ episode: 0, agent: g }]
  let steps = 0
  const snapshot = (done: boolean): Training<G> => ({
    seed,
    total,
    episodes: returns.length,
    budget,
    steps,
    every,
    returns: Float64Array.from(returns),
    lengths: Float64Array.from(lengths),
    terminated: Uint8Array.from(terminated),
    regret: hasRegret ? Float64Array.from(regret) : null,
    outcome: Int8Array.from(outcome),
    firstAction: Float64Array.from(firstAction),
    scalars: Object.fromEntries(
      Object.entries(scalars).map(([k, v]) => [
        k,
        Float64Array.from({ length: returns.length }, (_, i) => v[i] ?? NaN),
      ]),
    ),
    checkpoints: checkpoints.slice(),
    final: g,
    done,
  })
  const finished = () => (Number.isNaN(budget) ? returns.length >= total : steps >= budget)
  let nextYield = Number.isNaN(budget) ? chunk : stepChunk
  for (let e = 0; !finished(); e++) {
    const run = runEpisode(env, agent, g, child(r, 'step', e))
    g = run.agent
    const tr = run.trajectory
    returns.push(tr.episodeReturn)
    lengths.push(tr.actions.length)
    steps += tr.actions.length
    terminated.push(tr.reachedTerminal ? 1 : 0)
    if (hasRegret) regret.push(tr.regret)
    outcome.push(tr.ending ? (tr.ending.success ? 1 : -1) : 0)
    firstAction.push(typeof tr.actions[0] === 'number' ? tr.actions[0] : NaN)
    for (const [k, v] of Object.entries(agent.scalars?.(g) ?? {})) (scalars[k] ??= [])[e] = v
    if ((e + 1) % every === 0) {
      checkpoints.push({ episode: e + 1, agent: g })
      // A step budget does not know its episodes in advance: keep at most `maxCheckpoints` by thinning to every other.
      if (checkpoints.length > maxCheckpoints + 1) {
        every *= 2
        checkpoints = checkpoints.filter((c) => c.episode % every === 0)
      }
    }
    const progress = Number.isNaN(budget) ? e + 1 : steps
    if (!finished() && progress >= nextYield) {
      nextYield = progress + (Number.isNaN(budget) ? chunk : stepChunk)
      yield snapshot(false)
    }
  }
  const last = snapshot(true)
  yield last
  return last
}

/** Train with no display (see `training`): the complete run. */
export function train<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  options: TrainOptions,
): Training<G> {
  let last: Training<G> | undefined
  for (const t of training(env, agent, options)) last = t
  return last!
}

/** Re-run training episodes from the nearest checkpoint at or before `from`, up to episode `to`; calls `each`. */
function rerun<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  t: Pick<Training<G>, 'seed' | 'checkpoints'>,
  to: number,
  each?: (e: number, run: ReturnType<typeof runEpisode<S, O, A, G>>) => void,
): G {
  let c = t.checkpoints[0]
  for (const k of t.checkpoints) if (k.episode <= to && k.episode >= c.episode) c = k
  const r = root(t.seed)
  let g = c.agent
  for (let e = c.episode; e < to; e++) {
    const run = runEpisode(env, agent, g, child(r, 'step', e))
    each?.(e + 1, run)
    g = run.agent
  }
  return g
}

/** The agent's state after `e` episodes of a training run (re-run from the nearest checkpoint). */
export function agentAfter<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  t: Pick<Training<G>, 'seed' | 'checkpoints'>,
  e: number,
): G {
  return rerun(env, agent, t, e)
}

/** Training episode `e` (1-based) exactly as it happened, with exploration, and the agent's state before it. */
export function replay<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  t: Pick<Training<G>, 'seed' | 'checkpoints'>,
  e: number,
): { trajectory: Trajectory<S, O, A>; agent: G } {
  if (!(e >= 1)) throw new DomainError('replay', `replay: episodes are numbered from 1, got ${e}`)
  const before = rerun(env, agent, t, e - 1)
  const run = runEpisode(env, agent, before, child(root(t.seed), 'step', e - 1))
  return { trajectory: run.trajectory, agent: before }
}

/**
 * A fresh episode of the policy learnt after `e` episodes, with no exploration and no learning (the agent's `greedy`
 * action, or its `act` when it has none), on `child(stream(evaluationSeed), 'evaluate')`.
 */
export function evaluateEpisode<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  t: Pick<Training<G>, 'seed' | 'checkpoints'>,
  e: number,
  evaluationSeed: number | string = 'evaluate',
): { trajectory: Trajectory<S, O, A>; agent: G } {
  const g = rerun(env, agent, t, e)
  const run = runEpisode(env, agent, g, child(root(evaluationSeed), 'evaluate'), 'greedy')
  return { trajectory: run.trajectory, agent: g }
}
