/**
 * Headless training (docs/aifn-gym.md §4b): `train` runs an agent for a number of episodes with no display and returns
 * plain data, per-episode summaries (return, length, terminated, success or failure, pseudo-regret, first action and
 * the agent's own `scalars`) and agent-state checkpoints; `training` is the same run as a generator that yields partial
 * results, so a worker can stream progress. Episode $e$ (from 1) runs on `child(stream(seed), 'step', e - 1)` from an
 * agent initialised on `child(child(stream(seed), 'init'), 'agent')`, exactly as a trace of `episodes(env, agent)` on
 * `stream(seed)` would, so any training episode can be re-run from the nearest checkpoint: `replay` gives episode $e$
 * as it happened (with exploration), `agentAfter` the agent's state after $e$ episodes, and `evaluateEpisode` a fresh
 * greedy episode of that agent on an evaluation seed.
 */

import type { Agent, Checkpoint, Environment, Training, Trajectory } from 'aifn-compute/foundation/contracts'
import { child, stream, type Stream } from 'aifn-compute/foundation/random'
import { runEpisode } from './rollout'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Options for `train` and `training`. The budget is `episodes`, or `steps`: run episodes until at least this many
 * environment steps are done (`steps` wins when both are given). The episode that crosses the budget runs to its end
 * (it is not cut short), so a run may overshoot by up to one episode and every episode stays replayable as it
 * happened.
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

/**
 * The checkpoint spacing for a run: at least `checkpointEvery`, wide enough for at most `maxCheckpoints`.
 *
 * @param episodes The episodes the run will take (0 when unknown, under a step budget).
 * @param checkpointEvery The least spacing wanted, in episodes; rounded to an integer.
 * @param maxCheckpoints The most checkpoints wanted besides the initial state.
 * @returns The spacing in episodes, at least 1.
 *
 * @example Spacing widens for long runs
 * print('100 episodes', checkpointSpacing(100), '; 1000 episodes', checkpointSpacing(1000))
 * print('1000 episodes, every 10 at least', checkpointSpacing(1000, 10))
 */
export function checkpointSpacing(episodes: number, checkpointEvery = 1, maxCheckpoints = 200): number {
  return Math.max(1, Math.round(checkpointEvery), Math.ceil(episodes / Math.max(1, maxCheckpoints)))
}

/**
 * The root stream of a run.
 *
 * @param seed The run's seed.
 * @returns `stream(seed)`.
 */
const root = (seed: number | string): Stream => stream(seed)

/**
 * Training as a generator: yields the run so far after every `chunk` episodes (under a step budget, every twentieth of
 * the steps) and at the end (`done: true`), which it also returns. Each yielded value is a fresh snapshot (its arrays
 * are copies), safe to post from a worker. Throws `DomainError` when neither budget is given.
 *
 * @param env The environment.
 * @param agent The agent, initialised on the run's init stream and learning as it acts.
 * @param options The budget, the seed, the checkpoints and the chunk.
 * @returns The generator of snapshots.
 *
 * @example Progress in chunks of two episodes
 * const env = gymEnvironment('gridworldEnvironment', { stepReward: -0.04 })
 * const counter = {
 *   name: 'up, then right; counts its steps',
 *   init: () => 0,
 *   act: (g, o) => ({ action: o === 0 || o === 4 ? 0 : 1 }),
 *   learn: (g) => g + 1,
 * }
 * const progress = [...training(env, counter, { episodes: 6, chunk: 2 })]
 * print('episodes at each yield', progress.map((t) => t.episodes))
 * print('done at each yield', progress.map((t) => t.done))
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

/**
 * Train with no display (see `training`): the complete run.
 *
 * @param env The environment.
 * @param agent The agent.
 * @param options The budget, the seed and the checkpoints.
 * @returns The finished run, `done: true`.
 *
 * @example Five episodes, with an agent whose state is its step count
 * const env = gymEnvironment('gridworldEnvironment', { stepReward: -0.04 })
 * const counter = {
 *   name: 'up, then right; counts its steps',
 *   init: () => 0,
 *   act: (g, o) => ({ action: o === 0 || o === 4 ? 0 : 1 }),
 *   learn: (g) => g + 1,
 * }
 * const t = train(env, counter, { episodes: 5 })
 * print('returns', t.returns)
 * print('lengths', t.lengths, '; steps', t.steps, '; final agent', t.final)
 * print('outcomes', t.outcome)
 */
export function train<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  options: TrainOptions,
): Training<G> {
  let last: Training<G> | undefined
  for (const t of training(env, agent, options)) last = t
  return last!
}

/**
 * Re-run training episodes from the nearest checkpoint at or before episode `to`, up to episode `to`, calling `each`
 * after every episode re-run.
 *
 * @param env The environment of the run.
 * @param agent The agent of the run.
 * @param t The run's seed and checkpoints.
 * @param to The number of episodes after which to stop.
 * @param each Called with each re-run episode's number (from 1) and its result.
 * @returns The agent's state after `to` episodes.
 */
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

/**
 * The agent's state after `e` episodes of a training run (re-run from the nearest checkpoint).
 *
 * @param env The environment of the run.
 * @param agent The agent of the run.
 * @param t The run (its seed and checkpoints are read).
 * @param e The number of episodes; 0 gives the initial state.
 * @returns The agent's state.
 *
 * @example The step count after three episodes
 * const env = gymEnvironment('gridworldEnvironment', { stepReward: -0.04 })
 * const counter = {
 *   name: 'up, then right; counts its steps',
 *   init: () => 0,
 *   act: (g, o) => ({ action: o === 0 || o === 4 ? 0 : 1 }),
 *   learn: (g) => g + 1,
 * }
 * const t = train(env, counter, { episodes: 6, checkpointEvery: 2 })
 * print('lengths', t.lengths)
 * print('after 3 episodes', agentAfter(env, counter, t, 3))
 */
export function agentAfter<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  t: Pick<Training<G>, 'seed' | 'checkpoints'>,
  e: number,
): G {
  return rerun(env, agent, t, e)
}

/**
 * Training episode `e` exactly as it happened, with exploration, and the agent's state before it. Throws `DomainError`
 * when `e` is below 1.
 *
 * @param env The environment of the run.
 * @param agent The agent of the run.
 * @param t The run (its seed and checkpoints are read).
 * @param e The episode's number, from 1.
 * @returns The episode's `trajectory`, and `agent`, the state the agent began it in.
 *
 * @example Episode 3 again
 * const env = gymEnvironment('gridworldEnvironment', { stepReward: -0.04 })
 * const counter = {
 *   name: 'up, then right; counts its steps',
 *   init: () => 0,
 *   act: (g, o) => ({ action: o === 0 || o === 4 ? 0 : 1 }),
 *   learn: (g) => g + 1,
 * }
 * const t = train(env, counter, { episodes: 5 })
 * const { trajectory, agent } = replay(env, counter, t, 3)
 * print('return', trajectory.episodeReturn, '; recorded', t.returns[2])
 * print('agent before it', agent, '; steps of episodes 1 and 2', t.lengths[0] + t.lengths[1])
 */
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
 *
 * @param env The environment to evaluate in: the training one, or a variant (another start state, say).
 * @param agent The agent of the run.
 * @param t The run (its seed and checkpoints are read).
 * @param e The number of training episodes after which to evaluate.
 * @param evaluationSeed The seed of the evaluation episode's stream.
 * @returns The evaluation `trajectory`, and `agent`, the state it was played with.
 *
 * @example The route after five episodes, on two evaluation seeds
 * const env = gymEnvironment('gridworldEnvironment', { stepReward: -0.04 })
 * const counter = {
 *   name: 'up, then right; counts its steps',
 *   init: () => 0,
 *   act: (g, o) => ({ action: o === 0 || o === 4 ? 0 : 1 }),
 *   learn: (g) => g + 1,
 * }
 * const t = train(env, counter, { episodes: 5 })
 * print('seed 0', evaluateEpisode(env, counter, t, 5, 0).trajectory.episodeReturn)
 * print('seed 1', evaluateEpisode(env, counter, t, 5, 1).trajectory.episodeReturn)
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
