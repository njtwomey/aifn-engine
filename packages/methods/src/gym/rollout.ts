/**
 * Running agents in environments (docs/aifn-gym.md §4): `rollout` takes one environment step per algorithm step,
 * `episodes` one whole episode per step, and `compare` runs several agents over replicates with common random numbers.
 * Step t's environment draws come from `child(ctx.stream, 'env')` and the agent's from `child(ctx.stream, 'agent')`,
 * so two agents traced on one root stream face the same environment randomness at every step. The rollout truncates an
 * episode at the environment's `horizon` (Gymnasium's `TimeLimit`), passes `env.legal` to the agent when the
 * environment masks actions, and scores pseudo-regret when the environment's oracle knows expected rewards.
 */

import type { Agent, Environment, Status, StepContext, Trajectory, Transition } from 'aifn-compute/foundation/contracts'
import { child, replicate, stream as rootStream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'

// ── One step ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The transition of one step, with the agent's scores and action probabilities when it reports them. */
export interface RolloutTransition<O, A> extends Transition<O, A> {
  scores?: Float64Array
  probabilities?: Float64Array
  /** Best expected reward minus the action's in the state acted in (when the oracle knows both). */
  regret?: number
}

/** The rollout's view of a running episode. */
interface Position<S, O, G> {
  envState: S
  observation: O
  agent: G
  /** Steps taken in the current episode. */
  length: number
}

/** The pseudo-regret of action a in state s, when the oracle can tell. */
function regretOf<S, O, A>(env: Environment<S, O, A>, s: S, a: A): number | undefined {
  const o = env.oracle
  if (!o?.expectedReward || !o.bestExpectedReward) return undefined
  return o.bestExpectedReward(s) - o.expectedReward(s, a)
}

/** One environment step: act (within the legal actions), step, truncate at the horizon, learn. */
function advance<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  at: Position<S, O, G>,
  stream: Stream,
): { at: Position<S, O, G>; transition: RolloutTransition<O, A> } {
  const choice = agent.act(at.agent, at.observation, child(stream, 'agent'), env.legal?.(at.envState))
  const step = env.step(at.envState, choice.action, child(stream, 'env'))
  const length = at.length + 1
  const regret = regretOf(env, at.envState, choice.action)
  const terminated = step.terminated
  const transition: RolloutTransition<O, A> = {
    observation: at.observation,
    action: choice.action,
    reward: step.reward,
    next: step.observation,
    terminated,
    truncated: !terminated && (step.truncated || length >= env.horizon),
    ...(env.legal && !terminated && { nextLegal: env.legal(step.state) }),
    ...(choice.scores && { scores: choice.scores }),
    ...(choice.probabilities && { probabilities: choice.probabilities }),
    ...(regret !== undefined && { regret }),
  }
  return {
    at: { envState: step.state, observation: step.observation, agent: agent.learn(at.agent, transition), length },
    transition,
  }
}

/** A fresh episode from a reset drawn from `stream`. */
function begin<S, O, A, G>(env: Environment<S, O, A>, agent: G, stream: Stream): Position<S, O, G> {
  const { state, observation } = env.reset(stream)
  return { envState: state, observation, agent, length: 0 }
}

// ── rollout ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A rollout's state after step t: the environment, the agent, and the episode so far. */
export interface RolloutState<S, O, A, G> extends Status {
  /** Environment steps taken. */
  t: number
  /** Episodes completed. */
  episode: number
  /** The environment's state and the agent's observation after the last step (the reset state at t = 0). */
  envState: S
  observation: O
  /** The agent's state (what it has learnt). */
  agent: G
  /** The last step's transition (null at t = 0). */
  last: RolloutTransition<O, A> | null
  /** Steps taken in the current episode. */
  length: number
  /** The undiscounted sum of the current episode's rewards. */
  episodeReturn: number
  /** The undiscounted sum of every reward so far. */
  totalReward: number
  /** Σ_t (best expected reward − the action's), when the environment's oracle knows expected rewards; else 0. */
  cumulativeRegret: number
  /** The last step ended its episode: the next step resets the environment first. */
  ended: boolean
}

/** Options for `rollout`. */
export interface RolloutOptions {
  /** Stop (`done`) once this many episodes have ended. Default: no limit. */
  episodes?: number
}

/**
 * One environment step per algorithm step (no start). When a step ends its episode (`terminated`, or `truncated` by the
 * environment or at its `horizon`), the state shows the arrival and the next step resets the environment, from
 * `child(ctx.stream, 'reset')`, before acting. Step 0 resets from the init stream. A bandit (horizon 1) is the one-step
 * case: every step is a round.
 */
export function rollout<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  { episodes }: RolloutOptions = {},
): Algorithm<void, RolloutState<S, O, A, G>> {
  return {
    name: `${agent.name} on ${env.name}`,
    init(_, stream) {
      const at = begin(env, agent.init(env, child(stream, 'agent')), child(stream, 'env'))
      return {
        t: 0,
        episode: 0,
        ...at,
        last: null,
        episodeReturn: 0,
        totalReward: 0,
        cumulativeRegret: 0,
        ended: false,
      }
    },
    step(s, ctx: StepContext) {
      const from = s.ended ? begin(env, s.agent, child(ctx.stream, 'reset')) : s
      const { at, transition } = advance(env, agent, from, ctx.stream)
      const ended = transition.terminated || transition.truncated
      return {
        t: s.t + 1,
        episode: s.episode + (ended ? 1 : 0),
        ...at,
        last: transition,
        episodeReturn: (s.ended ? 0 : s.episodeReturn) + transition.reward,
        totalReward: s.totalReward + transition.reward,
        cumulativeRegret: s.cumulativeRegret + (transition.regret ?? 0),
        ended,
      }
    },
    done: (s) => episodes !== undefined && s.episode >= episodes,
  }
}

// ── episodes ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** How `runEpisode` acts: learning as in training, or a fixed policy (greedy when the agent has `greedy`). */
export type EpisodeMode = 'learn' | 'greedy'

/**
 * One episode from agent state `g`: the reset from `child(stream, 'reset')`, the k-th step on `child(stream, 'step', k)`
 * (split into `env` and `agent` as in `rollout`), until a terminal state or the `horizon`. With `mode: 'greedy'` the
 * agent does not learn and takes `agent.greedy` actions (its `act` when it has none). Returns the trajectory and the
 * agent's state after it.
 */
export function runEpisode<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  g: G,
  stream: Stream,
  mode: EpisodeMode = 'learn',
): { agent: G; trajectory: Trajectory<S, O, A> } {
  const actor: Agent<G, O, A> =
    mode === 'learn'
      ? agent
      : {
          ...agent,
          act: (a, o, s, legal) => (agent.greedy ? { action: agent.greedy(a, o, legal) } : agent.act(a, o, s, legal)),
          learn: (a) => a,
        }
  let at = begin(env, g, child(stream, 'reset'))
  const states = [at.envState]
  const observations = [at.observation]
  const actions: A[] = []
  const rewards: number[] = []
  let terminated = false
  let regret = 0
  for (let k = 0; ; k++) {
    const { at: next, transition } = advance(env, actor, at, child(stream, 'step', k))
    at = next
    states.push(at.envState)
    observations.push(transition.next)
    actions.push(transition.action)
    rewards.push(transition.reward)
    regret += transition.regret ?? 0
    terminated = transition.terminated
    if (transition.terminated || transition.truncated) break
  }
  const episodeReturn = rewards.reduce((a, b) => a + b, 0)
  const ending = env.ending?.(at.envState, terminated ? 'terminated' : 'truncated', actions.length) ?? null
  return {
    agent: at.agent,
    trajectory: { states, observations, actions, rewards, episodeReturn, reachedTerminal: terminated, regret, ending },
  }
}

/** An `episodes` state after e episodes: the agent and the last episode in full. */
export interface EpisodeState<O, A, G, S = unknown> extends Status, Trajectory<S, O, A> {
  /** Episodes completed. */
  t: number
  agent: G
}

/**
 * One whole episode per algorithm step (no start): episode e is `runEpisode` on the step stream `ctx.stream`. An
 * episode ends at a terminal state or at the environment's `horizon` (which must then be finite).
 */
export function episodes<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
): Algorithm<void, EpisodeState<O, A, G, S>> {
  return {
    name: `${agent.name} on ${env.name}, by episode`,
    init(_, stream) {
      const { state, observation } = env.reset(child(stream, 'env'))
      return {
        t: 0,
        agent: agent.init(env, child(stream, 'agent')),
        states: [state],
        observations: [observation],
        actions: [],
        rewards: [],
        episodeReturn: 0,
        reachedTerminal: false,
        regret: 0,
        ending: null,
      }
    },
    step(s, ctx) {
      const { agent: g, trajectory } = runEpisode(env, agent, s.agent, ctx.stream)
      return { t: s.t + 1, agent: g, ...trajectory }
    },
  }
}

// ── compare ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Mean, standard deviation and 10 % and 90 % quantiles across replicates, each [agents, points]. */
export interface Spread {
  mean: Tensor
  sd: Tensor
  lower: Tensor
  upper: Tensor
}

/** Several agents on one environment, summarised over replicates. */
export interface Comparison {
  /** The recorded steps (about `points` of them, always including the last). */
  t: Tensor
  /** The agents' names, in order. */
  names: string[]
  /** Cumulative reward Σ r up to each recorded step. */
  reward: Spread
  /** Cumulative pseudo-regret up to each recorded step, when the environment's oracle knows expected rewards. */
  regret: Spread | null
  /** Mean episodes completed by each recorded step, [agents, points]. */
  episodes: Tensor
  /** Mean times each action was taken over the run, [agents, actions] (discrete action domains; else [agents, 0]). */
  actions: Tensor
  /** Every replicate's final cumulative regret (or reward, without an oracle), [agents, replicates]. */
  final: Tensor
}

/** Options for `compare`. */
export interface CompareOptions {
  /** Independent runs per agent. */
  replicates: number
  /** Environment steps per run. */
  steps: number
  /** The root stream; replicate k is `child(stream, k)` for every agent. Default `stream(0)`. */
  stream?: Stream
  /** About how many steps to record. Default 200. */
  points?: number
}

/** One run as `trace(rollout(env, agent), …, { stream: root })` would make it, recording the cumulative quantities. */
function runOnce<S, O, A, G>(env: Environment<S, O, A>, agent: Agent<G, O, A>, steps: number, root: Stream) {
  const alg = rollout(env, agent)
  let s = alg.init(undefined, child(root, 'init'))
  const reward = new Float64Array(steps)
  const regret = new Float64Array(steps)
  const episodes = new Float64Array(steps)
  const counts = new Float64Array(env.action.kind === 'discrete' ? env.action.n : 0)
  for (let t = 0; t < steps; t++) {
    const st = child(root, 'step', t)
    s = alg.step(s, { t, stream: st })
    reward[t] = s.totalReward
    regret[t] = s.cumulativeRegret
    episodes[t] = s.episode
    if (counts.length) counts[s.last!.action as number] += 1
  }
  return { reward, regret, episodes, counts }
}

/**
 * Run each agent `replicates` times for `steps` environment steps and summarise the cumulative reward and, when the
 * oracle knows expected rewards, the cumulative pseudo-regret Σ_t (best expected reward − the action's). Replicate k of
 * every agent runs on `child(stream, k)` (through `replicate`), as a trace of `rollout` on that root would, so all
 * agents face the same environment draws (common random numbers) and differences between them come from the agents.
 */
export function compare<S, O, A>(
  env: Environment<S, O, A>,
  agents: readonly Agent<unknown, O, A>[],
  { replicates, steps, stream, points = 200 }: CompareOptions,
): Comparison {
  const s = stream ?? rootStream(0)
  const every = Math.max(1, Math.floor(steps / points))
  const ts: number[] = []
  for (let t = every; t <= steps; t += every) ts.push(t)
  if (ts[ts.length - 1] !== steps) ts.push(steps)
  const P = agents.length
  const K = ts.length
  const hasRegret = !!(env.oracle?.expectedReward && env.oracle.bestExpectedReward)
  const A = env.action.kind === 'discrete' ? env.action.n : 0
  const spread = () => ({
    mean: new Float64Array(P * K),
    sd: new Float64Array(P * K),
    lower: new Float64Array(P * K),
    upper: new Float64Array(P * K),
  })
  const reward = spread()
  const regret = spread()
  const episodes = new Float64Array(P * K)
  const actions = new Float64Array(P * A)
  const final = new Float64Array(P * replicates)
  const summarise = (into: ReturnType<typeof spread>, p: number, k: number, col: number[]) => {
    col.sort((a, b) => a - b)
    const m = col.reduce((a, b) => a + b, 0) / replicates
    into.mean[p * K + k] = m
    into.sd[p * K + k] = Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, replicates - 1))
    into.lower[p * K + k] = col[Math.floor(0.1 * (replicates - 1))]
    into.upper[p * K + k] = col[Math.ceil(0.9 * (replicates - 1))]
  }
  agents.forEach((agent, p) => {
    const runs = replicate(replicates, s, (r) => runOnce(env, agent, steps, r), { cache: false })
    ts.forEach((t, k) => {
      summarise(
        reward,
        p,
        k,
        runs.map((r) => r.reward[t - 1]),
      )
      if (hasRegret)
        summarise(
          regret,
          p,
          k,
          runs.map((r) => r.regret[t - 1]),
        )
      episodes[p * K + k] = runs.reduce((a, r) => a + r.episodes[t - 1], 0) / replicates
    })
    runs.forEach((r, k) => {
      r.counts.forEach((c, a) => (actions[p * A + a] += c / replicates))
      final[p * replicates + k] = hasRegret ? r.regret[steps - 1] : r.reward[steps - 1]
    })
  })
  const shape = [P, K]
  const tensors = (x: ReturnType<typeof spread>): Spread => ({
    mean: fromData(x.mean, shape),
    sd: fromData(x.sd, shape),
    lower: fromData(x.lower, shape),
    upper: fromData(x.upper, shape),
  })
  return {
    t: fromData(Float64Array.from(ts)),
    names: agents.map((a) => a.name),
    reward: tensors(reward),
    regret: hasRegret ? tensors(regret) : null,
    episodes: fromData(episodes, shape),
    actions: fromData(actions, [P, A]),
    final: fromData(final, [P, replicates]),
  }
}
