/**
 * Offline reinforcement learning on logged transitions: conservative Q-learning (CQL; Kumar et al., 2020) in its
 * discrete-action form, against the same training without the conservative term (offline DQN).
 *
 * A behaviour policy (a linear threshold policy a = [w·o > 0] taking a uniformly random action with probability ε)
 * plays some episodes, and its transitions are logged. A Q-network is then trained on the log alone by Adam on the
 * Huber TD error to y = r + γ (1 − terminated) max_a′ Q̄(o′, a′), with a target network Q̄ copied every `targetSync`
 * updates, plus α times the CQL regulariser
 *
 *   E_(o, a)~log [log Σ_a′ exp Q(o, a′) − Q(o, a)],
 *
 * which pushes down the values of actions the log does not take and up those it does. Without it (α = 0), the max in
 * the target picks actions the log never tried, whose values nothing corrects, and the estimates drift above any return
 * the greedy policy achieves. Checkpoints report the greedy policy's return in fresh episodes and its mean estimated
 * value on the logged states.
 */

import type { Agent, Environment, EnvironmentShape, FunctionInfo, Transition } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, integers, stream, uniform } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { domainDimension } from 'aifn-compute/foundation/space'
import { add, fromData, logsumexp, mean, mul, sub, sum, toFlat } from 'aifn-compute/foundation/tensor'
import { huber, oneHot } from 'aifn-compute/learning/losses'
import { runEpisode } from '../../rollout'
import { qValues } from '../dqn'
import { adamWithClip, argmax, gradientStep, networkCache, rows, scalarOf } from './shared'

/** Logged transitions with discrete actions, as flat columns. */
export interface OfflineLog {
  readonly dim: number
  readonly actions: number
  readonly n: number
  readonly observations: Float64Array
  readonly next: Float64Array
  readonly taken: Int32Array
  readonly rewards: Float64Array
  readonly terminated: Float64Array
  /** The behaviour policy's undiscounted return in each logged episode. */
  readonly returns: Float64Array
}

/** Options of `logTransitions`. */
export interface LogOptions {
  episodes?: number
  /** The behaviour policy: weights w of a = [w·o > 0] (default [0, 0, 1, 1], the cart-pole's pole angle and rate). */
  weights?: readonly number[]
  /** The chance of a uniformly random action instead (default 0.5). */
  epsilon?: number
  seed?: number | string
}

/** Run an ε-noisy linear threshold policy for some episodes and log its transitions. */
export function logTransitions<S>(env: Environment<S, Float64Array, number>, options: LogOptions = {}): OfflineLog {
  const { episodes = 20, weights = [0, 0, 1, 1], epsilon = 0.5, seed = 'log' } = options
  if (env.action.kind !== 'discrete') throw new TypeError('logTransitions: needs discrete actions')
  const k = env.action.n
  type Rec = { steps: Transition<Float64Array, number>[] }
  const behaviour: Agent<Rec, Float64Array, number> = {
    name: 'ε-noisy threshold policy',
    init: () => ({ steps: [] }),
    act(_, obs, s) {
      if ((uniform(child(s, 'explore')) as number) < epsilon) return { action: integers(child(s, 'action'), k) }
      let v = 0
      for (let i = 0; i < weights.length; i++) v += weights[i] * obs[i]
      return { action: v > 0 ? 1 : 0 }
    },
    // Recording only: the state is the growing list of transitions.
    learn: (g, t) => ({ steps: [...g.steps, t] }),
  }
  let g = behaviour.init(env as EnvironmentShape, stream(seed))
  const returns: number[] = []
  for (let e = 0; e < episodes; e++) {
    const run = runEpisode(env, behaviour, g, child(stream(seed), 'episode', e))
    g = run.agent
    returns.push(run.trajectory.episodeReturn)
  }
  const dim = domainDimension(env.observation)
  const n = g.steps.length
  const out = {
    dim,
    actions: k,
    n,
    observations: new Float64Array(n * dim),
    next: new Float64Array(n * dim),
    taken: new Int32Array(n),
    rewards: new Float64Array(n),
    terminated: new Float64Array(n),
    returns: Float64Array.from(returns),
  }
  g.steps.forEach((t, i) => {
    out.observations.set(t.observation, i * dim)
    out.next.set(t.next, i * dim)
    out.taken[i] = t.action
    out.rewards[i] = t.reward
    out.terminated[i] = t.terminated ? 1 : 0
  })
  return out
}

/** Options of `offlineQLearning`. */
export interface OfflineQOptions {
  /** The CQL weight α (default 1; 0 is offline DQN). */
  alpha?: number
  /** Gradient updates (default 3000). */
  steps?: number
  batchSize?: number
  learningRate?: number
  gamma?: number
  hidden?: readonly number[]
  /** Copy the target network every this many updates (default 100). */
  targetSync?: number
  /** Evaluate every this many updates (default steps/20) over this many greedy episodes (default 5). */
  evaluateEvery?: number
  evaluationEpisodes?: number
  seed?: number | string
}

/** One checkpoint of an offline run. */
export interface OfflineCheckpoint {
  readonly step: number
  /** Mean undiscounted return of the greedy policy in fresh episodes. */
  readonly greedyReturn: number
  /** Mean max_a Q(o, a) over the logged states, and the TD and CQL terms of the last update. */
  readonly meanQ: number
  readonly tdLoss: number
  readonly cqlTerm: number
}

/** Train a Q-network on a log, offline, yielding checkpoints (a generator a worker can stream). */
export function* offlineQLearning<S>(
  env: Environment<S, Float64Array, number>,
  log: OfflineLog,
  options: OfflineQOptions = {},
): Generator<{ checkpoints: OfflineCheckpoint[]; done: boolean }> {
  const { alpha = 1, steps = 3000, batchSize = 64, learningRate = 1e-3, hidden = [64, 64], targetSync = 100 } = options
  const { evaluationEpisodes = 5, seed = 'offline' } = options
  const gamma = options.gamma ?? env.gamma
  const evaluateEvery = options.evaluateEvery ?? Math.max(1, Math.round(steps / 20))
  const net = networkCache('relu')([log.dim, ...hidden, log.actions])
  const rule = adamWithClip(learningRate, 10)
  let params: Params[] = net.init(child(stream(seed), 'init'))
  let target = params
  let optimizer = rule.init(params)
  const greedyAgent: Agent<Params[], Float64Array, number> = {
    name: 'greedy Q',
    init: () => params,
    act: (p, obs) => ({ action: argmax(qValues(net, p, Float64Array.from(obs), log.dim)) }),
    learn: (p) => p,
  }
  const checkpoints: OfflineCheckpoint[] = []
  let last = { td: NaN, cql: NaN }
  const evaluate = (step: number) => {
    let total = 0
    for (let e = 0; e < evaluationEpisodes; e++)
      total += runEpisode(env, greedyAgent, params, child(stream(seed), 'evaluate', e), 'greedy').trajectory
        .episodeReturn
    const q = qValues(net, params, log.observations, log.dim)
    let meanQ = 0
    for (let i = 0; i < log.n; i++) {
      let m = -Infinity
      for (let a = 0; a < log.actions; a++) m = Math.max(m, q[i * log.actions + a])
      meanQ += m / log.n
    }
    checkpoints.push({ step, greedyReturn: total / evaluationEpisodes, meanQ, tdLoss: last.td, cqlTerm: last.cql })
  }
  evaluate(0)
  yield { checkpoints: [...checkpoints], done: false }
  for (let u = 1; u <= steps; u++) {
    const ids = Array.from(toFlat(integers(child(stream(seed), 'batch', u), log.n, { shape: [batchSize] })))
    const x = rows(
      ids.flatMap((i) => Array.from(log.observations.subarray(i * log.dim, (i + 1) * log.dim))),
      batchSize,
      log.dim,
    )
    const nextObs = Float64Array.from(ids.flatMap((i) => Array.from(log.next.subarray(i * log.dim, (i + 1) * log.dim))))
    const qNext = qValues(net, target, nextObs, log.dim)
    const y = fromData(
      Float64Array.from(ids, (i, k) => {
        let m = -Infinity
        for (let a = 0; a < log.actions; a++) m = Math.max(m, qNext[k * log.actions + a])
        return log.rewards[i] + gamma * (1 - log.terminated[i]) * m
      }),
      [batchSize],
    )
    const mask = oneHot(
      ids.map((i) => log.taken[i]),
      log.actions,
    )
    let tdValue = NaN
    let cqlValue = NaN
    const step = gradientStep(rule, params, optimizer, (p) => {
      const q = net.apply(p, x)
      const qa = sum(mul(q, mask), 1)
      const td = huber(qa, y, { delta: 1, reduction: 'mean' })
      const cql = mean(sub(logsumexp(q, 1), qa))
      tdValue = scalarOf(td)
      cqlValue = scalarOf(cql)
      return alpha > 0 ? add(td, mul(alpha, cql)) : td
    })
    params = step.params
    optimizer = step.optimizer
    last = { td: tdValue, cql: cqlValue }
    if (u % targetSync === 0) target = params
    if (u % evaluateEvery === 0 || u === steps) {
      evaluate(u)
      yield { checkpoints: [...checkpoints], done: u === steps }
    }
  }
}

/** Log with a behaviour policy, then train CQL and offline DQN on the same log: a run for a worker to stream. */
export function* offlineComparison<S>(
  env: Environment<S, Float64Array, number>,
  logOptions: LogOptions,
  options: Omit<OfflineQOptions, 'alpha'> & { alpha?: number },
): Generator<{
  log: { n: number; returns: Float64Array }
  cql: OfflineCheckpoint[]
  dqn: OfflineCheckpoint[]
  done: boolean
}> {
  const log = logTransitions(env, logOptions)
  const summary = { n: log.n, returns: log.returns }
  let cql: OfflineCheckpoint[] = []
  let dqn: OfflineCheckpoint[] = []
  for (const r of offlineQLearning(env, log, { ...options, alpha: options.alpha ?? 1 })) {
    cql = r.checkpoints
    yield { log: summary, cql, dqn, done: false }
  }
  for (const r of offlineQLearning(env, log, { ...options, alpha: 0 })) {
    dqn = r.checkpoints
    yield { log: summary, cql, dqn, done: r.done }
  }
}

const fn = definer<FunctionInfo>('function', 'gym/agents/policy')
const OFFLINE = ['offline-reinforcement-learning']

fn(
  {
    key: 'logTransitions',
    name: 'Log a behaviour policy',
    summary: 'Transitions of an ε-noisy linear threshold policy, as an offline dataset.',
    role: 'simulation',
    random: true,
    notes: OFFLINE,
  },
  logTransitions,
)
fn(
  {
    key: 'offlineQLearning',
    name: 'Conservative Q-learning (discrete)',
    summary: 'A Q-network trained on logged transitions with a TD loss plus α(logsumexp Q − Q of the logged action).',
    role: 'fit',
    random: true,
    notes: OFFLINE,
    cite: ['kumar2020'],
  },
  offlineQLearning,
)
fn(
  {
    key: 'offlineComparison',
    name: 'CQL against offline DQN',
    summary: 'Log a behaviour policy, then train with and without the conservative term; greedy returns and mean Q.',
    role: 'simulation',
    random: true,
    notes: OFFLINE,
    cite: ['kumar2020'],
  },
  offlineComparison,
)
