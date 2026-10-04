/**
 * Deep deterministic policy gradient (DDPG; Lillicrap et al., 2016, after Silver et al., 2014) for a box observation
 * and a one-dimensional box action, on the gym protocol. An actor μ_θ(o) = a_max tanh(·) proposes the action and a
 * critic Q_φ(o, a) scores it. Transitions go into the persistent replay buffer of `../dqn`; after `warmup` steps of
 * uniformly random actions, every `updateEvery` steps the agent takes `gradientSteps` pairs of updates on fresh
 * minibatches:
 *
 * - the critic regresses on y = r + γ (1 − terminated) Q_φ̄(o′, μ_θ̄(o′)) with target networks φ̄, θ̄;
 * - the actor ascends Q_φ(o, μ_θ(o)), the deterministic policy gradient, through the critic by autodiff;
 * - the targets follow by Polyak averaging, φ̄ ← τ φ + (1 − τ) φ̄ (likewise θ̄).
 *
 * Exploration adds Gaussian noise of standard deviation `noise` · a_max to the actor's action, clipped to the box
 * (TD3's choice over the original Ornstein–Uhlenbeck noise; Fujimoto et al., 2018). Minibatch u is drawn from
 * `child(stream(seed), 'batch', u)` with the seed fixed in `init`, so `learn` is pure.
 */

import type { Agent, AgentInfo, EnvironmentShape } from 'aifn-compute/foundation/contracts'
import { treeZip, type Params } from 'aifn-compute/foundation/pytree'
import { child, integers, normal, stream, uniform } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { domainDimension, int, real, space } from 'aifn-compute/foundation/space'
import {
  add,
  concat,
  fromData,
  mean,
  mul,
  neg,
  square,
  sub,
  tanh,
  toFlat,
  unwrap,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import type { ActivationName } from 'aifn-compute/nn/functional'
import { bufferSize, gatherMinibatch, pushTransition, replayBuffer, sampleIndices, type ReplayBuffer } from '../dqn'
import { adamWithClip, gradientStep, networkCache, rows } from './shared'

/** Options of `ddpgAgent`. */
export interface DdpgOptions {
  /** Hidden widths of actor and critic (default [64, 64]). */
  hidden?: readonly number[]
  activation?: ActivationName
  /** Adam step sizes (default 1e-3 both). */
  actorLearningRate?: number
  criticLearningRate?: number
  gamma?: number
  /** Polyak rate τ (default 0.005). */
  tau?: number
  /** Exploration noise as a fraction of a_max (default 0.1). */
  noise?: number
  batchSize?: number
  bufferSize?: number
  /** Uniformly random actions and no updates for the first this many steps (default 1000). */
  warmup?: number
  /** Train every this many steps (default 1), taking this many update pairs (default 1). */
  updateEvery?: number
  gradientSteps?: number
}

/** The DDPG agent's state: plain data. */
export interface DdpgState {
  seed: number
  dim: number
  low: number
  high: number
  gamma: number
  steps: number
  updates: number
  actor: Params[]
  critic: Params[]
  actorTarget: Params[]
  criticTarget: Params[]
  actorOptimizer: unknown
  criticOptimizer: unknown
  buffer: ReplayBuffer
  running: { actorLoss: number; criticLoss: number; q: number; n: number }
  last: { actorLoss: number; criticLoss: number; q: number }
}

/** A DDPG agent for one continuous action (module docs). */
export function ddpgAgent(options: DdpgOptions = {}): Agent<DdpgState, Float64Array, Float64Array> {
  const {
    hidden = [64, 64],
    activation = 'relu',
    actorLearningRate = 1e-3,
    criticLearningRate = 1e-3,
    tau = 0.005,
    noise = 0.1,
    batchSize = 64,
    bufferSize: capacity = 50_000,
    warmup = 1000,
    updateEvery = 1,
    gradientSteps = 1,
  } = options
  const net = networkCache(activation)
  const actorNet = (dim: number) => net([dim, ...hidden, 1])
  const criticNet = (dim: number) => net([dim + 1, ...hidden, 1])
  const actorRule = adamWithClip(actorLearningRate)
  const criticRule = adamWithClip(criticLearningRate)
  const polyak = (target: Params[], online: Params[]): Params[] =>
    treeZip([target, online], ([q, p]) => add(mul(1 - tau, q as Tensor), mul(tau, p as Tensor)))
  // μ(o) = centre + half-width · tanh(raw), for a batch [n, dim] → [n, 1].
  const policy = (
    g: Pick<DdpgState, 'dim' | 'low' | 'high'>,
    params: Params[],
    x: Tensor | ReturnType<typeof rows>,
  ) => {
    const half = (g.high - g.low) / 2
    const centre = (g.high + g.low) / 2
    return add(centre, mul(half, tanh(actorNet(g.dim).apply(params, x))))
  }
  const actionAt = (g: DdpgState, obs: ArrayLike<number>) =>
    toFlat(unwrap(policy(g, g.actor, rows(obs, 1, g.dim))) as Tensor)[0]

  const update = (g: DdpgState): DdpgState => {
    const b = gatherMinibatch(g.buffer, sampleIndices(g.buffer, batchSize, child(stream(g.seed), 'batch', g.updates)))
    const n = b.n
    const x = rows(b.observations, n, g.dim)
    const x2 = rows(b.next, n, g.dim)
    const a = fromData(Float64Array.from(b.actions), [n, 1])
    // Targets from the target networks (constants for the critic's gradient).
    const a2 = unwrap(policy(g, g.actorTarget, x2)) as Tensor
    const q2 = toFlat(unwrap(criticNet(g.dim).apply(g.criticTarget, concat([x2, a2], 1))) as Tensor)
    const y = fromData(
      Float64Array.from(q2, (q, i) => b.rewards[i] + g.gamma * (1 - b.terminated[i]) * q),
      [n, 1],
    )
    const critic = gradientStep(criticRule, g.critic, g.criticOptimizer, (p) =>
      mean(square(sub(criticNet(g.dim).apply(p, concat([x, a], 1)), y))),
    )
    let q = 0
    const actor = gradientStep(actorRule, g.actor, g.actorOptimizer, (p) => {
      const value = mean(criticNet(g.dim).apply(critic.params, concat([x, policy(g, p, x)], 1)))
      return neg(value)
    })
    q = -actor.loss
    return {
      ...g,
      actor: actor.params,
      critic: critic.params,
      actorOptimizer: actor.optimizer,
      criticOptimizer: critic.optimizer,
      actorTarget: polyak(g.actorTarget, actor.params),
      criticTarget: polyak(g.criticTarget, critic.params),
      updates: g.updates + 1,
      running: {
        actorLoss: g.running.actorLoss + actor.loss,
        criticLoss: g.running.criticLoss + critic.loss,
        q: g.running.q + q,
        n: g.running.n + 1,
      },
    }
  }

  return {
    name: 'DDPG',
    init: (env: EnvironmentShape, s) => {
      if (env.observation.kind !== 'box') throw new TypeError(`ddpgAgent: ${env.name} needs a box observation`)
      if (env.action.kind !== 'box' || domainDimension(env.action) !== 1)
        throw new TypeError(`ddpgAgent: ${env.name} needs a one-dimensional box action`)
      const dim = domainDimension(env.observation)
      const actor = actorNet(dim).init(child(s, 'actor'))
      const critic = criticNet(dim).init(child(s, 'critic'))
      return {
        seed: integers(child(s, 'seed'), 2 ** 31),
        dim,
        low: env.action.low[0],
        high: env.action.high[0],
        gamma: options.gamma ?? env.gamma,
        steps: 0,
        updates: 0,
        actor,
        critic,
        actorTarget: actor,
        criticTarget: critic,
        actorOptimizer: actorRule.init(actor),
        criticOptimizer: criticRule.init(critic),
        buffer: replayBuffer(dim, capacity),
        running: { actorLoss: 0, criticLoss: 0, q: 0, n: 0 },
        last: { actorLoss: NaN, criticLoss: NaN, q: NaN },
      }
    },
    act(g, obs, s) {
      if (g.steps < warmup) return { action: Float64Array.of(uniform(child(s, 'warmup'), g.low, g.high) as number) }
      const mu = actionAt(g, obs)
      const eps = ((normal(child(s, 'noise')) as number) * noise * (g.high - g.low)) / 2
      return { action: Float64Array.of(Math.min(g.high, Math.max(g.low, mu + eps))), scores: Float64Array.of(mu) }
    },
    greedy: (g, obs) => Float64Array.of(actionAt(g, obs)),
    learn(g, t) {
      let next: DdpgState = {
        ...g,
        steps: g.steps + 1,
        buffer: pushTransition(g.buffer, t.observation, t.action[0], t.reward, t.next, t.terminated),
      }
      if (next.steps > warmup && next.steps % updateEvery === 0 && bufferSize(next.buffer) >= batchSize)
        for (let k = 0; k < gradientSteps; k++) next = update(next)
      if (t.terminated || t.truncated) {
        const r = next.running
        next = {
          ...next,
          running: { actorLoss: 0, criticLoss: 0, q: 0, n: 0 },
          last: r.n > 0 ? { actorLoss: r.actorLoss / r.n, criticLoss: r.criticLoss / r.n, q: r.q / r.n } : next.last,
        }
      }
      return next
    },
    scalars: (g) => ({
      'critic loss': g.last.criticLoss,
      'mean Q(o, μ(o))': g.last.q,
      'buffer size': bufferSize(g.buffer),
    }),
  }
}

const agent = definer<AgentInfo>('agent', 'gym/agents/policy')

agent(
  {
    key: 'ddpgAgent',
    name: 'Deep deterministic policy gradient (DDPG)',
    summary:
      'A tanh actor and a Q critic from replayed transitions, the actor ascending the critic by the deterministic policy gradient, with Polyak target networks and Gaussian exploration.',
    params: space({
      actorLearningRate: real(1e-5, 1e-1, { default: 1e-3, scale: 'log', label: 'actor learning rate' }),
      criticLearningRate: real(1e-5, 1e-1, { default: 1e-3, scale: 'log', label: 'critic learning rate' }),
      noise: real(0, 1, { default: 0.1, label: 'exploration noise' }),
      warmup: int(0, 50_000, { default: 1000, label: 'random steps first' }),
    }),
    requires: { observation: 'box', action: 'box', families: ['control'] },
    notes: ['deterministic-policy-gradients', 'actor-critic'],
    cite: ['lillicrap2016', 'silver2014'],
    random: true,
  },
  ddpgAgent,
)
