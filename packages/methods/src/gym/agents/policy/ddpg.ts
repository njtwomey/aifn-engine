/**
 * Deep deterministic policy gradient (DDPG; Lillicrap et al., 2016, after Silver et al., 2014) for a box observation
 * and a one-dimensional box action $[a_{\min}, a_{\max}]$, on the gym protocol. An actor
 * $\mu_\theta(o) = c + h \tanh(\cdot)$ (with $c$ the box's centre and $h$ its half-width) proposes the action and a
 * critic $Q_\phi(o, a)$ scores it. Transitions go into the persistent replay buffer of `../dqn`; after `warmup` steps
 * of uniformly random actions, every `updateEvery` steps (once the buffer holds a minibatch) the agent takes
 * `gradientSteps` pairs of updates on fresh minibatches:
 *
 * - the critic regresses, by mean squared error, on
 *   $y = r + \gamma (1 - \text{terminated}) Q_{\bar\phi}(o', \mu_{\bar\theta}(o'))$ with target networks $\bar\phi$,
 *   $\bar\theta$;
 * - the actor ascends $Q_\phi(o, \mu_\theta(o))$ under the just-updated critic, the deterministic policy gradient,
 *   through the critic by autodiff;
 * - the targets follow by Polyak averaging, $\bar\phi \leftarrow \tau \phi + (1 - \tau) \bar\phi$ (likewise
 *   $\bar\theta$).
 *
 * Exploration adds Gaussian noise of standard deviation `noise` $\cdot h$ to the actor's action, clipped to the box
 * (TD3's choice over the original Ornstein–Uhlenbeck noise; Fujimoto et al., 2018). Minibatch $u$ is drawn from
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
  /** The hidden activation of both networks (default ReLU). */
  activation?: ActivationName
  /** The actor's Adam step size (default 1e-3). */
  actorLearningRate?: number
  /** The critic's Adam step size (default 1e-3). */
  criticLearningRate?: number
  /** The discount (default: the environment's). */
  gamma?: number
  /** Polyak rate $\tau$ (default 0.005). */
  tau?: number
  /** Exploration noise's standard deviation as a fraction of the action box's half-width (default 0.1). */
  noise?: number
  /** Transitions per minibatch (default 64). */
  batchSize?: number
  /** The replay window: the last this many transitions (default 50 000). */
  bufferSize?: number
  /** Uniformly random actions and no updates for the first this many steps (default 1000). */
  warmup?: number
  /** Train every this many steps (default 1). */
  updateEvery?: number
  /** Update pairs (critic, then actor) per training round (default 1). */
  gradientSteps?: number
}

/** The DDPG agent's state: plain data. */
export interface DdpgState {
  /** The seed of the minibatch draws, fixed in `init`. */
  seed: number
  /** The observation length $d$. */
  dim: number
  /** The action box's lower bound $a_{\min}$. */
  low: number
  /** The action box's upper bound $a_{\max}$. */
  high: number
  /** The discount $\gamma$. */
  gamma: number
  /** Environment steps learnt from. */
  steps: number
  /** Update pairs taken; also the index of the next minibatch's stream. */
  updates: number
  /** The actor's parameters $\theta$. */
  actor: Params[]
  /** The critic's parameters $\phi$. */
  critic: Params[]
  /** The target actor's parameters $\bar\theta$. */
  actorTarget: Params[]
  /** The target critic's parameters $\bar\phi$. */
  criticTarget: Params[]
  /** The actor's Adam state. */
  actorOptimizer: unknown
  /** The critic's Adam state. */
  criticOptimizer: unknown
  /** The replay buffer of every transition learnt from (the action in its action slot). */
  buffer: ReplayBuffer
  /** Sums over the current episode's updates: actor loss, critic loss, mean $Q(o, \mu(o))$, and their count. */
  running: { actorLoss: number; criticLoss: number; q: number; n: number }
  /** The means over the updates of the last episode that had any (NaN before the first). */
  last: { actorLoss: number; criticLoss: number; q: number }
}

/**
 * A DDPG agent for one continuous action (module docs). `init` throws `TypeError` unless the observation is a box and
 * the action a one-dimensional box. `act` draws from children of its stream's key, so it needs a fresh stream for
 * every step, as the gym's rollouts pass; after the warm-up its `scores` hold the actor's action before noise.
 * `greedy` is the actor's action, and `scalars` report the last episode's critic loss and mean $Q(o, \mu(o))$ and the
 * buffer size.
 *
 * @param options The networks, the optimisers and the schedule (see `DdpgOptions` for each default).
 * @returns The agent, named `'DDPG'`.
 *
 * @example A short run on a one-step task
 * // One-step episodes: the observation o is uniform on [-1, 1], and the action a in [-1, 1] earns -(a - o)^2.
 * const box = { kind: 'box', shape: [1], low: [-1], high: [1] }
 * const agent = ddpgAgent({ hidden: [16], warmup: 20, batchSize: 16, tau: 0.1 })
 * let g = agent.init({ name: 'track', observation: box, action: box, gamma: 0.9 }, stream(0))
 * const s = stream(1)
 * for (let t = 0; t < 60; t++) {
 *   const o = Float64Array.of(uniform(s, -1, 1))
 *   const action = agent.act(g, o, stream(`step ${t}`)).action
 *   const reward = -((action[0] - o[0]) ** 2)
 *   g = agent.learn(g, { observation: o, action, reward, next: o, terminated: true, truncated: false })
 * }
 * print('update pairs:', g.updates)
 * print('scalars:', agent.scalars(g))
 * print('greedy action at -0.5, 0, 0.5:', [-0.5, 0, 0.5].map((o) => agent.greedy(g, [o])[0]))
 */
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
    const a = fromData(b.actions, [n, 1])
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
