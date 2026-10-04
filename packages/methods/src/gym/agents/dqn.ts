/**
 * The deep Q-network (Mnih et al., 2015, "Human-level control through deep reinforcement learning", Nature 518) on the
 * gym protocol, for a box observation and discrete actions, with the schedule of Stable-Baselines3's `DQN` (Raffin et
 * al., 2021). A multilayer perceptron maps an observation to one value Q(o, a) per action, and every transition goes
 * into an experience-replay buffer. The schedule counts environment steps t:
 *
 * - while t < `warmup` (SB3's `learning_starts`) actions are uniformly random; afterwards they are ε-greedy, with ε
 *   falling linearly from `epsilonStart` to `epsilonEnd` over `epsilonSteps` steps;
 * - every `targetSync` steps (SB3's `target_update_interval`, counted in environment steps as its `_on_step` does) the
 *   target network moves to the online one, Q̄ ← τ Q + (1 − τ) Q̄ (τ = 1: a copy);
 * - every `updateEvery` steps (`train_freq`) once t > `warmup`, the agent takes `gradientSteps` (`gradient_steps`)
 *   Adam steps, each on a fresh minibatch, on the Huber loss between Q(o, a) and y = r + γ (1 − terminated) Q̄(o′, a*).
 *   The target sync of step t comes before step t's updates, as in SB3, so Q̄ is fixed during a burst.
 *
 * Without double DQN (SB3's DQN) a* is the target network's argmax at o′; with it (van Hasselt, Guez and Silver, 2016)
 * a* is the online network's. A truncated transition still bootstraps; only a terminated one does not.
 *
 * The state is plain data: the two networks as parameter pytrees, Adam's moments, counters and the buffer. `learn`
 * draws no randomness of its own: minibatch u is drawn from `child(stream(seed), 'batch', u)` with the seed fixed in
 * `init`, so a learning step is a pure function of the state and the transition.
 *
 * **The buffer is persistent.** A copied ring buffer would cost O(capacity) per step, and a mutated one would corrupt
 * every earlier state that shares it (checkpoints, traces). Here transitions are stored in sealed chunks of
 * `CHUNK` transitions, which are never written again and are shared by every state that holds them, plus a tail of
 * fewer than `CHUNK` transitions that each push copies. A state holds only the chunks that overlap its window of the
 * last `capacity` transitions. Each state therefore sees exactly its own transitions, and checkpoints share all but
 * their tails.
 */

import { valueAndGrad, type ValueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Agent, AgentInfo, EnvironmentShape } from 'aifn-compute/foundation/contracts'
import { treeZip, type Params } from 'aifn-compute/foundation/pytree'
import { child, integers, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, domainDimension, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { add, fromData, mul, sum, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { huber } from 'aifn-compute/learning/losses'
import type { ActivationName } from 'aifn-compute/nn/functional'
import { lecunUniform } from 'aifn-compute/nn/init'
import { ActivationLayer, LayerNorm, Linear, Sequential, type Layer } from 'aifn-compute/nn/layers'
import { adamRule, applyUpdates, chainRules, clipByGlobalNorm } from 'aifn-compute/optim/first-order'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── The replay buffer ────────────────────────────────────────────────────────────────────────────────────────────────

/** Transitions per sealed chunk of the replay buffer. */
export const CHUNK = 256

/**
 * An experience-replay buffer over the last `capacity` transitions, persistent (module docs). A transition is stored as
 * `width = 2 · dim + 3` numbers: observation, action, reward, terminated (1 or 0), next observation.
 */
export interface ReplayBuffer {
  readonly dim: number
  readonly width: number
  readonly capacity: number
  /** Transitions ever pushed; transition i (0-based) lives in chunk ⌊i / CHUNK⌋. */
  readonly count: number
  /** The index of `chunks[0]`: chunks before it have left the window and are not held. */
  readonly first: number
  /** Sealed chunks `first`, `first + 1`, …, each CHUNK · width numbers, never written again. */
  readonly chunks: readonly Float32Array[]
  /** The transitions after the last sealed chunk, (count mod CHUNK) · width numbers. */
  readonly tail: Float32Array
}

/** One stored transition, as read back from a buffer. */
export interface StoredTransition {
  observation: Float32Array
  action: number
  reward: number
  next: Float32Array
  terminated: boolean
}

/** An empty buffer for observations of `dim` numbers, keeping the last `capacity` transitions. */
export function replayBuffer(dim: number, capacity: number): ReplayBuffer {
  if (!(capacity >= 1))
    throw new DomainError('replayBuffer', `replayBuffer: capacity must be at least 1, got ${capacity}`)
  const width = 2 * dim + 3
  return { dim, width, capacity, count: 0, first: 0, chunks: [], tail: new Float32Array(0) }
}

/** The number of transitions a sample draws from: min(count, capacity). */
export const bufferSize = (b: ReplayBuffer): number => Math.min(b.count, b.capacity)

/** A new buffer with one more transition; `b` is unchanged. */
export function pushTransition(
  b: ReplayBuffer,
  observation: ArrayLike<number>,
  action: number,
  reward: number,
  next: ArrayLike<number>,
  terminated: boolean,
): ReplayBuffer {
  const { dim, width } = b
  const tail = new Float32Array(b.tail.length + width)
  tail.set(b.tail)
  const at = b.tail.length
  for (let i = 0; i < dim; i++) {
    tail[at + i] = observation[i]
    tail[at + dim + 3 + i] = next[i]
  }
  tail[at + dim] = action
  tail[at + dim + 1] = reward
  tail[at + dim + 2] = terminated ? 1 : 0
  const count = b.count + 1
  if (count % CHUNK !== 0) return { ...b, count, tail }
  // Seal the tail, and let go of the chunks wholly before the window of the last `capacity` transitions.
  const start = Math.max(0, count - b.capacity)
  const keepFrom = Math.floor(start / CHUNK)
  const chunks = [...b.chunks, tail].slice(keepFrom - b.first)
  return { ...b, count, first: keepFrom, chunks, tail: new Float32Array(0) }
}

/** Where transition i lives: its array and offset. */
function locate(b: ReplayBuffer, i: number): { data: Float32Array; offset: number } {
  if (!(i >= Math.max(0, b.count - b.capacity) && i < b.count))
    throw new DomainError('replay buffer', `replay buffer: transition ${i} is outside the window`)
  const c = Math.floor(i / CHUNK)
  const data = c - b.first < b.chunks.length ? b.chunks[c - b.first] : b.tail
  return { data, offset: (i % CHUNK) * b.width }
}

/** Transition i (0-based over every push), which must still be in the window. */
export function transitionAt(b: ReplayBuffer, i: number): StoredTransition {
  const { data, offset } = locate(b, i)
  const { dim } = b
  return {
    observation: data.slice(offset, offset + dim),
    action: data[offset + dim],
    reward: data[offset + dim + 1],
    next: data.slice(offset + dim + 3, offset + 2 * dim + 3),
    terminated: data[offset + dim + 2] === 1,
  }
}

/** n transition indices drawn uniformly, with replacement, from the window, using only `s`. */
export function sampleIndices(b: ReplayBuffer, n: number, s: Stream): Int32Array {
  const size = bufferSize(b)
  if (size === 0) throw new DomainError('replay buffer', 'replay buffer: cannot sample an empty buffer')
  const offsets = toFlat(integers(s, size, { shape: [n] }))
  const start = b.count - size
  return Int32Array.from(offsets, (k) => start + k)
}

/** A minibatch as flat arrays: observations and next observations [n, dim], actions, rewards and terminated [n]. */
export interface Minibatch {
  n: number
  observations: Float64Array
  next: Float64Array
  actions: Int32Array
  rewards: Float64Array
  terminated: Float64Array
}

/** The transitions at `indices`, gathered into a minibatch. */
export function gatherMinibatch(b: ReplayBuffer, indices: ArrayLike<number>): Minibatch {
  const { dim } = b
  const n = indices.length
  const out: Minibatch = {
    n,
    observations: new Float64Array(n * dim),
    next: new Float64Array(n * dim),
    actions: new Int32Array(n),
    rewards: new Float64Array(n),
    terminated: new Float64Array(n),
  }
  for (let k = 0; k < n; k++) {
    const { data, offset } = locate(b, indices[k])
    for (let i = 0; i < dim; i++) {
      out.observations[k * dim + i] = data[offset + i]
      out.next[k * dim + i] = data[offset + dim + 3 + i]
    }
    out.actions[k] = data[offset + dim]
    out.rewards[k] = data[offset + dim + 1]
    out.terminated[k] = data[offset + dim + 2]
  }
  return out
}

// ── Q-values and TD targets ──────────────────────────────────────────────────────────────────────────────────────────

/** Q-values of a batch: a network applied to [n, dim] observations, as a flat [n, actions] array. */
export function qValues(net: Layer<Params[]>, params: Params[], observations: Float64Array, dim: number): Float64Array {
  const n = observations.length / dim
  return Float64Array.from(toFlat(net.apply(params, fromData(observations, [n, dim])) as never))
}

/** The index of the largest of `q[row · actions …]`, the first on ties. */
function argmaxRow(q: ArrayLike<number>, row: number, actions: number): number {
  let best = 0
  for (let a = 1; a < actions; a++) if (q[row * actions + a] > q[row * actions + best]) best = a
  return best
}

/**
 * TD targets y_k = r_k + γ (1 − terminated_k) Q̄(o′_k, a*_k), with a* the argmax of `selector` (the online network's
 * Q at o′ for double DQN, the target network's own for DQN) and Q̄ = `evaluator` (the target network's Q at o′), both
 * flat [n, actions].
 */
export function tdTargets(
  batch: Pick<Minibatch, 'n' | 'rewards' | 'terminated'>,
  evaluator: ArrayLike<number>,
  selector: ArrayLike<number>,
  actions: number,
  gamma: number,
): Float64Array {
  const y = new Float64Array(batch.n)
  for (let k = 0; k < batch.n; k++) {
    const a = argmaxRow(selector, k, actions)
    y[k] = batch.rewards[k] + gamma * (1 - batch.terminated[k]) * evaluator[k * actions + a]
  }
  return y
}

// ── The network ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Q-network: dense layers through `sizes` (input, hidden widths, actions), each hidden layer followed by an
 * optional layer norm and the activation, and a linear output (one Q per action). Initialised as PyTorch does.
 */
export function qNetwork(
  sizes: readonly number[],
  activation: ActivationName = 'relu',
  layerNorm = false,
): Layer<Params[]> {
  // PyTorch's default for `nn.Linear` (which SB3 keeps): weights and biases uniform on ±1/√fanIn.
  const init = lecunUniform()
  const layers: Layer<Params>[] = []
  for (let k = 0; k + 1 < sizes.length; k++) {
    layers.push(Linear(sizes[k], sizes[k + 1], { init, biasInit: init }))
    if (k + 2 === sizes.length) break
    if (layerNorm) layers.push(LayerNorm(sizes[k + 1]))
    layers.push(ActivationLayer(activation))
  }
  return {
    ...Sequential(...layers),
    kind: 'QNetwork',
    label: `${sizes.join(' → ')}, ${activation}${layerNorm ? ', layer norm' : ''}`,
  }
}

// ── The agent ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `dqnAgent`. */
export interface DqnOptions {
  /** Hidden layer widths (default [64, 64]), ReLU between layers. */
  hidden?: readonly number[]
  /** The hidden activation (default ReLU). */
  activation?: ActivationName
  /** Layer normalisation before each hidden activation (default false). */
  layerNorm?: boolean
  /** Adam's step size (default 1e-3). */
  learningRate?: number
  /** The discount (default: the environment's). */
  gamma?: number
  /** Transitions per minibatch (default 64). */
  batchSize?: number
  /** The replay window: the last this many transitions (default 10 000). */
  bufferSize?: number
  /** Random actions, and no updates, for the first this many steps (SB3's `learning_starts`; default 1000). */
  warmup?: number
  /** Train every this many environment steps (SB3's `train_freq`; default 4). */
  updateEvery?: number
  /** Gradient steps per training round (SB3's `gradient_steps`; default 1). */
  gradientSteps?: number
  /** Move the target network every this many environment steps (`target_update_interval`; default 500). */
  targetSync?: number
  /** Q̄ ← τ Q + (1 − τ) Q̄ at each target sync (default 1: a hard copy). */
  tau?: number
  /** Rescale gradients whose global norm exceeds this (default 10; Infinity for none). */
  clipNorm?: number
  /** ε falls linearly from `epsilonStart` to `epsilonEnd` over `epsilonSteps` environment steps (1, 0.05, 10 000). */
  epsilonStart?: number
  epsilonEnd?: number
  epsilonSteps?: number
  /** Double DQN targets (default true). */
  double?: boolean
  /** The Huber loss's δ (default 1). */
  huberDelta?: number
}

/** The DQN agent's state: plain data (module docs). */
export interface DqnState {
  /** The seed of the minibatch draws, fixed in `init`. */
  seed: number
  dim: number
  actions: number
  gamma: number
  /** Environment steps learnt from, and gradient steps taken. */
  steps: number
  updates: number
  online: Params[]
  target: Params[]
  /** Adam's state: its step count and moments. */
  optimizer: unknown
  buffer: ReplayBuffer
  /** Sums over the current episode's updates: loss and mean max-Q of the online network at o′. */
  running: { loss: number; maxQ: number; n: number }
  /** The means over the updates of the last episode that had any (NaN before the first; SB3 trains in bursts). */
  last: { loss: number; maxQ: number }
}

/** ε after `steps` environment steps. */
const epsilonAt = (steps: number, start: number, end: number, over: number) =>
  over <= 0 ? end : start + (end - start) * Math.min(1, steps / over)

/**
 * ε's decay length in steps for SB3's `exploration_fraction`: a fraction of the run's step budget (`total_timesteps`).
 * The agent does not know the budget, so the caller converts.
 */
export const epsilonStepsFor = (fraction: number, budgetSteps: number): number => Math.round(fraction * budgetSteps)

/** The Zoo recipe's budget in environment steps (`total_timesteps`) and its `exploration_fraction`. */
export const SB3_CARTPOLE_STEPS = 50_000
export const SB3_CARTPOLE_EPSILON_FRACTION = 0.16

/**
 * The training recipe of the RL Baselines3 Zoo's tuned `DQN` for CartPole-v1 (its `hyperparams/dqn.yml`) with a
 * budget of 50 000 steps, as `dqnAgent` options: lr 2.3 × 10⁻³, batch 64, buffer 100 000, learning starts 1000,
 * γ 0.99, target interval 10 steps, 128 gradient steps every 256 steps, ε from 1 to 0.04 over 0.16 of the budget, plain
 * DQN. The Zoo's network (256 × 2) is left out: the network is chosen separately.
 */
export const SB3_CARTPOLE = {
  learningRate: 2.3e-3,
  batchSize: 64,
  bufferSize: 100_000,
  warmup: 1000,
  gamma: 0.99,
  targetSync: 10,
  updateEvery: 256,
  gradientSteps: 128,
  epsilonStart: 1,
  epsilonEnd: 0.04,
  epsilonSteps: epsilonStepsFor(SB3_CARTPOLE_EPSILON_FRACTION, SB3_CARTPOLE_STEPS),
  double: false,
  tau: 1,
  clipNorm: 10,
} as const satisfies DqnOptions

/** A deep Q-network agent (module docs). */
export function dqnAgent(options: DqnOptions = {}): Agent<DqnState, Float64Array, number> {
  const {
    hidden = [64, 64],
    activation = 'relu',
    layerNorm = false,
    learningRate = 1e-3,
    batchSize = 64,
    bufferSize: capacity = 10_000,
    warmup = 1000,
    updateEvery = 4,
    gradientSteps = 1,
    targetSync = 500,
    tau = 1,
    clipNorm = 10,
    epsilonStart = 1,
    epsilonEnd = 0.05,
    epsilonSteps = 10_000,
    double = true,
    huberDelta = 1,
  } = options
  const adam = adamRule({ stepSize: learningRate })
  const rule = Number.isFinite(clipNorm) ? chainRules(clipByGlobalNorm(clipNorm), adam) : adam
  // The network depends on the environment's sizes, so it is built in `init` and looked up from the state's sizes.
  const nets = new Map<string, Layer<Params[]>>()
  const netFor = (dim: number, actions: number) => {
    const key = `${dim}:${actions}`
    let net = nets.get(key)
    if (!net) nets.set(key, (net = qNetwork([dim, ...hidden, actions], activation, layerNorm)))
    return net
  }
  const sync = (target: Params[], online: Params[]): Params[] =>
    tau === 1 ? online : treeZip([target, online], ([q, p]) => add(mul(1 - tau, q as Tensor), mul(tau, p as Tensor)))
  const epsilon = (g: DqnState) => epsilonAt(g.steps, epsilonStart, epsilonEnd, epsilonSteps)
  const greedyAction = (g: DqnState, obs: ArrayLike<number>) => {
    const q = qValues(netFor(g.dim, g.actions), g.online, Float64Array.from(obs), g.dim)
    return { action: argmaxRow(q, 0, g.actions), q }
  }

  /** One gradient step on minibatch `g.updates`. */
  const update = (g: DqnState): DqnState => {
    const net = netFor(g.dim, g.actions)
    const batch = gatherMinibatch(
      g.buffer,
      sampleIndices(g.buffer, batchSize, child(stream(g.seed), 'batch', g.updates)),
    )
    const evaluator = qValues(net, g.target, batch.next, g.dim)
    const selector = qValues(net, g.online, batch.next, g.dim)
    const y = fromData(tdTargets(batch, evaluator, double ? selector : evaluator, g.actions, g.gamma), [batch.n])
    const onehot = new Float64Array(batch.n * g.actions)
    for (let k = 0; k < batch.n; k++) onehot[k * g.actions + batch.actions[k]] = 1
    const mask = fromData(onehot, [batch.n, g.actions])
    const x = fromData(batch.observations, [batch.n, g.dim])
    const loss = (params: Params[]): Value =>
      huber(sum(mul(net.apply(params, x), mask), 1), y, { delta: huberDelta, reduction: 'mean' })
    const lossAndGrad: (params: Params[]) => ValueAndGrad<Value, unknown> = valueAndGrad(loss, {})
    const { value, grad } = lossAndGrad(g.online)
    const step = rule.update(grad as Params, g.optimizer as never)
    const online = applyUpdates(g.online, step.updates)
    let maxQ = 0
    for (let k = 0; k < batch.n; k++) maxQ += selector[k * g.actions + argmaxRow(selector, k, g.actions)]
    return {
      ...g,
      online,
      optimizer: step.state,
      updates: g.updates + 1,
      running: {
        loss: g.running.loss + (typeof value === 'number' ? value : toFlat(value as Tensor)[0]),
        maxQ: g.running.maxQ + maxQ / batch.n,
        n: g.running.n + 1,
      },
    }
  }

  return {
    name: double ? 'double DQN' : 'DQN',
    init: (env: EnvironmentShape, s) => {
      if (env.observation.kind !== 'box') throw new TypeError(`dqnAgent: ${env.name} needs a box observation`)
      if (env.action.kind !== 'discrete') throw new TypeError(`dqnAgent: ${env.name} needs discrete actions`)
      const dim = domainDimension(env.observation)
      const actions = env.action.n
      const online = netFor(dim, actions).init(child(s, 'network'))
      return {
        seed: integers(child(s, 'seed'), 2 ** 31),
        dim,
        actions,
        gamma: options.gamma ?? env.gamma,
        steps: 0,
        updates: 0,
        online,
        target: online,
        optimizer: rule.init(online),
        buffer: replayBuffer(dim, capacity),
        running: { loss: 0, maxQ: 0, n: 0 },
        last: { loss: NaN, maxQ: NaN },
      }
    },
    act(g, obs, s) {
      // SB3's `learning_starts`: uniformly random actions until the warm-up is over.
      const eps = g.steps < warmup ? 1 : epsilon(g)
      const { action, q } = greedyAction(g, obs)
      const probabilities = new Float64Array(g.actions).fill(eps / g.actions)
      probabilities[action] += 1 - eps
      if (uniform(child(s, 'explore')) < eps)
        return { action: integers(child(s, 'action'), g.actions), scores: q, probabilities }
      return { action, scores: q, probabilities }
    },
    greedy: (g, obs) => greedyAction(g, obs).action,
    learn(g, t) {
      let next: DqnState = {
        ...g,
        steps: g.steps + 1,
        buffer: pushTransition(g.buffer, t.observation, t.action, t.reward, t.next, t.terminated),
      }
      // SB3's order: the target sync in `_on_step` after each environment step, then a training round.
      if (next.steps % targetSync === 0) next = { ...next, target: sync(next.target, next.online) }
      if (next.steps > warmup && next.steps % updateEvery === 0)
        for (let k = 0; k < gradientSteps; k++) next = update(next)
      if (t.terminated || t.truncated) {
        const { loss, maxQ, n } = next.running
        next = {
          ...next,
          running: { loss: 0, maxQ: 0, n: 0 },
          last: n > 0 ? { loss: loss / n, maxQ: maxQ / n } : next.last,
        }
      }
      return next
    },
    scalars: (g) => ({
      ε: epsilon(g),
      loss: g.last.loss,
      'mean max Q': g.last.maxQ,
      'buffer size': bufferSize(g.buffer),
    }),
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const agent = definer<AgentInfo>('agent', 'gym/agents')

agent(
  {
    key: 'dqnAgent',
    name: 'Deep Q-network',
    summary:
      'A multilayer perceptron Q(o, ·) trained by Adam on Huber TD errors from an experience-replay buffer, with a target network and ε-greedy exploration.',
    params: space({
      learningRate: real(1e-5, 1e-1, { default: 1e-3, scale: 'log', label: 'learning rate' }),
      updateEvery: int(1, 1000, { default: 4, label: 'train every (steps)' }),
      gradientSteps: int(1, 512, { default: 1, label: 'gradient steps per round' }),
      warmup: int(0, 50_000, { default: 1000, label: 'learning starts (steps)' }),
      epsilonSteps: int(0, 200_000, { default: 10_000, label: 'ε decay steps' }),
      targetSync: int(1, 10_000, { default: 500, label: 'target sync (steps)' }),
      bufferSize: int(100, 100_000, { default: 10_000, label: 'buffer size' }),
      double: bool({ default: true, label: 'double DQN' }),
      activation: oneOf(['relu', 'tanh', 'gelu', 'elu', 'silu'], { default: 'relu', label: 'activation' }),
      layerNorm: bool({ default: false, label: 'layer norm' }),
    }),
    requires: { observation: 'box', action: 'discrete', families: ['control'] },
    notes: ['deep-q-network'],
    random: true,
  },
  dqnAgent,
)
