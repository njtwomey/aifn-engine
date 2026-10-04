import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  bufferSize,
  CHUNK,
  dqnAgent,
  gatherMinibatch,
  pushTransition,
  qNetwork,
  qValues,
  replayBuffer,
  sampleIndices,
  tdTargets,
  transitionAt,
  type ReplayBuffer,
  randomAgent,
} from 'aifn-methods/gym/agents'
import { cartPoleEnvironment } from 'aifn-methods/gym/environments'
import { agentAfter, evaluateEpisode, replay, rollout, train, training } from 'aifn-methods/gym'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, revive } from 'aifn-compute/foundation/tensor'
import { Mlp } from 'aifn-compute/nn/layers'
import { canon, expectProtocol } from '../../protocol'

const env = cartPoleEnvironment()

/** A buffer of `n` transitions where transition i has observation (i, −i), action i mod 2 and reward i. */
function filled(n: number, capacity: number): ReplayBuffer[] {
  const states = [replayBuffer(2, capacity)]
  for (let i = 0; i < n; i++) states.push(pushTransition(states[i], [i, -i], i % 2, i, [i + 1, -i - 1], i % 7 === 6))
  return states
}

describe('the replay buffer', () => {
  it('stores transitions exactly and keeps a window of the last `capacity`', () => {
    const states = filled(3 * CHUNK + 10, CHUNK + 5)
    const b = states.at(-1)!
    expect(b.count).toBe(3 * CHUNK + 10)
    expect(bufferSize(b)).toBe(CHUNK + 5)
    // Only chunks overlapping the window are held: the window starts in chunk 1, so chunk 0 is gone.
    expect(b.first).toBe(1)
    expect(b.chunks.length).toBe(2)
    const t = transitionAt(b, 3 * CHUNK + 9)
    expect([...t.observation, t.action, t.reward, ...t.next, t.terminated]).toEqual([
      3 * CHUNK + 9,
      -(3 * CHUNK + 9),
      1,
      3 * CHUNK + 9,
      3 * CHUNK + 10,
      -(3 * CHUNK + 10),
      (3 * CHUNK + 9) % 7 === 6,
    ])
    expect(transitionAt(b, b.count - bufferSize(b)).reward).toBe(b.count - bufferSize(b))
    expect(() => transitionAt(b, b.count - bufferSize(b) - 1)).toThrow(DomainError)
    expect(() => transitionAt(b, b.count)).toThrow(DomainError)
  })

  it('is persistent: a push never changes an earlier buffer, and sealed chunks are shared', () => {
    const states = filled(CHUNK + 3, 1000)
    const early = states[5]
    expect(early.count).toBe(5)
    expect(transitionAt(early, 4).reward).toBe(4)
    expect(() => transitionAt(early, 5)).toThrow(DomainError)
    // The tail is copied on each push, so the early state still holds exactly its own five transitions.
    expect(early.tail.length).toBe(5 * early.width)
    const a = states[CHUNK + 1]
    const b = states[CHUNK + 3]
    expect(a.chunks[0]).toBe(b.chunks[0])
  })

  it('samples uniformly from the window, determined by the stream alone', () => {
    const b = filled(500, 100).at(-1)!
    const i = sampleIndices(b, 2000, stream(3))
    expect(Array.from(sampleIndices(b, 2000, stream(3)))).toEqual(Array.from(i))
    expect(Array.from(sampleIndices(b, 2000, stream(4)))).not.toEqual(Array.from(i))
    expect(Math.min(...i)).toBe(400)
    expect(Math.max(...i)).toBe(499)
    const m = gatherMinibatch(b, i)
    expect(Array.from(m.rewards)).toEqual(Array.from(i))
    expect(m.observations[2 * 7 + 1]).toBe(-i[7])
    expect(() => sampleIndices(replayBuffer(2, 10), 1, stream(0))).toThrow(DomainError)
  })
})

describe('TD targets', () => {
  // A one-layer network Q(o) = o W + b with two inputs and two actions, set by hand.
  const net = Mlp([2, 2])
  const params = (w: number[], b: number[]) => [
    { weight: fromData(Float64Array.from(w), [2, 2]), bias: fromData(Float64Array.from(b), [2]) },
  ]
  const online = params([1, 0, 0, 1], [0, 0]) // Q = (o₁, o₂)
  const target = params([2, 0, 0, 3], [1, -1]) // Q̄ = (2 o₁ + 1, 3 o₂ − 1)
  const next = Float64Array.of(1, 2, 3, 1)
  const batch = { n: 2, rewards: Float64Array.of(1, 0.5), terminated: Float64Array.of(0, 1) }

  it('are r + γ Q̄(o′, a*) by hand, and r alone at a terminal state', () => {
    const qOnline = qValues(net, online, next, 2)
    const qTarget = qValues(net, target, next, 2)
    expect(Array.from(qOnline)).toEqual([1, 2, 3, 1])
    expect(Array.from(qTarget)).toEqual([3, 5, 7, 2])
    // Double DQN: a* = argmax of the online Q at o′ (action 1 for the first), valued by the target: 1 + 0.9 · 5.
    expect(Array.from(tdTargets(batch, qTarget, qOnline, 2, 0.9))).toEqual([1 + 0.9 * 5, 0.5])
    // DQN: a* = argmax of the target Q itself.
    expect(Array.from(tdTargets(batch, qTarget, qTarget, 2, 0.9))).toEqual([1 + 0.9 * 5, 0.5])
    // Where the two networks disagree the targets differ: o′ = (3, 1) is valued Q̄ = (7, 2).
    const live = { ...batch, terminated: Float64Array.of(0, 0) }
    expect(tdTargets(live, qTarget, qOnline, 2, 0.9)[1]).toBeCloseTo(0.5 + 0.9 * 7)
    const swapped = Float64Array.of(1, 2, 0, 1)
    expect(tdTargets(live, qTarget, swapped, 2, 0.9)[1]).toBeCloseTo(0.5 + 0.9 * 2)
  })
})

describe('the DQN agent', () => {
  const small = { warmup: 64, batchSize: 16, updateEvery: 2, targetSync: 50, epsilonSteps: 400, hidden: [8] }

  it('keeps a plain-data state that clones and revives, and is a pure rollout', () => {
    expectProtocol(rollout(env, dqnAgent(small)), undefined, { n: 200, record: { r: (s) => s.episodeReturn } })
  })

  it('reports its scalars and acts greedily by argmax Q', () => {
    const t = train(env, dqnAgent(small), { episodes: 12, seed: 5 })
    const g = t.final
    expect(g.updates).toBeGreaterThan(0)
    expect(g.steps).toBe(t.lengths.reduce((a, b) => a + b, 0))
    expect(Object.keys(t.scalars)).toEqual(['ε', 'loss', 'mean max Q', 'buffer size'])
    expect(t.scalars['buffer size'].at(-1)).toBe(g.steps)
    expect(t.scalars['loss'].at(-1)).toBeGreaterThan(0)
    const agent = dqnAgent(small)
    const o = Float64Array.of(0.01, 0, -0.02, 0)
    const q = agent.act(g, o, stream(0)).scores!
    expect(agent.greedy!(g, o)).toBe(q[1] > q[0] ? 1 : 0)
  })

  it('replays a training episode exactly as it was recorded', () => {
    const agent = dqnAgent(small)
    const t = train(env, agent, { episodes: 30, seed: 2, checkpointEvery: 7 })
    expect(t.every).toBe(7)
    for (const e of [1, 9, 15, 30]) {
      const r = replay(env, agent, t, e).trajectory
      expect(r.episodeReturn, `episode ${e}`).toBe(t.returns[e - 1])
      expect(r.actions.length).toBe(t.lengths[e - 1])
    }
    expect(canon(agentAfter(env, agent, t, 30))).toEqual(canon(t.final))
    const a = evaluateEpisode(env, agent, t, 30, 'x').trajectory
    expect(evaluateEpisode(env, agent, t, 30, 'x').trajectory.actions).toEqual(a.actions)
  })
})

describe('the Stable-Baselines3 schedule', () => {
  const agent = dqnAgent({
    warmup: 20,
    updateEvery: 8,
    gradientSteps: 3,
    targetSync: 5,
    epsilonStart: 1,
    epsilonEnd: 0.1,
    epsilonSteps: 100,
    batchSize: 4,
    hidden: [4],
  })
  const o = Float64Array.of(0.01, -0.02, 0.03, 0)
  const step = { observation: o, action: 1, reward: 1, next: o, terminated: false, truncated: false }

  it('trains in bursts of gradient steps after learning starts, and syncs the target every interval', () => {
    let g = agent.init(env, stream(1))
    const seen: { steps: number; updates: number; synced: boolean }[] = []
    for (let t = 0; t < 40; t++) {
      g = agent.learn(g, step)
      seen.push({ steps: g.steps, updates: g.updates, synced: g.target === g.online })
    }
    const at = (t: number) => seen[t - 1]
    // train_freq 8 after learning_starts 20 (strictly after): rounds at steps 24, 32 and 40, three steps each.
    expect(at(23).updates).toBe(0)
    expect(at(24).updates).toBe(3)
    expect(at(31).updates).toBe(3)
    expect(at(32).updates).toBe(6)
    expect(at(40).updates).toBe(9)
    // The target is copied at every multiple of 5, before that step's round: at step 40 the round follows the copy.
    expect(at(25).synced).toBe(true)
    expect(at(24).synced).toBe(false)
    expect(at(30).synced).toBe(true)
    expect(at(40).synced).toBe(false)
  })

  it('plays randomly until learning starts, then ε-greedy with ε linear in the steps', () => {
    let g = agent.init(env, stream(1))
    const eps = () => agent.scalars!(g).ε
    expect(Array.from(agent.act(g, o, stream(0)).probabilities!)).toEqual([0.5, 0.5])
    for (let t = 0; t < 50; t++) g = agent.learn(g, step)
    expect(eps()).toBeCloseTo(1 - 0.9 * 0.5)
    expect(agent.act(g, o, stream(0)).probabilities!.reduce((a, b) => Math.max(a, b))).toBeCloseTo(1 - eps() / 2)
    for (let t = 0; t < 60; t++) g = agent.learn(g, step)
    expect(eps()).toBeCloseTo(0.1)
  })
})

describe('a step budget and a stopped run', () => {
  it('stops after the episode that crosses the budget, and thins the checkpoints', () => {
    const t = train(env, randomAgent(), { steps: 1000, seed: 3, maxCheckpoints: 5 })
    const lengths = Array.from(t.lengths)
    expect(t.steps).toBe(lengths.reduce((a, b) => a + b, 0))
    expect(t.steps).toBeGreaterThanOrEqual(1000)
    expect(t.steps - lengths.at(-1)!).toBeLessThan(1000)
    expect(Number.isNaN(t.total)).toBe(true)
    expect(t.checkpoints.length).toBeLessThanOrEqual(6)
    expect(t.checkpoints.every((c) => c.episode % t.every === 0)).toBe(true)
    expect(replay(env, randomAgent(), t, t.episodes).trajectory.episodeReturn).toBe(t.returns.at(-1))
  })

  it('keeps a valid, replayable run when stopped part way (a streamed partial, after a worker message)', () => {
    const agent = dqnAgent({ warmup: 64, batchSize: 16, updateEvery: 4, gradientSteps: 2, hidden: [8] })
    const gen = training(env, agent, { steps: 3000, seed: 4 })
    gen.next()
    const partial = revive(structuredClone(gen.next().value))
    expect(partial.done).toBe(false)
    expect(partial.steps).toBeLessThan(3000)
    for (const e of [1, Math.ceil(partial.episodes / 2), partial.episodes]) {
      const r = replay(env, agent, partial, e).trajectory
      expect(r.episodeReturn, `episode ${e}`).toBe(partial.returns[e - 1])
    }
    expect(canon(agentAfter(env, agent, partial, partial.episodes))).toEqual(canon(partial.final))
  })
})

describe('the Q-network', () => {
  it('stacks hidden layers of the given width, activation and optional layer norm', () => {
    const net = qNetwork([4, 8, 8, 2], 'tanh', true)
    expect(net.label).toBe('4 → 8 → 8 → 2, tanh, layer norm')
    const p = net.init(stream(0))
    // Linear, LayerNorm, tanh, Linear, LayerNorm, tanh, Linear.
    expect(p.length).toBe(7)
    expect(qValues(net, p, new Float64Array(12), 4).length).toBe(6)
    const t = train(
      env,
      dqnAgent({ ...{ warmup: 32, batchSize: 8, hidden: [8, 8] }, activation: 'gelu', layerNorm: true }),
      {
        episodes: 4,
        seed: 0,
      },
    )
    expect(t.final.updates).toBeGreaterThan(0)
  })
})

describe('DQN on the cart-pole', () => {
  it('learns: its mean return over the last 50 of 200 episodes is at least 3× the random agent’s on 3 of 3 seeds', () => {
    const mean = (x: Float64Array) => x.reduce((a, b) => a + b, 0) / x.length
    for (const seed of [0, 1, 2]) {
      const t = train(env, dqnAgent({ hidden: [32, 32], updateEvery: 2, epsilonSteps: 5000 }), { episodes: 200, seed })
      const random = train(env, randomAgent(), { episodes: 200, seed })
      expect(mean(t.returns.slice(-50)), `seed ${seed}`).toBeGreaterThan(3 * mean(random.returns.slice(-50)))
    }
  }, 120_000)
})
