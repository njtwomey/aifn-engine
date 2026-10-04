import { describe, expect, it } from 'vitest'
import { compare, episodes, greedyActions, rollout } from 'aifn-methods/gym'
import {
  qLearningAgent,
  randomAgent,
  ucb1,
  uniformPolicy,
  valueIteration,
  type TabularAgentState,
} from 'aifn-methods/gym/agents'
import { bernoulliBandit, mazeEnvironment } from 'aifn-methods/gym/environments'
import type { Agent } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../protocol'

const tiny = ['S..', '.#.', '..G']

describe('the maze environment', () => {
  it('declares discrete domains, a tabular model and a grid render', () => {
    const env = mazeEnvironment({ layout: tiny })
    expect(env.observation).toEqual({ kind: 'discrete', n: 9 })
    expect(env.action).toEqual({ kind: 'discrete', n: 4, names: ['up', 'right', 'down', 'left'] })
    expect(env.model.kind).toBe('tabular')
    expect(env.render?.kind).toBe('grid')
    expect(env.horizon).toBe(36)
  })

  it('terminates at the goal and pays its reward', () => {
    const env = mazeEnvironment({ layout: ['S.G'] })
    const { state } = env.reset(stream(0))
    const a = env.step(state, 1, stream(1))
    expect(a).toMatchObject({ state: 1, reward: -1, terminated: false, truncated: false })
    expect(env.step(a.state, 1, stream(2))).toMatchObject({ state: 2, reward: 10, terminated: true })
  })
})

describe('rollout', () => {
  it('passes the Algorithm protocol (purity, seek, extend, clone and revive) for both agents', () => {
    const env = mazeEnvironment({ layout: tiny })
    expectProtocol(rollout(env, randomAgent()), undefined, { n: 40 })
    expectProtocol(rollout(env, qLearningAgent()), undefined, {
      n: 40,
      record: { t: (s) => s.t, ret: (s) => s.episodeReturn },
    })
    expectProtocol(episodes(env, qLearningAgent()), undefined, { n: 6 })
  })

  it('truncates at the horizon, apart from termination, and resets on the next step', () => {
    // A random agent in a maze with no reachable goal within 3 steps is truncated, never terminated.
    const env = mazeEnvironment({ layout: ['S.......G'], horizon: 3 })
    const tr = trace(rollout(env, randomAgent()), undefined, 12, { keep: 'all', stream: stream(3) })
    const ends = tr.steps.filter((s) => s.ended)
    expect(ends.length).toBe(4)
    for (const s of ends) expect(s.last).toMatchObject({ truncated: true, terminated: false })
    expect(tr.steps.map((s) => s.length)).toEqual([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    // A terminal transition is terminated and not truncated, even on the horizon's last step.
    const corridor = mazeEnvironment({ layout: ['S.G'], horizon: 2 })
    const right: Agent<null, number, number> = {
      name: 'right',
      init: () => null,
      act: () => ({ action: 1 }),
      learn: () => null,
    }
    const s = run(rollout(corridor, right), undefined, 2, { stream: stream(0) })
    expect(s.last).toMatchObject({ terminated: true, truncated: false, reward: 10 })
    expect(s.episode).toBe(1)
    expect(s.episodeReturn).toBe(9)
  })

  it('gives two agents the same environment draws (common random numbers)', () => {
    const env = mazeEnvironment({ layout: tiny, slip: 0.4 })
    const a = run(rollout(env, randomAgent()), undefined, 30, { stream: stream(9) })
    const b = run(rollout(env, randomAgent()), undefined, 30, { stream: stream(9) })
    expect(a.envState).toBe(b.envState)
  })

  it('stops after the requested number of episodes', () => {
    const env = mazeEnvironment({ layout: tiny })
    const tr = trace(rollout(env, qLearningAgent(), { episodes: 5 }), undefined, 10_000, { stream: stream(1) })
    expect(tr.meta.stopped).toBe('done')
    expect(tr.final.episode).toBe(5)
  })
})

describe('Q-learning as an agent', () => {
  it("reaches value iteration's greedy policy on a small maze", () => {
    const env = mazeEnvironment({ layout: ['S..#', '.#..', '...G'], gamma: 0.9 })
    const learnt = run(episodes(env, qLearningAgent({ epsilon: 0.3, learningRate: 0.5 })), undefined, 400, {
      stream: stream('q'),
    })
    const Q = (learnt.agent as TabularAgentState).Q
    const optimal = run(valueIteration(env.model, { tolerance: 1e-10 }), undefined, 1000)
    const A = env.model.actions
    const qStar = toFlat(optimal.Q)
    const policy = greedyActions(env.model, Q.data)
    for (let s = 0; s < env.model.states; s++) {
      if (env.model.terminal[s]) continue
      const best = Math.max(...qStar.slice(s * A, (s + 1) * A))
      expect(qStar[s * A + policy[s]], `state ${s}`).toBeCloseTo(best, 6)
    }
  })

  it('does not bootstrap past a terminal transition but does past a truncated one', () => {
    const agent = qLearningAgent({ learningRate: 1, gamma: 0.5, initialQ: 4 })
    const g = agent.init(mazeEnvironment({ layout: ['S.G'] }), stream(0))
    const base = { observation: 0, action: 1, reward: 1, next: 1 }
    const term = agent.learn(g, { ...base, terminated: true, truncated: false })
    const trunc = agent.learn(g, { ...base, terminated: false, truncated: true })
    expect(toFlat(term.Q)[1]).toBe(1)
    expect(toFlat(trunc.Q)[1]).toBe(3)
  })
})

describe('compare', () => {
  it('equals traces of rollout on each replicate stream, with common random numbers across agents', () => {
    const env = bernoulliBandit({ means: [0.2, 0.6] })
    const agents = [uniformPolicy(), ucb1()] as Agent<unknown, number, number>[]
    const c = compare(env, agents, { replicates: 3, steps: 50, stream: stream('c'), points: 5 })
    expect(c.names).toEqual(['uniform', 'UCB1'])
    expect(toFlat(c.t)).toEqual([10, 20, 30, 40, 50])
    for (const [p, agent] of agents.entries())
      for (let k = 0; k < 3; k++) {
        const s = run(rollout(env, agent), undefined, 50, { stream: child(stream('c'), k) })
        expect(toFlat(c.final)[p * 3 + k]).toBeCloseTo(s.cumulativeRegret, 12)
      }
    // The same agent twice: identical curves, since every replicate's draws are shared.
    const twice = compare(env, [ucb1(), ucb1()] as Agent<unknown, number, number>[], { replicates: 4, steps: 40 })
    const m = toFlat(twice.regret!.mean)
    expect(m.slice(0, m.length / 2)).toEqual(m.slice(m.length / 2))
  })

  it('reports reward and episodes without an oracle regret on an MDP', () => {
    const env = mazeEnvironment({ layout: tiny })
    const c = compare(env, [randomAgent(), qLearningAgent()] as Agent<unknown, number, number>[], {
      replicates: 2,
      steps: 200,
      points: 4,
    })
    expect(c.regret).toBeNull()
    expect(c.actions.shape).toEqual([2, 4])
    const ep = toFlat(c.episodes)
    expect(ep[7]).toBeGreaterThan(ep[3])
  })
})
