import { describe, expect, it } from 'vitest'
import { cellState, episodes, greedyActions, rollout, tabularMdp } from 'aifn-methods/gym'
import {
  evaluatePolicy,
  expectedSarsaAgent,
  greedyPath,
  monteCarloControlAgent,
  nStepSarsaAgent,
  policyEvaluation,
  policyIteration,
  policyIterationAgent,
  qLearningAgent,
  reinforceAgent,
  sarsaAgent,
  tdPredictionAgent,
  valueIteration,
  valueIterationAgent,
  type TabularAgentState,
} from 'aifn-methods/gym/agents'
import {
  cliffWalking,
  cliffWalkingEnvironment,
  frozenLake,
  gridworld,
  gridworldEnvironment,
  maze,
  mazeEnvironment,
  mdpEnvironment,
} from 'aifn-methods/gym/environments'
import type { Agent } from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'

describe('planning', () => {
  it('value iteration solves a hand-worked corridor', () => {
    const corridor = gridworld({ width: 3, height: 1, walls: [], terminals: [{ x: 2, y: 0, value: 1 }], noise: 0 })
    const s = run(valueIteration(corridor), undefined, 100)
    expect(s.converged).toBe(true)
    const V = toFlat(s.V)
    expect(V[1]).toBeCloseTo(0.9, 12)
    expect(V[0]).toBeCloseTo(0.81, 12)
    expect(toFlat(s.policy)).toEqual([1, 1, -1])
  })

  it("value iteration reproduces Russell and Norvig's 4 × 3 utilities", () => {
    const g = gridworld({ stepReward: -0.04, gamma: 1, noise: 0.2 })
    const V = toFlat(run(valueIteration(g, { tolerance: 1e-12 }), undefined, 2000).V)
    const at = (x: number, y: number) => V[cellState(4, x, y)]
    expect(at(0, 0)).toBeCloseTo(0.705, 3)
    expect(at(0, 2)).toBeCloseTo(0.812, 3)
    expect(at(2, 2)).toBeCloseTo(0.918, 3)
    expect(at(2, 1)).toBeCloseTo(0.66, 3)
    expect(at(3, 0)).toBeCloseTo(0.388, 3)
  })

  it('policy iteration and value iteration agree, and exact evaluation matches iterative', () => {
    for (const mdp of [gridworld({ noise: 0.2 }), frozenLake(), maze(['..G', '.#.', 'S..'], { slip: 0.1 })]) {
      const vi = run(valueIteration(mdp), undefined, 5000)
      const pi = run(policyIteration(mdp), undefined, 50)
      expect(pi.converged).toBe(true)
      const a = toFlat(vi.V)
      toFlat(pi.V).forEach((v, i) => expect(v).toBeCloseTo(a[i], 6))
      const ev = run(policyEvaluation(mdp, pi.policy), undefined, 5000)
      toFlat(ev.V).forEach((v, i) => expect(v).toBeCloseTo(toFlat(evaluatePolicy(mdp, pi.policy))[i], 6))
    }
  })

  it('tabularMdp builds a general MDP', () => {
    // Two states: stay pays 1, leave to a terminal pays 5.
    const m = tabularMdp({
      transitions: [
        [
          [1, 0],
          [0, 1],
        ],
        [
          [0, 1],
          [0, 1],
        ],
      ],
      rewards: [
        [1, 5],
        [0, 0],
      ],
      gamma: 0.5,
      terminal: [1],
    })
    expect(toFlat(run(valueIteration(m), undefined, 200).V)[0]).toBeCloseTo(5, 10)
  })

  it('planners follow the trace protocol', () => {
    const g = gridworld({ noise: 0.2 })
    const record = { residual: (s: { residual: number }) => s.residual }
    expectProtocol(valueIteration(g), undefined, { record })
    expectProtocol(policyEvaluation(g, run(policyIteration(g), undefined, 50).policy), undefined, { record })
    expectProtocol(policyIteration(g), undefined, { n: 4 })
  })

  it('planning agents act optimally from the start on the environment’s model', () => {
    const env = mazeEnvironment({ layout: ['S..#', '.#..', '...G'] })
    for (const agent of [valueIterationAgent(), policyIterationAgent()]) {
      const s = run(episodes(env, agent), undefined, 1, { stream: stream(1) })
      // The shortest route: 5 moves, return 4 × (−1) + 10.
      expect(s.reachedTerminal).toBe(true)
      expect(s.episodeReturn).toBe(6)
    }
    expect(() => valueIterationAgent().init({ ...env, model: undefined }, stream(0))).toThrow(/tabular model/)
  })
})

type TabularAgent = Agent<TabularAgentState, number, number>

/** The greedy route of a learnt Q from the environment's start. */
const route = (env: ReturnType<typeof mazeEnvironment>, Q: TabularAgentState['Q']) =>
  toFlat(greedyPath(env.model, env.reset(stream(0)).state, greedyActions(env.model, Q.data)))

describe('learning agents', () => {
  const cliff = cliffWalkingEnvironment()

  it('Q-learning finds the cliff-edge path; SARSA earns more online by walking further away', () => {
    const rec = { record: { reward: (s: { episodeReturn: number }) => s.episodeReturn }, stream: stream(2) }
    const q = trace(episodes(cliff, qLearningAgent({ learningRate: 0.5, epsilon: 0.1 })), undefined, 500, rec)
    const s = trace(episodes(cliff, sarsaAgent({ learningRate: 0.5, epsilon: 0.1 })), undefined, 500, rec)
    const qPath = route(cliff, q.final.agent.Q)
    const sPath = route(cliff, s.final.agent.Q)
    expect(qPath.length).toBe(14)
    expect(qPath.at(-1)).toBe(11)
    // SARSA's greedy path leaves the row next to the cliff for a safer one (states 24 and up are two rows away).
    expect(Math.max(...sPath)).toBeGreaterThanOrEqual(24)
    const tail = (t: typeof q) =>
      toFlat(t.series.reward)
        .slice(-100)
        .reduce((a, b) => a + b, 0) / 100
    expect(tail(s)).toBeGreaterThan(tail(q))
  })

  it('expected SARSA, n-step SARSA and Monte Carlo control learn to reach the goal', () => {
    const m = mazeEnvironment({ layout: ['...G', '.#..', 'S...'] })
    for (const agent of [
      expectedSarsaAgent(),
      nStepSarsaAgent({ n: 3 }),
      monteCarloControlAgent({ epsilon: 0.2 }),
    ] as TabularAgent[]) {
      const last = run(episodes(m, agent), undefined, 300, { stream: stream(2) })
      expect(route(m, last.agent.Q).at(-1), agent.name).toBe(cellState(4, 3, 2))
    }
  })

  it('n-step SARSA with n = 1 is SARSA', () => {
    const m = mazeEnvironment({ layout: ['...G', '.#..', 'S...'], slip: 0.2 })
    const a = run(episodes(m, nStepSarsaAgent({ n: 1 })), undefined, 30, { stream: stream(6) })
    const b = run(episodes(m, sarsaAgent()), undefined, 30, { stream: stream(6) })
    toFlat(a.agent.Q).forEach((q, i) => expect(q).toBeCloseTo(toFlat(b.agent.Q)[i], 12))
  })

  it('TD(0) prediction approaches the exact value of a fixed policy', () => {
    const g = gridworld({ noise: 0.2 })
    const policy = run(valueIteration(g), undefined, 500).policy
    const exact = toFlat(evaluatePolicy(g, policy))
    const env = gridworldEnvironment({ noise: 0.2 })
    const est = run(episodes(env, tdPredictionAgent({ policy, learningRate: 0.02 })), undefined, 3000, {
      stream: stream(3),
    })
    expect(Math.abs(toFlat(est.agent.V)[0] - exact[0])).toBeLessThan(0.08)
  })

  it('REINFORCE improves the return on the corridor', () => {
    const corridor = mazeEnvironment({ layout: ['S....G'], rewards: { step: -1, goal: 10 } })
    const t = trace(episodes(corridor, reinforceAgent({ learningRate: 0.05 })), undefined, 400, {
      record: { reward: (s) => s.episodeReturn },
      stream: stream(4),
    })
    const r = toFlat(t.series.reward)
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length
    expect(mean(r.slice(-50))).toBeGreaterThan(mean(r.slice(1, 51)))
  })

  it('learners follow the trace protocol', () => {
    const policy = run(valueIteration(gridworld({ noise: 0.2 })), undefined, 500).policy
    const record = { g: (s: { episodeReturn: number }) => s.episodeReturn }
    for (const agent of [
      qLearningAgent(),
      sarsaAgent(),
      expectedSarsaAgent(),
      nStepSarsaAgent({ n: 3 }),
      monteCarloControlAgent(),
      reinforceAgent(),
    ] as Agent<unknown, number, number>[]) {
      expectProtocol(episodes(cliff, agent), undefined, { n: 4, record })
      expectProtocol(rollout(cliff, agent), undefined, { n: 30 })
    }
    expectProtocol(episodes(gridworldEnvironment({ noise: 0.2 }), tdPredictionAgent({ policy })), undefined, { n: 10 })
  })
})

describe('legal masking', () => {
  // State 0: action 0 stays (reward 1), action 1 is illegal, action 2 exits to the terminal state 1 (reward 5).
  const masked = tabularMdp({
    transitions: [
      [
        [1, 0],
        [0, 0],
        [0, 1],
      ],
      [
        [0, 1],
        [0, 1],
        [0, 1],
      ],
    ],
    rewards: [
      [1, 100, 5],
      [0, 0, 0],
    ],
    gamma: 0.5,
    terminal: [1],
  })

  it('planners never pick an illegal action; the environment lists the legal ones', () => {
    const vi = run(valueIteration(masked), undefined, 200)
    expect(toFlat(vi.Q)[1]).toBe(-Infinity)
    expect(toFlat(vi.policy)[0]).toBe(2)
    expect(toFlat(run(policyIteration(masked), undefined, 20).policy)[0]).toBe(2)
    const env = mdpEnvironment(masked, { horizon: 20 })
    expect(env.legal?.(0)).toEqual([0, 2])
    expect(() => env.step(0, 1, stream(0))).toThrow(/not legal/)
  })

  it('agents act only among legal actions', () => {
    const env = mdpEnvironment(masked, { horizon: 20 })
    for (const agent of [qLearningAgent({ epsilon: 1 }), sarsaAgent({ epsilon: 1 }), reinforceAgent()] as Agent<
      unknown,
      number,
      number
    >[]) {
      const tr = trace(rollout(env, agent), undefined, 200, { keep: 'all', stream: stream(8) })
      for (const s of tr.steps.slice(1)) expect(s.last!.action, agent.name).not.toBe(1)
    }
  })
})

describe('the MDP oracle', () => {
  it('gives V* by value iteration and the expected reward of an action', () => {
    const env = gridworldEnvironment({ noise: 0.2, stepReward: -0.04, gamma: 1 })
    const vStar = toFlat(env.oracle!.optimalValues!())
    const vi = toFlat(run(valueIteration(env.model, { tolerance: 1e-12 }), undefined, 5000).V)
    vStar.forEach((v, i) => expect(v).toBeCloseTo(vi[i], 8))
    expect(env.oracle!.expectedReward!(0, 0)).toBeCloseTo(-0.04, 12)
    expect(cliffWalking().states).toBe(48)
    expect(frozenLake().states).toBe(16)
    expect(maze(['S.G']).states).toBe(3)
  })
})
