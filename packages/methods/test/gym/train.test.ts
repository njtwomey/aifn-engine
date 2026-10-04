import { describe, expect, it } from 'vitest'
import {
  agentAfter,
  checkpointSpacing,
  episodes,
  evaluateEpisode,
  replay,
  runEpisode,
  train,
  training,
} from 'aifn-methods/gym'
import { qLearningAgent, reinforceAgent, ucb1, type TabularAgentState } from 'aifn-methods/gym/agents'
import { bernoulliBandit, mazeEnvironment } from 'aifn-methods/gym/environments'
import type { Agent, Trajectory } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

const env = mazeEnvironment({ layout: ['S..#', '.#..', '...G'], slip: 0.2 })
const agent = qLearningAgent({ epsilon: 0.2 }) as Agent<TabularAgentState, number, number>

describe('train', () => {
  it('matches a trace of episodes on the same seed, episode by episode', () => {
    const t = train(env, agent, { episodes: 40, seed: 3 })
    const tr = trace(episodes(env, agent), undefined, 40, { keep: 'all', stream: stream(3) })
    expect(Array.from(t.returns)).toEqual(tr.steps.slice(1).map((s) => s.episodeReturn))
    expect(Array.from(t.lengths)).toEqual(tr.steps.slice(1).map((s) => s.actions.length))
    expect(toFlat(t.final.Q)).toEqual(toFlat(tr.final.agent.Q))
    expect(t.done).toBe(true)
    expect(Object.keys(t.scalars)).toEqual(['mean max Q'])
    expect(t.regret).toBeNull()
  })

  it('keeps checkpoints at their spacing and within the memory bound', () => {
    expect(checkpointSpacing(100)).toBe(1)
    expect(checkpointSpacing(1000)).toBe(5)
    expect(checkpointSpacing(1000, 10)).toBe(10)
    expect(checkpointSpacing(50, 1, 10)).toBe(5)
    const t = train(env, agent, { episodes: 450, seed: 1, maxCheckpoints: 20 })
    expect(t.every).toBe(23)
    expect(t.checkpoints.length).toBeLessThanOrEqual(21)
    t.checkpoints.forEach((c, i) => expect(c.episode).toBe(i * t.every))
    // Each checkpoint is the agent's state after that many episodes.
    const c = t.checkpoints[3]
    expect(toFlat(agentAfter(env, agent, { seed: 1, checkpoints: t.checkpoints.slice(0, 1) }, c.episode).Q)).toEqual(
      toFlat(c.agent.Q),
    )
  })

  it('streams partial runs that are prefixes of the single run', () => {
    const whole = train(env, agent, { episodes: 60, seed: 'p' })
    const parts = [...training(env, agent, { episodes: 60, seed: 'p', chunk: 7 })]
    expect(parts.map((p) => p.episodes)).toEqual([7, 14, 21, 28, 35, 42, 49, 56, 60])
    for (const p of parts) {
      expect(Array.from(p.returns)).toEqual(Array.from(whole.returns.slice(0, p.episodes)))
      expect(Array.from(p.scalars['mean max Q'])).toEqual(Array.from(whole.scalars['mean max Q'].slice(0, p.episodes)))
      expect(p.done).toBe(p.episodes === 60)
    }
    expect(toFlat(parts.at(-1)!.final.Q)).toEqual(toFlat(whole.final.Q))
  })
})

describe('replay and evaluation', () => {
  it('replay of episode e equals the episode recorded during training', () => {
    const t = train(env, agent, { episodes: 120, seed: 5, maxCheckpoints: 7 })
    // The trajectories as they happened, recorded by running training by hand.
    const recorded: Trajectory<number, number, number>[] = []
    let g = agent.init(env, child(child(stream(5), 'init'), 'agent'))
    for (let e = 0; e < 120; e++) {
      const run = runEpisode(env, agent, g, child(stream(5), 'step', e))
      recorded.push(run.trajectory)
      g = run.agent
    }
    for (const e of [1, 17, 18, 60, 119, 120]) {
      const r = replay(env, agent, t, e)
      expect(r.trajectory, `episode ${e}`).toEqual(recorded[e - 1])
      expect(r.trajectory.episodeReturn).toBe(t.returns[e - 1])
    }
  })

  it('evaluation is greedy, does not learn, and is deterministic for a seed', () => {
    const t = train(env, agent, { episodes: 200, seed: 2 })
    const a = evaluateEpisode(env, agent, t, 200, 'ev')
    const b = evaluateEpisode(env, agent, t, 200, 'ev')
    expect(a.trajectory).toEqual(b.trajectory)
    expect(toFlat(a.agent.Q)).toEqual(toFlat(t.final.Q))
    // The greedy route of a trained agent reaches the goal; different evaluation seeds differ only through slips.
    expect(a.trajectory.reachedTerminal).toBe(true)
    const A = 4
    for (let k = 0; k < a.trajectory.actions.length; k++) {
      const o = a.trajectory.observations[k]
      expect(a.trajectory.actions[k]).toBe(agent.greedy!(a.agent, o))
      expect(Math.max(...toFlat(a.agent.Q).slice(o * A, (o + 1) * A))).toBe(
        toFlat(a.agent.Q)[o * A + a.trajectory.actions[k]],
      )
    }
    // REINFORCE has a greedy policy too; evaluation of an untrained agent is still deterministic.
    const rf = reinforceAgent() as Agent<unknown, number, number>
    const tr = train(env, rf, { episodes: 3 })
    expect(evaluateEpisode(env, rf, tr, 0, 1).trajectory).toEqual(evaluateEpisode(env, rf, tr, 0, 1).trajectory)
  })

  it('records regret and the pulled arm for a bandit', () => {
    const bandit = bernoulliBandit({ means: [0.2, 0.8] })
    const t = train(bandit, ucb1() as Agent<unknown, number, number>, { episodes: 300, seed: 1 })
    expect(Array.from(t.lengths).every((l) => l === 1)).toBe(true)
    expect(t.regret).not.toBeNull()
    const pulls = [0, 0]
    t.firstAction.forEach((a) => pulls[a]++)
    expect(pulls[1]).toBeGreaterThan(250)
    t.regret!.forEach((r, i) => expect(r).toBeCloseTo(t.firstAction[i] === 0 ? 0.6 : 0, 12))
  })
})

describe('episode endings', () => {
  it('a maze reports reaching the goal as a success and a time-out as a failure', () => {
    const corridor = mazeEnvironment({ layout: ['S.G'], horizon: 1 })
    const right: Agent<null, number, number> = {
      name: 'right',
      init: () => null,
      act: () => ({ action: 1 }),
      learn: () => null,
    }
    const short = runEpisode(corridor, right, null, stream(0)).trajectory
    expect(short.ending).toEqual({ success: false, reason: 'timed out after 1 step' })
    const long = runEpisode(mazeEnvironment({ layout: ['S.G'] }), right, null, stream(0)).trajectory
    expect(long.ending).toEqual({ success: true, reason: 'reached the goal' })
    const t = train(mazeEnvironment({ layout: ['S.G'] }), right, { episodes: 3 })
    expect(Array.from(t.outcome)).toEqual([1, 1, 1])
  })

  it('a bandit round has no outcome', () => {
    const t = train(bernoulliBandit(), ucb1() as Agent<unknown, number, number>, { episodes: 5 })
    expect(Array.from(t.outcome)).toEqual([0, 0, 0, 0, 0])
  })
})
