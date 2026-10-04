import { describe, expect, it } from 'vitest'
import { crossEntropyAgent, linearPolicyAgent, lqrBangBangAgent } from 'aifn-methods/gym/agents/control'
import { cartPoleEnvironment } from 'aifn-methods/gym/environments/control'
import { episodes, rollout } from 'aifn-methods/gym'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../../protocol'

const env = cartPoleEnvironment()

/** The returns of `n` episodes of an agent, on their own streams. */
function returns(agent: Parameters<typeof episodes>[1], n: number, seed: number, e = env): number[] {
  const tr = trace(episodes(e, agent), undefined, n, { stream: stream(seed), record: { r: (s) => s.episodeReturn } })
  return Array.from(toFlat(tr.series.r).slice(1))
}

describe('the LQR bang-bang agent', () => {
  it('balances for 500 steps from small perturbations', () => {
    expect(returns(lqrBangBangAgent(), 10, 1)).toEqual(new Array(10).fill(500))
    expect(returns(lqrBangBangAgent(), 5, 2, cartPoleEnvironment({ jitter: 0.1 }))).toEqual(new Array(5).fill(500))
  })
})

describe('the cross-entropy agent', () => {
  it('reaches a mean return ≥ 475 within 40 generations of 20 on at least 4 of 5 seeds', () => {
    let solved = 0
    for (let seed = 0; seed < 5; seed++) {
      const s = run(episodes(env, crossEntropyAgent()), undefined, 800, { stream: stream(seed) })
      expect(s.agent.generation).toBe(40)
      const r = returns(linearPolicyAgent(s.agent.mean), 10, 100 + seed)
      if (r.reduce((a, b) => a + b, 0) / r.length >= 475) solved++
    }
    expect(solved).toBeGreaterThanOrEqual(4)
  }, 60_000)

  it('keeps a plain-data state and is a pure rollout', () => {
    const s = run(episodes(env, crossEntropyAgent({ population: 4 })), undefined, 9, { stream: stream(3) })
    expect(JSON.parse(JSON.stringify(s.agent))).toEqual(s.agent)
    expect(s.agent.generation).toBe(2)
    expect(s.agent.last).toMatchObject({ generation: 1 })
    expectProtocol(rollout(env, crossEntropyAgent({ population: 4 })), undefined, { n: 80 })
    expectProtocol(rollout(env, lqrBangBangAgent()), undefined, { n: 30 })
  })
})
