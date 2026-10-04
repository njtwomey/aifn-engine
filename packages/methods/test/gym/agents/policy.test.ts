import { describe, expect, it } from 'vitest'
import { train } from 'aifn-methods/gym'
import { cartPoleEnvironment, pendulumEnvironment } from 'aifn-methods/gym/environments'
import {
  a2cAgent,
  ddpgAgent,
  logTransitions,
  offlineQLearning,
  ppoAgent,
  reinforceBaselineAgent,
} from 'aifn-methods/gym/agents'

const env = cartPoleEnvironment({})
const mean = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0) / a.length

describe('policy-gradient agents', () => {
  it('REINFORCE with a baseline learns to balance the cart-pole', () => {
    const r = train(env, reinforceBaselineAgent(), { steps: 20_000, seed: 1 })
    expect(mean(r.returns.slice(-10))).toBeGreaterThan(150)
    expect(mean(r.returns.slice(0, 10))).toBeLessThan(60)
    expect(Number.isFinite(r.scalars['value loss'].at(-1)!)).toBe(true)
  })
  it('A2C and PPO update and replay exactly from their checkpoints', () => {
    for (const agent of [a2cAgent({ nSteps: 8 }), ppoAgent({ horizon: 128, epochs: 2, batchSize: 32 })]) {
      const a = train(env, agent as never, { steps: 600, seed: 3 })
      const b = train(env, agent as never, { steps: 600, seed: 3 })
      expect(Array.from(a.returns)).toEqual(Array.from(b.returns))
      expect((a.final as { updates: number }).updates).toBeGreaterThan(0)
    }
    const p = train(env, ppoAgent({ horizon: 128, epochs: 2, batchSize: 32 }) as never, { steps: 600, seed: 3 })
    const clip = p.scalars['clip fraction'].filter(Number.isFinite)
    expect(clip.length).toBeGreaterThan(0)
    for (const c of clip) expect(c).toBeGreaterThanOrEqual(0)
  })
  it('DDPG acts in the box and trains its critic after the warm-up', () => {
    const pend = pendulumEnvironment({})
    const r = train(pend, ddpgAgent({ warmup: 200, batchSize: 32 }), { steps: 400, seed: 2 })
    expect(r.episodes).toBe(2)
    const g = r.final
    expect(g.updates).toBe(200)
    expect(Number.isFinite(g.last.criticLoss)).toBe(true)
    const a = ddpgAgent().greedy!(g, Float64Array.of(1, 0, 0))
    expect(Math.abs(a[0])).toBeLessThanOrEqual(2)
  })
})

describe('offline Q-learning', () => {
  it('logs a behaviour policy and trains with the conservative term', () => {
    const log = logTransitions(env, { episodes: 5, epsilon: 0.5, seed: 1 })
    expect(log.n).toBe(Array.from(log.returns).reduce((a, b) => a + b, 0))
    let last: { checkpoints: { meanQ: number; cqlTerm: number }[] } | undefined
    for (const r of offlineQLearning(env, log, { steps: 200, alpha: 1, evaluationEpisodes: 1 })) last = r
    const c = last!.checkpoints.at(-1)!
    expect(Number.isFinite(c.meanQ)).toBe(true)
    // logsumexp Q − Q(a) ≥ 0 always.
    expect(c.cqlTerm).toBeGreaterThanOrEqual(0)
  })
})
