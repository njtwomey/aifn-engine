import { describe, expect, it } from 'vitest'
import { compare, rollout } from 'aifn-methods/gym'
import {
  epsilonGreedy,
  exp3,
  exploreThenCommit,
  klBernoulli,
  klUcb,
  klUcbIndex,
  laiRobbinsBound,
  linearThompson,
  linUcb,
  thompsonBernoulli,
  thompsonGaussian,
  ucb1,
  uniformPolicy,
} from 'aifn-methods/gym/agents'
import { bernoulliBandit, gaussianBandit, linearBandit } from 'aifn-methods/gym/environments'
import type { Agent } from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'

const env = bernoulliBandit({ means: [0.3, 0.5, 0.7] })
type ArmAgent = Agent<unknown, number, number>

describe('bandit helpers', () => {
  it('KL-UCB index solves n kl(p, q) = level', () => {
    expect(klBernoulli(0.5, 0.5)).toBe(0)
    const q = klUcbIndex(0.4, 20, Math.log(100))
    expect(20 * klBernoulli(0.4, q)).toBeCloseTo(Math.log(100), 6)
    expect(klUcbIndex(1, 5, 1)).toBe(1)
  })
  it('Lai–Robbins constant', () => {
    const r = laiRobbinsBound([0.5, 0.6], [10])
    expect(r.constant).toBeCloseTo(0.1 / klBernoulli(0.5, 0.6), 12)
  })
})

describe('bandit environments', () => {
  it('are one-step environments that reveal the pulled arm and know the arms’ means', () => {
    expect(env.horizon).toBe(1)
    expect(env.action).toMatchObject({ kind: 'discrete', n: 3 })
    const step = env.step(0, 2, stream(1))
    expect(step.terminated).toBe(true)
    expect([0, 1]).toContain(step.reward)
    expect(env.oracle?.expectedReward?.(0, 1)).toBe(0.5)
    expect(env.oracle?.bestExpectedReward?.(0)).toBe(0.7)
    const lin = linearBandit({ arms: 4, mode: 'random' })
    expect(lin.observation).toMatchObject({ kind: 'box', shape: [4, 2] })
    const { state } = lin.reset(stream(2))
    expect(state.length).toBe(8)
  })

  it('draw every arm’s reward, so the pulled arm’s reward does not depend on the agent (common random numbers)', () => {
    // Arm 2's reward in round t is the same whichever agent pulled it.
    const a = env.step(0, 2, stream(5))
    const b = env.step(0, 2, stream(5))
    expect(a.reward).toBe(b.reward)
    const u = run(rollout(env, uniformPolicy()), undefined, 30, { stream: stream(5) })
    const v = run(rollout(env, uniformPolicy()), undefined, 30, { stream: stream(5) })
    expect(u.totalReward).toBe(v.totalReward)
  })
})

describe('regret', () => {
  it('UCB1, KL-UCB and Thompson sampling beat uniform play and grow sublinearly', () => {
    const r = compare(env, [uniformPolicy(), ucb1(), klUcb(), thompsonBernoulli()] as ArmAgent[], {
      steps: 1000,
      replicates: 20,
      stream: stream(1),
      points: 10,
    })
    const final = toFlat(r.regret!.mean).filter((_, i) => i % 10 === 9)
    expect(final[0]).toBeCloseTo(0.4 * 1000 * (1 / 3) + 0.2 * 1000 * (1 / 3), -1)
    for (const k of [1, 2, 3]) expect(final[k]).toBeLessThan(final[0] / 4)
    // Thompson sampling and KL-UCB are near-optimal for Bernoulli arms: well below UCB1.
    expect(final[3]).toBeLessThan(final[1])
    const pulls = toFlat(r.actions)
    expect(pulls.slice(9, 12).reduce((a, b) => a + b, 0)).toBeCloseTo(1000, 6)
    expect(pulls[11]).toBeGreaterThan(800)
    // Every step is one episode of a bandit.
    expect(toFlat(r.episodes).at(-1)).toBe(1000)
  })

  it('every context-free policy runs on Bernoulli and Gaussian arms', () => {
    const policies = [
      exploreThenCommit({ m: 10 }),
      epsilonGreedy(),
      epsilonGreedy({ decay: 5 }),
      exp3(),
      thompsonGaussian(),
    ] as ArmAgent[]
    for (const e of [env, gaussianBandit({ means: [0, 0.5, 1] })]) {
      const r = compare(e, policies, { steps: 300, replicates: 3, stream: stream(2), points: 5 })
      expect(toFlat(r.regret!.mean).every(Number.isFinite)).toBe(true)
    }
  })

  it('LinUCB and linear Thompson sampling learn a linear bandit', () => {
    const lin = linearBandit({ theta: [1, 0.5], mode: 'random' })
    const r = compare(
      lin,
      [linUcb({ alpha: 0.5 }), linearThompson(), linUcb({ alpha: 0 })] as Agent<unknown, Float64Array, number>[],
      { steps: 300, replicates: 5, stream: stream(3), points: 3 },
    )
    const m = toFlat(r.regret!.mean)
    // The per-round regret late in the run is far below the early rate.
    expect(m[2] - m[1]).toBeLessThan(m[0])
  })
})

describe('bandit rollouts follow the trace protocol', () => {
  it('every policy: same key, same trace; seek = run; extend = longer trace; clones step the same', () => {
    const lin = linearBandit({ theta: [1, 0.5], mode: 'random' })
    const record = { regret: (s: { cumulativeRegret: number }) => s.cumulativeRegret }
    for (const p of [
      uniformPolicy(),
      ucb1(),
      klUcb(),
      thompsonBernoulli(),
      exploreThenCommit({ m: 2 }),
      epsilonGreedy(),
      exp3(),
      thompsonGaussian(),
    ] as ArmAgent[])
      expectProtocol(rollout(env, p), undefined, { n: 30, record })
    for (const p of [linUcb({ alpha: 0.5 }), linearThompson()])
      expectProtocol(rollout(lin, p), undefined, { n: 20, record })
  })
})
