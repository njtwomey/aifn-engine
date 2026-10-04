import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { cellState, type TabularMdp } from 'aifn-methods/gym'
import {
  FROZEN_LAKE_MAPS,
  bernoulliBandit,
  cliffWalking,
  frozenLake,
  gaussianBandit,
  gridworld,
  linearBandit,
  maze,
} from 'aifn-methods/gym/environments'
import { child, stream } from 'aifn-compute/foundation/random'

/** Every non-terminal (state, action) has a probability distribution over outcomes; terminals have none. */
function wellFormed(m: TabularMdp) {
  expect(m.outcomes.length).toBe(m.states * m.actions)
  expect(m.start).toBeGreaterThanOrEqual(0)
  expect(m.start).toBeLessThan(m.states)
  for (let s = 0; s < m.states; s++)
    for (let a = 0; a < m.actions; a++) {
      const o = m.outcomes[s * m.actions + a]
      if (m.terminal[s]) expect(o).toEqual([])
      else {
        expect(o.reduce((p, x) => p + x.p, 0)).toBeCloseTo(1, 12)
        for (const x of o) expect(x.next >= 0 && x.next < m.states).toBe(true)
      }
    }
}

describe('bandit environments', () => {
  /** Arm a's reward in round t: the step stream child(root, t), whichever arm is pulled. */
  const pay = (env: ReturnType<typeof bernoulliBandit>, root: number, t: number, a: number) =>
    env.step(0, a, child(stream(root), t)).reward

  it('Bernoulli arms pay 0 or 1 at their means; the same stream gives the same rewards', () => {
    const env = bernoulliBandit({ means: [0.2, 0.7] })
    expect(env.action).toMatchObject({ kind: 'discrete', n: 2 })
    expect(env.observation).toMatchObject({ kind: 'discrete', n: 1 })
    expect(env.reset(stream(0))).toEqual({ state: 0, observation: 0 })
    expect([0, 1].map((a) => env.oracle!.expectedReward!(0, a))).toEqual([0.2, 0.7])
    const sums = [0, 0]
    for (let t = 0; t < 4000; t++)
      for (const a of [0, 1]) {
        const v = pay(env, 1, t, a)
        expect(v === 0 || v === 1).toBe(true)
        sums[a] += v
      }
    expect(sums[0] / 4000).toBeCloseTo(0.2, 1)
    expect(sums[1] / 4000).toBeCloseTo(0.7, 1)
    expect(pay(env, 2, 0, 1)).toBe(pay(env, 2, 0, 1))
    expect(() => bernoulliBandit({ means: [1.2] })).toThrow(DomainError)
  })

  it('Gaussian arms have the stated means and sds', () => {
    const env = gaussianBandit({ means: [0, 2], sd: [1, 0.5] })
    const draws = Array.from({ length: 4000 }, (_, t) => pay(env, 3, t, 1))
    const m = draws.reduce((a, b) => a + b, 0) / draws.length
    const sd = Math.sqrt(draws.reduce((a, b) => a + (b - m) ** 2, 0) / draws.length)
    expect(m).toBeCloseTo(2, 1)
    expect(sd).toBeCloseTo(0.5, 1)
  })

  it('a linear bandit pays xᵀθ* in expectation; fixed arms are unit vectors, random arms have lengths in [0.5, 1]', () => {
    const theta = [1, -0.5]
    const rows = (x: Float64Array) => Array.from({ length: x.length / 2 }, (_, a) => [x[2 * a], x[2 * a + 1]])
    const fixed = linearBandit({ theta, arms: 4 })
    const ctx = fixed.reset(stream(4)).observation
    expect(ctx.length).toBe(8)
    for (const x of rows(ctx)) expect(Math.hypot(...x)).toBeCloseTo(1, 12)
    expect(rows(ctx).map((_, a) => fixed.oracle!.expectedReward!(ctx, a))).toEqual(
      rows(ctx).map((x) => x[0] * theta[0] + x[1] * theta[1]),
    )
    const random = linearBandit({ theta, mode: 'random' })
    const a = random.reset(stream(5)).observation
    const b = random.reset(stream(6)).observation
    expect(a).not.toEqual(b)
    for (const x of rows(a)) expect(Math.hypot(...x) >= 0.5 - 1e-12 && Math.hypot(...x) <= 1 + 1e-12).toBe(true)
  })
})

describe('gridworlds', () => {
  it("Russell and Norvig's 4 × 3 world: a wall, two exits and sideways slips", () => {
    const g = gridworld()
    wellFormed(g)
    expect([g.states, g.actions, g.gamma]).toEqual([12, 4, 0.9])
    expect(g.terminalValue[cellState(4, 3, 2)]).toBe(1)
    expect(g.terminalValue[cellState(4, 3, 1)]).toBe(-1)
    // Moving up from (0, 0): 0.8 up to (0, 1), 0.1 right to (1, 0), 0.1 left into the edge (stay).
    const up = g.outcomes[cellState(4, 0, 0) * 4 + 0]
    const p = (next: number) => up.filter((o) => o.next === next).reduce((a, o) => a + o.p, 0)
    expect(p(cellState(4, 0, 1))).toBeCloseTo(0.8, 12)
    expect(p(cellState(4, 1, 0))).toBeCloseTo(0.1, 12)
    expect(p(cellState(4, 0, 0))).toBeCloseTo(0.1, 12)
  })

  it('cliff walking sends a fall back to the start at −100', () => {
    const c = cliffWalking()
    wellFormed(c)
    expect([c.states, c.start]).toEqual([48, 0])
    // Right (action 1) from the start (state 0) steps into the cliff; outcomes are indexed state · 4 + action.
    expect(c.outcomes[1]).toEqual([{ p: 1, next: 0, reward: -100 }])
    expect(c.terminal[11]).toBe(1)
  })

  it('mazes and frozen lakes parse their maps', () => {
    const m = maze(['..G', '.#T', 'S..'])
    wellFormed(m)
    expect(m.grid!.kinds.filter((k) => k === 'wall').length).toBe(1)
    expect(m.start).toBe(cellState(3, 0, 0))
    for (const map of ['4x4', '8x8'] as const) {
      const f = frozenLake({ map })
      wellFormed(f)
      const n = FROZEN_LAKE_MAPS[map].length
      expect(f.states).toBe(n * n)
      const holes = FROZEN_LAKE_MAPS[map].join('').split('H').length - 1
      expect(f.grid!.kinds.filter((k) => k === 'hole').length).toBe(holes)
    }
    // Slippery ice: the intended direction or either perpendicular one, 1/3 each.
    const f = frozenLake()
    const o = f.outcomes[f.start * 4 + 1]
    expect(o.map((x) => x.p).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(o.length).toBe(3)
    for (const x of o) expect(x.p).toBeCloseTo(1 / 3, 12)
    expect(frozenLake({ slippery: false }).outcomes[f.start * 4 + 1].length).toBe(1)
  })
})
