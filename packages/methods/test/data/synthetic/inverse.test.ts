/**
 * The inverse-problem generators (`bishopInverse`, `twoLinkArm`): seeded determinism, the data against the forward
 * map, and the truth: every solution maps back to x, the fold has three branches, p(t | x) integrates to one, the
 * conditional mean lies between the branches, and both elbow solutions reach the hand position.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { datasetRegistry, type InverseTruth } from 'aifn-methods/data'
import { bishopInverse, twoLinkArm, twoLinkInverse, twoLinkJoints } from 'aifn-methods/data/synthetic'

const flat = (t: Tensor) => Array.from(toFlat(t))
const truthOf = (d: { meta: { truth?: unknown } }) => d.meta.truth as InverseTruth

describe('bishopInverse', () => {
  const data = bishopInverse(stream('b'), { n: 200 })
  const truth = truthOf(data)

  it('is registered and seeded: the same stream gives the same data, another stream different data', () => {
    expect(datasetRegistry.bishopInverse.info.truth).toBe(true)
    expect(flat(bishopInverse(stream('b'), { n: 200 }).x)).toEqual(flat(data.x))
    expect(flat(bishopInverse(stream('c'), { n: 200 }).x)).not.toEqual(flat(data.x))
    expect(data.x.shape).toEqual([200, 1])
    expect(data.y!.shape).toEqual([200])
  })

  it('observes t through x = t + 0.3 sin 2πt + ε, with residuals of sd σ', () => {
    const x = flat(data.x)
    const t = flat(data.y!)
    const r = x.map((v, i) => v - (t[i] + 0.3 * Math.sin(2 * Math.PI * t[i])))
    const sd = Math.sqrt(r.reduce((a, v) => a + v * v, 0) / r.length)
    expect(sd).toBeGreaterThan(0.04)
    expect(sd).toBeLessThan(0.06)
    expect(Math.min(...t)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...t)).toBeLessThanOrEqual(1)
  })

  it('has three solutions in the fold and one at the ends, each mapping back to x', () => {
    expect(truth.solutions([0.5]).length).toBe(3)
    expect(
      truth
        .solutions([0.5])
        .map((s) => s.value[0])
        .sort(),
    ).toEqual([expect.closeTo(0.21, 2), expect.closeTo(0.5, 6), expect.closeTo(0.79, 2)])
    expect(truth.solutions([0.05]).length).toBe(1)
    expect(truth.solutions([0.95]).length).toBe(1)
    for (const x of [0.05, 0.3, 0.5, 0.62, 0.95])
      for (const s of truth.solutions([x])) expect(truth.forward(s.value)[0]).toBeCloseTo(x, 9)
    const w = truth.solutions([0.5]).map((s) => s.weight)
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('p(t | x) integrates to one, and the conditional mean falls between the branches', () => {
    for (const x of [0.1, 0.5, 0.8]) {
      const grid = Array.from({ length: 4000 }, (_, i) => (i + 0.5) / 4000)
      const lp = flat(truth.logLikelihood(tensor(grid.map(() => [x])), tensor(grid)))
      expect(lp.reduce((a, v) => a + Math.exp(v), 0) / grid.length).toBeCloseTo(1, 3)
    }
    // At x = 0.5 the branches are symmetric about t = 0.5; the mean sits on the middle one.
    expect(flat(truth.mean(tensor([[0.5]])))[0]).toBeCloseTo(0.5, 6)
    // At x = 0.42 the mean is on no branch.
    const m = flat(truth.mean(tensor([[0.42]])))[0]
    const sol = truth.solutions([0.42]).map((s) => s.value[0])
    expect(sol.length).toBe(3)
    expect(Math.min(...sol.map((s) => Math.abs(s - m)))).toBeGreaterThan(0.04)
    expect(truth.bayesRisk).toBeGreaterThan(0)
    expect(truth.bayesRisk).toBeLessThan(1 / 12)
  })
})

describe('twoLinkArm', () => {
  const data = twoLinkArm(stream('arm'), { n: 300 })
  const truth = truthOf(data)

  it('is registered and seeded; the hand is the forward kinematics of the angles plus noise', () => {
    expect(datasetRegistry.twoLinkArm.info.truth).toBe(true)
    expect(flat(twoLinkArm(stream('arm'), { n: 300 }).y!)).toEqual(flat(data.y!))
    expect(data.x.shape).toEqual([300, 2])
    expect(data.y!.shape).toEqual([300, 2])
    const p = flat(data.x)
    const a = flat(data.y!)
    for (let i = 0; i < 300; i++) {
      const { hand } = twoLinkJoints([a[2 * i], a[2 * i + 1]], [0.8, 0.5])
      expect(Math.hypot(hand[0] - p[2 * i], hand[1] - p[2 * i + 1])).toBeLessThan(0.06)
    }
  })

  it('has both elbow solutions, which reach the hand and bend opposite ways', () => {
    const target = [0.6, 0.6]
    const both = twoLinkInverse(target, [0.8, 0.5])
    expect(both.length).toBe(2)
    expect(both[0][1]).toBeCloseTo(-both[1][1], 12)
    for (const s of both) {
      const { hand } = twoLinkJoints(s, [0.8, 0.5])
      expect(hand[0]).toBeCloseTo(0.6, 10)
      expect(hand[1]).toBeCloseTo(0.6, 10)
    }
    const sol = truth.solutions(target)
    expect(sol.length).toBe(2)
    expect(sol.map((s) => s.weight)).toEqual([0.5, 0.5])
    // Out of reach: no solution.
    expect(twoLinkInverse([2, 0], [0.8, 0.5])).toEqual([])
    expect(truth.solutions([2, 0])).toEqual([])
  })

  it('the conditional mean of the angles reaches neither solution', () => {
    const target = [0.6, 0.6]
    const m = flat(truth.mean(tensor([target])))
    const { hand } = twoLinkJoints(m, [0.8, 0.5])
    expect(Math.hypot(hand[0] - 0.6, hand[1] - 0.6)).toBeGreaterThan(0.2)
  })
})
