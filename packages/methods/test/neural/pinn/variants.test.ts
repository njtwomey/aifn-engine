/**
 * aifn-methods/neural/pinn variants (the physics-informed notes' checks): a hard-constrained heat network meets its
 * initial and boundary values to rounding at every snapshot, the problems' lifts and distances meet and vanish on the
 * conditions, `heatProblem` is `heat`, balanced weights stop at `maxWeight`, and a fitted coefficient is reported at
 * every step. (The notes' training probes take seconds, so they are not unit tests.)
 */
import { describe, expect, it } from 'vitest'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  convectionProblem,
  heat,
  heatProblem,
  layerProblem,
  pinnVariantRun,
  poissonProblem,
  type PinnVariantSnapshot,
} from 'aifn-methods/neural/pinn'

const all = (g: Generator<PinnVariantSnapshot, PinnVariantSnapshot>) => {
  const out: PinnVariantSnapshot[] = []
  let r = g.next()
  while (!r.done) {
    out.push(r.value)
    r = g.next()
  }
  out.push(r.value)
  return out
}
const flat = (v: unknown) => toFlat(v as Tensor)

describe('PINN variants', () => {
  it('meets the heat conditions exactly with hard constraints, at every snapshot', () => {
    const runs = all(pinnVariantRun(heatProblem(0.1), { constraint: 'hard', adamSteps: 20, lbfgsSteps: 0, every: 10 }))
    expect(runs.at(-1)!.finished).toBe(true)
    expect(runs.at(-1)!.terms).toEqual(['residual'])
    runs.forEach((r) => expect(r.conditionError).toBeLessThan(1e-12))
  })

  it('builds lifts that meet the conditions and distances that vanish on them', () => {
    expect(heatProblem).toBe(heat)
    const p = heatProblem(0.1, 2)
    const X = tensor([
      [-1, 0.3],
      [0.5, 0],
      [1, 0.7],
      [0, 0.5],
    ])
    const lift = flat(p.lift!(X))
    expect(Math.abs(lift[0]) + Math.abs(lift[2])).toBeLessThan(1e-15)
    expect(lift[1]).toBeCloseTo(1, 15)
    expect(flat(p.distance!(X))).toEqual([0, 0, 0, 1])
    const q = poissonProblem(4)
    expect(flat(q.lift!(tensor([[0], [1]])))).toEqual([0, 1])
    expect(flat(q.distance!(tensor([[0], [0.5], [1]])))).toEqual([0, 1, 0])
    const l = layerProblem(0.02)
    const ends = flat(l.lift!(tensor([[0], [1]])))
    expect(ends[0]).toBeCloseTo(l.solution!(0, 0), 14)
    expect(ends[1]).toBeCloseTo(l.solution!(1, 0), 14)
    expect(convectionProblem(10).boundary(0.3)[0]).toBeCloseTo(Math.sin(-3), 15)
  })

  it('caps the gradient-norm weights at maxWeight', () => {
    const runs = all(
      pinnVariantRun(layerProblem(0.01), {
        hidden: [8],
        collocation: 16,
        balancing: 'gradient-norm',
        balanceEvery: 5,
        maxWeight: 10,
        adamSteps: 60,
        lbfgsSteps: 0,
        every: 60,
        grid: [21, 1],
      }),
    )
    const last = runs.at(-1)!
    expect(Math.max(...last.weights.boundary)).toBeLessThanOrEqual(10)
    expect(last.loss.length).toBe(60)
  })

  it('fits a coefficient from observations and reports its estimate at every step', () => {
    const p = heatProblem(0.1)
    const x = [-0.5, 0, 0.5, -0.25, 0.25]
    const t = [0.2, 0.4, 0.6, 0.8, 1]
    const y = x.map((v, i) => p.solution!(v, t[i]))
    const runs = all(
      pinnVariantRun(p, {
        hidden: [8],
        collocation: 32,
        initialPoints: 8,
        boundaryPoints: 4,
        trainable: { kappa: 0.3 },
        data: { x, t, y },
        adamSteps: 20,
        lbfgsSteps: 0,
        every: 20,
        grid: [11, 3],
      }),
    )
    const last = runs.at(-1)!
    expect(last.terms).toEqual(['residual', 'initial', 'boundary', 'data'])
    expect(last.coefficients.kappa.length).toBe(20)
    expect(last.coefficients.kappa.every((k) => k > 0)).toBe(true)
  })
})
