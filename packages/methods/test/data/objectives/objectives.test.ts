import { describe, expect, it } from 'vitest'
import { beale, himmelblau, quadraticBowl, rastrigin, rosenbrock } from 'aifn-methods/data/objectives'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

describe('test functions', () => {
  it('gradients and Hessians match finite differences', () => {
    const fns = [
      rosenbrock(),
      rosenbrock({ n: 4 }),
      himmelblau(),
      beale(),
      quadraticBowl({ condition: 30, angle: 0.4 }),
      rastrigin(),
    ]
    for (const fn of fns) {
      const x = Array.from({ length: fn.dimension }, (_, i) => 0.3 + 0.17 * i)
      const { grad } = fn.objective(tensor(x))
      const H = toFlat(fn.hessian(tensor(x)))
      const g = toFlat(grad as never)
      const h = 1e-6
      for (let i = 0; i < x.length; i++) {
        const up = [...x]
        const down = [...x]
        up[i] += h
        down[i] -= h
        expect((fn.value(tensor(up)) - fn.value(tensor(down))) / (2 * h)).toBeCloseTo(g[i], 4)
        const gu = toFlat(fn.objective(tensor(up)).grad as never)
        const gd = toFlat(fn.objective(tensor(down)).grad as never)
        for (let j = 0; j < x.length; j++) expect((gu[j] - gd[j]) / (2 * h)).toBeCloseTo(H[i * x.length + j], 3)
      }
      for (const m of fn.minima) expect(fn.value(m)).toBeCloseTo(fn.minimumValue, 10)
    }
  })

  it('rosenbrock minima for a=0, a=1 and a not in {0, 1}', () => {
    const r0 = rosenbrock({ a: 0, n: 3 })
    expect(r0.minima).toHaveLength(1)
    expect(Array.from(toFlat(r0.minima[0]))).toEqual([0, 0, 0])
    expect(r0.minimumValue).toBe(0)

    const r1 = rosenbrock({ a: 1, n: 3 })
    expect(r1.minima).toHaveLength(1)
    expect(Array.from(toFlat(r1.minima[0]))).toEqual([1, 1, 1])
    expect(r1.minimumValue).toBe(0)

    const r2 = rosenbrock({ a: 2, n: 3 })
    expect(r2.minima).toHaveLength(0)
    expect(Number.isNaN(r2.minimumValue)).toBe(true)
  })

  it('quadraticBowl validates center length against dimension n', () => {
    expect(() => quadraticBowl({ n: 3, center: [1, 2] })).toThrow(ShapeError)
    const valid = quadraticBowl({ n: 2, center: [1, 2] })
    expect(valid.dimension).toBe(2)
  })
})
