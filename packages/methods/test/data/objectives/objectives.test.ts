import { describe, expect, it } from 'vitest'
import { beale, himmelblau, quadraticBowl, rastrigin, rosenbrock } from 'aifn-methods/data/objectives'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'

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
})
