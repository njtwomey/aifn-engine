import { describe, expect, it } from 'vitest'
import {
  beale,
  branin,
  forrester,
  himmelblau,
  quadraticBowl,
  rastrigin,
  rosenbrock,
  svmTuningSurface,
} from 'aifn-methods/data/objectives'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'

describe('test functions', () => {
  it('gradients and Hessians match finite differences', () => {
    const fns = [
      rosenbrock(),
      rosenbrock({ n: 4 }),
      himmelblau(),
      beale(),
      quadraticBowl({ condition: 30, angle: 0.4 }),
      rastrigin(),
      forrester(),
      forrester({ fidelity: 'low' }),
      branin(),
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

describe('Bayesian optimisation test functions', () => {
  it('forrester: the minima of both fidelities, against scipy', () => {
    // scipy.optimize.minimize_scalar (bounded, xatol 1e-14) near the best of 100001 grid points.
    const hi = forrester()
    expect(toFlat(hi.minima[0])[0]).toBeCloseTo(0.7572487579285088, 7)
    expect(hi.minimumValue).toBeCloseTo(-6.0207400557670825, 12)
    const lo = forrester({ fidelity: 'low' })
    expect(toFlat(lo.minima[0])[0]).toBeCloseTo(0.0923928771562138, 7)
    expect(lo.minimumValue).toBeCloseTo(0.6650951230463615, 12)
    // The low fidelity is a f + b (x - 0.5) - c.
    for (const x of [0, 0.3, 0.8, 1])
      expect(lo.value(tensor([x]))).toBeCloseTo(0.5 * hi.value(tensor([x])) + 10 * (x - 0.5) + 5, 12)
    expect(hi.value(tensor([1]))).toBeCloseTo(15.829731945974109, 12)
    // A tilt that pushes the minimum to the boundary.
    expect(toFlat(forrester({ fidelity: 'low', b: 100 }).minima[0])[0]).toBe(0)
  })

  it('branin: three minima of value 10 / 8 pi, with zero gradient', () => {
    const f = branin()
    expect(f.minima).toHaveLength(3)
    expect(f.minimumValue).toBeCloseTo(0.397887357729738, 14)
    for (const m of f.minima) {
      expect(f.value(m)).toBeCloseTo(f.minimumValue, 13)
      for (const g of toFlat(f.objective(m).grad as never)) expect(Math.abs(g)).toBeLessThan(1e-12)
    }
    expect(toFlat(f.minima[1])).toEqual([Math.PI, 2.275])
  })
})

describe('svmTuningSurface', () => {
  const f = svmTuningSurface()
  const error = toArray(f.error) as number[][]
  const logC = toFlat(f.logC)
  const logGamma = toFlat(f.logGamma)

  it('reproduces the table at the nodes and is bilinear between them', () => {
    expect(f.error.shape).toEqual([21, 21])
    for (const [i, j] of [
      [0, 0],
      [7, 13],
      [20, 20],
      [12, 4],
    ])
      expect(f.value(tensor([logC[i], logGamma[j]]))).toBe(error[i][j])
    const mid = f.value(tensor([(logC[3] + logC[4]) / 2, (logGamma[9] + logGamma[10]) / 2]))
    expect(mid).toBeCloseTo((error[3][9] + error[4][9] + error[3][10] + error[4][10]) / 4, 14)
    // Outside the box: the nearest point of the box.
    expect(f.value(tensor([-10, 10]))).toBe(error[0][20])
  })

  it('has its minima at the nodes of least error, with the gradient and Hessian of the interpolant', () => {
    const lowest = Math.min(...error.flat())
    expect(f.minimumValue).toBe(lowest)
    for (const m of f.minima) expect(f.value(m)).toBe(lowest)
    const x = [0.37, -1.18]
    const h = 1e-7
    const g = toFlat(f.objective(tensor(x)).grad as never)
    expect((f.value(tensor([x[0] + h, x[1]])) - f.value(tensor([x[0] - h, x[1]]))) / (2 * h)).toBeCloseTo(g[0], 6)
    expect((f.value(tensor([x[0], x[1] + h])) - f.value(tensor([x[0], x[1] - h]))) / (2 * h)).toBeCloseTo(g[1], 6)
    const gu = toFlat(f.objective(tensor([x[0], x[1] + h])).grad as never)
    const gd = toFlat(f.objective(tensor([x[0], x[1] - h])).grad as never)
    expect((gu[0] - gd[0]) / (2 * h)).toBeCloseTo(toFlat(f.hessian(tensor(x)))[1], 5)
  })

  it('adds noise from a stream, reproducibly', () => {
    const noisy = svmTuningSurface({ noise: 0.01 })
    const x = tensor([0.1, -1.1])
    const a = noisy.value(x, stream(1))
    expect(noisy.value(x, stream(1))).toBe(a)
    expect(a).not.toBe(noisy.mean(x))
    expect(noisy.mean(x)).toBe(f.value(x))
    expect(() => noisy.value(x)).toThrow(DomainError)
    const own = svmTuningSurface({ noise: 0.01, stream: stream(2) })
    expect(own.value(x)).not.toBe(own.value(x))
    expect(() => svmTuningSurface({ noise: -1 })).toThrow(DomainError)
    expect(() => f.value(tensor([1, 2, 3]))).toThrow(ShapeError)
  })
})
