/**
 * aifn-methods/dynamics/pde periodic spectral solutions: the heat equation's mode-1 decay $e^{-0.1 \cdot 4\pi^2 \cdot
 * 0.1} = 0.673825$, series on a grid, and the Burgers solver's laws (a small profile diffuses as the heat equation,
 * the mean is conserved, bad input is refused). Its comparison with Cole–Hopf is in the operator datasets' test.
 */
import { describe, expect, it } from 'vitest'
import { fromData, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'
import { evaluateSeries, heatSolution, periodicGrid, solveBurgersPeriodic } from 'aifn-methods/dynamics/pde'

const data = (t: Tensor) => readonlyData(t) as Float64Array

describe('periodic spectral solutions', () => {
  it('decays mode k of the heat equation by exp(-D (2 pi k)^2 t)', () => {
    const f = { mean: 0.5, cos: Float64Array.of(1, 0), sin: Float64Array.of(1, 2) }
    const g = heatSolution(f, 0.1, 0.1)
    expect(g.sin[0] / f.sin[0]).toBeCloseTo(Math.exp(-0.1 * 4 * Math.PI ** 2 * 0.1), 12)
    expect(Math.exp(-0.1 * 4 * Math.PI ** 2 * 0.1)).toBeCloseTo(0.673825, 6)
    expect(g.sin[1] / f.sin[1]).toBeCloseTo(Math.exp(-0.1 * 16 * Math.PI ** 2 * 0.1), 12)
    expect(g.mean).toBe(0.5)
  })

  it('evaluates a series on the periodic grid', () => {
    expect(Array.from(periodicGrid(4))).toEqual([0, 0.25, 0.5, 0.75])
    const v = evaluateSeries({ mean: 1, cos: Float64Array.of(0), sin: Float64Array.of(2) }, periodicGrid(4))
    ;[1, 3, 1, -1].forEach((want, i) => expect(v[i]).toBeCloseTo(want, 14))
  })

  it('diffuses a small Burgers profile as the heat equation and conserves the mean', () => {
    const M = 64
    const x = periodicGrid(M)
    const eps = 1e-4
    const u0 = fromData(
      Float64Array.from(x, (v) => 0.3 + eps * Math.sin(2 * Math.PI * v)),
      [1, M],
    )
    const nu = 0.02
    const t = 0.2
    const u = data(solveBurgersPeriodic(u0, { viscosity: nu, time: t }))
    const mean = u.reduce((a, b) => a + b, 0) / M
    expect(mean).toBeCloseTo(0.3, 12)
    // The mean advects the wave at speed 0.3; its amplitude decays as the heat equation's.
    const amp = Math.hypot(...Array.from(u, (v) => v - mean)) * Math.sqrt(2 / M)
    expect(Math.abs(amp / eps - Math.exp(-nu * 4 * Math.PI ** 2 * t))).toBeLessThan(1e-3)
  })

  it('refuses non-finite profiles and odd grids', () => {
    expect(() => solveBurgersPeriodic(fromData(Float64Array.of(0, NaN), [1, 2]), { viscosity: 1, time: 1 })).toThrow(
      /finite/,
    )
    expect(() => solveBurgersPeriodic(fromData(Float64Array.of(0, 1, 0), [1, 3]), { viscosity: 1, time: 1 })).toThrow(
      /even/,
    )
  })
})
