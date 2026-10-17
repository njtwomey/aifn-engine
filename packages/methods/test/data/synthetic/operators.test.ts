/**
 * aifn-methods/data/synthetic operator datasets (the operator-learning notes' checks): the prior's mode-1 variance
 * (sample 0.310 against exact 0.301 over 4000 draws), the heat pairs' decay and grid values, the periodic Burgers
 * solver against the Cole–Hopf solution of `aifn-methods/neural/pinn` (error about 5e-8 at 128 points), and the
 * antiderivative pairs' trapezoid rule.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  antiderivativeData,
  burgersOperatorData,
  heatOperatorData,
  periodicGaussianField,
  priorVariance,
} from 'aifn-methods/data/synthetic'
import { evaluateSeries, periodicGrid, solveBurgersPeriodic } from 'aifn-methods/dynamics/pde'
import { burgersSolution } from 'aifn-methods/neural/pinn'

const data = (t: Tensor) => readonlyData(t) as Float64Array

describe('operator datasets', () => {
  it('draws mode-1 coefficients with the prior variance 2 sigma^2 ((2 pi)^2 + tau^2)^(-alpha)', () => {
    const fields = periodicGaussianField(stream('check/prior'), 4000, { modes: 4 })
    const v = fields.reduce((a, f) => a + f.cos[0] ** 2, 0) / fields.length
    expect(priorVariance(1)).toBeCloseTo(0.301, 3)
    expect(Math.abs(v / priorVariance(1) - 1)).toBeLessThan(0.06)
    expect(fields.every((f) => f.mean === 0)).toBe(true)
  })

  it('pairs heat inputs with their exact solutions: mode 1 decays by 0.673825', () => {
    const d = heatOperatorData(stream('check/heat'), { count: 3, grid: 16, diffusivity: 0.1, time: 0.1 })
    const s = d.series[0]
    expect(Math.abs(s.output.sin[0] / s.input.sin[0] - Math.exp(-0.1 * 4 * Math.PI ** 2 * 0.1))).toBeLessThan(1e-12)
    const direct = evaluateSeries(s.output, periodicGrid(16))
    const row = data(d.outputs).subarray(0, 16)
    row.forEach((v, i) => expect(Math.abs(v - direct[i])).toBeLessThan(1e-12))
  })

  it('solves Burgers as Cole-Hopf does (mapped from [-1, 1], nu/2, tau = t/2) to 1e-6', () => {
    const M = 128
    const nu = 0.05
    const tau = 0.25
    const xs = periodicGrid(M)
    const u0 = fromData(
      Float64Array.from(xs, (x) => Math.sin(2 * Math.PI * x)),
      [1, M],
    )
    const u = data(solveBurgersPeriodic(u0, { viscosity: nu / 2, time: tau }))
    let err = 0
    xs.forEach((x, j) => (err = Math.max(err, Math.abs(u[j] - burgersSolution(2 * x - 1, 2 * tau, nu)))))
    expect(err).toBeLessThan(1e-6)
  })

  it('subsamples Burgers pairs from the solver grid', () => {
    const d = burgersOperatorData(stream(3), { count: 2, grid: 16, viscosity: 0.05, time: 0.1, solverGrid: 32 })
    expect(d.inputs.shape).toEqual([2, 16])
    expect(d.fine.outputs.shape).toEqual([2, 32])
    expect(data(d.outputs)[17]).toBe(data(d.fine.outputs)[34])
    expect(() => burgersOperatorData(stream(3), { count: 1, grid: 12, viscosity: 0.1, time: 0.1 })).toThrow(/divide/)
  })

  it('integrates the antiderivative by the trapezoid rule from G(0) = 0', () => {
    const d = antiderivativeData(stream('check/anti'), { count: 2, sensors: 11, refine: 4 })
    const u = data(d.u)
    const G = data(d.G)
    const dx = 1 / (d.x.length - 1)
    expect(G[0]).toBe(0)
    expect(Math.abs((G[10] - G[9]) / dx - (u[9] + u[10]) / 2)).toBeLessThan(1e-12)
    expect(data(d.sensorValues)[3]).toBe(u[d.sensorIndex[3]])
  })
})
