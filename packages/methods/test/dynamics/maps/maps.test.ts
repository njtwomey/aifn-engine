import { describe, expect, it } from 'vitest'
import {
  bifurcationDiagram,
  cobweb,
  henonMap,
  logisticMap,
  lyapunovCurve,
  lyapunovExponent,
  lyapunovSpectrum,
  mapIteration,
  orbit,
  standardMap,
  tentMap,
} from 'aifn-methods/dynamics/maps'
import { mul, sub, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'

describe('iteration and orbits', () => {
  it('iterates the logistic map to its fixed point and 2-cycle', () => {
    const r = 2.8
    const s = run(mapIteration(logisticMap(r)), { x0: 0.2 }, 500)
    expect(s.x).toBeCloseTo(1 - 1 / r, 12)
    const o = toFlat(orbit(logisticMap(3.2), 0.2, 4, { discard: 1000 }))
    expect(o[0]).toBeCloseTo(o[2], 12)
    expect(o[1]).toBeCloseTo(o[3], 12)
    expect(Math.abs(o[0] - o[1])).toBeGreaterThan(0.1)
  })

  it('iterates the Hénon map onto its attractor', () => {
    const o = orbit(henonMap(), [0, 0], 100, { discard: 100 })
    expect(o.shape).toEqual([100, 2])
    toRows(o).forEach(([x, y]) => {
      expect(Math.abs(x)).toBeLessThan(1.5)
      expect(Math.abs(y)).toBeLessThan(0.5)
    })
  })

  it('draws a cobweb path', () => {
    const c = cobweb(logisticMap(2), 0.1, 2)
    expect(toFlat(c.x)).toEqual([
      0.1,
      0.1,
      expect.closeTo(0.18, 14),
      expect.closeTo(0.18, 14),
      expect.closeTo(0.2952, 14),
    ])
    expect(toFlat(c.y)[0]).toBe(0)
  })

  it('flags divergence', () => {
    const tr = trace(mapIteration({ f: (x: number) => x * x }), { x0: 10 }, 20)
    expect(tr.meta.stopped).toBe('diverged')
  })
})

describe('bifurcations and Lyapunov exponents', () => {
  it('gives ln 2 for the logistic map at r = 4 and ln μ for the tent map', () => {
    expect(lyapunovExponent(logisticMap(4), 0.1234, { keep: 200_000 }).exponent).toBeCloseTo(Math.LN2, 2)
    expect(lyapunovExponent(tentMap(1.9), 0.1234, { keep: 20_000 }).exponent).toBeCloseTo(Math.log(1.9), 10)
    // A stable fixed point has λ = ln|f′(x*)| = ln|2 − r|.
    expect(lyapunovExponent(logisticMap(2.5), 0.3).exponent).toBeCloseTo(Math.log(0.5), 8)
    const curve = toFlat(lyapunovCurve(logisticMap, [2.9, 3.83, 3.9]))
    expect(curve[0]).toBeLessThan(0)
    expect(curve[1]).toBeLessThan(0) // the period-3 window
    expect(curve[2]).toBeGreaterThan(0)
  })

  it('autodiff supplies a missing derivative', () => {
    // Written with primitives, so autodiff can differentiate it.
    const map = { f: (x: number) => mul(4, mul(x, sub(1, x))) }
    const a = lyapunovExponent(map, 0.3, { keep: 100 }).exponent
    const b = lyapunovExponent(logisticMap(4), 0.3, { keep: 100 }).exponent
    expect(a).toBeCloseTo(b, 10)
  })

  it('builds a bifurcation diagram with period doubling', () => {
    const d = bifurcationDiagram(logisticMap, [2.9, 3.2, 3.5], { keep: 16 })
    const at = (r: number) =>
      new Set(
        toFlat(d.x)
          .filter((_, i) => toFlat(d.r)[i] === r)
          .map((x) => x.toFixed(6)),
      ).size
    expect([at(2.9), at(3.2), at(3.5)]).toEqual([1, 2, 4])
  })

  it('Hénon and standard map spectra sum to the mean log-determinant', () => {
    const h = toFlat(lyapunovSpectrum(henonMap(), [0.1, 0.1], { keep: 20_000 }).exponents)
    expect(h[0]).toBeCloseTo(0.419, 1)
    expect(h[0] + h[1]).toBeCloseTo(Math.log(0.3), 8)
    const s = toFlat(lyapunovSpectrum(standardMap(5), [1, 1], { keep: 5000 }).exponents)
    expect(s[0] + s[1]).toBeCloseTo(0, 8)
    expect(s[0]).toBeGreaterThan(0.5)
  })
})

describe('trace protocol', () => {
  it('mapIteration follows the protocol for 1-D and 2-D maps', () => {
    expectProtocol(mapIteration(henonMap()), { x0: [0.1, 0] }, { n: 50 })
    expectProtocol(mapIteration(logisticMap(3.7)), { x0: 0.2 }, { n: 20 })
  })
})
