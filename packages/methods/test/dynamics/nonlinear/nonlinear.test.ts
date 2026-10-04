import { describe, expect, it } from 'vitest'
import { limitCycle, lyapunovCheck, poincareSection } from 'aifn-methods/dynamics/nonlinear'
import { mul, sum, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

// The Hopf normal form in Cartesian coordinates: ṙ = r(1 − r²), θ̇ = 1. The unit circle is a stable limit cycle of
// period 2π whose Floquet multiplier is exp(−2 · 2π) (the linearisation of ṙ at r = 1 is −2).
const hopf = (x: Tensor) => {
  const [a, b] = toFlat(x)
  const s = 1 - a * a - b * b
  return [a * s - b, b * s + a]
}
const section = { point: [0, 0], normal: [0, 1] }

describe('limit cycles and sections', () => {
  it('finds the unit circle of the Hopf normal form with period 2π', () => {
    const c = limitCycle(hopf, [0.3, 0.1], section)
    expect(c.converged).toBe(true)
    const [px, py] = toFlat(c.point)
    expect(px).toBeCloseTo(1, 6)
    expect(py).toBeCloseTo(0, 8)
    expect(c.period).toBeCloseTo(2 * Math.PI, 6)
    expect(Math.abs(c.multiplier)).toBeLessThan(1e-3)
    const orbit = toFlat(c.orbit)
    for (let i = 0; i < orbit.length; i += 2) expect(Math.hypot(orbit[i], orbit[i + 1])).toBeCloseTo(1, 4)
  })

  it('successive crossings approach the cycle and return every 2π', () => {
    const r = poincareSection(hopf, [2, 0.01], section, { crossings: 5 })
    const pts = toFlat(r.points)
    expect(r.points.shape).toEqual([5, 2])
    // The multiplier is e^{−4π} ≈ 3.5e-6: one return brings a start at r = 2 onto the circle.
    for (let k = 0; k < 5; k++) {
      expect(pts[2 * k]).toBeCloseTo(1, 5)
      expect(pts[2 * k + 1]).toBeCloseTo(0, 9)
    }
    toFlat(r.returnTimes).forEach((t) => expect(t).toBeCloseTo(2 * Math.PI, 6))
  })
})

// Van der Pol, ẍ − μ(1 − x²)ẋ + x = 0 with μ = 1: a stable relaxation cycle whose period, 6.6632868593…, is a
// reference value (Strogatz, 2015, §7.5; computed to 10 digits by shooting with a high-order integrator).
const vanDerPol = (x: Tensor) => {
  const [a, b] = toFlat(x)
  return [b, (1 - a * a) * b - a]
}

describe('the Van der Pol oscillator (μ = 1)', () => {
  it('has the golden period 6.6632868593 and a stable multiplier', () => {
    const c = limitCycle(vanDerPol, [0.5, 0], { point: [0, 0], normal: [1, 0] }, { tolerance: 1e-11 })
    expect(c.converged).toBe(true)
    expect(Math.abs(c.period - 6.6632868593)).toBeLessThan(1e-7)
    // The cycle crosses x = 0 going right at ẋ = 2.1727 (amplitude about 2).
    expect(toFlat(c.point)[1]).toBeCloseTo(2.1727, 3)
    expect(Math.abs(c.multiplier)).toBeLessThan(1)
    const xs = toFlat(c.orbit).filter((_, i) => i % 2 === 0)
    expect(Math.max(...xs)).toBeCloseTo(2.0086, 2)
  })
})

describe('lyapunovCheck', () => {
  const grid = { x: [-1, 1], y: [-1, 1], nx: 21, ny: 21 } as const
  // V is differentiated by the check, so it is written with tensor operations.
  const V = (x: Tensor) => sum(mul(x, x))
  it('certifies V = x² + y² for a stable linear system', () => {
    const c = lyapunovCheck(V, (x) => toFlat(x).map((v) => -v), [0, 0], grid)
    expect(c.positiveDefinite).toBe(true)
    expect(c.nonIncreasing).toBe(true)
    expect(c.decreasing).toBe(true)
  })
  it('rejects it for a centre (V̇ = 0) as asymptotic and for a source as stable', () => {
    const centre = lyapunovCheck(V, (x) => [-toFlat(x)[1], toFlat(x)[0]], [0, 0], grid)
    expect(centre.nonIncreasing).toBe(true)
    expect(centre.decreasing).toBe(false)
    const source = lyapunovCheck(V, (x) => toFlat(x), [0, 0], grid)
    expect(source.nonIncreasing).toBe(false)
  })
})
