import { describe, expect, it } from 'vitest'
import { banana, funnel, gaussianMixtureTarget, gaussianTarget, nonCentredFunnel } from 'aifn-methods/data/targets'
import type { LogDensity } from 'aifn-compute/foundation/contracts'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Normal } from 'aifn-compute/probability/distributions'

const value = (t: LogDensity, v: number[]) => {
  const r = t.logDensity(tensor(v))
  return typeof r === 'number' ? r : toFlat(r as Tensor)[0]
}

describe('targets', () => {
  const std2 = gaussianTarget(
    [0, 0],
    [
      [1, 0.5],
      [0.5, 1],
    ],
  )
  it.each([
    ['banana', banana({ a: 1.2, b: 0.7 })],
    ['gaussian', std2],
    [
      'mixture',
      gaussianMixtureTarget(
        [
          [-1, 0],
          [2, 1],
        ],
        0.8,
        [1, 2],
      ),
    ],
    ['funnel', funnel({ dim: 3 })],
    ['non-centred funnel', nonCentredFunnel({ dim: 3 })],
  ] as [string, LogDensity][])('%s: analytic gradient matches finite differences', (_, t) => {
    const x = Array.from({ length: t.dim }, (_, i) => 0.3 + 0.2 * i)
    if (!t.grad) return
    const g = Array.from(t.grad(tensor(x)) as ArrayLike<number>)
    for (let i = 0; i < t.dim; i++) {
      const up = [...x]
      const dn = [...x]
      up[i] += 1e-5
      dn[i] -= 1e-5
      expect(g[i]).toBeCloseTo((value(t, up) - value(t, dn)) / 2e-5, 5)
    }
  })

  it('gaussianTarget is normalised', () => {
    // At the mean, log N = −½ log|2πΣ| with |Σ| = 0.75.
    expect(value(std2, [0, 0])).toBeCloseTo(-Math.log(2 * Math.PI) - 0.5 * Math.log(0.75), 12)
  })

  it('nonCentredFunnel is the funnel through (v, z) ↦ (v, z e^{v/2}): N(0, s²) × N(0, 1)^(d − 1), same values', () => {
    const nc = nonCentredFunnel({ dim: 3, scale: 2 })
    const centred = funnel({ dim: 3, scale: 2 })
    for (const u of [
      [0.3, -1, 2],
      [-4, 0.5, 0.1],
      [2.5, 1, -1],
    ]) {
      const want =
        (Normal(0, 2).logProb(u[0]) as number) + u.slice(1).reduce((a, z) => a + (Normal(0, 1).logProb(z) as number), 0)
      expect(value(nc, u)).toBeCloseTo(want, 12)
      // The change of variables: log p_u(u) = log p_θ(T(u)) + (d − 1) v/2.
      const theta = Array.from(toFlat(nc.toOriginal(tensor(u)) as Tensor))
      expect(value(nc, u)).toBeCloseTo(value(centred, theta) + u[0], 12)
      expect(Array.from(toFlat(nc.fromOriginal!(tensor(theta)) as Tensor))).toEqual(
        u.map((x) => expect.closeTo(x, 12) as unknown as number),
      )
    }
    expect(nc.normalised).toBe(true)
    expect(nc.dim).toBe(3)
  })
})
