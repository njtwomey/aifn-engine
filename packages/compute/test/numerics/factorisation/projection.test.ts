/**
 * Random projections: the Johnson–Lindenstrauss dimension against scikit-learn's `johnson_lindenstrauss_min_dim`
 * (`fixtures/numerics/factorisation.json`), the entry laws of both matrices (mean 0, E[RᵀR] = I, sparse density and
 * values), and the lemma itself: at the JL dimension every squared distance of a point set stays within 1 ± ε.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  distanceDistortion,
  johnsonLindenstraussDimension,
  johnsonLindenstraussEpsilon,
  randomProjection,
  randomProjectionMatrix,
} from 'aifn-compute/numerics/factorisation'
import { fixture } from '../../fixtures'

const F = fixture<{ jl: { n: number; eps: number; k: number }[] }>('numerics/factorisation')

describe('johnsonLindenstraussDimension', () => {
  for (const c of F.jl)
    it(`matches scikit-learn for n = ${c.n}, ε = ${c.eps}`, () => {
      expect(johnsonLindenstraussDimension(c.n, c.eps)).toBe(c.k)
    })
  it('inverts the bound: the ε for k dimensions needs at most k, and a smaller ε needs more', () => {
    for (const [n, k] of [
      [100, 500],
      [1000, 2000],
      [10, 300],
    ]) {
      const e = johnsonLindenstraussEpsilon(n, k)
      expect(johnsonLindenstraussDimension(n, e)).toBeLessThanOrEqual(k)
      expect(johnsonLindenstraussDimension(n, e * 0.999)).toBeGreaterThan(k * 0.99)
    }
    expect(johnsonLindenstraussEpsilon(1000, 10)).toBeNaN()
  })

  it('rejects ε outside (0, 1)', () => {
    expect(() => johnsonLindenstraussDimension(10, 1)).toThrow()
    expect(() => johnsonLindenstraussDimension(10, 0)).toThrow()
  })
})

describe('randomProjectionMatrix', () => {
  it('Gaussian entries have mean 0 and variance 1/k', () => {
    const k = 50
    const v = toFlat(randomProjectionMatrix(400, k, stream(1)))
    const mean = v.reduce((a, b) => a + b, 0) / v.length
    const variance = v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length
    expect(Math.abs(mean)).toBeLessThan(0.005)
    expect(variance * k).toBeCloseTo(1, 1)
  })

  it('sparse entries take 0 or ±1/√(sk), with share s non-zero', () => {
    const k = 40
    const s = 1 / 3
    const v = toFlat(randomProjectionMatrix(300, k, stream(2), { kind: 'sparse', density: s }))
    const value = 1 / Math.sqrt(s * k)
    let nonzero = 0
    let positive = 0
    for (const u of v) {
      if (u === 0) continue
      nonzero++
      if (u > 0) positive++
      expect(Math.abs(u)).toBeCloseTo(value, 12)
    }
    expect(nonzero / v.length).toBeCloseTo(s, 1)
    expect(positive / nonzero).toBeCloseTo(0.5, 1)
  })

  it('preserves squared norms in expectation: mean ‖Rx‖² / ‖x‖² over draws ≈ 1', () => {
    const x = toFlat(normals(stream('x'), [100]))
    const norm = x.reduce((a, b) => a + b * b, 0)
    for (const kind of ['gaussian', 'sparse'] as const) {
      let total = 0
      const draws = 200
      for (let t = 0; t < draws; t++) {
        const { projected } = randomProjection([Array.from(x)], 20, stream(`d${t}`), { kind })
        total += toFlat(projected).reduce((a, b) => a + b * b, 0) / norm
      }
      expect(total / draws).toBeCloseTo(1, 1)
    }
  })
})

describe('the Johnson–Lindenstrauss lemma', () => {
  it('keeps every squared distance within 1 ± ε at the JL dimension', () => {
    const n = 30
    const eps = 0.5
    const k = johnsonLindenstraussDimension(n, eps)
    const X = normals(stream('points'), [n, 500])
    for (const kind of ['gaussian', 'sparse'] as const) {
      const { projected } = randomProjection(X, k, stream(kind), { kind })
      const d = distanceDistortion(X, projected)
      expect(d.ratios.length).toBe((n * (n - 1)) / 2)
      expect(d.maxDistortion).toBeLessThan(eps)
    }
  })

  it('identity map has zero distortion', () => {
    const X = normals(stream('id'), [6, 3])
    expect(distanceDistortion(X, X).maxDistortion).toBe(0)
  })
})
