/**
 * Global effects by their laws: ALE's local differences are exact for polynomial effects, so the accumulated effect
 * follows the true shape under correlated features (where partial dependence does not need to), and its count-weighted
 * centre is zero; Friedman's H is zero for an additive model and one for a pure product; functional ANOVA of the
 * Ishigami function on a midpoint grid gives its analytic Sobol indices and orthogonal, centred terms.
 */
import { describe, expect, it } from 'vitest'
import { normal, stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { accumulatedLocalEffects, functionalAnova, hStatistic } from 'aifn-compute/learning/explain'

const rowsOf = (f: (r: Float64Array) => number) => (X: Tensor) => {
  const x = dense.data(X)
  const d = X.shape[1]
  return Float64Array.from({ length: X.shape[0] }, (_, i) => f(x.subarray(i * d, (i + 1) * d)))
}

function correlated(n: number) {
  const z = toFlat(normal(stream('ale'), 0, 1, { shape: [n * 3] }))
  return fromData(
    Float64Array.from({ length: n * 3 }, (_, k) => {
      const i = Math.floor(k / 3)
      const c = k % 3
      return c === 0 ? z[3 * i] : c === 1 ? z[3 * i] + 0.2 * z[3 * i + 1] : z[3 * i + 2]
    }),
    [n, 3],
  )
}

describe('accumulated local effects', () => {
  const X = correlated(500)
  const f = rowsOf((r) => r[0] ** 2 + 2 * r[1] + r[1] * r[2])
  it('accumulate exact local differences', () => {
    const a = accumulatedLocalEffects(f, X, 0, { bins: 10 })
    for (let k = 1; k < a.edges.length; k++)
      expect(a.effect[k] - a.effect[0]).toBeCloseTo(a.edges[k] ** 2 - a.edges[0] ** 2, 10)
    const b = accumulatedLocalEffects(f, X, 1, { bins: 10 })
    // x₁'s local change also carries x₁x₂'s, 2Δ + Δ × (mean x₂ in the bin).
    expect(b.local.length).toBe(b.counts.length)
    let centre = 0
    for (let k = 0; k < b.counts.length; k++) centre += ((b.effect[k] + b.effect[k + 1]) / 2) * b.counts[k]
    expect(centre).toBeCloseTo(0, 10)
    expect(b.counts.reduce((p, q) => p + q, 0)).toBe(500)
  })
})

describe('H-statistic', () => {
  // A symmetric design: every sign pattern of (±1, ±2), so the product has no main effect.
  const grid: number[][] = []
  for (const a of [-1, -0.5, 0.5, 1]) for (const b of [-2, -1, 1, 2]) grid.push([a, b, 0.3])
  it('is zero for an additive model', () => {
    const h = hStatistic(
      rowsOf((r) => r[0] ** 3 + Math.sin(r[1]) + r[2]),
      grid,
    )
    expect(h.pairwise[1]).toBeCloseTo(0, 10)
    expect(h.overall[0]).toBeCloseTo(0, 10)
  })
  it('is one for a pure product', () => {
    const h = hStatistic(
      rowsOf((r) => r[0] * r[1]),
      grid,
    )
    expect(h.pairwise[1]).toBeCloseTo(1, 10)
    expect(h.overall[0]).toBeCloseTo(1, 10)
    expect(h.pairwise[2]).toBeCloseTo(0, 10)
  })
})

describe('functional ANOVA', () => {
  const a = 7
  const b = 0.1
  const ishigami = rowsOf((r) => Math.sin(r[0]) + a * Math.sin(r[1]) ** 2 + b * r[2] ** 4 * Math.sin(r[0]))
  // Midpoint grids: exact for the periodic x₁ and x₂ terms; fine for x₃'s polynomial.
  const midpoints = (g: number) => Float64Array.from({ length: g }, (_, k) => -Math.PI + ((k + 0.5) * 2 * Math.PI) / g)
  const g = 40
  const r = functionalAnova(ishigami, [midpoints(g), midpoints(g), midpoints(300)])
  it('gives the Ishigami function’s Sobol indices', () => {
    const V = a ** 2 / 8 + (b * Math.PI ** 4) / 5 + (b ** 2 * Math.PI ** 8) / 18 + 0.5
    const V1 = 0.5 * (1 + (b * Math.PI ** 4) / 5) ** 2
    const V2 = a ** 2 / 8
    const V13 = (b ** 2 * Math.PI ** 8) / 18 - (b ** 2 * Math.PI ** 8) / 50
    expect(r.variance).toBeCloseTo(V, 2)
    expect(r.mainVariance[0] / r.variance).toBeCloseTo(V1 / V, 3)
    expect(r.mainVariance[1] / r.variance).toBeCloseTo(V2 / V, 3)
    expect(r.mainVariance[2] / r.variance).toBeCloseTo(0, 10)
    // pairs: (0,1), (0,2), (1,2)
    expect(r.pairVariance[1] / r.variance).toBeCloseTo(V13 / V, 3)
    expect(r.pairVariance[0]).toBeCloseTo(0, 10)
    expect(r.higherVariance).toBeCloseTo(0, 8)
    expect(r.mean).toBeCloseTo(a / 2 + 0, 2)
  })
  it('has centred terms', () => {
    for (const m of r.main) expect(m.reduce((s, v) => s + v, 0)).toBeCloseTo(0, 9)
    const p = r.pairs[1].values
    for (let i = 0; i < g; i++) {
      let row = 0
      for (let j = 0; j < 300; j++) row += p[i * 300 + j]
      expect(row).toBeCloseTo(0, 8)
    }
  })
})
