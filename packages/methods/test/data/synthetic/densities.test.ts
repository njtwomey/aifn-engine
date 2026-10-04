import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { datasetRegistry, type ClassificationTruth, type Dataset } from 'aifn-methods/data'
import { annulus, gaussianGrid, gaussianRing, pinwheel, swissRoll2d } from 'aifn-methods/data/synthetic'
import { mixtureLogDensityOf } from 'aifn-methods/generative'

const model = (d: Dataset) => (d.meta.truth as ClassificationTruth).model

/** ∫ p(x) dx by the midpoint rule on [−L, L]² with g cells per side. */
function mass(d: Dataset, L: number, g: number): number {
  const h = (2 * L) / g
  const pts = new Float64Array(2 * g * g)
  for (let i = 0; i < g; i++)
    for (let j = 0; j < g; j++) {
      pts[2 * (i * g + j)] = -L + (j + 0.5) * h
      pts[2 * (i * g + j) + 1] = -L + (i + 0.5) * h
    }
  return mixtureLogDensityOf(model(d), fromData(pts, [g * g, 2])).reduce((s, v) => s + Math.exp(v) * h * h, 0)
}

const CASES: [string, (s: ReturnType<typeof stream>) => Dataset, number, number][] = [
  ['gaussianRing', (s) => gaussianRing(s, { n: 400 }), 3, 240],
  ['gaussianGrid', (s) => gaussianGrid(s, { n: 400 }), 3, 240],
  ['pinwheel', (s) => pinwheel(s, { n: 400 }), 3.5, 140],
  ['swissRoll2d', (s) => swissRoll2d(s, { n: 400 }), 3.5, 140],
  ['annulus', (s) => annulus(s, { n: 400 }), 4.5, 300],
]

describe('2-d densities for generative models', () => {
  it('are registered datasets with truth', () => {
    for (const [key] of CASES) {
      expect(datasetRegistry[key]?.info.truth).toBe(true)
      expect(datasetRegistry[key]?.info.task).toBe('clustering')
    }
  })

  for (const [name, make, L, g] of CASES) {
    it(`${name}: deterministic, and its density integrates to 1`, () => {
      const a = make(stream(3))
      const b = make(stream(3))
      expect(Array.from(toFlat(a.x))).toEqual(Array.from(toFlat(b.x)))
      expect(mass(a, L, g)).toBeCloseTo(1, 2)
    })
  }

  it('samples match the density: ring radii and per-mode means, annulus radii', () => {
    const ring = gaussianRing(stream(1), { n: 4000, modes: 8, radius: 2, sd: 0.05 })
    const x = toFlat(ring.x)
    const y = toFlat(ring.y!)
    let r = 0
    const sums = Array.from({ length: 8 }, () => [0, 0, 0])
    for (let i = 0; i < 4000; i++) {
      r += Math.hypot(x[2 * i], x[2 * i + 1]) / 4000
      sums[y[i]][0] += x[2 * i]
      sums[y[i]][1] += x[2 * i + 1]
      sums[y[i]][2]++
    }
    expect(r).toBeCloseTo(2, 2)
    sums.forEach(([sx, sy, n], j) => {
      expect(sx / n).toBeCloseTo(2 * Math.cos((2 * Math.PI * j) / 8), 1)
      expect(sy / n).toBeCloseTo(2 * Math.sin((2 * Math.PI * j) / 8), 1)
    })
    // Uniform on 3 ≤ ρ ≤ 4: E ρ² = (3² + 4²)/2.
    const shell = toFlat(annulus(stream(2), { n: 4000 }).x)
    let r2 = 0
    for (let i = 0; i < 4000; i++) r2 += (shell[2 * i] ** 2 + shell[2 * i + 1] ** 2) / 4000
    expect(r2).toBeCloseTo(12.5, 0)
  })

  it('a point is assigned to the mode it came from', () => {
    const grid = gaussianGrid(stream(5), { n: 250, sd: 0.05 })
    const post = toFlat((grid.meta.truth as ClassificationTruth).decide(grid.x))
    expect(Array.from(post)).toEqual(Array.from(toFlat(grid.y!)))
  })
})
