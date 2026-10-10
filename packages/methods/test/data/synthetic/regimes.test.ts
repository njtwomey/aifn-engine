/**
 * The regime generators (`piecewiseLinear`, `quadrantPlanes`, `interleavedFunctions`, `regressionMixture`): seeded
 * determinism, the returned regimes against the true gate, y against the regime functions, and the truth's laws (gate
 * rows sum to one, the mixture mean, the exact log-likelihood, the Bayes risk of a hard gate is σ²).
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { datasetRegistry, generate, type RegimeTruth } from 'aifn-methods/data'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  interleavedFunctions,
  piecewiseLinear,
  quadrantOf,
  quadrantPlanes,
  regressionMixture,
  type RegimeDataset,
} from 'aifn-methods/data/synthetic'

const GENERATORS = {
  piecewiseLinear: (seed: string) => piecewiseLinear(stream(seed), { n: 150, pieces: 4 }),
  quadrantPlanes: (seed: string) => quadrantPlanes(stream(seed), { n: 150 }),
  quadrantBoundaries: (seed: string) => quadrantPlanes(stream(seed), { n: 150, task: 'classification' }),
  interleavedFunctions: (seed: string) => interleavedFunctions(stream(seed), { n: 150, bands: 5 }),
  regressionMixture: (seed: string) => regressionMixture(stream(seed), { n: 150, regimes: 3 }),
}
const truthOf = (d: RegimeDataset) => d.meta.truth as RegimeTruth

describe('regime generators', () => {
  for (const [name, make] of Object.entries(GENERATORS)) {
    it(`${name}: the same stream gives the same data, another stream different data`, () => {
      const a = make('r')
      const b = make('r')
      expect(toFlat(a.x as Tensor)).toEqual(toFlat(b.x as Tensor))
      expect(toFlat(a.y as Tensor)).toEqual(toFlat(b.y as Tensor))
      expect(toFlat(a.regime)).toEqual(toFlat(b.regime))
      expect(toFlat(make('other').x as Tensor)).not.toEqual(toFlat(a.x as Tensor))
    })

    it(`${name}: every row's regime has positive gate, and the gate rows sum to one`, () => {
      const d = make('g')
      const gate = toRows(truthOf(d).gate(d.x as Tensor))
      const regime = toFlat(d.regime)
      gate.forEach((row, i) => {
        expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
        expect(row[regime[i]]).toBeGreaterThan(0)
      })
    })
  }

  it('hard gates: the regime is the gate argmax and f is the regime function', () => {
    for (const make of [GENERATORS.piecewiseLinear, GENERATORS.quadrantPlanes, GENERATORS.interleavedFunctions]) {
      const d = make('h')
      const t = truthOf(d)
      expect(toFlat(t.regime(d.x as Tensor))).toEqual(toFlat(d.regime))
      const means = toRows(t.regimeMean(d.x as Tensor))
      const r = toFlat(d.regime)
      toFlat(d.f!).forEach((f, i) => expect(f).toBeCloseTo(means[i][r[i]], 12))
    }
  })

  it('piecewise linear: K regimes on equal intervals, the jump at each breakpoint', () => {
    const d = piecewiseLinear(stream('p'), { n: 400, pieces: 3, jump: 1.5, noise: 0 })
    const t = truthOf(d)
    const xs = toFlat(d.x as Tensor)
    toFlat(d.regime).forEach((k, i) => expect(k).toBe(Math.min(2, Math.floor(((xs[i] + 3) / 6) * 3))))
    // At a breakpoint b, piece k ends and piece k + 1 starts 1.5 away (alternating sign).
    for (const [b, k] of [
      [-1, 0],
      [1, 1],
    ]) {
      const m = toRows(t.regimeMean(tensor([[b]])))[0]
      expect(Math.abs(m[k + 1] - m[k])).toBeCloseTo(1.5, 12)
    }
    // Noise 0: y is f.
    expect(toFlat(d.y as Tensor)).toEqual(toFlat(d.f!))
  })

  it('quadrants: the regime is the quadrant; classification labels follow each quadrant boundary', () => {
    const d = quadrantPlanes(stream('q'), { n: 300, task: 'classification', sharpness: 200 })
    const rows = toRows(d.x as Tensor)
    toFlat(d.regime).forEach((k, i) => expect(k).toBe(quadrantOf(rows[i][0], rows[i][1])))
    // A nearly noise-free link: the Bayes rule recovers almost every label.
    const decided = toFlat(truthOf(d).decide(d.x as Tensor))
    const agree = toFlat(d.y as Tensor).filter((y, i) => y === decided[i]).length / 300
    expect(agree).toBeGreaterThan(0.97)
    expect(truthOf(d).bayesRisk).toBeLessThan(0.02)
  })

  it('interleaved: the two lines alternate band by band', () => {
    const d = interleavedFunctions(stream('i'), { n: 200, bands: 6 })
    const xs = toFlat(d.x as Tensor)
    toFlat(d.regime).forEach((k, i) => expect(k).toBe(Math.min(5, Math.floor(((xs[i] + 3) / 6) * 6)) % 2))
  })

  it('a hard gate has Bayes risk σ²; an overlapping mixture more, by the spread of its lines', () => {
    expect(truthOf(piecewiseLinear(stream('b'), { noise: 0.3 })).bayesRisk).toBeCloseTo(0.09, 10)
    const soft = truthOf(regressionMixture(stream('b'), { noise: 0.3, overlap: 3 }))
    expect(soft.bayesRisk).toBeGreaterThan(0.09)
  })

  it('the mixture log-likelihood and mean agree with a direct computation', () => {
    const d = regressionMixture(stream('m'), { n: 20, regimes: 2, noise: 0.4, overlap: 2 })
    const t = truthOf(d)
    const x = d.x as Tensor
    const g = toRows(t.gate(x))
    const mu = toRows(t.regimeMean(x))
    const y = toFlat(d.y as Tensor)
    const ll = toFlat(t.logLikelihood(x, d.y as Tensor))
    const mean = toFlat(t.mean(x))
    y.forEach((v, i) => {
      const p = g[i].reduce(
        (a, w, k) => a + (w * Math.exp(-0.5 * ((v - mu[i][k]) / 0.4) ** 2)) / (0.4 * Math.sqrt(2 * Math.PI)),
        0,
      )
      expect(ll[i]).toBeCloseTo(Math.log(p), 10)
      expect(mean[i]).toBeCloseTo(g[i][0] * mu[i][0] + g[i][1] * mu[i][1], 12)
    })
  })

  it('a small overlap makes the soft gate nearly a hard split at x = 0', () => {
    const t = truthOf(regressionMixture(stream('o'), { regimes: 2, overlap: 0.02 }))
    const g = toRows(t.gate(tensor([[-0.5], [0.5]])))
    expect(g[0][0]).toBeGreaterThan(0.999)
    expect(g[1][1]).toBeGreaterThan(0.999)
  })

  it('are registered and replay through generate with their knobs', () => {
    for (const key of ['piecewiseLinear', 'quadrantPlanes', 'interleavedFunctions', 'regressionMixture']) {
      const entry = datasetRegistry[key]
      expect(entry?.info.truth).toBe(true)
      const d = generate(entry, stream('reg'), { n: 30 }) as RegimeDataset
      expect(d.x.shape[0]).toBe(30)
      expect(d.regime.shape).toEqual([30])
    }
  })

  it('interleavedFunctions validates bands', () => {
    expect(() => interleavedFunctions(stream('b'), { bands: 0 })).toThrow(DomainError)
    expect(() => interleavedFunctions(stream('b'), { bands: -2 })).toThrow(DomainError)
    expect(() => interleavedFunctions(stream('b'), { bands: 1.5 })).toThrow(DomainError)
  })
})
