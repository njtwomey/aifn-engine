import { describe, expect, it } from 'vitest'
import { datasetRegistry, recipe } from 'aifn-methods/data'
import type { ChangepointTruth } from 'aifn-methods/data'
import { arRegimes, meanShifts, poissonShifts, varianceShifts } from 'aifn-methods/data/synthetic'
import type { Dataset } from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import {
  detectChangepoints,
  laggedObservations,
  normalGamma,
  normalKnownVariance,
  poissonGamma,
  regressionNormalGamma,
} from 'aifn-compute/inference/filtering'

const truthOf = (d: Dataset) => d.meta!.truth as ChangepointTruth
const values = (d: Dataset) => Array.from(toFlat(d.x as never))
const mean = (a: number[]) => a.reduce((p, v) => p + v, 0) / a.length
const variance = (a: number[]) => {
  const m = mean(a)
  return a.reduce((p, v) => p + (v - m) ** 2, 0) / a.length
}

const generators = {
  meanShifts: (s: ReturnType<typeof stream>) => meanShifts(s, { n: 400 }),
  varianceShifts: (s: ReturnType<typeof stream>) => varianceShifts(s, { n: 400 }),
  poissonShifts: (s: ReturnType<typeof stream>) => poissonShifts(s, { n: 400 }),
  arRegimes: (s: ReturnType<typeof stream>) => arRegimes(s, { n: 400 }),
}

describe('changepoint generators', () => {
  it.each(Object.entries(generators))('%s: shapes, determinism and a consistent truth', (name, make) => {
    const a = make(stream('cp'))
    const b = make(stream('cp'))
    expect(values(a)).toEqual(values(b))
    expect(values(make(stream('other')))).not.toEqual(values(a))
    expect(a.x.shape).toEqual([400, 1])
    expect(a.y!.dtype).toBe('int32')
    expect(toFlat(a.t!)).toEqual(Array.from({ length: 400 }, (_, i) => i))
    expect(a.meta!.task).toBe('sequence')
    expect(a.meta!.recipe!.base).toBe(name)
    const t = truthOf(a)
    expect(t.kind).toBe('model')
    expect(t.task).toBe('changepoint')
    expect(t.n).toBe(400)
    // Segments tile 0 … n − 1; y is the segment index; the changepoints are the segment starts after 0.
    expect(t.segments[0].start).toBe(0)
    expect(t.segments.at(-1)!.end).toBe(400)
    expect(t.changepoints).toEqual(t.segments.slice(1).map((g) => g.start))
    const y = toFlat(a.y!)
    t.segments.forEach((g, j) => {
      for (let i = g.start; i < g.end; i++) expect(y[i]).toBe(j)
    })
    // decide = the segment at each time; expect = its mean; bayesRisk = the mean of the segments' risks.
    const times = tensor([0, 57, 399])
    expect(Array.from(toFlat(t.decide(times)))).toEqual([0, 57, 399].map((i) => y[i]))
    expect(Array.from(toFlat(t.expect(times)))).toEqual([0, 57, 399].map((i) => t.segments[y[i]].mean))
    const risk = t.segments.reduce((p, g) => p + g.risk * (g.end - g.start), 0) / 400
    expect(t.bayesRisk).toBeCloseTo(risk, 12)
  })

  it('given changepoints and parameters are kept exactly', () => {
    const d = meanShifts(stream(1), { n: 300, changepoints: [100, 200], means: [0, 5, -5], sd: 0.5 })
    const t = truthOf(d)
    expect(t.changepoints).toEqual([100, 200])
    expect(t.segments.map((g) => g.mean)).toEqual([0, 5, -5])
    const x = values(d)
    expect(mean(x.slice(100, 200))).toBeCloseTo(5, 0)
    expect(Math.sqrt(variance(x.slice(200)))).toBeCloseTo(0.5, 1)
    expect(() => meanShifts(stream(1), { n: 100, changepoints: [50, 40] })).toThrow(/ascending/)
    expect(() => meanShifts(stream(1), { n: 100, changepoints: [0, 1], means: [1, 2, 3] })).toThrow()
  })

  it('segment statistics follow each family', () => {
    const v = varianceShifts(stream(2), { n: 4000, changepoints: [2000], sds: [0.5, 2] })
    const xv = values(v)
    expect(Math.sqrt(variance(xv.slice(0, 2000)))).toBeCloseTo(0.5, 1)
    expect(Math.sqrt(variance(xv.slice(2000)))).toBeCloseTo(2, 1)
    const p = poissonShifts(stream(3), { n: 4000, changepoints: [2000], rates: [1, 6] })
    const xp = values(p)
    expect(xp.every((c) => Number.isInteger(c) && c >= 0)).toBe(true)
    expect(mean(xp.slice(0, 2000))).toBeCloseTo(1, 1)
    expect(mean(xp.slice(2000))).toBeCloseTo(6, 0)
    // AR(1) regime a: stationary variance sd²/(1 − a²) and lag-one autocorrelation a.
    const a = arRegimes(stream(4), { n: 8000, changepoints: [4000], regimes: [[0.8], [-0.5]] })
    const xa = values(a)
    const ta = truthOf(a)
    expect(ta.segments[0].variance).toBeCloseTo(1 / (1 - 0.64), 10)
    expect(ta.segments[1].variance).toBeCloseTo(1 / (1 - 0.25), 10)
    const lag1 = (s: number[]) => {
      const m = mean(s)
      let num = 0
      for (let i = 1; i < s.length; i++) num += (s[i] - m) * (s[i - 1] - m)
      return num / (variance(s) * s.length)
    }
    expect(lag1(xa.slice(100, 4000))).toBeCloseTo(0.8, 1)
    expect(lag1(xa.slice(4100))).toBeCloseTo(-0.5, 1)
  })

  it('the recipe builds changepoint bases and keeps their truth', () => {
    expect(datasetRegistry.poissonShifts.info.task).toBe('sequence')
    expect(datasetRegistry.moons.info.task).toBe('classification')
    const d = recipe({ base: 'meanShifts', knobs: { n: 200, sd: 0.5 }, seed: 3 })
    expect(truthOf(d).task).toBe('changepoint')
    expect(truthOf(d).segments.every((g) => Math.abs(g.variance - 0.25) < 1e-12)).toBe(true)
    expect(recipe({ base: 'poissonShifts', knobs: { n: 100, noise: 0.3 }, seed: 1 }).meta.ignored).toEqual(['noise'])
  })
})

describe('BOCPD on the generators', () => {
  // Pooled over several drawn series: the share of true changepoints with an estimate within `slack` steps, and the
  // share of estimates near a true changepoint.
  const pooled = (
    make: (seed: number) => { truth: readonly number[]; found: readonly number[] },
    slack: number,
    seeds = [1, 2, 3, 4, 5, 6],
  ) => {
    let hits = 0
    let total = 0
    let good = 0
    let found = 0
    for (const seed of seeds) {
      const r = make(seed)
      hits += r.truth.filter((c) => r.found.some((f) => Math.abs(f - c) <= slack)).length
      total += r.truth.length
      good += r.found.filter((f) => r.truth.some((c) => Math.abs(f - c) <= slack)).length
      found += r.found.length
    }
    return { recall: hits / total, precision: good / found }
  }
  const H = 1 / 80
  const run = (d: Dataset, detect: (x: number[]) => { changepoints: number[] }) => ({
    truth: truthOf(d).changepoints,
    found: detect(values(d)).changepoints,
  })

  it('recovers mean shifts (known and unknown variance)', () => {
    const known = pooled(
      (seed) =>
        run(meanShifts(stream(seed), { n: 400, meanGap: 80 }), (x) =>
          detectChangepoints(normalKnownVariance({ priorSd: 3, sd: 1 }), x, { hazard: H }),
        ),
      5,
    )
    const unknown = pooled(
      (seed) =>
        run(meanShifts(stream(seed), { n: 400, meanGap: 80 }), (x) =>
          detectChangepoints(normalGamma(), x, { hazard: H }),
        ),
      5,
    )
    expect(known.recall).toBeGreaterThanOrEqual(0.75)
    expect(known.precision).toBeGreaterThanOrEqual(0.75)
    expect(unknown.recall).toBeGreaterThanOrEqual(0.75)
    expect(unknown.precision).toBeGreaterThanOrEqual(0.75)
  })

  it('rows of the run-length posterior sum to one, with P(r = 0) = H under a constant hazard', () => {
    const d = meanShifts(stream(1), { n: 200 })
    const det = detectChangepoints(normalKnownVariance({ priorSd: 3, sd: 1 }), values(d), { hazard: H })
    const w = det.posterior.shape[1]
    const post = toFlat(det.posterior)
    for (const row of [0, 100, 199]) {
      const r = post.slice(row * w, (row + 1) * w)
      expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
      expect(r[0]).toBeCloseTo(H, 10)
    }
  })

  it('recovers variance shifts, Poisson rate shifts and AR regimes', () => {
    const v = pooled(
      (seed) =>
        run(varianceShifts(stream(seed), { n: 400, meanGap: 80 }), (x) =>
          detectChangepoints(normalGamma(), x, { hazard: H }),
        ),
      10,
    )
    const p = pooled(
      (seed) =>
        run(poissonShifts(stream(seed), { n: 400, meanGap: 80 }), (x) =>
          detectChangepoints(poissonGamma({ shape: 2, rate: 0.5 }), x, { hazard: H }),
        ),
      10,
    )
    // laggedObservations drops the first p values, so the estimated indices are one less than the true ones.
    const a = pooled((seed) => {
      const d = arRegimes(stream(seed), { n: 600, changepoints: [150, 300, 450], regimes: [[0.9], [-0.7]] })
      const lagged = laggedObservations(values(d), 1, { intercept: false })
      const found = detectChangepoints(regressionNormalGamma({ dimension: 1 }), lagged, { hazard: 1 / 150 })
      return { truth: [149, 299, 449], found: found.changepoints }
    }, 10)
    expect(v.recall).toBeGreaterThanOrEqual(0.75)
    expect(p.recall).toBeGreaterThanOrEqual(0.75)
    expect(a.recall).toBeGreaterThanOrEqual(0.75)
    for (const r of [v, p, a]) expect(r.precision).toBeGreaterThanOrEqual(0.75)
  })
})
