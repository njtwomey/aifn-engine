import { describe, expect, it } from 'vitest'
import {
  anisotropicBlobs,
  arSeries,
  barsAndStripes,
  blobs,
  casino,
  checkerboard,
  checkerboardImage,
  circles,
  clickLog,
  digitGlyphs,
  digits,
  friedman1,
  gaussians,
  gradientImage,
  linearRegressionData,
  moons,
  motifSeries,
  randomWalk,
  ratings,
  regression1d,
  rings,
  sCurve,
  seasonalSeries,
  shapesImage,
  shuffleDataset,
  spirals,
  swissRoll,
  xor,
  zipfCatalogue,
  zipfWeights,
} from 'aifn-methods/data/synthetic'
import { type Dataset } from 'aifn-methods/data'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

const generators: [string, (s: ReturnType<typeof stream>) => Dataset, number, number][] = [
  ['blobs', (s) => blobs(s, { n: 90 }), 90, 2],
  ['moons', (s) => moons(s, { n: 101 }), 101, 2],
  ['circles', (s) => circles(s, { n: 80 }), 80, 2],
  ['rings', (s) => rings(s, { n: 90 }), 90, 2],
  ['spirals', (s) => spirals(s, { n: 60, arms: 3 }), 60, 2],
  ['xor', (s) => xor(s, { n: 40 }), 40, 2],
  ['checkerboard', (s) => checkerboard(s, { n: 50 }), 50, 2],
  [
    'gaussians',
    (s) =>
      gaussians(s, {
        means: [
          [0, 0],
          [3, 3],
        ],
        covariances: [
          [
            [1, 0.5],
            [0.5, 1],
          ],
          [
            [0.5, 0],
            [0, 2],
          ],
        ],
        n: 40,
      }),
    40,
    2,
  ],
  ['anisotropic', (s) => anisotropicBlobs(s, { n: 30 }), 30, 2],
  ['swiss roll', (s) => swissRoll(s, { n: 70 }), 70, 3],
  ['s-curve', (s) => sCurve(s, { n: 70 }), 70, 3],
  ['regression1d', (s) => regression1d(s, { n: 33 }), 33, 1],
  ['linear', (s) => linearRegressionData(s, { n: 25, d: 4 }), 25, 4],
  ['friedman1', (s) => friedman1(s, { n: 20 }), 20, 10],
  ['digits', (s) => digits(s, { perClass: 3 }), 30, 35],
]

describe('synthetic datasets', () => {
  it.each(generators)('%s has the stated shape and is deterministic in its stream', (_name, make, n, d) => {
    const a = make(stream('datasets-test'))
    const b = make(stream('datasets-test'))
    const c = make(stream('other'))
    expect(a.x.shape).toEqual([n, d])
    if (a.y) expect(a.y.shape).toEqual([n])
    expect(toFlat(a.x)).toEqual(toFlat(b.x))
    expect(toFlat(a.x)).not.toEqual(toFlat(c.x))
    expect(a.meta.featureNames.length).toBe(d)
    expect(a.meta.description.length).toBeGreaterThan(10)
  })

  it('moons without noise lie on their half circles', () => {
    const m = moons(stream(1), { n: 20, noise: 0 })
    const x = toFlat(m.x)
    for (let i = 0; i < 10; i++) expect(Math.hypot(x[2 * i], x[2 * i + 1])).toBeCloseTo(1, 12)
    for (let i = 10; i < 20; i++) expect(Math.hypot(x[2 * i] - 1, x[2 * i + 1] - 0.5)).toBeCloseTo(1, 12)
  })

  it('xor labels are the sign disagreement; checkerboard labels the tile parity', () => {
    const d = xor(stream(2), { n: 100 })
    const x = toFlat(d.x)
    toFlat(d.y!).forEach((l, i) => expect(l).toBe(x[2 * i] * x[2 * i + 1] < 0 ? 1 : 0))
    const c = checkerboard(stream(2), { n: 100, tiles: 3 })
    const xc = toFlat(c.x)
    toFlat(c.y!).forEach((l, i) => expect(l).toBe((Math.floor(xc[2 * i]) + Math.floor(xc[2 * i + 1])) % 2))
  })

  it('shuffleDataset permutes rows together', () => {
    const d = blobs(stream(3), { n: 30 })
    const s = shuffleDataset(stream(4), d)
    const key = (x: number[], y: number[]) => y.map((l, i) => `${x[2 * i]},${x[2 * i + 1]},${l}`).sort()
    expect(key(toFlat(s.x), toFlat(s.y!))).toEqual(key(toFlat(d.x), toFlat(d.y!)))
  })

  it('regression1d recovers f with zero noise', () => {
    const r = regression1d(stream(5), { n: 10, noise: 0, fn: 'cubic' })
    expect(toFlat(r.y!)).toEqual(toFlat(r.f!))
  })
})

describe('sequences', () => {
  it('the casino emits faces 1–6 and loaded states show more sixes', () => {
    const c = casino(stream(6), { n: 5000 })
    const x = toFlat(c.x)
    const z = toFlat(c.z)
    expect(Math.min(...x)).toBe(1)
    expect(Math.max(...x)).toBe(6)
    const sixes = (state: number) => {
      const idx = z.map((v, i) => (v === state ? i : -1)).filter((i) => i >= 0)
      return idx.filter((i) => x[i] === 6).length / idx.length
    }
    expect(sixes(1)).toBeGreaterThan(0.4)
    expect(sixes(0)).toBeLessThan(0.25)
  })

  it('AR(1) has the stationary variance and flags an explosive model', () => {
    const a = arSeries(stream(7), { coefficients: [0.8], n: 20000 })
    const y = toFlat(a.y)
    const m = y.reduce((p, v) => p + v, 0) / y.length
    const v = y.reduce((p, x) => p + (x - m) ** 2, 0) / y.length
    expect(v).toBeCloseTo(1 / (1 - 0.64), 0)
    expect(arSeries(stream(7), { coefficients: [1.5], n: 3000, burn: 0 }).diverged).toBe(true)
    expect(seasonalSeries(stream(8)).y.shape).toEqual([120])
    expect(randomWalk(stream(9), { n: 5 }).y.shape).toEqual([5])

    const c0 = casino(stream(10), { toLoaded: 0, toFair: 0 })
    expect(Array.from(toFlat(c0.model.initial))).toEqual([0.5, 0.5])
    expect(() => seasonalSeries(stream(8), { persistence: 1 })).toThrow(DomainError)
    expect(() => seasonalSeries(stream(8), { persistence: -1.5 })).toThrow(DomainError)
    expect(() => motifSeries(stream(8), { m: 1 })).toThrow(DomainError)
  })
})

describe('recommendation data', () => {
  it('Zipf weights normalise and fall as a power', () => {
    const w = toFlat(zipfWeights(100, 1.2))
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(w[0] / w[1]).toBeCloseTo(2 ** 1.2, 10)
    const c = zipfCatalogue(stream(10), { items: 50, n: 2000 })
    expect(toFlat(c.counts).reduce((a, b) => a + b, 0)).toBe(2000)
  })

  it('ratings are on the scale and split into train and test', () => {
    const r = ratings(stream(11), { users: 20, items: 30 })
    const t = toFlat(r.truth)
    expect(Math.min(...t)).toBeGreaterThanOrEqual(1)
    expect(Math.max(...t)).toBeLessThanOrEqual(5)
    expect(r.train.shape[1]).toBe(3)
    expect(r.train.shape[0] + r.test.shape[0]).toBeGreaterThan(150)
  })

  it('click logs: clicks need examination; the cascade stops after the first click', () => {
    const log = clickLog(stream(12), { sessions: 100, model: 'cascade' })
    const ex = toFlat(log.examined)
    const cl = toFlat(log.clicked)
    cl.forEach((c, i) => expect(c <= ex[i]).toBe(true))
    for (let q = 0; q < 100; q++)
      expect(cl.slice(q * 10, q * 10 + 10).reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(1)
    const pbm = clickLog(stream(12), { sessions: 50, eta: 1 })
    expect(toFlat(pbm.propensity).slice(0, 3)).toEqual([1, 1 / 2, 1 / 3])
  })
})

describe('images and patterns', () => {
  it('images have the stated shapes and ranges', () => {
    expect(checkerboardImage({ size: 16, tile: 4 }).shape).toEqual([16, 16])
    const g = toFlat(gradientImage({ size: 9, angle: 0 }))
    expect(g[0]).toBeCloseTo(0, 12)
    expect(g[8]).toBeCloseTo(1, 12)
    expect(shapesImage().shape).toEqual([64, 64])
    expect(digitGlyphs().shape).toEqual([10, 7, 5])
    expect(barsAndStripes({ size: 4 }).shape).toEqual([30, 16])
  })
})
