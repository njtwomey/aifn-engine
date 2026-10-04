import { describe, expect, it } from 'vitest'
import { blobs, checkerboard, gaussians, moons, regression1d, xor } from 'aifn-methods/data/synthetic'
import { classCounts, type ClassificationTruth, type Dataset } from 'aifn-methods/data'
import {
  flippedMask,
  rotation2d,
  split,
  withCovariateShift,
  withLabelNoise,
  withMissing,
  withNuisanceFeatures,
  withOutliers,
  withPrevalence,
  withTransform,
} from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'

const counts = (d: Dataset, k = 2) => {
  const c = new Array<number>(k).fill(0)
  toFlat(d.y!).forEach((v) => c[v]++)
  return c
}
const truthOf = (d: Dataset) => d.meta.truth as ClassificationTruth
/** P(y = 1 | x) under a truth, at one point. */
const p1 = (t: ClassificationTruth, x: number[]) => toFlat(t.posterior(tensor([x])))[1]
const rows = (d: Dataset) => {
  const [n, dim] = d.x.shape
  const x = toFlat(d.x)
  return Array.from({ length: n }, (_, i) => x.slice(i * dim, (i + 1) * dim))
}

describe('class sizes', () => {
  it('classCounts uses the largest remainder and sums exactly', () => {
    expect(classCounts(100, [1, 2, 3])).toEqual([17, 33, 50])
    expect(classCounts(10, [1, 1, 1])).toEqual([4, 3, 3])
    expect(classCounts(7, [0.5, 0.5])).toEqual([4, 3])
    expect(classCounts(0, [1, 2])).toEqual([0, 0])
  })

  it('every labelled generator gives exact counts for a prevalence or weights', () => {
    expect(counts(moons(stream(1), { n: 400, prevalence: 0.2 }))).toEqual([320, 80])
    expect(counts(xor(stream(1), { n: 101, prevalence: 0.3 }))).toEqual([71, 30])
    expect(counts(xor(stream(1), { n: 50, prevalence: 0.1, kind: 'gaussian' }))).toEqual([45, 5])
    expect(counts(checkerboard(stream(1), { n: 90, classWeights: [2, 1] }))).toEqual([60, 30])
    expect(counts(blobs(stream(1), { n: 100, classWeights: [1, 2, 3] }), 3)).toEqual([17, 33, 50])
    expect(counts(gaussians(stream(1), { n: [10, 20] }))).toEqual([10, 20])
  })

  it('class-conditional xor and checkerboard keep their labelling rule', () => {
    const d = xor(stream(2), { n: 200, prevalence: 0.25 })
    const x = toFlat(d.x)
    toFlat(d.y!).forEach((l, i) => expect(l).toBe(x[2 * i] * x[2 * i + 1] < 0 ? 1 : 0))
    const c = checkerboard(stream(2), { n: 200, prevalence: 0.7, tiles: 3 })
    const xc = toFlat(c.x)
    toFlat(c.y!).forEach((l, i) => expect(l).toBe((Math.floor(xc[2 * i]) + Math.floor(xc[2 * i + 1])) % 2))
  })

  it('withPrevalence hits the target exactly by subsampling and oversampling', () => {
    const d = moons(stream(3), { n: 400 })
    expect(counts(withPrevalence(stream(4), d, { prevalence: 0.2 }))).toEqual([200, 50])
    expect(counts(withPrevalence(stream(4), d, { prevalence: 0.2, method: 'oversample' }))).toEqual([800, 200])
    expect(counts(withPrevalence(stream(4), d, { prevalence: 0.1, n: 100 }))).toEqual([90, 10])
  })
})

describe('label noise', () => {
  it('symmetric flips happen at the stated rate, and clean labels are kept', () => {
    const d = blobs(stream(5), { n: 20000, centers: 2, separation: 2 })
    const noisy = withLabelNoise(stream(6), d, { rate: 0.1 })
    const f = flippedMask(noisy)
    const rate = f.reduce((a, b) => a + b, 0) / f.length
    expect(Math.abs(rate - 0.1)).toBeLessThan(0.01)
    expect(toFlat(noisy.meta.cleanLabels!)).toEqual(toFlat(d.y!))
  })

  it('class-conditional flips follow the matrix row by row', () => {
    const d = blobs(stream(7), { n: 20000, centers: 2, separation: 2 })
    const noisy = withLabelNoise(stream(8), d, {
      matrix: [
        [0.9, 0.1],
        [0.3, 0.7],
      ],
    })
    const clean = toFlat(d.y!)
    const f = flippedMask(noisy)
    const rate = (j: number) => {
      const idx = clean.map((c, i) => (c === j ? i : -1)).filter((i) => i >= 0)
      return idx.reduce((a, i) => a + f[i], 0) / idx.length
    }
    expect(Math.abs(rate(0) - 0.1)).toBeLessThan(0.01)
    expect(Math.abs(rate(1) - 0.3)).toBeLessThan(0.015)
  })

  it('symmetric noise maps the Bayes error e to ρ + (1 − 2ρ) e', () => {
    const d = gaussians(stream(9), { n: 100, separation: 2 })
    const e = truthOf(d).bayesError
    const noisy = truthOf(withLabelNoise(stream(10), d, { rate: 0.1 }))
    expect(noisy.bayesErrorMethod).toBe('closed form')
    expect(noisy.bayesError).toBeCloseTo(0.1 + 0.8 * e, 12)
    expect(p1(noisy, [0.3, 0.1])).toBeCloseTo(0.1 + 0.8 * p1(truthOf(d), [0.3, 0.1]), 12)
  })
})

describe('known truth', () => {
  it('the Bayes posterior matches Monte Carlo label frequencies for Gaussian classes', () => {
    const d = gaussians(stream(11), {
      n: 40000,
      prevalence: 0.3,
      means: [
        [0, 0],
        [1.5, 0.5],
      ],
      covariances: [
        [
          [1, 0.3],
          [0.3, 1],
        ],
        [
          [0.5, 0],
          [0, 2],
        ],
      ],
    })
    const t = truthOf(d)
    const y = toFlat(d.y!)
    const bins = Array.from({ length: 10 }, () => ({ p: 0, y: 0, n: 0 }))
    const post = toFlat(t.posterior(d.x))
    rows(d).forEach((_, i) => {
      const p = post[2 * i + 1]
      const b = bins[Math.min(9, Math.floor(p * 10))]
      b.p += p
      b.y += y[i]
      b.n++
    })
    for (const b of bins.filter((b) => b.n > 500)) expect(Math.abs(b.y / b.n - b.p / b.n)).toBeLessThan(0.03)
  })

  it('the Bayes error of two 1-D Gaussians matches quadrature (closed form and Monte Carlo)', () => {
    const quad = (m: number[], sd: number[], pi: number[]) => {
      const pdf = (x: number, j: number) =>
        Math.exp(-0.5 * ((x - m[j]) / sd[j]) ** 2) / (sd[j] * Math.sqrt(2 * Math.PI))
      let s = 0
      const h = 1e-3
      for (let x = -15; x <= 15; x += h) s += Math.min(pi[0] * pdf(x, 0), pi[1] * pdf(x, 1)) * h
      return s
    }
    const equal = truthOf(
      gaussians(stream(12), { n: 10, prevalence: 0.3, means: [[0], [1.7]], covariances: [[[1]], [[1]]] }),
    )
    expect(equal.bayesErrorMethod).toBe('closed form')
    expect(equal.bayesError).toBeCloseTo(quad([0, 1.7], [1, 1], [0.7, 0.3]), 5)
    const unequal = truthOf(
      gaussians(stream(12), { n: 10, prevalence: 0.4, means: [[0], [1.5]], covariances: [[[1]], [[0.25]]] }),
    )
    expect(unequal.bayesErrorMethod).toBe('monte carlo')
    const q = quad([0, 1.5], [1, 0.5], [0.6, 0.4])
    expect(Math.abs(unequal.bayesError - q)).toBeLessThan(4 * unequal.bayesErrorSe + 1e-3)
  })

  it('a prevalence change gives the posterior under the new priors', () => {
    const d = gaussians(stream(13), { n: 400, separation: 2 })
    const shifted = truthOf(withPrevalence(stream(14), d, { prevalence: 0.2 }))
    const fresh = truthOf(gaussians(stream(13), { n: 10, separation: 2, prevalence: 0.2 }))
    for (const x of [
      [0, 0],
      [0.8, -1],
      [-1.5, 2],
    ])
      expect(p1(shifted, x)).toBeCloseTo(p1(fresh, x), 10)
    expect(shifted.prevalence[1]).toBeCloseTo(0.2, 12)
    expect(shifted.bayesError).toBeCloseTo(fresh.bayesError, 12)
  })

  it('covariate shift keeps the posterior; transforms and nuisance features carry it along', () => {
    const d = moons(stream(15), { n: 600, noise: 0.2 })
    const t = truthOf(d)
    const x = [0.4, 0.2]
    expect(p1(truthOf(withCovariateShift(stream(16), d)), x)).toBeCloseTo(p1(t, x), 10)
    const r = rotation2d(0.7)
    const moved = truthOf(withTransform(d, { matrix: r, offset: [1, -2] }))
    const x2 = [r[0][0] * x[0] + r[0][1] * x[1] + 1, r[1][0] * x[0] + r[1][1] * x[1] - 2]
    expect(p1(moved, x2)).toBeCloseTo(p1(t, x), 10)
    const wide = withNuisanceFeatures(stream(17), d, { count: 3 })
    expect(wide.x.shape).toEqual([600, 5])
    expect(p1(truthOf(wide), [...x, 5, -3, 1])).toBeCloseTo(p1(t, x), 10)
  })

  it('moons have a small Monte Carlo Bayes error that grows with the noise', () => {
    const low = truthOf(moons(stream(18), { n: 10, noise: 0.1 })).bayesError
    const high = truthOf(moons(stream(18), { n: 10, noise: 0.4 })).bayesError
    expect(low).toBeLessThan(0.01)
    expect(high).toBeGreaterThan(0.08)
  })

  it('regression generators carry the mean function and noise sd', () => {
    const r = regression1d(stream(19), { n: 20, fn: 'sine', noise: 0.3 })
    const t = r.meta.truth!
    expect(t.task).toBe('regression')
    if (t.task === 'regression' && 'noiseSd' in t) {
      expect(toFlat(t.mean(tensor([[1]])))[0]).toBeCloseTo(Math.sin(1), 14)
      expect(toFlat(t.expect(tensor([[1]])))[0]).toBeCloseTo(Math.sin(1), 14)
      expect(t.noiseSd).toBe(0.3)
      expect(t.bayesRisk).toBeCloseTo(0.09, 14)
    }
  })
})

describe('modifiers', () => {
  it('outliers, missing values and splits are marked and exact', () => {
    const d = moons(stream(20), { n: 200 })
    const o = withOutliers(stream(21), d, { fraction: 0.05 })
    expect(toFlat(o.meta.outliers!).reduce((a, b) => a + b, 0)).toBe(10)
    const m = withMissing(stream(22), d, { rate: 0.2, mechanism: 'mar' })
    const mask = toFlat(m.meta.missing!)
    expect(mask.filter((_, i) => i % 2 === 0).every((v) => v === 0)).toBe(true)
    toFlat(m.x).forEach((v, i) => expect(Number.isNaN(v)).toBe(mask[i] === 1))
    const { train, test } = split(stream(23), withPrevalence(stream(24), d, { prevalence: 0.2 }), { test: 0.3 })
    expect(counts(test)).toEqual([30, 8])
    expect(counts(train)).toEqual([70, 17])
  })
})

describe('modifier truths, exactly', () => {
  const d = gaussians(stream(30), { n: 400, separation: 2.5 })
  const clean = truthOf(d)
  const at = [
    [0, 0],
    [1.2, -0.4],
    [-2, 1],
  ]
  const exps = (t: ClassificationTruth, x: number[][]) => toFlat(t.logDensity(tensor(x))).map(Math.exp)

  it('outliers turn each class density into (1 − ε) p(x | j) + ε q(x) with one q shared by the classes', () => {
    const eps = 0.1
    const o = truthOf(withOutliers(stream(31), d, { fraction: eps }))
    const [b, l] = [exps(clean, at), exps(o, at)]
    for (let i = 0; i < at.length; i++) {
      const q0 = (l[2 * i] - (1 - eps) * b[2 * i]) / eps
      const q1 = (l[2 * i + 1] - (1 - eps) * b[2 * i + 1]) / eps
      expect(q0).toBeGreaterThan(0)
      expect(q1).toBeCloseTo(q0, 12)
    }
    // Far from both classes the broad q dominates, so the posterior returns to the priors.
    const far = toFlat(o.posterior(tensor([[40, -40]])))
    expect(far[1]).toBeCloseTo(o.priors[1], 6)
    expect(o.bayesError).toBeGreaterThan(clean.bayesError)
  })

  it('missing values: the truth and the complete data are kept; MCAR, MAR and MNAR follow their mechanism', () => {
    const big = gaussians(stream(32), { n: 4000, separation: 2 })
    const mcar = withMissing(stream(33), big, { rate: 0.3 })
    expect(mcar.meta.truth).toBe(big.meta.truth)
    expect(toFlat(mcar.meta.complete!)).toEqual(toFlat(big.x))
    const rate = toFlat(mcar.meta.missing!).reduce((a, b) => a + b, 0) / 8000
    expect(Math.abs(rate - 0.3)).toBeLessThan(0.02)
    // MAR: feature 0 complete; feature 1 is missing more often where feature 0 is large.
    const mar = withMissing(stream(34), big, { rate: 0.3, mechanism: 'mar' })
    const mask = toFlat(mar.meta.missing!)
    const x = toFlat(big.x)
    const meanOf = (pick: (i: number) => boolean, c: number) => {
      const v = Array.from({ length: 4000 }, (_, i) => i)
        .filter(pick)
        .map((i) => x[2 * i + c])
      return v.reduce((a, b) => a + b, 0) / v.length
    }
    expect(Array.from({ length: 4000 }, (_, i) => mask[2 * i]).every((v) => v === 0)).toBe(true)
    expect(meanOf((i) => mask[2 * i + 1] === 1, 0)).toBeGreaterThan(meanOf((i) => mask[2 * i + 1] === 0, 0) + 0.5)
    // MNAR: large values hide themselves.
    const mnar = toFlat(withMissing(stream(35), big, { rate: 0.3, mechanism: 'mnar' }).meta.missing!)
    expect(meanOf((i) => mnar[2 * i] === 1, 0)).toBeGreaterThan(meanOf((i) => mnar[2 * i] === 0, 0) + 0.5)
  })

  it('covariate shift adds the same log selection probability to every class density, rising along the direction', () => {
    const shifted = withCovariateShift(stream(36), d, { strength: 1.5, keep: 0.5 })
    const t = truthOf(shifted)
    const line = [
      [-2, 0],
      [0, 0],
      [2, 0],
    ]
    const [b, l] = [toFlat(clean.logDensity(tensor(line))), toFlat(t.logDensity(tensor(line)))]
    const shift = [0, 1, 2].map((i) => l[2 * i] - b[2 * i])
    ;[0, 1, 2].forEach((i) => expect(l[2 * i + 1] - b[2 * i + 1]).toBeCloseTo(shift[i], 12))
    shift.forEach((v) => expect(v).toBeLessThan(0))
    expect(shift[0]).toBeLessThan(shift[1])
    expect(shift[1]).toBeLessThan(shift[2])
    // About half the rows survive, and the survivors sit further along the first feature.
    const kept = shifted.x.shape[0]
    expect(Math.abs(kept - 200)).toBeLessThan(40)
    const first = (z: Dataset) => toFlat(z.x).filter((_, i) => i % 2 === 0)
    const mean = (v: number[]) => v.reduce((a, c) => a + c, 0) / v.length
    expect(mean(first(shifted))).toBeGreaterThan(mean(first(d)) + 0.3)
  })
})
