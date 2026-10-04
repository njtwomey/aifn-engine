import { describe, expect, test } from 'vitest'
import { ransac, ransacFit, ransacTrials, type RansacProblem } from 'aifn-compute/numerics/robust'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkProtocol } from '../../protocol'

// A line y = 2x + 1 through 60 inliers with noise 0.05, plus 40 uniform outliers.
const n = 100
const xs = toFlat(uniform(stream('x'), 0, 10, { shape: [n] }) as Tensor)
const noise = toFlat(normal(stream('e'), 0, 0.05, { shape: [n] }) as Tensor)
const junk = toFlat(uniform(stream('j'), -10, 30, { shape: [n] }) as Tensor)
const ys = xs.map((x, i) => (i < 60 ? 2 * x + 1 + noise[i] : junk[i]))

type Line = { a: number; b: number }
const lineProblem: RansacProblem<Line> = {
  count: n,
  sampleSize: 2,
  fit: (idx) => {
    if (idx.length === 2) {
      const [i, j] = idx
      if (xs[i] === xs[j]) return null
      const a = (ys[j] - ys[i]) / (xs[j] - xs[i])
      return { a, b: ys[i] - a * xs[i] }
    }
    // Least squares on the consensus set.
    const m = idx.length
    const mx = idx.reduce((s, i) => s + xs[i], 0) / m
    const my = idx.reduce((s, i) => s + ys[i], 0) / m
    const sxy = idx.reduce((s, i) => s + (xs[i] - mx) * (ys[i] - my), 0)
    const sxx = idx.reduce((s, i) => s + (xs[i] - mx) ** 2, 0)
    return { a: sxy / sxx, b: my - (sxy / sxx) * mx }
  },
  residuals: ({ a, b }) => xs.map((x, i) => Math.abs(ys[i] - a * x - b)),
}

describe('RANSAC', () => {
  test('recovers a line through 40% outliers and flags exactly the inliers', () => {
    const r = ransacFit(lineProblem, { threshold: 0.2, stream: stream('ransac') })
    expect(r.model!.a).toBeCloseTo(2, 2)
    expect(r.model!.b).toBeCloseTo(1, 1)
    expect(r.inlierCount).toBe(60)
    r.inliers.forEach((v, i) => expect(v).toBe(i < 60 ? 1 : 0))
    expect(r.samples).toBeLessThan(30)
  })

  test('the adaptive count: ⌈log(1 − p)/log(1 − wˢ)⌉', () => {
    expect(ransacTrials(0.5, 4, 0.99)).toBe(72)
    expect(ransacTrials(0.6, 2, 0.99)).toBe(Math.ceil(Math.log(0.01) / Math.log(1 - 0.36)))
    expect(ransacTrials(1, 8, 0.99)).toBe(1)
    expect(ransacTrials(0, 8, 0.99)).toBe(Infinity)
    // wˢ = 0.05¹² ≈ 2.4e−16: 1 − wˢ rounds to 1 − 2.2e−16, so log(1 − wˢ) was 9% off; log1p is exact to rounding and
    // −log(1 − wˢ) = wˢ to first order (review G2).
    const ws = 0.05 ** 12
    expect(Math.abs(ransacTrials(0.05, 12, 0.99) / (-Math.log(0.01) / ws) - 1)).toBeLessThan(1e-9)
  })

  test('the best inlier count never falls, and the algorithm meets the protocol', () => {
    const alg = ransac(lineProblem, { threshold: 0.2 })
    let s = alg.init(undefined, stream('a'))
    for (let t = 0; t < 20; t++) {
      const next = alg.step(s, { stream: stream(`b${t}`) } as never)
      expect(next.inlierCount).toBeGreaterThanOrEqual(s.inlierCount)
      s = next
    }
    checkProtocol(ransac(lineProblem, { threshold: 0.2 }), undefined, { steps: 5 })
  })
})
