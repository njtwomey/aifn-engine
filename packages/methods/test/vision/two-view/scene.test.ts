import { describe, expect, it } from 'vitest'
import { fundamentalProblem, homographyProblem, twoViewScene } from 'aifn-methods/vision/two-view'
import { applyHomography, sampsonDistance } from 'aifn-compute/numerics/geometry'
import { ransacFit } from 'aifn-compute/numerics/robust'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

describe('two-view scene and RANSAC', () => {
  it('noiseless inliers satisfy the true H and F exactly', () => {
    const s = twoViewScene(stream('scene'), { noise: 0, outlierFraction: 0 })
    const mapped = toFlat(applyHomography(s.H!, s.x1))
    toFlat(s.x2).forEach((v, i) => expect(v).toBeCloseTo(mapped[i], 6))
    expect(Math.max(...toFlat(sampsonDistance(s.F, s.x1, s.x2)))).toBeLessThan(1e-12)
    const d = twoViewScene(stream('scene'), { noise: 0, outlierFraction: 0, kind: 'depth' })
    expect(d.H).toBeNull()
    expect(Math.max(...toFlat(sampsonDistance(d.F, d.x1, d.x2)))).toBeLessThan(1e-12)
  })

  it('RANSAC recovers the homography and the inliers through 40% outliers', () => {
    const s = twoViewScene(stream('h'), { outlierFraction: 0.4, noise: 0.5 })
    const fit = ransacFit(homographyProblem(s.x1, s.x2), { threshold: 3, stream: stream('r') })
    const agree = fit.inliers.filter((v, i) => v === 1 - s.outlier[i]).length
    expect(agree / s.outlier.length).toBeGreaterThan(0.95)
    const err = toFlat(applyHomography(fit.model!, s.x1)).map((v, i) => v - toFlat(applyHomography(s.H!, s.x1))[i])
    expect(Math.max(...err.map(Math.abs))).toBeLessThan(3)
  })

  it('RANSAC recovers the epipolar geometry of a scene in depth', () => {
    const s = twoViewScene(stream('f'), { kind: 'depth', outlierFraction: 0.3, noise: 0.3, count: 120 })
    const fit = ransacFit(fundamentalProblem(s.x1, s.x2), { threshold: 1.5, stream: stream('r'), maxSamples: 3000 })
    const truth = s.outlier.map((o) => 1 - o)
    const tp = fit.inliers.filter((v, i) => v && truth[i]).length
    expect(tp / truth.filter(Boolean).length).toBeGreaterThan(0.9)
    const fp = fit.inliers.filter((v, i) => v && !truth[i]).length
    expect(fp).toBeLessThan(6)
  })
})
