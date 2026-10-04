import { describe, expect, it } from 'vitest'
import { caviGaussianMixture, mixturePredictiveDensity } from 'aifn-methods/inference/mixture-models'
import { child, normal, normals, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'

describe('CAVI Gaussian mixture', () => {
  it('CAVI Gaussian mixture: monotone ELBO, recovers well-separated components', () => {
    const s0 = stream('gmm')
    const x: number[][] = []
    const centres = [
      [-3, 0],
      [3, 1],
      [0, 4],
    ]
    for (let n = 0; n < 300; n++) {
      const c2 = centres[n % 3]
      const e = toFlat(normals(child(s0, n), 2))
      x.push([c2[0] + 0.6 * e[0], c2[1] + 0.6 * e[1]])
    }
    const alg = caviGaussianMixture(x, 6, { alpha0: 1e-3 })
    const tr = trace(alg, undefined, 300, { stream: stream(1), record: { elbo: (s) => s.elbo } })
    const e = toFlat(tr.series.elbo)
    for (let i = 1; i < e.length; i++) expect(e[i]).toBeGreaterThanOrEqual(e[i - 1] - 1e-8 * Math.abs(e[i]))
    const last = tr.steps.at(-1)!
    const w = toFlat(last.weights)
    const used = w.filter((v) => v > 0.05).length
    expect(used).toBe(3)
    const means = toFlat(last.means)
    for (const c2 of centres) {
      const hit = w.some((v, k) => v > 0.05 && Math.hypot(means[2 * k] - c2[0], means[2 * k + 1] - c2[1]) < 0.3)
      expect(hit).toBe(true)
    }
    // The predictive density integrates to about 1 (1-D check below) and is positive.
    const dens = toFlat(mixturePredictiveDensity(last, [[-3, 0]]).density)
    expect(dens[0]).toBeGreaterThan(0)
  })

  it('1-D mixture predictive density integrates to 1', () => {
    const xs = Array.from({ length: 200 }, (_, n) => normal(child(stream('x'), n), n % 2 ? 2 : -2, 0.7))
    const s = run(caviGaussianMixture(xs, 2), undefined, 200, { stream: stream(1) })
    const grid = Array.from({ length: 2001 }, (_, i) => -10 + i * 0.01)
    const d = toFlat(mixturePredictiveDensity(s, grid).density as Tensor)
    expect(d.reduce((a, b) => a + b, 0) * 0.01).toBeCloseTo(1, 3)
  })
})
