/**
 * Linear latent-variable models against scikit-learn (`fixtures/unsupervised/embedding/linear.json`): factor analysis
 * by EM reaches scikit-learn's maximum-likelihood covariance and log-likelihood; probabilistic PCA's closed form
 * matches numpy and scikit-learn's PCA covariance, and its EM reaches the same optimum; FastICA matches scikit-learn's
 * iterates from the same initial unmixing matrix and recovers the sources. Laws: EM never lowers the likelihood.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromRows, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dataset } from 'aifn-compute/learning/estimators'
import {
  factorAnalysis,
  fastIca,
  latentGaussianSteps,
  probabilisticPca,
} from 'aifn-methods/unsupervised/embedding/linear'
import { fixture } from '../../fixtures'

type F = {
  x: number[][]
  latent: number
  factorAnalysis: { covariance: number[][]; logLikelihood: number }
  ppca: { covariance: number[][]; noiseVariance: number; sklearnCovariance: number[][] }
  ica: {
    x: number[][]
    W0: number[][]
    runs: { steps: number; components: number[][]; mixing: number[][] }[]
    sources: number[][]
  }
}
const F = fixture<F>('unsupervised/embedding/linear')
const X = fromRows(F.x)
const close = (got: ArrayLike<number>, want: number[], tol: number) =>
  want.forEach((w, i) => expect(Math.abs(got[i] - w)).toBeLessThanOrEqual(tol * (1 + Math.abs(w))))

describe('factor analysis', () => {
  it("reaches scikit-learn's maximum-likelihood covariance and log-likelihood", () => {
    const m = factorAnalysis({ latent: F.latent, maxSteps: 5000, tolerance: 1e-13 }).fit(dataset(X), {
      stream: stream(1),
    })
    expect(m.logLikelihood).toBeCloseTo(F.factorAnalysis.logLikelihood, 5)
    close(toFlat(m.covariance), F.factorAnalysis.covariance.flat(), 1e-3)
    expect(m.averageLogLikelihood(X)).toBeCloseTo(m.logLikelihood, 10)
    expect(m.transform(X).shape).toEqual([F.x.length, F.latent])
  })

  for (const noise of ['diagonal', 'isotropic'] as const)
    it(`EM never lowers the log-likelihood (${noise} noise)`, () => {
      const t = trace(latentGaussianSteps(X, { latent: 2, noise, tolerance: 0 }), undefined, 80, { stream: stream(3) })
      const ll = t.steps.map((s) => s.logLikelihood)
      for (let i = 1; i < ll.length; i++) expect(ll[i]).toBeGreaterThanOrEqual(ll[i - 1] - 1e-10)
    })
})

describe('probabilistic PCA', () => {
  it('closed form matches numpy and scikit-learn’s PCA covariance (rescaled to 1/n)', () => {
    const m = probabilisticPca({ latent: F.latent }).fit(dataset(X))
    expect(m.noiseVariance).toBeCloseTo(F.ppca.noiseVariance, 10)
    close(toFlat(m.covariance), F.ppca.covariance.flat(), 1e-9)
    close(toFlat(m.covariance), F.ppca.sklearnCovariance.flat(), 1e-9)
  })

  it('EM reaches the closed-form optimum', () => {
    const closed = probabilisticPca({ latent: F.latent }).fit(dataset(X))
    const em = probabilisticPca({ latent: F.latent, method: 'em', maxSteps: 5000, tolerance: 1e-14 }).fit(dataset(X), {
      stream: stream(2),
    })
    expect(em.logLikelihood).toBeCloseTo(closed.logLikelihood, 7)
    close(toFlat(em.covariance), toFlat(closed.covariance) as unknown as number[], 1e-4)
  })
})

describe('FastICA', () => {
  const Xm = fromRows(F.ica.x)
  for (const r of F.ica.runs)
    it(`matches scikit-learn's components and mixing after ${r.steps} iteration(s) from the same start`, () => {
      const m = fastIca({ maxSteps: r.steps, tolerance: 0, init: fromRows(F.ica.W0) }).fit(dataset(Xm))
      close(toFlat(m.components), r.components.flat(), 1e-7)
      close(toFlat(m.mixing), r.mixing.flat(), 1e-7)
      expect(m.training.final.t).toBe(r.steps)
    })

  it('recovers the sources up to order, sign and scale', () => {
    const m = fastIca({ maxSteps: 200 }).fit(dataset(Xm), { stream: stream(4) })
    const S = toFlat(m.transform(Xm))
    const n = F.ica.sources.length
    for (let c = 0; c < 3; c++) {
      let best = 0
      for (let k = 0; k < 3; k++) {
        const a = Array.from({ length: n }, (_, i) => S[i * 3 + k])
        const b = F.ica.sources.map((row) => row[c])
        const ma = a.reduce((u, v) => u + v, 0) / n
        const mb = b.reduce((u, v) => u + v, 0) / n
        let sab = 0
        let saa = 0
        let sbb = 0
        for (let i = 0; i < n; i++) {
          sab += (a[i] - ma) * (b[i] - mb)
          saa += (a[i] - ma) ** 2
          sbb += (b[i] - mb) ** 2
        }
        best = Math.max(best, Math.abs(sab / Math.sqrt(saa * sbb)))
      }
      expect(best).toBeGreaterThan(0.98)
    }
    const back = toFlat(m.inverseTransform(m.transform(Xm)))
    close(back, F.ica.x.flat(), 1e-8)
  })
})
