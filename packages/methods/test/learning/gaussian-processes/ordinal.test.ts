/**
 * Gaussian-process ordinal regression (Chu & Ghahramani, 2005) by Laplace, against a direct numpy Laplace
 * (fixtures/gen/learning/generalised/ordinal.py, case `gp`).
 */
import { describe, expect, it } from 'vitest'
import { fromData, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { gpOrdinalRegression, ordinalLaplaceTerms } from 'aifn-methods/learning/gaussian-processes'
import { learningModelRegistry } from 'aifn-methods/learning'
import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'
import { fixture } from '../../fixtures'

type Gp = {
  n: number
  theta: number[]
  sigma: number
  mode: number[]
  logMarginal: number
  mean: number[]
  variance: number[]
  proba: number[][]
}
const F = fixture<{ x: number[][]; y: number[]; xq: number[][]; gp: Gp }>('learning/generalised/ordinal')
const G = F.gp
const X = tensor(F.x.slice(0, G.n))
const Y = fromData(Int32Array.from(F.y.slice(0, G.n)), [G.n])
const XQ = tensor(F.xq)

function close(got: ArrayLike<number>, want: number[], tol: number) {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) expect(Math.abs(got[i] - want[i]), `[${i}]`).toBeLessThan(tol)
}

describe('gpOrdinalRegression', () => {
  const fixed = () =>
    gpOrdinalRegression({ kernel: rbf(), thresholds: G.theta, noise: G.sigma, optimise: false }).fit(dataset(X, Y))

  it('at fixed hyperparameters: the mode, evidence and predictions match a direct Laplace', () => {
    const m = fixed()
    expect(m.converged).toBe(true)
    close(toFlat(m.mode), G.mode, 1e-7)
    expect(m.logMarginal).toBeCloseTo(G.logMarginal, 8)
    const { mean, variance } = m.latent(XQ)
    close(toFlat(mean), G.mean, 1e-7)
    close(toFlat(variance), G.variance, 1e-7)
    close(toFlat(m.probabilities(XQ)), G.proba.flat(), 1e-7)
  })

  it('the autodiff likelihood terms equal the closed form of the cumulative probit', () => {
    // One observation of class 1 between θ = (−1, 0.3, 1.5), σ = 0.5, at f = 0.2.
    const terms = ordinalLaplaceTerms(Int32Array.from([1]), [-1, 0.3, 1.5], 0.5)
    const { logLik, grad, W } = terms(Float64Array.from([0.2]))
    const phi = normalPdf
    const Phi = normalCdf
    const z1 = (0.3 - 0.2) / 0.5
    const z2 = (-1 - 0.2) / 0.5
    const Z = Phi(z1) - Phi(z2)
    expect(logLik).toBeCloseTo(Math.log(Z), 12)
    const r = (phi(z1) - phi(z2)) / Z
    expect(grad[0]).toBeCloseTo(-r / 0.5, 10)
    expect(W[0]).toBeCloseTo((r * r + (z1 * phi(z1) - z2 * phi(z2)) / Z) / 0.25, 10)
  })

  it('optimising the hyperparameters raises the evidence and keeps the thresholds ordered', () => {
    const before = fixed().logMarginal
    const m = gpOrdinalRegression({ kernel: rbf(), thresholds: G.theta, noise: G.sigma, hyperSteps: 150 }).fit(
      dataset(X, Y),
    )
    expect(m.logMarginal).toBeGreaterThan(before)
    const t = toFlat(m.thresholds)
    for (let k = 1; k < t.length; k++) expect(t[k]).toBeGreaterThan(t[k - 1])
    const P = toFlat(m.probabilities(XQ))
    for (let i = 0; i < F.xq.length; i++)
      expect(P.slice(i * 4, i * 4 + 4).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('is registered with its capabilities and rejects bad labels', () => {
    expect(learningModelRegistry.gpOrdinalRegression.info.capabilities).toEqual([
      'forward',
      'decide',
      'predictive',
      'expect',
      'score',
      'sample',
    ])
    expect(() =>
      gpOrdinalRegression({ kernel: rbf() }).fit(dataset(X, tensor(F.y.slice(0, G.n).map((v) => v + 0.5)))),
    ).toThrow(/class indices/)
  })
})
