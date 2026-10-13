/**
 * aifn-compute/learning/selection, Bayesian predictive criteria: `psisLoo` and `psisWeights` against arviz-stats 1.3
 * (`loo` with `reff=1`, `psislw`) on fixed log-likelihood draws, for both tail-length rules (3 sqrt(S) above 225 draws,
 * S / 5 below); `waic` against its definition in numpy (Vehtari, Gelman and Gabry, 2017); `dic` in closed form for a
 * normal mean with known variance.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { dic, psisLoo, psisWeights, waic } from 'aifn-compute/learning/selection'
import { fixture } from '../../fixtures'

interface Case {
  logLik: number[][]
  loo: { elpd: number; se: number; p: number; elpdPointwise: number[]; khat: number[]; goodK: number }
  logWeights: number[][]
  khat: number[]
  waic: { elpd: number; p: number; se: number; elpdPointwise: number[]; pPointwise: number[]; lppd: number }
}

const F = fixture<{ predictive: Case[] }>('learning/selection')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
}

describe.each(F.predictive.map((c) => [`${c.logLik.length} draws`, c] as const))('%s', (_, c) => {
  const S = c.logLik.length
  const n = c.logLik[0].length

  it('psisLoo matches arviz-stats loo', () => {
    const loo = psisLoo(tensor(c.logLik))
    expect(loo.elpd).toBeCloseTo(c.loo.elpd, 9)
    expect(loo.se).toBeCloseTo(c.loo.se, 9)
    expect(loo.p).toBeCloseTo(c.loo.p, 9)
    close(toFlat(loo.pointwise.elpd), c.loo.elpdPointwise, 1e-10)
    close(toFlat(loo.khat), c.loo.khat, 1e-10)
    expect(loo.khatThreshold).toBeCloseTo(c.loo.goodK, 14)
    expect(loo.bad).toBe(c.loo.khat.filter((k) => k > c.loo.goodK).length)
    expect(loo.looic).toBeCloseTo(-2 * c.loo.elpd, 9)
    expect(loo.lppd - loo.p).toBeCloseTo(loo.elpd, 10)
  })

  it('psisWeights matches arviz-stats psislw', () => {
    ;[0, n - 1].forEach((i, j) => {
      const w = psisWeights(c.logLik.map((row) => -row[i]))
      expect(w.khat).toBeCloseTo(c.khat[j], 10)
      close(toFlat(w.logWeights), c.logWeights[j], 1e-10)
      expect(w.tailLength).toBe(S > 225 ? Math.floor(3 * Math.sqrt(S)) : Math.floor(S / 5))
    })
  })

  it('waic matches its definition', () => {
    const w = waic(c.logLik)
    expect(w.elpd).toBeCloseTo(c.waic.elpd, 10)
    expect(w.p).toBeCloseTo(c.waic.p, 10)
    expect(w.se).toBeCloseTo(c.waic.se, 10)
    expect(w.lppd).toBeCloseTo(c.waic.lppd, 10)
    close(toFlat(w.pointwise.elpd), c.waic.elpdPointwise, 1e-10)
    close(toFlat(w.pointwise.p), c.waic.pPointwise, 1e-10)
    expect(w.waic).toBeCloseTo(-2 * c.waic.elpd, 10)
  })
})

describe('psis edge cases', () => {
  it('throws for too few draws; reports NaN k-hat for a constant tail, leaving the weights uniform', () => {
    expect(() => psisLoo(tensor(Array.from({ length: 20 }, () => [0])))).toThrow(/tail/)
    const flat = psisWeights(new Array(100).fill(1.5))
    expect(flat.khat).toBeNaN()
    close(toFlat(flat.logWeights), new Array(100).fill(-Math.log(100)), 1e-12)
  })
})

describe('dic', () => {
  it('normal mean, known variance, flat prior: p_D = 1 exactly on symmetric draws, p_V near 1', () => {
    // y_i ~ N(mu, 1); posterior mu ~ N(ybar, 1/n). The log-likelihood is quadratic in mu, so
    // p_D = n * mean((mu_s - ybar)^2) exactly when the draws are centred at ybar.
    const y = [0.2, -0.5, 1.1, 0.4, -0.1, 0.9, 0.3, -0.7]
    const n = y.length
    const ybar = y.reduce((a, b) => a + b, 0) / n
    const z = toFlat(normals(stream(11), [2000]))
    const zbar = z.reduce((a, b) => a + b, 0) / z.length
    const mus = z.map((v) => ybar + (v - zbar) / Math.sqrt(n))
    const ll = (mu: number) => y.map((v) => -0.5 * (v - mu) ** 2 - 0.5 * Math.log(2 * Math.PI))
    const matrix = mus.map(ll)
    const d = dic(matrix, ll(ybar))
    const spread = mus.reduce((a, m) => a + (m - ybar) ** 2, 0) / mus.length
    expect(d.pD).toBeCloseTo(n * spread, 10)
    expect(d.dic).toBeCloseTo(d.meanDeviance + d.pD, 12)
    expect(d.dic).toBeCloseTo(d.devianceAtMean + 2 * d.pD, 10)
    // p_V = var(D) / 2 with D = n (mu - ybar)^2 + const, a chi-squared variable with one degree of freedom.
    const chi = mus.map((m) => n * (m - ybar) ** 2)
    const chiMean = chi.reduce((a, b) => a + b, 0) / chi.length
    expect(d.pV).toBeCloseTo(chi.reduce((a, v) => a + (v - chiMean) ** 2, 0) / (chi.length - 1) / 2, 9)
    expect(Math.abs(d.pV - 1)).toBeLessThan(0.3)
    // Totals per draw and a summed log-likelihood at the mean give the same.
    const totals = matrix.map((r) => r.reduce((a, b) => a + b, 0))
    const e = dic(
      totals,
      ll(ybar).reduce((a, b) => a + b, 0),
    )
    expect(e.dic).toBeCloseTo(d.dic, 10)
    expect(e.pV).toBeCloseTo(d.pV, 10)
  })
})
