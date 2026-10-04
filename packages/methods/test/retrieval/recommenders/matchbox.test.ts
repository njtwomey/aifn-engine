import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import {
  matchbox,
  matchboxPredict,
  matchboxRatings,
  matchboxRun,
  matchboxUpdate,
  type MatchboxRun,
} from 'aifn-methods/retrieval/recommenders'

const last = <T>(g: Generator<T, T>): T => {
  let r = g.next()
  let v = r.value
  while (!r.done) {
    v = r.value
    r = g.next()
  }
  return r.value ?? v
}

const corr = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  const n = a.length
  const ma = Array.from(a).reduce((s, v) => s + v, 0) / n
  const mb = Array.from(b).reduce((s, v) => s + v, 0) / n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb)
    saa += (a[i] - ma) ** 2
    sbb += (b[i] - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

describe('Matchbox (ADF on the bilinear ordinal model)', () => {
  const x = { index: [0, 3], value: [1, 0.5] }
  const y = { index: [1], value: [1] }

  it('an update shrinks every touched variance and moves the prediction towards the rating', () => {
    const m0 = matchbox(stream('m'), 4, 3, { levels: 5 })
    for (const level of [0, 4]) {
      const m1 = matchboxUpdate(m0, x, y, level)
      for (const key of ['U', 'V', 'userBias', 'itemBias', 'thresholds'] as const)
        m1[key].variance.forEach((v, i) => expect(v).toBeLessThanOrEqual(m0[key].variance[i] + 1e-12))
      const before = matchboxPredict(m0, x, y).expected
      const after = matchboxPredict(m1, x, y).expected
      expect(Math.abs(after - level)).toBeLessThan(Math.abs(before - level))
      // Untouched features keep their prior.
      expect(m1.U.variance[1]).toBe(m0.U.variance[1])
    }
  })

  it('predictive probabilities form a distribution and thresholds stay ordered', () => {
    let m = matchbox(stream('m2'), 4, 3, { levels: 4 })
    const levels = [3, 0, 2, 1, 3, 3, 0]
    for (const l of levels) m = matchboxUpdate(m, x, y, l)
    const p = matchboxPredict(m, x, y).probabilities
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    for (let l = 1; l < m.thresholds.mean.length; l++)
      expect(m.thresholds.mean[l]).toBeGreaterThan(m.thresholds.mean[l - 1])
  })

  it('recovers the latent affinities and item trait space from synthetic ratings', () => {
    const data = matchboxRatings(stream('recovery'), { users: 200, items: 60, perUser: 40, noise: 0.3 })
    const run: MatchboxRun = last(matchboxRun(data, { seed: 1, checkpoints: 4 }))
    const K = 2
    // Affinity uᵀv for held-out pairs of known users: true against the posterior mean of Σ sₖtₖ.
    const { user, item } = run.test
    const truth = Array.from(user, (u, q) => {
      let a = 0
      for (let k = 0; k < K; k++) a += data.userTraits[u * K + k] * data.itemTraits[item[q] * K + k]
      return a
    })
    const fit = Array.from(user, (u, q) => {
      const p = matchboxPredict(
        run.model,
        { index: [u, 200, 201, 202], value: [1, ...data.userX.subarray(u * 3, u * 3 + 3)] },
        { index: [item[q], 60, 61, 62], value: [1, ...data.itemY.subarray(item[q] * 3, item[q] * 3 + 3)] },
      )
      return p.latentMean
    })
    expect(corr(truth, fit)).toBeGreaterThan(0.85)
    // Item traits are identified up to an invertible linear map: each true trait is a linear function of the fitted
    // ones (least squares in K = 2 dimensions, R² high).
    const T = run.checkpoints.at(-1)!.itemTraits
    for (let k = 0; k < K; k++) {
      const yk = Array.from({ length: 60 }, (_, i) => data.itemTraits[i * K + k])
      let a11 = 0
      let a12 = 0
      let a22 = 0
      let b1 = 0
      let b2 = 0
      for (let i = 0; i < 60; i++) {
        a11 += T[i * K] ** 2
        a12 += T[i * K] * T[i * K + 1]
        a22 += T[i * K + 1] ** 2
        b1 += T[i * K] * yk[i]
        b2 += T[i * K + 1] * yk[i]
      }
      const det = a11 * a22 - a12 * a12
      const c1 = (a22 * b1 - a12 * b2) / det
      const c2 = (a11 * b2 - a12 * b1) / det
      const pred = Array.from({ length: 60 }, (_, i) => c1 * T[i * K] + c2 * T[i * K + 1])
      expect(corr(pred, yk) ** 2).toBeGreaterThan(0.7)
    }
    const c = run.checkpoints.at(-1)!
    expect(c.rmse).toBeLessThan(0.8 * run.itemMeanRmse)
    expect(c.coldRmse).toBeLessThan(0.85 * run.coldGlobalRmse)
    expect(c.traitVariance).toBeLessThan(run.checkpoints[0].traitVariance / 5)
  })
})
