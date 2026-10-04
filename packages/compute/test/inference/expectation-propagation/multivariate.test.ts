/** Multivariate EP with rank-one sites: exactness for Gaussian factors, agreement with scalar EP, the fixed point. */
import { describe, expect, it } from 'vitest'
import { run } from 'aifn-compute/foundation/trace'
import { tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { logDet, solve } from 'aifn-compute/numerics/linalg'
import {
  epLogEvidence,
  expectationPropagation,
  multivariateExpectationPropagation,
  probitTilted,
  type Tilted,
} from 'aifn-compute/inference/expectation-propagation'
import { checkProtocol } from '../../protocol'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) =>
  Array.from(a).forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i]))))

/** The tilted moments of a Gaussian factor N(y; x, σ²) against a cavity N(m, v): exact, so EP is exact. */
const gaussianFactor =
  (y: number[], s2: number) =>
  (i: number, c: { mean: number; variance: number }): Tilted => {
    const S = c.variance + s2
    return {
      logZ: -0.5 * (Math.log(2 * Math.PI * S) + (y[i] - c.mean) ** 2 / S),
      mean: c.mean + (c.variance * (y[i] - c.mean)) / S,
      variance: c.variance - (c.variance * c.variance) / S,
    }
  }

const A = [
  [1, 0, 0.5],
  [0, 1, -1],
  [1, 1, 0],
  [0.3, -0.2, 1],
]
const mu0 = [0.5, -0.2, 0.1]
const S0 = [
  [1, 0.3, 0],
  [0.3, 2, 0.4],
  [0, 0.4, 1.5],
]
const y = [0.7, -1.1, 0.4, 1.3]
const s2 = 0.5

describe('multivariateExpectationPropagation', () => {
  it('is exact for Gaussian factors: Bayesian linear regression posterior and evidence', () => {
    const alg = multivariateExpectationPropagation({
      prior: { mean: tensor(mu0), covariance: tensor(S0) },
      projections: tensor(A),
      tilted: gaussianFactor(y, s2),
    })
    const s = run(alg, undefined, 100)
    expect(s.converged).toBe(true)
    // Exact: Σ = (Σ₀⁻¹ + AᵀA/σ²)⁻¹, μ = Σ(Σ₀⁻¹μ₀ + Aᵀy/σ²); evidence log N(y; Aμ₀, AΣ₀Aᵀ + σ²I).
    const precision = toRows(
      solve(
        tensor(S0),
        tensor([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]),
      ),
    )
    const AtA = A[0].map((_, i) => A[0].map((_, j) => A.reduce((acc, r) => acc + (r[i] * r[j]) / s2, 0)))
    const Lambda = precision.map((r, i) => r.map((v, j) => v + AtA[i][j]))
    const h = toFlat(solve(tensor(S0), tensor(mu0))).map(
      (v, i) => v + A.reduce((acc, r, k) => acc + (r[i] * y[k]) / s2, 0),
    )
    close(
      toFlat(s.covariance),
      toFlat(
        solve(
          tensor(Lambda),
          tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ]),
        ),
      ),
      1e-10,
    )
    close(toFlat(s.mean), toFlat(solve(tensor(Lambda), tensor(h))), 1e-10)
    const m = A.map((r) => r.reduce((acc, v, j) => acc + v * mu0[j], 0))
    const C = A.map((ri) =>
      A.map((rj) => ri.reduce((acc, v, a) => acc + v * S0[a].reduce((t, w, b) => t + w * rj[b], 0), 0)),
    )
    C.forEach((r, i) => (r[i] += s2))
    const r = y.map((v, i) => v - m[i])
    const Cr = toFlat(solve(tensor(C), tensor(r)))
    const exact =
      -0.5 *
      (y.length * Math.log(2 * Math.PI) + (logDet(tensor(C)) as number) + r.reduce((a, v, i) => a + v * Cr[i], 0))
    expect(s.logEvidence).toBeCloseTo(exact, 10)
  })
  it('with d = 1 and unit projections it is scalar EP', () => {
    const cuts = [-1, 0.5, 0.2, 1.5, -0.3]
    const signs = [1, 1, -1, -1, 1]
    const tilted = (i: number, c: { mean: number; variance: number }) =>
      probitTilted(c.mean, c.variance, signs[i], { offset: cuts[i] })
    const scalar = run(
      expectationPropagation({ prior: { mean: 0.3, variance: 2 }, factors: 5, tilted }),
      undefined,
      200,
    )
    const mv = run(
      multivariateExpectationPropagation({
        prior: { mean: tensor([0.3]), covariance: tensor([[2]]) },
        projections: tensor(cuts.map(() => [1])),
        tilted,
      }),
      undefined,
      200,
    )
    expect(mv.converged).toBe(true)
    expect(toFlat(mv.mean)[0]).toBeCloseTo(scalar.posterior.mean, 9)
    expect(toFlat(mv.covariance)[0]).toBeCloseTo(scalar.posterior.variance, 9)
    expect(mv.logEvidence).toBeCloseTo(epLogEvidence(scalar), 8)
  })
  it('GP probit classification (identity projections): at the fixed point every tilted marginal matches q', () => {
    const x = [-2, -1, 0, 0.5, 1.5, 2.5]
    const labels = [-1, -1, 1, -1, 1, 1]
    const K = x.map((a) => x.map((b) => 1.5 * Math.exp(-0.5 * ((a - b) / 1.2) ** 2) + (a === b ? 1e-8 : 0)))
    const tilted = (i: number, c: { mean: number; variance: number }) => probitTilted(c.mean, c.variance, labels[i])
    const s = run(
      multivariateExpectationPropagation({ prior: { mean: tensor(x.map(() => 0)), covariance: tensor(K) }, tilted }),
      undefined,
      2000,
    )
    expect(s.converged).toBe(true)
    expect(Number.isFinite(s.logEvidence)).toBe(true)
    const S = toRows(s.covariance)
    const m = toFlat(s.mean)
    const tau = toFlat(s.sitePrecision)
    const nu = toFlat(s.siteShift)
    x.forEach((_, i) => {
      const ct = 1 / S[i][i] - tau[i]
      const cn = m[i] / S[i][i] - nu[i]
      const t = tilted(i, { mean: cn / ct, variance: 1 / ct })
      expect(t.mean).toBeCloseTo(m[i], 6)
      expect(t.variance).toBeCloseTo(S[i][i], 6)
    })
    // The smooth latent mean follows the labels at the ends of the line.
    expect(Math.sign(m[0])).toBe(-1)
    expect(Math.sign(m[5])).toBe(1)
  })
  it('satisfies the Algorithm protocol', () => {
    checkProtocol(
      multivariateExpectationPropagation({
        prior: {
          mean: tensor([0, 0]),
          covariance: tensor([
            [1, 0.2],
            [0.2, 1],
          ]),
        },
        tilted: (i, c) => probitTilted(c.mean, c.variance, i === 0 ? 1 : -1),
      }),
      undefined,
      { steps: 6 },
    )
  })
})
