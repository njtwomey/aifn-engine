import { describe, expect, it } from 'vitest'
import * as I from 'aifn-compute/probability/information'
import { Normal } from 'aifn-compute/probability/distributions'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

/** Information measures against scipy, scikit-learn and direct numpy (gen/probability/information.py). */
type Fixture = {
  p: number[]
  q: number[]
  qz: number[]
  P: number[][]
  Q: number[][]
  entropy: { p: number; pBits: number; rows: number[] }
  klDivergence: { pq: number; pqBits: number; pqz: number; rows: number[] }
  crossEntropy: { pq: number; rows: number[] }
  jensenShannonDivergence: { pq: number; pqBits: number }
  jensenShannonDistance: { pq: number; pqBits: number }
  totalVariation: { pq: number }
  hellingerDistance: { pq: number }
  fDivergence: Record<keyof typeof I.fGenerators, number>
  fDivergenceZero: Record<string, number>
  joint: number[][]
  jointEntropy: number
  conditionalEntropy: { givenX: number; givenY: number }
  mutualInformation: { nats: number; bits: number }
  pointwiseMutualInformation: { pmi: number[][]; npmi: number[][] }
  gaussianMutualInformation: { covariance: number[][]; x: number[]; y: number[]; value: number; scalar: number }
  ksgMutualInformation: {
    x: number[]
    y: number[]
    k3: number
    k7: number
    x2: number[][]
    y2: number[][]
    k4MultiDim: number
  }
  differentialEntropy: { normalBits: number }
}
const F = fixture<Fixture>('probability/information')

const num = (v: unknown) => v as number
const flat = (v: unknown) => Array.from(toFlat(v as Tensor))
const near = (got: number, want: number, tol = 1e-13) =>
  Number.isFinite(want)
    ? expect(Math.abs(got - want), `${got} vs ${want}`).toBeLessThanOrEqual(tol * (1 + Math.abs(want)))
    : expect(got).toBe(want)
const nearAll = (got: number[], want: number[], tol = 1e-13) => want.forEach((w, i) => near(got[i], w, tol))

describe('entropies and divergences match scipy.stats.entropy', () => {
  it('entropy of counts, in bits, and per row', () => {
    near(num(I.entropy(F.p)), F.entropy.p)
    near(num(I.entropy(F.p, { base: 2 })), F.entropy.pBits)
    nearAll(flat(I.entropy(tensor(F.P))), F.entropy.rows)
  })
  it('KL divergence, infinite where q has no mass, and per row', () => {
    near(num(I.klDivergence(F.p, F.q)), F.klDivergence.pq)
    near(num(I.klDivergence(F.p, F.q, { base: 2 })), F.klDivergence.pqBits)
    near(num(I.klDivergence(F.p, F.qz)), F.klDivergence.pqz)
    nearAll(flat(I.klDivergence(tensor(F.P), tensor(F.Q))), F.klDivergence.rows)
  })
  it('cross entropy', () => {
    near(num(I.crossEntropy(F.p, F.q)), F.crossEntropy.pq)
    nearAll(flat(I.crossEntropy(tensor(F.P), tensor(F.Q))), F.crossEntropy.rows)
  })
  it('Jensen–Shannon divergence and distance (scipy.spatial.distance.jensenshannon)', () => {
    near(num(I.jensenShannonDivergence(F.p, F.q)), F.jensenShannonDivergence.pq)
    near(num(I.jensenShannonDivergence(F.p, F.q, { base: 2 })), F.jensenShannonDivergence.pqBits)
    near(num(I.jensenShannonDistance(F.p, F.q)), F.jensenShannonDistance.pq)
    near(num(I.jensenShannonDistance(F.p, F.q, { base: 2 })), F.jensenShannonDistance.pqBits)
  })
  it('total variation, Hellinger and every f-divergence generator', () => {
    near(num(I.totalVariation(F.p, F.q)), F.totalVariation.pq)
    near(num(I.hellingerDistance(F.p, F.q)), F.hellingerDistance.pq)
    for (const [name, value] of Object.entries(F.fDivergence))
      near(I.fDivergence(F.p, F.q, I.fGenerators[name as keyof typeof I.fGenerators]), value)
    for (const [name, value] of Object.entries(F.fDivergenceZero))
      near(I.fDivergence(F.p, F.qz, I.fGenerators[name as keyof typeof I.fGenerators]), value)
  })
})

describe('joint tables', () => {
  it('joint and conditional entropy', () => {
    near(num(I.jointEntropy(F.joint)), F.jointEntropy)
    near(num(I.conditionalEntropy(F.joint)), F.conditionalEntropy.givenX)
    near(num(I.conditionalEntropy(F.joint, { given: 'y' })), F.conditionalEntropy.givenY)
  })
  it("mutual information equals scikit-learn's mutual_info_score", () => {
    near(num(I.mutualInformation(F.joint)), F.mutualInformation.nats)
    near(num(I.mutualInformation(F.joint, { base: 2 })), F.mutualInformation.bits)
  })
  it('pointwise mutual information, plain and normalised', () => {
    nearAll(flat(I.pointwiseMutualInformation(F.joint)), F.pointwiseMutualInformation.pmi.flat())
    nearAll(flat(I.pointwiseMutualInformation(F.joint, { normalised: true })), F.pointwiseMutualInformation.npmi.flat())
  })
})

describe('continuous mutual information', () => {
  it('Gaussian closed form', () => {
    const g = F.gaussianMutualInformation
    near(num(I.gaussianMutualInformation(tensor(g.covariance), g.x, g.y)), g.value, 1e-12)
    near(num(I.gaussianMutualInformation(tensor(g.covariance), [0], [1])), g.scalar, 1e-12)
  })
  it("KSG matches scikit-learn's estimator and a direct multivariate computation", () => {
    const k = F.ksgMutualInformation
    near(I.ksgMutualInformation(k.x, k.y, { k: 3 }), k.k3, 1e-12)
    near(I.ksgMutualInformation(k.x, k.y, { k: 7 }), k.k7, 1e-12)
    near(I.ksgMutualInformation(k.x2, k.y2, { k: 4 }), k.k4MultiDim, 1e-12)
  })
  it('differential entropy in bits', () => {
    near(num(I.differentialEntropy(Normal(0.3, 2.5), { base: 2 })), F.differentialEntropy.normalBits)
  })
})
