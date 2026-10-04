/**
 * The pieces of label propagation for label proportions: Gaussian point affinities (exp(−γd²), zero diagonal,
 * symmetric k-NN restriction), the random-walk matrix D⁻¹W (rows sum to one) and the spreading resolvent
 * (1 − α)(I − αS)⁻¹, which inverts (I − αS)/(1 − α) and is the limit of F ← αSF + (1 − α)Y.
 */
import { describe, expect, it } from 'vitest'
import { dense, fromRows } from 'aifn-compute/foundation/tensor'
import { pointAffinity, randomWalkMatrix, spreadingResolvent } from 'aifn-compute/graph/propagation'

const X = fromRows([
  [0, 0],
  [1, 0],
  [0, 2],
  [3, 1],
  [-1, -1],
])
const n = 5

describe('pointAffinity', () => {
  it('is exp(−γ‖xᵢ − xⱼ‖²) off the diagonal, 0 on it', () => {
    const W = dense.data(pointAffinity(X, { gamma: 0.7 }))
    const P = dense.data(X)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const d2 = (P[2 * i] - P[2 * j]) ** 2 + (P[2 * i + 1] - P[2 * j + 1]) ** 2
        expect(W[i * n + j]).toBeCloseTo(i === j ? 0 : Math.exp(-0.7 * d2), 14)
      }
  })
  it('keeps a symmetric k-NN graph', () => {
    const W = dense.data(pointAffinity(X, { gamma: 0.1, neighbours: 1 }))
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) expect(W[i * n + j]).toBe(W[j * n + i])
      expect([...W.slice(i * n, (i + 1) * n)].filter((w) => w > 0).length).toBeGreaterThanOrEqual(1)
    }
  })
})

describe('randomWalkMatrix and spreadingResolvent', () => {
  const W = pointAffinity(X, { gamma: 0.3 })
  const S = dense.data(randomWalkMatrix(W))
  it('the walk’s rows sum to one', () => {
    for (let i = 0; i < n; i++) expect(S.slice(i * n, (i + 1) * n).reduce((a, v) => a + v, 0)).toBeCloseTo(1, 14)
  })
  it('R(I − αS) = (1 − α)I, and R is row-stochastic', () => {
    const a = 0.6
    const R = dense.data(spreadingResolvent(randomWalkMatrix(W), a))
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let v = 0
        for (let k = 0; k < n; k++) v += R[i * n + k] * ((k === j ? 1 : 0) - a * S[k * n + j])
        expect(v).toBeCloseTo(i === j ? 1 - a : 0, 12)
      }
      expect(R.slice(i * n, (i + 1) * n).reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
    }
  })
  it('is the limit of F ← αSF + (1 − α)Y', () => {
    const a = 0.5
    const Y = [1, 0, 0, 0, 1]
    let F = [...Y]
    for (let t = 0; t < 200; t++)
      F = F.map((_, i) => a * S.slice(i * n, (i + 1) * n).reduce((s, v, j) => s + v * F[j], 0) + (1 - a) * Y[i])
    const R = dense.data(spreadingResolvent(randomWalkMatrix(W), a))
    for (let i = 0; i < n; i++)
      expect(F[i]).toBeCloseTo(
        R.slice(i * n, (i + 1) * n).reduce((s, v, j) => s + v * Y[j], 0),
        10,
      )
    const raw = dense.data(spreadingResolvent(randomWalkMatrix(W), a, { scaled: false }))
    for (let k = 0; k < n * n; k++) expect(raw[k] * (1 - a)).toBeCloseTo(R[k], 12)
  })
})
