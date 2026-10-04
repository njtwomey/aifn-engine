import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { expectile, expectiles } from 'aifn-compute/probability/stats'
import { expectileLoss } from 'aifn-compute/learning/losses'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Normal } from 'aifn-compute/probability/distributions'

/** Σ wᵢ ρ_τ(xᵢ − m), the objective the expectile minimises. */
const objective = (x: number[], tau: number, m: number, w?: number[]) =>
  x.reduce((s, v, i) => s + (w ? w[i] : 1) * (v < m ? 1 - tau : tau) * (v - m) ** 2, 0)

describe('expectile', () => {
  it('matches the worked example of the note: 1, 2, 3, 4, 10', () => {
    const x = [10, 3, 1, 4, 2]
    expect(expectile(x, 0.5)).toBeCloseTo(4, 12)
    expect(expectile(x, 0.7)).toBeCloseTo(10 / 1.9, 12)
  })

  it('minimises the asymmetric squared loss, with weights', () => {
    const x = [0.3, -1.2, 2.5, 0.9, 0.1, 4, -0.4]
    const w = [1, 2, 0.5, 1, 3, 0.2, 1]
    for (const tau of [0.02, 0.2, 0.5, 0.8, 0.97]) {
      const m = expectile(x, tau, { weights: w })
      for (const d of [-1e-4, 1e-4]) expect(objective(x, tau, m + d, w)).toBeGreaterThan(objective(x, tau, m, w))
    }
    // τ = ½ is the weighted mean.
    const mean = x.reduce((s, v, i) => s + w[i] * v, 0) / w.reduce((a, b) => a + b, 0)
    expect(expectile(x, 0.5, { weights: w })).toBeCloseTo(mean, 12)
  })

  it('of a standard normal from its quantiles: e₀.₉ = 0.862 (Φ(0.862) = 0.806)', () => {
    const m = 20000
    const levels = fromData(
      Float64Array.from({ length: m }, (_, i) => (i + 0.5) / m),
      [m],
    )
    const z = Normal(0, 1).quantile(levels) as Tensor
    expect(expectile(z, 0.9)).toBeCloseTo(0.8616, 3)
  })

  it('increases with τ; expectiles sorts once', () => {
    const x = [5, 1, 1, 2, 9, 3]
    const e = expectiles(x, [0.1, 0.4, 0.6, 0.9])
    for (let i = 1; i < e.length; i++) expect(e[i]).toBeGreaterThan(e[i - 1])
    expect(e[2]).toBeCloseTo(expectile(x, 0.6), 14)
  })

  it('rejects τ outside (0, 1) and empty data', () => {
    expect(() => expectile([1, 2], 1)).toThrow(DomainError)
    expect(() => expectile([], 0.5)).toThrow()
  })
})

describe('expectileLoss', () => {
  it('weights under-predictions by τ and over-predictions by 1 − τ', () => {
    const loss = toFlat(
      expectileLoss(fromData(new Float64Array(2), [2]), [2, -1], { expectile: 0.9, reduction: 'none' }) as Tensor,
    )
    expect(loss[0]).toBeCloseTo(0.9 * 4, 12)
    expect(loss[1]).toBeCloseTo(0.1 * 1, 12)
  })
})
