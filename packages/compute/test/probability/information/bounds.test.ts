/** aifn-compute/probability/information: the InfoNCE, NWJ and Donsker–Varadhan bounds from critic scores. */
import { describe, expect, it } from 'vitest'
import {
  correlationForMutualInformation,
  donskerVaradhanBound,
  infoNceBound,
  informationFunctions,
  nwjBound,
} from 'aifn-compute/probability/information'

// The worked 3 × 3 batch of the theory-of-representation-learning note.
const T = [2.0, 0.1, -0.3, 0.4, 1.5, 0.0, -0.2, 0.3, 2.5]

describe('mutual-information bounds', () => {
  it('give the worked batch’s values', () => {
    expect(infoNceBound(T, 3)).toBeCloseTo(0.822295560882564, 12)
    expect(nwjBound(T, 3)).toBeCloseTo(1.6010713257093847, 12)
    expect(donskerVaradhanBound(T, 3)).toBeCloseTo(1.9189726392483597, 12)
  })

  it('InfoNCE reaches log n for separated scores and never exceeds it', () => {
    const big = Array.from({ length: 16 }, (_, k) => (k % 5 === 0 ? 50 : 0))
    expect(Math.abs(infoNceBound(big, 4) - Math.log(4))).toBeLessThan(1e-9)
    expect(infoNceBound(big, 4)).toBeLessThanOrEqual(Math.log(4) + 1e-12)
    const huge = Array.from({ length: 9 }, (_, k) => (k % 4 === 0 ? 1e4 : -1e4))
    expect(infoNceBound(huge, 3)).toBeCloseTo(Math.log(3), 12)
  })

  it('InfoNCE is log n minus the cross-entropy of the score rows against the diagonal', () => {
    let ce = 0
    for (let i = 0; i < 3; i++) {
      const row = T.slice(3 * i, 3 * i + 3)
      const lse = Math.log(row.reduce((a, v) => a + Math.exp(v), 0))
      ce += (lse - row[i]) / 3
    }
    expect(infoNceBound(T, 3)).toBeCloseTo(Math.log(3) - ce, 12)
  })

  it('DV is invariant to a constant shift of the critic; NWJ is not', () => {
    const shifted = T.map((v) => v + 5)
    expect(donskerVaradhanBound(shifted, 3)).toBeCloseTo(donskerVaradhanBound(T, 3), 12)
    expect(infoNceBound(shifted, 3)).toBeCloseTo(infoNceBound(T, 3), 12)
    expect(nwjBound(shifted, 3)).not.toBeCloseTo(nwjBound(T, 3), 3)
  })

  it('correlationForMutualInformation inverts -½ log(1 - ρ²) per dimension', () => {
    const rho = correlationForMutualInformation(2)
    expect(-0.5 * Math.log(1 - rho ** 2)).toBeCloseTo(2, 12)
    const r3 = correlationForMutualInformation(1.5, 3)
    expect(-1.5 * Math.log(1 - r3 ** 2)).toBeCloseTo(1.5, 12)
    expect(correlationForMutualInformation(0)).toBe(0)
  })

  it('refuses bad shapes and arguments, and is registered', () => {
    expect(() => infoNceBound(T, 2)).toThrow(/n²/)
    expect(() => nwjBound([1], 1)).toThrow(/at least 2/)
    expect(() => correlationForMutualInformation(-1)).toThrow(/non-negative/)
    for (const key of ['infoNceBound', 'nwjBound', 'donskerVaradhanBound', 'correlationForMutualInformation'])
      expect(informationFunctions[key].info.key).toBe(key)
  })
})
