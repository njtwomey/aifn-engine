import { describe, expect, it } from 'vitest'
import { fusedLassoSignal, proxTotalVariation1d, totalVariationDenoise1d } from 'aifn-compute/optim/proximal'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { normals, stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'

type Fixture = {
  totalVariation: {
    signals: Record<string, number[]>
    tv: { signal: string; lambda: number; x: number[] }[]
    fused: { signal: string; lambda1: number; lambda2: number; x: number[] }[]
  }
}
const F = fixture<Fixture>('optim/proximal')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

/** The objective of total variation denoising. */
const tvObjective = (y: ArrayLike<number>, x: ArrayLike<number>, lambda: number) => {
  let s = 0
  for (let i = 0; i < y.length; i++) s += 0.5 * (y[i] - x[i]) ** 2
  for (let i = 1; i < x.length; i++) s += lambda * Math.abs(x[i] - x[i - 1])
  return s
}

/**
 * The optimality conditions of total variation denoising: the dual u_k = sum_{i <= k} (y_i - x_i) lies in
 * [-lambda, lambda], ends at 0, and equals -lambda sign(x_{k+1} - x_k) where x jumps.
 */
const checkOptimal = (y: ArrayLike<number>, x: ArrayLike<number>, lambda: number, tol: number) => {
  let u = 0
  for (let k = 0; k < y.length - 1; k++) {
    u += y[k] - x[k]
    expect(Math.abs(u)).toBeLessThanOrEqual(lambda + tol)
    const jump = x[k + 1] - x[k]
    if (Math.abs(jump) > 1e-9) expect(Math.abs(u + lambda * Math.sign(jump))).toBeLessThanOrEqual(tol)
  }
  expect(Math.abs(u + y[y.length - 1] - x[y.length - 1])).toBeLessThanOrEqual(tol)
}

describe('total variation', () => {
  it('matches the dual box QP solved by BVLS, and satisfies the optimality conditions', () => {
    for (const { signal, lambda, x } of F.totalVariation.tv) {
      const y = F.totalVariation.signals[signal]
      const ours = toFlat(totalVariationDenoise1d(y, lambda))
      close(ours, x, 1e-10)
      checkOptimal(y, ours, lambda, 1e-12)
      expect(tvObjective(y, ours, lambda)).toBeLessThanOrEqual(tvObjective(y, x, lambda) + 1e-12)
    }
  })

  it('is optimal on random signals of many lengths and weights', () => {
    for (let n = 1; n <= 40; n++) {
      const y = toFlat(normals(stream(`tv/${n}`), [n]))
      for (const lambda of [0.01, 0.1, 0.5, 2, 100])
        checkOptimal(y, toFlat(totalVariationDenoise1d(y, lambda)), lambda, 1e-12)
    }
    // A large weight gives the mean; zero gives the signal.
    close(toFlat(totalVariationDenoise1d([1, 2, 6], 100)), [3, 3, 3], 1e-14)
    expect(toFlat(totalVariationDenoise1d([1, 2, 6], 0))).toEqual([1, 2, 6])
    expect(toFlat(totalVariationDenoise1d([], 1))).toEqual([])
    expect(() => totalVariationDenoise1d([1, 2], -1)).toThrow(DomainError)
  })

  it('the fused lasso signal approximator matches BVLS on its dual', () => {
    for (const { signal, lambda1, lambda2, x } of F.totalVariation.fused) {
      const y = F.totalVariation.signals[signal]
      close(toFlat(fusedLassoSignal(y, { lambda1, lambda2 })), x, 1e-10)
    }
  })

  it('proxTotalVariation1d scales the weight by the step', () => {
    const v = [0, 0.2, 1.5, 1.3, 0.1]
    const g = proxTotalVariation1d(2)
    close(toFlat(g.prox(tensor(v), 0.25)), toFlat(totalVariationDenoise1d(v, 0.5)), 0)
    expect(g.value(tensor(v))).toBeCloseTo(2 * (0.2 + 1.3 + 0.2 + 1.2), 14)
  })
})
