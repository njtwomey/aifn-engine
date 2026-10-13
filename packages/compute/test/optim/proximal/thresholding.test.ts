import { describe, expect, it } from 'vitest'
import { proxL1, thresholding, thresholdingPenalty, type ThresholdRule } from 'aifn-compute/optim/proximal'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'

type Fixture = {
  thresholding: {
    z: number[]
    theta: number[]
    rules: { rule: ThresholdRule; lambda: number; gamma: number; value: number[]; penalty: number[] }[]
  }
}
const F = fixture<Fixture>('optim/proximal')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

describe('thresholding', () => {
  it('matches the closed forms, checked by brute force in the generator', () => {
    const { z, theta, rules } = F.thresholding
    for (const { rule, lambda, gamma, value, penalty } of rules) {
      const options = { rule, lambda, gamma: gamma || undefined }
      close(toFlat(thresholding(z, options)), value, 1e-14)
      close(toFlat(thresholdingPenalty(theta, options)), penalty, 1e-14)
    }
  })

  it('each rule minimises the penalised univariate problem', () => {
    const grid = Array.from({ length: 20001 }, (_, i) => -10 + i * 1e-3)
    for (const rule of ['soft', 'hard', 'scad', 'mcp'] as const) {
      const options = { rule, lambda: 1.3 }
      const pen = toFlat(thresholdingPenalty(grid, options))
      for (const zi of [-4.4, -2.9, -1.8, -0.7, 0.4, 1.5, 2.2, 3.1, 4.6, 6]) {
        const t = toFlat(thresholding([zi], options))[0]
        const value = 0.5 * (zi - t) ** 2 + toFlat(thresholdingPenalty([t], options))[0]
        const best = Math.min(...grid.map((g, i) => 0.5 * (zi - g) ** 2 + pen[i]))
        expect(value).toBeLessThanOrEqual(best + 1e-12)
      }
    }
  })

  it('agrees with proxL1, is continuous except for the hard rule, and checks its options', () => {
    const z = tensor([-3, -0.5, 0.2, 1.7, 4])
    close(toFlat(thresholding(z, { lambda: 1 })), toFlat(proxL1(1).prox(z, 1)), 0)
    for (const rule of ['scad', 'mcp'] as const) {
      const lambda = 1
      const gamma = rule === 'scad' ? 3.7 : 3
      for (const knot of [lambda, 2 * lambda, gamma * lambda]) {
        const [a, b] = toFlat(thresholding([knot - 1e-9, knot + 1e-9], { rule, lambda }))
        expect(Math.abs(a - b)).toBeLessThan(1e-8)
      }
    }
    expect(toFlat(thresholding([1, 1.0001], { rule: 'hard', lambda: 1 }))).toEqual([0, 1.0001])
    expect(() => thresholding([1], { rule: 'scad', lambda: 1, gamma: 2 })).toThrow(DomainError)
    expect(() => thresholding([1], { rule: 'mcp', lambda: 1, gamma: 1 })).toThrow(DomainError)
    expect(() => thresholding([1], { lambda: -1 })).toThrow(DomainError)
    expect(() => thresholding([1], { rule: 'nope' as ThresholdRule, lambda: 1 })).toThrow(DomainError)
  })
})
