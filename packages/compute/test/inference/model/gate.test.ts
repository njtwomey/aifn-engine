import { describe, expect, it } from 'vitest'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  discreteFactor,
  factorMarginalise,
  factorProduct,
  factorReduce,
  gateFactor,
  normaliseFactor,
} from 'aifn-compute/inference/model'

describe('gateFactor', () => {
  // Variables: 0 = selector c (2 values), 1 = x (3 values), 2 = y (2 values).
  const card = [2, 3, 2]
  const p0 = discreteFactor([1], card, [0.7, 0.2, 0.1])
  const p1 = discreteFactor([1, 2], card, [0.05, 0.05, 0.1, 0.1, 0.3, 0.4])
  const gate = gateFactor(0, [p0, p1], card)
  it('switches between its cases, constant along variables a case does not touch', () => {
    expect(gate.scope).toEqual([0, 1, 2])
    const t = toFlat(gate.table)
    // c = 0: p0(x) for both y; c = 1: p1(x, y).
    expect(Array.from(t.slice(0, 6))).toEqual([0.7, 0.7, 0.2, 0.2, 0.1, 0.1])
    expect(Array.from(t.slice(6))).toEqual([0.05, 0.05, 0.1, 0.1, 0.3, 0.4])
  })
  it('gives the posterior of the selector by Bayes’ rule', () => {
    const prior = discreteFactor([0], card, [0.6, 0.4])
    const joint = factorProduct(prior, gate, card)
    const evidence = factorReduce(
      joint,
      new Map([
        [1, 2],
        [2, 1],
      ]),
    )
    const posterior = normaliseFactor(factorMarginalise(evidence, [])).factor
    const expected0 = (0.6 * 0.1) / (0.6 * 0.1 + 0.4 * 0.4)
    expect(toFlat(posterior.table)[0]).toBeCloseTo(expected0, 12)
  })
  it('refuses a case count that does not match the selector', () => {
    expect(() => gateFactor(0, [p0], card)).toThrow(ShapeError)
  })
})
