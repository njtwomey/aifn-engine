/**
 * aifn-methods/data/synthetic recall and state-tracking tasks: selective copying puts its content tokens in order among
 * noise, parity strings are labelled by their count of ones, and MQAR is the same function as in
 * `aifn-methods/neural/sequence`.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { multiQueryAssociativeRecall, parityStrings, selectiveCopy } from 'aifn-methods/data/synthetic'
import { multiQueryAssociativeRecall as fromNeural } from 'aifn-methods/neural/sequence'

describe('recall tasks', () => {
  it('copies four content tokens in order from sixteen positions', () => {
    const c = selectiveCopy(stream(2), { length: 16, tokens: 4, vocab: 8 })
    expect(c.input.length).toBe(16)
    expect(c.input.filter((x) => x > 0)).toEqual(c.target)
    expect(c.target.every((t) => t >= 1 && t <= 8)).toBe(true)
    c.positions.forEach((p, i) => expect(c.input[p]).toBe(c.target[i]))
  })

  it('labels bit strings by parity', () => {
    const p = parityStrings(stream(3), { length: 5, n: 4 })
    expect(p.bits.length).toBe(4)
    p.bits.forEach((b, i) => expect(b.reduce((a, x) => a + x, 0) % 2).toBe(p.parity[i]))
  })

  it('re-exports the MQAR generator of neural/sequence', () => {
    expect(multiQueryAssociativeRecall).toBe(fromNeural)
  })
})
