import { describe, expect, it } from 'vitest'
import { knapsack, knapsackProgram, unboundedKnapsack } from 'aifn-methods/algorithms/dynamic-programming'
import { dynamicProgram, editDistance, lcs, needlemanWunsch, smithWaterman } from 'aifn-compute/optim/programming'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { expectProtocol } from '../../protocol'

describe('dynamic programming problems', () => {
  it('knapsacks', () => {
    const r = knapsack([60, 100, 120], [10, 20, 30], 50)
    expect(r.value).toBe(220)
    expect(toFlat(r.take)).toEqual([0, 1, 1])
    const u = unboundedKnapsack([10, 30, 20], [5, 10, 15], 100)
    expect(u.value).toBe(300)
  })

  it('LCS, edit distance and alignments', () => {
    const l = lcs('ABCBDAB', 'BDCABA')
    expect(l.length).toBe(4)
    expect(l.subsequence).toHaveLength(4)
    expect(editDistance('kitten', 'sitting').distance).toBe(3)
    const g = needlemanWunsch('GATTACA', 'GCATGCU')
    expect(g.score).toBe(0)
    expect((g.alignedA as string).replace(/-/g, '')).toBe('GATTACA')
    const s = smithWaterman('TGTTACGG', 'GGTTGACTA', { match: 3, mismatch: -3, gap: -2 })
    expect(s.score).toBe(13)
    expect(s.alignedA).toBe('GTT-AC')
    expect(s.alignedB).toBe('GTTGAC')
  })

  it('the knapsack table fills by the Algorithm protocol', () => {
    expectProtocol(dynamicProgram(knapsackProgram([60, 100, 120], [10, 20, 30], 50)), {}, { n: 3 })
  })
})
