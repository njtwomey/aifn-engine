/**
 * Conjunctive hypotheses (aifn-compute/logic/induction `conjunctions.ts`): the size of the language, the Hasse diagram
 * of the generality order, the refinement operators and the order itself.
 */
import { describe, expect, it } from 'vitest'
import {
  allConjunctions,
  allInstances,
  ANY,
  conjunctionCoverage,
  conjunctionLevel,
  conjunctionText,
  consistent,
  covers,
  generalisations,
  generaliseToCover,
  hasseEdges,
  instanceText,
  moreGeneralOrEqual,
  mostGeneral,
  sameConjunction,
  specialisations,
  specialiseToExclude,
  strictlyMoreGeneral,
  type AttributeSpace,
} from 'aifn-compute/logic/induction'

const ENJOY_SPORT: AttributeSpace = {
  names: ['sky', 'air', 'humidity', 'wind', 'water', 'forecast'],
  values: [
    ['sunny', 'cloudy', 'rainy'],
    ['warm', 'cold'],
    ['normal', 'high'],
    ['strong', 'weak'],
    ['warm', 'cool'],
    ['same', 'change'],
  ],
}
const PICNIC: AttributeSpace = {
  names: ['sky', 'temp', 'humidity', 'wind'],
  values: [
    ['sunny', 'cloudy', 'rainy'],
    ['warm', 'cold'],
    ['normal', 'high'],
    ['weak', 'strong'],
  ],
}
const BINARY: AttributeSpace = {
  names: ['a', 'b', 'c'],
  values: [
    ['0', '1'],
    ['0', '1'],
    ['0', '1'],
  ],
}

describe('the language', () => {
  it('has 1 + prod (k_i + 1) conjunctions', () => {
    expect(allConjunctions(ENJOY_SPORT)).toHaveLength(973)
    expect(allConjunctions(PICNIC)).toHaveLength(109)
    expect(allConjunctions(PICNIC, false)).toHaveLength(108)
    expect(allInstances(PICNIC)).toHaveLength(24)
  })

  it('three binary attributes: 28 hypotheses and 62 Hasse edges (6 + 24 + 24 + 8)', () => {
    const hs = allConjunctions(BINARY)
    expect(hs).toHaveLength(28)
    const edges = hasseEdges(hs, BINARY)
    expect(edges).toHaveLength(62)
    const fromLevel = [0, 1, 2, 3].map((l) => edges.filter(([i]) => conjunctionLevel(hs[i], BINARY) === l).length)
    expect(fromLevel).toEqual([6, 24, 24, 8])
  })

  it('refines the top into 6 and generalises the bottom into 8', () => {
    expect(specialisations([ANY, ANY, ANY], BINARY)).toHaveLength(6)
    expect(generalisations(null, BINARY)).toHaveLength(8)
    expect(specialisations([0, 1, 0], BINARY)).toEqual([null])
    expect(specialisations(null, BINARY)).toEqual([])
    expect(generalisations(mostGeneral(BINARY), BINARY)).toEqual([])
  })
})

describe('the generality order', () => {
  it('orders by coverage, with the empty hypothesis at the bottom', () => {
    expect(moreGeneralOrEqual([0, ANY, ANY], [0, 1, ANY])).toBe(true)
    expect(moreGeneralOrEqual([0, 1, ANY], [0, ANY, ANY])).toBe(false)
    expect(moreGeneralOrEqual([0, 1, 1], null)).toBe(true)
    expect(strictlyMoreGeneral([0, ANY, ANY], [0, ANY, ANY])).toBe(false)
    expect(sameConjunction(null, null)).toBe(true)
    expect(covers(null, [0, 0, 0])).toBe(false)
  })

  it('generalises to cover and specialises to exclude minimally', () => {
    expect(generaliseToCover(null, [1, 0, 1])).toEqual([1, 0, 1])
    expect(generaliseToCover([1, 0, 1], [1, 1, 1])).toEqual([1, ANY, 1])
    const spec = specialiseToExclude([ANY, ANY, 0], [1, 1, 0], BINARY)
    expect(spec).toEqual([
      [0, ANY, 0],
      [ANY, 0, 0],
    ])
    expect(specialiseToExclude([0, ANY, ANY], [1, 1, 0], BINARY)).toEqual([[0, ANY, ANY]])
  })

  it('reports coverage and consistency, and prints in three notations', () => {
    const ex = [
      { x: [1, 0, 0], positive: true },
      { x: [1, 1, 0], positive: true },
      { x: [0, 1, 0], positive: false },
    ]
    expect(conjunctionCoverage([ANY, ANY, 0], ex)).toEqual({ positives: [0, 1], negatives: [2] })
    expect(consistent([1, ANY, ANY], ex)).toBe(true)
    expect(conjunctionText([0, ANY, ANY, 0], PICNIC)).toBe('⟨sunny, ?, ?, weak⟩')
    expect(conjunctionText([1, ANY, 0], BINARY, 'logic')).toBe('a = 1 ∧ c = 0')
    expect(conjunctionText(mostGeneral(BINARY), BINARY, 'logic')).toBe('true')
    expect(conjunctionText(null, BINARY, 'tex')).toBe('\\bot')
    expect(instanceText([2, 1, 1, 1], PICNIC)).toBe('⟨rainy, cold, high, strong⟩')
  })
})
