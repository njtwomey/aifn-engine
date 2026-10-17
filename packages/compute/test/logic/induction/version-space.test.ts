/**
 * Candidate elimination (aifn-compute/logic/induction `candidateElimination`): Mitchell's EnjoySport boundaries, the
 * picnic table checked against the brute-force version space at every step, and collapses under noise and a
 * disjunctive target.
 */
import { describe, expect, it } from 'vitest'
import { run } from 'aifn-compute/foundation/trace'
import {
  allConjunctions,
  candidateElimination,
  conjunctionText,
  consistentSet,
  findS,
  versionSpaceMembers,
  versionSpaceVote,
  type AttributeSpace,
  type Conjunction,
  type LabelledInstance,
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
const PICNIC_EXAMPLES: LabelledInstance[] = [
  { x: [0, 0, 0, 0], positive: true },
  { x: [0, 0, 1, 0], positive: true },
  { x: [2, 1, 1, 1], positive: false },
  { x: [0, 1, 1, 0], positive: true },
  { x: [1, 0, 0, 0], positive: false },
  { x: [0, 0, 0, 1], positive: false },
]

const texts = (hs: readonly Conjunction[], space: AttributeSpace) => hs.map((h) => conjunctionText(h, space)).sort()

/** Version-space size, |S| and |G| after 0, 1, … examples. */
function sizes(examples: readonly LabelledInstance[]): [number, number, number][] {
  const alg = candidateElimination(PICNIC, examples)
  let st = alg.init(undefined, undefined as never)
  const out: [number, number, number][] = []
  for (let k = 0; k <= examples.length; k++) {
    out.push([versionSpaceMembers(PICNIC, st.S, st.G).length, st.S.length, st.G.length])
    expect(versionSpaceMembers(PICNIC, st.S, st.G).length).toBe(consistentSet(PICNIC, examples.slice(0, k)).length)
    if (k < examples.length) st = alg.step(st, { t: k, stream: undefined as never })
  }
  return out
}

describe('candidateElimination', () => {
  it('EnjoySport after four examples: S4, G4 and six members', () => {
    const ex: LabelledInstance[] = [
      { x: [0, 0, 0, 0, 0, 0], positive: true },
      { x: [0, 0, 1, 0, 0, 0], positive: true },
      { x: [2, 1, 1, 0, 0, 1], positive: false },
      { x: [0, 0, 1, 0, 1, 1], positive: true },
    ]
    const s = run(candidateElimination(ENJOY_SPORT, ex), undefined, 10)
    expect(texts(s.S, ENJOY_SPORT)).toEqual(['⟨sunny, warm, ?, strong, ?, ?⟩'])
    expect(texts(s.G, ENJOY_SPORT)).toEqual(['⟨?, warm, ?, ?, ?, ?⟩', '⟨sunny, ?, ?, ?, ?, ?⟩'])
    const vs = versionSpaceMembers(ENJOY_SPORT, s.S, s.G)
    expect(vs).toHaveLength(6)
    expect(texts(vs, ENJOY_SPORT)).toEqual(texts(consistentSet(ENJOY_SPORT, ex), ENJOY_SPORT))
    expect(allConjunctions(ENJOY_SPORT)).toHaveLength(973)
  })

  it('picnic: boundaries equal brute force at every step (109, 16, 8, 7, 3, 2, 1)', () => {
    expect(sizes(PICNIC_EXAMPLES).map((x) => x[0])).toEqual([109, 16, 8, 7, 3, 2, 1])
    const s = run(candidateElimination(PICNIC, PICNIC_EXAMPLES), undefined, 10)
    expect(texts(s.S, PICNIC)).toEqual(['⟨sunny, ?, ?, weak⟩'])
    expect(texts(s.G, PICNIC)).toEqual(['⟨sunny, ?, ?, weak⟩'])
    expect(conjunctionText(findS(PICNIC_EXAMPLES), PICNIC)).toBe('⟨sunny, ?, ?, weak⟩')
  })

  it('negatives first grow G to 13', () => {
    const order = [2, 4, 5, 0, 1, 3].map((i) => PICNIC_EXAMPLES[i])
    const out = sizes(order)
    expect(out.map((x) => x[2])).toEqual([1, 5, 13, 13, 1, 1, 1])
    expect(out.map((x) => x[0])).toEqual([109, 93, 78, 67, 4, 2, 1])
  })

  it('a mislabelled repeat collapses the space at example 7', () => {
    const repeat = [...PICNIC_EXAMPLES, { x: [0, 1, 1, 0], positive: false }]
    const s = run(candidateElimination(PICNIC, repeat), undefined, 10)
    expect(s.collapsed).toBe(true)
    expect(s.seen).toBe(7)
    expect(s.S).toHaveLength(0)
  })

  it('a disjunctive target collapses at example 3', () => {
    const disjunctive: LabelledInstance[] = [
      { x: [0, 0, 0, 1], positive: true },
      { x: [2, 0, 0, 0], positive: true },
      { x: [1, 0, 0, 1], positive: false },
    ]
    const s = run(candidateElimination(PICNIC, disjunctive), undefined, 10)
    expect(s.collapsed).toBe(true)
    expect(s.seen).toBe(3)
  })

  it('the version space votes', () => {
    const ex = PICNIC_EXAMPLES.slice(0, 3)
    const members = consistentSet(PICNIC, ex)
    expect(versionSpaceVote(members, [0, 0, 0, 0]).verdict).toBe('positive')
    expect(versionSpaceVote(members, [2, 1, 1, 1]).verdict).toBe('negative')
    expect(versionSpaceVote(members, [0, 1, 0, 1]).verdict).toBe('disagree')
    expect(versionSpaceVote([], [0, 0, 0, 0]).verdict).toBe('empty')
  })
})
