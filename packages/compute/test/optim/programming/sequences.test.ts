/**
 * The sequence programmes of aifn-compute/optim/programming against references (fixtures/gen/optim/programming.py): edit
 * distances from nltk, LCS lengths and Needleman–Wunsch and Smith–Waterman scores from Biopython's PairwiseAligner;
 * and the edit operations replay a into b with counts that add up.
 */
import { describe, expect, it } from 'vitest'
import { editDistance, lcs, needlemanWunsch, smithWaterman } from 'aifn-compute/optim/programming'
import { fixture } from '../../fixtures'

type Pair = { a: string; b: string }
type Fx = {
  editDistance: (Pair & { unit: number; substitute2: number })[]
  lcs: (Pair & { length: number })[]
  needlemanWunsch: (Pair & { match: number; mismatch: number; gap: number; score: number })[]
  smithWaterman: (Pair & { match: number; mismatch: number; gap: number; score: number })[]
}
const F = fixture<Fx>('optim/programming')
const name = (c: Pair) => `${JSON.stringify(c.a)} → ${JSON.stringify(c.b)}`

describe('edit distance (nltk)', () => {
  it.each(F.editDistance.map((c) => [name(c), c] as const))('%s', (_, c) => {
    const r = editDistance(c.a, c.b)
    expect(r.distance).toBe(c.unit)
    expect(editDistance(c.a, c.b, { substitute: 2 }).distance).toBe(c.substitute2)
    // The operations turn a into b, and the counts are theirs.
    const out: string[] = []
    for (const o of r.operations) if (o.op !== 'delete') out.push(c.b[o.j])
    expect(out.join('')).toBe(c.b)
    expect(r.counts.match + r.counts.substitute + r.counts.delete).toBe(c.a.length)
    expect(r.counts.match + r.counts.substitute + r.counts.insert).toBe(c.b.length)
    expect(r.counts.substitute + r.counts.delete + r.counts.insert).toBe(c.unit)
  })
})

describe('longest common subsequence (Biopython)', () => {
  it.each(F.lcs.map((c) => [name(c), c] as const))('%s', (_, c) => {
    const r = lcs(c.a, c.b)
    expect(r.length).toBe(c.length)
    // The subsequence is common to both.
    const sub = r.subsequence as string
    const isSub = (s: string) => {
      let k = 0
      for (const ch of s) if (ch === sub[k]) k++
      return k === sub.length
    }
    expect(isSub(c.a) && isSub(c.b)).toBe(true)
  })
})

describe('alignment scores (Biopython PairwiseAligner)', () => {
  it.each(F.needlemanWunsch.map((c, i) => [`${name(c)} (${i})`, c] as const))('Needleman–Wunsch %s', (_, c) => {
    expect(needlemanWunsch(c.a, c.b, { match: c.match, mismatch: c.mismatch, gap: c.gap }).score).toBe(c.score)
  })
  it.each(F.smithWaterman.map((c, i) => [`${name(c)} (${i})`, c] as const))('Smith–Waterman %s', (_, c) => {
    expect(smithWaterman(c.a, c.b, { match: c.match, mismatch: c.mismatch, gap: c.gap }).score).toBe(c.score)
  })
})
