/** Topic coherence against a NumPy reference of gensim's boolean-document NPMI and UMass (`fixtures/text/cooccurrence.json`). */
import { describe, expect, it } from 'vitest'
import { topicCoherence } from 'aifn-compute/text/cooccurrence'
import { fixture } from '../../fixtures'

const F = fixture<{ documents: number[][]; topics: number[][]; npmi: number[]; umass: number[] }>('text/cooccurrence')

describe('topicCoherence', () => {
  it('NPMI matches the reference', () => {
    const r = topicCoherence(F.topics, F.documents)
    F.npmi.forEach((v, i) => expect(r.topics[i]).toBeCloseTo(v, 10))
    expect(r.mean).toBeCloseTo(F.npmi.reduce((a, b) => a + b) / F.npmi.length, 10)
  })
  it('UMass matches the reference', () => {
    const r = topicCoherence(F.topics, F.documents, { measure: 'umass' })
    F.umass.forEach((v, i) => expect(r.topics[i]).toBeCloseTo(v, 10))
  })
  it('words that always occur together have NPMI 1', () => {
    const docs = [[0, 1], [0, 1, 2], [2], [3]]
    expect(topicCoherence([[0, 1]], docs).topics[0]).toBeCloseTo(1, 9)
  })
})
