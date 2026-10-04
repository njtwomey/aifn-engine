/** The topic corpus: labels and word topics, prefix stability, and the structure LSA should find in it. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { lsa, nearestByCosine, termTermMatrix } from 'aifn-compute/text/representations'
import { tokenise } from 'aifn-compute/text/tokenise'
import { tokenId } from 'aifn-compute/text/vocabulary'
import { topicCorpus, TOPIC_CORPUS_TOPICS } from 'aifn-methods/text/corpora'

describe('topicCorpus', () => {
  const c = topicCorpus(stream(0), { sentences: 300 })
  it('labels every sentence and knows every word’s topic', () => {
    expect(c.documents).toHaveLength(300)
    expect(c.labels).toHaveLength(300)
    expect(c.meta.labelNames).toEqual(TOPIC_CORPUS_TOPICS)
    for (const d of c.documents) for (const w of d.split(' ')) expect(c.wordTopics![w]).toBeDefined()
    expect(new Set(c.labels).size).toBe(5)
  })
  it('extends a shorter corpus with the same seed', () => {
    expect(topicCorpus(stream(0), { sentences: 50 }).documents).toEqual(c.documents.slice(0, 50))
  })
  it('puts topic words together under PPMI and a rank-10 SVD', () => {
    const docs = c.documents.map((d) => tokenise(d).tokens)
    const M = termTermMatrix(docs, { window: 2, weighting: 'ppmi' })
    const rows = lsa(M.matrix, 10).rows
    for (const w of ['cat', 'bakes', 'bus', 'red', 'tourists']) {
      const nn = nearestByCosine(rows, tokenId(M.terms, w), { count: 3 })
      const same = nn.filter((x) => c.wordTopics![M.terms.tokens[x.index]] === c.wordTopics![w]).length
      expect(same, w).toBeGreaterThanOrEqual(2)
    }
  })
})
