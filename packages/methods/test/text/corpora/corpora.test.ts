/** `aifn-methods/text/corpora`: the named corpora, the seeded generator and their registration. */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { tokenise } from 'aifn-compute/text/tokenise'
import { bpe, bpeEncode } from 'aifn-compute/text/subword'
import { corpusDatasets, namedCorpus, NAMED_CORPORA, toyCorpus } from 'aifn-methods/text/corpora'
import { expectInfo } from '../../registry'

describe('corpora', () => {
  it('registers both generators with well-formed metadata, running at their default knobs', () => {
    expectInfo(corpusDatasets, 'dataset')
    for (const entry of Object.values(corpusDatasets)) {
      const f = entry as unknown as (...a: unknown[]) => { kind: string; documents: string[]; meta: { task: string } }
      const knobs = defaults(entry.info.knobs)
      const made = entry.info.random ? f(stream(1), knobs) : f(knobs)
      expect(made.kind).toBe('corpus')
      expect(made.meta.task).toBe('text')
      expect(made.documents.length).toBeGreaterThan(0)
    }
  })
  it('gives the worked-example corpora', () => {
    expect(namedCorpus().documents).toHaveLength(5)
    expect(namedCorpus({ name: 'pets' }).documents).toHaveLength(13)
    const words = tokenise(namedCorpus({ name: 'low-lower-newest-widest' }).documents[0]).tokens
    expect(words).toHaveLength(16)
    expect(Object.keys(NAMED_CORPORA)).toContain('hug-pug-pun')
  })
  it('is seeded and prefix-stable', () => {
    const a = toyCorpus(stream(3), { sentences: 10 })
    const b = toyCorpus(stream(3), { sentences: 20 })
    expect(b.documents.slice(0, 10)).toEqual(a.documents)
    expect(toyCorpus(stream(4), { sentences: 10 }).documents).not.toEqual(a.documents)
    expect(toyCorpus(child(stream(3), 'x'), { topics: 1 }).meta.description).toContain('pets')
  })
  it('feeds the subword trainers', () => {
    const words = toyCorpus(stream(5), { sentences: 200 }).documents.flatMap((d) => tokenise(d).tokens)
    const model = bpe(words, { merges: 50 })
    expect(model.merges).toHaveLength(50)
    expect(bpeEncode(model, 'the hungry dogs').tokens.length).toBeLessThan(15)
  })
})
