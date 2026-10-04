/**
 * WordPiece: the Hugging Face course corpus (hug 10, pug 5, pun 12, bun 4, hugs 5), its first merges and scores by the
 * count ratio, the likelihood criterion, and greedy longest-match-first encoding (BERT's "unaffable").
 */
import { describe, expect, it } from 'vitest'
import { run } from 'aifn-compute/foundation/trace'
import { wordPiece, wordPieceEncode, wordPieceModel, wordPieceSegment, wordPieceSteps } from 'aifn-compute/text/subword'

const COURSE = { hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }

describe('wordPieceSteps', () => {
  it('starts from first characters and ## continuations', () => {
    const s = run(wordPieceSteps(COURSE), undefined, 0)
    expect(s.segmentations[4]).toEqual(['h', '##u', '##g', '##s'])
    expect(s.vocabulary).toEqual(['[UNK]', 'b', 'h', 'p', '##g', '##n', '##s', '##u'])
  })
  it('merges ##g ##s (1/20), then h ##u (a tie at 1/36 broken by order), then hu ##gs (1/15)', () => {
    const s = run(wordPieceSteps(COURSE, { merges: 3 }), undefined, 10)
    expect(s.merges.map((m) => [m.left, m.right, m.merged])).toEqual([
      ['##g', '##s', '##gs'],
      ['h', '##u', 'hu'],
      ['hu', '##gs', 'hugs'],
    ])
    expect(s.merges[0].score).toBeCloseTo(1 / 20, 15)
    expect(s.merges[1].score).toBeCloseTo(1 / 36, 15)
    expect(s.merges[2].score).toBeCloseTo(1 / 15, 15)
    expect(s.vocabulary.slice(-3)).toEqual(['##gs', 'hu', 'hugs'])
  })
  it('prefers the rare-but-inseparable pair to the frequent one (the note’s q u against t h)', () => {
    // t 200, h 60, t h 50; q 10, u 100, q u 10: BPE would merge t h; WordPiece merges q u.
    const words: Record<string, number> = { th: 50, qu: 10, t: 150, h: 10, u: 90 }
    const s = run(wordPieceSteps(words, { merges: 1, prefix: '' }), undefined, 5)
    expect([s.merges[0].left, s.merges[0].right]).toEqual(['q', 'u'])
    expect(s.merges[0].score).toBeCloseTo(10 / (10 * 100), 15)
  })
  it('raises the unigram log-likelihood with the likelihood criterion', () => {
    const s = run(wordPieceSteps(COURSE, { criterion: 'likelihood', merges: 4 }), undefined, 10)
    expect(s.merges).toHaveLength(4)
    const s0 = run(wordPieceSteps(COURSE), undefined, 0)
    expect(s.logLikelihood).toBeGreaterThan(s0.logLikelihood)
  })
})

describe('wordPiece encoding', () => {
  it('splits unaffable as un ##aff ##able with BERT-style pieces', () => {
    const model = wordPieceModel(['[UNK]', 'un', 'aff', '##aff', '##able', 'able', 'u', '##n'])
    expect(wordPieceSegment(model, 'unaffable').map((p) => p.token)).toEqual(['un', '##aff', '##able'])
    expect(wordPieceSegment(model, 'unaffablez').map((p) => p.token)).toEqual(['[UNK]'])
  })
  it('encodes text with offsets after training', () => {
    const model = wordPiece(COURSE, { merges: 6 })
    const t = wordPieceEncode(model, 'hugs, bugs')
    // "," was never seen in training, so it is unknown.
    expect(t.tokens).toEqual(['hugs', '[UNK]', 'bu', '##gs'])
    expect(t.source.slice(t.offsets.data[4], t.offsets.data[5])).toBe('bu')
    expect(wordPieceEncode(model, 'hugz').tokens).toEqual(['[UNK]']) // no "##z" piece
  })
})
