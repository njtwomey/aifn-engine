/**
 * The unigram-LM tokeniser against the worked example of the WordPiece-and-unigram note ("hugs" with h, u, g 0.05,
 * s 0.10, hu 0.08, ug 0.10, hug 0.07, gs 0.03): the six segmentations, their posteriors, the marginal, Viterbi,
 * expected counts and the pruning losses; then training, encoding and sampling.
 */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  unigramLm,
  unigramLmEncode,
  unigramLmLosses,
  unigramLmMarginal,
  unigramLmModel,
  unigramLmSample,
  unigramLmSegmentations,
  unigramLmSteps,
  unigramLmViterbi,
} from 'aifn-compute/text/subword'

const PIECES = ['h', 'u', 'g', 's', 'hu', 'ug', 'hug', 'gs']
const P = [0.05, 0.05, 0.05, 0.1, 0.08, 0.1, 0.07, 0.03]
const model = unigramLmModel({ pieces: PIECES, probabilities: P }, { boundary: '', normalise: false })

describe('the worked example', () => {
  it('lists the six segmentations with their probabilities and posteriors', () => {
    const segs = unigramLmSegmentations(model, 'hugs')
    expect(segs.map((s) => s.pieces.join(' '))).toEqual(['hug s', 'hu gs', 'h ug s', 'hu g s', 'h u gs', 'h u g s'])
    expect(segs.map((s) => s.probability)).toEqual(
      [0.007, 0.0024, 0.0005, 0.0004, 0.000075, 0.0000125].map((x) => expect.closeTo(x, 12)),
    )
    expect(segs.map((s) => Math.round(1000 * s.posterior) / 1000)).toEqual([0.674, 0.231, 0.048, 0.039, 0.007, 0.001])
  })
  it('sums them in the marginal and picks hug s by Viterbi', () => {
    expect(Math.exp(unigramLmMarginal(model, 'hugs'))).toBeCloseTo(0.0103875, 12)
    const v = unigramLmViterbi(model, 'hugs')
    expect(v.pieces.map((p) => p.token)).toEqual(['hug', 's'])
    expect(Math.exp(v.logProb)).toBeCloseTo(0.007, 12)
  })
  it('gives the pruning losses: 1.12 nats for hug, 0.27 for gs, 0.05 for ug', () => {
    const loss = toFlat(unigramLmLosses(model, { chars: [[...'hugs']], counts: [1] }))
    const of = (p: string) => loss[PIECES.indexOf(p)]
    expect(of('hug')).toBeCloseTo(Math.log(0.0103875 / 0.0033875), 12)
    expect(of('gs')).toBeCloseTo(Math.log(0.0103875 / 0.0079125), 12)
    expect(of('ug')).toBeCloseTo(Math.log(0.0103875 / 0.0098875), 12)
    expect([of('hug'), of('gs'), of('ug')].map((x) => Math.round(100 * x) / 100)).toEqual([1.12, 0.27, 0.05])
    expect(of('s')).toBeCloseTo(Math.log(0.0103875 / 0.002475), 12) // hu gs and h u gs remain
  })
  it('samples segmentations in proportion to their posteriors', () => {
    const root = stream(7)
    let hugS = 0
    const n = 4000
    for (let k = 0; k < n; k++)
      if (unigramLmSample(child(root, k), model, 'hugs').length === 2) {
        const pieces = unigramLmSample(child(root, k), model, 'hugs').map((p) => p.token)
        if (pieces[0] === 'hug') hugS++
      }
    expect(Math.abs(hugS / n - 0.674)).toBeLessThan(0.03)
    // a large exponent concentrates on the Viterbi segmentation
    const sharp = unigramLmSample(child(root, 'x'), model, 'hugs', { alpha: 200 }).map((p) => p.token)
    expect(sharp).toEqual(['hug', 's'])
  })
})

const CORPUS = new Map([
  ['hug', 10],
  ['pug', 5],
  ['pun', 12],
  ['bun', 4],
  ['hugs', 5],
  ['bugs', 3],
  ['puns', 2],
])

describe('unigramLmSteps', () => {
  it('prunes to the target, keeps every character, and records the losses of what it removed', () => {
    const tr = trace(unigramLmSteps(CORPUS, { vocabularySize: 14, shrink: 0.7 }), undefined, 50)
    const final = tr.final
    expect(final.done).toBe(true)
    expect(final.pieces.length).toBe(14)
    for (const c of '▁hugpnbs') expect(final.pieces).toContain(c)
    const s1 = tr.steps![1]
    expect(s1.pruned.length).toBeGreaterThan(0)
    expect(s1.pruned.map((p) => p.loss)).toEqual([...s1.pruned.map((p) => p.loss)].sort((a, b) => a - b))
    expect(toFlat(final.logProbs).reduce((s, x) => s + Math.exp(x), 0)).toBeCloseTo(1, 12)
  })
  it('does not lower the likelihood by EM when nothing is pruned', () => {
    const s0 = run(unigramLmSteps(CORPUS, { vocabularySize: 1e6 }), undefined, 0)
    const s1 = run(unigramLmSteps(CORPUS, { vocabularySize: 1e6 }), undefined, 1)
    expect(s1.done).toBe(true)
    expect(s1.logLikelihood).toBeGreaterThanOrEqual(s0.logLikelihood - 1e-9)
  })
  it('encodes with offsets, unknown characters as <unk>', () => {
    const m = unigramLm(CORPUS, { vocabularySize: 14 })
    const t = unigramLmEncode(m, 'hugs zap')
    expect(t.tokens.join('')).toContain('▁')
    expect(t.tokens).toContain('<unk>')
    const rows = toRows(t.offsets)
    expect(rows[0][0]).toBe(0)
    expect(rows.at(-1)![1]).toBe(8)
  })
})
