/**
 * aifn-methods/neural/language-models: the Kneser–Ney n-gram model by its laws (every distribution sums to one; the
 * discount and continuation counts on a hand-worked corpus; modified discounts; perplexity falls with order on its own
 * corpus) and the tiny GPT (shapes, causality, gradients, a short training run that lowers the loss, decoding).
 */
import { describe, expect, it } from 'vitest'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, sum, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { greedyDecoding } from 'aifn-compute/nn/decoding'
import {
  charCorpus,
  decodeChars,
  encodeChars,
  Gpt,
  gptLogits,
  gptTraining,
  kneserNey,
  nextTokenWindows,
} from 'aifn-methods/neural/language-models'

describe('Kneser–Ney n-gram model', () => {
  const corpus = charCorpus()

  it('every next-token distribution sums to one, for every order and both discount schemes', () => {
    for (const order of [1, 2, 3, 5])
      for (const modified of [false, true]) {
        const m = kneserNey({ order, modified }).fit(corpus)
        for (const ctx of [[], corpus.ids.slice(0, 1), corpus.ids.slice(10, 20), encodeChars(corpus, 'zq')]) {
          const p = m.distribution(ctx)
          expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
          expect(p.every((v) => v > 0)).toBe(true)
        }
      }
  })

  it('matches a hand-worked bigram case: discount, freed mass and continuation counts', () => {
    // Tokens a = 0, b = 1, c = 2 in "abacab": bigrams ab ×2, ba, ac, ca. Counts of counts n1 = 3, n2 = 1, so
    // D = 3/(3 + 2) = 0.6. Continuation counts N₁₊(• w): a ← {b, c} = 2, b ← {a} = 1, c ← {a} = 1; their counts of
    // counts n1 = 2, n2 = 1, D′ = 2/4 = 0.5; P₁(w) = (N − D′)/4 + (3·D′/4)·(1/3).
    const tiny = charCorpus('abacab')
    const m = kneserNey({ order: 2 }).fit(tiny)
    const p1 = [(2 - 0.5) / 4 + 1.5 / 4 / 3, (1 - 0.5) / 4 + 1.5 / 4 / 3, (1 - 0.5) / 4 + 1.5 / 4 / 3]
    // After "a": c(a ·) = 3 (ab ×2, ac), freed = 2 × 0.6 = 1.2.
    const afterA = [0, (2 - 0.6) / 3, (1 - 0.6) / 3].map((v, w) => v + (1.2 / 3) * p1[w])
    m.distribution([0]).forEach((v, w) => expect(v).toBeCloseTo(afterA[w], 14))
    expect(m.discounts.counts[1]).toEqual([0.6, 0.6, 0.6])
  })

  it('higher orders fit their own corpus better, and the model decodes', () => {
    const ppl = [1, 2, 3, 4].map((order) => kneserNey({ order }).fit(corpus).perplexity(corpus.ids.slice(0, 400)))
    for (let i = 1; i < ppl.length; i++) expect(ppl[i]).toBeLessThan(ppl[i - 1])
    // Order 6 reads five characters, "pty d", which only "humpty dumpty" continues.
    const m = kneserNey({ order: 6 }).fit(corpus)
    const s = run(greedyDecoding(m.logits, { prompt: encodeChars(corpus, 'humpty d'), maxTokens: 5 }), undefined, 5)
    expect(decodeChars(corpus, s.tokens)).toBe('humpty dumpty')
  })

  it('score reads a 1-D tensor of N ids as N one-token contexts, and rejects other ranks', () => {
    const m = kneserNey({ order: 3 }).fit(corpus)
    const ids = encodeChars(corpus, 'hum')
    const flat = toFlat(m.score(fromData(Int32Array.from(ids), [3])))
    const rows = toFlat(m.score(fromData(Int32Array.from(ids), [3, 1])))
    expect(Array.from(flat)).toEqual(Array.from(rows))
    const V = corpus.vocabulary.tokens.length
    m.distribution([ids[1]]).forEach((p, w) => expect(flat[V + w]).toBeCloseTo(Math.log(p), 14))
    expect(() => m.score(fromData(Int32Array.from(ids), [1, 3, 1]))).toThrow(ShapeError)
  })
})

describe('tiny GPT', () => {
  const corpus = charCorpus()
  const V = corpus.vocabulary.tokens.length

  it('maps ids to next-token logits, causally, for every positional scheme', () => {
    for (const position of ['learned', 'sinusoidal', 'rope', 'alibi', 'none'] as const) {
      const model = Gpt({ vocabulary: V, width: 16, layers: 1, heads: 2, context: 8, position })
      const p = model.init(stream(position))
      const ids = corpus.ids.slice(0, 8)
      const full = toFlat(unwrap(model.apply(p, ids)) as Tensor)
      expect(full).toHaveLength(8 * V)
      // Causality: the logits at position 3 do not depend on later tokens.
      const changed = [...ids.slice(0, 4), ...ids.slice(4).map((i) => (i + 1) % V)]
      const other = toFlat(unwrap(model.apply(p, changed)) as Tensor)
      for (let j = 0; j < 4 * V; j++) expect(other[j]).toBeCloseTo(full[j], 12)
    }
  })

  it('has correct gradients', () => {
    const model = Gpt({
      vocabulary: V,
      width: 8,
      layers: 1,
      heads: 2,
      context: 4,
      position: 'rope',
      feedForward: 'swiglu',
    })
    const p = model.init(stream('g'))
    const ids = fromData(Int32Array.from(corpus.ids.slice(0, 4)), [4])
    expect(gradCheck((q: typeof p) => sum(model.apply(q, ids)), p, { rtol: 1e-4, atol: 1e-6 }).ok).toBe(true)
  })

  it('a short training run lowers the next-token loss, and greedy decoding reads its logits', () => {
    const model = Gpt({ vocabulary: V, width: 16, layers: 1, heads: 2, context: 16 })
    const tr = trace(
      gptTraining(model, corpus, { batchSize: 8, stepSize: 0.02 }),
      { params: model.init(stream('t')) },
      40,
      {
        record: { loss: (s) => s.loss },
      },
    )
    const loss = toFlat(tr.series.loss)
    const early = loss.slice(0, 5).reduce((a, b) => a + b) / 5
    const late = loss.slice(-5).reduce((a, b) => a + b) / 5
    expect(late).toBeLessThan(early - 0.3)
    const s = run(greedyDecoding(gptLogits(model, tr.final.params), { prompt: [1, 2], maxTokens: 20 }), undefined, 20)
    expect(s.tokens).toHaveLength(22)
    expect(nextTokenWindows([0, 1, 2, 3, 4], 2).y.shape).toEqual([3, 2])
  })
})
