/**
 * `aifn-compute/text/features` against scikit-learn (CountVectorizer, TfidfVectorizer, HashingVectorizer and its analyzers,
 * fixture `text/features`) and against the worked examples of the TF-IDF and weighting-variants notes.
 */
import { describe, expect, it } from 'vitest'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { tokenise } from 'aifn-compute/text/tokenise'
import { tokenId } from 'aifn-compute/text/vocabulary'
import {
  bagOfWords,
  bm25,
  bm25Weights,
  characterNgrams,
  featureHash,
  hashedFeatures,
  inverseDocumentFrequency,
  murmurHash3,
  smartWeighting,
  termFrequency,
  tfidf,
  wordNgrams,
  type TfidfOptions,
} from 'aifn-compute/text/features'
import { fixture } from '../../fixtures'

type M = { vocabulary: string[]; matrix: number[][]; idf?: number[] | null; options?: Record<string, unknown> }
const F = fixture<{
  docs: string[]
  vectorisers: Record<string, M>
  analyzers: Record<string, string[]> & { text: string }
  murmurhash3: { words: string[]; hashes: number[]; seeded: number[] }
}>('text/features')

const sk = (doc: string) => tokenise(doc.toLowerCase(), { pattern: 'alphanumeric' }).tokens
const DOCS = F.docs.map(sk)
const close = (a: number[][], b: number[][], tol = 1e-12) =>
  a.forEach((row, i) => row.forEach((x, j) => expect(x).toBeCloseTo(b[i][j], -Math.log10(tol))))

describe('bagOfWords against CountVectorizer', () => {
  it('counts over an alphabetical vocabulary', () => {
    const b = bagOfWords(DOCS)
    expect(b.vocabulary.tokens).toEqual(F.vectorisers.count.vocabulary)
    expect(toRows(b.matrix)).toEqual(F.vectorisers.count.matrix)
  })
  it('marks presence of unigrams and bigrams', () => {
    const b = bagOfWords(
      DOCS.map((d) => wordNgrams(d, [1, 2])),
      { binary: true },
    )
    expect(b.vocabulary.tokens).toEqual(F.vectorisers.count_binary_bigrams.vocabulary)
    expect(toRows(b.matrix)).toEqual(F.vectorisers.count_binary_bigrams.matrix)
  })
  it('filters by document frequency', () => {
    const b = bagOfWords(DOCS, { minDocuments: 2, maxDocumentShare: 0.7 })
    expect(b.vocabulary.tokens).toEqual(F.vectorisers.count_df_limits.vocabulary)
  })
})

describe('tfidf against TfidfVectorizer', () => {
  const counts = bagOfWords(DOCS).matrix
  const cases: [string, TfidfOptions][] = [
    ['default', {}],
    ['sublinear', { tf: 'log' }],
    ['no_smooth', { idf: 'standardPlusOne' }],
    ['l1', { norm: 'l1' }],
    ['no_norm_no_idf', { idf: 'none', norm: 'none' }],
  ]
  it.each(cases)('%s', (name, options) => {
    const ref = F.vectorisers[`tfidf_${name}`]
    close(toRows(tfidf(counts, options)), ref.matrix)
    if (ref.idf) close([toFlat(inverseDocumentFrequency(counts, options.idf ?? 'smooth'))], [ref.idf])
  })
})

// The five documents of the TF-IDF notes, tokenised keeping one-letter words.
const NOTE = [
  'the cat sat on the mat',
  'the dog sat on the log',
  'the cat chased the dog',
  'a dog and a cat played',
  'the bird sang',
]
const noteBag = bagOfWords(NOTE.map((d) => tokenise(d).tokens))
const col = (t: string) => tokenId(noteBag.vocabulary, t)
const query = (...terms: string[]) => {
  const q = new Array<number>(noteBag.vocabulary.tokens.length).fill(0)
  for (const t of terms) q[col(t)]++
  return q
}

describe('the TF-IDF note', () => {
  it('weights the by 0.223, cat by 0.511 and mat by 1.609, and scores "cat mat"', () => {
    const idf = toFlat(inverseDocumentFrequency(noteBag.matrix))
    expect([idf[col('the')], idf[col('cat')], idf[col('mat')]].map((x) => +x.toFixed(3))).toEqual([0.223, 0.511, 1.609])
    const w = toRows(tfidf(noteBag.matrix, { idf: 'standard', norm: 'none' }))
    const scores = w.map((row) => row[col('cat')] + row[col('mat')])
    expect(scores.map((x) => +x.toFixed(3))).toEqual([2.12, 0, 0.511, 0.511, 0])
  })
  it('gives BM25 scores 1.811, 0.548 and 0.507', () => {
    const s = toFlat(bm25(noteBag.matrix, query('cat', 'mat')))
    expect(s.map((x) => +x.toFixed(3))).toEqual([1.811, 0, 0.548, 0.507, 0])
  })
  it('makes the Robertson–Spärck Jones IDF of "the" negative (−1.10)', () => {
    expect(toFlat(inverseDocumentFrequency(noteBag.matrix, 'robertson'))[col('the')]).toBeCloseTo(
      Math.log(1.5 / 4.5),
      12,
    )
  })
  it('scores "cat mat" by lnc.ltc with base-10 logarithms: 0.526, 0.140, 0.127', () => {
    const docs = toRows(tfidf(noteBag.matrix, smartWeighting('lnc', { logBase: 10 })))
    // The query takes the collection's idf (ltc), then cosine normalisation.
    const idf = toFlat(inverseDocumentFrequency(noteBag.matrix, 'standard', { logBase: 10 }))
    const q = query('cat', 'mat').map((x, t) => x * idf[t])
    const z = Math.hypot(...q)
    const scores = docs.map((row) => row.reduce((s, x, t) => s + (x * q[t]) / z, 0))
    expect(scores.map((x) => +x.toFixed(3))).toEqual([0.526, 0, 0.14, 0.127, 0])
  })
})

describe('term frequency and BM25 variants', () => {
  const counts = [
    [4, 1, 0],
    [0, 2, 2],
  ]
  it('applies each term-frequency function where tf > 0', () => {
    expect(toRows(termFrequency(counts, 'binary'))).toEqual([
      [1, 1, 0],
      [0, 1, 1],
    ])
    expect(toRows(termFrequency(counts, 'augmented'))[0]).toEqual([1, 0.625, 0])
    expect(toRows(termFrequency(counts, 'log', { logBase: 2 }))[0]).toEqual([3, 1, 0])
    const la = toRows(termFrequency(counts, 'logAverage'))[0]
    expect(la[0]).toBeCloseTo((1 + Math.log(4)) / (1 + Math.log(2.5)), 14)
  })
  it('adds δ to every present term in BM25+, and reduces to presence with k1 = 0', () => {
    const plain = toRows(bm25Weights(counts, { idf: 'none' }))
    const plus = toRows(bm25Weights(counts, { idf: 'none', delta: 1 }))
    expect(plus[0][0] - plain[0][0]).toBeCloseTo(1, 14)
    expect(plus[0][2]).toBe(0)
    expect(toRows(bm25Weights(counts, { idf: 'none', k1: 0 }))).toEqual([
      [1, 1, 0],
      [0, 1, 1],
    ])
    // b = 0 ignores length: tf (k1 + 1) / (tf + k1)
    expect(toRows(bm25Weights(counts, { idf: 'none', b: 0 }))[0][0]).toBeCloseTo((4 * 2.2) / (4 + 1.2), 14)
  })
  it('rejects a SMART code it does not know', () => {
    expect(() => smartWeighting('lnu')).toThrow()
    expect(smartWeighting('ltc')).toEqual({ tf: 'log', idf: 'standard', norm: 'l2', logBase: Math.E })
  })
})

describe('n-grams against the scikit-learn analyzers', () => {
  const A = F.analyzers
  it('char', () => expect(characterNgrams(A.text, [2, 3])).toEqual(A.char_2_3))
  it('char_wb', () => {
    expect(characterNgrams(A.text, [2, 4], { wordBoundaries: true })).toEqual(A.char_wb_2_4)
    expect(characterNgrams(A.text, 5, { wordBoundaries: true })).toEqual(A.char_wb_5)
  })
  it('word', () => expect(wordNgrams(tokenise(A.text, { pattern: 'alphanumeric' }).tokens, [1, 3])).toEqual(A.word_1_3))
  it('joins with a custom joiner and rejects bad n', () => {
    expect(wordNgrams(['a', 'b', 'c'], 2, '_')).toEqual(['a_b', 'b_c'])
    expect(() => wordNgrams(['a'], [2, 1])).toThrow()
  })
})

describe('feature hashing against HashingVectorizer', () => {
  it('hashes like sklearn.utils.murmurhash3_32', () => {
    const { words, hashes, seeded } = F.murmurhash3
    expect(words.map((w) => murmurHash3(w))).toEqual(hashes)
    expect(words.map((w) => murmurHash3(w, 42))).toEqual(seeded)
  })
  it('gives the signed, unnormalised 8-column matrix of the feature-hashing note', () => {
    const docs = F.docs.map((d) => tokenise(d.toLowerCase(), { pattern: /(?<![\p{L}\p{N}_])[\p{L}\p{N}_]+/u }).tokens)
    expect(toRows(featureHash(docs, { features: 8, norm: 'none' }))).toEqual(F.vectorisers.hashing_8.matrix)
    expect(toRows(featureHash(docs, { features: 8, norm: 'none' }))[0]).toEqual([0, 0, 0, -1, 1, 0, -2, 0])
  })
  it('gives the unsigned, l2-normalised 16-column matrix', () => {
    close(toRows(featureHash(DOCS, { features: 16, signed: false })), F.vectorisers.hashing_16_unsigned_l2.matrix)
  })
  it('gives sparse rows with cancelled columns dropped', () => {
    const { columns, values } = hashedFeatures(['the', 'cat', 'sat', 'on', 'the', 'mat'], { features: 8 })
    expect(toFlat(columns)).toEqual([3, 4, 6])
    expect(toFlat(values)).toEqual([-1, 1, -2])
    expect(columns.dtype).toBe('int32')
  })
})
