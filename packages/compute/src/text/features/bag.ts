/**
 * The bag of words: a document–term matrix of counts (or presence) over a vocabulary, the representation that TF-IDF,
 * BM25, naive Bayes and latent semantic analysis start from (Salton, Wong & Yang 1975).
 */

import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { buildVocabulary, tokenCounts, tokenId, vocabularyOf, type Vocabulary } from 'aifn-compute/text/vocabulary'

/** A document–term matrix with the vocabulary of its columns. */
export interface BagOfWords {
  readonly kind: 'bag-of-words'
  /** Counts or presence (float64 [D, V]). */
  readonly matrix: Tensor
  readonly vocabulary: Vocabulary
}

/** Options of {@link bagOfWords}. */
export interface BagOfWordsOptions {
  /** The columns; default built from the documents, in code-point order, without specials (as `CountVectorizer`). */
  vocabulary?: Vocabulary
  /** Presence (1/0) instead of counts (default false). */
  binary?: boolean
  /** When building the vocabulary: keep terms in at least this many documents (default 1). */
  minDocuments?: number
  /** When building the vocabulary: drop terms in more than this share of the documents (default 1, keep all). */
  maxDocumentShare?: number
  /** When building the vocabulary: keep at most this many terms, the most frequent in the corpus. */
  maxSize?: number
}

/**
 * The document–term matrix of tokenised documents. A token outside the vocabulary is counted in the unknown column if
 * the vocabulary has one, and ignored otherwise.
 */
export function bagOfWords(documents: readonly (readonly string[])[], options: BagOfWordsOptions = {}): BagOfWords {
  let vocabulary = options.vocabulary
  if (!vocabulary) {
    const { tokens, documentCounts } = tokenCounts(documents)
    const df = toFlat(documentCounts)
    const { minDocuments = 1, maxDocumentShare = 1 } = options
    const keep = new Set(tokens.filter((_, k) => df[k] >= minDocuments && df[k] <= maxDocumentShare * documents.length))
    const filtered = documents.map((d) => d.filter((t) => keep.has(t)))
    vocabulary =
      options.maxSize === undefined
        ? vocabularyOf([...keep].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)))
        : buildVocabulary(filtered, { specials: [], order: 'alphabetical', maxSize: options.maxSize })
  }
  const V = vocabulary.tokens.length
  const out = new Float64Array(documents.length * V)
  documents.forEach((doc, d) => {
    for (const t of doc) {
      let id = tokenId(vocabulary!, t)
      if (id < 0) id = vocabulary!.unknown
      if (id < 0) continue
      out[d * V + id] = options.binary ? 1 : out[d * V + id] + 1
    }
  })
  return { kind: 'bag-of-words', matrix: fromData(out, [documents.length, V]), vocabulary }
}
