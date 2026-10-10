/**
 * The bag of words: a document–term matrix of counts (or presence) over a vocabulary, the representation that TF-IDF,
 * BM25, naive Bayes and latent semantic analysis start from (Salton, Wong & Yang 1975).
 */

import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { buildVocabulary, tokenCounts, tokenId, vocabularyOf, type Vocabulary } from 'aifn-compute/text/vocabulary'

/** A document–term matrix with the vocabulary of its columns. */
export interface BagOfWords {
  /** The tag `'bag-of-words'`. */
  readonly kind: 'bag-of-words'
  /** Counts or presence (float64 [D, V]). */
  readonly matrix: Tensor
  /** The vocabulary: column $j$ counts the token with id $j$. */
  readonly vocabulary: Vocabulary
}

/** Options of {@link bagOfWords}. */
export interface BagOfWordsOptions {
  /**
   * The columns; default built from the documents, in code-unit order (code-point order outside the astral planes),
   * without specials (as `CountVectorizer`).
   */
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
 * The document–term matrix of tokenised documents, as scikit-learn's `CountVectorizer` on pre-tokenised input. A token
 * outside the vocabulary is counted in the unknown column if the vocabulary has one, and ignored otherwise.
 *
 * @param documents The tokenised documents, one row each.
 * @param options The vocabulary (or how to build one) and whether to record presence; see {@link BagOfWordsOptions}.
 * @returns The matrix (float64 [D, V]) and the vocabulary of its columns.
 *
 * @example Counts and presence
 * const docs = [['the', 'cat', 'sat'], ['the', 'cat', 'saw', 'the', 'cat'], ['a', 'dog']]
 * const b = bagOfWords(docs)
 * print('terms ', b.vocabulary.tokens)
 * print('counts', b.matrix)
 * print('binary', bagOfWords(docs, { binary: true }).matrix)
 *
 * @example Dropping terms in every document but one, as `max_df`
 * const docs = [['the', 'cat', 'sat'], ['the', 'cat', 'saw', 'the', 'cat'], ['the', 'dog']]
 * print(bagOfWords(docs, { maxDocumentShare: 0.5 }).vocabulary.tokens)
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
