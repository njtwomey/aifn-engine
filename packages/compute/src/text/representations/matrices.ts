/**
 * The two matrices of distributional semantics (Turney & Pantel 2010): the term–document matrix, whose rows say which
 * documents a word occurs in (the input of latent semantic analysis, Deerwester et al. 1990), and the term–term
 * matrix, whose rows say which words occur near it in a context window (as in HAL, Lund & Burgess 1996, and the
 * count models of Levy, Goldberg & Dagan 2015). Both are counted, then weighted.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, transpose, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { cooccurrence, ppmi, type CooccurrenceOptions, type PmiOptions } from 'aifn-compute/text/cooccurrence'
import { bagOfWords, tfidf, type TfidfOptions } from 'aifn-compute/text/features'
import { buildVocabulary, type Vocabulary } from 'aifn-compute/text/vocabulary'

/**
 * A cell weighting of a count matrix whose columns are documents or contexts: `raw` counts; `binary` presence;
 * `log` log(1 + count), which damps repeats; `tfidf` each column (a document) weighted by TF-IDF over the columns and
 * normalised to unit length, a row (term) that occurs in every column getting the least weight; `ppmi` positive
 * pointwise mutual information of row and column.
 */
export type MatrixWeighting = 'raw' | 'binary' | 'log' | 'tfidf' | 'ppmi'

/** Options of {@link weightMatrix}. */
export interface WeightOptions {
  /** TF-IDF options (default scikit-learn's: raw tf, smooth idf, l2 per column). */
  tfidf?: TfidfOptions
  /** PPMI options: context smoothing α and shift k. */
  ppmi?: PmiOptions & { shift?: number }
}

/** A count matrix (rows × columns) under a {@link MatrixWeighting} (float64, same shape). */
export function weightMatrix(counts: MatrixLike, weighting: MatrixWeighting, options: WeightOptions = {}): Tensor {
  const { data, m, n } = dense.toMatrixF64(counts, 'weightMatrix')
  switch (weighting) {
    case 'raw':
      return fromData(Float64Array.from(data), [m, n])
    case 'binary':
      return fromData(
        data.map((x) => (x > 0 ? 1 : 0)),
        [m, n],
      )
    case 'log':
      return fromData(
        data.map((x) => Math.log1p(x)),
        [m, n],
      )
    case 'tfidf':
      // tfidf weighs documents × terms; the columns here are the documents.
      return transpose(tfidf(transpose(fromData(data, [m, n])), options.tfidf)) as Tensor
    case 'ppmi':
      return ppmi(fromData(data, [m, n]), options.ppmi)
    default:
      throw new DomainError('weightMatrix', `weightMatrix: unknown weighting '${String(weighting)}'`)
  }
}

/** A weighted term × column matrix, with its counts and the names of its rows and columns. */
export interface TermMatrix {
  readonly kind: 'term-matrix'
  /** The weighted matrix (float64 [V, C]). */
  readonly matrix: Tensor
  /** The raw counts (float64 [V, C]). */
  readonly counts: Tensor
  /** The rows: one word each. */
  readonly terms: Vocabulary
  /** The columns: document names `d1 … dD`, or context words. */
  readonly columns: readonly string[]
  readonly weighting: MatrixWeighting
}

/** Options of {@link termDocumentMatrix}. */
export interface TermDocumentOptions extends WeightOptions {
  /** Default `raw`. */
  weighting?: MatrixWeighting
  /** The rows (default every word seen at least `minCount` times, most frequent first, without specials). */
  terms?: Vocabulary
  /** Default 1. */
  minCount?: number
}

const vocabularyFor = (documents: readonly (readonly string[])[], minCount = 1) =>
  buildVocabulary(documents, { specials: [], minCount })

/**
 * The term–document matrix of tokenised documents (float64 [V, D]): entry (t, d) is the weighted count of term t in
 * document d. It is the transpose of the bag of words.
 */
export function termDocumentMatrix(
  documents: readonly (readonly string[])[],
  options: TermDocumentOptions = {},
): TermMatrix {
  const { weighting = 'raw' } = options
  const terms = options.terms ?? vocabularyFor(documents, options.minCount)
  const counts = transpose(bagOfWords(documents, { vocabulary: terms }).matrix) as Tensor
  return {
    kind: 'term-matrix',
    matrix: weightMatrix(counts, weighting, options),
    counts,
    terms,
    columns: documents.map((_, d) => `d${d + 1}`),
    weighting,
  }
}

/** Options of {@link termTermMatrix}: the window and distance weighting of {@link cooccurrence}, and a cell weighting. */
export interface TermTermOptions extends WeightOptions, Omit<CooccurrenceOptions, 'words' | 'contexts' | 'weighting'> {
  /** Default `raw`. */
  weighting?: MatrixWeighting
  /** How a context counts by its distance (`cooccurrence`'s `weighting`; default `uniform`). */
  distance?: CooccurrenceOptions['weighting']
  /** The rows (default every word seen at least `minCount` times, most frequent first). */
  terms?: Vocabulary
  /** The columns (default the rows). */
  contexts?: Vocabulary
  /** Default 1. */
  minCount?: number
}

/**
 * The term–term (word × context) matrix of tokenised documents (float64 [V, C]): entry (w, c) is the weighted,
 * distance-weighted count of c within the window around w. Symmetric windows with uniform weights give a symmetric
 * matrix; `left`/`right` make the window asymmetric and `distance: 'hal'` weighs near contexts more, as HAL does.
 */
export function termTermMatrix(documents: readonly (readonly string[])[], options: TermTermOptions = {}): TermMatrix {
  const { weighting = 'raw', distance = 'uniform' } = options
  const terms = options.terms ?? vocabularyFor(documents, options.minCount)
  const contexts = options.contexts ?? terms
  const c = cooccurrence(documents, {
    window: options.window,
    left: options.left,
    right: options.right,
    rightOnly: options.rightOnly,
    weighting: distance,
    words: terms,
    contexts,
  })
  const empty = (c.matrix.data as Float64Array).every((x) => x === 0)
  if (weighting === 'ppmi' && empty)
    throw new DomainError('termTermMatrix', 'termTermMatrix: no co-occurrences in the window, so no PPMI')
  return {
    kind: 'term-matrix',
    matrix: weightMatrix(c.matrix, weighting, options),
    counts: c.matrix,
    terms,
    columns: contexts.tokens,
    weighting,
  }
}
