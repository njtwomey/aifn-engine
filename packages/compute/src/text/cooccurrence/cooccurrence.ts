/**
 * Word–context co-occurrence counts in a window, pointwise mutual information (Church & Hanks 1990) with positive
 * clipping, context-distribution smoothing and shifting (Levy, Goldberg & Dagan 2015; Levy & Goldberg 2014), and dense
 * word vectors from a truncated singular value decomposition, U_d Σ_d^p.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, toFlat, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { buildVocabulary, tokenId, type Vocabulary } from 'aifn-compute/text/vocabulary'
import { truncatedSvd } from './svd'

/** A word × context count matrix with the vocabularies of its rows and columns. */
export interface Cooccurrence {
  readonly kind: 'cooccurrence'
  /** Weighted pair counts #(w, c) (float64 [V, C]). */
  readonly matrix: Tensor
  readonly words: Vocabulary
  readonly contexts: Vocabulary
}

/** Options of {@link cooccurrence}. */
export interface CooccurrenceOptions {
  /** Context words up to this distance on each side (default 2). */
  window?: number
  /** Count only contexts to the right (default false: both sides, so the matrix is symmetric). */
  rightOnly?: boolean
  /** Context words up to this distance on the left (default `window`, or 0 with `rightOnly`): an asymmetric window. */
  left?: number
  /** Context words up to this distance on the right (default `window`). */
  right?: number
  /**
   * How a context at distance d on a side of size L counts: `uniform` 1 (default), `harmonic` 1/d (GloVe), `linear`
   * (L − d + 1)/L (word2vec's dynamic window in expectation), `hal` L − d + 1 (the Hyperspace Analogue to Language,
   * Lund & Burgess 1996: the nearest context counts L, the farthest 1).
   */
  weighting?: 'uniform' | 'harmonic' | 'linear' | 'hal'
  /** The row vocabulary (default every word, by descending frequency, without specials). */
  words?: Vocabulary
  /** The column vocabulary (default the row vocabulary). */
  contexts?: Vocabulary
}

/**
 * Windowed co-occurrence counts: for every token w of every document (sentence), each token c at distance
 * d = 1 … left before it or d = 1 … right after it, inside the same document, adds weight(d) to #(w, c). Tokens outside
 * the vocabularies are skipped.
 */
export function cooccurrence(
  documents: readonly (readonly string[])[],
  options: CooccurrenceOptions = {},
): Cooccurrence {
  const { window = 2, rightOnly = false, weighting = 'uniform' } = options
  const left = options.left ?? (rightOnly ? 0 : window)
  const right = options.right ?? window
  const side = (L: number) => Number.isInteger(L) && L >= 0
  if (!(Number.isInteger(window) && window >= 1) || !side(left) || !side(right) || left + right < 1)
    throw new DomainError('cooccurrence', 'cooccurrence: the window sizes must be integers ≥ 0, not both 0')
  const words = options.words ?? buildVocabulary(documents, { specials: [] })
  const contexts = options.contexts ?? words
  const V = words.tokens.length
  const C = contexts.tokens.length
  const out = new Float64Array(V * C)
  const weight = (d: number, L: number) =>
    weighting === 'harmonic' ? 1 / d : weighting === 'linear' ? (L - d + 1) / L : weighting === 'hal' ? L - d + 1 : 1
  for (const doc of documents) {
    const w = doc.map((t) => tokenId(words, t))
    const c = doc.map((t) => tokenId(contexts, t))
    for (let i = 0; i < doc.length; i++) {
      if (w[i] < 0) continue
      for (let d = 1; d <= left; d++) {
        const j = i - d
        if (j >= 0 && c[j] >= 0) out[w[i] * C + c[j]] += weight(d, left)
      }
      for (let d = 1; d <= right; d++) {
        const j = i + d
        if (j < doc.length && c[j] >= 0) out[w[i] * C + c[j]] += weight(d, right)
      }
    }
  }
  return { kind: 'cooccurrence', matrix: fromData(out, [V, C]), words, contexts }
}

/** Options of {@link pmi} and {@link ppmi}. */
export interface PmiOptions {
  /** Context-distribution smoothing: P_α(c) ∝ #(c)^α (default 1, none; 0.75 as word2vec). */
  alpha?: number
  /** The base of the logarithm (default e; 2 for bits). */
  base?: number
}

function pmiOf(counts: MatrixLike, op: string, { alpha = 1, base = Math.E }: PmiOptions) {
  const { data, m, n } = dense.toMatrixF64(counts, op)
  const row = new Float64Array(m)
  const col = new Float64Array(n)
  let total = 0
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      const x = data[i * n + j]
      if (x < 0) throw new DomainError(op, `${op}: counts must be non-negative`)
      row[i] += x
      col[j] += x
      total += x
    }
  if (!(total > 0)) throw new DomainError(op, `${op}: the matrix has no counts`)
  let smoothed = 0
  for (const c of col) smoothed += c ** alpha
  const lb = Math.log(base)
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      const x = data[i * n + j]
      // log [ p(w, c) / (p(w) P_α(c)) ] with p(w, c) = x / |D|, p(w) = row / |D|, P_α(c) = col^α / Σ col^α.
      out[i * n + j] = x > 0 ? Math.log(x / total / ((row[i] / total) * (col[j] ** alpha / smoothed))) / lb : -Infinity
    }
  return { out, m, n }
}

/**
 * Pointwise mutual information of every cell of a count matrix (float64 [V, C]):
 * PMI(w, c) = log p̂(w, c) / (p̂(w) P_α(c)), −∞ where the count is 0.
 */
export function pmi(counts: MatrixLike, options: PmiOptions = {}): Tensor {
  const { out, m, n } = pmiOf(counts, 'pmi', options)
  return fromData(out, [m, n])
}

/**
 * Positive (shifted) PMI: max(PMI(w, c) − log k, 0), 0 where the count is 0 (float64 [V, C]). `shift` k = 1 (default)
 * is PPMI; k > 1 is the shifted PPMI that matches skip-gram with k negative samples (Levy & Goldberg 2014).
 */
export function ppmi(counts: MatrixLike, options: PmiOptions & { shift?: number } = {}): Tensor {
  const { shift = 1, base = Math.E } = options
  if (!(shift > 0)) throw new DomainError('ppmi', 'ppmi: shift must be positive')
  const { out, m, n } = pmiOf(counts, 'ppmi', options)
  const s = Math.log(shift) / Math.log(base)
  return fromData(
    out.map((x) => Math.max(x - s, 0)),
    [m, n],
  )
}

/**
 * Dense word vectors from a word × context matrix (counts, PPMI, …): the rows of U_d Σ_d^p from its singular value
 * decomposition, truncated to the `dimensions` largest singular values (float64 [V, d]), from {@link truncatedSvd}. `power` p = 1 is
 * the textbook truncated SVD (latent semantic analysis); Levy et al. (2015) found p = 0.5 (default) or 0 better on
 * similarity tasks.
 */
export function wordVectors(matrix: MatrixLike, dimensions: number, options: { power?: number } = {}): Tensor {
  const { power = 0.5 } = options
  const { m, n } = dense.toMatrixF64(matrix, 'wordVectors')
  const k = Math.min(m, n)
  if (!(Number.isInteger(dimensions) && dimensions >= 1 && dimensions <= k))
    throw new DomainError('wordVectors', `wordVectors: dimensions must be an integer in [1, ${k}]`)
  const { U, S } = truncatedSvd(matrix, dimensions)
  const u = toFlat(U)
  const s = toFlat(S)
  const out = new Float64Array(m * dimensions)
  for (let i = 0; i < m; i++)
    for (let j = 0; j < dimensions; j++) out[i * dimensions + j] = u[i * dimensions + j] * s[j] ** power
  return fromData(out, [m, dimensions])
}
