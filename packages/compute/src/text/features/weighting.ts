/**
 * Term weighting of a document–term count matrix: a term-frequency function, a document-frequency function and a
 * normalisation (Salton & Buckley 1988; SMART notation as in Manning, Raghavan & Schütze 2008, table 6.15), and the
 * BM25 family (Robertson & Zaragoza 2009), with BM25+'s lower bound on the term-frequency factor (Lv & Zhai 2011).
 *
 * Every function takes a document $\times$ term count matrix, one row per document, such as the `matrix` of
 * `bagOfWords`; $\mathrm{tf}_{t,d}$ is a cell, $\mathrm{df}_t$ the number of documents that contain term $t$ and $N$
 * the number of documents.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type MatrixLike, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'

/**
 * A term-frequency function $f(\mathrm{tf})$, applied where $\mathrm{tf} > 0$ (absent terms weigh 0): `raw`
 * $\mathrm{tf}$; `binary` 1; `log` $1 + \log \mathrm{tf}$ (sublinear); `augmented`
 * $0.5 + 0.5\,\mathrm{tf} / \max_s \mathrm{tf}_s$ over the document; `logAverage`
 * $(1 + \log \mathrm{tf}) / (1 + \log \operatorname{ave}_s \mathrm{tf}_s)$, the average over the document's present
 * terms.
 */
export type TfScheme = 'raw' | 'binary' | 'log' | 'augmented' | 'logAverage'

/**
 * A document-frequency function $g(\mathrm{df})$ over $N$ documents: `none` 1; `standard` $\log(N / \mathrm{df})$;
 * `standardPlusOne` $\log(N / \mathrm{df}) + 1$ (scikit-learn with `smooth_idf=False`); `smooth`
 * $\log((1 + N) / (1 + \mathrm{df})) + 1$ (scikit-learn's default); `probabilistic`
 * $\max(0, \log((N - \mathrm{df}) / \mathrm{df}))$; `robertson`
 * $\log((N - \mathrm{df} + \frac{1}{2}) / (\mathrm{df} + \frac{1}{2}))$, the Robertson–Spärck Jones weight, negative
 * for terms in more than half the documents; `nonNegative`
 * $\log(1 + (N - \mathrm{df} + \frac{1}{2}) / (\mathrm{df} + \frac{1}{2}))$, Lucene's BM25 IDF.
 */
export type IdfScheme =
  'none' | 'standard' | 'standardPlusOne' | 'smooth' | 'probabilistic' | 'robertson' | 'nonNegative'

/**
 * A normalisation of each document's weight vector: `none`, `l1` ($\sum_t \lvert w_t \rvert = 1$) or `l2` (unit
 * length; cosine).
 */
export type NormScheme = 'none' | 'l1' | 'l2'

/** Options of {@link tfidf}. */
export interface TfidfOptions {
  /** The term-frequency function (default `raw`). */
  tf?: TfScheme
  /** The document-frequency function (default `smooth`). */
  idf?: IdfScheme
  /** The row normalisation (default `l2`). */
  norm?: NormScheme
  /** The base of every logarithm (default $e$). */
  logBase?: number
}

/**
 * A count matrix as a dense row-major float64 array, for reading.
 *
 * @param counts The matrix, documents $\times$ terms.
 * @param op The caller's name, for error messages.
 * @returns Its entries `data` and its numbers of rows `m` and columns `n`.
 */
const matrix = (counts: MatrixLike, op: string) => dense.toMatrixF64(counts, op)

/**
 * Document frequencies $\mathrm{df}_t$, the number of documents (rows) with a positive count of each term (float64
 * [V]).
 *
 * @param counts The count matrix, documents $\times$ terms.
 * @returns One frequency per term.
 *
 * @example How many documents hold each term
 * const docs = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['the', 'dog', 'sat'], ['the', 'cat', 'ate', 'the', 'fish']]
 * const { matrix, vocabulary } = bagOfWords(docs)
 * print(vocabulary.tokens)
 * print(documentFrequency(matrix))
 */
export function documentFrequency(counts: MatrixLike): Tensor {
  const { data, m, n } = matrix(counts, 'documentFrequency')
  const df = new Float64Array(n)
  for (let d = 0; d < m; d++) for (let t = 0; t < n; t++) if (data[d * n + t] > 0) df[t]++
  return fromData(df)
}

/**
 * The inverse document frequency of each term under a scheme. Throws `DomainError` for an unknown scheme.
 *
 * @param df The document frequency of each term.
 * @param N The number of documents.
 * @param scheme The document-frequency function.
 * @param log The logarithm to use.
 * @returns One weight per term.
 */
function idfOf(df: ArrayLike<number>, N: number, scheme: IdfScheme, log: (x: number) => number): Float64Array {
  return Float64Array.from(df, (f) => {
    switch (scheme) {
      case 'none':
        return 1
      case 'standard':
        return log(N / f)
      case 'standardPlusOne':
        return log(N / f) + 1
      case 'smooth':
        return log((1 + N) / (1 + f)) + 1
      case 'probabilistic':
        return Math.max(0, log((N - f) / f))
      case 'robertson':
        return log((N - f + 0.5) / (f + 0.5))
      case 'nonNegative':
        return log(1 + (N - f + 0.5) / (f + 0.5))
      default:
        throw new DomainError(
          'inverseDocumentFrequency',
          `inverseDocumentFrequency: unknown scheme '${String(scheme)}'`,
        )
    }
  })
}

/**
 * The logarithm to a base.
 *
 * @param base The base; `Math.E` gives `Math.log` itself.
 * @returns A function of $x$ giving $\log_{\text{base}} x$.
 */
const logIn = (base: number) => (base === Math.E ? Math.log : (x: number) => Math.log(x) / Math.log(base))

/**
 * The inverse document frequency of every term of a count matrix (documents $\times$ terms) under `scheme` (default
 * `standard`), float64 [V]. A term in no document gets $+\infty$ under `standard`; its weight is 0 in `tfidf` since its
 * counts are. Throws `DomainError` for an unknown scheme.
 *
 * @param counts The count matrix, documents $\times$ terms.
 * @param scheme The document-frequency function; see {@link IdfScheme}.
 * @param options The base of the logarithm.
 * @param options.logBase The base (default $e$).
 * @returns One weight per term.
 *
 * @example scikit-learn's smooth IDF, and the plain one
 * const docs = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['the', 'dog', 'sat'], ['the', 'cat', 'ate', 'the', 'fish']]
 * const { matrix, vocabulary } = bagOfWords(docs)
 * print(vocabulary.tokens)
 * print('smooth  ', inverseDocumentFrequency(matrix, 'smooth'))
 * print('standard', inverseDocumentFrequency(matrix))
 */
export function inverseDocumentFrequency(
  counts: MatrixLike,
  scheme: IdfScheme = 'standard',
  options: { logBase?: number } = {},
): Tensor {
  const { m } = matrix(counts, 'inverseDocumentFrequency')
  return fromData(idfOf(documentFrequency(counts).data, m, scheme, logIn(options.logBase ?? Math.E)))
}

/**
 * The term-frequency factor $f(\mathrm{tf})$ of every cell (float64 [D, V]), zero where $\mathrm{tf} = 0$.
 *
 * @param counts The count matrix, documents $\times$ terms.
 * @param scheme The term-frequency function; see {@link TfScheme}. An unknown name is read as `logAverage`.
 * @param options The base of the logarithm.
 * @param options.logBase The base (default $e$).
 * @returns The factors, with the shape of `counts`.
 *
 * @example Raw, sublinear and augmented term frequency of one document
 * const counts = tensor([[1, 2, 4, 0]])
 * print('raw      ', termFrequency(counts))
 * print('log      ', termFrequency(counts, 'log'))
 * print('augmented', termFrequency(counts, 'augmented'))
 */
export function termFrequency(
  counts: MatrixLike,
  scheme: TfScheme = 'raw',
  options: { logBase?: number } = {},
): Tensor {
  const { data, m, n } = matrix(counts, 'termFrequency')
  const log = logIn(options.logBase ?? Math.E)
  const out = new Float64Array(m * n)
  for (let d = 0; d < m; d++) {
    let max = 0
    let sum = 0
    let present = 0
    for (let t = 0; t < n; t++) {
      const c = data[d * n + t]
      if (c > 0) [max, sum, present] = [Math.max(max, c), sum + c, present + 1]
    }
    for (let t = 0; t < n; t++) {
      const c = data[d * n + t]
      if (!(c > 0)) continue
      out[d * n + t] =
        scheme === 'raw'
          ? c
          : scheme === 'binary'
            ? 1
            : scheme === 'log'
              ? 1 + log(c)
              : scheme === 'augmented'
                ? 0.5 + (0.5 * c) / max
                : (1 + log(c)) / (1 + log(sum / present))
    }
  }
  return fromData(out, [m, n])
}

/**
 * Normalise each row of a row-major matrix in place; a zero row is left as it is.
 *
 * @param w The matrix, $m \times n$, row-major; overwritten.
 * @param m The number of rows.
 * @param n The number of columns.
 * @param norm The normalisation: `none` leaves `w` unchanged.
 */
function normaliseRows(w: Float64Array, m: number, n: number, norm: NormScheme): void {
  if (norm === 'none') return
  for (let d = 0; d < m; d++) {
    let z = 0
    for (let t = 0; t < n; t++) z += norm === 'l1' ? Math.abs(w[d * n + t]) : w[d * n + t] ** 2
    if (norm === 'l2') z = Math.sqrt(z)
    // An empty document stays a zero row (scikit-learn does the same).
    if (z > 0) for (let t = 0; t < n; t++) w[d * n + t] /= z
  }
}

/**
 * TF-IDF weights $w_{t,d} = f(\mathrm{tf}_{t,d}) \cdot g(\mathrm{df}_t)$, each row then normalised, from a count matrix
 * (documents $\times$ terms; float64 [D, V]). The defaults (raw tf, smooth idf, l2) equal scikit-learn's
 * `TfidfVectorizer`; `tf: 'log'` is its `sublinear_tf`. Throws `DomainError` for an unknown IDF scheme.
 *
 * @param counts The count matrix, documents $\times$ terms.
 * @param options The term-frequency and document-frequency functions, the normalisation and the base of the logarithm;
 *   see {@link TfidfOptions}.
 * @returns The weights, with the shape of `counts`.
 *
 * @example TF-IDF of three short documents, as scikit-learn's `TfidfVectorizer`
 * const docs = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['the', 'dog', 'sat'], ['the', 'cat', 'ate', 'the', 'fish']]
 * const { matrix, vocabulary } = bagOfWords(docs)
 * print(vocabulary.tokens)
 * print(tfidf(matrix))
 *
 * @example The SMART weighting ltc
 * const docs = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['the', 'dog', 'sat'], ['the', 'cat', 'ate', 'the', 'fish']]
 * const { matrix, vocabulary } = bagOfWords(docs)
 * print(tfidf(matrix, smartWeighting('ltc')))
 */
export function tfidf(counts: MatrixLike, options: TfidfOptions = {}): Tensor {
  const { tf = 'raw', idf = 'smooth', norm = 'l2', logBase = Math.E } = options
  const { m, n } = matrix(counts, 'tfidf')
  const f = termFrequency(counts, tf, { logBase }).data as Float64Array
  const g = idfOf(documentFrequency(counts).data, m, idf, logIn(logBase))
  const w = new Float64Array(m * n)
  for (let d = 0; d < m; d++) for (let t = 0; t < n; t++) if (f[d * n + t] !== 0) w[d * n + t] = f[d * n + t] * g[t]
  normaliseRows(w, m, n, norm)
  return fromData(w, [m, n])
}

/**
 * TF-IDF options from a SMART code of three letters (Salton & Buckley 1988): term frequency n (raw), l (log),
 * a (augmented), b (binary), L (log average); document frequency n (none), t (standard), p (probabilistic);
 * normalisation n (none), c (cosine, l2). "lnc" and "ltc" are the document and query weightings of lnc.ltc. Throws
 * `DomainError` for a code outside these letters.
 *
 * @param code The three letters: term frequency, document frequency, normalisation.
 * @param options The base of the logarithm.
 * @param options.logBase The base (default $e$).
 * @returns Options for `tfidf`.
 *
 * @example The two halves of lnc.ltc
 * print('lnc', smartWeighting('lnc'))
 * print('ltc', smartWeighting('ltc', { logBase: 10 }))
 */
export function smartWeighting(code: string, options: { logBase?: number } = {}): Required<TfidfOptions> {
  const tf: Record<string, TfScheme> = { n: 'raw', l: 'log', a: 'augmented', b: 'binary', L: 'logAverage' }
  const idf: Record<string, IdfScheme> = { n: 'none', t: 'standard', p: 'probabilistic' }
  const norm: Record<string, NormScheme> = { n: 'none', c: 'l2' }
  const [a, b, c] = code
  if (code.length !== 3 || !(a in tf) || !(b in idf) || !(c in norm))
    throw new DomainError('smartWeighting', `smartWeighting: '${code}' is not a supported SMART code (e.g. 'ltc')`)
  return { tf: tf[a], idf: idf[b], norm: norm[c], logBase: options.logBase ?? Math.E }
}

/** Options of {@link bm25Weights} and {@link bm25}. */
export interface Bm25Options {
  /** Term-frequency saturation $k_1 \ge 0$ (default 1.2). */
  k1?: number
  /** Length normalisation $b \in [0, 1]$ (default 0.75). */
  b?: number
  /** BM25+'s lower bound $\delta$ added to the term-frequency factor of every present term (default 0, plain BM25). */
  delta?: number
  /**
   * The IDF (default `nonNegative`, Lucene's $\log(1 + (N - \mathrm{df} + \frac{1}{2}) / (\mathrm{df} +
   * \frac{1}{2}))$), always with the natural logarithm.
   */
  idf?: IdfScheme
  /** Document lengths $L_d$, one per document (default the row sums of the counts). */
  lengths?: VectorLike
}

/**
 * The BM25 weight of every term in every document (float64 [D, V]):
 * $w_{d,t} = \mathrm{idf}_t \cdot (\mathrm{tf}\,(k_1 + 1) / (\mathrm{tf} + k_1 (1 - b + b L_d / \bar L)) + \delta)$
 * where $\mathrm{tf} > 0$, and 0 elsewhere, with $\bar L$ the mean document length. A query's score is the sum of its
 * terms' columns ({@link bm25}). Throws `DomainError` unless $k_1 \ge 0$ and $0 \le b \le 1$.
 *
 * @param counts The count matrix, documents $\times$ terms.
 * @param options The parameters $k_1$, $b$ and $\delta$, the IDF and the document lengths; see {@link Bm25Options}.
 * @returns The weights, with the shape of `counts`.
 *
 * @example The weight of a term saturates as its count grows
 * const counts = tensor([[1, 1], [2, 0], [8, 0], [0, 3]])
 * print(bm25Weights(counts))
 * print('BM25+', bm25Weights(counts, { delta: 1 }))
 */
export function bm25Weights(counts: MatrixLike, options: Bm25Options = {}): Tensor {
  const { k1 = 1.2, b = 0.75, delta = 0, idf = 'nonNegative' } = options
  if (!(k1 >= 0) || !(b >= 0 && b <= 1))
    throw new DomainError('bm25Weights', 'bm25Weights: need k1 ≥ 0 and b in [0, 1]')
  const { data, m, n } = matrix(counts, 'bm25Weights')
  const lengths = options.lengths
    ? dense.toF64(options.lengths, 'bm25Weights')
    : Float64Array.from({ length: m }, (_, d) => data.subarray(d * n, (d + 1) * n).reduce((s, x) => s + x, 0))
  const avg = lengths.reduce((s, x) => s + x, 0) / m
  const g = idfOf(documentFrequency(counts).data, m, idf, Math.log)
  const w = new Float64Array(m * n)
  for (let d = 0; d < m; d++) {
    const B = avg > 0 ? 1 - b + (b * lengths[d]) / avg : 1
    for (let t = 0; t < n; t++) {
      const tf = data[d * n + t]
      if (tf > 0) w[d * n + t] = g[t] * ((tf * (k1 + 1)) / (tf + k1 * B) + delta)
    }
  }
  return fromData(w, [m, n])
}

/**
 * BM25 (or BM25+ with `delta`) scores of every document for a query (float64 [D]): the sum, over the distinct terms
 * the query contains (`query` is its count vector over the same $V$ terms), of the documents' BM25 weights. Throws
 * `DomainError` when the query's length differs from the number of terms.
 *
 * @param counts The count matrix, documents $\times$ terms.
 * @param query The query's count of each term; only whether a count is positive matters.
 * @param options The parameters of the weights; see {@link Bm25Options}.
 * @returns One score per document.
 *
 * @example Rank three documents for the query "cat sat"
 * const docs = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['the', 'dog', 'sat'], ['the', 'cat', 'ate', 'the', 'fish']]
 * const { matrix, vocabulary } = bagOfWords(docs)
 * const query = vocabulary.tokens.map((t) => (t === 'cat' || t === 'sat' ? 1 : 0))
 * print('scores', bm25(matrix, query))
 */
export function bm25(counts: MatrixLike, query: VectorLike, options: Bm25Options = {}): Tensor {
  const w = bm25Weights(counts, options)
  const [m, n] = w.shape
  const q = dense.toF64(query, 'bm25')
  if (q.length !== n) throw new DomainError('bm25', `bm25: the query has ${q.length} terms, the counts ${n}`)
  const out = new Float64Array(m)
  const wd = w.data as Float64Array
  for (let d = 0; d < m; d++) for (let t = 0; t < n; t++) if (q[t] > 0) out[d] += wd[d * n + t]
  return fromData(out)
}
