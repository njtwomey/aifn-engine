/**
 * Term weighting of a document–term count matrix: a term-frequency function, a document-frequency function and a
 * normalisation (Salton & Buckley 1988; SMART notation as in Manning, Raghavan & Schütze 2008, table 6.15), and the
 * BM25 family (Robertson & Zaragoza 2009), with BM25+'s lower bound on the term-frequency factor (Lv & Zhai 2011).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type MatrixLike, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'

/**
 * A term-frequency function f(tf), applied where tf > 0 (absent terms weigh 0): `raw` tf; `binary` 1; `log`
 * 1 + log tf (sublinear); `augmented` 0.5 + 0.5 tf / max_s tf_s over the document; `logAverage`
 * (1 + log tf) / (1 + log ave_s tf_s), the average over the document's present terms.
 */
export type TfScheme = 'raw' | 'binary' | 'log' | 'augmented' | 'logAverage'

/**
 * A document-frequency function g(df) over N documents: `none` 1; `standard` log(N / df); `standardPlusOne`
 * log(N / df) + 1 (scikit-learn with `smooth_idf=False`); `smooth` log((1 + N) / (1 + df)) + 1 (scikit-learn's
 * default); `probabilistic` max(0, log((N − df) / df)); `robertson` log((N − df + ½) / (df + ½)), the Robertson–Spärck
 * Jones weight, negative for terms in more than half the documents; `nonNegative` log(1 + (N − df + ½) / (df + ½)),
 * Lucene's BM25 IDF.
 */
export type IdfScheme =
  'none' | 'standard' | 'standardPlusOne' | 'smooth' | 'probabilistic' | 'robertson' | 'nonNegative'

/** A normalisation of each document's weight vector: `none`, `l1` (sum of |w| = 1) or `l2` (unit length; cosine). */
export type NormScheme = 'none' | 'l1' | 'l2'

/** Options of {@link tfidf}. */
export interface TfidfOptions {
  /** Default `raw`. */
  tf?: TfScheme
  /** Default `smooth`. */
  idf?: IdfScheme
  /** Default `l2`. */
  norm?: NormScheme
  /** The base of every logarithm (default e). */
  logBase?: number
}

const matrix = (counts: MatrixLike, op: string) => dense.toMatrixF64(counts, op)

/** Document frequencies df_t, the number of documents (rows) with a positive count of each term (float64 [V]). */
export function documentFrequency(counts: MatrixLike): Tensor {
  const { data, m, n } = matrix(counts, 'documentFrequency')
  const df = new Float64Array(n)
  for (let d = 0; d < m; d++) for (let t = 0; t < n; t++) if (data[d * n + t] > 0) df[t]++
  return fromData(df)
}

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

const logIn = (base: number) => (base === Math.E ? Math.log : (x: number) => Math.log(x) / Math.log(base))

/**
 * The inverse document frequency of every term of a count matrix (documents × terms) under `scheme` (default
 * `standard`), float64 [V]. A term in no document gets +∞ under `standard`; its weight is 0 in `tfidf` since its
 * counts are.
 */
export function inverseDocumentFrequency(
  counts: MatrixLike,
  scheme: IdfScheme = 'standard',
  options: { logBase?: number } = {},
): Tensor {
  const { m } = matrix(counts, 'inverseDocumentFrequency')
  return fromData(idfOf(documentFrequency(counts).data, m, scheme, logIn(options.logBase ?? Math.E)))
}

/** The term-frequency factor f(tf) of every cell (float64 [D, V]), zero where tf = 0. */
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
 * TF-IDF weights w_{t,d} = f(tf_{t,d}) · g(df_t), each row then normalised, from a count matrix (documents × terms;
 * float64 [D, V]). The defaults (raw tf, smooth idf, l2) equal scikit-learn's `TfidfVectorizer`; `tf: 'log'` is its
 * `sublinear_tf`.
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
 * normalisation n (none), c (cosine, l2). "lnc" and "ltc" are the document and query weightings of lnc.ltc.
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
  /** Term-frequency saturation k₁ ≥ 0 (default 1.2). */
  k1?: number
  /** Length normalisation b ∈ [0, 1] (default 0.75). */
  b?: number
  /** BM25+'s lower bound δ added to the term-frequency factor of every present term (default 0, plain BM25). */
  delta?: number
  /** The IDF (default `nonNegative`, Lucene's log(1 + (N − df + ½) / (df + ½))). */
  idf?: IdfScheme
  /** Document lengths L_d (default the row sums of the counts). */
  lengths?: VectorLike
}

/**
 * The BM25 weight of every term in every document (float64 [D, V]):
 * w_{d,t} = idf_t · (tf (k₁ + 1) / (tf + k₁ (1 − b + b L_d / L̄)) + δ) where tf > 0, and 0 elsewhere. A query's score
 * is the sum of its terms' columns ({@link bm25}).
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
 * the query contains (`query` is its count vector over the same V terms), of the documents' BM25 weights.
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
