/**
 * Latent semantic analysis (Deerwester et al. 1990): the truncated SVD A ≈ U_k Σ_k V_kᵀ of a term × document (or term ×
 * context) matrix gives each term the coordinates U_k Σ_k and each column V_k Σ_k, and terms are compared by the cosine
 * of their coordinates. Nearest neighbours by cosine, and analogies by vector offset (Mikolov et al. 2013): the answer
 * to a − b + c is the word whose unit vector is closest to â − b̂ + ĉ, the inputs excluded (3CosAdd, Levy & Goldberg
 * 2014).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, toFlat, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { truncatedSvd } from 'aifn-compute/text/cooccurrence'
import { tokenId, type Vocabulary } from 'aifn-compute/text/vocabulary'

/** The latent semantic analysis of a matrix at rank k. */
export interface Lsa {
  readonly kind: 'lsa'
  /** Row (term) coordinates U_k Σ_k (float64 [m, k]). */
  readonly rows: Tensor
  /** Column (document or context) coordinates V_k Σ_k (float64 [n, k]). */
  readonly columns: Tensor
  /** The k largest singular values (float64 [k]). */
  readonly singularValues: Tensor
  /** σᵢ² / ‖A‖²_F per component (float64 [k]). */
  readonly energy: Tensor
}

/** The rank-k latent semantic analysis of a term × column matrix (counts, TF-IDF, PPMI, …). */
export function lsa(matrix: MatrixLike, k: number): Lsa {
  const { U, S, V, energy } = truncatedSvd(matrix, k)
  const s = toFlat(S)
  const scale = (t: Tensor) => {
    const [r] = t.shape
    const d = t.data as Float64Array
    const out = new Float64Array(r * k)
    for (let i = 0; i < r; i++) for (let j = 0; j < k; j++) out[i * k + j] = d[i * k + j] * s[j]
    return fromData(out, [r, k])
  }
  return { kind: 'lsa', rows: scale(U), columns: scale(V), singularValues: S, energy }
}

/** Rows scaled to unit length; a zero row stays zero. */
function unitRows(data: ArrayLike<number>, m: number, n: number): Float64Array {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) {
    let z = 0
    for (let j = 0; j < n; j++) z += data[i * n + j] ** 2
    z = Math.sqrt(z)
    if (z > 0) for (let j = 0; j < n; j++) out[i * n + j] = data[i * n + j] / z
  }
  return out
}

/**
 * The cosine similarity of every pair of rows (float64 [m, m]), xᵢ·xⱼ / (‖xᵢ‖ ‖xⱼ‖); 0 where either row is zero, so a
 * zero vector is similar to nothing.
 */
export function cosineSimilarities(vectors: MatrixLike): Tensor {
  const { data, m, n } = dense.toMatrixF64(vectors, 'cosineSimilarities')
  const u = unitRows(data, m, n)
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let j = i; j < m; j++) {
      let s = 0
      for (let c = 0; c < n; c++) s += u[i * n + c] * u[j * n + c]
      out[i * m + j] = out[j * m + i] = s
    }
  return fromData(out, [m, m])
}

/** A ranked neighbour: its row and its cosine similarity to the query. */
export interface Neighbour {
  readonly index: number
  readonly cosine: number
}

/**
 * The `count` rows with the highest cosine similarity to a query (a row index, or a vector of the same width), most
 * similar first, ties by index. A row-index query excludes itself; `exclude` drops more rows.
 */
export function nearestByCosine(
  vectors: MatrixLike,
  query: number | ArrayLike<number>,
  options: { count?: number; exclude?: readonly number[] } = {},
): Neighbour[] {
  const { data, m, n } = dense.toMatrixF64(vectors, 'nearestByCosine')
  const { count = 10 } = options
  let q: Float64Array
  if (typeof query === 'number') {
    if (!(Number.isInteger(query) && query >= 0 && query < m))
      throw new DomainError('nearestByCosine', `nearestByCosine: row ${query} is out of range [0, ${m})`)
    q = data.slice(query * n, (query + 1) * n)
  } else {
    q = Float64Array.from(query)
    if (q.length !== n)
      throw new DomainError('nearestByCosine', `nearestByCosine: the query has ${q.length} values, the rows ${n}`)
  }
  const skip = new Set(options.exclude ?? [])
  if (typeof query === 'number') skip.add(query)
  const qn = Math.hypot(...q)
  const out: Neighbour[] = []
  for (let i = 0; i < m; i++) {
    if (skip.has(i)) continue
    let s = 0
    let z = 0
    for (let c = 0; c < n; c++) {
      s += data[i * n + c] * q[c]
      z += data[i * n + c] ** 2
    }
    out.push({ index: i, cosine: z > 0 && qn > 0 ? s / Math.sqrt(z) / qn : 0 })
  }
  return out.sort((a, b) => b.cosine - a.cosine || a.index - b.index).slice(0, count)
}

/** An answer to an analogy: the word, its row and its cosine similarity to the offset vector. */
export interface AnalogyAnswer extends Neighbour {
  readonly word: string
}

/**
 * The answers to the analogy a − b + c ("king − man + woman"): the words whose unit vectors are nearest by cosine to
 * â − b̂ + ĉ, the three inputs excluded, best first (3CosAdd). Throws when an input is not a row of the vocabulary.
 */
export function analogy(
  vectors: MatrixLike,
  vocabulary: Vocabulary,
  a: string,
  b: string,
  c: string,
  options: { count?: number } = {},
): AnalogyAnswer[] {
  const { data, m, n } = dense.toMatrixF64(vectors, 'analogy')
  if (vocabulary.tokens.length !== m)
    throw new DomainError('analogy', `analogy: ${vocabulary.tokens.length} words for ${m} rows`)
  const ids = [a, b, c].map((w) => {
    const id = tokenId(vocabulary, w)
    if (id < 0) throw new DomainError('analogy', `analogy: '${w}' is not in the vocabulary`)
    return id
  })
  const u = unitRows(data, m, n)
  const target = new Float64Array(n)
  for (let j = 0; j < n; j++) target[j] = u[ids[0] * n + j] - u[ids[1] * n + j] + u[ids[2] * n + j]
  return nearestByCosine(fromData(u, [m, n]), target, { count: options.count ?? 5, exclude: ids }).map((x) => ({
    ...x,
    word: vocabulary.tokens[x.index],
  }))
}

/**
 * A map of the rows' cosine geometry (float64 [m, d], default d = 2): the rows scaled to unit length, centred, and
 * projected on their top d principal axes. Rows that point the same way land together whatever their lengths, so the
 * picture shows what cosine nearest neighbours see. A zero row maps to minus the mean.
 */
export function cosineMap(vectors: MatrixLike, dimensions = 2): Tensor {
  const { data, m, n } = dense.toMatrixF64(vectors, 'cosineMap')
  if (!(Number.isInteger(dimensions) && dimensions >= 1 && dimensions <= Math.min(m, n)))
    throw new DomainError('cosineMap', `cosineMap: dimensions must be an integer in [1, ${Math.min(m, n)}]`)
  const u = unitRows(data, m, n)
  for (let j = 0; j < n; j++) {
    let mean = 0
    for (let i = 0; i < m; i++) mean += u[i * n + j]
    mean /= m
    for (let i = 0; i < m; i++) u[i * n + j] -= mean
  }
  return lsa(fromData(u, [m, n]), dimensions).rows
}
