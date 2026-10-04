/**
 * Locality-sensitive hashing (Indyk and Motwani 1998, "Approximate nearest neighbors: towards removing the curse of
 * dimensionality", STOC). A family of hash functions is locality sensitive when near points collide more often than
 * far ones. Two families on vectors:
 *
 * - **Random hyperplanes** for the cosine distance (Charikar 2002, "Similarity estimation techniques from rounding
 *   algorithms", STOC): h(x) = [rᵀx ≥ 0] with r ~ N(0, I). Two vectors at angle θ collide with probability 1 − θ/π.
 * - **p-stable projections** for the Euclidean distance (Datar, Immorlica, Indyk and Mirrokni 2004, "Locality-sensitive
 *   hashing scheme based on p-stable distributions", SoCG): h(x) = ⌊(aᵀx + b)/w⌋ with a ~ N(0, I), b ~ U(0, w). Two
 *   points at distance c collide with probability p(c) = 1 − 2Φ(−w/c) − (2c/(√(2π) w))(1 − e^{−w²/(2c²)}).
 *
 * MinHash (`aifn-compute/text/features`) is the family for Jaccard similarity of sets. Every family is amplified the same way:
 * a signature of b·r hashes is cut into b bands (tables) of r rows; a band's r values are one bucket key (AND), and two
 * items are candidates when any band agrees (OR), with probability 1 − (1 − pʳ)ᵇ for a per-hash collision probability
 * p: an S-curve that rises around p = (1/b)^{1/r} (Leskovec, Rajaraman and Ullman, "Mining of Massive Datasets", ch. 3).
 * A query is answered exactly over the union of its buckets.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalCdf } from 'aifn-compute/numerics/special'
import {
  checkK,
  distanceOf,
  kBest,
  queryOf,
  rowsOf,
  stackResults,
  type Neighbours,
  type NeighbourMetric,
  type QueryResult,
} from './search'

// ── Banding: the AND–OR amplification shared by every family ────────────────────────────────────────────────────────

/** Banding of a signature: `bands` b and `rows` r per band, using the first b·r positions. */
export interface Banding {
  bands: number
  rows: number
}

function checkBanding({ bands, rows }: Banding, length: number, op: string): void {
  if (!(Number.isInteger(bands) && bands >= 1 && Number.isInteger(rows) && rows >= 1))
    throw new DomainError(op, `${op}: bands and rows must be positive integers`)
  if (bands * rows > length)
    throw new DomainError(
      op,
      `${op}: ${bands} bands × ${rows} rows need ${bands * rows} hashes, the signature has ${length}`,
    )
}

/** The bucket key of each band of a signature: band index and its r values, as strings (length b). */
export function lshBands(signature: VectorLike, banding: Banding): string[] {
  const s = dense.toF64(signature, 'lshBands')
  checkBanding(banding, s.length, 'lshBands')
  const { bands, rows } = banding
  return Array.from({ length: bands }, (_, b) => `${b}:${Array.from(s.subarray(b * rows, (b + 1) * rows)).join(',')}`)
}

/**
 * The candidate pairs of LSH banding over the rows of a signature matrix (N × k): every pair i < j that shares the
 * bucket of at least one band, with the number of bands they share, in order of (i, j).
 */
export function lshCandidates(signatures: MatrixLike, banding: Banding): { i: number; j: number; bands: number }[] {
  const { data, m, n } = dense.toMatrixF64(signatures, 'lshCandidates')
  checkBanding(banding, n, 'lshCandidates')
  const shared = new Map<number, number>()
  for (let b = 0; b < banding.bands; b++) {
    const buckets = new Map<string, number[]>()
    for (let i = 0; i < m; i++) {
      const at = i * n + b * banding.rows
      const key = Array.from(data.subarray(at, at + banding.rows)).join(',')
      const list = buckets.get(key)
      if (list) list.push(i)
      else buckets.set(key, [i])
    }
    for (const list of buckets.values())
      for (let p = 0; p < list.length; p++)
        for (let q = p + 1; q < list.length; q++) {
          const key = list[p] * m + list[q]
          shared.set(key, (shared.get(key) ?? 0) + 1)
        }
  }
  return [...shared.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([key, bands]) => ({ i: Math.floor(key / m), j: key % m, bands }))
}

/**
 * The probability 1 − (1 − pʳ)ᵇ that two items whose single hashes collide with probability p become a candidate pair
 * under b bands of r rows (for MinHash, p is the Jaccard similarity).
 */
export function lshProbability(similarity: number, banding: Banding): number {
  if (!(similarity >= 0 && similarity <= 1))
    throw new DomainError('lshProbability', 'lshProbability: the similarity must be in [0, 1]')
  checkBanding(banding, Infinity, 'lshProbability')
  return 1 - (1 - similarity ** banding.rows) ** banding.bands
}

/** The collision probability (1/b)^{1/r} near which the S-curve of b bands of r rows is steepest: the threshold. */
export function lshThreshold(banding: Banding): number {
  checkBanding(banding, Infinity, 'lshThreshold')
  return (1 / banding.bands) ** (1 / banding.rows)
}

// ── Families on vectors ──────────────────────────────────────────────────────────────────────────────────────────────

/** A drawn family of b·r hash functions on vectors of width d (b tables of r hashes). */
export interface LshFamily {
  readonly kind: 'lsh-family'
  readonly family: 'hyperplane' | 'p-stable'
  readonly d: Size
  readonly banding: Banding
  /** The projection directions, one row per hash [b·r, d]. */
  readonly projections: Tensor
  /** p-stable: the offsets b ~ U(0, w) [b·r]. */
  readonly offsets?: Tensor
  /** p-stable: the bucket width w. */
  readonly width?: number
}

/** Options of the family constructors: b tables of r hashes each, and the stream they are drawn from. */
export interface LshFamilyOptions {
  tables: Size
  hashesPerTable: Size
  stream: Stream
}

/** Random-hyperplane hashes for the cosine distance (Charikar 2002): b·r directions r ~ N(0, I_d). */
export function hyperplaneFamily(d: Size, options: LshFamilyOptions): LshFamily {
  const banding = { bands: options.tables, rows: options.hashesPerTable }
  checkBanding(banding, Infinity, 'hyperplaneFamily')
  const projections = normal(child(options.stream, 'hyperplanes'), 0, 1, { shape: [banding.bands * banding.rows, d] })
  return { kind: 'lsh-family', family: 'hyperplane', d, banding, projections }
}

/** p-stable (Gaussian) hashes for the Euclidean distance (Datar et al. 2004) with bucket width w. */
export function pStableFamily(d: Size, options: LshFamilyOptions & { width: number }): LshFamily {
  const banding = { bands: options.tables, rows: options.hashesPerTable }
  checkBanding(banding, Infinity, 'pStableFamily')
  if (!(options.width > 0)) throw new DomainError('pStableFamily', 'pStableFamily: the width must be positive')
  const h = banding.bands * banding.rows
  const projections = normal(child(options.stream, 'projections'), 0, 1, { shape: [h, d] })
  const offsets = uniform(child(options.stream, 'offsets'), 0, options.width, { shape: [h] })
  return { kind: 'lsh-family', family: 'p-stable', d, banding, projections, offsets, width: options.width }
}

/** The signature of every row of x (n × d) under a family: [n, b·r] integer hash values (bits for hyperplanes). */
export function lshSignatures(family: LshFamily, x: MatrixLike): Tensor {
  const X = rowsOf(x, 'lshSignatures')
  const P = dense.data(family.projections)
  const off = family.offsets ? dense.data(family.offsets) : undefined
  const h = family.banding.bands * family.banding.rows
  const out = new Float64Array(X.n * h)
  for (let i = 0; i < X.n; i++)
    for (let j = 0; j < h; j++) {
      let s = 0
      for (let c = 0; c < X.d; c++) s += P[j * X.d + c] * X.data[i * X.d + c]
      out[i * h + j] = family.family === 'hyperplane' ? (s >= 0 ? 1 : 0) : Math.floor((s + off![j]) / family.width!)
    }
  return fromData(out, [X.n, h])
}

/** The probability that one hash of the family puts two points at distance c (angle θ for hyperplanes) together. */
export function lshCollisionProbability(family: Pick<LshFamily, 'family' | 'width'>, distance: number): number {
  if (family.family === 'hyperplane') return 1 - Math.min(Math.PI, Math.max(0, distance)) / Math.PI
  const w = family.width!
  if (distance <= 0) return 1
  const r = w / distance
  return 1 - 2 * normalCdf(-r) - (2 / (Math.sqrt(2 * Math.PI) * r)) * (1 - Math.exp(-(r * r) / 2))
}

/** An LSH index: the family, the stored points, and each table's buckets (key → point indices). */
export interface LshIndex {
  readonly kind: 'lsh-index'
  readonly family: LshFamily
  readonly n: Size
  readonly data: Float64Array
  readonly signatures: Tensor
  readonly tables: readonly Readonly<Record<string, readonly number[]>>[]
  readonly metric: NeighbourMetric
}

/** Hash the rows of x into the family's tables. The metric reranks candidates (default: cosine for hyperplanes). */
export function lshIndex(x: MatrixLike, family: LshFamily, options: { metric?: NeighbourMetric } = {}): LshIndex {
  const X = rowsOf(x, 'lshIndex')
  if (X.d !== family.d) throw new DomainError('lshIndex', `lshIndex: the family hashes width ${family.d}, data ${X.d}`)
  const signatures = lshSignatures(family, x)
  const S = dense.data(signatures)
  const { bands, rows } = family.banding
  const h = bands * rows
  const tables = Array.from({ length: bands }, (_, b) => {
    const t: Record<string, number[]> = {}
    for (let i = 0; i < X.n; i++) {
      const key = Array.from(S.subarray(i * h + b * rows, i * h + (b + 1) * rows)).join(',')
      ;(t[key] ??= []).push(i)
    }
    return t
  })
  const metric = options.metric ?? (family.family === 'hyperplane' ? 'cosine' : 'euclidean')
  return { kind: 'lsh-index', family, n: X.n, data: Float64Array.from(X.data), signatures, tables, metric }
}

/** The answer to one LSH query, with the query's bucket in each table and the candidates reranked. */
export interface LshQueryResult extends QueryResult {
  /** The query's bucket key per table. */
  readonly buckets: readonly string[]
  /** The distinct points that shared a bucket with the query. */
  readonly candidates: readonly number[]
}

/** The k nearest candidates (by exact distance) among the points that share a bucket with the query in any table. */
export function lshQuery(index: LshIndex, query: VectorLike, k: Size): LshQueryResult {
  const { family } = index
  const q = queryOf(query, family.d, 'lshQuery')
  checkK(k, index.n, 'lshQuery')
  const sig = dense.data(lshSignatures(family, fromData(q, [1, family.d])))
  const { rows } = family.banding
  const buckets = index.tables.map((_, b) => Array.from(sig.subarray(b * rows, (b + 1) * rows)).join(','))
  const seen = new Set<number>()
  buckets.forEach((key, b) => index.tables[b][key]?.forEach((i) => seen.add(i)))
  const candidates = [...seen].sort((a, b) => a - b)
  const best = kBest(k)
  for (const j of candidates) best.offer(j, distanceOf(index.metric, q, 0, index.data, j, family.d))
  return {
    indices: best.indices,
    distances: best.distances,
    distanceEvaluations: candidates.length,
    buckets,
    candidates,
  }
}

/** {@link lshQuery} for every row of `queries`; a query with fewer than k candidates is padded with −1 and Infinity. */
export function lshSearch(index: LshIndex, queries: MatrixLike, k: Size): Neighbours {
  const Q = rowsOf(queries, 'lshSearch')
  return stackResults(
    Array.from({ length: Q.n }, (_, i) => lshQuery(index, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k)),
    k,
  )
}
