/**
 * Vector-quantisation codebooks, the coarse and fine quantisers of inverted files and product quantisation: k-means++
 * seeding (Arthur and Vassilvitskii 2007, "k-means++: the advantages of careful seeding", SODA), the assignment of rows
 * to their nearest codeword, the centroid update of Lloyd's algorithm (Lloyd 1982, "Least squares quantization in
 * PCM", IEEE Trans. Inf. Theory 28(2)) and a codebook trained by running them. `aifn-methods`'s k-means model steps
 * through the same assignment and update.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { squaredRowDistance } from 'aifn-compute/numerics/linalg'
import { rowsOf } from './search'

/** The result of k-means++ seeding. */
export interface KMeansPlusPlus {
  /** The chosen rows, in the order picked. */
  indices: Tensor
  /** Their coordinates [k, d]. */
  centroids: Tensor
  /** Pick j's sampling probabilities over the rows [k, n]: uniform for the first, D(x)² / Σ D² after. */
  probabilities: Tensor
}

/**
 * k-means++ seeding: the first centre is a uniform row; each next centre is row x with probability D(x)² / Σ D², D the
 * distance to the nearest centre so far. With `trials` > 1 each pick draws that many candidates and keeps the one that
 * lowers the potential most (scikit-learn's greedy variant uses 2 + ⌊log k⌋). Pick j draws from `child(s, 'pick', j)`.
 */
export function kmeansPlusPlus(s: Stream, x: MatrixLike, k: Size, params: { trials?: number } = {}): KMeansPlusPlus {
  const { n, d, data: v } = rowsOf(x, 'kmeansPlusPlus')
  if (!(Number.isInteger(k) && k >= 1 && k <= n))
    throw new DomainError('kmeansPlusPlus', `kmeansPlusPlus: k must lie in 1 … ${n}`)
  const trials = params.trials ?? 1
  const indices: number[] = []
  const probs = new Float64Array(k * n)
  const D2 = new Float64Array(n).fill(Infinity)
  const draw = (sub: Stream, weights: Float64Array, total: number): number => {
    let u = uniform(sub) * total
    for (let i = 0; i < n; i++) {
      u -= weights[i]
      if (u < 0) return i
    }
    for (let i = n - 1; i >= 0; i--) if (weights[i] > 0) return i
    return n - 1
  }
  for (let j = 0; j < k; j++) {
    const sub = child(s, 'pick', j)
    let pick: number
    if (j === 0) {
      probs.fill(1 / n, 0, n)
      pick = integers(sub, n)
    } else {
      let total = 0
      for (let i = 0; i < n; i++) total += D2[i]
      for (let i = 0; i < n; i++) probs[j * n + i] = total > 0 ? D2[i] / total : 1 / n
      if (total === 0) pick = integers(sub, n)
      else {
        pick = draw(sub, D2, total)
        let bestPotential = Infinity
        for (let t = 0; t < trials; t++) {
          const c = t === 0 ? pick : draw(sub, D2, total)
          let potential = 0
          for (let i = 0; i < n; i++) potential += Math.min(D2[i], squaredRowDistance(v, i, v, c, d))
          if (potential < bestPotential) {
            bestPotential = potential
            pick = c
          }
        }
      }
    }
    indices.push(pick)
    for (let i = 0; i < n; i++) D2[i] = Math.min(D2[i], squaredRowDistance(v, i, v, pick, d))
  }
  const centroids = new Float64Array(k * d)
  indices.forEach((i, j) => centroids.set(v.subarray(i * d, (i + 1) * d), j * d))
  return {
    indices: fromData(Int32Array.from(indices), [k]),
    centroids: fromData(centroids, [k, d]),
    probabilities: fromData(probs, [k, n]),
  }
}

/** The nearest codeword of every row. */
export interface Assignment {
  /** Codeword index per row [n] (int32), ties to the lower index. */
  labels: Tensor
  /** Rows per codeword [k]. */
  sizes: Tensor
  /** Σᵢ ‖xᵢ − c_labelᵢ‖², the quantisation error (k-means inertia). */
  inertia: number
}

/** Assign every row of x (n × d) to its nearest row of the codebook (k × d) in squared Euclidean distance. */
export function assignNearest(x: MatrixLike, codebook: MatrixLike): Assignment {
  const X = rowsOf(x, 'assignNearest')
  const C = rowsOf(codebook, 'assignNearest')
  if (C.d !== X.d) throw new ShapeError('assignNearest', `assignNearest: codebook has ${C.d} columns, data ${X.d}`)
  const labels = new Int32Array(X.n)
  const sizes = new Float64Array(C.n)
  let inertia = 0
  for (let i = 0; i < X.n; i++) {
    let best = 0
    let bestD = Infinity
    for (let j = 0; j < C.n; j++) {
      const t = squaredRowDistance(X.data, i, C.data, j, X.d)
      if (t < bestD) {
        bestD = t
        best = j
      }
    }
    labels[i] = best
    sizes[best]++
    inertia += bestD
  }
  return { labels: fromData(labels, [X.n]), sizes: fromData(sizes, [C.n]), inertia }
}

/** The update step of Lloyd's algorithm. */
export interface LloydUpdate {
  /** Each codeword moved to the mean of its rows [k, d]; an empty one stays where it was. */
  centroids: Tensor
  /** Codewords that had no rows. */
  empty: number[]
  /** Total squared movement of the codewords. */
  shift: number
}

/** Move every codeword (k × d) to the mean of the rows of x assigned to it by `labels`. */
export function lloydUpdate(x: MatrixLike, labels: Tensor, codebook: MatrixLike): LloydUpdate {
  const X = rowsOf(x, 'lloydUpdate')
  const C = rowsOf(codebook, 'lloydUpdate')
  const lab = dense.data(labels)
  const { n, d } = X
  const k = C.n
  const sums = new Float64Array(k * d)
  const counts = new Float64Array(k)
  for (let i = 0; i < n; i++) {
    counts[lab[i]]++
    for (let j = 0; j < d; j++) sums[lab[i] * d + j] += X.data[i * d + j]
  }
  const c = new Float64Array(k * d)
  const empty: number[] = []
  let shift = 0
  for (let j = 0; j < k; j++) {
    if (counts[j] === 0) empty.push(j)
    for (let t = 0; t < d; t++) {
      c[j * d + t] = counts[j] ? sums[j * d + t] / counts[j] : C.data[j * d + t]
      shift += (c[j * d + t] - C.data[j * d + t]) ** 2
    }
  }
  return { centroids: fromData(c, [k, d]), empty, shift }
}

/** A trained codebook. */
export interface Codebook extends Assignment {
  /** The codewords [k, d]. */
  centroids: Tensor
  /** Lloyd iterations run. */
  iterations: number
}

/** Options of {@link trainCodebook}. */
export interface CodebookOptions {
  stream: Stream
  /** Most Lloyd iterations (default 25, as faiss's `Clustering`). */
  iterations?: number
  /** Start Lloyd's iterations from these codewords (k × d) instead of a k-means++ seeding (a warm start). */
  initial?: MatrixLike
}

/**
 * A k-codeword vector quantiser of the rows of x: k-means++ seeding from `child(stream, 'seed')` (or `initial`), then Lloyd's
 * iterations until no label changes or `iterations` is reached. When k ≥ n every row is its own codeword.
 */
export function trainCodebook(x: MatrixLike, k: Size, options: CodebookOptions): Codebook {
  const X = rowsOf(x, 'trainCodebook')
  const m = Math.min(k, X.n)
  const rows = fromData(X.data, [X.n, X.d])
  let centroids: Tensor
  if (options.initial) {
    const I = rowsOf(options.initial, 'trainCodebook')
    if (I.n !== m || I.d !== X.d)
      throw new ShapeError(
        'trainCodebook',
        `trainCodebook: initial codewords are ${I.n} × ${I.d}, expected ${m} × ${X.d}`,
      )
    centroids = fromData(I.data, [m, X.d])
  } else centroids = kmeansPlusPlus(child(options.stream, 'seed'), rows, m).centroids
  let a = assignNearest(rows, centroids)
  const most = options.iterations ?? 25
  let t = 0
  while (t < most) {
    t++
    centroids = lloydUpdate(rows, a.labels, centroids).centroids
    const next = assignNearest(rows, centroids)
    const before = dense.data(a.labels)
    const after = dense.data(next.labels)
    a = next
    let same = true
    for (let i = 0; i < X.n && same; i++) same = before[i] === after[i]
    if (same) break
  }
  return { centroids, ...a, iterations: t }
}
