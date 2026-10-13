/**
 * Spatial cross-validation: splitters for rows that carry coordinates, where nearby rows are alike and a random split
 * tests a model on points it has all but seen.
 *
 * - Blocks (Roberts et al., 2017, "Cross-validation strategies for data with temporal, spatial, hierarchical, or
 *   phylogenetic structure", Ecography 40): `spatialBlocks` cuts the coordinates into a grid of cells and labels each
 *   row with its cell, for `groupKFold`.
 * - Buffers: `buffered` wraps any splitter and drops from each training set the rows within a radius $r$ of a test row;
 *   around `leaveOneOut` it is buffered leave-one-out.
 * - Matching the prediction task (Milà et al., 2022, "Nearest neighbour distance matching leave-one-out
 *   cross-validation for map validation", Methods in Ecology and Evolution 13; Linnenbrink et al., 2024, "kNNDM
 *   CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation", Geoscientific Model
 *   Development 17): `nndmLeaveOneOut` and `knndm` choose the training sets so that the distribution of the distances
 *   from each test row to its nearest training row, $G^*_j(r)$, matches that of the distances from the prediction
 *   points to their nearest sample, $G_{ij}(r)$. Both follow the R package CAST (`nndm`, `knndm`) step for step, in
 *   geographical space with Euclidean distances.
 *
 * Coordinates are an $n \times d$ matrix, one row per data row (and one per prediction point), and distances are
 * Euclidean. The splitters are not random once built: `knndm` draws from its stream only when it falls back to a
 * random assignment. A splitter given data with a different number of rows than its coordinates throws `ShapeError`.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { eigh } from 'aifn-compute/numerics/linalg'
import { ksTest } from 'aifn-compute/probability/tests'
import { wasserstein1d } from 'aifn-compute/transport'
import { fromTestFolds, inputOf, sorted, type Split, type Splitter } from './splitters'

/** Points as a row-major working copy: $n$ points of $d$ coordinates. */
interface Points {
  /** The coordinates, point $i$ at entries `i * d` to `i * d + d - 1`. */
  data: Float64Array
  /** The number of points. */
  n: number
  /** The number of coordinates of each point. */
  d: number
}

/** Nearest-neighbour distances that `nndmLeaveOneOut` and `knndm` match. */
export interface NearestDistances {
  /** $G_{ij}$: from each prediction point to its nearest data row. */
  prediction: Tensor
  /** $G_j$: from each data row to its nearest other data row (as random leave-one-out sees them). */
  sample: Tensor
  /** $G^*_j$: from each data row, when tested, to its nearest training row under the chosen splits. */
  cv: Tensor
}

/** The splitter of `nndmLeaveOneOut`, with what it matched. */
export interface NndmSplitter extends Splitter {
  /** The distance $\phi$ up to which the distributions were matched. */
  readonly phi: number
  /** For each row, the rows left out of its training set (besides itself), sorted. */
  readonly excluded: number[][]
  /** The nearest-neighbour distances: prediction, sample, and under the splits. */
  readonly nearest: NearestDistances
}

/** The splitter of `knndm`, with the folds it chose and how well they match. */
export interface KnndmSplitter extends Splitter {
  /** The test fold of each row, $0, \dots, k - 1$ (int32). */
  readonly folds: Tensor
  /** The Wasserstein distance $W_1$ between $G^*_j$ under the folds and $G_{ij}$. */
  readonly W: number
  /** The number of clusters $q$ whose merge gave the folds, or `'random'` for the random fallback. */
  readonly clusters: number | 'random'
  /** The $p$-value of the one-sided Kolmogorov–Smirnov test that $G_j$ lies below $G_{ij}$. */
  readonly ksPValue: number
  /** The nearest-neighbour distances: prediction, sample, and under the folds. */
  readonly nearest: NearestDistances
}

/**
 * Coordinates as a row-major working copy.
 *
 * @param coords The $n \times d$ coordinates.
 * @param where The caller's name, for error messages.
 * @returns The points.
 */
function pointsOf(coords: MatrixLike, where: string): Points {
  const { data, m, n } = dense.toMatrixF64(coords, where)
  if (m < 1) throw new DomainError(where, `${where}: no points`)
  return { data: Float64Array.from(data), n: m, d: n }
}

/**
 * The Euclidean distance between point $i$ of `a` and point $j$ of `b`.
 *
 * @param a The first point set.
 * @param i The index of a point of `a`.
 * @param b The second point set, with the same $d$.
 * @param j The index of a point of `b`.
 * @returns $\lVert \avec_i - \bvec_j \rVert$.
 */
function distance(a: Points, i: number, b: Points, j: number): number {
  let s = 0
  for (let k = 0; k < a.d; k++) {
    const t = a.data[i * a.d + k] - b.data[j * b.d + k]
    s += t * t
  }
  return Math.sqrt(s)
}

/**
 * Throws `ShapeError` unless the data a splitter is given has one row per point.
 *
 * @param n The number of data rows.
 * @param points The coordinates the splitter was built with.
 * @param where The caller's name, for error messages.
 */
function checkRows(n: number, points: Points, where: string) {
  if (n !== points.n) throw new ShapeError(where, `${where}: ${n} rows for ${points.n} coordinates`)
}

/**
 * The pairwise distances of the data rows, $n \times n$ row-major.
 *
 * @param p The points.
 * @returns The distance matrix, zero on the diagonal.
 */
function distanceMatrix(p: Points): Float64Array {
  const D = new Float64Array(p.n * p.n)
  for (let i = 0; i < p.n; i++) for (let j = i + 1; j < p.n; j++) D[i * p.n + j] = D[j * p.n + i] = distance(p, i, p, j)
  return D
}

/**
 * $G_{ij}$: the distance from each prediction point to its nearest data row.
 *
 * @param pred The prediction points.
 * @param t The data rows.
 * @returns One distance per prediction point.
 */
function predictionDistances(pred: Points, t: Points): Float64Array {
  if (pred.d !== t.d) throw new ShapeError('spatial', `spatial: ${pred.d} prediction coordinates for ${t.d}`)
  return Float64Array.from({ length: pred.n }, (_, i) => {
    let best = Infinity
    for (let j = 0; j < t.n; j++) best = Math.min(best, distance(pred, i, t, j))
    return best
  })
}

/**
 * $G^*_j$ under a fold assignment: each row's distance to its nearest row in another fold.
 *
 * @param D The $n \times n$ distance matrix.
 * @param n The number of rows.
 * @param fold The fold of each row.
 * @returns One distance per row (infinite when every row shares its fold).
 */
function foldDistances(D: Float64Array, n: number, fold: ArrayLike<number>): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => {
    let best = Infinity
    for (let j = 0; j < n; j++) if (fold[j] !== fold[i]) best = Math.min(best, D[i * n + j])
    return best
  })
}

/**
 * Group labels of spatial blocks (Roberts et al., 2017): the coordinates are cut into a grid of cells, of side `size`
 * from `origin` (default the smallest coordinate on each axis), or `blocks` cells per axis spanning the coordinates'
 * range, and each row is labelled with its cell. The label of the cell with indices $c_0, \dots, c_{d-1}$ is
 * $c_0 + m_0 (c_1 + m_1 (c_2 + \dots))$, $m_j$ the number of cells on axis $j$, so labels are stable across calls.
 * Pass them as `groups` to `groupKFold` for block cross-validation. Throws `DomainError` unless exactly one of `size`
 * and `blocks` is given, for a non-positive size or block count, and for a point below `origin`.
 *
 * @param options The coordinates and the grid.
 * @param options.coords The $n \times d$ coordinates of the rows.
 * @param options.size The side of a cell: one length for every axis, or one per axis.
 * @param options.blocks The number of cells on each axis (one count, or one per axis), spanning the coordinates.
 * @param options.origin The lower corner of the grid, one value per axis. Left out, the smallest coordinates.
 * @returns The block label of each row, $n$ whole numbers.
 *
 * @example Six points in a 2 by 2 grid of blocks, then grouped folds
 * const coords = [[0, 0], [0.2, 0.9], [0.9, 0.1], [1, 1], [0.1, 0.2], [0.8, 0.8]]
 * const groups = spatialBlocks({ coords, blocks: 2 })
 * print('blocks:', groups)
 * print('test rows:', groupKFold({ k: 2 }).split({ n: 6, groups }).map((s) => s.test))
 *
 * @example Cells of a fixed side from a chosen origin
 * print(spatialBlocks({ coords: [[0.5, 0.5], [1.5, 0.5], [2.5, 1.5]], size: 1, origin: [0, 0] }))
 */
export function spatialBlocks({
  coords,
  size,
  blocks,
  origin,
}: {
  coords: MatrixLike
  size?: number | readonly number[]
  blocks?: number | readonly number[]
  origin?: readonly number[]
}): number[] {
  const where = 'spatialBlocks'
  const p = pointsOf(coords, where)
  if ((size === undefined) === (blocks === undefined))
    throw new DomainError(where, `${where}: give one of size and blocks`)
  const axis = (v: number | readonly number[], j: number) => {
    const value = typeof v === 'number' ? v : v[j]
    if (typeof v !== 'number' && v.length !== p.d)
      throw new ShapeError(where, `${where}: ${v.length} values for ${p.d} axes`)
    return value
  }
  const cell = Array.from({ length: p.n }, () => 0)
  let stride = 1
  for (let j = 0; j < p.d; j++) {
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < p.n; i++) {
      lo = Math.min(lo, p.data[i * p.d + j])
      hi = Math.max(hi, p.data[i * p.d + j])
    }
    if (origin !== undefined) {
      if (origin.length !== p.d) throw new ShapeError(where, `${where}: an origin of ${origin.length} for ${p.d} axes`)
      if (lo < origin[j]) throw new DomainError(where, `${where}: a point lies below the origin on axis ${j}`)
      lo = origin[j]
    }
    let width: number
    let count: number
    if (blocks !== undefined) {
      count = axis(blocks, j)
      if (!(Number.isInteger(count) && count >= 1))
        throw new DomainError(where, `${where}: blocks must be whole numbers ≥ 1`)
      width = (hi - lo) / count
    } else {
      width = axis(size!, j)
      if (!(width > 0)) throw new DomainError(where, `${where}: size must be positive`)
      count = Math.floor((hi - lo) / width) + 1
    }
    for (let i = 0; i < p.n; i++) {
      const c = width > 0 ? Math.min(Math.floor((p.data[i * p.d + j] - lo) / width), count - 1) : 0
      cell[i] += c * stride
    }
    stride *= count
  }
  return cell
}

/**
 * A splitter with a buffer: the splits of `splitter`, each with the training rows within distance $r$ of any of its
 * test rows ($\lVert \xvec_i - \xvec_j \rVert \le r$) removed, so that no model is tested next to the points it was
 * trained on. Around `leaveOneOut` it is buffered leave-one-out (spatial leave-one-out with a buffer, as in Roberts
 * et al., 2017); around `groupKFold` on `spatialBlocks`, buffered block cross-validation. `split` throws
 * `ShapeError` when the data's rows do not match the coordinates, and `DomainError` when a buffer leaves a split no
 * training rows.
 *
 * @param splitter The splitter whose splits are buffered; random when it is.
 * @param coords The $n \times d$ coordinates of the rows.
 * @param options The buffer.
 * @param options.radius The buffer radius $r \ge 0$, in the units of the coordinates.
 * @returns A splitter of the same splits, minus the buffered training rows.
 *
 * @example Buffered leave-one-out on a line of five points
 * const coords = [[0], [1], [2], [3], [4]]
 * const splits = buffered(leaveOneOut(), coords, { radius: 1 }).split({ n: 5 })
 * print('test: ', splits.map((s) => s.test))
 * print('train:', splits.map((s) => s.train))
 */
export function buffered(splitter: Splitter, coords: MatrixLike, { radius }: { radius: number }): Splitter {
  const where = 'buffered'
  const p = pointsOf(coords, where)
  if (!(radius >= 0)) throw new DomainError(where, `${where}: the radius must be non-negative`)
  return {
    name: `buffered(${splitter.name}, ${radius})`,
    randomised: splitter.randomised,
    split(data, s) {
      checkRows(inputOf(data).n, p, where)
      return splitter.split(data, s).map(({ train, test }): Split => {
        const tested = Array.from(test.data)
        const kept = Array.from(train.data).filter((i) => tested.every((j) => distance(p, i, p, j) > radius))
        if (kept.length === 0) throw new DomainError(where, `${where}: a radius of ${radius} leaves no training rows`)
        return { train: sorted(kept), test }
      })
    },
  }
}

/**
 * Nearest neighbour distance matching leave-one-out (NNDM LOO; Milà et al., 2022, Algorithm 1, as `CAST::nndm`): a
 * leave-one-out splitter whose training sets leave out the neighbours of each test row until the distribution
 * $G^*_j(r)$ of test-to-nearest-training distances matches the distribution $G_{ij}(r)$ of prediction-to-nearest-sample
 * distances, for $r \le \phi$. Starting from random leave-one-out ($G^*_j = G_j$), it walks up the distances: at the
 * smallest $r$ (with test row $j$), when $(\#\{G^*_j \le r\} - 1)/n \ge \hat G_{ij}(r)$ and row $j$ still trains on
 * more than `minTrain` of the rows, it removes $j$'s nearest neighbours at distance $r$ from $j$'s training set and
 * updates $G^*_j$; otherwise it moves to the next distance. Not random. Throws `ShapeError` for prediction points
 * of another dimension, and `split` when the data's rows do not match the coordinates.
 *
 * @param coords The $n \times d$ coordinates of the data rows.
 * @param predictionCoords The coordinates of points where the model will predict, $m \times d$: a sample of the
 *   prediction area.
 * @param options The matching range and the least training share.
 * @param options.phi The distance $\phi$ up to which to match (CAST's `phi`). Left out, the largest distance between
 *   any two points, plus $10^{-9}$.
 * @param options.minTrain The share of the rows each training set must keep (CAST's `min_train`), in $[0, 1]$.
 * @returns A splitter of $n$ splits, with $\phi$, the rows each split leaves out and the three distance samples.
 *
 * @example Clustered samples and a spread-out prediction area: neighbours are left out
 * const coords = [[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5], [5, 5.1]]
 * const area = [[0, 0], [2.5, 2.5], [5, 5], [0, 5], [5, 0], [1, 4]]
 * const nndm = nndmLeaveOneOut(coords, area)
 * print('left out of each training set:', nndm.excluded)
 * print('G_ij:', nndm.nearest.prediction)
 * print('G*_j:', nndm.nearest.cv)
 */
export function nndmLeaveOneOut(
  coords: MatrixLike,
  predictionCoords: MatrixLike,
  { phi, minTrain = 0.5 }: { phi?: number; minTrain?: number } = {},
): NndmSplitter {
  const where = 'nndmLeaveOneOut'
  const t = pointsOf(coords, where)
  const n = t.n
  if (n < 2) throw new DomainError(where, `${where}: needs at least two rows`)
  if (!(minTrain >= 0 && minTrain <= 1)) throw new DomainError(where, `${where}: minTrain must be in [0, 1]`)
  if (phi !== undefined && !(phi >= 0)) throw new DomainError(where, `${where}: phi must be non-negative`)
  const gij = predictionDistances(pointsOf(predictionCoords, where), t)
  const D = distanceMatrix(t)
  for (let i = 0; i < n; i++) D[i * n + i] = NaN
  const rowMin = (i: number) => {
    let best = Infinity
    for (let j = 0; j < n; j++) if (D[i * n + j] < best) best = D[i * n + j]
    return best
  }
  const gj = Float64Array.from({ length: n }, (_, i) => rowMin(i))
  let largest = 0
  for (const v of gij) largest = Math.max(largest, v)
  for (const v of D) if (v > largest) largest = v
  const range = phi ?? largest + 1e-9
  const sortedGij = Float64Array.from(gij).sort()
  /** The empirical cdf of $G_{ij}$ at $r$. */
  const cdfGij = (r: number) => {
    let lo = 0
    let hi = sortedGij.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (sortedGij[mid] <= r) lo = mid + 1
      else hi = mid
    }
    return lo / sortedGij.length
  }
  const gStar = Float64Array.from(gj)
  const first = (r: number) => gStar.findIndex((v) => v === r)
  const atDistance = (j: number, r: number) => {
    const out: number[] = []
    for (let k = 0; k < n; k++) if (D[j * n + k] === r) out.push(k)
    return out
  }
  const nextAbove = (r: number, inclusive: boolean) => {
    let best = Infinity
    for (const v of gStar) if ((inclusive ? v >= r : v > r) && v < best) best = v
    return best
  }
  let rmin = Math.min(...gStar)
  let jmin = first(rmin)
  let kmin = atDistance(jmin, rmin)
  while (rmin <= range) {
    let atOrBelow = 0
    let above = 0
    for (const v of gStar) {
      if (v <= rmin) atOrBelow++
      else above++
    }
    let training = 0
    for (let k = 0; k < n; k++) if (!Number.isNaN(D[jmin * n + k])) training++
    if ((atOrBelow - 1) / n >= cdfGij(rmin) && training / n > minTrain) {
      for (const k of kmin) D[jmin * n + k] = NaN
      gStar[jmin] = rowMin(jmin)
      rmin = nextAbove(rmin, true)
    } else if (above === 0) {
      break
    } else {
      rmin = nextAbove(rmin, false)
    }
    if (!Number.isFinite(rmin)) break
    jmin = first(rmin)
    kmin = atDistance(jmin, rmin)
  }
  const excluded: number[][] = []
  const splits: Split[] = []
  for (let i = 0; i < n; i++) {
    const train: number[] = []
    const out: number[] = []
    for (let k = 0; k < n; k++) (Number.isNaN(D[i * n + k]) ? out : train).push(k)
    excluded.push(out.filter((k) => k !== i))
    splits.push({ train: sorted(train), test: sorted([i]) })
  }
  return {
    name: `nndm-loo(phi ${range.toPrecision(4)})`,
    randomised: false,
    phi: range,
    excluded,
    nearest: { prediction: fromData(gij, [gij.length]), sample: fromData(gj, [n]), cv: fromData(gStar, [n]) },
    split(data) {
      checkRows(inputOf(data).n, t, where)
      return splits.map((s) => ({ ...s }))
    },
  }
}

/**
 * Ward's agglomerative clustering (`ward.D2` of R's `hclust`, `ward` of scipy) by the nearest-neighbour chain, on
 * squared Euclidean distances updated by the Lance–Williams formula
 * $d^2(k, i \cup j) = ((n_i + n_k) d^2(k, i) + (n_j + n_k) d^2(k, j) - n_k d^2(i, j)) / (n_i + n_j + n_k)$.
 *
 * @param p The points.
 * @returns The $n - 1$ merges as pairs of member rows, sorted by height (the square root of the merge cost).
 */
function wardMerges(p: Points): { a: number; b: number; height: number }[] {
  const n = p.n
  const D = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) D[i * n + j] = D[j * n + i] = distance(p, i, p, j) ** 2
  const size = new Float64Array(n).fill(1)
  const active = new Uint8Array(n).fill(1)
  const merges: { a: number; b: number; height: number }[] = []
  const chain: number[] = []
  let left = n
  while (left > 1) {
    if (chain.length === 0) chain.push(active.indexOf(1))
    const a = chain[chain.length - 1]
    const previous = chain.length > 1 ? chain[chain.length - 2] : -1
    let b = previous
    let best = previous >= 0 ? D[a * n + previous] : Infinity
    for (let k = 0; k < n; k++) {
      if (!active[k] || k === a) continue
      if (D[a * n + k] < best) {
        best = D[a * n + k]
        b = k
      }
    }
    if (b !== previous) {
      chain.push(b)
      continue
    }
    chain.length -= 2
    const [keep, drop] = a < b ? [a, b] : [b, a]
    merges.push({ a: keep, b: drop, height: Math.sqrt(best) })
    const total = size[keep] + size[drop]
    for (let k = 0; k < n; k++) {
      if (!active[k] || k === keep || k === drop) continue
      const v =
        ((size[keep] + size[k]) * D[keep * n + k] + (size[drop] + size[k]) * D[drop * n + k] - size[k] * best) /
        (total + size[k])
      D[keep * n + k] = D[k * n + keep] = v
    }
    size[keep] = total
    active[drop] = 0
    left--
  }
  return merges.sort((x, y) => x.height - y.height)
}

/**
 * Cut a dendrogram into $q$ clusters (R's `cutree`): apply the $n - q$ lowest merges and number the clusters in the
 * order of their first row.
 *
 * @param merges The merges, sorted by height.
 * @param n The number of rows.
 * @param q The number of clusters.
 * @returns The cluster of each row, $0, \dots, q - 1$.
 */
function cutTree(merges: readonly { a: number; b: number }[], n: number, q: number): Int32Array {
  const parent = Int32Array.from({ length: n }, (_, i) => i)
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]]
    return i
  }
  for (let m = 0; m < n - q; m++) parent[find(merges[m].a)] = find(merges[m].b)
  const id = new Map<number, number>()
  return Int32Array.from({ length: n }, (_, i) => {
    const root = find(i)
    if (!id.has(root)) id.set(root, id.size)
    return id.get(root)!
  })
}

/**
 * The first principal axis of the points, as `prcomp(rank = 1)`, with its sign fixed so that its largest entry (in
 * absolute value, the first of ties) is positive.
 *
 * @param p The points.
 * @returns The centre and the unit axis.
 */
function principalAxis(p: Points): { centre: Float64Array; axis: Float64Array } {
  const { n, d } = p
  const centre = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) centre[j] += p.data[i * d + j] / n
  const C = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++)
      for (let b = 0; b < d; b++) C[a * d + b] += (p.data[i * d + a] - centre[a]) * (p.data[i * d + b] - centre[b])
  const V = dense.data(eigh(fromData(C, [d, d])).vectors)
  const axis = Float64Array.from({ length: d }, (_, j) => V[j * d])
  let top = 0
  for (let j = 1; j < d; j++) if (Math.abs(axis[j]) > Math.abs(axis[top])) top = j
  if (axis[top] < 0) for (let j = 0; j < d; j++) axis[j] = -axis[j]
  return { centre, axis }
}

/**
 * $k$-fold nearest neighbour distance matching (kNNDM; Linnenbrink et al., 2024, as `CAST::knndm` with hierarchical
 * clustering): $k$ folds chosen so that the test-to-nearest-training distances $G^*_j$ match the
 * prediction-to-nearest-sample distances $G_{ij}$. When a one-sided Kolmogorov–Smirnov test does not find $G_j$
 * below $G_{ij}$ ($p \ge 0.05$: the samples are as spread as the prediction points), random $k$-fold is returned.
 * Otherwise the rows are clustered by Ward's method, and for each number of clusters $q$ on a grid from $k$ to
 * $n - 2$ (`gridSize` values, log-spaced, rounded and deduplicated) the clusters are ordered along the first
 * principal axis of the coordinates; clusters of at least $n/k$ rows become folds of their own and the others are
 * dealt round-robin to the remaining folds. Among the assignments whose largest fold holds at most `maxp` of the rows,
 * the one with the smallest Wasserstein distance $W_1(G^*_j, G_{ij})$ is chosen. The sign of the principal axis is
 * fixed (its largest entry positive) where R's is the LAPACK's; the test uses the exact law where R's `ks.test`
 * switches to the asymptotic one for large samples. Throws `DomainError` unless $1/k < $ `maxp` $< 1$ and
 * $2 \le k \le n - 2$, when no assignment satisfies `maxp`, and when the random fallback has no stream.
 *
 * @param coords The $n \times d$ coordinates of the data rows.
 * @param predictionCoords The coordinates of points where the model will predict, $m \times d$.
 * @param options The number of folds, the largest fold share, the grid size and the stream of the random fallback.
 * @param options.k The number of folds $k$.
 * @param options.maxp The largest share of the rows one fold may hold, in $(1/k, 1)$ (CAST's `maxp`).
 * @param options.gridSize The number of cluster counts tried (CAST's `nk_len`), before deduplication.
 * @param options.stream Randomness for the random fallback, which draws from `child(stream, 'knndm')`.
 * @returns A splitter of $k$ folds, with the fold of each row, $W_1$, the chosen number of clusters, the test's
 *   $p$-value and the three distance samples.
 *
 * @example Samples in two clumps, predictions across the square between them
 * const coords = [
 *   [0, 0], [0.2, 0.1], [0.1, 0.3], [0.3, 0.2], [0.2, 0.4], [0.4, 0.1],
 *   [3, 3], [3.2, 3.1], [3.1, 3.3], [3.3, 3.2], [3.2, 3.4], [3.4, 3.1],
 * ]
 * const area = Array.from({ length: 25 }, (_, i) => [(i % 5) * 0.9, Math.floor(i / 5) * 0.9])
 * const folds = knndm(coords, area, { k: 2, maxp: 0.6 })
 * print('folds:', folds.folds)
 * print('clusters, W:', folds.clusters, folds.W)
 * print('KS p-value:', folds.ksPValue)
 */
export function knndm(
  coords: MatrixLike,
  predictionCoords: MatrixLike,
  {
    k = 10,
    maxp = 0.5,
    gridSize = 100,
    stream,
  }: { k?: number; maxp?: number; gridSize?: number; stream?: Stream } = {},
): KnndmSplitter {
  const where = 'knndm'
  const t = pointsOf(coords, where)
  const n = t.n
  if (!(Number.isInteger(k) && k >= 2 && k <= n - 2))
    throw new DomainError(where, `${where}: k must be a whole number from 2 to n - 2`)
  if (!(maxp > 1 / k && maxp < 1)) throw new DomainError(where, `${where}: maxp must lie strictly between 1/k and 1`)
  const gij = predictionDistances(pointsOf(predictionCoords, where), t)
  const D = distanceMatrix(t)
  const gj = foldDistances(
    D,
    n,
    Int32Array.from({ length: n }, (_, i) => i),
  )
  const ksPValue = ksTest(gj, gij, { alternative: 'greater' }).pValue
  let fold: Int32Array
  let W: number
  let clusters: number | 'random'
  if (ksPValue >= 0.05) {
    if (!stream) throw new DomainError(where, `${where}: the random fallback needs a stream`)
    const deck = Array.from({ length: k * Math.ceil(n / k) }, (_, i) => i % k)
    const order = permutation(child(stream, 'knndm'), deck.length).data
    fold = Int32Array.from({ length: n }, (_, i) => deck[order[i]])
    W = wasserstein1d(foldDistances(D, n, fold), gij) as number
    clusters = 'random'
  } else {
    const merges = wardMerges(t)
    const { centre, axis } = principalAxis(t)
    const grid: number[] = []
    for (let g = 0; g < gridSize; g++) {
      const q = Math.round(
        Math.exp(Math.log(k) + (gridSize > 1 ? (g * (Math.log(n - 2) - Math.log(k))) / (gridSize - 1) : 0)),
      )
      if (!grid.includes(q)) grid.push(q)
    }
    let best: { fold: Int32Array; W: number; q: number } | undefined
    for (const q of grid) {
      const cluster = cutTree(merges, n, q)
      const count = new Float64Array(q)
      const sum = new Float64Array(q * t.d)
      for (let i = 0; i < n; i++) {
        count[cluster[i]]++
        for (let j = 0; j < t.d; j++) sum[cluster[i] * t.d + j] += t.data[i * t.d + j]
      }
      const projection = Float64Array.from({ length: q }, (_, c) => {
        let s = 0
        for (let j = 0; j < t.d; j++) s += (sum[c * t.d + j] / count[c] - centre[j]) * axis[j]
        return s
      })
      const order = Array.from({ length: q }, (_, c) => c).sort((a, b) => projection[a] - projection[b])
      // Large clusters keep a fold each; the others are dealt round-robin to the remaining folds.
      const label = new Int32Array(q).fill(-1)
      let next = 0
      for (const c of order) if (count[c] >= n / k) label[c] = next++
      const free = Array.from({ length: k - next }, (_, f) => next + f)
      let dealt = 0
      for (const c of order) if (label[c] < 0) label[c] = free[dealt++ % free.length]
      const candidate = Int32Array.from(cluster, (c) => label[c])
      const sizes = new Float64Array(k)
      for (const f of candidate) sizes[f]++
      if (Math.max(...sizes) / n > maxp) continue
      const w = wasserstein1d(foldDistances(D, n, candidate), gij) as number
      if (!best || w < best.W) best = { fold: candidate, W: w, q }
    }
    if (!best) throw new DomainError(where, `${where}: no assignment keeps every fold within maxp = ${maxp}`)
    ;({ fold, W, q: clusters } = best)
  }
  const gStar = foldDistances(D, n, fold)
  const splits = fromTestFolds(fold, k).filter((s) => s.test.shape[0] > 0)
  return {
    name: `knndm(${k})`,
    randomised: false,
    folds: fromData(fold, [n]),
    W,
    clusters,
    ksPValue,
    nearest: { prediction: fromData(gij, [gij.length]), sample: fromData(gj, [n]), cv: fromData(gStar, [n]) },
    split(data) {
      checkRows(inputOf(data).n, t, where)
      return splits.map((s) => ({ ...s }))
    },
  }
}
