/**
 * Distance-based anomaly scores from the $k$ nearest neighbours (found exactly, by brute force, by
 * `aifn-compute/numerics/neighbours`): the distance to the $k$-th neighbour (Ramaswamy, Rastogi and Shim, 2000) or the
 * mean distance to the $k$ nearest (Angiulli and Pizzuti, 2002), and the local outlier factor (Breunig et al., 2000),
 * which compares a point's local density with its neighbours'. Distances are Euclidean. A training point scored
 * against the training set is left out of its own neighbours, so $k$ may be at most $n - 1$ there (and $n$ for new
 * points); a larger $k$ throws `DomainError`.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'

/**
 * A float64 copy of a matrix argument as a tensor.
 *
 * @param x The points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param where The caller's name for error messages.
 * @returns The $n \times d$ tensor.
 */
const rowsOf = (x: MatrixLike, where: string) => {
  const m = dense.toMatrixF64(x, where)
  return fromData(Float64Array.from(m.data), [m.m, m.n])
}

/**
 * Neighbours of `queries` among `train`, leaving a training point out of its own list when the queries are the training
 * set.
 *
 * @param train The points searched, $n \times d$.
 * @param queries The points whose neighbours are found, $m \times d$; left out, the training points themselves.
 * @param k The number of neighbours per query.
 * @param where The caller's name for error messages.
 * @returns `ids` and `dist`, the neighbours' row indices and distances, nearest first, as flat row-major $m \times k$
 *   arrays; and `m`, the number of queries.
 */
function neighboursOf(train: MatrixLike, queries: MatrixLike | undefined, k: Size, where: string) {
  const X = rowsOf(train, where)
  const Q = queries === undefined ? X : rowsOf(queries, where)
  const nn = bruteForceNeighbours(X, Q, k, { excludeSelf: queries === undefined })
  return { ids: toFlat(nn.indices), dist: toFlat(nn.distances), m: Q.shape[0] }
}

/** Options of `knnScore`. */
export type KnnScoreOptions = {
  /** Neighbours $k$ (default 5). */
  k?: Size
  /** `kth`: the distance to the $k$-th neighbour; `mean`: the mean distance to the $k$ nearest (default `kth`). */
  aggregate?: 'kth' | 'mean'
}

/**
 * The $k$-nearest-neighbour anomaly score of each query among the training points (of each training point among the
 * others when `queries` is omitted): far from its neighbours means anomalous.
 *
 * @param train The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param queries The points to score, $m \times d$; left out, the training points are scored, each without itself.
 * @param options The number of neighbours and how their distances are aggregated.
 * @returns The score of each query ($m$ values, or $n$ without queries).
 *
 * @example The far point has the largest distance to its second neighbour
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [6, 6]]
 * print('k-th', knnScore(x, undefined, { k: 2 }))
 * print('mean', knnScore(x, undefined, { k: 2, aggregate: 'mean' }))
 * print('new points', knnScore(x, [[0.5, 0.5], [3, 3]], { k: 2 }))
 */
export function knnScore(train: MatrixLike, queries?: MatrixLike, options: KnnScoreOptions = {}): Float64Array {
  const { k = 5, aggregate = 'kth' } = options
  const nn = neighboursOf(train, queries, k, 'knnScore')
  const out = new Float64Array(nn.m)
  for (let q = 0; q < nn.m; q++) {
    if (aggregate === 'kth') out[q] = nn.dist[q * k + k - 1]
    else {
      let s = 0
      for (let j = 0; j < k; j++) s += nn.dist[q * k + j]
      out[q] = s / k
    }
  }
  return out
}

/** A fitted local outlier factor model: the training points, their k-distances and local reachability densities. */
export type LocalOutlierFactor = {
  /** The training points, as given (not copied). */
  train: MatrixLike
  /** The number of neighbours $k$. */
  k: Size
  /** The distance from each training point to its $k$-th neighbour (other than itself). */
  kDistance: Float64Array
  /** The local reachability density of each training point. */
  density: Float64Array
  /** LOF of each training point (about 1 inside a cluster, much larger than 1 for outliers). */
  factor: Float64Array
}

/**
 * The local outlier factor of the training points (Breunig, Kriegel, Ng and Sander, 2000), as scikit-learn's
 * `LocalOutlierFactor` (whose `negative_outlier_factor_` is minus `factor`): with $N_k(p)$ the $k$ nearest other
 * points, the reachability distance of $p$ from $o$ is $\max(d_k(o), d(p, o))$, $d_k(o)$ the distance from $o$ to its
 * $k$-th neighbour; the local reachability density is
 * $\text{lrd}(p) = 1 / (\frac{1}{k} \sum_{o \in N_k(p)} \max(d_k(o), d(p, o)) + 10^{-10})$; and
 * $\text{LOF}(p) = \frac{1}{k} \sum_{o \in N_k(p)} \text{lrd}(o) / \text{lrd}(p)$. A point in a sparser region
 * than its neighbours has $\text{LOF} \gg 1$. Unlike scikit-learn, a $k$ of $n$ or more throws `DomainError` rather
 * than being lowered to $n - 1$.
 *
 * @param train The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param options The number of neighbours `k` (default 20, as scikit-learn), at most $n - 1$.
 * @returns The model: the training points, `k`, and each point's $k$-distance, density and `factor`.
 *
 * @example A line of points and one far away
 * const x = [[0], [1], [2], [3], [4], [12]]
 * const model = localOutlierFactor(x, { k: 2 })
 * print('LOF', model.factor)
 * print('density', model.density)
 */
export function localOutlierFactor(train: MatrixLike, options: { k?: Size } = {}): LocalOutlierFactor {
  const { k = 20 } = options
  const nn = neighboursOf(train, undefined, k, 'localOutlierFactor')
  const n = nn.m
  const kDistance = Float64Array.from({ length: n }, (_, p) => nn.dist[p * k + k - 1])
  const density = new Float64Array(n)
  for (let p = 0; p < n; p++) {
    let s = 0
    for (let j = 0; j < k; j++) s += Math.max(kDistance[nn.ids[p * k + j]], nn.dist[p * k + j])
    density[p] = 1 / (s / k + 1e-10)
  }
  const factor = new Float64Array(n)
  for (let p = 0; p < n; p++) {
    let s = 0
    for (let j = 0; j < k; j++) s += density[nn.ids[p * k + j]]
    factor[p] = s / k / density[p]
  }
  return { train, k, kDistance, density, factor }
}

/**
 * The LOF of new points relative to a fitted model (scikit-learn's `novelty=True`): their $k$ neighbours are found
 * among all the training points, and their density is compared with those neighbours' training densities.
 *
 * @param model The fitted model, as `localOutlierFactor` returns it.
 * @param queries The points to score, $m \times d$: nested arrays or a rank-2 tensor.
 * @returns The LOF of each query ($m$ values).
 *
 * @example New points inside and outside a line of points
 * const model = localOutlierFactor([[0], [1], [2], [3], [4]], { k: 2 })
 * print('LOF', localOutlierScore(model, [[2.5], [10]]))
 */
export function localOutlierScore(model: LocalOutlierFactor, queries: MatrixLike): Float64Array {
  const { k } = model
  const nn = neighboursOf(model.train, queries, k, 'localOutlierScore')
  const out = new Float64Array(nn.m)
  for (let q = 0; q < nn.m; q++) {
    let reach = 0
    let dens = 0
    for (let j = 0; j < k; j++) {
      const o = nn.ids[q * k + j]
      reach += Math.max(model.kDistance[o], nn.dist[q * k + j])
      dens += model.density[o]
    }
    const lrd = 1 / (reach / k + 1e-10)
    out[q] = dens / k / lrd
  }
  return out
}
