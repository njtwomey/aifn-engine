/**
 * Distance-based anomaly scores from the k nearest neighbours (found exactly by `aifn-compute/numerics/neighbours`): the
 * distance to the k-th neighbour (Ramaswamy, Rastogi and Shim, 2000) or the mean distance to the k nearest (Angiulli
 * and Pizzuti, 2002), and the local outlier factor (Breunig et al., 2000), which compares a point's local density with
 * its neighbours'.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'

const rowsOf = (x: MatrixLike, where: string) => {
  const m = dense.toMatrixF64(x, where)
  return fromData(Float64Array.from(m.data), [m.m, m.n])
}

/** Neighbours of `queries` among `train`, leaving a training point out of its own list when the queries are the training set. */
function neighboursOf(train: MatrixLike, queries: MatrixLike | undefined, k: Size, where: string) {
  const X = rowsOf(train, where)
  const Q = queries === undefined ? X : rowsOf(queries, where)
  const nn = bruteForceNeighbours(X, Q, k, { excludeSelf: queries === undefined })
  return { ids: toFlat(nn.indices), dist: toFlat(nn.distances), m: Q.shape[0] }
}

/** Options of `knnScore`. */
export type KnnScoreOptions = {
  /** Neighbours k (default 5). */
  k?: Size
  /** `kth`: the distance to the k-th neighbour; `mean`: the mean distance to the k nearest (default `kth`). */
  aggregate?: 'kth' | 'mean'
}

/**
 * The k-nearest-neighbour anomaly score of each query among the training points (of each training point among the
 * others when `queries` is omitted): far from its neighbours means anomalous.
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
  train: MatrixLike
  k: Size
  /** The distance from each training point to its k-th neighbour. */
  kDistance: Float64Array
  /** The local reachability density of each training point. */
  density: Float64Array
  /** LOF of each training point (≈ 1 inside a cluster, ≫ 1 for outliers). */
  factor: Float64Array
}

/**
 * The local outlier factor of the training points (Breunig, Kriegel, Ng and Sander, 2000), as scikit-learn's
 * `LocalOutlierFactor`: with N_k(p) the k nearest other points, the reachability distance of p from o is
 * max(k-distance(o), d(p, o)); the local reachability density is lrd(p) = 1/(mean_{o ∈ N_k(p)} reach-dist(p, o) + 10⁻¹⁰);
 * and LOF(p) = mean_{o ∈ N_k(p)} lrd(o) / lrd(p). A point in a sparser region than its neighbours has LOF ≫ 1.
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

/** The LOF of new points relative to a fitted model (scikit-learn's `novelty=True`): neighbours among the training points. */
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
