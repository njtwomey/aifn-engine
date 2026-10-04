/**
 * k-nearest-neighbour classification and regression (Fix and Hodges, 1951; Cover and Hart, 1967) by exhaustive
 * search: small data, exact answers, and the neighbours themselves exposed for figures.
 */

import type {
  AnyUnivariate,
  Decides,
  Estimator,
  Expects,
  Fitted,
  Predicts,
  Scores,
  Supervised,
} from 'aifn-compute/learning/estimators'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { rowDistance, type PairwiseMetric } from 'aifn-compute/numerics/linalg'
import { classLabels, inputs, matrix, probabilityModel, targets } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Distances between points: Minkowski of order p (Euclidean p = 2, Manhattan p = 1) or Chebyshev (p = ∞). */
export type Metric = 'euclidean' | 'manhattan' | 'chebyshev' | { minkowski: number }

/** Hyperparameters of the k-NN estimators. */
export interface NeighboursParams {
  /** Number of neighbours (default 5). */
  k?: number
  /**
   * `uniform` (default): every neighbour votes equally. `distance`: weight 1/d, and a query at distance 0 from some
   * training points takes their vote only (scikit-learn's convention).
   */
  weights?: 'uniform' | 'distance'
  metric?: Metric
}

/** The k nearest training rows of each query: indices and distances [m, k], nearest first (ties to the lower index). */
export interface Neighbours {
  index: Tensor
  distance: Tensor
}

/** The row-pair distance of a k-NN metric: linalg's `rowDistance` (Minkowski orders map to its named metrics). */
function distanceFn(metric: Metric): (a: Float64Array, i: number, b: Float64Array, j: number, d: number) => number {
  if (typeof metric === 'string') return (a, i, b, j, d) => rowDistance(a, i, b, j, d, metric)
  const p = metric.minkowski
  if (!(p >= 1))
    throw new DomainError('kNearestNeighbours', 'kNearestNeighbours: the Minkowski order must be at least 1')
  const named: PairwiseMetric =
    p === Infinity ? 'chebyshev' : p === 1 ? 'manhattan' : p === 2 ? 'euclidean' : 'minkowski'
  return (a, i, b, j, d) => rowDistance(a, i, b, j, d, named, p)
}

/** Exhaustive k-nearest-neighbour search of the rows of `q` among the rows of `x`. */
function search(
  x: Float64Array,
  n: number,
  q: Float64Array,
  m: number,
  d: number,
  k: number,
  metric: Metric,
): { index: Int32Array; distance: Float64Array } {
  const dist = distanceFn(metric)
  const index = new Int32Array(m * k)
  const distance = new Float64Array(m * k)
  const all = new Float64Array(n)
  const order = new Int32Array(n)
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      all[j] = dist(q, i, x, j, d)
      order[j] = j
    }
    // A stable sort by distance keeps ties in index order.
    const sorted = Array.from(order).sort((a, b) => all[a] - all[b] || a - b)
    for (let r = 0; r < k; r++) {
      index[i * k + r] = sorted[r]
      distance[i * k + r] = all[sorted[r]]
    }
  }
  return { index, distance }
}

/** Neighbour weights for one query row: uniform, or 1/d with exact matches taking all the weight. */
function neighbourWeights(distance: Float64Array, i: number, k: number, mode: 'uniform' | 'distance'): Float64Array {
  const w = new Float64Array(k)
  if (mode === 'uniform') return w.fill(1)
  let exact = false
  for (let r = 0; r < k; r++) if (distance[i * k + r] === 0) exact = true
  for (let r = 0; r < k; r++) {
    const dr = distance[i * k + r]
    w[r] = exact ? (dr === 0 ? 1 : 0) : 1 / dr
  }
  return w
}

/** A fitted k-NN classifier. */
export interface NeighboursClassifier
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Predicts<Tensor, AnyUnivariate> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'k-nearest-neighbours'
  readonly k: number
  readonly classes: number
  readonly weights: 'uniform' | 'distance'
  readonly metric: Metric
  /** The training inputs [n, d] and labels [n] (the model is the data). */
  readonly x: Tensor
  readonly y: Tensor
  /** The k nearest training rows of each query row. */
  neighbours(x: Tensor): Neighbours
}

/**
 * k-nearest-neighbour classification: the predictive is the (weighted) share of each class among the k nearest
 * training points. `forward` and `score` return those shares [m, K]; `decide` the class with the largest share (ties to
 * the lowest label, as scikit-learn).
 */
export function kNearestNeighbours(
  params: NeighboursParams = {},
): Estimator<Supervised<Tensor, Tensor>, NeighboursClassifier> {
  const { k = 5, weights = 'uniform', metric = 'euclidean' } = params
  if (!(Number.isInteger(k) && k >= 1))
    throw new DomainError('kNearestNeighbours', 'kNearestNeighbours: k must be a positive integer')
  return {
    name: 'k-nearest-neighbours',
    params: { k, weights, metric },
    fit({ x, y }) {
      const { n, d, v } = matrix(x, 'kNearestNeighbours')
      if (k > n) throw new DomainError('kNearestNeighbours', `kNearestNeighbours: k = ${k} but only ${n} training rows`)
      const { y: labels, k: K } = classLabels(y, n, 'kNearestNeighbours')
      const neighbours = (q: Tensor) => {
        const { n: m, v: qv } = inputs(q, d, 'kNearestNeighbours')
        return search(v, n, qv, m, d, k, metric)
      }
      const shares = (q: Tensor): Float64Array => {
        const { index, distance } = neighbours(q)
        const m = q.shape[0]
        const out = new Float64Array(m * K)
        for (let i = 0; i < m; i++) {
          const w = neighbourWeights(distance, i, k, weights)
          let total = 0
          for (let r = 0; r < k; r++) {
            out[i * K + labels[index[i * k + r]]] += w[r]
            total += w[r]
          }
          for (let c = 0; c < K; c++) out[i * K + c] /= total
        }
        return out
      }
      return {
        kind: 'model',
        name: 'k-nearest-neighbours',
        k,
        classes: K,
        weights,
        metric,
        x,
        y,
        neighbours: (q: Tensor): Neighbours => {
          const r = neighbours(q)
          return { index: fromData(r.index, [q.shape[0], k]), distance: fromData(r.distance, [q.shape[0], k]) }
        },
        ...probabilityModel(shares, (h) => h, K),
      }
    },
  }
}

/** A fitted k-NN regressor. */
export interface NeighboursRegressor extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Expects<Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'k-nearest-neighbours-regression'
  readonly k: number
  readonly x: Tensor
  readonly y: Tensor
  neighbours(x: Tensor): Neighbours
  /** The weighted standard deviation of the neighbours' targets [m]: a descriptive spread, not a calibrated one. */
  spread(x: Tensor): Tensor
}

/**
 * k-nearest-neighbour regression: the prediction is the (weighted) mean of the neighbours' targets, and `expect(x, f)`
 * the weighted mean of f over them (the neighbours' empirical law of y).
 */
export function kNearestNeighboursRegression(
  params: NeighboursParams = {},
): Estimator<Supervised<Tensor, Tensor>, NeighboursRegressor> {
  const { k = 5, weights = 'uniform', metric = 'euclidean' } = params
  return {
    name: 'k-nearest-neighbours-regression',
    params: { k, weights, metric },
    fit({ x, y }) {
      const { n, d, v } = matrix(x, 'kNearestNeighboursRegression')
      if (k > n)
        throw new DomainError(
          'kNearestNeighboursRegression',
          `kNearestNeighboursRegression: k = ${k} but only ${n} training rows`,
        )
      const t = targets(y, n, 'kNearestNeighboursRegression')
      const moments = (q: Tensor) => {
        const { n: m, v: qv } = inputs(q, d, 'kNearestNeighboursRegression')
        const { index, distance } = search(v, n, qv, m, d, k, metric)
        const mean = new Float64Array(m)
        const sd = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          const w = neighbourWeights(distance, i, k, weights)
          let total = 0
          let s = 0
          for (let r = 0; r < k; r++) {
            total += w[r]
            s += w[r] * t[index[i * k + r]]
          }
          mean[i] = s / total
          let q2 = 0
          for (let r = 0; r < k; r++) q2 += w[r] * (t[index[i * k + r]] - mean[i]) ** 2
          sd[i] = Math.sqrt(q2 / total)
        }
        return { mean, sd, index, distance, m }
      }
      const predict = (q: Tensor) => fromData(moments(q).mean, [q.shape[0]])
      return {
        kind: 'model',
        name: 'k-nearest-neighbours-regression',
        k,
        x,
        y,
        neighbours: (q: Tensor): Neighbours => {
          const r = moments(q)
          return { index: fromData(r.index, [r.m, k]), distance: fromData(r.distance, [r.m, k]) }
        },
        forward: predict,
        decide: predict,
        expect: (q: Tensor, f?: (y: number) => number) => {
          if (!f) return predict(q)
          const r = moments(q)
          return fromData(
            Float64Array.from({ length: r.m }, (_, i) => {
              const w = neighbourWeights(r.distance, i, k, weights)
              let total = 0
              let s = 0
              for (let j = 0; j < k; j++) {
                total += w[j]
                s += w[j] * f(t[r.index[i * k + j]])
              }
              return s / total
            }),
            [r.m],
          )
        },
        spread: (q: Tensor) => fromData(moments(q).sd, [q.shape[0]]),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'kNearestNeighbours',
    module: 'learning/neighbours',
    name: 'k-nearest neighbours',
    summary: 'Majority (or distance-weighted) vote of the k nearest training points.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({
      k: int(1, 50, { default: 5 }),
      weights: oneOf(['uniform', 'distance']),
      metric: oneOf(['euclidean', 'manhattan', 'chebyshev']),
    }),
    notes: ['k-nearest-neighbours'],
    cite: ['cover1967'],
  },
  kNearestNeighbours,
)

defineModel(
  {
    key: 'kNearestNeighboursRegression',
    module: 'learning/neighbours',
    name: 'k-nearest neighbours regression',
    summary: 'The (distance-weighted) mean target of the k nearest training points.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'expect'],
    hyper: space({
      k: int(1, 50, { default: 5 }),
      weights: oneOf(['uniform', 'distance']),
      metric: oneOf(['euclidean', 'manhattan', 'chebyshev']),
    }),
    notes: ['k-nearest-neighbours'],
    cite: ['cover1967'],
  },
  kNearestNeighboursRegression,
)
