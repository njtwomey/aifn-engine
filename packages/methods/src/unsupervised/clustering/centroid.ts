/**
 * k-means and relatives:
 *
 * - `kmeansSteps`, `kmeans`: Lloyd's algorithm (Lloyd, 1957/1982) with k-means++ seeding (Arthur and Vassilvitskii,
 *   2007) and restarts, as scikit-learn's `KMeans(algorithm='lloyd')`.
 * - k-means++ seeding is `aifn-compute/numerics/neighbours`'s `kmeansPlusPlus`.
 * - `miniBatchKMeansSteps`, `miniBatchKMeans`: Sculley's (2010) mini-batch k-means with per-centre learning rates.
 * - `kMedoidsSteps`, `kMedoids`: PAM, BUILD then SWAP (Kaufman and Rousseeuw, 1990, "Finding Groups in Data", ch. 2).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type {
  Decides,
  Estimator,
  FitOptions,
  Fitted,
  Scores,
  Trained,
  Transforms,
} from 'aifn-compute/learning/estimators'
import type { Dataset } from 'aifn-compute/learning/estimators'
import { child, integers, stream } from 'aifn-compute/foundation/random'
import { pairwiseDistances, squaredDistances } from 'aifn-compute/numerics/linalg'
import { assignNearest, kmeansPlusPlus, lloydUpdate } from 'aifn-compute/numerics/neighbours'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { ints, mat, matrix, nearest, values, vec } from './util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { ShapeError } from 'aifn-compute/foundation/errors'

// ── Lloyd ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One state of Lloyd's algorithm: centroids and the assignment to them. */
export interface KMeansState extends Status {
  /** Lloyd iterations done. */
  t: number
  /** Centroids [k, d]. */
  centroids: Tensor
  /** The nearest centroid of each row [n] (int32). */
  labels: Tensor
  /** Σᵢ ‖xᵢ − c_labelᵢ‖² at these centroids. */
  inertia: number
  /** Rows per cluster [k]. */
  sizes: Tensor
  /** Total squared movement of the centroids in the last update (0 at the start). */
  shift: number
  /** Clusters that were empty in the last update; their centroids stay where they were. */
  empty: number[]
  /** The assignment did not change (strict convergence) or the shift fell to `tolerance`. */
  converged: boolean
}

/** How to start Lloyd's algorithm: given centroids [k, d], or a seeding drawn from the stream. */
export interface KMeansInit {
  centroids?: Tensor
  seeding?: 'k-means++' | 'random'
}

/** The nearest centroid of every row (core `assignNearest`, ties to the lower index) as plain arrays. */
function assign(v: Float64Array, n: number, c: Float64Array, k: number, d: number) {
  const a = assignNearest(mat(v, n, d), mat(c, k, d))
  return { labels: Int32Array.from(values(a.labels)), sizes: Float64Array.from(values(a.sizes)), inertia: a.inertia }
}

/**
 * Lloyd's algorithm as a traceable algorithm on the rows of x [n, d]: each step moves every centroid to the mean of
 * its rows and reassigns every row to its nearest centroid (ties to the lower index). Converged when no label changes
 * or the centroids' total squared shift is at most `tolerance` (default 0). An empty cluster keeps its centroid and is
 * reported in `empty`. `init` takes centroids, or seeds them from the `init` stream (k-means++ by default).
 */
export function kmeansSteps(x: Tensor, params: { k: number; tolerance?: number }): Algorithm<KMeansInit, KMeansState> {
  const { n, d, v } = matrix(x, 'kmeansSteps')
  const { k, tolerance: tol = 0 } = params
  return {
    name: 'lloyd',
    init: ({ centroids, seeding = 'k-means++' } = {}, s) => {
      let c: Float64Array
      if (centroids) {
        if (centroids.shape[0] !== k || centroids.shape[1] !== d)
          throw new ShapeError('kmeansSteps', `kmeansSteps: centroids must be [${k}, ${d}]`)
        c = Float64Array.from(values(centroids))
      } else {
        const st = s
        if (seeding === 'random') {
          const picks = new Set<number>()
          const sub = child(st, 'random-seeding')
          while (picks.size < k) picks.add(integers(sub, n))
          c = new Float64Array(k * d)
          ;[...picks].forEach((i, j) => c.set(v.subarray(i * d, (i + 1) * d), j * d))
        } else c = Float64Array.from(values(kmeansPlusPlus(st, x, k).centroids))
      }
      const a = assign(v, n, c, k, d)
      return {
        centroids: mat(c, k, d),
        labels: fromData(a.labels, [n]),
        inertia: a.inertia,
        sizes: vec(a.sizes),
        shift: 0,
        empty: [],
        t: 0,
        converged: false,
      }
    },
    step: (state) => {
      const labels = values(state.labels)
      // The update step is compute's `lloydUpdate`: each centroid to the mean of its rows, an empty one kept.
      const u = lloydUpdate(x, state.labels, state.centroids)
      const c = Float64Array.from(values(u.centroids))
      const { empty, shift } = u
      const a = assign(v, n, c, k, d)
      let same = true
      for (let i = 0; i < n; i++) if (a.labels[i] !== labels[i]) same = false
      return {
        centroids: mat(c, k, d),
        labels: fromData(a.labels, [n]),
        inertia: a.inertia,
        sizes: vec(a.sizes),
        shift,
        empty,
        t: state.t + 1,
        converged: same || shift <= tol,
        diverged: !Number.isFinite(a.inertia),
      }
    },
  }
}

/** A fitted k-means (or mini-batch k-means) model. */
export interface KMeansModel
  extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Scores<Tensor>, Transforms<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'kmeans' | 'mini-batch-kmeans'
  readonly centroids: Tensor
  readonly inertia: number
  /** Iterations (or batches) taken. */
  readonly steps: number
  readonly converged: boolean
  /** The inertia of every restart (k-means). */
  readonly restarts: Tensor
}

function centroidModel(centroids: Tensor, d: number) {
  const k = centroids.shape[0]
  const sqd = (q: Tensor) => {
    const { n: m, v, d: dq } = matrix(q, 'kmeans')
    if (dq !== d) throw new ShapeError('kmeans', `kmeans: fitted on ${d} features, given ${dq}`)
    const out = values(squaredDistances(mat(v, m, d), centroids))
    return { out, m }
  }
  return {
    /** Negative squared distances to the centroids [m, k]. */
    forward: (q: Tensor) => {
      const { out, m } = sqd(q)
      return mat(
        out.map((t) => -t),
        m,
        k,
      )
    },
    score: (q: Tensor) => {
      const { out, m } = sqd(q)
      return mat(
        out.map((t) => -t),
        m,
        k,
      )
    },
    /** Euclidean distances to the centroids [m, k]. */
    transform: (q: Tensor) => {
      const { out, m } = sqd(q)
      return mat(out.map(Math.sqrt), m, k)
    },
    decide: (q: Tensor) => {
      const { out, m } = sqd(q)
      const labels = new Int32Array(m)
      for (let i = 0; i < m; i++) for (let j = 1; j < k; j++) if (out[i * k + j] < out[i * k + labels[i]]) labels[i] = j
      return fromData(labels, [m])
    },
  }
}

/**
 * k-means by Lloyd's algorithm: `restarts` runs (default 10) from k-means++ seedings, each traced on the root
 * `child(stream, 'restart', r)`, keeping the lowest inertia; or one run from given `centroids`. `decide` gives the
 * nearest centroid (training labels are `decide(x)`), `transform` the distances to the centroids, `score` their negated
 * squares. The best run is kept in `training`.
 */
export function kmeans(params: {
  k: number
  restarts?: number
  centroids?: Tensor
  seeding?: 'k-means++' | 'random'
  maxSteps?: number
  tolerance?: number
}): Estimator<Dataset<Tensor>, KMeansModel & Trained<KMeansState>> {
  const { k, restarts = 10, centroids, seeding = 'k-means++', maxSteps = 300, tolerance = 0 } = params
  return {
    name: 'kmeans',
    params: { k, restarts, seeding, maxSteps, tolerance },
    fit({ x }, options: FitOptions = {}) {
      const { d } = matrix(x, 'kmeans')
      const alg = kmeansSteps(x, { k, tolerance })
      const s = options.stream ?? stream(0)
      const runs = centroids ? 1 : restarts
      let best: Trace<KMeansState> | null = null
      const inertias: number[] = []
      for (let r = 0; r < runs; r++) {
        const t = trace(alg, { centroids, seeding }, maxSteps, {
          stream: child(s, 'restart', r),
          every: options.trace?.every ?? 1,
          checkpointEvery: options.trace?.checkpointEvery,
          record: {
            inertia: (st) => st.inertia,
            ...(options.trace?.record as Record<string, (st: KMeansState, i: number) => number> | undefined),
          },
        })
        const final = t.final
        inertias.push(final.inertia)
        if (!best || final.inertia < best.final.inertia) best = t
      }
      const final = best!.final
      return {
        kind: 'model',
        name: 'kmeans',
        centroids: final.centroids,
        inertia: final.inertia,
        steps: final.t,
        converged: final.converged,
        restarts: vec(inertias),
        training: best!,
        ...centroidModel(final.centroids, d),
      }
    },
  }
}

// ── Mini-batch k-means ───────────────────────────────────────────────────────────────────────────────────────────

/** One state of mini-batch k-means. */
export interface MiniBatchKMeansState extends Status {
  /** Batches done. */
  t: number
  centroids: Tensor
  /** How many rows each centroid has absorbed so far [k] (its learning rate is 1/count). */
  counts: Tensor
  /** The rows of the latest batch (int32). */
  batch: Tensor
  /** The inertia of the whole data at these centroids. */
  inertia: number
}

/**
 * Mini-batch k-means (Sculley, 2010, Algorithm 1): each step draws `batchSize` rows uniformly with replacement from
 * the step's stream, assigns them to their nearest centroids, and moves each centroid towards each of its rows by the
 * per-centre rate 1/count. `init` takes centroids or seeds them by k-means++ from the `init` stream.
 */
export function miniBatchKMeansSteps(
  x: Tensor,
  params: { k: number; batchSize?: number },
): Algorithm<KMeansInit, MiniBatchKMeansState> {
  const { n, d, v } = matrix(x, 'miniBatchKMeansSteps')
  const { k, batchSize = Math.min(n, 64) } = params
  return {
    name: 'mini-batch-kmeans',
    init: ({ centroids } = {}, s) => {
      const c = centroids
        ? Float64Array.from(values(centroids))
        : Float64Array.from(values(kmeansPlusPlus(child(s, 'seeding'), x, k).centroids))
      return {
        centroids: mat(c, k, d),
        counts: vec(new Float64Array(k)),
        batch: ints([]),
        inertia: assign(v, n, c, k, d).inertia,
        t: 0,
      }
    },
    step: (state, ctx) => {
      const c = Float64Array.from(values(state.centroids))
      const counts = Float64Array.from(values(state.counts))
      const sub = ctx.stream
      const batch = Int32Array.from({ length: batchSize }, () => integers(sub, n))
      const nearestOf = Int32Array.from(batch, (i) => nearest(v, i, c, k, d)[0])
      batch.forEach((i, b) => {
        const j = nearestOf[b]
        counts[j]++
        const eta = 1 / counts[j]
        for (let t = 0; t < d; t++) c[j * d + t] = (1 - eta) * c[j * d + t] + eta * v[i * d + t]
      })
      return {
        centroids: mat(c, k, d),
        counts: vec(counts),
        batch: fromData(batch, [batchSize]),
        inertia: assign(v, n, c, k, d).inertia,
        t: state.t + 1,
      }
    },
  }
}

/** Mini-batch k-means for `steps` batches (default 100). */
export function miniBatchKMeans(params: {
  k: number
  batchSize?: number
  steps?: number
  centroids?: Tensor
}): Estimator<Dataset<Tensor>, KMeansModel & Trained<MiniBatchKMeansState>> {
  const { k, batchSize, steps = 100, centroids } = params
  return {
    name: 'mini-batch-kmeans',
    params: { k, batchSize, steps },
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'miniBatchKMeans')
      const training = trace(miniBatchKMeansSteps(x, { k, batchSize }), { centroids }, steps, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        record: { inertia: (s) => s.inertia },
      })
      const final = training.final
      const a = assign(v, n, values(final.centroids), k, d)
      return {
        kind: 'model',
        name: 'mini-batch-kmeans',
        centroids: final.centroids,
        inertia: a.inertia,
        steps: final.t,
        converged: false,
        restarts: vec([a.inertia]),
        training,
        ...centroidModel(final.centroids, d),
      }
    },
  }
}

// ── k-medoids (PAM) ──────────────────────────────────────────────────────────────────────────────────────────────

/** One PAM state. */
export interface KMedoidsState extends Status {
  /** Swaps tried. */
  t: number
  /** The medoids (row indices, int32 [k]). */
  medoids: Tensor
  /** The nearest medoid's position in `medoids` for each row [n]. */
  labels: Tensor
  /** Σᵢ distance to the nearest medoid. */
  cost: number
  /** The swap made in this step, [medoid out, row in], or null. */
  swap: [number, number] | null
  converged: boolean
}

function medoidCost(D: Float64Array, n: number, medoids: readonly number[]) {
  const labels = new Int32Array(n)
  let cost = 0
  for (let i = 0; i < n; i++) {
    let best = Infinity
    medoids.forEach((m, j) => {
      if (D[i * n + m] < best) {
        best = D[i * n + m]
        labels[i] = j
      }
    })
    cost += best
  }
  return { labels, cost }
}

/**
 * PAM as a traceable algorithm on a distance matrix [n, n]: `init` runs BUILD (greedily add the medoid that lowers
 * the cost most) unless medoids are given; each step makes the best cost-lowering swap of a medoid with a non-medoid,
 * and it is done when no swap lowers the cost.
 */
export function kMedoidsSteps(
  distances: Tensor,
  params: { k: number },
): Algorithm<{ medoids?: readonly number[] }, KMedoidsState> {
  const [n, n2] = distances.shape
  if (n !== n2) throw new ShapeError('kMedoidsSteps', 'kMedoidsSteps: distances must be [n, n]')
  const D = values(distances)
  const { k } = params
  const make = (medoids: number[], t: number, swap: [number, number] | null, converged: boolean): KMedoidsState => {
    const { labels, cost } = medoidCost(D, n, medoids)
    return { medoids: ints(medoids), labels: fromData(labels, [n]), cost, swap, t, converged }
  }
  return {
    name: 'pam',
    init: ({ medoids } = {}) => {
      if (medoids) return make([...medoids], 0, null, false)
      const chosen: number[] = []
      for (let j = 0; j < k; j++) {
        let best = -1
        let bestCost = Infinity
        for (let c = 0; c < n; c++) {
          if (chosen.includes(c)) continue
          const cost = medoidCost(D, n, [...chosen, c]).cost
          if (cost < bestCost) {
            bestCost = cost
            best = c
          }
        }
        chosen.push(best)
      }
      return make(chosen, 0, null, false)
    },
    step: (state) => {
      const medoids = Array.from(state.medoids.data as Int32Array)
      let best: [number, number] | null = null
      let bestCost = state.cost
      for (let j = 0; j < k; j++) {
        for (let c = 0; c < n; c++) {
          if (medoids.includes(c)) continue
          const trial = medoids.slice()
          trial[j] = c
          const cost = medoidCost(D, n, trial).cost
          if (cost < bestCost - 1e-12 * Math.max(1, Math.abs(bestCost))) {
            bestCost = cost
            best = [j, c]
          }
        }
      }
      if (!best) return { ...state, t: state.t + 1, swap: null, converged: true }
      const out = medoids[best[0]]
      medoids[best[0]] = best[1]
      return make(medoids, state.t + 1, [out, best[1]], false)
    },
  }
}

/** A fitted k-medoids model. */
export interface KMedoidsModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Scores<Tensor>,
    Transforms<Tensor, Tensor>,
    Trained<KMedoidsState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'k-medoids'
  /** Medoid row indices [k] and their coordinates [k, d]. */
  readonly medoids: Tensor
  readonly centres: Tensor
  readonly cost: number
}

/**
 * k-medoids by PAM on Euclidean distances between the rows of x. `decide` assigns new rows to the nearest medoid,
 * `transform` gives the distances to the medoids, `forward`/`score` their negated squares (as k-means).
 */
export function kMedoids(params: { k: number; maxSteps?: number }): Estimator<Dataset<Tensor>, KMedoidsModel> {
  const { k, maxSteps = 100 } = params
  return {
    name: 'k-medoids',
    params,
    fit({ x }, options: FitOptions = {}) {
      const { d, v } = matrix(x, 'kMedoids')
      const training = trace(kMedoidsSteps(pairwiseDistances(x) as Tensor, { k }), {}, maxSteps, {
        every: options.trace?.every ?? 1,
        record: { cost: (s) => s.cost },
      })
      const final = training.final
      const m = final.medoids.data as Int32Array
      const centres = new Float64Array(k * d)
      m.forEach((i, j) => centres.set(v.subarray(i * d, (i + 1) * d), j * d))
      return {
        kind: 'model',
        name: 'k-medoids',
        medoids: final.medoids,
        centres: mat(centres, k, d),
        cost: final.cost,
        training,
        ...centroidModel(mat(centres, k, d), d),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'kmeans',
    module: 'unsupervised/clustering',
    name: 'k-means',
    summary: "Lloyd's algorithm from k-means++ seeds, best of several restarts.",
    task: 'clustering',
    capabilities: ['forward', 'decide', 'score', 'transform'],
    hyper: space({
      k: int(1, 20, { default: 3 }),
      restarts: int(1, 50, { default: 10 }),
      seeding: oneOf(['k-means++', 'random']),
      maxSteps: int(1, 1000, { default: 300 }),
      tolerance: real(0, 1, { default: 0 }),
    }),
    notes: ['k-means'],
    cite: ['lloyd1982', 'arthur2007'],
  },
  kmeans,
)

defineModel(
  {
    key: 'miniBatchKMeans',
    module: 'unsupervised/clustering',
    name: 'Mini-batch k-means',
    summary: 'k-means with centroid updates from random mini-batches.',
    task: 'clustering',
    capabilities: ['forward', 'decide', 'score', 'transform'],
    hyper: space({
      k: int(1, 20, { default: 3 }),
      batchSize: int(1, 1024, { default: 64 }),
      steps: int(1, 10000, { default: 100 }),
    }),
    notes: ['k-means'],
  },
  miniBatchKMeans,
)

defineModel(
  {
    key: 'kMedoids',
    module: 'unsupervised/clustering',
    name: 'k-medoids',
    summary: 'Clusters represented by data points (medoids), by alternating assignment and medoid update.',
    task: 'clustering',
    capabilities: ['forward', 'decide', 'score', 'transform'],
    hyper: space({ k: int(1, 20, { default: 3 }), maxSteps: int(1, 1000, { default: 100 }) }),
    notes: ['k-means'],
  },
  kMedoids,
)
