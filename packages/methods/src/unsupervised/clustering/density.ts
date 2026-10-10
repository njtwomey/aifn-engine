/**
 * Density-based and mode-seeking clustering, all on Euclidean distances between the rows of the data, with the full
 * $n \times n$ distance matrix held for DBSCAN and OPTICS:
 *
 * - `dbscan`: DBSCAN (Ester, Kriegel, Sander and Xu, 1996) with core, border and noise points, as scikit-learn.
 * - `optics`: OPTICS (Ankerst, Breunig, Kriegel and Sander, 1999): the ordering, reachability and core distances, as
 *   scikit-learn's `OPTICS(max_eps=inf)`, and the model's `clustersAt` extracting DBSCAN-like clusters at any
 *   $\varepsilon$.
 * - `meanShiftSteps`, `meanShift`: mean shift (Fukunaga and Hostetler, 1975; Comaniciu and Meer, 2002) with a flat or
 *   Gaussian kernel, every point a seed, as scikit-learn's `MeanShift` for the flat kernel.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type { Dataset, Decides, Estimator, FitOptions, Trained, Transforms } from 'aifn-compute/learning/estimators'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { pairwiseDistances } from 'aifn-compute/numerics/linalg'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { mat, matrix, pairwise, sq, vec } from './util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── DBSCAN ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** A point's role in DBSCAN: 0 core, 1 border, 2 noise. This is the core role. */
export const CORE = 0
/** The DBSCAN role of a point within `eps` of a core point but not core itself. */
export const BORDER = 1
/** The DBSCAN role of a point in no cluster. */
export const NOISE = 2

/** A fitted DBSCAN. */
export interface DbscanModel extends Decides<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'dbscan'
  /** The neighbourhood radius $\varepsilon$ it was fitted with. */
  readonly eps: number
  /** The neighbours (itself included) that make a point core. */
  readonly minSamples: number
  /** Cluster labels $0, 1, \dots$ in order of discovery, $-1$ for noise, $n$ values (int32). */
  readonly labels: Tensor
  /** The role of each point: `CORE`, `BORDER` or `NOISE`, $n$ values (int32). */
  readonly roles: Tensor
  /** Number of $\varepsilon$-neighbours of each point, itself included, $n$ values (int32). */
  readonly neighbourCounts: Tensor
  /** The number of clusters found. */
  readonly clusters: number
}

/**
 * DBSCAN: a core point has at least `minSamples` points (itself included) within distance `eps`; clusters are the
 * connected components of core points under $\varepsilon$-adjacency, plus the border points within $\varepsilon$ of
 * them. Rows are scanned in order, so a border point near two clusters joins the first one found (as scikit-learn).
 * `decide` gives a new point the label of its nearest core point within $\varepsilon$, else $-1$. The fit is
 * deterministic and holds all $n^2$ distances.
 *
 * @param params The hyperparameters.
 * @param params.eps The neighbourhood radius $\varepsilon$: points at Euclidean distance at most `eps` are neighbours.
 * @param params.minSamples The neighbours, the point itself included, that make a point core (default 5).
 * @returns The estimator; `fit({ x })` takes the data, $n \times d$.
 *
 * @example A tight group and one outlier, which is noise
 * const x = concat([normals(stream(0), [8, 2], 0, 0.3), tensor([[4, 4]])])
 * const model = dbscan({ eps: 1, minSamples: 3 }).fit({ x })
 * print('labels', model.labels)
 * print('roles (0 core, 1 border, 2 noise)', model.roles)
 * print('new points', model.decide(tensor([[0.2, 0.1], [3, 3]])))
 */
export function dbscan(params: { eps: number; minSamples?: number }): Estimator<Dataset<Tensor>, DbscanModel> {
  const { eps, minSamples = 5 } = params
  return {
    name: 'dbscan',
    params: { eps, minSamples },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'dbscan')
      const D = pairwise(x)
      const neighbours: number[][] = Array.from({ length: n }, (_, i) => {
        const out: number[] = []
        for (let j = 0; j < n; j++) if (D[i * n + j] <= eps) out.push(j)
        return out
      })
      const core = neighbours.map((nb) => nb.length >= minSamples)
      const labels = new Int32Array(n).fill(-1)
      let cluster = 0
      for (let i = 0; i < n; i++) {
        if (labels[i] !== -1 || !core[i]) continue
        const stack = [i]
        labels[i] = cluster
        while (stack.length) {
          const p = stack.pop()!
          if (!core[p]) continue
          for (const q of neighbours[p]) {
            if (labels[q] !== -1) continue
            labels[q] = cluster
            if (core[q]) stack.push(q)
          }
        }
        cluster++
      }
      const roles = Int32Array.from(labels, (l, i) => (core[i] ? CORE : l >= 0 ? BORDER : NOISE))
      return {
        kind: 'model',
        name: 'dbscan',
        eps,
        minSamples,
        labels: fromData(labels, [n]),
        roles: fromData(roles, [n]),
        neighbourCounts: fromData(
          Int32Array.from(neighbours, (nb) => nb.length),
          [n],
        ),
        clusters: cluster,
        decide: (q: Tensor) => {
          const { n: m, v: qv } = matrix(q, 'dbscan')
          const out = new Int32Array(m).fill(-1)
          for (let i = 0; i < m; i++) {
            let best = Infinity
            for (let j = 0; j < n; j++) {
              if (!core[j]) continue
              const t = Math.sqrt(sq(qv, i, v, j, d))
              if (t <= eps && t < best) {
                best = t
                out[i] = labels[j]
              }
            }
          }
          return fromData(out, [m])
        },
      }
    },
  }
}

// ── OPTICS ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted OPTICS ordering. */
export interface OpticsModel {
  readonly kind: 'model'
  /** OPTICS orders the training points only: it cannot place new inputs. */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'optics'
  /** The neighbours, the point itself included, that define the core distance. */
  readonly minSamples: number
  /** The processing order of the points, $n$ indices (int32). */
  readonly ordering: Tensor
  /**
   * Reachability distance of each point, indexed by point, $n$ values: $\infty$ for the first point processed, which
   * with unbounded $\varepsilon$ is the only one reached from none.
   */
  readonly reachability: Tensor
  /** Core distance of each point: distance to its `minSamples`-th nearest point, itself included, $n$ values. */
  readonly coreDistances: Tensor
  /** The point each was reached from ($-1$ when none), $n$ values (int32). */
  readonly predecessor: Tensor
  /**
   * DBSCAN-like clusters at radius `eps` (scikit-learn's `cluster_optics_dbscan`): labels $0, 1, \dots$ along the
   * ordering, $-1$ for noise.
   */
  clustersAt(eps: number): Tensor
}

/**
 * OPTICS with unbounded $\varepsilon$: repeatedly take the unprocessed point $p$ of smallest reachability (the lowest
 * index among ties, so point 0 first) and lower every unprocessed point $q$'s reachability to
 * $\max(\text{core}(p), d(p, q))$ when that is smaller. Throws `DomainError` when `minSamples` exceeds $n$. The fit is
 * deterministic and holds all $n^2$ distances; the model is transductive (it has no `decide`).
 *
 * @param params The hyperparameters: `minSamples`, the neighbours (the point itself included) whose farthest sets a
 *   point's core distance (default 5).
 * @returns The estimator; `fit({ x })` takes the data, $n \times d$.
 *
 * @example Two groups on a line, read off at two radii
 * const x = tensor([[0], [0.1], [0.3], [5], [5.2], [5.3], [9]])
 * const model = optics({ minSamples: 2 }).fit({ x })
 * print('ordering', model.ordering)
 * print('reachability', model.reachability)
 * print('clusters at 1', model.clustersAt(1))
 * print('clusters at 5', model.clustersAt(5))
 */
export function optics(params: { minSamples?: number } = {}): Estimator<Dataset<Tensor>, OpticsModel> {
  const { minSamples = 5 } = params
  return {
    name: 'optics',
    params: { minSamples },
    fit({ x }) {
      const { n } = matrix(x, 'optics')
      if (minSamples > n) throw new DomainError('optics', 'optics: minSamples exceeds the number of points')
      const D = pairwise(x)
      const core = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        const row = Array.from(D.subarray(i * n, (i + 1) * n)).sort((a, b) => a - b)
        core[i] = row[minSamples - 1]
      }
      const reach = new Float64Array(n).fill(Infinity)
      const pred = new Int32Array(n).fill(-1)
      const done = new Uint8Array(n)
      const ordering: number[] = []
      for (let t = 0; t < n; t++) {
        let p = -1
        for (let i = 0; i < n; i++) if (!done[i] && (p < 0 || reach[i] < reach[p])) p = i
        done[p] = 1
        ordering.push(p)
        for (let q = 0; q < n; q++) {
          if (done[q]) continue
          const r = Math.max(core[p], D[p * n + q])
          if (r < reach[q]) {
            reach[q] = r
            pred[q] = p
          }
        }
      }
      return {
        kind: 'model',
        transductive: true,
        name: 'optics',
        minSamples,
        ordering: fromData(Int32Array.from(ordering), [n]),
        reachability: vec(reach),
        coreDistances: vec(core),
        predecessor: fromData(pred, [n]),
        clustersAt: (eps: number) => {
          // scikit-learn's cluster_optics_dbscan: a new cluster starts where reachability exceeds ε at a core point.
          const labels = new Int32Array(n)
          let c = -1
          for (const p of ordering) {
            if (reach[p] > eps) {
              if (core[p] <= eps) labels[p] = ++c
              else labels[p] = -1
            } else labels[p] = c
          }
          for (let i = 0; i < n; i++) if (reach[i] > eps && core[i] > eps) labels[i] = -1
          return fromData(labels, [n])
        },
      }
    },
  }
}

// ── Mean shift ───────────────────────────────────────────────────────────────────────────────────────────────────

/** One state of mean shift: every seed's position. */
export interface MeanShiftState extends Status {
  /** Steps taken. */
  t: number
  /** Seed positions, $s \times d$. */
  seeds: Tensor
  /** Which seeds have stopped moving, $s$ flags (int32, 1 when settled). */
  settled: Tensor
  /** Largest shift of any seed in the last step ($\infty$ at the start). */
  shift: number
  /** True once every seed has settled. */
  converged: boolean
}

/**
 * Mean shift as a traceable algorithm: every seed moves to the (kernel-weighted) mean of the data within its window,
 * and settles once a step moves it at most $10^{-3} h$ ($h$ the bandwidth), or when its window holds no point. `flat`
 * uses the points within distance $h$ (scikit-learn); `gaussian` weights every point by
 * $\exp(-\lVert \xvec - \svec \rVert^2 / 2h^2)$ for a seed at $\svec$. Seeds default to the data points.
 *
 * @param x The data, $n \times d$, one point per row.
 * @param params The hyperparameters: `bandwidth`, the window radius (flat) or standard deviation (Gaussian) $h$; and
 *   `kernel`, `'flat'` (default) or `'gaussian'`.
 * @returns The algorithm, whose `init` takes optional starting `seeds` ($s \times d$; the rows of `x` when left out)
 *   and whose state is a `MeanShiftState`.
 *
 * @example Five seeds climb to two modes
 * const x = tensor([[0], [0.5], [1], [6], [6.5]])
 * const final = run(meanShiftSteps(x, { bandwidth: 1 }), {}, 20)
 * print('seeds', final.seeds)
 * print('steps', final.t, 'converged', final.converged)
 */
export function meanShiftSteps(
  x: Tensor,
  params: { bandwidth: number; kernel?: 'flat' | 'gaussian' },
): Algorithm<{ seeds?: Tensor }, MeanShiftState> {
  const { n, d, v } = matrix(x, 'meanShiftSteps')
  const { bandwidth: h, kernel = 'flat' } = params
  const stop = 1e-3 * h
  return {
    name: 'mean-shift',
    init: ({ seeds } = {}) => {
      const s = seeds ? Float64Array.from(seeds.data as Float64Array) : Float64Array.from(v)
      const m = s.length / d
      return {
        seeds: mat(s, m, d),
        settled: fromData(new Int32Array(m), [m]),
        shift: Infinity,
        t: 0,
        converged: false,
      }
    },
    step: (state) => {
      const old = state.seeds.data as Float64Array
      const m = old.length / d
      const settled = Int32Array.from(state.settled.data as ArrayLike<number>)
      const s = Float64Array.from(old)
      let shift = 0
      for (let a = 0; a < m; a++) {
        if (settled[a]) continue
        const mean = new Float64Array(d)
        let w = 0
        for (let i = 0; i < n; i++) {
          const t = sq(old, a, v, i, d)
          const wi = kernel === 'flat' ? (t <= h * h ? 1 : 0) : Math.exp(-t / (2 * h * h))
          if (!wi) continue
          w += wi
          for (let j = 0; j < d; j++) mean[j] += wi * v[i * d + j]
        }
        if (w === 0) {
          settled[a] = 1
          continue
        }
        let move = 0
        for (let j = 0; j < d; j++) {
          mean[j] /= w
          move += (mean[j] - old[a * d + j]) ** 2
          s[a * d + j] = mean[j]
        }
        move = Math.sqrt(move)
        shift = Math.max(shift, move)
        if (move <= stop) settled[a] = 1
      }
      const all = settled.every((u) => u === 1)
      return {
        seeds: mat(s, m, d),
        settled: fromData(settled, [m]),
        shift,
        t: state.t + 1,
        converged: all,
      }
    },
  }
}

/** A fitted mean shift. */
export interface MeanShiftModel extends Decides<Tensor, Tensor>, Transforms<Tensor, Tensor>, Trained<MeanShiftState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'mean-shift'
  /** The modes found, $c \times d$, by decreasing number of points within the bandwidth. */
  readonly centres: Tensor
}

/**
 * Mean shift: runs every seed to a mode (`meanShiftSteps`), merges modes within the bandwidth of each other (keeping
 * the one with more points within the bandwidth, as scikit-learn), and drops modes with no point within it. `decide`
 * labels a point by its nearest mode (training labels are `decide(x)`), `transform` gives its distances to the modes.
 * The fit is deterministic (the fit's stream is not used). For the Gaussian kernel, too, a mode's count of points is
 * taken within the flat window of radius `bandwidth`.
 *
 * @param params The hyperparameters.
 * @param params.bandwidth The window radius (flat) or standard deviation (Gaussian) $h$, also the distance within
 *   which two modes merge.
 * @param params.kernel `'flat'` (default) or `'gaussian'`.
 * @param params.maxSteps The most mean-shift steps (default 300).
 * @returns The estimator; `fit({ x })` takes the data, $n \times d$.
 *
 * @example Two blobs, two modes
 * const s = stream(0)
 * const x = concat([normals(s, [10, 2], 0, 0.3), normals(s, [10, 2], 4, 0.3)])
 * const model = meanShift({ bandwidth: 1 }).fit({ x })
 * print('modes', model.centres)
 * print('labels', model.decide(x))
 */
export function meanShift(params: {
  bandwidth: number
  kernel?: 'flat' | 'gaussian'
  maxSteps?: number
}): Estimator<Dataset<Tensor>, MeanShiftModel> {
  const { bandwidth: h, kernel = 'flat', maxSteps = 300 } = params
  return {
    name: 'mean-shift',
    params,
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'meanShift')
      const training = trace(meanShiftSteps(x, { bandwidth: h, kernel }), {}, maxSteps, {
        every: options.trace?.every ?? 1,
        stopOnNonFinite: false,
        record: { shift: (s) => (Number.isFinite(s.shift) ? s.shift : NaN) },
      })
      const seeds = training.final.seeds.data as Float64Array
      const m = seeds.length / d
      const intensity = Array.from({ length: m }, (_, a) => {
        let c = 0
        for (let i = 0; i < n; i++) if (sq(seeds, a, v, i, d) <= h * h) c++
        return c
      })
      // Most intense first; ties by the coordinates, largest first (scikit-learn sorts (intensity, centre) descending).
      const byCoordinates = (a: number, b: number) => {
        for (let j = 0; j < d; j++)
          if (seeds[a * d + j] !== seeds[b * d + j]) return seeds[b * d + j] - seeds[a * d + j]
        return a - b
      }
      const order = Array.from({ length: m }, (_, a) => a)
        .filter((a) => intensity[a] > 0)
        .sort((a, b) => intensity[b] - intensity[a] || byCoordinates(a, b))
      const kept: number[] = []
      for (const a of order) if (kept.every((b) => sq(seeds, a, seeds, b, d) > h * h)) kept.push(a)
      const centres = new Float64Array(kept.length * d)
      kept.forEach((a, j) => centres.set(seeds.subarray(a * d, (a + 1) * d), j * d))
      const decide = (q: Tensor) => {
        const { n: rows, v: qv } = matrix(q, 'meanShift')
        const out = new Int32Array(rows)
        for (let i = 0; i < rows; i++) {
          let best = Infinity
          for (let j = 0; j < kept.length; j++) {
            const t = sq(qv, i, centres, j, d)
            if (t < best) {
              best = t
              out[i] = j
            }
          }
        }
        return fromData(out, [rows])
      }
      const transform = (q: Tensor) => {
        const { n: rows, v: qv } = matrix(q, 'meanShift')
        return pairwiseDistances(mat(qv, rows, d), mat(centres, kept.length, d))
      }
      return { kind: 'model', name: 'mean-shift', centres: mat(centres, kept.length, d), training, decide, transform }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'dbscan',
    module: 'unsupervised/clustering',
    name: 'DBSCAN',
    summary: 'Density-connected core points form clusters; the rest is noise.',
    task: 'clustering',
    capabilities: ['decide'],
    hyper: space({
      eps: real(1e-3, 10, { default: 0.5, label: 'ε', scale: 'log' }),
      minSamples: int(1, 50, { default: 5 }),
    }),
    notes: ['density-based-spatial-clustering'],
    cite: ['ester1996'],
  },
  dbscan,
)

defineModel(
  {
    key: 'optics',
    module: 'unsupervised/clustering',
    name: 'OPTICS',
    summary: 'The reachability ordering of the points, from which DBSCAN clusterings at every ε can be read.',
    task: 'clustering',
    capabilities: [],
    transductive: true,
    hyper: space({ minSamples: int(1, 50, { default: 5 }) }),
    notes: ['density-based-spatial-clustering'],
  },
  optics,
)

defineModel(
  {
    key: 'meanShift',
    module: 'unsupervised/clustering',
    name: 'Mean shift',
    summary: 'Points climb a kernel density estimate to its modes, which are the clusters.',
    task: 'clustering',
    capabilities: ['decide', 'transform'],
    hyper: space({
      bandwidth: real(1e-2, 10, { default: 1, scale: 'log' }),
      kernel: oneOf(['flat', 'gaussian']),
      maxSteps: int(1, 1000, { default: 300 }),
    }),
    notes: ['density-based-spatial-clustering'],
  },
  meanShift,
)
