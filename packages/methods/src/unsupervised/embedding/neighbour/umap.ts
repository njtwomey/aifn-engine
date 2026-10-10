/**
 * UMAP, simplified but faithful to McInnes, Healy and Melville (2018, "UMAP: Uniform Manifold Approximation and
 * Projection for dimension reduction", arXiv:1802.03426) and umap-learn's defaults.
 *
 * The $k$ nearest neighbours of each point (itself included; exact for small $n$, by nearest-neighbour descent above
 * `DESCENT_ABOVE` rows) give a fuzzy simplicial set: local connectivity $\rho_i$ (the nearest nonzero distance),
 * memberships $w_{ij} = \exp(-\max(0, d_{ij} - \rho_i)/\sigma_i)$ with bandwidths $\sigma_i$ such that
 * $\sum_j w_{ij} = \log_2 k$ over the other neighbours, and their fuzzy union $w_{ij} + w_{ji} - w_{ij}w_{ji}$. The
 * output curve $1/(1 + a d^{2b})$ is fitted to `minDist` and `spread`, and the layout is stochastic: each epoch
 * samples edges by weight (edge $e$ every $\max_{e'} w_{e'} / w_e$ epochs), pulls their ends together and pushes the
 * edge's lower-indexed end away from `negativeSamples` random points, with a learning rate falling linearly to 0.
 *
 * Simplifications: random (not random-projection-tree) initial lists for the descent, a Laplacian-eigenmap or random
 * start, each edge stored once (umap-learn holds both directions, so both ends get negative samples), and $\sigma_i$
 * floored at $10^{-3}$ of the point's own mean neighbour distance.
 */

import type { Dataset, Estimator, FitOptions, Trained } from 'aifn-compute/learning/estimators'
import { eigh, eigsh } from 'aifn-compute/numerics/linalg'
import type { Status } from 'aifn-compute/foundation/contracts'
import { child, integers, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { nearestNeighbourDescent } from 'aifn-compute/numerics/neighbours'
import { squaredDistances } from '../neighbourhoods'
import { mat, matrix, values, vec } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The fuzzy graph of the data: per-point $\rho$ and $\sigma$, and the symmetric membership strengths. */
export interface FuzzyGraph {
  /** Distance to the nearest neighbour at a nonzero distance, 0 if none (local connectivity 1); $n$ values. */
  rho: Tensor
  /** Bandwidth found by bisection ($n$ values). */
  sigma: Tensor
  /** The $k$ nearest neighbours of each point, itself first ($n \times k$, int32). */
  neighbours: Tensor
  /** Edges ($i < j$, as parallel arrays) with their membership strength after the fuzzy union. */
  edges: { from: Int32Array; to: Int32Array; weight: Float64Array }
}

/** How `fuzzyGraph` finds neighbours: exact search, nearest-neighbour descent, or `auto` (descent above 2000 rows). */
export type NeighbourSearch = 'exact' | 'descent' | 'auto'

/** Above this many rows `auto` neighbour search uses nearest-neighbour descent (umap-learn switches at 4096). */
export const DESCENT_ABOVE = 2000

/** Options of {@link fuzzyGraph}. */
export interface FuzzyGraphOptions {
  /** How neighbours are found (default `'auto'`). */
  search?: NeighbourSearch
  /** Randomness of nearest-neighbour descent (default: a fixed stream). */
  stream?: Stream
}

/**
 * The $k$ nearest neighbours of each row, itself first, and their distances: exact (all pairwise distances, $O(n^2)$
 * memory; ties to the lower index) or by nearest-neighbour descent (`aifn-compute/numerics/neighbours`). Exact search
 * is used for `'exact'`, for `'auto'` up to `DESCENT_ABOVE` rows, and whenever $k = 1$.
 *
 * @param v The points as a row-major array of $n \times d$ values.
 * @param n The number of points.
 * @param d The number of features.
 * @param k The number of neighbours per point, itself included, at most $n$.
 * @param options The search and the stream of the descent (default the fixed stream `'nearest-neighbour-descent'`).
 * @returns `nb`, the neighbours' indices, and `dist`, their Euclidean distances, each a row-major $n \times k$ array
 *   (row $i$ in entries `i * k` to `i * k + k - 1`, nearest first).
 */
function neighbourLists(v: Float64Array, n: number, d: number, k: number, options: FuzzyGraphOptions) {
  const { search = 'auto' } = options
  const nb = new Int32Array(n * k)
  const dist = new Float64Array(n * k)
  if (search === 'exact' || (search === 'auto' && n <= DESCENT_ABOVE) || k === 1) {
    const D = Float64Array.from(squaredDistances(v, n, d), Math.sqrt)
    for (let i = 0; i < n; i++) {
      const order = Array.from({ length: n }, (_, j) => j).sort((a, b) =>
        a === i ? -1 : b === i ? 1 : D[i * n + a] - D[i * n + b] || a - b,
      )
      for (let r = 0; r < k; r++) {
        nb[i * k + r] = order[r]
        dist[i * k + r] = D[i * n + order[r]]
      }
    }
    return { nb, dist }
  }
  const found = nearestNeighbourDescent(fromData(v, [n, d]), k - 1, {
    stream: options.stream ?? stream('nearest-neighbour-descent'),
  })
  const idx = found.indices.data as Int32Array
  const dst = toFlat(found.distances)
  for (let i = 0; i < n; i++) {
    nb[i * k] = i
    for (let r = 1; r < k; r++) {
      nb[i * k + r] = idx[i * (k - 1) + r - 1]
      dist[i * k + r] = dst[i * (k - 1) + r - 1]
    }
  }
  return { nb, dist }
}

/**
 * The fuzzy simplicial set of the rows of `x` (McInnes, Healy and Melville, 2018, §3): memberships
 * $w_{ij} = \exp(-\max(0, d_{ij} - \rho_i)/\sigma_i)$ over each point's neighbours, then the fuzzy union
 * $\Wmat + \Wmat^\top - \Wmat \circ \Wmat^\top$. Each $\sigma_i$ is found by bisection from 1 (tolerance $10^{-5}$, at
 * most 64 steps) so that the memberships of the point's $k - 1$ other neighbours sum to $\log_2 k$, then floored at
 * $10^{-3}$ of their mean distance. Throws `ShapeError` when `x` is not a matrix.
 *
 * @param x The points ($n \times d$), one per row.
 * @param neighbours The number of neighbours $k$, the point itself included as in umap-learn; more than $n$ is cut to
 *   $n$.
 * @param options How neighbours are found: exact up to `DESCENT_ABOVE` rows and by nearest-neighbour descent beyond
 *   (`search: 'auto'`, default), and the descent's stream.
 * @returns The graph: $\rho$, $\sigma$, the neighbour lists and the symmetric edges.
 *
 * @example Four points on a line
 * const g = fuzzyGraph(tensor([[0], [1], [3], [6]]), 3)
 * print('rho =', g.rho, 'sigma =', g.sigma)
 * print('neighbours =', g.neighbours)
 * print('edges [i, j, weight] =', Array.from(g.edges.from, (i, e) => [i, g.edges.to[e], g.edges.weight[e]]))
 */
export function fuzzyGraph(x: Tensor, neighbours = 15, options: FuzzyGraphOptions = {}): FuzzyGraph {
  const { n, d, v } = matrix(x, 'fuzzyGraph')
  const k = Math.min(neighbours, n)
  const { nb, dist } = neighbourLists(v, n, d, k, options)
  const target = Math.log2(k)
  const rho = new Float64Array(n)
  const sigma = new Float64Array(n)
  const W = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    let first = 0
    for (let r = 1; r < k; r++) {
      const t = dist[i * k + r]
      if (t > 0) {
        first = t
        break
      }
    }
    rho[i] = first
    let lo = 0
    let hi = Infinity
    let s = 1
    for (let it = 0; it < 64; it++) {
      let psum = 0
      for (let r = 1; r < k; r++) psum += Math.exp(-Math.max(0, dist[i * k + r] - rho[i]) / s)
      if (Math.abs(psum - target) < 1e-5) break
      if (psum > target) {
        hi = s
        s = (lo + hi) / 2
      } else {
        lo = s
        s = hi === Infinity ? s * 2 : (lo + hi) / 2
      }
    }
    // umap-learn floors σ at 10⁻³ of the mean neighbour distance.
    let mean = 0
    for (let r = 1; r < k; r++) mean += dist[i * k + r] / (k - 1)
    sigma[i] = Math.max(s, 1e-3 * mean)
    for (let r = 1; r < k; r++) {
      const j = nb[i * k + r]
      W.set(i * n + j, Math.exp(-Math.max(0, dist[i * k + r] - rho[i]) / sigma[i]))
    }
  }
  const from: number[] = []
  const to: number[] = []
  const weight: number[] = []
  const seen = new Set<number>()
  for (const key of [...W.keys()].sort((a, b) => a - b)) {
    const i = Math.floor(key / n)
    const j = key % n
    const a = Math.min(i, j)
    const b = Math.max(i, j)
    if (seen.has(a * n + b)) continue
    seen.add(a * n + b)
    const wij = W.get(i * n + j) ?? 0
    const wji = W.get(j * n + i) ?? 0
    from.push(a)
    to.push(b)
    weight.push(wij + wji - wij * wji)
  }
  return {
    rho: vec(rho),
    sigma: vec(sigma),
    neighbours: fromData(nb, [n, k]),
    edges: { from: Int32Array.from(from), to: Int32Array.from(to), weight: Float64Array.from(weight) },
  }
}

/**
 * The output curve's parameters: $a$ and $b$ minimising the squared error of $1/(1 + a x^{2b})$ against 1 for
 * $x <$ `minDist` and $\exp(-(x - \text{minDist})/\text{spread})$ beyond, on 300 points of $[0, 3 \cdot \text{spread}]$
 * (umap-learn's `find_ab_params`), by Gauss-Newton with step halving from $a = b = 1$, keeping both positive.
 *
 * @param minDist The distance below which the curve should be flat at 1: how tightly points may pack.
 * @param spread The scale of the curve's exponential decay beyond `minDist`.
 * @returns The fitted $a$ and $b$.
 *
 * @example umap-learn's defaults give a = 1.577, b = 0.895
 * print('minDist 0.1:', curveParameters())
 * print('minDist 0.5:', curveParameters(0.5, 1))
 */
export function curveParameters(minDist = 0.1, spread = 1): { a: number; b: number } {
  const xs = Array.from({ length: 300 }, (_, i) => (3 * spread * i) / 299)
  const ys = xs.map((x) => (x < minDist ? 1 : Math.exp(-(x - minDist) / spread)))
  const loss = (a: number, b: number) => xs.reduce((s, x, i) => s + (1 / (1 + a * x ** (2 * b)) - ys[i]) ** 2, 0)
  let a = 1
  let b = 1
  for (let it = 0; it < 200; it++) {
    let JtJ00 = 0
    let JtJ01 = 0
    let JtJ11 = 0
    let g0 = 0
    let g1 = 0
    xs.forEach((x, i) => {
      if (x === 0) return
      const p = x ** (2 * b)
      const f = 1 / (1 + a * p)
      const r = f - ys[i]
      const da = -p * f * f
      const db = -a * p * 2 * Math.log(x) * f * f
      JtJ00 += da * da
      JtJ01 += da * db
      JtJ11 += db * db
      g0 += da * r
      g1 += db * r
    })
    const det = JtJ00 * JtJ11 - JtJ01 * JtJ01
    if (!(Math.abs(det) > 0)) break
    const s0 = (JtJ11 * g0 - JtJ01 * g1) / det
    const s1 = (JtJ00 * g1 - JtJ01 * g0) / det
    const before = loss(a, b)
    let t = 1
    while (t > 1e-8 && !(a - t * s0 > 0 && b - t * s1 > 0 && loss(a - t * s0, b - t * s1) <= before)) t /= 2
    if (t <= 1e-8) break
    a -= t * s0
    b -= t * s1
    if (Math.abs(t * s0) < 1e-12 && Math.abs(t * s1) < 1e-12) break
  }
  return { a, b }
}

/** Up to this many rows UMAP's spectral start uses a dense eigendecomposition; beyond it, thick-restart Lanczos. */
export const DENSE_SPECTRAL_UP_TO = 500

/**
 * The Laplacian eigenmap of a fuzzy graph, UMAP's spectral start: with $\Wmat$ the symmetric edge weights and $\Dmat$
 * their row sums, the eigenvectors $2, \dots,$ `dims` $+ 1$ of $\Mmat = \Dmat^{-1/2}\Wmat\Dmat^{-1/2}$ by descending
 * eigenvalue (the smallest of the normalised Laplacian $\Imat - \Mmat$, skipping the trivial $\Dmat^{1/2}\ones$),
 * each scaled to $[0, 10]$. `method` `'lanczos'` finds them with `eigsh` on the sparse product $\Mmat\vvec$
 * ($O(\text{edges})$ per product, so large $n$ stays cheap); `'dense'` builds the $n \times n$ matrix and runs `eigh`
 * ($O(n^3)$). Each eigenvector's sign is fixed so that its entry of largest magnitude is positive (before scaling), so
 * both methods give the same layout up to rounding when the eigenvalues are distinct and no two entries tie for the
 * largest magnitude (on symmetric data they can, and the two layouts may then be mirror images). Throws `DomainError`
 * unless $n >$ `dims` $+ 1$.
 *
 * @param graph The fuzzy graph, as `fuzzyGraph` returns it.
 * @param dims The dimension of the layout.
 * @param options The eigensolver: `'lanczos'` (default) or `'dense'`.
 * @returns The layout as a row-major array of $n \times$ `dims` values (a constant column is all 0).
 *
 * @example The dense and Lanczos solvers give the same start
 * const g = fuzzyGraph(normals(stream(1), [12, 2]), 5)
 * print('dense =', spectralLayout(g, 1, { method: 'dense' }))
 * print('Lanczos =', spectralLayout(g, 1, { method: 'lanczos' }))
 */
export function spectralLayout(
  graph: FuzzyGraph,
  dims = 2,
  options: { method?: 'dense' | 'lanczos' } = {},
): Float64Array {
  const n = graph.rho.shape[0]
  const { from, to, weight } = graph.edges
  const k = dims + 1
  if (!(k < n))
    throw new DomainError('spectral layout', `spectral layout: need more than ${k} rows for ${dims} dimensions`)
  const deg = new Float64Array(n)
  for (let e = 0; e < from.length; e++) {
    deg[from[e]] += weight[e]
    deg[to[e]] += weight[e]
  }
  const inv = Float64Array.from(deg, (d) => (d > 0 ? 1 / Math.sqrt(d) : 0))
  // column(c)(i): component i of the eigenvector with the (c + 2)-th largest eigenvalue of M.
  let column: (c: number) => (i: number) => number
  if ((options.method ?? 'lanczos') === 'dense') {
    const M = new Float64Array(n * n)
    for (let e = 0; e < from.length; e++)
      M[from[e] * n + to[e]] = M[to[e] * n + from[e]] = weight[e] * inv[from[e]] * inv[to[e]]
    const V = values(eigh(fromData(M, [n, n])).vectors)
    column = (c) => (i) => V[i * n + c + 1]
  } else {
    // M·v over the edge list; eigenvalues of M lie in [−1, 1] and the wanted ones are its largest.
    const product = (v: Tensor): Float64Array => {
      const x = values(v)
      const out = new Float64Array(n)
      for (let e = 0; e < from.length; e++) {
        const i = from[e]
        const j = to[e]
        const m = weight[e] * inv[i] * inv[j]
        out[i] += m * x[j]
        out[j] += m * x[i]
      }
      return out
    }
    const r = eigsh(product, n, { k, which: 'largest', tolerance: 1e-8, start: stream('umap-spectral') })
    const V = values(r.vectors)
    column = (c) => (i) => V[i * k + c + 1]
  }
  const Y = new Float64Array(n * dims)
  for (let c = 0; c < dims; c++) {
    const u = column(c)
    let lo = Infinity
    let hi = -Infinity
    let big = 0
    for (let i = 0; i < n; i++) {
      const x = u(i)
      lo = Math.min(lo, x)
      hi = Math.max(hi, x)
      if (Math.abs(x) > Math.abs(big)) big = x
    }
    // Flip so the entry of largest magnitude is positive, then scale to [0, 10].
    const flip = big < 0
    for (let i = 0; i < n; i++) Y[i * dims + c] = hi > lo ? (flip ? 10 * (hi - u(i)) : 10 * (u(i) - lo)) / (hi - lo) : 0
  }
  return Y
}

/** One epoch of UMAP's layout. */
export interface UmapState extends Status {
  /** The layout ($n \times$ `dims`). */
  embedding: Tensor
  /** Epochs done. */
  t: number
  /** The learning rate used in the epoch that produced this state (`learningRate` at the start). */
  alpha: number
  /** Edge samples taken in that epoch. */
  samples: number
}

/**
 * UMAP's stochastic layout as a traceable algorithm (McInnes, Healy and Melville, 2018, arXiv:1802.03426), one step
 * per epoch, done after `epochs`. In epoch $e$ each edge due by its weight pulls its two ends together along the
 * gradient of $\log(1 + a d^{2b})$, and its lower-indexed end is pushed from `negativeSamples` random points (the
 * step's stream), each gradient coordinate clipped to $[-4, 4]$ and scaled by the learning rate
 * $\text{learningRate} \cdot (1 - (e - 1)/\text{epochs})$. Edges lighter than $\max_e w_e /$ `epochs` are never
 * sampled. `init` takes an embedding, or a spectral start (`start: 'spectral'`, default; `spectralLayout`, dense up to
 * `DENSE_SPECTRAL_UP_TO` rows) or a uniform random one in $[-10, 10]^{\text{dims}}$ from the run's stream
 * (`start: 'random'`, also used when $n \le$ `dims` $+ 1$).
 *
 * @param graph The fuzzy graph to lay out, as `fuzzyGraph` returns it.
 * @param params The settings of the layout.
 * @param params.dims The dimension of the layout (default 2).
 * @param params.epochs The number of epochs, which also sets the learning-rate schedule and the lightest edge sampled
 *   (default 200).
 * @param params.minDist How tightly points may pack (default 0.1), with `spread` setting the curve's $a$ and $b$.
 * @param params.spread The scale of the curve's decay (default 1).
 * @param params.negativeSamples The random points each sampled edge's end is pushed from (default 5).
 * @param params.learningRate The initial learning rate (default 1).
 * @returns The algorithm, for `run` or `trace`; its states are `UmapState`s.
 *
 * @example The learning rate falls to almost nothing over the epochs
 * const x = concat([normals(stream(1), [10, 3]), add(normals(stream(2), [10, 3]), 10)], 0)
 * const steps = umapSteps(fuzzyGraph(x, 5), { epochs: 50 })
 * const first = run(steps, { start: 'random' }, 1, { stream: stream(3) })
 * const last = run(steps, { start: 'random' }, 50, { stream: stream(3) })
 * print('epoch 1: learning rate', first.alpha, 'edges sampled', first.samples)
 * print('epoch 50: learning rate', last.alpha, 'edges sampled', last.samples)
 */
export function umapSteps(
  graph: FuzzyGraph,
  params: {
    dims?: number
    epochs?: number
    minDist?: number
    spread?: number
    negativeSamples?: number
    learningRate?: number
  } = {},
): Algorithm<{ embedding?: Tensor; start?: 'spectral' | 'random' }, UmapState> {
  const { dims = 2, epochs = 200, minDist = 0.1, spread = 1, negativeSamples = 5, learningRate = 1 } = params
  const { a, b } = curveParameters(minDist, spread)
  const n = graph.rho.shape[0]
  const { from, to, weight } = graph.edges
  let wmax = 0
  for (const w of weight) wmax = Math.max(wmax, w)
  // Edges below wmax/epochs are never sampled (umap-learn drops them).
  const period = Float64Array.from(weight, (w) => (w >= wmax / epochs ? wmax / w : Infinity))
  const clip = (g: number) => Math.max(-4, Math.min(4, g))
  return {
    name: 'umap-layout',
    init: ({ embedding, start = 'spectral' } = {}, s) => {
      let Y: Float64Array
      if (embedding) Y = Float64Array.from(values(embedding))
      else if (start === 'random' || n <= dims + 1) {
        const r = child(s, 'layout')
        Y = Float64Array.from({ length: n * dims }, () => 20 * uniform(r) - 10)
      } else Y = spectralLayout(graph, dims, { method: n > DENSE_SPECTRAL_UP_TO ? 'lanczos' : 'dense' })
      return { embedding: mat(Y, n, dims), t: 0, alpha: learningRate, samples: 0 }
    },
    step: (state, ctx) => {
      const Y = Float64Array.from(values(state.embedding))
      const e = state.t + 1
      const alpha = learningRate * (1 - state.t / epochs)
      const r = ctx.stream
      let samples = 0
      for (let k = 0; k < from.length; k++) {
        // Edge k is sampled in the epochs e where ⌊e / period⌋ increases.
        if (!Number.isFinite(period[k]) || Math.floor(e / period[k]) === Math.floor((e - 1) / period[k])) continue
        samples++
        const i = from[k]
        const j = to[k]
        let d2 = 0
        for (let c = 0; c < dims; c++) d2 += (Y[i * dims + c] - Y[j * dims + c]) ** 2
        const attract = d2 > 0 ? (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b) : 0
        for (let c = 0; c < dims; c++) {
          const g = clip(attract * (Y[i * dims + c] - Y[j * dims + c]))
          Y[i * dims + c] += alpha * g
          Y[j * dims + c] -= alpha * g
        }
        for (const u of [i, j]) {
          for (let s = 0; s < negativeSamples; s++) {
            const m = integers(r, n)
            if (m === u) continue
            let q2 = 0
            for (let c = 0; c < dims; c++) q2 += (Y[u * dims + c] - Y[m * dims + c]) ** 2
            const repel = (2 * b) / ((0.001 + q2) * (1 + a * q2 ** b))
            for (let c = 0; c < dims; c++)
              Y[u * dims + c] += alpha * (q2 > 0 ? clip(repel * (Y[u * dims + c] - Y[m * dims + c])) : 4)
          }
        }
      }
      return { embedding: mat(Y, n, dims), t: e, alpha, samples }
    },
    done: (state) => state.t >= epochs,
  }
}

/** A fitted UMAP embedding, with its run (`training`). */
export interface UmapModel extends Trained<UmapState> {
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** This UMAP places the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'umap'
  /** The coordinates of the training rows ($n \times$ `dims`). */
  readonly embedding: Tensor
  /** The fuzzy graph that was laid out. */
  readonly graph: FuzzyGraph
  /** The output curve's $a$, from `curveParameters`. */
  readonly a: number
  /** The output curve's $b$. */
  readonly b: number
}

/**
 * UMAP of the training rows (McInnes, Healy and Melville, 2018): the fuzzy graph of `fuzzyGraph`, laid out by
 * `umapSteps` for `epochs` epochs. The fit options' `stream` drives the layout's negative samples and random start,
 * and a child of it the neighbour descent; the run is traced every 5 epochs (or every `trace.every` of the fit
 * options).
 *
 * @param params The settings of the estimator.
 * @param params.neighbours The number of neighbours $k$, each point itself included (default 15).
 * @param params.dims The dimension of the embedding (default 2).
 * @param params.epochs The number of layout epochs (default 200).
 * @param params.minDist How tightly points may pack (default 0.1).
 * @param params.spread The scale of the output curve's decay (default 1).
 * @param params.negativeSamples The random points each sampled edge's end is pushed from (default 5).
 * @param params.start The initial layout: `'spectral'` (default) or `'random'`.
 * @param params.search How neighbours are found (default `'auto'`).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `UmapModel`.
 *
 * @example Two well-separated blobs stay apart
 * // Two blobs of 10 points in three dimensions, their centres 17 apart.
 * const x = concat([normals(stream(1), [10, 3]), add(normals(stream(2), [10, 3]), 10)], 0)
 * const model = umap({ neighbours: 5, epochs: 50 }).fit({ x }, { stream: stream(3) })
 * const Y = toArray(model.embedding)
 * const centre = (rows) => [0, 1].map((c) => rows.reduce((s, r) => s + r[c], 0) / rows.length)
 * const blobs = [Y.slice(0, 10), Y.slice(10)]
 * const [a, b] = blobs.map(centre)
 * const radius = (rows, m) => Math.max(...rows.map((r) => Math.hypot(r[0] - m[0], r[1] - m[1])))
 * print('distance between the blob centres =', Math.hypot(a[0] - b[0], a[1] - b[1]))
 * print('largest distance of a point from its centre =', Math.max(radius(blobs[0], a), radius(blobs[1], b)))
 */
export function umap(
  params: {
    neighbours?: number
    dims?: number
    epochs?: number
    minDist?: number
    spread?: number
    negativeSamples?: number
    start?: 'spectral' | 'random'
    /** Neighbour search (default 'auto': exact up to `DESCENT_ABOVE` rows, nearest-neighbour descent beyond). */
    search?: NeighbourSearch
  } = {},
): Estimator<Dataset<Tensor>, UmapModel> {
  const {
    neighbours = 15,
    epochs = 200,
    start = 'spectral',
    minDist = 0.1,
    spread = 1,
    search = 'auto',
    ...rest
  } = params
  return {
    name: 'umap',
    params: { neighbours, epochs, start, minDist, spread, search, ...rest },
    fit({ x }, options: FitOptions = {}) {
      const graph = fuzzyGraph(x, neighbours, {
        search,
        ...(options.stream ? { stream: child(options.stream, 'neighbours') } : {}),
      })
      const training = trace(umapSteps(graph, { epochs, minDist, spread, ...rest }), { start }, epochs, {
        stream: options.stream,
        every: options.trace?.every ?? 5,
      })
      const final = training.final
      const { a, b } = curveParameters(minDist, spread)
      return { kind: 'model', transductive: true, name: 'umap', embedding: final.embedding, graph, a, b, training }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'umap',
    module: 'unsupervised/embedding/neighbour',
    name: 'UMAP',
    summary: 'A fuzzy nearest-neighbour graph laid out by stochastic gradient descent on a cross-entropy.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({
      neighbours: int(2, 100, { default: 15 }),
      dims: int(1, 3, { default: 2 }),
      epochs: int(1, 2000, { default: 200 }),
      minDist: real(0, 1, { default: 0.1 }),
      spread: real(0.1, 5, { default: 1 }),
      start: oneOf(['spectral', 'random']),
      search: oneOf(['auto', 'exact', 'descent']),
    }),
    notes: ['uniform-manifold-approximation-and-projection'],
    cite: ['mcinnes2018'],
  },
  umap,
)
