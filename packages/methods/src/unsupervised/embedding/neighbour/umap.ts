/**
 * UMAP, simplified but faithful to McInnes, Healy and Melville (2018, "UMAP: Uniform Manifold Approximation and
 * Projection for dimension reduction", arXiv:1802.03426) and umap-learn's defaults: k-nearest neighbours (exact for
 * small n, by nearest-neighbour descent above `DESCENT_ABOVE` rows), a
 * fuzzy simplicial set (local connectivity ρᵢ, bandwidths σᵢ with Σⱼ exp(−(dᵢⱼ − ρᵢ)/σᵢ) = log₂ k, fuzzy union
 * w + wᵀ − w∘wᵀ), the output curve 1/(1 + a d^(2b)) fitted to `minDist` and `spread`, and a stochastic layout: each
 * epoch samples edges by weight (edge e every 1/wₑ·max w epochs), pulls their ends together and pushes each end away
 * from `negativeSamples` random points, with a learning rate falling linearly to 0.
 *
 * Simplifications: random (not random-projection-tree) initial lists for the descent, and a Laplacian-eigenmap or
 * random start.
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

/** The fuzzy graph of the data: per-point ρ and σ, and the symmetric membership strengths. */
export interface FuzzyGraph {
  /** Distance to the nearest neighbour (local connectivity 1) [n]. */
  rho: Tensor
  /** Bandwidth found by bisection [n]. */
  sigma: Tensor
  /** The k nearest neighbours of each point, itself first [n, k] (int32). */
  neighbours: Tensor
  /** Edges (i < j) with their membership strength after the fuzzy union. */
  edges: { from: Int32Array; to: Int32Array; weight: Float64Array }
}

/** How `fuzzyGraph` finds neighbours: exact search, nearest-neighbour descent, or `auto` (descent above 2000 rows). */
export type NeighbourSearch = 'exact' | 'descent' | 'auto'

/** Above this many rows `auto` neighbour search uses nearest-neighbour descent (umap-learn switches at 4096). */
export const DESCENT_ABOVE = 2000

/** Options of {@link fuzzyGraph}. */
export interface FuzzyGraphOptions {
  /** Default 'auto'. */
  search?: NeighbourSearch
  /** Randomness of nearest-neighbour descent (default: a fixed stream). */
  stream?: Stream
}

/**
 * The k nearest neighbours of each row, itself first, and their distances, flat [n·k]: exact (all pairwise distances,
 * O(n²) memory) or by nearest-neighbour descent.
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
 * The fuzzy simplicial set of the rows of x with `neighbours` k (default 15, itself included, as umap-learn):
 * wᵢⱼ = exp(−max(0, dᵢⱼ − ρᵢ)/σᵢ), then w + wᵀ − w∘wᵀ. Neighbours are exact up to `DESCENT_ABOVE` rows and found by
 * nearest-neighbour descent beyond (`options.search`).
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
 * The output curve's parameters: a and b minimising the squared error of 1/(1 + a x^(2b)) against 1 for x < minDist
 * and exp(−(x − minDist)/spread) beyond, on 300 points of [0, 3·spread] (umap-learn's `find_ab_params`), by
 * Gauss–Newton with step halving.
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
 * The Laplacian eigenmap of a fuzzy graph, UMAP's spectral start: with W the symmetric edge weights and D their row
 * sums, the eigenvectors 2 … dims + 1 of M = D^(−½) W D^(−½) by descending eigenvalue (the smallest of the normalised
 * Laplacian I − M, skipping the trivial D^(½)1), each scaled to [0, 10], as rows [n · dims]. `method` 'lanczos' finds
 * them with `eigsh` on the sparse product M·v (O(edges) per product, so large n stays cheap); 'dense' builds the n × n
 * matrix and runs `eigh` (O(n³)). Each eigenvector's sign is fixed so that its entry of largest magnitude is positive,
 * so both methods give the same layout up to rounding when the eigenvalues are distinct.
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
  embedding: Tensor
  /** Epochs done. */
  t: number
  /** The learning rate used in the epoch that produced this state. */
  alpha: number
  /** Edge samples taken in that epoch. */
  samples: number
}

/**
 * UMAP's stochastic layout as a traceable algorithm (McInnes, Healy and Melville, 2018, arXiv:1802.03426; one step =
 * one epoch, `epochs` default 200). Each epoch's negative samples come from the step's stream. `init` takes an
 * embedding, or a spectral start (default: the Laplacian eigenmap of the fuzzy graph, scaled to [0, 10]) or a uniform
 * random one in [−10, 10]² from the `init` stream.
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
        for (let s = 0; s < negativeSamples; s++) {
          const m = integers(r, n)
          if (m === i) continue
          let q2 = 0
          for (let c = 0; c < dims; c++) q2 += (Y[i * dims + c] - Y[m * dims + c]) ** 2
          const repel = (2 * b) / ((0.001 + q2) * (1 + a * q2 ** b))
          for (let c = 0; c < dims; c++)
            Y[i * dims + c] += alpha * (q2 > 0 ? clip(repel * (Y[i * dims + c] - Y[m * dims + c])) : 4)
        }
      }
      return { embedding: mat(Y, n, dims), t: e, alpha, samples }
    },
    done: (state) => state.t >= epochs,
  }
}

/** A fitted UMAP embedding. */
export interface UmapModel extends Trained<UmapState> {
  readonly kind: 'model'
  /** This UMAP places the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'umap'
  readonly embedding: Tensor
  readonly graph: FuzzyGraph
  readonly a: number
  readonly b: number
}

/** UMAP of the rows of x (see `fuzzyGraph`, `umapSteps`). */
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
