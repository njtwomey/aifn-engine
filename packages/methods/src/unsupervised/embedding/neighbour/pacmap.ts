/**
 * PaCMAP, lite (Wang, Huang, Rudin and Shaposhnik, 2021): an embedding fitted on three kinds of pairs with a loss whose
 * weights change in three phases.
 *
 * - Neighbour pairs: each row's `neighbours` nearest rows under the scaled distance $d_{ij}^2/(\sigma_i\sigma_j)$,
 *   $\sigma_i$ the mean distance to the 4th to 6th nearest rows, chosen from its `neighbours` $+ 50$ Euclidean nearest
 *   (exact search: the "lite" part).
 * - Mid-near pairs: for each row, round(`midNearRatio` $\cdot$ `neighbours`) times, the second nearest of six random
 *   rows.
 * - Further pairs: round(`furtherRatio` $\cdot$ `neighbours`) random rows per row that are not its neighbours.
 *
 * With $\tilde{d}_{ij} = 1 + \lVert \yvec_i - \yvec_j \rVert^2$, the loss is the sum of
 * $w_{\text{NB}} \sum \tilde{d}_{ij}/(10 + \tilde{d}_{ij})$ over neighbour pairs,
 * $w_{\text{MN}} \sum \tilde{d}_{ij}/(10^4 + \tilde{d}_{ij})$ over mid-near pairs and
 * $w_{\text{FP}} \sum 1/(1 + \tilde{d}_{ij})$ over further pairs, minimised by Adam (step 1) from a PCA start scaled
 * by 0.01. The weights $(w_{\text{NB}}, w_{\text{MN}}, w_{\text{FP}})$ go from $(2, 1000, 1)$ to $(2, 3, 1)$ over the
 * first 100 iterations, are $(3, 3, 1)$ for the next 100 and $(1, 0, 1)$ for the rest (the phases shrink for runs
 * shorter than 450), so mid-near pairs first fix the global layout and neighbour pairs then refine the local one.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { child, integers, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Dataset, Estimator, FitOptions, Trained } from 'aifn-compute/learning/estimators'
import { defineModel } from 'aifn-compute/learning/estimators'
import { eigh } from 'aifn-compute/numerics/linalg'
import { adamRule, applyUpdates } from 'aifn-compute/optim/first-order'
import { int, real, space } from 'aifn-compute/foundation/space'
import { squaredDistances } from '../neighbourhoods'
import { mat, matrix, values } from '../util'

/**
 * The three pair sets of PaCMAP, each a flat list of index pairs: pair $p$ is (entry $2p$, entry $2p + 1$), and the
 * pairs of row $i$ are consecutive.
 */
export interface PacmapPairs {
  /** Each row with its nearest rows under the scaled distance. */
  neighbour: Int32Array
  /** Each row with rows a little further out, for the global layout. */
  midNear: Int32Array
  /** Each row with random non-neighbours, pushed apart. */
  further: Int32Array
}

/**
 * Build PaCMAP's pairs for the rows of `x` (see the file comment), with exact distances ($O(n^2)$). The number of
 * neighbours is capped at $n - 1$, and a row gets fewer further pairs when $50$ tries per pair do not find them.
 * Throws `ShapeError` when `x` is not a matrix.
 *
 * @param x The points ($n \times d$), one per row.
 * @param s The stream the mid-near and further pairs are drawn from.
 * @param params The numbers of pairs per row.
 * @param params.neighbours The neighbour pairs per row, $k$ (default 10).
 * @param params.midNearRatio Mid-near pairs per row, as a multiple of $k$ (default 0.5).
 * @param params.furtherRatio Further pairs per row, as a multiple of $k$ (default 2).
 * @returns The three pair lists.
 *
 * @example Eight points on a line, two neighbours each
 * const x = tensor([[0], [1], [2], [3], [4], [5], [6], [7]])
 * const pairs = pacmapPairs(x, stream(1), { neighbours: 2 })
 * print('neighbour pairs of rows 0 and 3:', pairs.neighbour.slice(0, 4), pairs.neighbour.slice(12, 16))
 * print('pairs of each kind:', pairs.neighbour.length / 2, pairs.midNear.length / 2, pairs.further.length / 2)
 */
export function pacmapPairs(
  x: Tensor,
  s: Stream,
  params: { neighbours?: number; midNearRatio?: number; furtherRatio?: number } = {},
): PacmapPairs {
  const { n, d, v } = matrix(x, 'pacmapPairs')
  const { neighbours: k0 = 10, midNearRatio = 0.5, furtherRatio = 2 } = params
  const k = Math.min(k0, n - 1)
  const D2 = squaredDistances(v, n, d)
  const order = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D2[i * n + a] - D2[i * n + b] || a - b),
  )
  const sigma = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const picks = order[i].slice(3, 6)
    let m = 0
    for (const j of picks) m += Math.sqrt(D2[i * n + j]) / picks.length
    sigma[i] = Math.max(m, 1e-10)
  }
  const nb: number[] = []
  const isNeighbour = new Set<number>()
  for (let i = 0; i < n; i++) {
    const candidates = order[i].slice(0, Math.min(n - 1, k + 50))
    candidates.sort((a, b) => D2[i * n + a] / (sigma[i] * sigma[a]) - D2[i * n + b] / (sigma[i] * sigma[b]) || a - b)
    for (const j of candidates.slice(0, k)) {
      nb.push(i, j)
      isNeighbour.add(i * n + j)
    }
  }
  const mn: number[] = []
  const nMn = Math.round(midNearRatio * k)
  for (let i = 0; i < n; i++)
    for (let t = 0; t < nMn; t++) {
      const sample: number[] = []
      while (sample.length < Math.min(6, n - 1)) {
        const j = integers(s, n)
        if (j !== i && !sample.includes(j)) sample.push(j)
      }
      sample.sort((a, b) => D2[i * n + a] - D2[i * n + b] || a - b)
      mn.push(i, sample[Math.min(1, sample.length - 1)])
    }
  const fp: number[] = []
  const nFp = Math.round(furtherRatio * k)
  for (let i = 0; i < n; i++) {
    let found = 0
    for (let tries = 0; found < nFp && tries < 50 * nFp; tries++) {
      const j = integers(s, n)
      if (j === i || isNeighbour.has(i * n + j)) continue
      fp.push(i, j)
      found++
    }
  }
  return { neighbour: Int32Array.from(nb), midNear: Int32Array.from(mn), further: Int32Array.from(fp) }
}

/**
 * The loss weights $(w_{\text{NB}}, w_{\text{MN}}, w_{\text{FP}})$ at an iteration: $w_{\text{MN}}$ falls linearly from
 * 1000 to 3 in the first phase, with $w_{\text{NB}} = 2$; then $(3, 3, 1)$; then $(1, 0, 1)$. The phases change at
 * iterations 100 and 200, or at $\text{iterations}/4.5$ and twice that when the run is shorter than 450.
 *
 * @param t The iteration, from 0.
 * @param iterations The length of the run, which scales the phases when below 450.
 * @returns The weights of the neighbour, mid-near and further pairs.
 *
 * @example The three phases of a default run
 * for (const t of [0, 50, 100, 200]) {
 *   const w = pacmapWeights(t)
 *   print('iteration', t, ':', w.neighbour, w.midNear, w.further)
 * }
 */
export function pacmapWeights(t: number, iterations = 450): { neighbour: number; midNear: number; further: number } {
  const unit = Math.min(100, iterations / 4.5)
  if (t < unit) return { neighbour: 2, midNear: 1000 * (1 - t / unit) + 3 * (t / unit), further: 1 }
  if (t < 2 * unit) return { neighbour: 3, midNear: 3, further: 1 }
  return { neighbour: 1, midNear: 0, further: 1 }
}

/** A state of `pacmapSteps`. */
export interface PacmapState extends Status {
  /** Adam steps done. */
  t: number
  /** The embedding ($n \times$ `dims`). */
  embedding: Tensor
  /**
   * The loss of the embedding the step that produced this state started from, with that step's weights (at the start,
   * the initial embedding's).
   */
  loss: number
  /** Adam's state. */
  optimiser: unknown
}

/**
 * PaCMAP's optimisation as a step-through algorithm on fixed pairs: one Adam step (moments 0.9 and 0.999,
 * $\epsilon = 10^{-7}$) per step with the weights of `pacmapWeights`, done after `iterations`. The start is the rows'
 * projections onto their top `dims` principal axes, scaled by 0.01 (zero beyond $d$ axes), or a given `embedding`; no
 * randomness is used. Throws `ShapeError` when `x` is not a matrix.
 *
 * @param x The points ($n \times d$), one per row; used for the start only.
 * @param pairs The pairs to fit, as `pacmapPairs` returns them for `x`.
 * @param params The settings of the optimisation.
 * @param params.dims The dimension of the embedding (default 2).
 * @param params.iterations The length of the run, which also places the phases (default 450).
 * @param params.learningRate Adam's step size (default 1).
 * @returns The algorithm, for `run` or `trace`; its states are `PacmapState`s.
 *
 * @example In the last phase the loss falls
 * const x = concat([normals(stream(1), [10, 3]), add(normals(stream(2), [10, 3]), 10)], 0)
 * const steps = pacmapSteps(x, pacmapPairs(x, stream(3), { neighbours: 5 }), { iterations: 100 })
 * print('loss at step 50 =', run(steps, undefined, 50).loss)
 * print('loss at step 100 =', run(steps, undefined, 100).loss)
 */
export function pacmapSteps(
  x: Tensor,
  pairs: PacmapPairs,
  params: { dims?: number; iterations?: number; learningRate?: number } = {},
): Algorithm<{ embedding?: Tensor } | void, PacmapState> {
  const { n, d, v } = matrix(x, 'pacmapSteps')
  const { dims = 2, iterations = 450, learningRate = 1 } = params
  const rule = adamRule({ stepSize: learningRate, beta1: 0.9, beta2: 0.999, epsilon: 1e-7 })
  const lossAndGrad = (Y: Float64Array, t: number) => {
    const w = pacmapWeights(t, iterations)
    const G = new Float64Array(n * dims)
    let loss = 0
    const add = (list: Int32Array, weight: number, kind: 'nb' | 'mn' | 'fp') => {
      if (weight === 0) return
      for (let p = 0; p < list.length; p += 2) {
        const i = list[p]
        const j = list[p + 1]
        let q = 1
        for (let c = 0; c < dims; c++) q += (Y[i * dims + c] - Y[j * dims + c]) ** 2
        let dLdq: number
        if (kind === 'nb') {
          loss += (weight * q) / (10 + q)
          dLdq = (weight * 10) / (10 + q) ** 2
        } else if (kind === 'mn') {
          loss += (weight * q) / (10000 + q)
          dLdq = (weight * 10000) / (10000 + q) ** 2
        } else {
          loss += weight / (1 + q)
          dLdq = -weight / (1 + q) ** 2
        }
        for (let c = 0; c < dims; c++) {
          const g = 2 * dLdq * (Y[i * dims + c] - Y[j * dims + c])
          G[i * dims + c] += g
          G[j * dims + c] -= g
        }
      }
    }
    add(pairs.neighbour, w.neighbour, 'nb')
    add(pairs.midNear, w.midNear, 'mn')
    add(pairs.further, w.further, 'fp')
    return { loss, G }
  }
  return {
    name: 'pacmap',
    init: (input) => {
      let Y: Float64Array
      if (input && input.embedding) Y = Float64Array.from(values(input.embedding))
      else {
        const mean = new Float64Array(d)
        for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) mean[c] += v[i * d + c] / n
        const S = new Float64Array(d * d)
        for (let i = 0; i < n; i++)
          for (let a = 0; a < d; a++)
            for (let b = 0; b < d; b++) S[a * d + b] += ((v[i * d + a] - mean[a]) * (v[i * d + b] - mean[b])) / n
        const V = values(eigh(mat(S, d, d)).vectors)
        Y = new Float64Array(n * dims)
        for (let i = 0; i < n; i++)
          for (let c = 0; c < Math.min(dims, d); c++) {
            let s = 0
            for (let a = 0; a < d; a++) s += (v[i * d + a] - mean[a]) * V[a * d + c]
            Y[i * dims + c] = 0.01 * s
          }
      }
      const embedding = mat(Y, n, dims)
      return { t: 0, embedding, loss: lossAndGrad(Y, 0).loss, optimiser: rule.init(embedding) }
    },
    step: (st) => {
      const Y = values(st.embedding)
      const { loss, G } = lossAndGrad(Y, st.t)
      const { updates, state } = rule.update(fromData(G, [n, dims]), st.optimiser as never, st.embedding)
      const embedding = applyUpdates(st.embedding, updates) as Tensor
      return { t: st.t + 1, embedding, loss, optimiser: state, diverged: !Number.isFinite(loss) }
    },
    done: (st) => st.t >= iterations,
  }
}

/** A fitted PaCMAP embedding, with its run (`training`). */
export interface PacmapModel extends Trained<PacmapState> {
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** PaCMAP places the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'pacmap'
  /** The coordinates of the training rows ($n \times$ `dims`). */
  readonly embedding: Tensor
  /** The pairs the embedding was fitted on. */
  readonly pairs: PacmapPairs
}

/**
 * PaCMAP of the training rows (Wang, Huang, Rudin and Shaposhnik, 2021): the pairs of `pacmapPairs`, drawn from a
 * child of the fit options' `stream` (or a fixed stream), fitted by `pacmapSteps`. The run is traced every 5
 * iterations (or every `trace.every` of the fit options).
 *
 * @param params The settings of the estimator.
 * @param params.neighbours The neighbour pairs per row, $k$ (default 10).
 * @param params.midNearRatio Mid-near pairs per row, as a multiple of $k$ (default 0.5).
 * @param params.furtherRatio Further pairs per row, as a multiple of $k$ (default 2).
 * @param params.dims The dimension of the embedding (default 2).
 * @param params.iterations The number of Adam steps (default 450).
 * @param params.learningRate Adam's step size (default 1).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `PacmapModel`.
 *
 * @example Two well-separated blobs stay apart
 * // Two blobs of 10 points in three dimensions, their centres 17 apart.
 * const x = concat([normals(stream(1), [10, 3]), add(normals(stream(2), [10, 3]), 10)], 0)
 * const model = pacmap({ neighbours: 5, iterations: 100 }).fit({ x }, { stream: stream(3) })
 * const Y = toArray(model.embedding)
 * const centre = (rows) => [0, 1].map((c) => rows.reduce((s, r) => s + r[c], 0) / rows.length)
 * const blobs = [Y.slice(0, 10), Y.slice(10)]
 * const [a, b] = blobs.map(centre)
 * const radius = (rows, m) => Math.max(...rows.map((r) => Math.hypot(r[0] - m[0], r[1] - m[1])))
 * print('distance between the blob centres =', Math.hypot(a[0] - b[0], a[1] - b[1]))
 * print('largest distance of a point from its centre =', Math.max(radius(blobs[0], a), radius(blobs[1], b)))
 */
export function pacmap(
  params: {
    neighbours?: number
    midNearRatio?: number
    furtherRatio?: number
    dims?: number
    iterations?: number
    learningRate?: number
  } = {},
): Estimator<Dataset<Tensor>, PacmapModel> {
  const { neighbours = 10, midNearRatio = 0.5, furtherRatio = 2, iterations = 450, ...rest } = params
  return {
    name: 'pacmap',
    params: { neighbours, midNearRatio, furtherRatio, iterations, ...rest },
    fit({ x }, options: FitOptions = {}) {
      const s = options.stream
      const pairs = pacmapPairs(x, s ? child(s, 'pairs') : stream('pacmap'), { neighbours, midNearRatio, furtherRatio })
      const training = trace(pacmapSteps(x, pairs, { iterations, ...rest }), undefined, iterations, {
        stream: s,
        every: options.trace?.every ?? 5,
        record: { loss: (st) => st.loss },
      })
      return {
        kind: 'model',
        transductive: true,
        name: 'pacmap',
        embedding: training.final.embedding,
        pairs,
        training,
      }
    },
  }
}

defineModel(
  {
    key: 'pacmap',
    module: 'unsupervised/embedding/neighbour',
    name: 'PaCMAP',
    summary: 'Neighbour, mid-near and further pairs with phased loss weights, so the layout keeps global structure.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({
      neighbours: int(2, 50, { default: 10 }),
      midNearRatio: real(0, 2, { default: 0.5 }),
      furtherRatio: real(0, 5, { default: 2 }),
      iterations: int(10, 2000, { default: 450 }),
    }),
    notes: ['pacmap-trimap-and-largevis'],
    cite: ['wang2021b'],
  },
  pacmap,
)
