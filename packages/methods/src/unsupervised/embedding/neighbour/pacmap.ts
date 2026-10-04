/**
 * PaCMAP, lite (Wang, Huang, Rudin and Shaposhnik, 2021): an embedding fitted on three kinds of pairs with a loss whose
 * weights change in three phases.
 *
 * - Neighbour pairs: each row's `neighbours` nearest rows under the scaled distance d²ᵢⱼ/(σᵢσⱼ), σᵢ the mean distance
 *   to the 4th–6th nearest rows, chosen from its `neighbours` + 50 Euclidean nearest (exact search: the "lite" part).
 * - Mid-near pairs: for each row, ⌊ratio·neighbours⌉ times, the second nearest of six random rows.
 * - Further pairs: random rows that are not neighbours.
 *
 * With d̃ = 1 + ‖yᵢ − yⱼ‖², the loss is w_NB Σ d̃/(10 + d̃) + w_MN Σ d̃/(10⁴ + d̃) + w_FP Σ 1/(1 + d̃), minimised by
 * Adam (step 1) from a PCA start scaled by 0.01. The weights (w_NB, w_MN, w_FP) go from (2, 1000 → 3, 1) over the
 * first 100 iterations to (3, 3, 1) for the next 100 and (1, 0, 1) for the rest, so mid-near pairs first fix the
 * global layout and neighbour pairs then refine the local one.
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

/** The three pair sets of PaCMAP, as flat [i, j] index lists. */
export interface PacmapPairs {
  neighbour: Int32Array
  midNear: Int32Array
  further: Int32Array
}

/** Build PaCMAP's pairs for the rows of x (see the module comment); random choices draw from `s`. */
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

/** The weights (w_NB, w_MN, w_FP) at iteration t of `iterations` (phases at 100 and 200, scaled when shorter). */
export function pacmapWeights(t: number, iterations = 450): { neighbour: number; midNear: number; further: number } {
  const unit = Math.min(100, iterations / 4.5)
  if (t < unit) return { neighbour: 2, midNear: 1000 * (1 - t / unit) + 3 * (t / unit), further: 1 }
  if (t < 2 * unit) return { neighbour: 3, midNear: 3, further: 1 }
  return { neighbour: 1, midNear: 0, further: 1 }
}

/** A state of `pacmapSteps`. */
export interface PacmapState extends Status {
  t: number
  embedding: Tensor
  /** The loss at the start of the step, with that step's weights. */
  loss: number
  /** Adam's state. */
  optimiser: unknown
}

/**
 * PaCMAP's optimisation as a step-through algorithm (one Adam step per step, `iterations` default 450) on fixed
 * `pairs` for the rows of x; the start is x's top principal axes scaled by 0.01, or a given embedding.
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

/** A fitted PaCMAP embedding. */
export interface PacmapModel extends Trained<PacmapState> {
  readonly kind: 'model'
  /** PaCMAP places the training rows only (no out-of-sample transform). */
  readonly transductive: true
  readonly name: 'pacmap'
  readonly embedding: Tensor
  readonly pairs: PacmapPairs
}

/** PaCMAP of the rows of x (see `pacmapPairs`, `pacmapSteps`). */
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
