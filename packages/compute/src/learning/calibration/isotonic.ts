/**
 * Isotonic regression by the pool-adjacent-violators algorithm (Ayer et al., 1955; Best and Chakravarti, 1990), as a
 * traceable algorithm: the least-squares fit ŷ to y, weighted by w, subject to ŷ₁ ≤ ŷ₂ ≤ … ≤ ŷₙ (or ≥ for a
 * decreasing fit). The solution is piecewise constant: each block of pooled points takes the weighted mean of its y.
 */

import type { Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'

type F64 = Float64Array

/** Options of `poolAdjacentViolatorsSteps` and `isotonicRegression`. */
export type IsotonicOptions = {
  /** Non-negative weights (default all 1). */
  weights?: VectorLike
  /** Fit a non-decreasing (default) or non-increasing sequence. */
  increasing?: boolean
}

/** What the last step of `poolAdjacentViolatorsSteps` did. */
export type PavEvent = 'start' | 'add' | 'pool' | 'done'

/** One state of `poolAdjacentViolatorsSteps`. */
export interface PavState extends Status {
  /** The index of the first point of each block, in order. */
  starts: Tensor
  /** The weighted mean of y over each block. */
  values: Tensor
  /** The total weight of each block. */
  blockWeights: Tensor
  /** The points added so far: 0 … next − 1 are in blocks. */
  next: number
  /** The current fit of the points added so far: each point takes its block's value (length n; NaN beyond `next`). */
  fit: Tensor
  /** The weighted sum of squares Σ wᵢ(yᵢ − ŷᵢ)² over the points added so far. */
  sse: number
  event: PavEvent
  done: boolean
}

/**
 * Pool adjacent violators as a traceable algorithm on a sequence y (ordered by its covariate). Each step either adds
 * the next point as a block of its own (`add`) or, when the last two blocks violate the order, pools them into one
 * block with their weighted mean (`pool`). Pooling only merges a violating pair, so the blocks are always in order
 * up to the last one, and the fit after the last point is the isotonic regression. Each point is added once and each pool
 * removes a block, so it finishes in at most 2n − 1 steps.
 */
export function poolAdjacentViolatorsSteps(y: VectorLike, options: IsotonicOptions = {}): Algorithm<void, PavState> {
  const where = 'poolAdjacentViolatorsSteps'
  const ys = dense.toF64(y, where)
  const n = ys.length
  const ws = options.weights === undefined ? new Float64Array(n).fill(1) : dense.toF64(options.weights, where)
  if (ws.length !== n) throw new DomainError(where, `${where}: ${ws.length} weights for ${n} values`)
  if (ws.some((w) => !(w >= 0))) throw new DomainError(where, `${where}: weights must be non-negative`)
  // A decreasing fit is the increasing fit of −y, negated.
  const sign = options.increasing === false ? -1 : 1
  const violates = (values: F64, k: number) => sign * values[k - 1] > sign * values[k]
  const state = (t: number, starts: number[], values: number[], weights: number[], next: number, event: PavEvent) => {
    const fit = new Float64Array(n).fill(NaN)
    let sse = 0
    for (let b = 0; b < starts.length; b++) {
      const end = b + 1 < starts.length ? starts[b + 1] : next
      for (let i = starts[b]; i < end; i++) {
        fit[i] = values[b]
        sse += ws[i] * (ys[i] - values[b]) ** 2
      }
    }
    const pending = starts.length >= 2 && violates(Float64Array.from(values), starts.length - 1)
    const done = next === n && !pending
    return {
      t,
      starts: dense.vec(Float64Array.from(starts)),
      values: dense.vec(Float64Array.from(values)),
      blockWeights: dense.vec(Float64Array.from(weights)),
      next,
      fit: dense.vec(fit),
      sse,
      event: done ? ('done' as const) : event,
      done,
    }
  }
  return {
    name: 'pool-adjacent-violators',
    init: () => state(0, [], [], [], 0, 'start'),
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1 }
      const starts = Array.from(dense.data(s.starts))
      const values = Array.from(dense.data(s.values))
      const weights = Array.from(dense.data(s.blockWeights))
      const k = values.length - 1
      if (k >= 1 && violates(Float64Array.from(values), k)) {
        const w = weights[k - 1] + weights[k]
        // Zero total weight pools to the plain mean, so the fit stays defined.
        const v =
          w > 0 ? (weights[k - 1] * values[k - 1] + weights[k] * values[k]) / w : (values[k - 1] + values[k]) / 2
        starts.pop()
        values.splice(k - 1, 2, v)
        weights.splice(k - 1, 2, w)
        return state(s.t + 1, starts, values, weights, s.next, 'pool')
      }
      starts.push(s.next)
      values.push(ys[s.next])
      weights.push(ws[s.next])
      return state(s.t + 1, starts, values, weights, s.next + 1, 'add')
    },
    done: (s) => s.done,
  }
}

/** The result of `isotonicRegression`. */
export type IsotonicFit = {
  /** The fitted values, in the order of the input. */
  fit: Tensor
  /** The covariate at each block's first point (the sorted x when x was given, else the index). */
  thresholds: Tensor
  /** The value of each block. */
  values: Tensor
}

/**
 * The isotonic regression of y on x (or on the order of y when x is omitted): points are sorted by x, points with equal
 * x are first pooled to their weighted mean (as scikit-learn), then `poolAdjacentViolatorsSteps` runs to the end. The
 * fit is returned in the input order.
 */
export function isotonicRegression(y: VectorLike, options: IsotonicOptions & { x?: VectorLike } = {}): IsotonicFit {
  const where = 'isotonicRegression'
  const ys = dense.toF64(y, where)
  const n = ys.length
  const ws = options.weights === undefined ? new Float64Array(n).fill(1) : dense.toF64(options.weights, where)
  const xs = options.x === undefined ? Float64Array.from({ length: n }, (_, i) => i) : dense.toF64(options.x, where)
  if (xs.length !== n) throw new DomainError(where, `${where}: ${xs.length} covariates for ${n} values`)
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => xs[a] - xs[b] || a - b)
  // Pool ties in x.
  const groupOf = new Int32Array(n)
  const gx: number[] = []
  const gy: number[] = []
  const gw: number[] = []
  for (const i of order) {
    const g = gx.length - 1
    if (g >= 0 && xs[i] === gx[g]) {
      const w = gw[g] + ws[i]
      gy[g] = w > 0 ? (gw[g] * gy[g] + ws[i] * ys[i]) / w : (gy[g] + ys[i]) / 2
      gw[g] = w
    } else {
      gx.push(xs[i])
      gy.push(ys[i])
      gw.push(ws[i])
    }
    groupOf[i] = gx.length - 1
  }
  const s = run(poolAdjacentViolatorsSteps(gy, { weights: gw, increasing: options.increasing }), undefined, 2 * n + 1)
  const groupFit = dense.data(s.fit)
  const fit = Float64Array.from({ length: n }, (_, i) => groupFit[groupOf[i]])
  const starts = dense.data(s.starts)
  return {
    fit: dense.vec(fit),
    thresholds: dense.vec(Float64Array.from(starts, (g) => gx[g])),
    values: dense.vec(Float64Array.from(dense.data(s.values))),
  }
}
