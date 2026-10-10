/**
 * Isotonic regression by the pool-adjacent-violators algorithm (Ayer et al., 1955; Best and Chakravarti, 1990), as a
 * traceable algorithm: the fit $\hat y$ minimising $\sum_i w_i (y_i - \hat y_i)^2$ subject to
 * $\hat y_1 \le \hat y_2 \le \dots \le \hat y_n$ (or $\ge$ for a decreasing fit). The solution is piecewise constant:
 * each block of pooled points takes the weighted mean of its $y_i$.
 *
 * A negative or NaN weight, or weights and values that differ in number, throw `DomainError`. Nothing is
 * differentiable.
 */

import type { Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'

type F64 = Float64Array

/** Options of `poolAdjacentViolatorsSteps` and `isotonicRegression`. */
export type IsotonicOptions = {
  /** Non-negative weights $w_i$, one per value (default all 1). */
  weights?: VectorLike
  /** Fit a non-decreasing sequence (`true`, the default) or a non-increasing one (`false`). */
  increasing?: boolean
}

/** What the last step of `poolAdjacentViolatorsSteps` did. */
export type PavEvent = 'start' | 'add' | 'pool' | 'done'

/** One state of `poolAdjacentViolatorsSteps`. */
export interface PavState extends Status {
  /** The index of the first point of each block, in order. */
  starts: Tensor
  /** The weighted mean of $y_i$ over each block. */
  values: Tensor
  /** The total weight of each block. */
  blockWeights: Tensor
  /** The number of points added so far: points $0, \dots, \text{next} - 1$ are in blocks. */
  next: number
  /** The current fit: each point added takes its block's value ($n$ values; NaN from `next` on). */
  fit: Tensor
  /** The weighted sum of squares $\sum_i w_i (y_i - \hat y_i)^2$ over the points added so far. */
  sse: number
  /** What the last step did: `start`, `add`, `pool`, or `done` once finished. */
  event: PavEvent
  /** True when every point is added and no two adjacent blocks violate the order: `fit` is the isotonic regression. */
  done: boolean
}

/**
 * Pool adjacent violators as a traceable algorithm on a sequence $y_1, \dots, y_n$ (ordered by its covariate). Each
 * step either adds the next point as a block of its own (`add`) or, when the last two blocks violate the order, pools
 * them into one block with their weighted mean (`pool`; a pair of zero total weight takes the plain mean). Pooling only
 * merges a violating pair, so the blocks are always in order up to the last one, and the fit after the last point is
 * the isotonic regression. Each point is added once and each pool removes a block, so it finishes in at most $2n - 1$
 * steps. Throws `DomainError` when the weights and values differ in number, or for a negative or NaN weight.
 *
 * @param y The values $y_i$, already in the order of their covariate (not modified).
 * @param options The weights $w_i$ and the direction of the fit.
 * @returns The algorithm: run it from `undefined` (`run(alg, undefined, 2 * n)`) and read `fit` from the final state.
 *
 * @example Run to the end: the violating pair 3, 2 is pooled to 2.5
 * const s = run(poolAdjacentViolatorsSteps([1, 3, 2, 4]), undefined, 10)
 * print('fit =', s.fit)
 * print('blocks start at', s.starts, 'with values', s.values)
 * print('steps =', s.t, 'done =', s.done)
 *
 * @example Step by step: what each step did
 * const alg = poolAdjacentViolatorsSteps([1, 3, 2, 4])
 * for (let k = 1; k <= 5; k++) {
 *   const s = run(alg, undefined, k)
 *   print(`step ${k}:`, s.event, 'fit =', s.fit)
 * }
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
  /** The fitted values $\hat y_i$, in the order of the input. */
  fit: Tensor
  /** The covariate at each block's first point (the sorted x when x was given, else the index). */
  thresholds: Tensor
  /** The value of each block, in the order of `thresholds`. */
  values: Tensor
}

/**
 * The isotonic regression of $y$ on $x$ (or on the order of $y$ when $x$ is omitted): points are sorted by $x$, points
 * with equal $x$ are first pooled to their weighted mean (as scikit-learn), then `poolAdjacentViolatorsSteps` runs to
 * the end. The fit is returned in the input order. Throws `DomainError` when `x` and `y` differ in length, or for a
 * negative or NaN weight.
 *
 * @param y The values $y_i$ to fit, in any order (not modified).
 * @param options The weights $w_i$ (one per value) and direction, as `IsotonicOptions`, and `x`, the covariate of each
 *   value (default the index $0, \dots, n - 1$, so $y$ is taken in order).
 * @returns The fit in the input order, with the blocks as covariate thresholds and values.
 *
 * @example A short sequence: two violations, each pooled to its mean
 * const { fit, thresholds, values } = isotonicRegression([1, 3, 2, 4, 3, 5])
 * print('fit =', fit)
 * print('thresholds =', thresholds)
 * print('values =', values)
 *
 * @example Unsorted covariates with a tie: the two points at x = 2 are pooled first
 * const { fit, thresholds, values } = isotonicRegression([4, 1, 3, 1], { x: [3, 1, 2, 2] })
 * print('fit =', fit)
 * print('thresholds =', thresholds)
 * print('values =', values)
 *
 * @example A decreasing fit
 * print('fit =', isotonicRegression([5, 3, 4, 1], { increasing: false }).fit)
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
