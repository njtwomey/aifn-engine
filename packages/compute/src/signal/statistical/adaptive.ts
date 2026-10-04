/**
 * Adaptive FIR filters as step-through algorithms, one input sample per step: least mean squares (`lms`; Widrow and
 * Hoff), normalised LMS (`nlms`) and recursive least squares (`rls`). Each adapts M taps w so that the output
 * y[n] = wᵀu[n], with the regressor u[n] = [x[n], x[n − 1], …, x[n − M + 1]] (zeros before the first sample), tracks a
 * desired signal d[n]. The error is the a-priori error e[n] = d[n] − w[n − 1]ᵀu[n] (Sayed, 2008, "Adaptive Filters",
 * ch. 10, 11 and 30). A run ends (`terminated`) once every sample has been used.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { readSamples } from '../signal'

/** The state shared by the adaptive filters after t samples. */
export type AdaptiveFilterState = Status & {
  /** Samples used so far. */
  t: Size
  /** The taps w (length M); w[k] multiplies x[n − k]. */
  w: Tensor
  /** The last output y[n] = w[n − 1]ᵀu[n] (0 at t = 0). */
  y: number
  /** The last a-priori error e[n] = d[n] − y[n] (0 at t = 0). */
  e: number
  /** e[n]², the instantaneous squared error. */
  squaredError: number
  /** True once every sample has been used. */
  terminated: boolean
}

/** The state of `rls`: also the inverse correlation matrix P (M × M) and the last gain vector k. */
export type RlsState = AdaptiveFilterState & { P: Tensor; gain: Tensor }

/** The start of an adaptive filter: initial taps (default zeros). */
export type AdaptiveFilterStart = { w0?: VectorLike } | undefined

/** Options every adaptive filter takes. */
export type AdaptiveFilterOptions = {
  /** Number of taps M. */
  order: Size
}

function signals(input: VectorLike, desired: VectorLike, order: Size, where: string) {
  if (!(Number.isInteger(order) && order >= 1))
    throw new ShapeError(where, `${where}: order must be a positive integer`)
  const x = readSamples(input, where).values
  const d = readSamples(desired, where).values
  if (x.length !== d.length) throw new ShapeError(where, `${where}: input and desired differ in length`)
  return { x, d, n: x.length }
}

function startTaps(start: AdaptiveFilterStart, order: Size, where: string): Float64Array {
  if (!start?.w0) return new Float64Array(order)
  const w = readSamples(start.w0, where).values
  if (w.length !== order) throw new ShapeError(where, `${where}: w0 has ${w.length} taps, expected ${order}`)
  return w
}

/** The regressor u[t] = [x[t], x[t − 1], …, x[t − M + 1]]. */
function regressor(x: Float64Array, t: number, order: number): Float64Array {
  const u = new Float64Array(order)
  for (let k = 0; k < order && t - k >= 0; k++) u[k] = x[t - k]
  return u
}

const dot = (a: Float64Array, b: Float64Array) => {
  let s = 0
  for (let k = 0; k < a.length; k++) s += a[k] * b[k]
  return s
}

/** The common shape of LMS and NLMS: w ← w + step(u)·e·u. */
function gradientFilter(
  name: string,
  input: VectorLike,
  desired: VectorLike,
  order: Size,
  step: (u: Float64Array) => number,
): Algorithm<AdaptiveFilterStart, AdaptiveFilterState> {
  const { x, d, n } = signals(input, desired, order, name)
  return {
    name,
    init: (start) => ({
      t: 0,
      w: fromData(startTaps(start, order, name), [order]),
      y: 0,
      e: 0,
      squaredError: 0,
      terminated: n === 0,
    }),
    step: (s) => {
      const u = regressor(x, s.t, order)
      const w = Float64Array.from(s.w.data as Float64Array)
      const y = dot(w, u)
      const e = d[s.t] - y
      const mu = step(u)
      for (let k = 0; k < order; k++) w[k] += mu * e * u[k]
      return { t: s.t + 1, w: fromData(w, [order]), y, e, squaredError: e * e, terminated: s.t + 1 >= n }
    },
  }
}

/**
 * The least-mean-squares filter (Widrow and Hoff, 1960; Widrow et al., 1976): w ← w + μ·e[n]·u[n], a stochastic-
 * gradient step on the mean squared error. It converges in the mean for 0 < μ < 2/λ_max of the input's correlation
 * matrix. `init` takes `{ w0 }` (default zeros).
 */
export function lms(
  input: VectorLike,
  desired: VectorLike,
  options: AdaptiveFilterOptions & { stepSize: Scalar },
): Algorithm<AdaptiveFilterStart, AdaptiveFilterState> {
  const mu = options.stepSize
  return gradientFilter('lms', input, desired, options.order, () => mu)
}

/**
 * The normalised LMS filter: w ← w + μ/(ε + ‖u[n]‖²)·e[n]·u[n], LMS with a step scaled by the regressor's energy, so it
 * is insensitive to the input's scale and stable for 0 < μ < 2. `epsilon` (default 1e-6) guards against a zero
 * regressor. `init` takes `{ w0 }`.
 */
export function nlms(
  input: VectorLike,
  desired: VectorLike,
  options: AdaptiveFilterOptions & { stepSize: Scalar; epsilon?: Scalar },
): Algorithm<AdaptiveFilterStart, AdaptiveFilterState> {
  const mu = options.stepSize
  const eps = options.epsilon ?? 1e-6
  return gradientFilter('nlms', input, desired, options.order, (u) => mu / (eps + dot(u, u)))
}

/**
 * The recursive least-squares filter: the exact minimiser of Σᵢ λ^{n−i} e²[i] + δλⁿ‖w‖², updated per sample by the
 * matrix inversion lemma: k = Pu/(λ + uᵀPu), w ← w + k·e[n], P ← (P − k uᵀP)/λ, from P = I/δ. `forgetting` λ (default
 * 0.99) sets a memory of about 1/(1 − λ) samples; `delta` δ (default 0.01) is the initial ridge. P is symmetrised each
 * step against rounding. `init` takes `{ w0 }`.
 */
export function rls(
  input: VectorLike,
  desired: VectorLike,
  options: AdaptiveFilterOptions & { forgetting?: Scalar; delta?: Scalar },
): Algorithm<AdaptiveFilterStart, RlsState> {
  const name = 'rls'
  const M = options.order
  const { x, d, n } = signals(input, desired, M, name)
  const lambda = options.forgetting ?? 0.99
  const delta = options.delta ?? 0.01
  if (!(lambda > 0 && lambda <= 1)) throw new DomainError('rls', 'rls: forgetting must lie in (0, 1]')
  if (!(delta > 0)) throw new DomainError('rls', 'rls: delta must be positive')
  return {
    name,
    init: (start) => {
      const P = new Float64Array(M * M)
      for (let i = 0; i < M; i++) P[i * M + i] = 1 / delta
      return {
        t: 0,
        w: fromData(startTaps(start, M, name), [M]),
        P: fromData(P, [M, M]),
        gain: fromData(new Float64Array(M), [M]),
        y: 0,
        e: 0,
        squaredError: 0,
        terminated: n === 0,
      }
    },
    step: (s) => {
      const u = regressor(x, s.t, M)
      const P = s.P.data as Float64Array
      const w = Float64Array.from(s.w.data as Float64Array)
      const Pu = new Float64Array(M)
      for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) Pu[i] += P[i * M + j] * u[j]
      const denominator = lambda + dot(u, Pu)
      const k = Pu.map((v) => v / denominator)
      const y = dot(w, u)
      const e = d[s.t] - y
      for (let i = 0; i < M; i++) w[i] += k[i] * e
      // P ← (P − k (Pu)ᵀ)/λ, using uᵀP = (Pu)ᵀ for symmetric P; then symmetrise.
      const next = new Float64Array(M * M)
      for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) next[i * M + j] = (P[i * M + j] - k[i] * Pu[j]) / lambda
      for (let i = 0; i < M; i++)
        for (let j = i + 1; j < M; j++) {
          const v = 0.5 * (next[i * M + j] + next[j * M + i])
          next[i * M + j] = v
          next[j * M + i] = v
        }
      return {
        t: s.t + 1,
        w: fromData(w, [M]),
        P: fromData(next, [M, M]),
        gain: fromData(k, [M]),
        y,
        e,
        squaredError: e * e,
        terminated: s.t + 1 >= n,
      }
    },
  }
}
