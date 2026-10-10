/**
 * Adaptive FIR filters as step-through algorithms, one input sample per step: least mean squares (`lms`; Widrow and
 * Hoff), normalised LMS (`nlms`) and recursive least squares (`rls`). Each adapts $M$ taps $\wvec$ so that the output
 * $y[n] = \wvec^\top \uvec[n]$, with the regressor $\uvec[n] = (x[n], x[n - 1], \dots, x[n - M + 1])$ (zeros before
 * the first sample), tracks a desired signal $d[n]$. The error is the a-priori error
 * $e[n] = d[n] - \wvec[n - 1]^\top \uvec[n]$, taken with the taps before the update (Sayed, 2008, "Adaptive
 * Filters", ch. 10, 11 and 30). A run ends (`terminated`) once every sample has been used, so `run` with a step count
 * of at least the series' length filters all of it.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { readSamples } from '../signal'

/** The state shared by the adaptive filters after t samples. */
export type AdaptiveFilterState = Status & {
  /** Samples used so far. */
  t: Size
  /** The taps $\wvec$ (length $M$); $w_k$ multiplies $x[n - k]$. */
  w: Tensor
  /** The last output $y[n] = \wvec[n - 1]^\top \uvec[n]$ (0 at $t = 0$). */
  y: number
  /** The last a-priori error $e[n] = d[n] - y[n]$ (0 at $t = 0$). */
  e: number
  /** $e[n]^2$, the instantaneous squared error. */
  squaredError: number
  /** True once every sample has been used. */
  terminated: boolean
}

/**
 * The state of `rls`: also `P`, the inverse correlation matrix $\Pmat$ ($M \times M$), and `gain`, the last gain
 * vector $\kvec$ (zeros at $t = 0$).
 */
export type RlsState = AdaptiveFilterState & { P: Tensor; gain: Tensor }

/** The start of an adaptive filter: initial taps (default zeros). */
export type AdaptiveFilterStart = { w0?: VectorLike } | undefined

/** Options every adaptive filter takes. */
export type AdaptiveFilterOptions = {
  /** Number of taps $M$, a positive integer. */
  order: Size
}

/**
 * Reads the input and desired signals of an adaptive filter. Throws `ShapeError` unless the order is a positive
 * integer and the two signals have the same length.
 *
 * @param input The input $x$ the filter sees.
 * @param desired The desired signal $d$, the same length as `input`.
 * @param order The number of taps $M$, checked here.
 * @param where The caller's name for error messages.
 * @returns The samples `x` and `d` and their length `n`.
 */
function signals(input: VectorLike, desired: VectorLike, order: Size, where: string) {
  if (!(Number.isInteger(order) && order >= 1))
    throw new ShapeError(where, `${where}: order must be a positive integer`)
  const x = readSamples(input, where).values
  const d = readSamples(desired, where).values
  if (x.length !== d.length) throw new ShapeError(where, `${where}: input and desired differ in length`)
  return { x, d, n: x.length }
}

/**
 * The initial taps: `start.w0` when given, else zeros. Throws `ShapeError` when `w0` does not have `order` taps.
 *
 * @param start The start passed to `init`, possibly undefined.
 * @param order The number of taps $M$.
 * @param where The caller's name for error messages.
 * @returns A fresh array of $M$ taps.
 */
function startTaps(start: AdaptiveFilterStart, order: Size, where: string): Float64Array {
  if (!start?.w0) return new Float64Array(order)
  const w = readSamples(start.w0, where).values
  if (w.length !== order) throw new ShapeError(where, `${where}: w0 has ${w.length} taps, expected ${order}`)
  return w
}

/**
 * The regressor $\uvec[t] = (x[t], x[t - 1], \dots, x[t - M + 1])$, with zeros for samples before the first.
 *
 * @param x The input samples.
 * @param t The current sample index.
 * @param order The number of taps $M$.
 * @returns A fresh array of $M$ values, newest first.
 */
function regressor(x: Float64Array, t: number, order: number): Float64Array {
  const u = new Float64Array(order)
  for (let k = 0; k < order && t - k >= 0; k++) u[k] = x[t - k]
  return u
}

/**
 * The dot product of two arrays, over the length of `a`.
 *
 * @param a The first array.
 * @param b The second array, at least as long as `a`.
 * @returns $\sum_k a_k b_k$.
 */
const dot = (a: Float64Array, b: Float64Array) => {
  let s = 0
  for (let k = 0; k < a.length; k++) s += a[k] * b[k]
  return s
}

/**
 * The common shape of LMS and NLMS: $\wvec \leftarrow \wvec + \mu(\uvec) \, e[n] \, \uvec[n]$, with a step size
 * $\mu(\uvec)$ that may depend on the regressor.
 *
 * @param name The algorithm's name, also used in error messages.
 * @param input The input $x$.
 * @param desired The desired signal $d$, the same length as `input`.
 * @param order The number of taps $M$.
 * @param step The step size $\mu(\uvec)$ for a regressor: a constant for LMS, normalised by the regressor's energy
 *   for NLMS.
 * @returns The algorithm, whose `init` takes `{ w0 }` or undefined.
 */
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
 * The least-mean-squares filter (Widrow and Hoff, 1960; Widrow et al., 1976):
 * $\wvec \leftarrow \wvec + \mu \, e[n] \, \uvec[n]$, a stochastic-gradient step on the mean squared error. It
 * converges in the mean for $0 < \mu < 2/\lambda_{\max}$, with $\lambda_{\max}$ the largest eigenvalue of the
 * input's correlation matrix, so the step size that works depends on the input's scale (`nlms` removes that). `init`
 * takes `{ w0 }` (default zeros). Throws `ShapeError` for a bad order or signals of different lengths.
 *
 * @param input The input $x$ the filter sees.
 * @param desired The desired signal $d$, the same length as `input`.
 * @param options `order`, the number of taps $M$, and `stepSize`, the step size $\mu$.
 * @returns The algorithm, one sample per step.
 *
 * @example Identify an unknown two-tap filter from white noise
 * // The unknown filter: d[n] = 0.5 x[n] - 0.3 x[n - 1].
 * const x = toArray(normals(stream(1), 400))
 * const d = x.map((v, t) => 0.5 * v - 0.3 * (t > 0 ? x[t - 1] : 0))
 * const alg = lms(x, d, { order: 2, stepSize: 0.05 })
 * for (const n of [10, 100, 400]) print(`taps after ${n} samples:`, run(alg, undefined, n).w)
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
 * The normalised LMS filter:
 * $\wvec \leftarrow \wvec + \frac{\mu}{\epsilon + \lVert \uvec[n] \rVert^2} \, e[n] \, \uvec[n]$, LMS with a step
 * scaled by the regressor's energy, so it is insensitive to the input's scale and stable for $0 < \mu < 2$. `init`
 * takes `{ w0 }`. Throws `ShapeError` for a bad order or signals of different lengths.
 *
 * @param input The input $x$ the filter sees.
 * @param desired The desired signal $d$, the same length as `input`.
 * @param options `order`, the number of taps $M$; `stepSize`, the normalised step $\mu$; and `epsilon`, the
 *   regulariser $\epsilon$ that guards against a zero regressor (default 1e-6).
 * @returns The algorithm, one sample per step.
 *
 * @example An input a hundred times larger: LMS with its old step diverges, NLMS does not care
 * // The unknown filter: d[n] = 0.5 x[n] - 0.3 x[n - 1].
 * const x = toArray(normals(stream(1), 400, 0, 100))
 * const d = x.map((v, t) => 0.5 * v - 0.3 * (t > 0 ? x[t - 1] : 0))
 * print('NLMS taps =', run(nlms(x, d, { order: 2, stepSize: 0.5 }), undefined, 100).w)
 * print('LMS taps =', run(lms(x, d, { order: 2, stepSize: 0.05 }), undefined, 100).w)
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
 * The recursive least-squares filter: the exact minimiser of
 * $\sum_i \lambda^{n-i} e[i]^2 + \delta\lambda^n \lVert \wvec \rVert^2$ (from zero initial taps), updated per sample
 * by the matrix inversion lemma: $\kvec = \Pmat\uvec / (\lambda + \uvec^\top \Pmat \uvec)$,
 * $\wvec \leftarrow \wvec + \kvec \, e[n]$, $\Pmat \leftarrow (\Pmat - \kvec \uvec^\top \Pmat)/\lambda$, from
 * $\Pmat = \Imat/\delta$. $\Pmat$ is symmetrised each step against rounding. `init` takes `{ w0 }`. Throws
 * `ShapeError` for a bad order or signals of different lengths, and `DomainError` for a forgetting factor outside
 * $(0, 1]$ or a $\delta$ that is not positive.
 *
 * @param input The input $x$ the filter sees.
 * @param desired The desired signal $d$, the same length as `input`.
 * @param options `order`, the number of taps $M$; `forgetting`, the factor $\lambda$ (default 0.99), which sets a
 *   memory of about $1/(1 - \lambda)$ samples; and `delta`, the initial ridge $\delta$ (default 0.01).
 * @returns The algorithm, one sample per step; its state also holds $\Pmat$ and the gain.
 *
 * @example RLS finds the unknown filter within a few samples
 * // The unknown filter: d[n] = 0.5 x[n] - 0.3 x[n - 1].
 * const x = toArray(normals(stream(1), 400))
 * const d = x.map((v, t) => 0.5 * v - 0.3 * (t > 0 ? x[t - 1] : 0))
 * const alg = rls(x, d, { order: 2 })
 * for (const n of [2, 5, 20]) print(`taps after ${n} samples:`, run(alg, undefined, n).w)
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
