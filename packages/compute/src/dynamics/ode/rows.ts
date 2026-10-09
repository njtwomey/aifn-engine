/**
 * Many independent initial-value problems solved together, each with its own adaptive steps: the Dormand–Prince 5(4)
 * pair of `dormandPrince` (Dormand & Prince, 1980) run row by row on a batch, with the field evaluated once per stage
 * on every row still integrating. Row i's states, step sizes, rejections and function evaluations are exactly those
 * of `dormandPrince` on row i alone, so the per-instance work of a batch (the number of function evaluations, NFE,
 * each instance needs) is measured at the cost of batched evaluations. A batched solve with one shared step size
 * instead takes the steps its hardest row needs, and its NFE is the batch's maximum.
 */

import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { DORMAND_PRINCE } from './adaptive'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The field on the $m$ rows still integrating: `t` their times ($m$ values), `x` their states ($m \times d$,
 * row-major) and `rows` their indices in the batch ($m$ values); returns $f(t_i, \xvec_i)$ for each, $m \times d$
 * row-major.
 */
export type RowsRhs = (t: Float64Array, x: Float64Array, rows: Int32Array) => ArrayLike<number>

/** Options of {@link dormandPrinceRows}. */
export type RowsSolveOptions = {
  /** Start time of every row. Default 0. */
  t0?: Scalar
  /** End time of every row (required; below `t0` integrates backwards). */
  tEnd: Scalar
  /** The relative tolerance (default 1e-3, as `dormandPrince`). */
  rtol?: Scalar
  /** The absolute tolerance, one value for every component (default 1e-6, as `dormandPrince`). */
  atol?: Scalar
  /** The most accepted steps per row before it is marked failed. Default 10 000. */
  maxSteps?: Size
  /**
   * Continue earlier solves (a `RowsSolution` will do) rather than starting afresh: each row's proposed step size
   * (its magnitude, capped by the interval) and $f(t_0, \xvec_0)$ from the previous solve's `nextStepSize` and
   * `derivative` (the field must be unchanged for that row). A row whose step is not a finite non-zero number starts
   * afresh (its $f(t_0, \xvec_0)$ and starting step cost two evaluations).
   */
  resume?: { nextStepSize: ArrayLike<number>; derivative: ArrayLike<number> }
}

/** The solution of {@link dormandPrinceRows}. */
export type RowsSolution = {
  /** States at `tEnd`, $B \times d$ row-major (a failed row keeps its last accepted state). */
  x: Float64Array
  /** Function evaluations of each row, including the start's evaluation and the starting-step probe. */
  evaluations: Int32Array
  /** Accepted steps of each row. */
  steps: Int32Array
  /** Rejected step attempts of each row. */
  rejected: Int32Array
  /** 1 where a row stopped early (step size underflow or `maxSteps`). */
  failed: Uint8Array
  /** The step each row would try next, signed (to `resume` a later solve). */
  nextStepSize: Float64Array
  /** $f$ at each row's end state, $B \times d$ row-major (to `resume` a later solve). */
  derivative: Float64Array
}

const SAFETY = 0.9
const MIN_FACTOR = 0.2
const MAX_FACTOR = 10

/**
 * Solve $B$ independent IVPs $\xvec_i' = f(t, \xvec_i)$, $\xvec_i(t_0) = \xvec_{0,i}$ with Dormand–Prince, each row
 * with its own step-size control (the controller, starting step and first-same-as-last reuse of `dormandPrince`).
 * Each stage calls `f` once on the rows still integrating. A row stops early, marked in `failed`, when its step size
 * underflows or it reaches `maxSteps`; a non-finite attempt is retried with a fifth of the step. Non-finite `t0` or
 * `tEnd`, or a `d` that is not a positive integer, throws `DomainError`; an `x0` that is not whole rows of $d$ values,
 * or an `f` that returns the wrong number of values, throws `ShapeError`.
 *
 * @param f The field on the rows still integrating, called with their times, states and indices in the batch.
 * @param x0 The initial states, $B \times d$ row-major ($B$ is its length over $d$); copied, not modified.
 * @param d The dimension of each row's state.
 * @param options The interval, the tolerances, the step limit per row and the solves to resume.
 * @returns The end states and, per row, the work done, whether it failed, and the step and derivative to resume from.
 *
 * @example A different rate per row: the faster decay takes more evaluations
 * // x′ = −kᵢ x on two rows of d = 1.
 * const k = [0.5, 20]
 * const sol = dormandPrinceRows((t, x, rows) => x.map((v, j) => -k[rows[j]] * v), Float64Array.of(1, 1), 1, {
 *   tEnd: 1,
 * })
 * print('x(1) =', sol.x)
 * print('exact =', k.map((r) => Math.exp(-r)))
 * print('evaluations =', sol.evaluations)
 * print('accepted, rejected =', sol.steps, sol.rejected)
 *
 * @example Resuming a solve skips the start-up evaluations
 * const f = (t, x, rows) => x.map((v) => -v)
 * const first = dormandPrinceRows(f, Float64Array.of(1, 2), 1, { tEnd: 0.5 })
 * const resumed = dormandPrinceRows(f, first.x, 1, { t0: 0.5, tEnd: 1, resume: first })
 * const afresh = dormandPrinceRows(f, first.x, 1, { t0: 0.5, tEnd: 1 })
 * print('x(1) =', resumed.x, ' exact =', [1, 2].map((v) => v * Math.exp(-1)))
 * print('evaluations resumed =', resumed.evaluations, ' afresh =', afresh.evaluations)
 */
export function dormandPrinceRows(f: RowsRhs, x0: ArrayLike<number>, d: Size, options: RowsSolveOptions): RowsSolution {
  const { t0 = 0, tEnd, rtol = 1e-3, atol = 1e-6, maxSteps = 10_000 } = options
  if (!Number.isFinite(tEnd) || !Number.isFinite(t0))
    throw new DomainError('dormandPrinceRows', 'dormandPrinceRows: t0 and tEnd must be finite')
  if (!(Number.isInteger(d) && d >= 1))
    throw new DomainError('dormandPrinceRows', 'dormandPrinceRows: d must be a positive integer')
  if (x0.length % d !== 0)
    throw new ShapeError('dormandPrinceRows', 'dormandPrinceRows: x0 must hold B rows of d values')
  const B = x0.length / d
  const tab = DORMAND_PRINCE
  const S = tab.c.length
  const errW = tab.b.map((b, i) => b - tab.bHat![i])
  const dir = Math.sign(tEnd - t0) || 1
  const interval = Math.abs(tEnd - t0)

  const x = Float64Array.from(x0)
  const time = new Float64Array(B).fill(t0)
  const h = new Float64Array(B)
  const k0 = new Float64Array(B * d)
  const evaluations = new Int32Array(B)
  const steps = new Int32Array(B)
  const rejected = new Int32Array(B)
  const failed = new Uint8Array(B)
  const retried = new Uint8Array(B)
  const out: RowsSolution = { x, evaluations, steps, rejected, failed, nextStepSize: h, derivative: k0 }
  if (B === 0 || interval === 0) return out

  /** f on the given rows at the given times and states (packed), counted per row. */
  const call = (rows: Int32Array, t: Float64Array, z: Float64Array): Float64Array => {
    for (const r of rows) evaluations[r]++
    const v = f(t, z, rows)
    if (v.length !== rows.length * d)
      throw new ShapeError(
        'dormandPrinceRows',
        `dormandPrinceRows: f returned ${v.length} values, expected ${rows.length * d}`,
      )
    return Float64Array.from(v)
  }

  // Rows resumed from an earlier solve keep its step (capped by the interval) and its f(t₀, x₀).
  const resumed = (i: number) => {
    const hr = options.resume?.nextStepSize[i]
    return hr !== undefined && Number.isFinite(hr) && hr !== 0
  }
  for (let i = 0; i < B; i++)
    if (resumed(i)) {
      h[i] = dir * Math.min(Math.abs(options.resume!.nextStepSize[i]), interval)
      for (let j = 0; j < d; j++) k0[i * d + j] = options.resume!.derivative[i * d + j]
    }
  // f(t₀, x₀) for every other row, then Hairer's starting step (as `startingStep`, one more evaluation per row).
  const all = Int32Array.from({ length: B }, (_, i) => i)
  const fresh = Int32Array.from(all.filter((i) => !resumed(i)))
  if (fresh.length > 0) {
    const pack = (src: Float64Array) => {
      const outp = new Float64Array(fresh.length * d)
      fresh.forEach((i, a) => outp.set(src.subarray(i * d, (i + 1) * d), a * d))
      return outp
    }
    const xf = pack(x)
    const kf = call(fresh, new Float64Array(fresh.length).fill(t0), xf)
    const scaled = (v: ArrayLike<number>, at: number, ref: Float64Array) => {
      let s = 0
      for (let j = 0; j < d; j++) s += (v[at + j] / (atol + Math.abs(ref[at + j]) * rtol)) ** 2
      return Math.sqrt(s / d)
    }
    const h0 = new Float64Array(fresh.length)
    const probe = new Float64Array(fresh.length * d)
    for (let a = 0; a < fresh.length; a++) {
      const d0 = scaled(xf, a * d, xf)
      const d1 = scaled(kf, a * d, xf)
      h0[a] = Math.min(d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : (0.01 * d0) / d1, interval)
      for (let j = 0; j < d; j++) probe[a * d + j] = xf[a * d + j] + dir * h0[a] * kf[a * d + j]
    }
    const f1 = call(
      fresh,
      Float64Array.from(h0, (v) => t0 + dir * v),
      probe,
    )
    for (let a = 0; a < fresh.length; a++) {
      const i = fresh[a]
      let s = 0
      for (let j = 0; j < d; j++) s += ((f1[a * d + j] - kf[a * d + j]) / (atol + Math.abs(xf[a * d + j]) * rtol)) ** 2
      const d1 = scaled(kf, a * d, xf)
      const d2 = Math.sqrt(s / d) / h0[a]
      const h1 = Math.max(d1, d2) <= 1e-15 ? Math.max(1e-6, h0[a] * 1e-3) : (0.01 / Math.max(d1, d2)) ** (1 / 5)
      h[i] = dir * Math.min(100 * h0[a], h1, interval)
      k0.set(kf.subarray(a * d, (a + 1) * d), i * d)
    }
  }

  const done = (i: number) => failed[i] === 1 || Math.abs(tEnd - time[i]) <= 1e-12 * Math.max(1, Math.abs(tEnd))
  const k: Float64Array[] = Array.from({ length: S }, () => new Float64Array(0))
  for (;;) {
    const active = Int32Array.from(all.filter((i) => !done(i)))
    const m = active.length
    if (m === 0) break
    // This attempt's step per active row: never past tEnd; underflow fails the row.
    const hs = new Float64Array(m)
    for (let a = 0; a < m; a++) {
      const i = active[a]
      let hi = h[i]
      if (dir * (time[i] + hi - tEnd) > 0) hi = tEnd - time[i]
      hs[a] = hi
    }
    const liveRows: number[] = []
    const liveSteps: number[] = []
    for (let a = 0; a < m; a++) {
      const i = active[a]
      if (Math.abs(hs[a]) < 1e-12 * Math.max(1, Math.abs(time[i]))) failed[i] = 1
      else {
        liveRows.push(i)
        liveSteps.push(hs[a])
      }
    }
    if (liveRows.length === 0) continue
    const live = Int32Array.from(liveRows)
    const hl = Float64Array.from(liveSteps)
    const L = live.length
    const xs = new Float64Array(L * d)
    const ts = new Float64Array(L)
    k[0] = new Float64Array(L * d)
    for (let a = 0; a < L; a++) {
      const i = live[a]
      for (let j = 0; j < d; j++) {
        xs[a * d + j] = x[i * d + j]
        k[0][a * d + j] = k0[i * d + j]
      }
    }
    // Stages 2 … 7 (the first is the stored derivative: first same as last).
    for (let s = 1; s < S; s++) {
      const z = new Float64Array(L * d)
      for (let a = 0; a < L; a++) {
        ts[a] = time[live[a]] + tab.c[s] * hl[a]
        for (let j = 0; j < d; j++) {
          let acc = xs[a * d + j]
          for (let q = 0; q < s; q++) acc += hl[a] * tab.a[s][q] * k[q][a * d + j]
          z[a * d + j] = acc
        }
      }
      k[s] = call(live, ts, z)
    }
    for (let a = 0; a < L; a++) {
      const i = live[a]
      const y = new Float64Array(d)
      let err = 0
      let finite = true
      for (let j = 0; j < d; j++) {
        let acc = xs[a * d + j]
        let e = 0
        for (let s = 0; s < S; s++) {
          acc += hl[a] * tab.b[s] * k[s][a * d + j]
          e += hl[a] * errW[s] * k[s][a * d + j]
        }
        y[j] = acc
        if (!Number.isFinite(acc)) finite = false
        const sc = atol + rtol * Math.max(Math.abs(xs[a * d + j]), Math.abs(acc))
        err += (e / sc) ** 2
      }
      err = Math.sqrt(err / d)
      if (!finite || !Number.isFinite(err)) {
        rejected[i]++
        retried[i] = 1
        h[i] = hl[a] * MIN_FACTOR
        continue
      }
      const factor = err === 0 ? MAX_FACTOR : Math.min(MAX_FACTOR, Math.max(MIN_FACTOR, SAFETY * err ** (-1 / 5)))
      if (err <= 1) {
        // After a rejection within this step, do not grow the step again at once.
        const grow = retried[i] ? Math.min(1, factor) : factor
        time[i] += hl[a]
        for (let j = 0; j < d; j++) {
          x[i * d + j] = y[j]
          k0[i * d + j] = k[S - 1][a * d + j]
        }
        h[i] = hl[a] * grow
        retried[i] = 0
        steps[i]++
        if (steps[i] >= maxSteps && !done(i)) failed[i] = 1
      } else {
        rejected[i]++
        retried[i] = 1
        h[i] = hl[a] * factor
      }
    }
  }
  return out
}
