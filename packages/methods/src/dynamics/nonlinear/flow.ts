/**
 * Poincaré sections and limit cycles of autonomous flows $\dot{\xvec} = \fvec(\xvec)$, part of
 * `aifn-methods/dynamics/nonlinear` (Strogatz, 2015, "Nonlinear Dynamics and Chaos", 2nd ed., §8.7; Guckenheimer and
 * Holmes, 1983, §1.5).
 *
 * A section is a hyperplane crossed in one direction; its crossings are located by event detection on an adaptive
 * Dormand–Prince integration, and give the orbit of the first-return map $P$. A limit cycle is a fixed point of $P$,
 * found by iterating it. Integrating the time-reversed flow finds the repelling cycles too.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toFlat, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { autonomous, trajectory, type VectorField } from 'aifn-compute/dynamics/fields'
import { dormandPrince, withEvents, type EventHit, type Rhs } from 'aifn-compute/dynamics/ode'

const toF64 = dense.toF64

/**
 * A Poincaré section: the hyperplane $\{\xvec : \nvec \cdot (\xvec - \pvec) = 0\}$ through `point` $\pvec$ with
 * `normal` $\nvec$ (each $n$ values), crossed in the direction of $\nvec$.
 */
export type Section = { point: VectorLike; normal: VectorLike }

/** Options for Poincaré sections and limit cycles. */
export type SectionOptions = {
  /** Stop after this many crossings. Default 20. */
  crossings?: number
  /** Give up after this much time. Default 1000. */
  tMax?: number
  /** Relative tolerance of the Dormand–Prince integration. Default 1e-9. */
  rtol?: number
  /** Absolute tolerance of the Dormand–Prince integration. Default 1e-11. */
  atol?: number
  /** Integrate backwards in time (finds unstable cycles). Default false. */
  reverse?: boolean
}

/**
 * The successive crossings of a trajectory from $\xvec_0$ with a Poincaré section, in the direction of its normal
 * (with `reverse`, of the time-reversed flow): the orbit of the first-return map $P$. Located by event detection on
 * an adaptive Dormand–Prince integration, which stops after `crossings` crossings or at `tMax`, whichever comes
 * first; a start on the section may count as a crossing at $t = 0$.
 *
 * @param f The vector field $\fvec$.
 * @param x0 The starting point $\xvec_0$ ($n$ values).
 * @param section The section's point and normal.
 * @param options The number of crossings, the time limit, the tolerances and the direction of time.
 * @returns The crossing times `t` (negative with `reverse`), the crossing `points` ($k \times n$, one per row) and
 *   the `returnTimes` between successive crossings ($k - 1$ of them, positive).
 *
 * @example The van der Pol oscillator settles onto its cycle
 * const vdp = (x) => {
 *   const [a, b] = toFlat(x)
 *   return [b, (1 - a * a) * b - a]
 * }
 * const r = poincareSection(vdp, [1, 1], { point: [0, 0], normal: [0, 1] }, { crossings: 4 })
 * print('points =', r.points)
 * print('return times =', r.returnTimes)
 */
export function poincareSection(
  f: VectorField,
  x0: VectorLike,
  section: Section,
  options: SectionOptions = {},
): { t: Vector; points: Matrix; returnTimes: Vector } {
  const { crossings = 20, tMax = 1000, rtol = 1e-9, atol = 1e-11, reverse = false } = options
  const p = toF64(section.point, 'poincareSection')
  const nrm = toF64(section.normal, 'poincareSection')
  const sign = reverse ? -1 : 1
  const rhs: Rhs = reverse ? (_t, x) => toF64(f(x), 'poincareSection').map((v) => -v) : autonomous(f)
  const g = (_t: number, x: Tensor) => {
    const d = toFlat(x)
    let s = 0
    for (let i = 0; i < d.length; i++) s += nrm[i] * (d[i] - p[i])
    return s
  }
  const alg = withEvents(dormandPrince(rhs, { tEnd: tMax, rtol, atol }), rhs, [{ g, direction: 1 }])
  let hits: readonly EventHit[] = []
  let guard = 0
  for (const { state, stopped } of live(alg, { x0 })) {
    hits = state.events
    if (stopped || hits.length >= crossings || guard++ >= 1_000_000) break
  }
  hits = hits.slice(0, crossings)
  const n = p.length
  const pts = new Float64Array(hits.length * n)
  hits.forEach((e, k) => pts.set(toFlat(e.x), k * n))
  const times = Float64Array.from(hits, (e) => sign * e.time)
  const returns = Float64Array.from({ length: Math.max(0, hits.length - 1) }, (_, k) =>
    Math.abs(times[k + 1] - times[k]),
  )
  return {
    t: fromData(times, [hits.length]),
    points: fromData(pts, [hits.length, n]),
    returnTimes: fromData(returns, [returns.length]),
  }
}

/** A limit cycle found by iterating the first-return map. */
export type LimitCycle = {
  /** Whether successive returns agreed to `tolerance`. */
  converged: boolean
  /** The point where the cycle crosses the section. */
  point: Vector
  /** The period (time of one return). */
  period: number
  /**
   * One period of the orbit ($m \times n$, $m$ = `orbitPoints` + 1), from `point`; one row when no return was
   * found.
   */
  orbit: Matrix
  /**
   * For a planar flow, the derivative of the return map along the section (the nontrivial Floquet multiplier): the
   * cycle is stable when $\lvert \text{multiplier} \rvert < 1$. NaN in higher dimensions, or when the iteration did
   * not converge.
   */
  multiplier: number
  /**
   * Distances between successive returns, which shrink geometrically by $\lvert \text{multiplier} \rvert$ near a
   * cycle.
   */
  residuals: Vector
}

/**
 * A limit cycle through a Poincaré section, found by iterating the return map $P$ from the first crossing of the
 * trajectory from $\xvec_0$ until successive crossings agree to `tolerance` (relative: the distance is at most
 * $\text{tolerance} \cdot (1 + \lVert \xvec \rVert)$). This finds attracting cycles; with `reverse: true` it iterates
 * the return map of the time-reversed flow and so finds repelling ones. The Floquet multiplier of a planar cycle is
 * $P'$ along the section, estimated by central differences on the return map (and reported for forward time also
 * with `reverse`). Throws when the trajectory never crosses the section.
 *
 * @param f The vector field $\fvec$.
 * @param x0 The starting point $\xvec_0$ ($n$ values), near the cycle sought.
 * @param section The section's point and normal; the cycle must cross it once per period in the normal's direction.
 * @param options The `SectionOptions` of each return (`crossings` is set by the search), and: `tolerance`, the
 *   relative agreement of successive returns that counts as converged (default 1e-8); `maxReturns`, the most returns
 *   iterated (default 200); `orbitPoints`, the RK4 steps of the one-period `orbit` returned (default 200).
 * @returns The cycle's crossing point, its period, one period of its orbit, the Floquet multiplier, the residuals and
 *   whether the search converged.
 *
 * @example The van der Pol cycle: period about 6.66, amplitude about 2
 * const vdp = (x) => {
 *   const [a, b] = toFlat(x)
 *   return [b, (1 - a * a) * b - a]
 * }
 * const c = limitCycle(vdp, [1, 1], { point: [0, 0], normal: [0, 1] }, { orbitPoints: 40 })
 * print('converged =', c.converged, ' after', c.residuals.shape[0], 'returns')
 * print('point =', c.point, ' period =', c.period)
 * print('multiplier =', c.multiplier)
 */
export function limitCycle(
  f: VectorField,
  x0: VectorLike,
  section: Section,
  options: SectionOptions & { tolerance?: number; maxReturns?: number; orbitPoints?: number } = {},
): LimitCycle {
  const { tolerance: tol = 1e-8, maxReturns = 200, orbitPoints = 200 } = options
  const once = (x: VectorLike) => {
    // The start is on the section already, so the first crossing is the return.
    const r = poincareSection(f, x, section, { ...options, crossings: 2 })
    const pts = toFlat(r.points)
    const n = r.points.shape[1] ?? 0
    if (r.points.shape[0] === 0) return null
    // A start exactly on the section may be reported as a crossing at t ≈ 0; skip it.
    const k = Math.abs(toFlat(r.t)[0]) < 1e-9 && r.points.shape[0] > 1 ? 1 : 0
    return { x: Float64Array.from(pts.slice(k * n, (k + 1) * n)), t: Math.abs(toFlat(r.t)[k]) }
  }
  const first = poincareSection(f, x0, section, { ...options, crossings: 1 })
  if (first.points.shape[0] === 0) throw new Error('limitCycle: the trajectory never crosses the section')
  let x = Float64Array.from(toFlat(first.points))
  let period = NaN
  let converged = false
  const residuals: number[] = []
  for (let k = 0; k < maxReturns; k++) {
    const next = once(x)
    if (!next) break
    const d = Math.hypot(...next.x.map((v, i) => v - x[i]))
    residuals.push(d)
    x = next.x
    period = next.t
    if (d <= tol * (1 + Math.hypot(...x))) {
      converged = true
      break
    }
  }
  let multiplier = NaN
  const nrm = toF64(section.normal, 'limitCycle')
  if (x.length === 2 && converged) {
    const tangent = [-nrm[1], nrm[0]].map((v) => v / Math.hypot(nrm[0], nrm[1]))
    const eps = 1e-5 * (1 + Math.hypot(...x))
    const along = (s: number) => {
      const r = once(x.map((v, i) => v + s * eps * tangent[i]))
      return r ? (r.x[0] - x[0]) * tangent[0] + (r.x[1] - x[1]) * tangent[1] : NaN
    }
    multiplier = (along(1) - along(-1)) / (2 * eps)
    if (options.reverse) multiplier = 1 / multiplier
  }
  const orbit = Number.isFinite(period)
    ? trajectory(f, x, options.reverse ? -period : period, { steps: orbitPoints }).x
    : fromData(Float64Array.from(x), [1, x.length])
  return {
    converged,
    point: fromData(x, [x.length]),
    period,
    orbit,
    multiplier,
    residuals: fromData(Float64Array.from(residuals), [residuals.length]),
  }
}
