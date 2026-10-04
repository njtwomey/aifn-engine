/**
 * Event detection: find the times at which scalar functions g(t, x(t)) cross zero along a solution, as scipy's
 * `solve_ivp(events=…)` does. A sign change of g between two accepted steps brackets a crossing; the crossing is then
 * located by Brent's method on the cubic Hermite interpolant of the step (Hairer, Nørsett & Wanner, 1993, §II.6,
 * "Dense output" and "Discontinuities"; Shampine & Thompson, 2000).
 */

import { findRoot } from 'aifn-compute/numerics/roots'
import { dense, fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Index, Scalar } from 'aifn-compute/foundation/contracts'
import { evaluate } from './explicit'
import type { OdeState, Rhs } from './types'

type F64 = dense.F64

/** An event: the zero of g(t, x). */
export type OdeEvent = {
  name?: string
  g: (t: Scalar, x: Tensor) => Scalar
  /** Stop the integration at the first crossing. Default false. */
  terminal?: boolean
  /** Only crossings in this direction: +1 (g increasing), −1 (decreasing) or 0 (both, the default). */
  direction?: -1 | 0 | 1
}

/** A located event. */
export type EventHit = {
  /** The position of the event in the list passed to `withEvents`. */
  event: Index
  name: string
  /** The time of the crossing. */
  time: Scalar
  x: Vector
  /** +1 if g increased through zero, −1 if it decreased. */
  direction: 1 | -1
}

/** A solver state with the events found so far. */
export type EventState<S extends OdeState> = S & {
  /** Every crossing found so far, in time order. */
  events: EventHit[]
  /** g at the current point, one per event. */
  eventValues: Scalar[]
  /** True once a terminal event has stopped the run (the runner stops on it). */
  terminated: boolean
}

/**
 * The cubic Hermite interpolant on [t₀, t₁] through (t₀, x₀) and (t₁, x₁) with slopes f₀ and f₁: third-order accurate,
 * which is enough to locate events to well within a step's own error.
 */
export function hermite(t0: Scalar, x0: F64, f0: F64, t1: Scalar, x1: F64, f1: F64): (t: Scalar) => F64 {
  const h = t1 - t0
  return (t) => {
    const s = (t - t0) / h
    const h00 = (1 + 2 * s) * (1 - s) ** 2
    const h10 = s * (1 - s) ** 2
    const h01 = s * s * (3 - 2 * s)
    const h11 = s * s * (s - 1)
    return Float64Array.from(x0, (v, i) => h00 * v + h10 * h * f0[i] + h01 * x1[i] + h11 * h * f1[i])
  }
}

/**
 * Wraps a solver so that it detects the zeros of the given event functions. After each step, every event whose g
 * changes sign (in its `direction`) is located by Brent's method on the step's Hermite interpolant, which costs two
 * extra evaluations of f per step with a crossing. A terminal event ends the run at the event: the state is moved to
 * (t_e, x(t_e)) and `terminated` is set, which stops the runners. Works with any solver whose state is an `OdeState`.
 *
 * A crossing is a strict sign change into or onto zero (g < 0 → g ≥ 0, or g > 0 → g ≤ 0), so a step that ends
 * exactly on g = 0 reports the event once, and a run that starts on g = 0 reports nothing there. scipy's `solve_ivp`
 * counts g ≤ 0 → g ≥ 0, so it reports a start on zero, and a zero at a step's end in both steps.
 */
export function withEvents<Opts, S extends OdeState>(
  solver: Algorithm<Opts, S>,
  f: Rhs,
  events: readonly OdeEvent[],
): Algorithm<Opts, EventState<S>> {
  const values = (t: Scalar, x: Vector) => events.map((e) => e.g(t, x))
  return {
    name: `${solver.name}+events`,
    init: (opts, s) => {
      const state = solver.init(opts, s)
      return { ...state, events: [], eventValues: values(state.time, state.x), terminated: false }
    },
    step: (s, ctx) => {
      const next = solver.step(s, ctx)
      const g1 = values(next.time, next.x)
      const hits: EventHit[] = []
      let evaluations = 0
      if (!next.diverged) {
        const x0 = dense.data(s.x)
        const x1 = dense.data(next.x)
        let interpolant: ((t: Scalar) => F64) | null = null
        events.forEach((e, k) => {
          const a = s.eventValues[k]
          const b = g1[k]
          const up = a < 0 && b >= 0
          const down = a > 0 && b <= 0
          if (!(up || down)) return
          if ((e.direction ?? 0) === 1 && !up) return
          if ((e.direction ?? 0) === -1 && !down) return
          if (!interpolant) {
            const f0 = evaluate(f, s.time, x0, 'withEvents')
            const f1 = evaluate(f, next.time, x1, 'withEvents')
            evaluations += 2
            interpolant = hermite(s.time, x0, f0, next.time, x1, f1)
          }
          const at = interpolant
          const lo = Math.min(s.time, next.time)
          const hi = Math.max(s.time, next.time)
          const r =
            b === 0
              ? { x: next.time }
              : findRoot((t) => e.g(t, fromData(at(t), [x0.length])), [lo, hi], { xtol: 1e-14 })
          const xe = at(r.x)
          hits.push({
            event: k,
            name: e.name ?? `event ${k}`,
            time: r.x,
            x: fromData(xe, [xe.length]),
            direction: up ? 1 : -1,
          })
        })
      }
      // Keep hits in the order the solution meets them.
      const forward = next.time >= s.time
      hits.sort((u, v) => (forward ? u.time - v.time : v.time - u.time))
      const firstTerminal = hits.find((hit) => events[hit.event].terminal)
      if (firstTerminal) {
        const kept = hits.filter((hit) => (forward ? hit.time <= firstTerminal.time : hit.time >= firstTerminal.time))
        return {
          ...next,
          time: firstTerminal.time,
          x: firstTerminal.x,
          stepSize: firstTerminal.time - s.time,
          evaluations: next.evaluations + evaluations,
          events: [...s.events, ...kept],
          eventValues: values(firstTerminal.time, firstTerminal.x),
          terminated: true,
        }
      }
      return {
        ...next,
        evaluations: next.evaluations + evaluations,
        events: [...s.events, ...hits],
        eventValues: g1,
        terminated: false,
      }
    },
    done: (s) => solver.done?.(s) ?? false,
  }
}
