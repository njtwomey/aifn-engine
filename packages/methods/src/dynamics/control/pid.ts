/**
 * A discrete PID controller closing a loop around a SISO plant, simulated as a traceable algorithm: filtered
 * derivative, actuator limits with anti-windup, set-point and load-disturbance inputs, and an optional input delay.
 */

import type { LtiSystem, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { discretise, toStateSpace } from 'aifn-compute/systems'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** PID gains: u = k_p e + k_i ∫e dt + k_d de/dt, the derivative filtered by a first-order lag of time constant T_f. */
export type PidGains = {
  kp: number
  ki?: number
  kd?: number
  /** Derivative filter time constant T_f (the derivative is k_d s/(1 + T_f s)). Default 0: unfiltered. */
  filter?: number
}

/** How the integrator is kept from winding up while the actuator saturates. */
export type AntiWindup = 'none' | 'clamp' | 'back-calculation'

/** Options for `pid`. */
export type PidOptions = {
  /** Controller sampling interval (the plant is advanced exactly between samples with the input held). */
  dt: number
  /** Set point r(t): a constant or a function of time. Default 1 (a unit step). */
  setpoint?: number | ((t: number) => number)
  /** Load disturbance d(t) added to the plant input. Default 0. */
  disturbance?: number | ((t: number) => number)
  /** Actuator limits; the applied input is clipped to [uMin, uMax] and `saturated` reports when. Default ±∞. */
  uMin?: number
  uMax?: number
  /** Anti-windup scheme. Default `clamp` (conditional integration). */
  antiWindup?: AntiWindup
  /** Tracking time constant T_t for back-calculation. Default √(T_i T_d) or T_i when k_d = 0 (Åström & Hägglund). */
  tracking?: number
  /** Derivative on the error (default) or on the measurement only (no derivative kick on set-point steps). */
  derivativeOn?: 'error' | 'measurement'
  /** A pure delay τ on the plant input, rounded to whole samples. Default 0. */
  delay?: number
  /** Stop at this time. */
  tEnd?: number
}

/** The state of `pidLoop`: `t` counts samples (the runner's `Status`), `time` is t · dt. */
export type PidState = Status & {
  /** Samples taken. */
  t: number
  /** The current time. */
  time: number
  /** Plant state x. */
  x: Vector
  /** Measured output y (the plant's first output). */
  y: number
  /** Set point r and error e = r − y. */
  r: number
  e: number
  /** The P, I and D contributions and their unsaturated sum. */
  proportional: number
  integral: number
  derivative: number
  uRaw: number
  /** The applied (clipped) input u, held until the next sample. */
  u: number
  saturated: boolean
  /** Computed inputs still in the delay line (length = delay in samples), oldest first. */
  pending: number[]
  /** The previous derivative input (e or −y), for the difference quotient. */
  previousD: number
  diverged: boolean
}

const value = (f: number | ((t: number) => number) | undefined, t: number, fallback: number) =>
  f === undefined ? fallback : typeof f === 'function' ? f(t) : f

/**
 * A PID loop around a SISO plant (continuous or discrete state space; the first input and output are used), in the
 * parallel form u = k_p e + k_i ∫e + k_d ė with the derivative filtered by 1/(1 + T_f s) and discretised by backward
 * Euler: D_k = (T_f D_{k−1} + k_d (e_k − e_{k−1})) / (T_f + dt) (Åström & Murray, 2021, "Feedback Systems", 2nd ed.,
 * §11.5). The integrator is updated by forward Euler after the output is computed. Anti-windup:
 * - `clamp`: skip the integrator update while the actuator is saturated and the error would drive it further in;
 * - `back-calculation`: add (u − u_raw)/T_t to the integrator's rate, bleeding it off while saturated.
 * `init` takes `{ x0 }` (default rest).
 */
export function pidLoop(
  plant: LtiSystem,
  gains: PidGains,
  options: PidOptions,
): Algorithm<{ x0?: VectorLike }, PidState> {
  const { dt } = options
  if (!(dt > 0)) throw new DomainError('pidLoop', 'pidLoop: dt must be positive')
  if (plant.dt !== null && Math.abs(plant.dt - dt) > 1e-12 * dt)
    throw new DomainError('pidLoop', 'pidLoop: a discrete plant must have the controller interval dt')
  const ss = (plant.domain === 'continuous' ? discretise(plant, dt, 'zoh') : toStateSpace(plant)).repr
  const n = ss.A.shape[0]
  const inputs = ss.B.shape[1]
  const A = dense.data(ss.A)
  const B = dense.data(ss.B)
  const b = Float64Array.from({ length: n }, (_, i) => B[i * inputs])
  const c = dense.data(ss.C).slice(0, n)
  // A feedthrough D would make u depend on y and y on u within one sample (an algebraic loop).
  if (dense.data(ss.D)[0] !== 0) throw new DomainError('pidLoop', 'pidLoop: the plant must be strictly proper (D = 0)')
  const kp = gains.kp
  const ki = gains.ki ?? 0
  const kd = gains.kd ?? 0
  const Tf = gains.filter ?? 0
  const uMin = options.uMin ?? -Infinity
  const uMax = options.uMax ?? Infinity
  const scheme = options.antiWindup ?? 'clamp'
  const Ti = ki !== 0 ? kp / ki : Infinity
  const Td = kp !== 0 ? kd / kp : 0
  const Tt = options.tracking ?? (Td > 0 && Number.isFinite(Ti) ? Math.sqrt(Ti * Td) : Number.isFinite(Ti) ? Ti : 1)
  const lag = Math.round((options.delay ?? 0) / dt)
  const onMeasurement = options.derivativeOn === 'measurement'
  const tEnd = options.tEnd ?? Infinity

  // The controller output at a sample, given the plant state and the controller's memory.
  const control = (
    step: number,
    x: Float64Array,
    integral: number,
    derivative: number,
    previousD: number | null,
    pending: number[],
  ): PidState => {
    const time = step * dt
    const y = dense.dot(c, x)
    const r = value(options.setpoint, time, 1)
    const e = r - y
    const dIn = onMeasurement ? -y : e
    const D = previousD === null ? 0 : (Tf * derivative + kd * (dIn - previousD)) / (Tf + dt)
    const P = kp * e
    const uRaw = P + integral + D
    const u = Math.min(uMax, Math.max(uMin, uRaw))
    const saturated = u !== uRaw
    return {
      t: step,
      time,
      x: fromData(Float64Array.from(x), [n]),
      y,
      r,
      e,
      proportional: P,
      integral,
      derivative: D,
      uRaw,
      u,
      saturated,
      pending,
      previousD: dIn,
      diverged: !Number.isFinite(y) || !x.every(Number.isFinite),
    }
  }

  return {
    name: 'pid-loop',
    init: ({ x0 } = {}) => {
      const x = x0 === undefined ? new Float64Array(n) : dense.toF64(x0, 'pidLoop x0')
      if (x.length !== n) throw new ShapeError('pidLoop', `pidLoop: x0 must have ${n} components`)
      return control(0, x, 0, 0, null, new Array(lag).fill(0))
    },
    step: (s) => {
      // The newly computed u enters the delay line; the plant receives the oldest entry (u itself without delay).
      const applied = lag > 0 ? s.pending[0] : s.u
      const line = lag > 0 ? [...s.pending.slice(1), s.u] : []
      const uPlant = applied + value(options.disturbance, s.time, 0)
      const x = dense.axpy(uPlant, b, dense.matVec(A, dense.data(s.x), n, n))
      let integral = s.integral
      if (scheme === 'back-calculation') integral += dt * (ki * s.e + (s.u - s.uRaw) / Tt)
      else if (scheme === 'clamp') {
        const pushingIn = s.saturated && Math.sign(s.e * ki) === Math.sign(s.uRaw - s.u)
        if (!pushingIn) integral += dt * ki * s.e
      } else integral += dt * ki * s.e
      return control(s.t + 1, x, integral, s.derivative, s.previousD, line)
    },
    done: (s) => s.time >= tEnd - 1e-9 * dt,
  }
}
