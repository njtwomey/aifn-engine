/**
 * The inverted pendulum (Gymnasium's `Pendulum-v1`): a uniform rod of mass $m$ and length $l$ on a frictionless pivot,
 * driven by a torque $u$. The angle $\theta$ is measured from upright ($\theta = 0$ at the top, positive
 * anticlockwise), so
 * $\ddot\theta = \frac{3g}{2l} \sin\theta + \frac{3}{m l^2} u - c \dot\theta$,
 * with an optional viscous damping $c$ (0 in Gymnasium). The agent observes
 * $(\cos\theta, \sin\theta, \dot\theta)$ and earns $-(\theta^2 + 0.1 \dot\theta^2 + 0.001 u^2)$ per step with
 * $\theta$ wrapped to $[-\pi, \pi)$; an episode lasts 200 steps of $\Delta t = 0.05$ s.
 *
 * **Integrator.** One environment step is one classical RK4 step of size $\Delta t$ (`rungeKutta(…, 'rk4')` from
 * `aifn-compute/dynamics/ode`), with $u$ held constant over the step, then $\dot\theta$ clipped to $\pm$`maxSpeed`.
 * Gymnasium uses semi-implicit Euler ($\dot\theta \leftarrow \dot\theta + \Delta t\, \ddot\theta$, then
 * $\theta \leftarrow \theta + \Delta t\, \dot\theta$), so trajectories agree only to first order in $\Delta t$, not
 * exactly; RK4 conserves the undamped, unforced energy to $O(\Delta t^4)$ per unit time, which the tests check. The
 * step is written with tensor primitives, so `model.transition` is differentiable in $\xvec$ and $\uvec$ (autodiff
 * Jacobians for LQR, iLQR, MPC), and `step` calls that same transition.
 *
 * Actions are a box $[-u_{\max}, u_{\max}]$ ($u_{\max}$ the `maxTorque`; a length-1 `Float64Array`), or, with
 * `torques: n`, the $n$ evenly spaced torque levels of that interval as a discrete domain, for agents that need discrete
 * actions.
 *
 * Reference: Gymnasium, `gymnasium/envs/classic_control/pendulum.py` (Farama Foundation, 2023).
 */

import type {
  BoxDomain,
  DiscreteDomain,
  DynamicsModel,
  EnvironmentInfo,
  Environment,
  PendulumRender,
  Value,
} from 'aifn-compute/foundation/contracts'
import { run } from 'aifn-compute/foundation/trace'
import { rungeKutta } from 'aifn-compute/dynamics/ode'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { boxDomain, discreteDomain, int, real, space } from 'aifn-compute/foundation/space'
import {
  add,
  clip,
  fromData,
  get,
  mul,
  neg,
  sin,
  square,
  stack,
  sub,
  toFlat,
  unwrap,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The pendulum's state: the angle $\theta$ and the angular velocity $\dot\theta$. */
export interface PendulumState {
  /** The angle $\theta$ from upright, anticlockwise, in radians (not wrapped). */
  theta: number
  /** The angular velocity $\dot\theta$, in rad/s. */
  thetaDot: number
}

/** A pendulum action: a torque as a length-1 array (box), or the index of a torque level (discrete). */
export type PendulumAction = Float64Array | number

/** Options of `pendulumEnvironment`; the defaults are Gymnasium's `Pendulum-v1`. */
export interface PendulumOptions {
  /** Gravity $g$ (m/s$^2$, default 10). */
  g?: number
  /** Rod mass $m$ (kg, default 1). */
  mass?: number
  /** Rod length $l$ (m, default 1). */
  length?: number
  /** Viscous damping $c$ in $\ddot\theta$ (1/s, default 0). */
  damping?: number
  /** Torque limit $u_{\max}$ (N m, default 2): actions are clipped to $[-u_{\max}, u_{\max}]$. */
  maxTorque?: number
  /** Speed limit (rad/s, default 8): $\dot\theta$ is clipped to $\pm$ this after each step. */
  maxSpeed?: number
  /** Time step $\Delta t$ (s, default 0.05). */
  dt?: number
  /** Episode length in steps (default 200). */
  horizon?: number
  /** Discount the problem is posed with (default 0.99). */
  gamma?: number
  /**
   * The number $n \ge 2$ of discrete torque levels, evenly spaced on $[-u_{\max}, u_{\max}]$; 0 or absent for a box
   * action.
   */
  torques?: number
  /**
   * A fixed start state; by default $\theta \sim \Unif[-\pi, \pi)$ and $\dot\theta \sim \Unif[-1, 1)$, as in
   * Gymnasium.
   */
  start?: PendulumState
}

/** The physical constants of a pendulum, resolved from its options. */
export interface PendulumParameters {
  /** Gravity $g$ (m/s$^2$). */
  g: number
  /** Rod mass $m$ (kg). */
  mass: number
  /** Rod length $l$ (m). */
  length: number
  /** Viscous damping $c$ (1/s). */
  damping: number
  /** Torque limit $u_{\max}$ (N m). */
  maxTorque: number
  /** Speed limit on $\lvert \dot\theta \rvert$ (rad/s). */
  maxSpeed: number
  /** Time step $\Delta t$ (s). */
  dt: number
  /** Gravity's gain $3g / 2l$ in $\ddot\theta$ (1/s$^2$). */
  gravityGain: number
  /** The torque's gain $3 / m l^2$ in $\ddot\theta$ (1/(kg m$^2$)). */
  torqueGain: number
  /** The torque of each discrete action, or null for a box action. */
  levels: readonly number[] | null
}

/** A pendulum environment, with its parameters for model-based agents and the lab. */
export interface PendulumEnvironment extends Environment<PendulumState, Float64Array, PendulumAction> {
  /** The constants it was built with. */
  readonly parameters: PendulumParameters
  /**
   * The differentiable dynamics on $\xvec = (\theta, \dot\theta)$ and $\uvec = (u)$: one RK4 step and the reward.
   */
  readonly model: DynamicsModel<PendulumState>
  /** The drawing of the rod, with $\theta$ (wrapped) and $\dot\theta$ as series. */
  readonly render: PendulumRender<PendulumState>
}

const TWO_PI = 2 * Math.PI

/**
 * An angle wrapped to $[-\pi, \pi)$.
 *
 * @param theta The angle $\theta$, in radians.
 * @returns $\theta - 2\pi k$ for the integer $k$ that puts it in $[-\pi, \pi)$.
 *
 * @example Angles past a half turn
 * print('wrap(1.5 pi) =', wrapAngle(1.5 * Math.PI))
 * print('wrap(pi) =', wrapAngle(Math.PI), '; wrap(-pi) =', wrapAngle(-Math.PI))
 * print('wrap(7) =', wrapAngle(7))
 */
export const wrapAngle = (theta: number): number => theta - TWO_PI * Math.floor((theta + Math.PI) / TWO_PI)

/**
 * $\theta$ wrapped to $[-\pi, \pi)$ as a value: $\theta$ minus a multiple of $2\pi$ read from its primal, so the
 * derivative is 1 (the shift is piecewise constant).
 *
 * @param theta The angle $\theta$ in radians: a number, or a scalar value that may be traced.
 * @returns The wrapped angle, `theta` itself when it is already in range.
 */
function wrapValue(theta: Value): Value {
  const raw = unwrap(theta)
  const t = typeof raw === 'number' ? raw : toFlat(raw)[0]
  const k = Math.floor((t + Math.PI) / TWO_PI)
  return k === 0 ? theta : sub(theta, TWO_PI * k)
}

/**
 * The specific mechanical energy $\frac12 \dot\theta^2 + \frac{3g}{2l} \cos\theta$ (the energy divided by the rod's
 * moment of inertia $m l^2 / 3$): constant without torque and damping, $3g / 2l$ at rest upright and $-3g / 2l$ at rest
 * hanging.
 *
 * @param p The pendulum's constants (its `parameters`); only `gravityGain` is read.
 * @param s The state.
 * @returns The energy, in 1/s$^2$.
 *
 * @example Energy is conserved by the free swing
 * const env = pendulumEnvironment({ start: { theta: 1, thetaDot: 0 } })
 * const s = stream(0)
 * let { state } = env.reset(s)
 * print('at the start', pendulumEnergy(env.parameters, state))
 * for (let k = 0; k < 20; k++) state = env.step(state, Float64Array.of(0), s).state
 * print('after 20 steps', pendulumEnergy(env.parameters, state))
 * print('upright at rest', pendulumEnergy(env.parameters, { theta: 0, thetaDot: 0 }))
 */
export function pendulumEnergy(p: PendulumParameters, s: PendulumState): number {
  return 0.5 * s.thetaDot * s.thetaDot + p.gravityGain * Math.cos(s.theta)
}

/**
 * The pendulum of Gymnasium's `Pendulum-v1`, stepped by RK4 on compute's ODE solver (see the file's introduction).
 * Throws `DomainError` when `torques` is neither 0 nor an integer of at least 2, and `step` throws for a number action
 * on a box pendulum (`TypeError`) or an index with no torque level (`DomainError`).
 *
 * @param options The physical constants, the action domain, the start and the episode; every field defaults to
 *   Gymnasium's value.
 * @returns The environment: state $(\theta, \dot\theta)$, observation $(\cos\theta, \sin\theta, \dot\theta)$, never
 *   terminated (the rollout truncates it at `horizon`), with its `parameters`, a dynamics `model` and a `render`.
 *
 * @example Full torque from hanging at rest
 * const env = pendulumEnvironment({ start: { theta: Math.PI, thetaDot: 0 } })
 * const s = stream(0)
 * const { state, observation } = env.reset(s)
 * print('observation', observation)
 * const a = env.step(state, Float64Array.of(2), s)
 * const b = env.step(a.state, Float64Array.of(2), s)
 * print('rewards', a.reward, b.reward)
 * print('state after two steps', b.state)
 *
 * @example Discrete torque levels
 * const env = pendulumEnvironment({ torques: 5 })
 * print('actions', env.action.names)
 * print('torques', env.parameters.levels)
 */
export function pendulumEnvironment(options: PendulumOptions = {}): PendulumEnvironment {
  const {
    g = 10,
    mass = 1,
    length = 1,
    damping = 0,
    maxTorque = 2,
    maxSpeed = 8,
    dt = 0.05,
    horizon = 200,
    gamma = 0.99,
    torques = 0,
    start,
  } = options
  if (torques !== 0 && !(Number.isInteger(torques) && torques >= 2))
    throw new DomainError(
      'pendulumEnvironment',
      `pendulumEnvironment: torques must be 0 (a box) or an integer ≥ 2, got ${torques}`,
    )
  const levels = torques
    ? Array.from({ length: torques }, (_, k) => -maxTorque + (2 * maxTorque * k) / (torques - 1))
    : null
  const parameters: PendulumParameters = {
    g,
    mass,
    length,
    damping,
    maxTorque,
    maxSpeed,
    dt,
    gravityGain: (3 * g) / (2 * length),
    torqueGain: 3 / (mass * length * length),
    levels,
  }
  const { gravityGain: a, torqueGain: b } = parameters

  /** One RK4 step from $\xvec = (\theta, \dot\theta)$ under torque $\uvec$ (length 1), then the speed clip. */
  function transition(x: Value, u: Value): Value {
    const torque = clip(get(u, 0), -maxTorque, maxTorque)
    const rhs = (_t: number, y: Tensor) => {
      const w = get(y, 1)
      const accel = add(mul(a, sin(get(y, 0))), mul(b, torque))
      return stack([w, damping === 0 ? accel : sub(accel, mul(damping, w))])
    }
    const y = run(rungeKutta(rhs, 'rk4', { stepSize: dt }), { x0: x as Tensor }, 1).x as Value
    return stack([get(y, 0), clip(get(y, 1), -maxSpeed, maxSpeed)])
  }

  /** $-(\theta^2 + 0.1 \dot\theta^2 + 0.001 u^2)$ with $\theta$ wrapped and $u$ clipped, at the state acted in. */
  function reward(x: Value, u: Value): Value {
    const torque = clip(get(u, 0), -maxTorque, maxTorque)
    return neg(add(add(square(wrapValue(get(x, 0))), mul(0.1, square(get(x, 1)))), mul(0.001, square(torque))))
  }

  const encode = (s: PendulumState): Tensor => fromData(Float64Array.of(s.theta, s.thetaDot), [2])
  const decode = (x: Value): PendulumState => {
    const v = toFlat(unwrap(x) as Tensor)
    return { theta: v[0], thetaDot: v[1] }
  }
  const observe = (s: PendulumState) => Float64Array.of(Math.cos(s.theta), Math.sin(s.theta), s.thetaDot)
  const torqueOf = (action: PendulumAction): number => {
    if (typeof action === 'number') {
      if (!levels) throw new TypeError('pendulumEnvironment: a box pendulum takes a torque array, got a number')
      const l = levels[action]
      if (l === undefined)
        throw new DomainError('pendulumEnvironment', `pendulumEnvironment: no torque level ${action}`)
      return l
    }
    return action[0]
  }

  const action: BoxDomain | DiscreteDomain = levels
    ? discreteDomain(
        levels.length,
        levels.map((l) => `${l > 0 ? '+' : ''}${+l.toFixed(2)}`),
      )
    : boxDomain([-maxTorque], [maxTorque], { names: ['torque'] })

  return {
    name: levels ? `pendulum (${levels.length} torques)` : 'pendulum',
    observation: boxDomain([-1, -1, -maxSpeed], [1, 1, maxSpeed], { names: ['cos θ', 'sin θ', 'θ̇'] }),
    action,
    gamma,
    horizon,
    parameters,
    reset(stream: Stream) {
      const state = start ?? {
        theta: uniform(stream, -Math.PI, Math.PI),
        thetaDot: uniform(stream, -1, 1),
      }
      return { state, observation: observe(state) }
    },
    step(state, act) {
      const u = fromData(Float64Array.of(torqueOf(act)), [1])
      const x = encode(state)
      const r = reward(x, u) as number
      const next = decode(transition(x, u))
      return { state: next, observation: observe(next), reward: r, terminated: false, truncated: false }
    },
    model: { kind: 'dynamics', stateSize: 2, actionSize: 1, transition, reward, encode, decode },
    render: {
      kind: 'pendulum',
      length,
      angle: (s) => s.theta,
      series: [
        { name: 'θ (rad)', value: (s) => wrapAngle(s.theta) },
        { name: 'θ̇ (rad/s)', value: (s) => s.thetaDot },
      ],
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<EnvironmentInfo>('environment', 'gym/environments/control')(
  {
    key: 'pendulumEnvironment',
    name: 'Inverted pendulum',
    summary:
      'Gymnasium’s Pendulum-v1: swing a torque-limited rod up and balance it; RK4 dynamics with a differentiable model.',
    family: 'control',
    params: space({
      maxTorque: real(0.5, 4, { default: 2, label: 'torque limit' }),
      damping: real(0, 1, { default: 0, label: 'damping c' }),
      torques: int(0, 9, { default: 0, label: 'torque levels (0: box)' }),
      horizon: int(1, 10000, { default: 200 }),
      gamma: real(0, 1, { default: 0.99, label: 'γ' }),
    }),
    observation: 'box',
    action: 'box',
    capabilities: ['model', 'render'],
    notes: ['model-based-reinforcement-learning', 'linear-quadratic-regulator'],
  },
  pendulumEnvironment,
)
