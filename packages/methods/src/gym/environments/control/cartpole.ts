/**
 * The cart-pole (Gymnasium's `CartPole-v1`, after Barto, Sutton and Anderson, 1983, "Neuronlike adaptive elements that
 * can solve difficult learning control problems", IEEE Trans. SMC 13(5)): a pole hinged on a cart that moves along a
 * frictionless track. The agent pushes the cart left or right with a force of $\pm 10$ N and earns $+1$ for every step
 * the pole stays up. The episode terminates when $\lvert\theta\rvert > 12^\circ$ or $\lvert x \rvert > 2.4$ m, and is
 * truncated at 500 steps.
 *
 * With total mass $M = m_c + m_p$, pole half-length $l$ and force $F$ (Florian, 2007, "Correct equations for the
 * dynamics of the cart-pole system", the equations Gymnasium uses, without friction):
 *
 * - $q = (F + m_p l \dot\theta^2 \sin\theta) / M$,
 * - $\ddot\theta = (g \sin\theta - q \cos\theta) / \big(l (4/3 - m_p \cos^2\theta / M)\big)$,
 * - $\ddot x = q - m_p l \ddot\theta \cos\theta / M$.
 *
 * $\theta$ is measured from upright, positive towards $+x$. **Integrator: explicit Euler, as Gymnasium** (`euler`,
 * $\tau = 0.02$ s): $x \leftarrow x + \tau \dot x$, $\dot x \leftarrow \dot x + \tau \ddot x$,
 * $\theta \leftarrow \theta + \tau \dot\theta$, $\dot\theta \leftarrow \dot\theta + \tau \ddot\theta$, every
 * right-hand side at the old state. So trajectories match Gymnasium's to rounding. The step is written with tensor
 * primitives, so `model.transition(x, u)` ($u$ a force in newtons, any real value) is differentiable for LQR and MPC,
 * and `step` calls the same equations on numbers.
 *
 * Reference: Gymnasium, `gymnasium/envs/classic_control/cartpole.py` (Farama Foundation, 2023).
 */

import type {
  BoxDomain,
  CartPoleRender,
  DiscreteDomain,
  DynamicsModel,
  Environment,
  EnvironmentInfo,
  Value,
} from 'aifn-compute/foundation/contracts'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { boxDomain, discreteDomain, int, real, space } from 'aifn-compute/foundation/space'
import {
  add,
  cos,
  div,
  fromData,
  get,
  mul,
  sin,
  square,
  stack,
  sub,
  toFlat,
  unwrap,
} from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The cart-pole's state: the cart's position $x$ and velocity $\dot x$, the pole's angle $\theta$ and rate. */
export interface CartPoleState {
  /** The cart's position $x$ on the track (m). */
  x: number
  /** The cart's velocity $\dot x$ (m/s). */
  xDot: number
  /** The pole's angle $\theta$ from upright, positive towards $+x$ (rad). */
  theta: number
  /** The pole's angular velocity $\dot\theta$ (rad/s). */
  thetaDot: number
}

/** Options of `cartPoleEnvironment`; the defaults are Gymnasium's `CartPole-v1`. */
export interface CartPoleOptions {
  /** Gravity $g$ (m/s$^2$, default 9.8). */
  g?: number
  /** Cart mass (kg, default 1). */
  cartMass?: number
  /** Pole mass (kg, default 0.1). */
  poleMass?: number
  /** Pole half-length $l$ (m, default 0.5). */
  halfLength?: number
  /** The push force (N, default 10): action 0 pushes left with $-$`force`, action 1 right with $+$`force`. */
  force?: number
  /** Time step $\tau$ (s, default 0.02). */
  tau?: number
  /** The pole falls when $\lvert\theta\rvert$ exceeds this (rad, default $12^\circ = 0.2094$). */
  thetaLimit?: number
  /** The cart leaves the track when $\lvert x \rvert$ exceeds this (m, default 2.4). */
  trackLimit?: number
  /** Episode length in steps (default 500). */
  horizon?: number
  /** Discount the problem is posed with (default 0.99). */
  gamma?: number
  /** Each state element starts uniform on $[-\text{jitter}, \text{jitter}]$ (default 0.05). */
  jitter?: number
}

/** The constants of a cart-pole, resolved from its options. */
export interface CartPoleParameters {
  /** Gravity $g$ (m/s$^2$). */
  g: number
  /** Cart mass $m_c$ (kg). */
  cartMass: number
  /** Pole mass $m_p$ (kg). */
  poleMass: number
  /** Pole half-length $l$ (m). */
  halfLength: number
  /** The push force (N). */
  force: number
  /** Time step $\tau$ (s). */
  tau: number
  /** The largest $\lvert\theta\rvert$ before the pole falls (rad). */
  thetaLimit: number
  /** The largest $\lvert x \rvert$ before the cart leaves the track (m). */
  trackLimit: number
  /** The half-width of the uniform start distribution of each state element. */
  jitter: number
}

/**
 * A cart-pole environment: box observations $(x, \dot x, \theta, \dot\theta)$, two actions (push left, push right).
 */
export interface CartPoleEnvironment extends Environment<CartPoleState, Float64Array, number> {
  /** The constants it was built with. */
  readonly parameters: CartPoleParameters
  /**
   * The differentiable dynamics on $\xvec = (x, \dot x, \theta, \dot\theta)$ and $\uvec = (F)$, a force in newtons:
   * one Euler step, and a reward of 1.
   */
  readonly model: DynamicsModel<CartPoleState>
  /** The drawing of the cart and pole, with $x$ and $\theta$ as series. */
  readonly render: CartPoleRender<CartPoleState>
}

/**
 * One Euler step of the cart-pole under force $F$, on numbers or traced values alike (the equations of the file's
 * introduction).
 *
 * @param p The cart-pole's constants.
 * @param x The cart's position $x$ (m).
 * @param xDot The cart's velocity $\dot x$ (m/s).
 * @param theta The pole's angle $\theta$ from upright (rad).
 * @param thetaDot The pole's angular velocity $\dot\theta$ (rad/s).
 * @param force The force $F$ on the cart (N), positive towards $+x$.
 * @returns The next $(x, \dot x, \theta, \dot\theta)$, numbers for numbers in and traced values otherwise.
 */
function eulerStep(p: CartPoleParameters, x: Value, xDot: Value, theta: Value, thetaDot: Value, force: Value) {
  const total = p.cartMass + p.poleMass
  const pml = p.poleMass * p.halfLength
  const s = sin(theta)
  const c = cos(theta)
  const q = div(add(force, mul(pml, mul(square(thetaDot), s))), total)
  const thetaAcc = div(
    sub(mul(p.g, s), mul(c, q)),
    mul(p.halfLength, sub(4 / 3, div(mul(p.poleMass, square(c)), total))),
  )
  const xAcc = sub(q, div(mul(pml, mul(thetaAcc, c)), total))
  return [
    add(x, mul(p.tau, xDot)),
    add(xDot, mul(p.tau, xAcc)),
    add(theta, mul(p.tau, thetaDot)),
    add(thetaDot, mul(p.tau, thetaAcc)),
  ] as const
}

/**
 * The cart-pole of Gymnasium's `CartPole-v1` (see the file's introduction). `step` pays 1 per step and throws
 * `DomainError` for an action other than 0 or 1.
 *
 * @param options The physical constants, the limits, the start and the episode; every field defaults to Gymnasium's
 *   value.
 * @returns The environment: state and observation $(x, \dot x, \theta, \dot\theta)$, with its `parameters`, a
 *   dynamics `model` and a `render`.
 *
 * @example Push right until the pole falls
 * const env = cartPoleEnvironment()
 * const s = stream(0)
 * let { state } = env.reset(s)
 * let steps = 0
 * let done = false
 * while (!done && steps < 100) {
 *   const step = env.step(state, 1, s)
 *   state = step.state
 *   done = step.terminated
 *   steps++
 * }
 * print('terminated after', steps, 'steps')
 * print('ending', env.ending(state, 'terminated', steps).reason)
 *
 * @example The differentiable model's Jacobian at upright
 * const env = cartPoleEnvironment()
 * const x0 = tensor([0, 0, 0, 0])
 * print('d next / d u', jacobian((u) => env.model.transition(x0, u))(tensor([0])))
 */
export function cartPoleEnvironment(options: CartPoleOptions = {}): CartPoleEnvironment {
  const {
    g = 9.8,
    cartMass = 1,
    poleMass = 0.1,
    halfLength = 0.5,
    force = 10,
    tau = 0.02,
    thetaLimit = (12 * 2 * Math.PI) / 360,
    trackLimit = 2.4,
    horizon = 500,
    gamma = 0.99,
    jitter = 0.05,
  } = options
  const parameters: CartPoleParameters = {
    g,
    cartMass,
    poleMass,
    halfLength,
    force,
    tau,
    thetaLimit,
    trackLimit,
    jitter,
  }

  const transition = (x: Value, u: Value): Value =>
    stack([...eulerStep(parameters, get(x, 0), get(x, 1), get(x, 2), get(x, 3), get(u, 0))])
  const encode = (s: CartPoleState) => fromData(Float64Array.of(s.x, s.xDot, s.theta, s.thetaDot), [4])
  const decode = (x: Value): CartPoleState => {
    const raw = unwrap(x)
    const v = typeof raw === 'number' ? [raw] : toFlat(raw)
    return { x: v[0], xDot: v[1], theta: v[2], thetaDot: v[3] }
  }
  const observe = (s: CartPoleState) => Float64Array.of(s.x, s.xDot, s.theta, s.thetaDot)
  const action: DiscreteDomain = discreteDomain(2, ['push left', 'push right'])
  const observation: BoxDomain = boxDomain(
    [-2 * trackLimit, -Infinity, -2 * thetaLimit, -Infinity],
    [2 * trackLimit, Infinity, 2 * thetaLimit, Infinity],
    { names: ['x', 'ẋ', 'θ', 'θ̇'] },
  )

  return {
    name: 'cart-pole',
    observation,
    action,
    gamma,
    horizon,
    parameters,
    reset(stream: Stream) {
      const v = toFlat(uniform(stream, -jitter, jitter, { shape: [4] }))
      const state = { x: v[0], xDot: v[1], theta: v[2], thetaDot: v[3] }
      return { state, observation: observe(state) }
    },
    step(s, a) {
      if (a !== 0 && a !== 1)
        throw new DomainError('cartPoleEnvironment', `cartPoleEnvironment: action must be 0 or 1, got ${a}`)
      const [x, xDot, theta, thetaDot] = eulerStep(
        parameters,
        s.x,
        s.xDot,
        s.theta,
        s.thetaDot,
        a === 1 ? force : -force,
      )
      const next = { x, xDot, theta, thetaDot } as CartPoleState
      const terminated = Math.abs(next.x) > trackLimit || Math.abs(next.theta) > thetaLimit
      return { state: next, observation: observe(next), reward: 1, terminated, truncated: false }
    },
    ending(s, how, steps) {
      if (how === 'truncated') return { success: true, reason: `survived ${steps} steps` }
      const deg = (r: number) => ((r * 180) / Math.PI).toFixed(1)
      return Math.abs(s.theta) > thetaLimit
        ? { success: false, reason: `pole fell: θ = ${deg(s.theta)}°` }
        : { success: false, reason: `cart left the track: x = ${s.x.toFixed(2)} m` }
    },
    model: { kind: 'dynamics', stateSize: 4, actionSize: 1, transition, reward: () => 1, encode, decode },
    render: {
      kind: 'cartpole',
      poleLength: 2 * halfLength,
      trackLimit,
      cart: (s) => s.x,
      angle: (s) => s.theta,
      series: [
        { name: 'x (m)', value: (s) => s.x },
        { name: 'θ (rad)', value: (s) => s.theta },
      ],
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<EnvironmentInfo>('environment', 'gym/environments/control')(
  {
    key: 'cartPoleEnvironment',
    name: 'Cart-pole',
    summary:
      'Gymnasium’s CartPole-v1: push a cart left or right to keep its pole up for 500 steps; Euler dynamics with a differentiable model.',
    family: 'control',
    params: space({
      force: real(1, 30, { default: 10, label: 'push force (N)' }),
      jitter: real(0, 0.2, { default: 0.05, label: 'initial jitter' }),
      horizon: int(1, 10000, { default: 500 }),
      gamma: real(0, 1, { default: 0.99, label: 'γ' }),
    }),
    observation: 'box',
    action: 'discrete',
    capabilities: ['model', 'render'],
    notes: ['reinforcement-learning', 'linear-quadratic-regulator'],
  },
  cartPoleEnvironment,
)
