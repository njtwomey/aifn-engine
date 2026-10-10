/**
 * Model-based control of the inverted pendulum: an LQR controller about the upright, linearised with autodiff
 * Jacobians of the environment's differentiable dynamics, and an energy-based swing-up that hands over to it near the
 * top.
 *
 * - **Linearisation.** `lineariseDynamics(model, x, u)` returns $\Amat = \partial f / \partial \xvec$ and
 *   $\Bmat = \partial f / \partial \uvec$ of the discrete-time transition $\xvec_{k+1} = f(\xvec_k, \uvec_k)$ at a
 *   point $(\bar{\xvec}, \bar{\uvec})$, by `jacobian` of `model.transition` (exact for the model's own step, such as
 *   the pendulum's RK4 step, not a continuous-time approximation).
 * - **LQR.** `dlqr` (`aifn-compute/dynamics/control`, Riccati doubling) on $(\Amat, \Bmat)$ with the reward's weights
 *   $\Qmat = \diag(1, 0.1)$, $\Rmat = r$ (default 0.001, Gymnasium's) gives $u = -\Kmat (\theta, \dot\theta)$, with
 *   $\theta$ read from the observation in $(-\pi, \pi]$, clipped to the torque limit. Alone it balances only near the
 *   top: the linearisation and the torque limit fail far from it.
 * - **Energy swing-up** (Åström and Furuta, 2000, "Swinging up a pendulum by energy control", Automatica 36(2)). With
 *   $e = \tfrac{1}{2}\dot\theta^2 + (3g/2l) \cos\theta$ the specific energy, $\dot{e} = \dot\theta (3/ml^2) u$ without
 *   damping, so $u = k (e^* - e) \sgn(\dot\theta)$, clipped, pumps $e$ towards the upright's $e^* = 3g/2l$
 *   monotonically. The agent switches to LQR once $\lvert \theta \rvert <$ `switchAngle`.
 *
 * Both act from the observation $(\cos\theta, \sin\theta, \dot\theta)$. On a discrete-torque pendulum they play the
 * nearest level.
 */

import { dlqr } from 'aifn-compute/dynamics/control'
import { jacobian } from 'aifn-compute/foundation/autodiff'
import type {
  AgentInfo,
  Agent,
  Decision,
  DynamicsModel,
  EnvironmentShape,
  Value,
} from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, real, space } from 'aifn-compute/foundation/space'
import { fromData, slice, toRows, type Matrix, type Tensor } from 'aifn-compute/foundation/tensor'

/**
 * What the controllers need of a pendulum (`pendulumEnvironment` provides it): its differentiable model on
 * $\xvec = (\theta, \dot\theta)$, the gravity gain $3g/2l$, the torque limit, and the torque of each discrete action
 * (null for a box).
 */
export interface PendulumPlant {
  /** The differentiable dynamics on $\xvec = (\theta, \dot\theta)$ and the torque $u$, $\theta = 0$ upright. */
  readonly model: DynamicsModel<unknown>
  /**
   * The constants the controllers read: `gravityGain` $3g/2l$ (also the upright's specific energy), `maxTorque` the
   * torque limit, and `levels` the torque of each discrete action (null for a box of torques).
   */
  readonly parameters: {
    readonly gravityGain: number
    readonly maxTorque: number
    readonly levels: readonly number[] | null
  }
}

/** A pendulum action: a torque as a length-1 array (box), or the index of a torque level (discrete). */
export type TorqueAction = Float64Array | number

/**
 * The linearisation $\xvec_{k+1} - \bar{\xvec}' \approx \Amat (\xvec_k - \bar{\xvec}) + \Bmat (\uvec_k - \bar{\uvec})$
 * of a dynamics model, with $\bar{\xvec}' = f(\bar{\xvec}, \bar{\uvec})$, $\Amat$ $n \times n$ and $\Bmat$
 * $n \times m$.
 */
export interface Linearisation {
  /** $\Amat = \partial f / \partial \xvec$, $n \times n$. */
  A: Matrix
  /** $\Bmat = \partial f / \partial \uvec$, $n \times m$. */
  B: Matrix
}

/**
 * $\Amat = \partial f / \partial \xvec$ and $\Bmat = \partial f / \partial \uvec$ of `model.transition` at
 * $(\bar{\xvec}, \bar{\uvec})$, by automatic differentiation (one `jacobian` of the transition on the joined vector
 * $(\xvec, \uvec)$).
 *
 * @param model The differentiable dynamics; `transition`, `stateSize` $n$ and `actionSize` $m$ are read.
 * @param x The point $\bar{\xvec}$, $n$ values.
 * @param u The action $\bar{\uvec}$, $m$ values.
 * @returns $\Amat$ and $\Bmat$.
 *
 * @example The pendulum linearised upright and hanging down
 * // A pendulum (gravity gain 3g/2l = 15, torque gain 3) stepped by semi-implicit Euler with dt = 0.05.
 * const transition = (x, u) => {
 *   const w = add(slice(x, [1, 2]), mul(0.05, add(mul(15, sin(slice(x, [0, 1]))), mul(3, u))))
 *   return concat([add(slice(x, [0, 1]), mul(0.05, w)), w])
 * }
 * const model = { kind: 'dynamics', stateSize: 2, actionSize: 1, transition }
 * const up = lineariseDynamics(model, [0, 0], [0])
 * print('upright: A =', up.A, 'B =', up.B)
 * print('hanging: A =', lineariseDynamics(model, [Math.PI, 0], [0]).A)
 */
export function lineariseDynamics<S>(
  model: DynamicsModel<S>,
  x: ArrayLike<number>,
  u: ArrayLike<number>,
): Linearisation {
  const n = model.stateSize
  const m = model.actionSize
  const f = (z: Value) => model.transition(slice(z, [0, n]), slice(z, [n, n + m]))
  const z = fromData(Float64Array.from([...Array.from(x), ...Array.from(u)]), [n + m])
  const rows = toRows(jacobian(f)(z) as Tensor)
  return {
    A: fromData(Float64Array.from(rows.flatMap((r) => r.slice(0, n))), [n, n]),
    B: fromData(Float64Array.from(rows.flatMap((r) => r.slice(n))), [n, m]),
  }
}

/** Options of `swingUpAgent`. */
export interface SwingUpOptions {
  /** Pump energy until near the top (default true); false gives the LQR controller alone. */
  swingUp?: boolean
  /** The energy gain $k$ of the swing-up (default 1). */
  energyGain?: number
  /** Hand over to LQR once $\lvert \theta \rvert$ is below this (radians, default 0.6). */
  switchAngle?: number
  /** The LQR's torque weight $\Rmat$ (default 0.001, as in the reward). */
  r?: number
}

/**
 * The agent's state: the plant it controls and the LQR gain $\Kmat$ ($1 \times 2$) on $(\theta, \dot\theta)$. It
 * learns nothing.
 */
export interface SwingUpState {
  /** The pendulum's constants: gravity gain, torque limit and torque levels. */
  plant: PendulumPlant['parameters']
  /** The LQR gain $\Kmat$ on $(\theta, \dot\theta)$. */
  K: readonly number[]
}

/**
 * The environment as a pendulum plant. Throws `TypeError` when it has no dynamics model or no `parameters` with a
 * gravity gain.
 *
 * @param env The environment the agent is initialised for.
 * @returns The same object, typed as a `PendulumPlant`.
 */
function plantOf(env: EnvironmentShape): PendulumPlant {
  const plant = env as EnvironmentShape & Partial<PendulumPlant>
  if (plant.model?.kind !== 'dynamics' || typeof plant.parameters?.gravityGain !== 'number')
    throw new TypeError(`swingUpAgent: ${env.name} is not a pendulum (needs a dynamics model and its parameters)`)
  return plant as PendulumPlant
}

/**
 * The energy swing-up with an LQR hand-over (module docs), for a pendulum environment (box or discrete torques). `init`
 * reads the plant from the environment (`TypeError` when it is not a pendulum) and computes the LQR gain from its
 * model, once per run. `act` is deterministic: its `scores` hold 1 when the LQR acted and 0 for the swing-up. On
 * discrete torques it plays the level nearest the clipped torque. It learns nothing.
 *
 * @param options The swing-up and the hand-over.
 * @param options.swingUp Whether to pump energy until near the top; false gives the LQR controller alone.
 * @param options.energyGain The energy gain $k$ of $u = k (e^* - e) \sgn(\dot\theta)$.
 * @param options.switchAngle The angle from upright, in radians, below which the LQR takes over.
 * @param options.r The LQR's torque weight $\Rmat$.
 * @returns The agent, named `'energy swing-up + LQR'` or `'LQR'`.
 *
 * @example Swing up from hanging at rest
 * // A pendulum (gravity gain 3g/2l = 15, torque gain 3) stepped by semi-implicit Euler with dt = 0.05.
 * const transition = (x, u) => {
 *   const w = add(slice(x, [1, 2]), mul(0.05, add(mul(15, sin(slice(x, [0, 1]))), mul(3, u))))
 *   return concat([add(slice(x, [0, 1]), mul(0.05, w)), w])
 * }
 * const model = { kind: 'dynamics', stateSize: 2, actionSize: 1, transition }
 * const parameters = { gravityGain: 15, maxTorque: 2, levels: null }
 * const env = { name: 'pendulum', model, parameters }
 * const agent = swingUpAgent()
 * const g = agent.init(env, stream(0))
 * let x = [Math.PI, 0]
 * let lqrSteps = 0
 * for (let t = 0; t < 200; t++) {
 *   const d = agent.act(g, Float64Array.of(Math.cos(x[0]), Math.sin(x[0]), x[1]), stream(t))
 *   lqrSteps += d.scores[0]
 *   x = Array.from(transition(tensor(x), tensor(Array.from(d.action))).data)
 * }
 * print('angle from upright after 10 s:', Math.atan2(Math.sin(x[0]), Math.cos(x[0])))
 * print('steps under LQR:', lqrSteps)
 */
export function swingUpAgent({
  swingUp = true,
  energyGain = 1,
  switchAngle = 0.6,
  r = 0.001,
}: SwingUpOptions = {}): Agent<SwingUpState, Float64Array, TorqueAction> {
  const emit = (p: PendulumPlant['parameters'], u: number): TorqueAction => {
    const torque = Math.min(p.maxTorque, Math.max(-p.maxTorque, u))
    const levels = p.levels
    if (!levels) return Float64Array.of(torque)
    let best = 0
    for (let k = 1; k < levels.length; k++) if (Math.abs(levels[k] - torque) < Math.abs(levels[best] - torque)) best = k
    return best
  }
  /** The swing-up or LQR action from an observation; scores[0] says which acted: 1 for LQR, 0 for the swing-up. */
  const control = ({ plant: p, K }: SwingUpState, obs: Float64Array): Decision<TorqueAction> => {
    const theta = Math.atan2(obs[1], obs[0])
    const w = obs[2]
    if (!swingUp || Math.abs(theta) < switchAngle)
      return { action: emit(p, -(K[0] * theta + K[1] * w)), scores: Float64Array.of(1) }
    // Pump the specific energy towards the upright's, e* = 3g/2l.
    const e = 0.5 * w * w + p.gravityGain * Math.cos(theta)
    return { action: emit(p, energyGain * (p.gravityGain - e) * (w >= 0 ? 1 : -1)), scores: Float64Array.of(0) }
  }
  return {
    name: swingUp ? 'energy swing-up + LQR' : 'LQR',
    init: (env) => {
      const plant = plantOf(env)
      return { plant: plant.parameters, K: pendulumLqr(plant, r).K }
    },
    act: (g, obs) => control(g, obs),
    greedy: (g, obs) => control(g, obs).action,
    learn: (g) => g,
  }
}

/**
 * The gain of `swingUpAgent`'s LQR controller for a pendulum, and the linearisation it came from: the model linearised
 * at the upright at rest, and `dlqr` with $\Qmat = \diag(1, 0.1)$ and $\Rmat = r$.
 *
 * @param env The pendulum: only its `model` is read.
 * @param r The torque weight $\Rmat$.
 * @returns $\Amat$, $\Bmat$ and the gain $\Kmat$ ($u = -\Kmat (\theta, \dot\theta)$).
 *
 * @example The LQR gain balances the pendulum from a small tilt
 * // A pendulum (gravity gain 3g/2l = 15, torque gain 3) stepped by semi-implicit Euler with dt = 0.05.
 * const transition = (x, u) => {
 *   const w = add(slice(x, [1, 2]), mul(0.05, add(mul(15, sin(slice(x, [0, 1]))), mul(3, u))))
 *   return concat([add(slice(x, [0, 1]), mul(0.05, w)), w])
 * }
 * const model = { kind: 'dynamics', stateSize: 2, actionSize: 1, transition }
 * const { K } = pendulumLqr({ model, parameters: { gravityGain: 15, maxTorque: 2, levels: null } })
 * print('K =', K)
 * let x = tensor([0.3, 0])
 * for (let t = 0; t < 60; t++) x = transition(x, tensor([-(K[0] * x.data[0] + K[1] * x.data[1])]))
 * print('state after 3 s:', x)
 */
export function pendulumLqr(env: PendulumPlant, r = 0.001): Linearisation & { K: readonly number[] } {
  const lin = lineariseDynamics(env.model, [0, 0], [0])
  return {
    ...lin,
    K: toRows(
      dlqr(
        lin,
        [
          [1, 0],
          [0, 0.1],
        ],
        [[r]],
      ).K,
    )[0],
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<AgentInfo>('agent', 'gym/agents/control')(
  {
    key: 'swingUpAgent',
    name: 'Energy swing-up + LQR',
    summary:
      'Pumps the pendulum’s energy to the upright’s, then balances it with an LQR gain from autodiff Jacobians of the model.',
    params: space({
      swingUp: bool({ default: true, label: 'swing up' }),
      energyGain: real(0.1, 5, { default: 1, label: 'energy gain k' }),
      switchAngle: real(0.1, 1.5, { default: 0.6, label: 'switch angle (rad)' }),
      r: real(1e-4, 1, { default: 0.001, scale: 'log', label: 'torque weight R' }),
    }),
    requires: { observation: 'box', action: 'box', model: 'dynamics', families: ['control'] },
    notes: ['linear-quadratic-regulator', 'model-based-reinforcement-learning'],
  },
  swingUpAgent,
)
