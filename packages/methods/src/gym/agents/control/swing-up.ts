/**
 * Model-based control of the inverted pendulum: an LQR controller about the upright, linearised with autodiff
 * Jacobians of the environment's differentiable dynamics, and an energy-based swing-up that hands over to it near the
 * top.
 *
 * - **Linearisation.** `lineariseDynamics(model, x̄, ū)` returns A = ∂f/∂x and B = ∂f/∂u of the discrete-time transition
 *   x_{k+1} = f(x_k, u_k) at (x̄, ū), by `jacobian` of `model.transition` (exact for the RK4 step, not a
 *   continuous-time approximation).
 * - **LQR.** `dlqr` (`aifn-compute/dynamics/control`, Riccati doubling) on (A, B) with the reward's weights Q = diag(1, 0.1),
 *   R = r (default 0.001, Gymnasium's) gives u = −K (θ, θ̇) with θ wrapped to [−π, π), clipped to the torque limit.
 *   Alone it balances only near the top: the linearisation and the torque limit fail far from it.
 * - **Energy swing-up** (Åström and Furuta, 2000, "Swinging up a pendulum by energy control", Automatica 36(2)). With
 *   e = ½θ̇² + (3g/2l) cos θ the specific energy, ė = θ̇ (3/ml²) u without damping, so u = k (e* − e) sign(θ̇), clipped,
 *   pumps e towards the upright's e* = 3g/2l monotonically. The agent switches to LQR once |θ| < `switchAngle`.
 *
 * Both act from the observation (cos θ, sin θ, θ̇). On a discrete-torque pendulum they play the nearest level.
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
 * x = (θ, θ̇), the gravity gain 3g/2l, the torque limit, and the torque of each discrete action (null for a box).
 */
export interface PendulumPlant {
  readonly model: DynamicsModel<unknown>
  readonly parameters: {
    readonly gravityGain: number
    readonly maxTorque: number
    readonly levels: readonly number[] | null
  }
}

/** A pendulum action: a torque as a length-1 array (box), or the index of a torque level (discrete). */
export type TorqueAction = Float64Array | number

/** The linearisation x_{k+1} − x̄′ ≈ A (x_k − x̄) + B (u_k − ū) of a dynamics model, A n×n and B n×m. */
export interface Linearisation {
  A: Matrix
  B: Matrix
}

/** A = ∂f/∂x and B = ∂f/∂u of `model.transition` at (x̄, ū), by automatic differentiation. */
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
  /** The energy gain k of the swing-up (default 1). */
  energyGain?: number
  /** Hand over to LQR once |θ| is below this (radians, default 0.6). */
  switchAngle?: number
  /** The LQR's torque weight R (default 0.001, as in the reward). */
  r?: number
}

/** The agent's state: the plant it controls and the LQR gain K [1 × 2] on (θ, θ̇). It learns nothing. */
export interface SwingUpState {
  plant: PendulumPlant['parameters']
  K: readonly number[]
}

/** The environment as a pendulum plant, or an error naming what is missing. */
function plantOf(env: EnvironmentShape): PendulumPlant {
  const plant = env as EnvironmentShape & Partial<PendulumPlant>
  if (plant.model?.kind !== 'dynamics' || typeof plant.parameters?.gravityGain !== 'number')
    throw new TypeError(`swingUpAgent: ${env.name} is not a pendulum (needs a dynamics model and its parameters)`)
  return plant as PendulumPlant
}

/**
 * The energy swing-up with an LQR hand-over (module docs), for a pendulum environment (box or discrete torques). `init`
 * reads the plant from the environment and computes the LQR gain from its model, once per run.
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

/** The gain of `swingUpAgent`'s LQR controller for a pendulum, and the linearisation it came from. */
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
