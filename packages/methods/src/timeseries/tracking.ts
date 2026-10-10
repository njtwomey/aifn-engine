/**
 * Target-tracking state-space models for `aifn-compute/inference/filtering`: the nearly-constant-velocity model with
 * continuous white-noise acceleration (Bar-Shalom, Li & Kirubarajan, 2001, "Estimation with Applications to Tracking
 * and Navigation", §6.2.2), observed through noisy positions.
 */

import type { StateSpaceModel } from 'aifn-compute/inference/filtering'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `constantVelocityModel`. */
export type ConstantVelocityOptions = {
  /** The number of spatial dimensions $d$ (default 2). */
  dim?: number
  /** The time step $\Delta t$ between observations (default 1). */
  dt?: number
  /** The spectral density $q$ of the white-noise acceleration, per axis (default 1). */
  processNoise?: number
  /** The standard deviation $\sigma$ of each position measurement (default 1). */
  measurementStd?: number
  /** The mean $\mvec_0$ of the initial state, $2d$ values, positions first (default: at rest at the origin). */
  initialMean?: readonly number[]
  /** The standard deviation of each initial position (default 10). */
  initialPositionStd?: number
  /** The standard deviation of each initial velocity (default 1). */
  initialVelocityStd?: number
}

/**
 * The nearly-constant-velocity model in $d$ dimensions, with state $\zvec = (p_1, \dots, p_d, v_1, \dots, v_d)$,
 * positions then velocities: per axis $\Amat = \begin{pmatrix} 1 & \Delta t \\ 0 & 1 \end{pmatrix}$ and
 * $\Qmat = q \begin{pmatrix} \Delta t^3/3 & \Delta t^2/2 \\ \Delta t^2/2 & \Delta t \end{pmatrix}$ (the exact
 * discretisation of white-noise acceleration with spectral density $q$), and $\yvec = (p_1, \dots, p_d) + \vvec$,
 * $\vvec \sim \Gauss(\zeros, \sigma^2\Imat)$. The prior $\Pmat_0$ is diagonal. Throws `ShapeError` when
 * `initialMean` does not have $2d$ values.
 *
 * @param options The dimension, time step, noise levels and prior (see `ConstantVelocityOptions`).
 * @returns The model $\Amat$, $\Cmat$, $\Qmat$, $\Rmat$, $\mvec_0$, $\Pmat_0$ as rows of numbers, ready for
 *   `aifn-compute/inference/filtering`.
 *
 * @example One axis with a half-second step
 * const m = constantVelocityModel({ dim: 1, dt: 0.5 })
 * print('A =', m.A)
 * print('Q =', m.Q)
 * print('C =', m.C)
 *
 * @example In the plane the positions come first
 * const m = constantVelocityModel({ dim: 2, dt: 0.1 })
 * print('A =', m.A)
 * print('C =', m.C)
 */
export function constantVelocityModel(options: ConstantVelocityOptions = {}): StateSpaceModel {
  const {
    dim = 2,
    dt = 1,
    processNoise = 1,
    measurementStd = 1,
    initialPositionStd = 10,
    initialVelocityStd = 1,
  } = options
  const n = 2 * dim
  const zeros = () => Array.from({ length: n }, () => new Array<number>(n).fill(0))
  const A = zeros()
  const Q = zeros()
  const P0 = zeros()
  for (let i = 0; i < dim; i++) {
    const p = i
    const v = dim + i
    A[p][p] = 1
    A[p][v] = dt
    A[v][v] = 1
    Q[p][p] = (processNoise * dt ** 3) / 3
    Q[p][v] = Q[v][p] = (processNoise * dt ** 2) / 2
    Q[v][v] = processNoise * dt
    P0[p][p] = initialPositionStd ** 2
    P0[v][v] = initialVelocityStd ** 2
  }
  const C = Array.from({ length: dim }, (_, i) => Array.from({ length: n }, (_, j) => (j === i ? 1 : 0)))
  const R = Array.from({ length: dim }, (_, i) =>
    Array.from({ length: dim }, (_, j) => (i === j ? measurementStd ** 2 : 0)),
  )
  const m0 = options.initialMean ? [...options.initialMean] : new Array<number>(n).fill(0)
  if (m0.length !== n)
    throw new ShapeError(
      'constantVelocityModel',
      `constantVelocityModel: initialMean has ${m0.length} values for a state of ${n}`,
    )
  return { A, C, Q, R, m0, P0 }
}
