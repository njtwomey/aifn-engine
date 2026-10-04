/**
 * Target-tracking state-space models for `aifn-compute/inference/filtering`: the nearly-constant-velocity model with
 * continuous white-noise acceleration (Bar-Shalom, Li & Kirubarajan, 2001, "Estimation with Applications to Tracking
 * and Navigation", §6.2.2), observed through noisy positions.
 */

import type { StateSpaceModel } from 'aifn-compute/inference/filtering'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `constantVelocityModel`. */
export type ConstantVelocityOptions = {
  /** Spatial dimensions d (default 2). */
  dim?: number
  /** Time step Δt (default 1). */
  dt?: number
  /** Spectral density q of the white-noise acceleration, per axis (default 1). */
  processNoise?: number
  /** Standard deviation σ of each position measurement (default 1). */
  measurementStd?: number
  /** Mean and standard deviations of the initial state (defaults: at rest at the origin, sd 10 in position and 1 in velocity). */
  initialMean?: readonly number[]
  initialPositionStd?: number
  initialVelocityStd?: number
}

/**
 * The nearly-constant-velocity model in d dimensions, state z = (position₁…d, velocity₁…d): per axis
 * A = [[1, Δt], [0, 1]] and Q = q·[[Δt³/3, Δt²/2], [Δt²/2, Δt]] (the exact discretisation of white-noise acceleration
 * with spectral density q), and y = position + N(0, σ²I).
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
