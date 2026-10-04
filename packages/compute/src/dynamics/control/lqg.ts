/**
 * Linear-quadratic-Gaussian (LQG) control: the LQR gain applied to the Kalman filter's state estimate. By the
 * separation principle the two are designed independently, and the closed-loop poles are those of A − BK and of
 * A − LC together (Athans, 1971, "The role and use of the stochastic linear-quadratic-Gaussian problem in control
 * system design", IEEE TAC 16(6); Anderson & Moore, 1990, "Optimal Control: Linear Quadratic Methods", ch. 8).
 *
 * The steady-state Kalman gain is the LQR gain of the dual problem (Aᵀ, Cᵀ) with weights (W, V) (Kalman, 1960), so it is
 * computed by the same Riccati solvers (`lqr`, `dlqr`) rather than a second implementation.
 */

import { dense, fromData, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { child, normal, type Stream } from 'aifn-compute/foundation/random'
import { cholesky, eig } from 'aifn-compute/numerics/linalg'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dlqr, lqr, type LqrResult } from './lqr'

type F64 = dense.F64

/** A plant with process and measurement noise: x′ = Ax + Bu + w, y = Cx + v, w ~ N(0, W), v ~ N(0, V). */
export type LqgPlant = { A: MatrixLike; B: MatrixLike; C: MatrixLike }

/** The weights and noise covariances of an LQG problem. */
export type LqgWeights = {
  /** State and input weights of the cost. */
  Q: MatrixLike
  R: MatrixLike
  /** Process and measurement noise covariances. */
  W: MatrixLike
  V: MatrixLike
}

/** An LQG design. */
export type LqgDesign = {
  /** The LQR gain: u = −K x̂. */
  K: Matrix
  /**
   * The steady-state Kalman gain. Continuous: x̂′ = Ax̂ + Bu + L(y − Cx̂). Discrete (predictor form):
   * x̂_{k+1} = Ax̂_k + Bu_k + L(y_k − Cx̂_k).
   */
  L: Matrix
  /** The control Riccati solution and the steady-state (prior) error covariance. */
  P: Matrix
  S: Matrix
  /** The poles of the regulator A − BK and of the estimator A − LC (complex128). */
  regulatorPoles: Tensor
  estimatorPoles: Tensor
  /** The output-feedback compensator from y to u: A − BK − LC, L, −K, 0. */
  controller: { A: Matrix; B: Matrix; C: Matrix; D: Matrix }
  converged: boolean
}

const T = (a: F64, m: number, n: number) => fromData(dense.transpose(a, m, n), [n, m])

/**
 * The LQG design for a continuous plant, or a discrete one with `discrete: true` (Anderson & Moore, 1990, ch. 8):
 * K from the LQR of (A, B, Q, R) and L from the dual LQR of (Aᵀ, Cᵀ, W, V), L = Kᵈᵤₐₗᵀ. Continuous: L = SCᵀV⁻¹ with S the
 * filter CARE's solution. Discrete: L = ASCᵀ(CSCᵀ + V)⁻¹, the predictor gain, with S the prior covariance.
 */
export function lqg(
  plant: LqgPlant,
  weights: LqgWeights,
  { discrete = false }: { discrete?: boolean } = {},
): LqgDesign {
  const a = dense.toMatrixF64(plant.A, 'lqg A')
  const n = a.m
  const b = dense.toMatrixF64(plant.B, 'lqg B', n)
  const c = dense.toMatrixF64(plant.C, 'lqg C', undefined, n)
  const m = b.n
  const p = c.m
  const solve = discrete ? dlqr : lqr
  const ctrl: LqrResult = solve({ A: plant.A, B: plant.B }, weights.Q, weights.R)
  const est: LqrResult = solve({ A: T(a.data, n, n), B: T(c.data, p, n) }, weights.W, weights.V)
  if (!ctrl.converged || !est.converged)
    throw new DomainError('lqg', 'lqg: a Riccati equation did not converge (is the plant stabilisable and detectable?)')
  const K = dense.data(ctrl.K)
  const L = dense.transpose(dense.data(est.K), p, n)
  const BK = dense.matMul(b.data, K, n, m, n)
  const LC = dense.matMul(L, c.data, n, p, n)
  const regulator = dense.sub(a.data, BK)
  const estimator = dense.sub(a.data, LC)
  const Ac = dense.sub(regulator, LC)
  return {
    K: ctrl.K,
    L: fromData(L, [n, p]),
    P: ctrl.P,
    S: est.P,
    regulatorPoles: eig(fromData(regulator, [n, n]), { vectors: false }).values,
    estimatorPoles: eig(fromData(estimator, [n, n]), { vectors: false }).values,
    controller: {
      A: fromData(Ac, [n, n]),
      B: fromData(L, [n, p]),
      C: fromData(dense.scale(-1, K), [m, n]),
      D: fromData(new Float64Array(m * p), [m, p]),
    },
    converged: true,
  }
}

/** The state of `lqgSimulation`. */
export interface LqgSimulationState extends Status {
  t: Size
  /** The true state, the estimate and the estimation error x − x̂. */
  x: Vector
  xHat: Vector
  /** The measurement y_k = Cx_k + v_k and the input u_k = −Kx̂_k. */
  y: Vector
  u: Vector
  /** Running cost Σ xᵀQx + uᵀRu. */
  cost: Scalar
  diverged: boolean
}

/**
 * A discrete LQG loop as a traceable algorithm: the plant x_{k+1} = Ax_k + Bu_k + w_k, y_k = Cx_k + v_k with Gaussian
 * noise drawn from the step's stream, the predictor-form estimator and u_k = −Kx̂_k. With `feedback: 'state'` the
 * regulator uses the true state instead (the full-information LQR), for comparison. `init` takes `{ x0, xHat0 }`.
 */
export function lqgSimulation(
  plant: LqgPlant,
  weights: LqgWeights,
  design: LqgDesign,
  { feedback = 'estimate' }: { feedback?: 'estimate' | 'state' } = {},
): Algorithm<{ x0: VectorLike; xHat0?: VectorLike }, LqgSimulationState> {
  const a = dense.toMatrixF64(plant.A, 'lqgSimulation A')
  const n = a.m
  const b = dense.toMatrixF64(plant.B, 'lqgSimulation B', n)
  const c = dense.toMatrixF64(plant.C, 'lqgSimulation C', undefined, n)
  const m = b.n
  const p = c.m
  const Q = dense.toMatrixF64(weights.Q, 'lqgSimulation Q', n, n).data
  const R = dense.toMatrixF64(weights.R, 'lqgSimulation R', m, m).data
  const cholOf = (M: MatrixLike, k: number) => {
    const ch = cholesky(fromData(dense.toMatrixF64(M, 'lqgSimulation noise', k, k).data, [k, k]))
    return dense.data(ch.L)
  }
  const Lw = cholOf(weights.W, n)
  const Lv = cholOf(weights.V, p)
  const K = dense.data(design.K)
  const L = dense.data(design.L)
  const draw = (s: Stream, k: number, chol: F64) => {
    const z = dense.toF64(normal(s, 0, 1, { shape: [k] }) as Tensor, 'lqgSimulation noise')
    return dense.matVec(chol, z, k, k)
  }
  const vec = (x: F64) => fromData(x, [x.length])
  const measure = (x: F64, s: Stream) => dense.add(dense.matVec(c.data, x, p, n), draw(child(s, 'v'), p, Lv))
  const control = (x: F64, xHat: F64) => dense.scale(-1, dense.matVec(K, feedback === 'state' ? x : xHat, m, n))
  return {
    name: 'lqg-simulation',
    init: ({ x0, xHat0 }, stream) => {
      const x = dense.toF64(x0, 'lqgSimulation x0')
      const xHat = xHat0 === undefined ? new Float64Array(n) : dense.toF64(xHat0, 'lqgSimulation xHat0')
      const y = measure(x, stream)
      const u = control(x, xHat)
      return { t: 0, x: vec(x), xHat: vec(xHat), y: vec(y), u: vec(u), cost: 0, diverged: false }
    },
    step: (s, ctx) => {
      const x = dense.data(s.x)
      const xHat = dense.data(s.xHat)
      const u = dense.data(s.u)
      const y = dense.data(s.y)
      const stage = dense.dot(x, dense.matVec(Q, x, n, n)) + dense.dot(u, dense.matVec(R, u, m, m))
      const xNext = dense.add(
        dense.add(dense.matVec(a.data, x, n, n), dense.matVec(b.data, u, n, m)),
        draw(child(ctx.stream, 'w'), n, Lw),
      )
      const innovation = dense.sub(y, dense.matVec(c.data, xHat, p, n))
      const xHatNext = dense.add(
        dense.add(dense.matVec(a.data, xHat, n, n), dense.matVec(b.data, u, n, m)),
        dense.matVec(L, innovation, n, p),
      )
      const yNext = measure(xNext, ctx.stream)
      const uNext = control(xNext, xHatNext)
      return {
        t: s.t + 1,
        x: vec(xNext),
        xHat: vec(xHatNext),
        y: vec(yNext),
        u: vec(uNext),
        cost: s.cost + stage,
        diverged: !dense.allFinite(xNext),
      }
    },
  }
}
