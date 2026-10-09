/**
 * Linear-quadratic-Gaussian (LQG) control: the LQR gain applied to the Kalman filter's state estimate. By the
 * separation principle the two are designed independently, and the closed-loop poles are those of
 * $\Amat - \Bmat\Kmat$ and of $\Amat - \Lmat\Cmat$ together (Athans, 1971, "The role and use of the stochastic
 * linear-quadratic-Gaussian problem in control system design", IEEE TAC 16(6); Anderson & Moore, 1990, "Optimal
 * Control: Linear Quadratic Methods", ch. 8).
 *
 * The steady-state Kalman gain is the LQR gain of the dual problem $(\Amat^\top, \Cmat^\top)$ with weights
 * $(\Wmat, \Vmat)$ (Kalman, 1960), so it is computed by the same Riccati solvers (`lqr`, `dlqr`) rather than a second
 * implementation.
 */

import { dense, fromData, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { child, normal, type Stream } from 'aifn-compute/foundation/random'
import { cholesky, eig } from 'aifn-compute/numerics/linalg'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dlqr, lqr, type LqrResult } from './lqr'

type F64 = dense.F64

/**
 * A plant with process and measurement noise: $\xvec' = \Amat\xvec + \Bmat\uvec + \wvec$,
 * $\yvec = \Cmat\xvec + \vvec$, with $\wvec \sim \Gauss(\zeros, \Wmat)$ and $\vvec \sim \Gauss(\zeros, \Vmat)$ (or the
 * same in discrete time). `A` is $n \times n$, `B` $n \times m$ and `C` $p \times n$.
 */
export type LqgPlant = { A: MatrixLike; B: MatrixLike; C: MatrixLike }

/** The weights and noise covariances of an LQG problem. */
export type LqgWeights = {
  /** State weight $\Qmat$ of the cost ($n \times n$, symmetric positive semi-definite). */
  Q: MatrixLike
  /** Input weight $\Rmat$ of the cost ($m \times m$, symmetric positive definite). */
  R: MatrixLike
  /** Process noise covariance $\Wmat$ ($n \times n$). */
  W: MatrixLike
  /** Measurement noise covariance $\Vmat$ ($p \times p$, positive definite). */
  V: MatrixLike
}

/** An LQG design. */
export type LqgDesign = {
  /** The LQR gain $\Kmat$ ($m \times n$): $\uvec = -\Kmat\hat{\xvec}$. */
  K: Matrix
  /**
   * The steady-state Kalman gain $\Lmat$ ($n \times p$). Continuous:
   * $\hat{\xvec}' = \Amat\hat{\xvec} + \Bmat\uvec + \Lmat(\yvec - \Cmat\hat{\xvec})$. Discrete (predictor form):
   * $\hat{\xvec}_{k+1} = \Amat\hat{\xvec}_k + \Bmat\uvec_k + \Lmat(\yvec_k - \Cmat\hat{\xvec}_k)$.
   */
  L: Matrix
  /** The control Riccati solution $\Pmat$ ($n \times n$). */
  P: Matrix
  /** The steady-state estimation error covariance $\Smat$ ($n \times n$; the prior covariance in discrete time). */
  S: Matrix
  /** The poles of the regulator $\Amat - \Bmat\Kmat$ (complex128). */
  regulatorPoles: Tensor
  /** The poles of the estimator $\Amat - \Lmat\Cmat$ (complex128). */
  estimatorPoles: Tensor
  /**
   * The output-feedback compensator from $\yvec$ to $\uvec$ as a state-space system: `A` is
   * $\Amat - \Bmat\Kmat - \Lmat\Cmat$, `B` is $\Lmat$, `C` is $-\Kmat$ and `D` is $\zeros$.
   */
  controller: { A: Matrix; B: Matrix; C: Matrix; D: Matrix }
  /** Always true: a Riccati equation that does not converge throws instead. */
  converged: boolean
}

/**
 * The transpose of a row-major matrix, as a tensor.
 *
 * @param a The matrix's entries, row-major, $m \times n$.
 * @param m Its number of rows.
 * @param n Its number of columns.
 * @returns The $n \times m$ transpose.
 */
const T = (a: F64, m: number, n: number) => fromData(dense.transpose(a, m, n), [n, m])

/**
 * The LQG design for a continuous plant, or a discrete one with `discrete: true` (Anderson & Moore, 1990, ch. 8):
 * $\Kmat$ from the LQR of $(\Amat, \Bmat, \Qmat, \Rmat)$ and $\Lmat$ from the dual LQR of
 * $(\Amat^\top, \Cmat^\top, \Wmat, \Vmat)$, $\Lmat = \Kmat_{\text{dual}}^\top$. Continuous:
 * $\Lmat = \Smat\Cmat^\top\Vmat^{-1}$ with $\Smat$ the filter CARE's solution. Discrete:
 * $\Lmat = \Amat\Smat\Cmat^\top(\Cmat\Smat\Cmat^\top + \Vmat)^{-1}$, the predictor gain, with $\Smat$ the prior
 * covariance. Throws `DomainError` when either Riccati equation does not converge (the plant is then usually not
 * stabilisable or not detectable).
 *
 * @param plant The plant: `A` ($n \times n$), `B` ($n \times m$) and `C` ($p \times n$).
 * @param weights The cost weights `Q` and `R`, and the noise covariances `W` and `V`.
 * @param options Whether the plant is continuous or discrete.
 * @param options.discrete True for a discrete-time plant ($\xvec_{k+1} = \Amat\xvec_k + \dots$), solved with `dlqr`;
 *   false (default) for a continuous one, solved with `lqr`.
 * @returns The two gains, the two Riccati solutions, the regulator and estimator poles, and the compensator.
 *
 * @example A scalar discrete plant
 * // x_{k+1} = x_k + u_k + w_k, y_k = x_k + v_k with every weight 1: the regulator and the estimator are the same
 * // Riccati problem, so K = L = 0.618 and both poles are 0.382 (dlqr's example).
 * const design = lqg({ A: [[1]], B: [[1]], C: [[1]] }, { Q: [[1]], R: [[1]], W: [[1]], V: [[1]] }, { discrete: true })
 * print('K =', design.K, ' L =', design.L)
 * print('regulator pole =', design.regulatorPoles, ' estimator pole =', design.estimatorPoles)
 * print('compensator A =', design.controller.A)
 *
 * @example A continuous double integrator measured in position
 * const plant = { A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]] }
 * const weights = { Q: [[1, 0], [0, 1]], R: [[1]], W: [[1, 0], [0, 1]], V: [[1]] }
 * const design = lqg(plant, weights)
 * print('K =', design.K)
 * print('L =', design.L)
 * print('estimator poles =', design.estimatorPoles)
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
  /** The sample number $k$. */
  t: Size
  /** The true state $\xvec_k$. */
  x: Vector
  /** The estimate $\hat{\xvec}_k$. */
  xHat: Vector
  /** The measurement $\yvec_k = \Cmat\xvec_k + \vvec_k$. */
  y: Vector
  /** The input $\uvec_k$ to be applied: $-\Kmat\hat{\xvec}_k$ (or $-\Kmat\xvec_k$ with `feedback: 'state'`). */
  u: Vector
  /** Running cost $\sum_{j<k} (\xvec_j^\top\Qmat\xvec_j + \uvec_j^\top\Rmat\uvec_j)$ over the samples applied. */
  cost: Scalar
  /** True when the true state is not finite. */
  diverged: boolean
}

/**
 * A discrete LQG loop as a traceable algorithm: the plant
 * $\xvec_{k+1} = \Amat\xvec_k + \Bmat\uvec_k + \wvec_k$, $\yvec_k = \Cmat\xvec_k + \vvec_k$ with Gaussian noise drawn
 * from the step's stream, the predictor-form estimator and $\uvec_k = -\Kmat\hat{\xvec}_k$. With `feedback: 'state'`
 * the regulator uses the true state instead (the full-information LQR), for comparison. `init` takes
 * `{ x0, xHat0 }`; the noise is drawn as $\Lmat_{\Wmat}\zvec$ with $\Lmat_{\Wmat}$ the Cholesky factor of $\Wmat$
 * (likewise for $\Vmat$).
 *
 * @param plant The discrete plant: `A` ($n \times n$), `B` ($n \times m$) and `C` ($p \times n$).
 * @param weights `Q` and `R`, which weigh the running cost, and `W` and `V`, the covariances the noise is drawn with.
 * @param design The gains to apply, from `lqg` with `discrete: true` (the estimator is the predictor form, so a
 *   continuous design's `L` is not the right gain; this is not checked).
 * @param options Which state the regulator feeds back.
 * @param options.feedback `'estimate'` (default) for LQG output feedback on $\hat{\xvec}$, `'state'` for the
 *   full-information LQR on the true $\xvec$.
 * @returns The algorithm: `init` takes the true initial state `x0` and the initial estimate `xHat0` (default
 *   $\zeros$) and takes the first measurement; each `step` moves the plant, the estimator and the input on by one
 *   sample.
 *
 * @example Output feedback against full information
 * const plant = { A: [[1]], B: [[1]], C: [[1]] }
 * const weights = { Q: [[1]], R: [[1]], W: [[0.01]], V: [[0.01]] }
 * const design = lqg(plant, weights, { discrete: true })
 * const lqgLoop = lqgSimulation(plant, weights, design)
 * const lqrLoop = lqgSimulation(plant, weights, design, { feedback: 'state' })
 * const a = run(lqgLoop, { x0: [5] }, 20, { stream: stream(1) })
 * const b = run(lqrLoop, { x0: [5] }, 20, { stream: stream(1) })
 * print('LQG:  x =', a.x, ' estimate =', a.xHat, ' cost =', a.cost)
 * print('LQR:  x =', b.x, ' cost =', b.cost)
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
