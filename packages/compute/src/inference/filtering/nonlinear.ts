/**
 * Nonlinear state-space filters for z_t = f(z_{t−1}) + w_t, y_t = h(z_t) + v_t with Gaussian noise: the extended
 * Kalman filter (linearise f and h by their Jacobians, from `aifn-compute/foundation/autodiff`) and the unscented Kalman filter
 * (propagate sigma points through f and h). Both return the linear filter's result shape and share its tensor algebra.
 */

import { jacobian } from 'aifn-compute/foundation/autodiff'
import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  add,
  full,
  isTensor,
  matmul,
  mul,
  outer,
  reshape,
  sub,
  tensor,
  toFlat,
  transpose,
  zeros,
  type Matrix,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { packFilter, type FilterRun, type KalmanFilterResult, type KalmanStep } from './kalman'
import {
  asMatrix,
  asSeries,
  asVector,
  sandwich,
  quadratic,
  solveOrNull,
  sqrtPsd,
  symmetrise,
  type MatrixLike,
  type VectorLike,
} from './gaussian'

/**
 * A nonlinear state-space model. `f` and `h` take the state as a length-n vector and return a vector (n and m long).
 * For the extended filter they must be written with `aifn-compute/foundation/tensor` primitives so that `aifn-compute/foundation/autodiff` can
 * differentiate them (e.g. `stack([add(get(z, 0), get(z, 1)), sin(get(z, 0))])`).
 */
export type NonlinearStateSpaceModel = {
  f: (z: Vector) => Value
  h: (z: Vector) => Value
  Q: MatrixLike | number
  R: MatrixLike | number
  m0: VectorLike | number
  P0: MatrixLike | number
}

/** f(z) or h(z) as a float64 vector. */
const asOutput = (v: Value, where: string): Vector => {
  if (typeof v === 'number') return tensor([v]) as Vector
  if (!isTensor(v)) throw new ShapeError(where, `${where}: f and h must return numbers or tensors`)
  return reshape(v, [-1]) as Vector
}

function parse(model: NonlinearStateSpaceModel, where: string) {
  return {
    Q: asMatrix(model.Q, where),
    R: asMatrix(model.R, where),
    m0: asVector(model.m0, where),
    P0: asMatrix(model.P0, where),
  }
}

/**
 * The update shared by both filters, given the predicted moments, the predicted observation ŷ, its covariance S and
 * the state–observation cross-covariance Σ_zy: K = Σ_zy S⁻¹ (solved as Kᵀ = S⁻¹ Σ_zyᵀ), μ = μ⁻ + K(y − ŷ),
 * P = P⁻ − K S Kᵀ. Appends the step to `out`.
 */
function update(
  out: FilterRun,
  t: number,
  y: readonly number[],
  predictedMean: Vector,
  predictedCov: Matrix,
  yHat: Vector,
  innovationCov: Matrix,
  cross: Tensor,
): KalmanStep {
  const [n] = predictedMean.shape
  const [m] = innovationCov.shape
  let step: KalmanStep = {
    predictedMean,
    predictedCov,
    mean: predictedMean,
    cov: predictedCov,
    gain: zeros([n, m]) as Matrix,
    innovation: full([m], NaN) as Vector,
    innovationCov,
    term: 0,
    singular: false,
  }
  if (!y.some((v) => Number.isNaN(v))) {
    const innovation = sub(tensor([...y]), yHat) as Vector
    const sol = solveOrNull(innovationCov, transpose(cross) as Tensor)
    if (sol === null) step = { ...step, innovation, term: NaN, singular: true }
    else {
      const gain = transpose(sol.x) as Matrix
      const quad = quadratic(innovation, solveOrNull(innovationCov, innovation)!.x)
      step = {
        ...step,
        innovation,
        gain,
        mean: add(predictedMean, matmul(gain, innovation)) as Vector,
        cov: symmetrise(sub(predictedCov, sandwich(gain, innovationCov)) as Tensor) as Matrix,
        term: -0.5 * (m * Math.log(2 * Math.PI) + sol.logAbsDet + quad),
      }
    }
  }
  out.steps.push(step)
  out.logLikelihood += step.term
  if (step.singular) out.singularSteps.push(t)
  return step
}

const empty = (): FilterRun => ({ steps: [], logLikelihood: 0, singularSteps: [] })

/**
 * The extended Kalman filter (Jazwinski, 1970; Särkkä, 2013, Algorithm 5.4): predict μ⁻ = f(μ), P⁻ = F P Fᵀ + Q with
 * F = ∂f/∂z at μ; update with H = ∂h/∂z at μ⁻, S = H P⁻ Hᵀ + R, K = P⁻ Hᵀ S⁻¹. Jacobians come from
 * `aifn-compute/foundation/autodiff`'s `jacobian`. The log-likelihood is that of the linearised model.
 */
export function extendedKalmanFilter(model: NonlinearStateSpaceModel, y: VectorLike | MatrixLike): KalmanFilterResult {
  const where = 'extendedKalmanFilter'
  const { Q, R, m0, P0 } = parse(model, where)
  const ys = asSeries(y, where)
  const jf = jacobian(model.f as (z: Value) => Value)
  const jh = jacobian(model.h as (z: Value) => Value)
  const [n] = m0.shape
  // The Jacobian has shape [...shape of f(z), n]; read it as rows × n whatever the output's rank.
  const jac = (J: Value, rows: number): Matrix => reshape(typeof J === 'number' ? tensor([J]) : J, [rows, n]) as Matrix
  const out = empty()
  let m: Vector = m0
  let P: Matrix = P0
  ys.forEach((yt, t) => {
    const mPred = asOutput(model.f(m), where)
    const F = jac(jf(m), n)
    const PPred = symmetrise(add(sandwich(F, P), Q) as Tensor) as Matrix
    const yHat = asOutput(model.h(mPred), where)
    const H = jac(jh(mPred), yHat.shape[0])
    const S = symmetrise(add(sandwich(H, PPred), R) as Tensor) as Matrix
    ;({ mean: m, cov: P } = update(out, t, yt, mPred, PPred, yHat, S, matmul(PPred, transpose(H)) as Tensor))
  })
  return packFilter(out, n, R.shape[0])
}

/** Options of the unscented transform (Wan & van der Merwe, 2000): spread α, prior-knowledge β and secondary κ. */
export type UnscentedOptions = { alpha?: number; beta?: number; kappa?: number }

/**
 * The unscented Kalman filter (Julier & Uhlmann, 1997; Wan & van der Merwe, 2000): 2n + 1 sigma points
 * μ, μ ± √(n + λ) S_i with S Sᵀ = P and λ = α²(n + κ) − n, pushed through f (predict) and h (update), with mean
 * weights W₀ = λ/(n + λ), covariance weight W₀ + 1 − α² + β and 1/(2(n + λ)) for the rest. Defaults α = 1, β = 2,
 * κ = 0 (positive weights). The square root is a symmetric eigen-root, which exists for singular P.
 */
export function unscentedKalmanFilter(
  model: NonlinearStateSpaceModel,
  y: VectorLike | MatrixLike,
  { alpha = 1, beta = 2, kappa = 0 }: UnscentedOptions = {},
): KalmanFilterResult {
  const where = 'unscentedKalmanFilter'
  const { Q, R, m0, P0 } = parse(model, where)
  const ys = asSeries(y, where)
  const [n] = m0.shape
  const lambda = alpha * alpha * (n + kappa) - n
  const wm = [lambda / (n + lambda), ...new Array<number>(2 * n).fill(1 / (2 * (n + lambda)))]
  const wc = [wm[0] + 1 - alpha * alpha + beta, ...wm.slice(1)]
  // The sigma points as rows of a [2n + 1, n] matrix: μ, μ + √(n + λ) Sᵢ, μ − √(n + λ) Sᵢ (Sᵢ the columns of S).
  const sigma = (m: Vector, P: Matrix): Vector[] => {
    const cols = toRowsOf(transpose(mul(Math.sqrt(n + lambda), sqrtPsd(P).S)) as Tensor)
    return [m, ...cols.map((c) => add(m, c) as Vector), ...cols.map((c) => sub(m, c) as Vector)]
  }
  const weighted = (pts: Vector[]): Vector =>
    pts.reduce((acc, p, k) => add(acc, mul(wm[k], p)) as Vector, mul(0, pts[0]) as Vector)
  const cov = (a: Vector[], ma: Vector, b: Vector[], mb: Vector): Matrix =>
    a.reduce(
      (acc, p, k) => add(acc, mul(wc[k], outer(sub(p, ma), sub(b[k], mb)))) as Matrix,
      zeros([ma.shape[0], mb.shape[0]]) as Matrix,
    )
  const through = (g: (z: Vector) => Value, pts: Vector[]) => pts.map((p) => asOutput(g(p), where))
  const out = empty()
  let m: Vector = m0
  let P: Matrix = P0
  ys.forEach((yt, t) => {
    const fx = through(model.f, sigma(m, P))
    const mPred = weighted(fx)
    const PPred = symmetrise(add(cov(fx, mPred, fx, mPred), Q) as Tensor) as Matrix
    const pts = sigma(mPred, PPred)
    const hx = through(model.h, pts)
    const yHat = weighted(hx)
    const S = symmetrise(add(cov(hx, yHat, hx, yHat), R) as Tensor) as Matrix
    ;({ mean: m, cov: P } = update(out, t, yt, mPred, PPred, yHat, S, cov(pts, mPred, hx, yHat)))
  })
  return packFilter(out, n, R.shape[0])
}

/** The rows of a matrix as vectors. */
function toRowsOf(a: Tensor): Vector[] {
  const [r, c] = a.shape
  const flat = toFlat(a)
  return Array.from({ length: r }, (_, i) => tensor(flat.slice(i * c, (i + 1) * c)) as Vector)
}
