/**
 * Nonlinear state-space filters for $\zvec_t = f(\zvec_{t-1}) + \wvec_t$, $\yvec_t = h(\zvec_t) + \vvec_t$ with
 * Gaussian noise $\wvec_t \sim \Gauss(\zeros, \Qmat)$, $\vvec_t \sim \Gauss(\zeros, \Rmat)$ and
 * $\zvec_0 \sim \Gauss(\mvec_0, \Pmat_0)$: the extended Kalman filter (linearise $f$ and $h$ by their Jacobians, from
 * `aifn-compute/foundation/autodiff`) and the unscented Kalman filter (propagate sigma points through $f$ and $h$).
 *
 * Both return the linear filter's `KalmanFilterResult` and share its tensor algebra and its conventions: $\zvec_0$ is
 * not observed, so each step predicts before it updates, and a singular innovation covariance skips the update and is
 * reported in `singularSteps`. Unlike the linear filter, a step whose observation has any NaN entry is a prediction
 * only, and the shapes of $\Qmat$, $\Rmat$ and $\Pmat_0$ are not checked against each other.
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
 * A nonlinear state-space model. `f` and `h` take the state as a length-$n$ vector and return a vector ($n$ and $m$
 * long; a number counts as length 1). For the extended filter they must be written with
 * `aifn-compute/foundation/tensor` primitives so that `aifn-compute/foundation/autodiff` can differentiate them (e.g.
 * `stack([add(get(z, 0), get(z, 1)), sin(get(z, 0))])`).
 */
export type NonlinearStateSpaceModel = {
  /** The transition $f$: the mean of $\zvec_t$ given $\zvec_{t-1}$. */
  f: (z: Vector) => Value
  /** The observation function $h$: the mean of $\yvec_t$ given $\zvec_t$. */
  h: (z: Vector) => Value
  /** Process-noise covariance $\Qmat$ ($n \times n$). */
  Q: MatrixLike | number
  /** Observation-noise covariance $\Rmat$ ($m \times m$). */
  R: MatrixLike | number
  /** Mean $\mvec_0$ of $\zvec_0$ (length $n$; it sets $n$). */
  m0: VectorLike | number
  /** Covariance $\Pmat_0$ of $\zvec_0$ ($n \times n$). */
  P0: MatrixLike | number
}

/**
 * $f(\zvec)$ or $h(\zvec)$ as a float64 vector: a number becomes length 1 and a tensor of any rank is flattened.
 * Anything else throws `ShapeError`.
 *
 * @param v What `f` or `h` returned.
 * @param where The caller's name, for error messages.
 * @returns The output as a vector.
 */
const asOutput = (v: Value, where: string): Vector => {
  if (typeof v === 'number') return tensor([v]) as Vector
  if (!isTensor(v)) throw new ShapeError(where, `${where}: f and h must return numbers or tensors`)
  return reshape(v, [-1]) as Vector
}

/**
 * The model's covariances and initial mean as tensors (shapes not checked against each other).
 *
 * @param model The nonlinear model.
 * @param where The caller's name, for error messages.
 * @returns $\Qmat$, $\Rmat$, $\mvec_0$ and $\Pmat_0$ as tensors.
 */
function parse(model: NonlinearStateSpaceModel, where: string) {
  return {
    Q: asMatrix(model.Q, where),
    R: asMatrix(model.R, where),
    m0: asVector(model.m0, where),
    P0: asMatrix(model.P0, where),
  }
}

/**
 * The update shared by both filters, given the predicted moments, the predicted observation $\hat\yvec$, its covariance
 * $\Smat$ and the state–observation cross-covariance $\Sigmamat_{zy}$: $\Kmat = \Sigmamat_{zy}\Smat^{-1}$ (solved as
 * $\Kmat^\top = \Smat^{-1}\Sigmamat_{zy}^\top$), $\muvec = \muvec^- + \Kmat(\yvec - \hat\yvec)$,
 * $\Pmat = \Pmat^- - \Kmat\Smat\Kmat^\top$. Appends the step to `out`. An observation with any NaN entry skips
 * the update (the step is the prediction, with term 0); a singular $\Smat$ skips it too, with term NaN.
 *
 * @param out The run so far; the step is appended to its `steps`, its term added to `logLikelihood`, and `t` to
 *   `singularSteps` when $\Smat$ is singular. Modified in place.
 * @param t The index of the step (0-based), recorded when $\Smat$ is singular.
 * @param y The observation, $m$ numbers.
 * @param predictedMean The predicted mean $\muvec^-$ (length $n$).
 * @param predictedCov The predicted covariance $\Pmat^-$ ($n \times n$).
 * @param yHat The predicted observation $\hat\yvec$ (length $m$).
 * @param innovationCov The innovation covariance $\Smat$ ($m \times m$).
 * @param cross The cross-covariance $\Sigmamat_{zy}$ ($n \times m$).
 * @returns The step, as appended.
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

/** An empty run, for the filters to append their steps to. */
const empty = (): FilterRun => ({ steps: [], logLikelihood: 0, singularSteps: [] })

/**
 * The extended Kalman filter (Jazwinski, 1970; Särkkä, 2013, Algorithm 5.4): predict $\muvec^- = f(\muvec)$,
 * $\Pmat^- = \Fmat\Pmat\Fmat^\top + \Qmat$ with $\Fmat = \partial f / \partial \zvec$ at $\muvec$; update with
 * $\Hmat = \partial h / \partial \zvec$ at $\muvec^-$, $\Smat = \Hmat\Pmat^-\Hmat^\top + \Rmat$,
 * $\Kmat = \Pmat^-\Hmat^\top\Smat^{-1}$. Jacobians come from `aifn-compute/foundation/autodiff`'s `jacobian`. The
 * log-likelihood is that of the linearised model. For linear $f$ and $h$ it is the Kalman filter.
 *
 * @param model The model; `f` and `h` must be differentiable (written with tensor primitives).
 * @param y The observations, `[T, m]` (or a length-$T$ vector when $m = 1$); a row with any NaN is skipped.
 * @returns The predicted and filtered moments, gains, innovations and log-likelihood of every step, as
 *   `kalmanFilter` returns them.
 *
 * @example On a linear model it is the Kalman filter
 * const walk = { f: (z) => z, h: (z) => z, Q: 1, R: 1, m0: 0, P0: 0 }
 * print('extended =', extendedKalmanFilter(walk, [1, 1, 1]).mean)
 * print('kalman   =', kalmanFilter({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, [1, 1, 1]).mean)
 *
 * @example A state seen through its square
 * // z grows by about 0.1 a step from 1, and is seen as y = z^2 + noise.
 * const model = { f: (z) => add(z, 0.1), h: (z) => mul(z, z), Q: 0.001, R: 0.01, m0: 1, P0: 0.01 }
 * const y = [1.25, 1.4, 1.72]
 * print('filtered z =', extendedKalmanFilter(model, y).mean)
 * print('sqrt(y) =', y.map(Math.sqrt))
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

/**
 * Options of the unscented transform (Wan & van der Merwe, 2000): `alpha`, the spread $\alpha$ of the sigma points
 * (default 1); `beta`, the prior-knowledge parameter $\beta$ (default 2, optimal for a Gaussian); and `kappa`, the
 * secondary scaling $\kappa$ (default 0).
 */
export type UnscentedOptions = { alpha?: number; beta?: number; kappa?: number }

/**
 * The unscented Kalman filter (Julier & Uhlmann, 1997; Wan & van der Merwe, 2000): $2n + 1$ sigma points
 * $\muvec$, $\muvec \pm \sqrt{n + \lambda}\,\svec_i$ with $\svec_i$ the columns of $\Smat$, $\Smat\Smat^\top = \Pmat$
 * and $\lambda = \alpha^2(n + \kappa) - n$, pushed through $f$ (predict) and, redrawn from the predicted moments,
 * through $h$ (update), with mean weights $W_0 = \lambda/(n + \lambda)$, covariance weight
 * $W_0 + 1 - \alpha^2 + \beta$ and $1/(2(n + \lambda))$ for the rest. Defaults $\alpha = 1$, $\beta = 2$,
 * $\kappa = 0$ (non-negative weights). The square root is a symmetric eigen-root, which exists for singular $\Pmat$.
 * `f` and `h` need not be differentiable.
 *
 * @param model The model.
 * @param y The observations, `[T, m]` (or a length-$T$ vector when $m = 1$); a row with any NaN is skipped.
 * @param options The unscented transform's parameters.
 * @param options.alpha The spread $\alpha$ of the sigma points around the mean (default 1).
 * @param options.beta The prior-knowledge parameter $\beta$, added to the centre point's covariance weight (default
 *   2, optimal for a Gaussian).
 * @param options.kappa The secondary scaling $\kappa$ (default 0).
 * @returns The predicted and filtered moments, gains, innovations and log-likelihood of every step, as
 *   `kalmanFilter` returns them.
 *
 * @example On a linear model it is the Kalman filter
 * const walk = { f: (z) => z, h: (z) => z, Q: 1, R: 1, m0: 0, P0: 0 }
 * print('unscented =', unscentedKalmanFilter(walk, [1, 1, 1]).mean)
 * print('kalman    =', kalmanFilter({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, [1, 1, 1]).mean)
 *
 * @example A state seen through its square
 * // z grows by about 0.1 a step from 1, and is seen as y = z^2 + noise.
 * const model = { f: (z) => add(z, 0.1), h: (z) => mul(z, z), Q: 0.001, R: 0.01, m0: 1, P0: 0.01 }
 * const y = [1.25, 1.4, 1.72]
 * print('filtered z =', unscentedKalmanFilter(model, y).mean)
 * print('sqrt(y) =', y.map(Math.sqrt))
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

/**
 * The rows of a matrix as vectors.
 *
 * @param a A matrix, $r \times c$.
 * @returns $r$ new vectors of length $c$.
 */
function toRowsOf(a: Tensor): Vector[] {
  const [r, c] = a.shape
  const flat = toFlat(a)
  return Array.from({ length: r }, (_, i) => tensor(flat.slice(i * c, (i + 1) * c)) as Vector)
}
