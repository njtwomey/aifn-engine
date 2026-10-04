/**
 * Linear-Gaussian state-space models z_t = A z_{t−1} + w_t, y_t = C z_t + v_t, w_t ~ N(0, Q), v_t ~ N(0, R), with
 * z₀ ~ N(m₀, P₀): simulation, the Kalman filter, the Rauch–Tung–Striebel smoother, the steady-state filter, and the
 * filter and smoother as step-through `Algorithm`s.
 *
 * Convention: (m₀, P₀) describes z₀, which is not observed; the first observation y₁ is of z₁ = A z₀ + w₁, so the
 * filter predicts before its first update. NaN entries of y are missing: a step updates with its observed entries only,
 * and predicts through a row that is entirely NaN.
 *
 * Everything is written on tensors (`aifn-compute/foundation/tensor` arithmetic, `aifn-compute/numerics/linalg` factorisations): one
 * filter step (`kalmanStep`) and one smoother step (`rtsStep`) are the definitions, and the batch functions, the
 * algorithms and EM (`aifn-methods/timeseries`) all call them.
 */

import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { solveDense } from 'aifn-compute/numerics/linalg'
import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  eye,
  fromData,
  matmul,
  sub,
  tensor,
  toFlat,
  transpose,
  zeros,
  dense,
  type Matrix,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import {
  asMatrix,
  asSeries,
  asVector,
  sandwich,
  quadratic,
  solveOrNull,
  sqrtPsd,
  stackSteps,
  symmetrise,
  type MatrixLike,
  type VectorLike,
} from './gaussian'

/** A linear-Gaussian state-space model. Scalars stand for 1×1 matrices (and a length-1 m₀). */
export type StateSpaceModel = {
  /** Transition A (n×n). */
  A: MatrixLike | number
  /** Observation C (m×n). */
  C: MatrixLike | number
  /** Process-noise covariance Q (n×n, positive semi-definite; zero rows are allowed). */
  Q: MatrixLike | number
  /** Observation-noise covariance R (m×m, positive semi-definite; zero rows are allowed). */
  R: MatrixLike | number
  /** Mean of z₀ (length n). */
  m0: VectorLike | number
  /** Covariance of z₀ (n×n). */
  P0: MatrixLike | number
}

/** The model with every part a tensor: the form the filter, the smoother and EM share. */
export type Model = { A: Matrix; C: Matrix; Q: Matrix; R: Matrix; m0: Vector; P0: Matrix }

/** Read a `StateSpaceModel` into tensors, checking that the shapes agree (throws `ShapeError` naming `where`). */
export function parseModel(model: StateSpaceModel, where: string): Model {
  const A = asMatrix(model.A, where)
  const C = asMatrix(model.C, where)
  const Q = asMatrix(model.Q, where)
  const R = asMatrix(model.R, where)
  const m0 = asVector(model.m0, where)
  const P0 = asMatrix(model.P0, where)
  const [n, nA] = A.shape
  const [m, nC] = C.shape
  const square = (M: Tensor, k: number) => M.shape[0] === k && M.shape[1] === k
  const ok = nA === n && nC === n && square(Q, n) && square(R, m) && m0.shape[0] === n && square(P0, n)
  if (!ok) throw new ShapeError(where, `${where}: inconsistent model shapes (A ${n}×${nA}, C ${m}×${nC})`)
  return { A, C, Q, R, m0, P0 }
}

/**
 * Draw a trajectory of length T and its observations. Noise is drawn as S ε with S Sᵀ = Q (or R) from a symmetric
 * eigendecomposition, so covariances with deterministic components (zero rows) give exact zeros rather than NaN.
 * Returns z₁ … z_T ([T, n]) and y₁ … y_T ([T, m]) and the drawn z₀.
 */
export function simulateStateSpace(
  s: Stream,
  model: StateSpaceModel,
  T: number,
): { states: Matrix; observations: Matrix; initial: Tensor } {
  const { A, C, Q, R, m0, P0 } = parseModel(model, 'simulateStateSpace')
  const [n] = A.shape
  const [m] = C.shape
  const draw = (S: Matrix, key: string, t: number) => matmul(S, normals(child(s, key, t), S.shape[1])) as Tensor
  const Sq = sqrtPsd(Q).S
  const Sr = sqrtPsd(R).S
  let z = add(m0, draw(sqrtPsd(P0).S, 'initial', 0)) as Tensor
  const initial = z
  const states: Tensor[] = []
  const obs: Tensor[] = []
  for (let t = 0; t < T; t++) {
    z = add(matmul(A, z), draw(Sq, 'state', t)) as Tensor
    states.push(z)
    obs.push(add(matmul(C, z), draw(Sr, 'observation', t)) as Tensor)
  }
  return { states: stackSteps(states, [n]) as Matrix, observations: stackSteps(obs, [m]) as Matrix, initial }
}

// ── One filter step ──────────────────────────────────────────────────────────────────────────────────────────────────

/** One step of the Kalman filter: the prediction, the update and the step's log-likelihood term. */
export type KalmanStep = {
  /** μ_{t|t−1}. */
  predictedMean: Vector
  /** P_{t|t−1}. */
  predictedCov: Matrix
  /** μ_{t|t}. */
  mean: Vector
  /** P_{t|t}. */
  cov: Matrix
  /** K_t (n×m); zero for a missing or singular step, and in the columns of missing entries. */
  gain: Matrix
  /** y_t − C μ_{t|t−1}; NaN in missing entries. */
  innovation: Vector
  /** S_t = C P_{t|t−1} Cᵀ + R. */
  innovationCov: Matrix
  /** log N(y_t; C μ_{t|t−1}, S_t); 0 when missing, NaN when S_t is singular. */
  term: number
  /** True when S_t was singular (the update was skipped). */
  singular: boolean
}

/**
 * One step of the Kalman filter (Kalman, 1960) from (μ_{t−1|t−1}, P_{t−1|t−1}) and the observation y_t (NaN entries
 * are missing: the update then uses the observed rows of C and y and the observed block of R, and a step with every
 * entry missing is a prediction only). Predict μ⁻ = A μ, P⁻ = A P Aᵀ + Q; update with S = C P⁻ Cᵀ + R,
 * K = P⁻ Cᵀ S⁻¹ (solved as Kᵀ = S⁻¹ C P⁻), μ = μ⁻ + K(y − C μ⁻) and the covariance in Joseph form
 * (I − KC) P⁻ (I − KC)ᵀ + K R Kᵀ, which keeps P symmetric positive semi-definite in finite precision.
 */
export function kalmanStep(md: Model, mean: Vector, cov: Matrix, y: readonly number[]): KalmanStep {
  const d = denseModel(md)
  const { n, m } = d
  const r = stepDense(d, dense.data(mean), dense.data(cov), y)
  return {
    predictedMean: fromData(r.predictedMean, [n]) as Vector,
    predictedCov: fromData(r.predictedCov, [n, n]) as Matrix,
    mean: fromData(r.mean, [n]) as Vector,
    cov: fromData(r.cov, [n, n]) as Matrix,
    gain: fromData(r.gain, [n, m]) as Matrix,
    innovation: fromData(r.innovation, [m]) as Vector,
    innovationCov: fromData(r.innovationCov, [m, m]) as Matrix,
    term: r.term,
    singular: r.singular,
  }
}

/** The model's matrices as row-major arrays (zero-copy views where contiguous), cached per model. */
type F64 = dense.F64
type DenseModel = { n: number; m: number; A: F64; C: F64; Q: F64; R: F64; I: F64 }
const denseModels = new WeakMap<Model, DenseModel>()
function denseModel(md: Model): DenseModel {
  let d = denseModels.get(md)
  if (d === undefined) {
    const [n] = md.A.shape
    const [m] = md.C.shape
    d = {
      n,
      m,
      A: dense.data(md.A),
      C: dense.data(md.C),
      Q: dense.data(md.Q),
      R: dense.data(md.R),
      I: dense.identity(n),
    }
    denseModels.set(md, d)
  }
  return d
}

/**
 * `kalmanStep` on row-major arrays with `aifn-compute/foundation/tensor`'s `dense` kernels and `aifn-compute/numerics/linalg`'s
 * `solveDense` (the LU of the solve primitive): the per-step recursion runs inside likelihood optimisations (ARMA,
 * EM), where tensor dispatch on 2×2 matrices would dominate. One LU of S solves for Kᵀ and S⁻¹v together.
 */
function stepDense(d: DenseModel, mean: F64, cov: F64, y: readonly number[]) {
  const { n, m, A, C, Q, R } = d
  const predictedMean = dense.matVec(A, mean, n, n)
  const predictedCov = dense.symmetrise(dense.add(dense.sandwich(A, cov, n, n), Q), n)
  const innovationCov = dense.symmetrise(dense.add(dense.sandwich(C, predictedCov, m, n), R), m)
  const out = {
    predictedMean,
    predictedCov,
    mean: predictedMean,
    cov: predictedCov,
    gain: new Float64Array(n * m) as F64,
    innovation: new Float64Array(m).fill(NaN) as F64,
    innovationCov,
    term: 0,
    singular: false,
  }
  // Missing entries drop out: the update uses the observed rows of C, the observed block of R and the observed y
  // (a fully missing step is a pure prediction).
  const observed: number[] = []
  y.forEach((v, i) => {
    if (!Number.isNaN(v)) observed.push(i)
  })
  if (observed.length === 0) return out
  const k = observed.length
  const all = k === m
  const Co = all
    ? C
    : (Float64Array.from({ length: k * n }, (_, t) => C[observed[Math.floor(t / n)] * n + (t % n)]) as F64)
  const Ro = all
    ? R
    : (Float64Array.from({ length: k * k }, (_, t) => R[observed[Math.floor(t / k)] * m + observed[t % k]]) as F64)
  const So = all
    ? innovationCov
    : (Float64Array.from(
        { length: k * k },
        (_, t) => innovationCov[observed[Math.floor(t / k)] * m + observed[t % k]],
      ) as F64)
  const yo = observed.map((i) => y[i])
  const innovationO = dense.sub(yo, dense.matVec(Co, predictedMean, k, n))
  for (let i = 0; i < k; i++) out.innovation[observed[i]] = innovationO[i]
  // The right-hand side [C P⁻ | v] (k × (n + 1)): Kᵀ = S⁻¹ C P⁻ and S⁻¹ v from one factorisation.
  const CP = dense.matMul(Co, predictedCov, k, n, n)
  const rhs = new Float64Array(k * (n + 1))
  for (let i = 0; i < k; i++) {
    rhs.set(CP.subarray(i * n, (i + 1) * n), i * (n + 1))
    rhs[i * (n + 1) + n] = innovationO[i]
  }
  const sol = solveDense(So, rhs, k)
  if (sol.x === null) return { ...out, term: NaN, singular: true }
  const x = sol.x
  const gainO = new Float64Array(n * k) as F64
  let quad = 0
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < n; j++) gainO[j * k + i] = x[i * (n + 1) + j]
    quad += innovationO[i] * x[i * (n + 1) + n]
  }
  const gain = new Float64Array(n * m) as F64
  for (let j = 0; j < n; j++) for (let i = 0; i < k; i++) gain[j * m + observed[i]] = gainO[j * k + i]
  const J = dense.sub(d.I, dense.matMul(gainO, Co, n, k, n))
  return {
    ...out,
    gain,
    mean: dense.add(predictedMean, dense.matVec(gainO, innovationO, n, k)),
    cov: dense.symmetrise(dense.add(dense.sandwich(J, predictedCov, n, n), dense.sandwich(gainO, Ro, n, k)), n),
    term: -0.5 * (k * Math.log(2 * Math.PI) + sol.logAbsDet + quad),
  }
}

/** The filter over a whole series: every step, the total log-likelihood and the steps with a singular S_t. */
export type FilterRun = { steps: KalmanStep[]; logLikelihood: number; singularSteps: number[] }

/** Run `kalmanStep` over the rows of `ys` (T rows of m observations, NaN entries missing) from (m₀, P₀). */
export function filterAll(md: Model, ys: readonly (readonly number[])[]): FilterRun {
  let mean = md.m0
  let cov = md.P0
  const out: FilterRun = { steps: [], logLikelihood: 0, singularSteps: [] }
  ys.forEach((y, t) => {
    const step = kalmanStep(md, mean, cov, y)
    out.steps.push(step)
    out.logLikelihood += step.term
    if (step.singular) out.singularSteps.push(t)
    mean = step.mean
    cov = step.cov
  })
  return out
}

/** The Kalman filter's output, over t = 1 … T. */
export type KalmanFilterResult = {
  /** μ_{t|t−1} ([T, n]). */
  predictedMean: Matrix
  /** P_{t|t−1} ([T, n, n]). */
  predictedCov: Tensor
  /** μ_{t|t} ([T, n]). */
  mean: Matrix
  /** P_{t|t} ([T, n, n]). */
  cov: Tensor
  /** K_t ([T, n, m]); zero for a missing or singular step. */
  gain: Tensor
  /** Innovations y_t − C μ_{t|t−1} ([T, m]); NaN for a missing step. */
  innovation: Matrix
  /** Innovation covariances S_t = C P_{t|t−1} Cᵀ + R ([T, m, m]). */
  innovationCov: Tensor
  /** log p(y₁, …, y_T) = Σ log N(y_t; C μ_{t|t−1}, S_t) over observed steps. */
  logLikelihood: number
  /** Each step's term of the log-likelihood (0 for a missing step). */
  logLikelihoodTerms: Tensor
  /** Steps whose S_t was singular (the update was skipped and the step's likelihood term is NaN). */
  singularSteps: number[]
}

/** The filter's steps stacked into the public `KalmanFilterResult` (n states, m observations). */
export function packFilter(f: FilterRun, n: number, m: number): KalmanFilterResult {
  const col = <K extends keyof KalmanStep>(k: K) => f.steps.map((s) => s[k] as Tensor)
  return {
    predictedMean: stackSteps(col('predictedMean'), [n]) as Matrix,
    predictedCov: stackSteps(col('predictedCov'), [n, n]),
    mean: stackSteps(col('mean'), [n]) as Matrix,
    cov: stackSteps(col('cov'), [n, n]),
    gain: stackSteps(col('gain'), [n, m]),
    innovation: stackSteps(col('innovation'), [m]) as Matrix,
    innovationCov: stackSteps(col('innovationCov'), [m, m]),
    logLikelihood: f.logLikelihood,
    logLikelihoodTerms: tensor(f.steps.map((s) => s.term)),
    singularSteps: f.singularSteps,
  }
}

/** The observations of a model as rows, checked against C (throws naming `where`). */
function observations(md: Model, y: VectorLike | MatrixLike, where: string): number[][] {
  const ys = asSeries(y, where)
  const [m] = md.C.shape
  if (ys.length && ys[0].length !== m)
    throw new ShapeError(where, `${where}: observations have ${ys[0].length} columns, C has ${m} rows`)
  return ys
}

/**
 * The Kalman filter (Kalman, 1960) over a whole series: `kalmanStep` at every t. `y` is [T, m] (or a length-T vector
 * when m = 1). No Cholesky factor is taken, so Q or R with zero rows filter normally; a singular S is reported in
 * `singularSteps`, never turned into NaN states. `kalmanFilterSteps` steps through the same recursion.
 */
export function kalmanFilter(model: StateSpaceModel, y: VectorLike | MatrixLike): KalmanFilterResult {
  const md = parseModel(model, 'kalmanFilter')
  const ys = observations(md, y, 'kalmanFilter')
  return packFilter(filterAll(md, ys), md.A.shape[0], md.C.shape[0])
}

// ── One smoother step ────────────────────────────────────────────────────────────────────────────────────────────────

/** One backward step of the RTS smoother: the smoothed moments of z_t and the gain that produced them. */
export type SmootherStep = {
  /** μ_{t|T}. */
  mean: Vector
  /** P_{t|T}. */
  cov: Matrix
  /** G_t = P_{t|t} Aᵀ P_{t+1|t}⁻¹. */
  gain: Matrix
  /** True when P_{t+1|t} was singular and a ridge 1e-12·max diag was added to solve. */
  singular: boolean
}

/**
 * One step of the Rauch–Tung–Striebel smoother (Rauch, Tung & Striebel, 1965), from the filtered moments of z_t
 * (μ_{t|t}, P_{t|t}; the prior for z₀), the predicted moments of z_{t+1} (μ_{t+1|t}, P_{t+1|t}) and the smoothed ones
 * of z_{t+1}: G = P_{t|t} Aᵀ P_{t+1|t}⁻¹ (solved as Gᵀ = P_{t+1|t}⁻¹ A P_{t|t}), μ_{t|T} = μ_{t|t} + G(μ_{t+1|T} −
 * μ_{t+1|t}), P_{t|T} = P_{t|t} + G(P_{t+1|T} − P_{t+1|t})Gᵀ.
 */
export function rtsStep(
  md: Model,
  filtered: { mean: Vector; cov: Matrix },
  predicted: { mean: Vector; cov: Matrix },
  smoothed: { mean: Vector; cov: Matrix },
): SmootherStep {
  const { n, A } = denseModel(md)
  const Pf = dense.data(filtered.cov)
  const Pp = dense.data(predicted.cov)
  const rhs = dense.matMul(A, Pf, n, n, n)
  let sol = solveDense(Pp, rhs, n).x
  let singular = false
  if (sol === null) {
    singular = true
    let scale = 0
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(Pp[i * n + i]))
    const ridged = Float64Array.from(Pp)
    for (let i = 0; i < n; i++) ridged[i * n + i] += 1e-12 * (scale || 1)
    sol = solveDense(ridged, rhs, n).x
  }
  const gain = sol === null ? new Float64Array(n * n) : dense.transpose(sol, n, n)
  const dm = dense.sub(dense.data(smoothed.mean), dense.data(predicted.mean))
  const dP = dense.sub(dense.data(smoothed.cov), Pp)
  return {
    gain: fromData(gain, [n, n]) as Matrix,
    mean: fromData(dense.add(dense.data(filtered.mean), dense.matVec(gain, dm, n, n)), [n]) as Vector,
    cov: fromData(dense.symmetrise(dense.add(Pf, dense.sandwich(gain, dP, n, n)), n), [n, n]) as Matrix,
    singular,
  }
}

/** The smoother over a whole series: smoothed moments and gains for t = 1 … T, lag-one covariances, and z₀'s. */
export type SmootherRun = {
  mean: Vector[]
  cov: Matrix[]
  /** G_t for t = 1 … T (the last is zero). */
  gain: Matrix[]
  /** Cov(z_t, z_{t−1} | y) = P_{t|T} G_{t−1}ᵀ, with G₀ the gain of the step back to z₀. */
  lag: Matrix[]
  initialMean: Vector
  initialCov: Matrix
  /** Steps whose P_{t+1|t} was singular (−1 for the step back to z₀). */
  singularSteps: number[]
}

/** The RTS smoother from the filter's steps: `rtsStep` from t = T − 1 back to z₀. */
export function smoothAll(md: Model, f: FilterRun): SmootherRun {
  const T = f.steps.length
  const [n] = md.A.shape
  const mean: Vector[] = new Array(T)
  const cov: Matrix[] = new Array(T)
  const gain: Matrix[] = new Array<Matrix>(T).fill(zeros([n, n]) as Matrix)
  const singularSteps: number[] = []
  if (T === 0) return { mean, cov, gain, lag: [], initialMean: md.m0, initialCov: md.P0, singularSteps }
  mean[T - 1] = f.steps[T - 1].mean
  cov[T - 1] = f.steps[T - 1].cov
  const back = (t: number, filtered: { mean: Vector; cov: Matrix }) => {
    const next = f.steps[t + 1]
    const s = rtsStep(
      md,
      filtered,
      { mean: next.predictedMean, cov: next.predictedCov },
      {
        mean: mean[t + 1],
        cov: cov[t + 1],
      },
    )
    if (s.singular) singularSteps.push(t)
    return s
  }
  for (let t = T - 2; t >= 0; t--) {
    const s = back(t, f.steps[t])
    gain[t] = s.gain
    mean[t] = s.mean
    cov[t] = s.cov
  }
  // One more step back to z₀, whose "filtered" moments are the prior (m₀, P₀); index −1 in `singularSteps`.
  const first = f.steps[0]
  const s0 = rtsStep(
    md,
    { mean: md.m0, cov: md.P0 },
    { mean: first.predictedMean, cov: first.predictedCov },
    { mean: mean[0], cov: cov[0] },
  )
  if (s0.singular) singularSteps.push(-1)
  // Cov(z_t, z_{t−1} | y) = P_{t|T} G_{t−1}ᵀ (de Jong, 1989; Shumway & Stoffer, 2017, Property 6.3 in this form).
  const lag = mean.map((_, t) => matmul(cov[t], transpose(t === 0 ? s0.gain : gain[t - 1])) as Matrix)
  return { mean, cov, gain, lag, initialMean: s0.mean, initialCov: s0.cov, singularSteps }
}

/** The RTS smoother's output: smoothed moments for t = 1 … T, and for z₀. */
export type SmootherResult = {
  /** μ_{t|T} ([T, n]). */
  mean: Matrix
  /** P_{t|T} ([T, n, n]). */
  cov: Tensor
  /** Smoother gains G_t = P_{t|t} Aᵀ P_{t+1|t}⁻¹ ([T, n, n]; the last is zero). */
  gain: Tensor
  /** Cov(z_t, z_{t−1} | y) = P_{t|T} G_{t−1}ᵀ ([T, n, n]; entry t pairs z_t with z_{t−1}, z₀ for the first). */
  lagOneCov: Tensor
  /** μ_{0|T} and P_{0|T}. */
  initialMean: Tensor
  initialCov: Matrix
  /** Steps whose predicted covariance P_{t+1|t} was singular; a pseudo-solve (ridge 1e-12·scale) was used there. */
  singularSteps: number[]
}

/**
 * The Rauch–Tung–Striebel smoother (Rauch, Tung & Striebel, 1965): the filter forwards, then `rtsStep` backwards from
 * the last filtered estimate. Also returns the lag-one covariances EM needs and the smoothed z₀.
 * `rtsSmootherSteps` steps through the backward pass.
 */
export function rtsSmoother(model: StateSpaceModel, y: VectorLike | MatrixLike): SmootherResult {
  const md = parseModel(model, 'rtsSmoother')
  const s = smoothAll(md, filterAll(md, observations(md, y, 'rtsSmoother')))
  const [n] = md.A.shape
  return {
    mean: stackSteps(s.mean, [n]) as Matrix,
    cov: stackSteps(s.cov, [n, n]),
    gain: stackSteps(s.gain, [n, n]),
    lagOneCov: stackSteps(s.lag, [n, n]),
    initialMean: s.initialMean,
    initialCov: s.initialCov,
    singularSteps: s.singularSteps,
  }
}

// ── The filter and smoother as algorithms ────────────────────────────────────────────────────────────────────────────

/** The state of `kalmanFilterSteps` after t observations. */
export type KalmanFilterState = Status & {
  /** Observations absorbed so far (0 … T). */
  t: number
  /** μ_{t|t} (m₀ at t = 0). */
  mean: Vector
  /** P_{t|t} (P₀ at t = 0). */
  cov: Matrix
  /** The last step's prediction, gain, innovation and term (null at t = 0). */
  step: KalmanStep | null
  /** log p(y₁, …, y_t). */
  logLikelihood: number
  /** Steps so far whose S was singular. */
  singularSteps: readonly number[]
  /** True once every observation is absorbed. */
  terminated: boolean
}

/**
 * The Kalman filter as a step-through `Algorithm` (start: none): step t absorbs y_t by `kalmanStep`, so `run(alg,
 * undefined, T)` ends on the filtered moments `kalmanFilter` gives at T, and a trace records every step's moments,
 * gain and innovation. Terminates after the last observation.
 */
export function kalmanFilterSteps(
  model: StateSpaceModel,
  y: VectorLike | MatrixLike,
): Algorithm<void, KalmanFilterState> {
  const md = parseModel(model, 'kalmanFilterSteps')
  const ys = observations(md, y, 'kalmanFilterSteps')
  const T = ys.length
  return {
    name: 'kalman-filter',
    init: () => ({
      t: 0,
      mean: md.m0,
      cov: md.P0,
      step: null,
      logLikelihood: 0,
      singularSteps: [],
      terminated: T === 0,
    }),
    step: (s) => {
      const step = kalmanStep(md, s.mean, s.cov, ys[s.t])
      return {
        t: s.t + 1,
        mean: step.mean,
        cov: step.cov,
        step,
        logLikelihood: s.logLikelihood + step.term,
        singularSteps: step.singular ? [...s.singularSteps, s.t] : s.singularSteps,
        terminated: s.t + 1 >= T,
      }
    },
  }
}

/** The state of `rtsSmootherSteps`: the smoothed moments of z_t, the backward pass's position. */
export type RtsSmootherState = Status & {
  /** Backward steps taken (0 … T). */
  t: number
  /** The time index smoothed last: T − 1 (0-based) after init, down to −1 for z₀. */
  index: number
  /** μ_{t|T}. */
  mean: Vector
  /** P_{t|T}. */
  cov: Matrix
  /** G_t of the last backward step (zero after init). */
  gain: Matrix
  /** Steps whose P_{t+1|t} was singular. */
  singularSteps: readonly number[]
  /** True once z₀ is smoothed. */
  terminated: boolean
}

/**
 * The RTS smoother's backward pass as a step-through `Algorithm` (start: none). The forward filter runs once when the
 * algorithm is made (pass `filtered`, the steps of `filterAll`, to reuse a run); init holds the last filtered
 * moments, which are already smoothed, and each step applies `rtsStep` one index back, ending at z₀ after T steps.
 */
export function rtsSmootherSteps(
  model: StateSpaceModel,
  y: VectorLike | MatrixLike,
  options: { filtered?: FilterRun } = {},
): Algorithm<void, RtsSmootherState> {
  const md = parseModel(model, 'rtsSmootherSteps')
  const f = options.filtered ?? filterAll(md, observations(md, y, 'rtsSmootherSteps'))
  const T = f.steps.length
  const [n] = md.A.shape
  const filteredAt = (t: number) => (t < 0 ? { mean: md.m0, cov: md.P0 } : f.steps[t])
  return {
    name: 'rts-smoother',
    init: () => {
      const last = filteredAt(T - 1)
      return {
        t: 0,
        index: T - 1,
        mean: last.mean,
        cov: last.cov,
        gain: zeros([n, n]) as Matrix,
        singularSteps: [],
        terminated: T === 0,
      }
    },
    step: (s) => {
      const index = s.index - 1
      const next = f.steps[s.index]
      const r = rtsStep(md, filteredAt(index), { mean: next.predictedMean, cov: next.predictedCov }, s)
      return {
        t: s.t + 1,
        index,
        mean: r.mean,
        cov: r.cov,
        gain: r.gain,
        singularSteps: r.singular ? [...s.singularSteps, index] : s.singularSteps,
        terminated: index < 0,
      }
    },
  }
}

// ── Steady state and consistency checks ──────────────────────────────────────────────────────────────────────────────

/**
 * The steady-state Kalman filter: iterate the Riccati recursion P⁻ ← A (P⁻ − P⁻Cᵀ S⁻¹ C P⁻) Aᵀ + Q from P₀ until the
 * largest change is below `tolerance` relative to P, and return the limiting gain and covariances. `converged` is false
 * when `maxSteps` (default 10 000) ran out, e.g. for an undetectable unstable mode (Anderson & Moore, 1979,
 * "Optimal Filtering", §4.4).
 */
export function steadyStateKalman(
  model: StateSpaceModel,
  { tolerance = 1e-12, maxSteps = 10000 }: { tolerance?: number; maxSteps?: number } = {},
): { gain: Matrix; predictedCov: Matrix; cov: Matrix; iterations: number; converged: boolean } {
  const md = parseModel(model, 'steadyStateKalman')
  const { A, C, Q, R } = md
  const [n] = A.shape
  const [m] = C.shape
  const I = eye(n)
  let Pp = symmetrise(add(sandwich(A, md.P0), Q) as Tensor) as Matrix
  let K = zeros([n, m]) as Matrix
  let Pf = Pp
  const result = (iterations: number, converged: boolean) => ({
    gain: K,
    predictedCov: Pp,
    cov: Pf,
    iterations,
    converged,
  })
  for (let it = 1; it <= maxSteps; it++) {
    const sol = solveOrNull(add(sandwich(C, Pp), R) as Tensor, matmul(C, Pp) as Tensor)
    if (sol === null) return result(it, false)
    K = transpose(sol.x) as Matrix
    const J = sub(I, matmul(K, C)) as Tensor
    Pf = symmetrise(add(sandwich(J, Pp), sandwich(K, R)) as Tensor) as Matrix
    const next = symmetrise(add(sandwich(A, Pf), Q) as Tensor) as Matrix
    const a = toFlat(next)
    const b = toFlat(Pp)
    const change = Math.max(...a.map((v, k) => Math.abs(v - b[k])))
    const size = Math.max(...a.map(Math.abs))
    Pp = next
    if (change <= tolerance * (1 + size)) return result(it, true)
  }
  return result(maxSteps, false)
}

/** Rows of a [T, k] tensor and slices of a [T, k, k] tensor, checked (throws `ShapeError` naming `where`). */
function rowsOf(t: Tensor, rank: 2 | 3, where: string): Tensor[] {
  if (t.shape.length !== rank)
    throw new ShapeError(where, `${where}: expected a rank-${rank} [T, …] tensor, got [${t.shape.join(', ')}]`)
  const [T, ...rest] = t.shape
  const size = rest.reduce((a, b) => a * b, 1)
  const flat = Float64Array.from(toFlat(t))
  return Array.from({ length: T }, (_, i) => fromData(flat.slice(i * size, (i + 1) * size), rest))
}

/** eᵀ S⁻¹ e for each row, NaN where e has a NaN or S is singular. */
function quadForms(e: Tensor[], S: Tensor[]): Tensor {
  return tensor(
    e.map((v, t) => {
      if (toFlat(v).some(Number.isNaN)) return NaN
      const sol = solveOrNull(S[t], v)
      return sol ? quadratic(v, sol.x) : NaN
    }),
  )
}

/**
 * The normalised innovation squared NIS_t = v_tᵀ S_t⁻¹ v_t of a Kalman filter's innovations ([T]; NaN at missing
 * steps). For a consistent filter (the model matches the data) each NIS_t is χ² with m degrees of freedom, so its
 * average over T steps should lie within the χ²_{mT}/T band (Bar-Shalom, Li & Kirubarajan, 2001, "Estimation with
 * Applications to Tracking and Navigation", §5.4.2).
 */
export function normalisedInnovationSquared(filter: Pick<KalmanFilterResult, 'innovation' | 'innovationCov'>): Tensor {
  const where = 'normalisedInnovationSquared'
  return quadForms(rowsOf(filter.innovation, 2, where), rowsOf(filter.innovationCov, 3, where))
}

/**
 * The normalised estimation error squared NEES_t = (z_t − μ_t)ᵀ P_t⁻¹ (z_t − μ_t) of estimates (μ_t, P_t) against the
 * true states z_t ([T]). Only a simulation knows z_t; for a consistent estimator each NEES_t is χ² with n degrees of
 * freedom (Bar-Shalom, Li & Kirubarajan, 2001, §5.4.2). `mean` and `truth` are [T, n], `cov` is [T, n, n]: pass the
 * filter's or the smoother's moments.
 */
export function normalisedEstimationErrorSquared(mean: Matrix, cov: Tensor, truth: Matrix): Tensor {
  const where = 'normalisedEstimationErrorSquared'
  const m = rowsOf(mean, 2, where)
  const z = rowsOf(truth, 2, where)
  if (m.length !== z.length) throw new ShapeError(where, `${where}: ${m.length} estimates but ${z.length} true states`)
  return quadForms(
    m.map((mi, t) => sub(z[t], mi) as Tensor),
    rowsOf(cov, 3, where),
  )
}
