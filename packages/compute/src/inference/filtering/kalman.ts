/**
 * Linear-Gaussian state-space models $\zvec_t = \Amat\zvec_{t-1} + \wvec_t$, $\yvec_t = \Cmat\zvec_t + \vvec_t$,
 * $\wvec_t \sim \Gauss(\zeros, \Qmat)$, $\vvec_t \sim \Gauss(\zeros, \Rmat)$, with
 * $\zvec_0 \sim \Gauss(\mvec_0, \Pmat_0)$: simulation, the Kalman filter, the Rauch–Tung–Striebel smoother, the
 * steady-state filter, the filter and smoother as step-through `Algorithm`s, and the consistency checks NIS and NEES.
 *
 * Convention: $(\mvec_0, \Pmat_0)$ describes $\zvec_0$, which is not observed; the first observation $\yvec_1$ is of
 * $\zvec_1 = \Amat\zvec_0 + \wvec_1$, so the filter predicts before its first update. NaN entries of $\yvec$ are
 * missing: a step updates with its observed entries only, and predicts through a row that is entirely NaN. $n$ is the
 * state dimension and $m$ the observation dimension throughout; per-step results are stacked along a first axis of
 * length $T$.
 *
 * Everything is written on tensors (`aifn-compute/foundation/tensor` arithmetic, `aifn-compute/numerics/linalg`
 * factorisations): one filter step (`kalmanStep`) and one smoother step (`rtsStep`) are the definitions, and the batch
 * functions, the algorithms and EM (`aifn-methods/timeseries`) all call them. Neither throws for a singular matrix:
 * the step is reported (`singular`, `singularSteps`) instead.
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

/** A linear-Gaussian state-space model. Scalars stand for $1 \times 1$ matrices (and a length-1 $\mvec_0$). */
export type StateSpaceModel = {
  /** Transition $\Amat$ ($n \times n$). */
  A: MatrixLike | number
  /** Observation $\Cmat$ ($m \times n$). */
  C: MatrixLike | number
  /** Process-noise covariance $\Qmat$ ($n \times n$, positive semi-definite; zero rows are allowed). */
  Q: MatrixLike | number
  /** Observation-noise covariance $\Rmat$ ($m \times m$, positive semi-definite; zero rows are allowed). */
  R: MatrixLike | number
  /** Mean $\mvec_0$ of $\zvec_0$ (length $n$). */
  m0: VectorLike | number
  /** Covariance $\Pmat_0$ of $\zvec_0$ ($n \times n$). */
  P0: MatrixLike | number
}

/**
 * The model with every part a tensor: the form the filter, the smoother and EM share. The fields are those of
 * `StateSpaceModel`: $\Amat$, $\Cmat$, $\Qmat$, $\Rmat$ and $\Pmat_0$ as matrices, $\mvec_0$ as a vector.
 */
export type Model = { A: Matrix; C: Matrix; Q: Matrix; R: Matrix; m0: Vector; P0: Matrix }

/**
 * Read a `StateSpaceModel` into tensors, checking that the shapes agree (throws `ShapeError` naming `where`). Tensor
 * parts are used as they are, not copied; values are not checked (a covariance need not be positive semi-definite).
 *
 * @param model The model, with numbers, arrays or tensors for its parts.
 * @param where The caller's name, for error messages.
 * @returns The model with every part a tensor.
 *
 * @example Numbers stand for 1 × 1 matrices
 * const md = parseModel({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, 'example')
 * print('A =', md.A, 'm0 =', md.m0)
 *
 * @example Shapes that disagree are reported
 * try {
 *   parseModel({ A: [[1, 1], [0, 1]], C: [[1, 0]], Q: 1, R: 1, m0: [0, 0], P0: [[1, 0], [0, 1]] }, 'example')
 * } catch (e) {
 *   print(e.message)
 * }
 */
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
 * Draw a trajectory of length $T$ and its observations. Noise is drawn as $\Smat\epsilonvec$ with
 * $\Smat\Smat^\top = \Qmat$ (or $\Rmat$) from a symmetric eigendecomposition, so covariances with deterministic
 * components (zero rows) give exact zeros rather than NaN. Returns $\zvec_1, \dots, \zvec_T$ (`[T, n]`) and
 * $\yvec_1, \dots, \yvec_T$ (`[T, m]`) and the drawn $\zvec_0$.
 *
 * @param s The random stream; not advanced: each draw comes from its own child stream (`'initial'`, then `'state'`
 *   and `'observation'` with the step index), so a longer simulation from the same stream extends a shorter one.
 * @param model The model to simulate.
 * @param T The number of steps to draw.
 * @returns `states` ($T \times n$), `observations` ($T \times m$) and `initial`, the drawn $\zvec_0$.
 *
 * @example A seeded random walk
 * // A scalar random walk z_t = z_{t-1} + w_t seen as y_t = z_t + v_t, unit noises, starting from z_0 = 0 exactly.
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const { states, observations } = simulateStateSpace(stream(0), walk, 5)
 * print('states =', states)
 * print('observations =', observations)
 *
 * @example A zero process noise keeps the state where it started
 * const still = { A: 1, C: 1, Q: 0, R: 1, m0: 2, P0: 0 }
 * print('states =', simulateStateSpace(stream(0), still, 4).states)
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
  /** The predicted mean $\muvec_{t \mid t-1}$. */
  predictedMean: Vector
  /** The predicted covariance $\Pmat_{t \mid t-1}$. */
  predictedCov: Matrix
  /** The filtered mean $\muvec_{t \mid t}$ (the predicted one when the update was skipped). */
  mean: Vector
  /** The filtered covariance $\Pmat_{t \mid t}$ (the predicted one when the update was skipped). */
  cov: Matrix
  /** $\Kmat_t$ ($n \times m$); zero for a missing or singular step, and in the columns of missing entries. */
  gain: Matrix
  /** $\yvec_t - \Cmat\muvec_{t \mid t-1}$; NaN in missing entries. */
  innovation: Vector
  /** $\Smat_t = \Cmat\Pmat_{t \mid t-1}\Cmat^\top + \Rmat$ (all $m$ rows, missing or not). */
  innovationCov: Matrix
  /**
   * $\log \Gauss(\yvec_t; \Cmat\muvec_{t \mid t-1}, \Smat_t)$ over the observed entries; 0 when every entry is
   * missing, NaN when $\Smat_t$ is singular.
   */
  term: number
  /** True when S_t was singular (the update was skipped). */
  singular: boolean
}

/**
 * One step of the Kalman filter (Kalman, 1960) from $(\muvec_{t-1 \mid t-1}, \Pmat_{t-1 \mid t-1})$ and the
 * observation $\yvec_t$ (NaN entries are missing: the update then uses the observed rows of $\Cmat$ and $\yvec$ and
 * the observed block of $\Rmat$, and a step with every entry missing is a prediction only). Predict
 * $\muvec^- = \Amat\muvec$, $\Pmat^- = \Amat\Pmat\Amat^\top + \Qmat$; update with
 * $\Smat = \Cmat\Pmat^-\Cmat^\top + \Rmat$, $\Kmat = \Pmat^-\Cmat^\top\Smat^{-1}$ (solved as
 * $\Kmat^\top = \Smat^{-1}\Cmat\Pmat^-$), $\muvec = \muvec^- + \Kmat(\yvec - \Cmat\muvec^-)$ and the covariance in
 * Joseph form $(\Imat - \Kmat\Cmat)\Pmat^-(\Imat - \Kmat\Cmat)^\top + \Kmat\Rmat\Kmat^\top$, which keeps
 * $\Pmat$ symmetric positive semi-definite in finite precision. A singular $\Smat$ skips the update and is reported
 * in `singular`, never thrown.
 *
 * @param md The model as tensors (`parseModel`). Its dense arrays are cached per model object, so reuse one object
 *   across steps.
 * @param mean The previous filtered mean $\muvec_{t-1 \mid t-1}$ (length $n$; $\mvec_0$ for the first step).
 * @param cov The previous filtered covariance $\Pmat_{t-1 \mid t-1}$ ($n \times n$; $\Pmat_0$ for the first step).
 * @param y The observation $\yvec_t$ as $m$ numbers, NaN where missing.
 * @returns The step's prediction, update, gain, innovation and log-likelihood term.
 *
 * @example One step of a scalar random walk
 * // z_0 = 0 exactly, Q = R = 1: predict variance 1, then the gain is 1 / (1 + 1) = 0.5.
 * const md = parseModel({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, 'example')
 * const step = kalmanStep(md, md.m0, md.P0, [1])
 * print('gain =', step.gain, 'mean =', step.mean, 'cov =', step.cov)
 * print('log N(1; 0, 2) =', step.term)
 *
 * @example A missing observation is a prediction only
 * const md = parseModel({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, 'example')
 * const step = kalmanStep(md, md.m0, md.P0, [NaN])
 * print('mean =', step.mean, 'cov =', step.cov, 'term =', step.term)
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

/** A float64 working array (`dense.F64`): the model's matrices as row-major arrays, zero-copy where contiguous. */
type F64 = dense.F64
/**
 * The model on row-major arrays: the dimensions $n$ and $m$, $\Amat$, $\Cmat$, $\Qmat$, $\Rmat$ and the
 * $n \times n$ identity `I`.
 */
type DenseModel = { n: number; m: number; A: F64; C: F64; Q: F64; R: F64; I: F64 }
const denseModels = new WeakMap<Model, DenseModel>()
/**
 * The model's matrices as row-major arrays, made once per model object and cached (a `WeakMap`, so a dropped model is
 * not kept alive).
 *
 * @param md The model as tensors.
 * @returns Its dense form.
 */
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
 * `kalmanStep` on row-major arrays with `aifn-compute/foundation/tensor`'s `dense` kernels and
 * `aifn-compute/numerics/linalg`'s `solveDense` (the LU of the solve primitive): the per-step recursion runs inside
 * likelihood optimisations (ARMA, EM), where tensor dispatch on $2 \times 2$ matrices would dominate. One LU of
 * $\Smat$ solves for $\Kmat^\top$ and $\Smat^{-1}\vvec$ together.
 *
 * @param d The model in dense form.
 * @param mean The previous filtered mean, $n$ values (not modified).
 * @param cov The previous filtered covariance, row-major $n \times n$ (not modified).
 * @param y The observation, $m$ numbers with NaN where missing.
 * @returns The fields of `KalmanStep` as row-major arrays (`gain` $n \times m$), with `term` and `singular`.
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

/**
 * The filter over a whole series: every step (`steps`), the total log-likelihood (`logLikelihood`, NaN when a step was
 * singular) and the 0-based indices of the steps with a singular $\Smat_t$ (`singularSteps`).
 */
export type FilterRun = { steps: KalmanStep[]; logLikelihood: number; singularSteps: number[] }

/**
 * Run `kalmanStep` over the rows of `ys` ($T$ rows of $m$ observations, NaN entries missing) from
 * $(\mvec_0, \Pmat_0)$.
 *
 * @param md The model as tensors (`parseModel`).
 * @param ys The observations, one row of $m$ numbers per step; the row lengths are not checked against $\Cmat$.
 * @returns Every step, the summed log-likelihood and the singular steps.
 *
 * @example Filter a scalar random walk
 * const md = parseModel({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, 'example')
 * const f = filterAll(md, [[1], [1], [1]])
 * print('means =', f.steps.map((s) => s.mean))
 * print('log-likelihood =', f.logLikelihood)
 */
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

/** The Kalman filter's output, over $t = 1, \dots, T$ (row $t - 1$ of each tensor). */
export type KalmanFilterResult = {
  /** $\muvec_{t \mid t-1}$ (`[T, n]`). */
  predictedMean: Matrix
  /** $\Pmat_{t \mid t-1}$ (`[T, n, n]`). */
  predictedCov: Tensor
  /** $\muvec_{t \mid t}$ (`[T, n]`). */
  mean: Matrix
  /** $\Pmat_{t \mid t}$ (`[T, n, n]`). */
  cov: Tensor
  /** $\Kmat_t$ (`[T, n, m]`); zero for a missing or singular step. */
  gain: Tensor
  /** Innovations $\yvec_t - \Cmat\muvec_{t \mid t-1}$ (`[T, m]`); NaN in missing entries. */
  innovation: Matrix
  /** Innovation covariances $\Smat_t = \Cmat\Pmat_{t \mid t-1}\Cmat^\top + \Rmat$ (`[T, m, m]`). */
  innovationCov: Tensor
  /**
   * $\log p(\yvec_1, \dots, \yvec_T) = \sum_t \log \Gauss(\yvec_t; \Cmat\muvec_{t \mid t-1}, \Smat_t)$ over the
   * observed entries.
   */
  logLikelihood: number
  /** Each step's term of the log-likelihood (0 for a missing step). */
  logLikelihoodTerms: Tensor
  /** Steps (0-based) whose $\Smat_t$ was singular (the update was skipped and the step's likelihood term is NaN). */
  singularSteps: number[]
}

/**
 * The filter's steps stacked into the public `KalmanFilterResult` ($n$ states, $m$ observations).
 *
 * @param f The run of `filterAll`.
 * @param n The state dimension, which shapes the empty result of a run with no steps.
 * @param m The observation dimension, likewise.
 * @returns The per-step moments, gains and innovations stacked along a first axis of length $T$.
 *
 * @example Stack a run into tensors
 * const md = parseModel({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, 'example')
 * const result = packFilter(filterAll(md, [[1], [1], [1]]), 1, 1)
 * print('mean =', result.mean)
 * print('gain =', result.gain)
 */
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

/**
 * The observations of a model as rows, checked against $\Cmat$ (throws `ShapeError` naming `where`).
 *
 * @param md The model as tensors.
 * @param y The observations: a vector of $T$ scalars (for $m = 1$) or a $T \times m$ matrix, NaN where missing.
 * @param where The caller's name, for error messages.
 * @returns $T$ rows of $m$ numbers.
 */
function observations(md: Model, y: VectorLike | MatrixLike, where: string): number[][] {
  const ys = asSeries(y, where)
  const [m] = md.C.shape
  if (ys.length && ys[0].length !== m)
    throw new ShapeError(where, `${where}: observations have ${ys[0].length} columns, C has ${m} rows`)
  return ys
}

/**
 * The Kalman filter (Kalman, 1960) over a whole series: `kalmanStep` at every $t$. No Cholesky factor is taken, so
 * $\Qmat$ or $\Rmat$ with zero rows filter normally; a singular $\Smat$ is reported in `singularSteps`, never turned
 * into NaN states. `kalmanFilterSteps` steps through the same recursion. Shapes that disagree throw `ShapeError`.
 *
 * @param model The state-space model.
 * @param y The observations, `[T, m]` (or a length-$T$ vector when $m = 1$), NaN where missing.
 * @returns The predicted and filtered moments, gains, innovations and log-likelihood of every step.
 *
 * @example Track a scalar random walk
 * // A scalar random walk z_t = z_{t-1} + w_t seen as y_t = z_t + v_t, unit noises, starting from z_0 = 0 exactly.
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const f = kalmanFilter(walk, [1, 1, 1])
 * print('filtered means =', f.mean)
 * print('gains =', f.gain)
 * print('log-likelihood =', f.logLikelihood)
 *
 * @example A missing observation widens the estimate
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const f = kalmanFilter(walk, [1, NaN, 1])
 * print('filtered variances =', f.cov)
 * print('likelihood terms =', f.logLikelihoodTerms)
 */
export function kalmanFilter(model: StateSpaceModel, y: VectorLike | MatrixLike): KalmanFilterResult {
  const md = parseModel(model, 'kalmanFilter')
  const ys = observations(md, y, 'kalmanFilter')
  return packFilter(filterAll(md, ys), md.A.shape[0], md.C.shape[0])
}

// ── One smoother step ────────────────────────────────────────────────────────────────────────────────────────────────

/** One backward step of the RTS smoother: the smoothed moments of $\zvec_t$ and the gain that produced them. */
export type SmootherStep = {
  /** $\muvec_{t \mid T}$. */
  mean: Vector
  /** $\Pmat_{t \mid T}$. */
  cov: Matrix
  /** $\Gmat_t = \Pmat_{t \mid t}\Amat^\top\Pmat_{t+1 \mid t}^{-1}$. */
  gain: Matrix
  /**
   * True when $\Pmat_{t+1 \mid t}$ was singular and a ridge of $10^{-12}$ times its largest diagonal entry was added
   * to solve (the gain is zero if that too failed).
   */
  singular: boolean
}

/**
 * One step of the Rauch–Tung–Striebel smoother (Rauch, Tung & Striebel, 1965), from the filtered moments of
 * $\zvec_t$ ($\muvec_{t \mid t}$, $\Pmat_{t \mid t}$; the prior for $\zvec_0$), the predicted moments of
 * $\zvec_{t+1}$ ($\muvec_{t+1 \mid t}$, $\Pmat_{t+1 \mid t}$) and the smoothed ones of $\zvec_{t+1}$:
 * $\Gmat = \Pmat_{t \mid t}\Amat^\top\Pmat_{t+1 \mid t}^{-1}$ (solved as
 * $\Gmat^\top = \Pmat_{t+1 \mid t}^{-1}\Amat\Pmat_{t \mid t}$),
 * $\muvec_{t \mid T} = \muvec_{t \mid t} + \Gmat(\muvec_{t+1 \mid T} - \muvec_{t+1 \mid t})$,
 * $\Pmat_{t \mid T} = \Pmat_{t \mid t} + \Gmat(\Pmat_{t+1 \mid T} - \Pmat_{t+1 \mid t})\Gmat^\top$.
 *
 * @param md The model as tensors (`parseModel`); only $\Amat$ is used.
 * @param filtered The filtered mean and covariance of $\zvec_t$ (a `KalmanStep` will do).
 * @param predicted The predicted mean and covariance of $\zvec_{t+1}$: the next step's `predictedMean` and
 *   `predictedCov`.
 * @param smoothed The smoothed mean and covariance of $\zvec_{t+1}$ (for the last step, its filtered moments).
 * @returns The smoothed moments of $\zvec_t$, the gain, and whether the predicted covariance was singular.
 *
 * @example One step back from the last observation
 * // After y = (1, 1): z_1 filtered N(0.5, 0.5), z_2 predicted N(0.5, 1.5) and filtered N(0.8, 0.6).
 * const md = parseModel({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, 'example')
 * const [first, last] = filterAll(md, [[1], [1]]).steps
 * const back = rtsStep(md, first, { mean: last.predictedMean, cov: last.predictedCov }, last)
 * print('gain =', back.gain, 'exact', 0.5 / 1.5)
 * print('smoothed mean =', back.mean, 'cov =', back.cov)
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

/**
 * The smoother over a whole series: smoothed moments and gains for $t = 1, \dots, T$, lag-one covariances, and
 * $\zvec_0$'s.
 */
export type SmootherRun = {
  /** $\muvec_{t \mid T}$ for $t = 1, \dots, T$. */
  mean: Vector[]
  /** $\Pmat_{t \mid T}$ for $t = 1, \dots, T$. */
  cov: Matrix[]
  /** $\Gmat_t$ for $t = 1, \dots, T$ (the last is zero). */
  gain: Matrix[]
  /**
   * $\cov(\zvec_t, \zvec_{t-1} \mid \yvec) = \Pmat_{t \mid T}\Gmat_{t-1}^\top$, with $\Gmat_0$ the gain of the step
   * back to $\zvec_0$.
   */
  lag: Matrix[]
  /** $\muvec_{0 \mid T}$. */
  initialMean: Vector
  /** $\Pmat_{0 \mid T}$. */
  initialCov: Matrix
  /** Steps (0-based) whose $\Pmat_{t+1 \mid t}$ was singular ($-1$ for the step back to $\zvec_0$). */
  singularSteps: number[]
}

/**
 * The RTS smoother from the filter's steps: `rtsStep` from $t = T - 1$ back to $\zvec_0$, then the lag-one
 * covariances (de Jong, 1989; Shumway & Stoffer, 2017, Property 6.3). With no steps the result is empty and
 * $\zvec_0$ keeps its prior.
 *
 * @param md The model as tensors (`parseModel`).
 * @param f The run of `filterAll` on the same model.
 * @returns The smoothed moments and gains of every step, the lag-one covariances, $\zvec_0$'s smoothed moments, and
 *   the singular steps.
 *
 * @example Smooth a filtered random walk
 * const md = parseModel({ A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }, 'example')
 * const s = smoothAll(md, filterAll(md, [[1], [1], [1]]))
 * print('smoothed means =', s.mean)
 * print('z_0 =', s.initialMean, 'with variance', s.initialCov)
 */
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

/** The RTS smoother's output: smoothed moments for $t = 1, \dots, T$, and for $\zvec_0$. */
export type SmootherResult = {
  /** $\muvec_{t \mid T}$ (`[T, n]`). */
  mean: Matrix
  /** $\Pmat_{t \mid T}$ (`[T, n, n]`). */
  cov: Tensor
  /** Smoother gains $\Gmat_t = \Pmat_{t \mid t}\Amat^\top\Pmat_{t+1 \mid t}^{-1}$ (`[T, n, n]`; the last is zero). */
  gain: Tensor
  /**
   * $\cov(\zvec_t, \zvec_{t-1} \mid \yvec) = \Pmat_{t \mid T}\Gmat_{t-1}^\top$ (`[T, n, n]`; entry $t$ pairs
   * $\zvec_t$ with $\zvec_{t-1}$, $\zvec_0$ for the first).
   */
  lagOneCov: Tensor
  /** $\muvec_{0 \mid T}$. */
  initialMean: Tensor
  /** $\Pmat_{0 \mid T}$. */
  initialCov: Matrix
  /**
   * Steps (0-based; $-1$ for $\zvec_0$) whose predicted covariance $\Pmat_{t+1 \mid t}$ was singular; a pseudo-solve
   * (a ridge of $10^{-12}$ times its largest diagonal entry) was used there.
   */
  singularSteps: number[]
}

/**
 * The Rauch–Tung–Striebel smoother (Rauch, Tung & Striebel, 1965): the filter forwards, then `rtsStep` backwards from
 * the last filtered estimate. Also returns the lag-one covariances EM needs and the smoothed $\zvec_0$.
 * `rtsSmootherSteps` steps through the backward pass. Shapes that disagree throw `ShapeError`.
 *
 * @param model The state-space model.
 * @param y The observations, `[T, m]` (or a length-$T$ vector when $m = 1$), NaN where missing.
 * @returns The smoothed moments and gains of every step, the lag-one covariances, and $\zvec_0$'s smoothed moments.
 *
 * @example Smoothing uses the later observations too
 * // A scalar random walk z_t = z_{t-1} + w_t seen as y_t = z_t + v_t, unit noises, starting from z_0 = 0 exactly.
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * print('filtered =', kalmanFilter(walk, [1, 1, 1]).mean)
 * const s = rtsSmoother(walk, [1, 1, 1])
 * print('smoothed =', s.mean)
 * print('smoothed variances =', s.cov)
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

/** The state of `kalmanFilterSteps` after $t$ observations. */
export type KalmanFilterState = Status & {
  /** Observations absorbed so far ($0, \dots, T$). */
  t: number
  /** $\muvec_{t \mid t}$ ($\mvec_0$ at $t = 0$). */
  mean: Vector
  /** $\Pmat_{t \mid t}$ ($\Pmat_0$ at $t = 0$). */
  cov: Matrix
  /** The last step's prediction, gain, innovation and term (null at t = 0). */
  step: KalmanStep | null
  /** $\log p(\yvec_1, \dots, \yvec_t)$. */
  logLikelihood: number
  /** Steps so far (0-based) whose $\Smat$ was singular. */
  singularSteps: readonly number[]
  /** True once every observation is absorbed. */
  terminated: boolean
}

/**
 * The Kalman filter as a step-through `Algorithm` (start: none): step $t$ absorbs $\yvec_t$ by `kalmanStep`, so
 * `run(alg, undefined, T)` ends on the filtered moments `kalmanFilter` gives at $T$, and a trace records every step's
 * moments, gain and innovation. Terminates after the last observation. The model and observations are read and
 * checked when the algorithm is made (`ShapeError`).
 *
 * @param model The state-space model.
 * @param y The observations, `[T, m]` (or a length-$T$ vector when $m = 1$), NaN where missing.
 * @returns The algorithm; it takes no start and draws no random numbers.
 *
 * @example Absorb the observations one at a time
 * // A scalar random walk z_t = z_{t-1} + w_t seen as y_t = z_t + v_t, unit noises, starting from z_0 = 0 exactly.
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const filter = kalmanFilterSteps(walk, [1, 1, 1])
 * const after2 = run(filter, undefined, 2)
 * print('t =', after2.t, 'mean =', after2.mean, 'cov =', after2.cov)
 * const done = run(filter, undefined, 10)
 * print('t =', done.t, 'mean =', done.mean, 'terminated =', done.terminated)
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

/** The state of `rtsSmootherSteps`: the smoothed moments of $\zvec_t$, the backward pass's position. */
export type RtsSmootherState = Status & {
  /** Backward steps taken ($0, \dots, T$). */
  t: number
  /** The time index smoothed last: $T - 1$ (0-based) after init, down to $-1$ for $\zvec_0$. */
  index: number
  /** The smoothed mean of the state at `index`. */
  mean: Vector
  /** The smoothed covariance of the state at `index`. */
  cov: Matrix
  /** The gain $\Gmat$ of the last backward step (zero after init). */
  gain: Matrix
  /** Indices (0-based, $-1$ for $\zvec_0$) whose $\Pmat_{t+1 \mid t}$ was singular. */
  singularSteps: readonly number[]
  /** True once $\zvec_0$ is smoothed. */
  terminated: boolean
}

/**
 * The RTS smoother's backward pass as a step-through `Algorithm` (start: none). The forward filter runs once when the
 * algorithm is made (pass `filtered`, the steps of `filterAll`, to reuse a run); init holds the last filtered
 * moments, which are already smoothed, and each step applies `rtsStep` one index back, ending at $\zvec_0$ after $T$
 * steps.
 *
 * @param model The state-space model.
 * @param y The observations, `[T, m]` (or a length-$T$ vector when $m = 1$), NaN where missing. Not read when
 *   `options.filtered` is given.
 * @param options `filtered`, a run of `filterAll` on the same model to smooth instead of filtering `y` again.
 * @returns The algorithm; it takes no start and draws no random numbers.
 *
 * @example Step back from the last observation to z_0
 * // A scalar random walk z_t = z_{t-1} + w_t seen as y_t = z_t + v_t, unit noises, starting from z_0 = 0 exactly.
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const smoother = rtsSmootherSteps(walk, [1, 1, 1])
 * const one = run(smoother, undefined, 1)
 * print('index', one.index, 'mean =', one.mean)
 * const done = run(smoother, undefined, 10)
 * print('index', done.index, 'mean =', done.mean, 'terminated =', done.terminated)
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
 * The steady-state Kalman filter: iterate the Riccati recursion
 * $\Pmat^- \leftarrow \Amat(\Pmat^- - \Pmat^-\Cmat^\top\Smat^{-1}\Cmat\Pmat^-)\Amat^\top + \Qmat$ from
 * $\Amat\Pmat_0\Amat^\top + \Qmat$ until the largest change of an entry is at most `tolerance` times
 * $1 + \max_{ij} \lvert P^-_{ij} \rvert$, and return the limiting gain and covariances. `converged` is false when
 * `maxSteps` ran out, e.g. for an undetectable unstable mode (Anderson & Moore, 1979, "Optimal Filtering", §4.4), or
 * when $\Smat$ became singular.
 *
 * @param model The state-space model; $\mvec_0$ is not used.
 * @param options The stopping rule.
 * @param options.tolerance The relative change at which the recursion has converged (default $10^{-12}$).
 * @param options.maxSteps The most iterations to run (default 10 000).
 * @returns The gain $\Kmat$ ($n \times m$), the predicted covariance $\Pmat^-$ and the filtered covariance $\Pmat$
 *   (both $n \times n$), the number of iterations, and whether it converged.
 *
 * @example The scalar random walk's steady gain is the golden ratio's inverse
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const { gain, predictedCov, cov, converged } = steadyStateKalman(walk)
 * print('gain =', gain, 'exact', (Math.sqrt(5) - 1) / 2)
 * print('predicted variance =', predictedCov, 'filtered variance =', cov)
 * print('converged =', converged)
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

/**
 * Rows of a `[T, k]` tensor and slices of a `[T, k, k]` tensor, checked (throws `ShapeError` naming `where`).
 *
 * @param t The stacked tensor.
 * @param rank Its expected rank: 2 for rows, 3 for matrices.
 * @param where The caller's name, for error messages.
 * @returns $T$ new tensors, each of the shape of one slice.
 */
function rowsOf(t: Tensor, rank: 2 | 3, where: string): Tensor[] {
  if (t.shape.length !== rank)
    throw new ShapeError(where, `${where}: expected a rank-${rank} [T, …] tensor, got [${t.shape.join(', ')}]`)
  const [T, ...rest] = t.shape
  const size = rest.reduce((a, b) => a * b, 1)
  const flat = Float64Array.from(toFlat(t))
  return Array.from({ length: T }, (_, i) => fromData(flat.slice(i * size, (i + 1) * size), rest))
}

/**
 * $\evec^\top\Smat^{-1}\evec$ for each row, NaN where $\evec$ has a NaN or $\Smat$ is singular.
 *
 * @param e The vectors $\evec_t$.
 * @param S The matrices $\Smat_t$, one per vector.
 * @returns The $T$ quadratic forms.
 */
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
 * The normalised innovation squared $\operatorname{NIS}_t = \vvec_t^\top\Smat_t^{-1}\vvec_t$ of a Kalman filter's
 * innovations (`[T]`; NaN at a step with any entry missing or a singular $\Smat_t$). For a consistent filter (the
 * model matches the data) each $\operatorname{NIS}_t$ is $\chi^2$ with $m$ degrees of freedom, so its average over $T$
 * steps should lie within the $\chi^2_{mT}/T$ band (Bar-Shalom, Li & Kirubarajan, 2001, "Estimation with Applications
 * to Tracking and Navigation", §5.4.2).
 *
 * @param filter The innovations (`[T, m]`) and innovation covariances (`[T, m, m]`), as `kalmanFilter` returns them.
 * @returns The $T$ values of $\operatorname{NIS}_t$.
 *
 * @example A filter of the right model averages about m = 1
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const { observations } = simulateStateSpace(stream(0), walk, 200)
 * const nis = normalisedInnovationSquared(kalmanFilter(walk, observations))
 * print('mean NIS =', mean(nis))
 * // The same data filtered with R ten times too large: the innovations look too small.
 * const wrong = normalisedInnovationSquared(kalmanFilter({ ...walk, R: 10 }, observations))
 * print('mean NIS, R too large =', mean(wrong))
 */
export function normalisedInnovationSquared(filter: Pick<KalmanFilterResult, 'innovation' | 'innovationCov'>): Tensor {
  const where = 'normalisedInnovationSquared'
  return quadForms(rowsOf(filter.innovation, 2, where), rowsOf(filter.innovationCov, 3, where))
}

/**
 * The normalised estimation error squared
 * $\operatorname{NEES}_t = (\zvec_t - \muvec_t)^\top\Pmat_t^{-1}(\zvec_t - \muvec_t)$ of estimates
 * $(\muvec_t, \Pmat_t)$ against the true states $\zvec_t$ (`[T]`). Only a simulation knows $\zvec_t$; for a
 * consistent estimator each $\operatorname{NEES}_t$ is $\chi^2$ with $n$ degrees of freedom (Bar-Shalom, Li &
 * Kirubarajan, 2001, §5.4.2). Pass the filter's or the smoother's moments. A different number of estimates and true
 * states throws `ShapeError`.
 *
 * @param mean The estimated means $\muvec_t$ (`[T, n]`).
 * @param cov Their covariances $\Pmat_t$ (`[T, n, n]`), with as many slices as `mean` has rows.
 * @param truth The true states $\zvec_t$ (`[T, n]`).
 * @returns The $T$ values of $\operatorname{NEES}_t$ (NaN where $\Pmat_t$ is singular).
 *
 * @example The smoother's errors are consistent too
 * const walk = { A: 1, C: 1, Q: 1, R: 1, m0: 0, P0: 0 }
 * const { states, observations } = simulateStateSpace(stream(0), walk, 200)
 * const f = kalmanFilter(walk, observations)
 * const s = rtsSmoother(walk, observations)
 * print('mean NEES, filter =', mean(normalisedEstimationErrorSquared(f.mean, f.cov, states)))
 * print('mean NEES, smoother =', mean(normalisedEstimationErrorSquared(s.mean, s.cov, states)))
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
