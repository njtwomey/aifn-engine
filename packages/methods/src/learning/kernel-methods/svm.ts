/**
 * Support vector machines for two classes.
 *
 * - `smoSteps`, `supportVectorMachine`: the kernel SVM's dual solved by sequential minimal optimisation (Platt, 1998),
 *   in the form of LIBSVM (Chang and Lin, 2011): the working pair is the maximal violating pair (Keerthi et al., 2001)
 *   or chosen with second-order information (Fan, Chen and Lin, 2005, "Working set selection using second order
 *   information for training support vector machines", JMLR 6, WSS 3); the pair is updated in closed form and clipped
 *   to the box; the bias is LIBSVM's $-\rho$.
 * - `dualCoordinateSteps`, `pegasosSteps`, `linearSvm`: the linear hinge-loss SVM by dual coordinate descent (Hsieh
 *   et al., 2008, "A dual coordinate descent method for large-scale linear SVM", ICML) or by Pegasos stochastic
 *   subgradient steps (Shalev-Shwartz, Singer and Srebro, 2007).
 *
 * The primal is $\min \frac{1}{2}\lVert \wvec \rVert^2 + C \sum_i \max(0, 1 - y_i f(\xvec_i))$ with labels
 * $y_i = \pm 1$; the dual is $\min_{\alphavec} \frac{1}{2} \alphavec^\top\Qmat\alphavec - \ones^\top\alphavec$ subject
 * to $0 \le \alpha_i \le C$ and $\yvec^\top\alphavec = 0$, with $Q_{ij} = y_i y_j k(\xvec_i, \xvec_j)$. The estimators
 * (`supportVectorMachine`, `linearSvm`) take labels 0/1 and map them to $-1$/$+1$; the step algorithms take $\pm 1$.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type {
  Decides,
  Estimator,
  FitOptions,
  Fitted,
  Scores,
  Supervised,
  Trained,
} from 'aifn-compute/learning/estimators'
import { bernoulliPredictive, type AnyUnivariate } from 'aifn-compute/learning/estimators'
import { integers, permutation } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { gram, rbf, type Kernel } from 'aifn-compute/learning/kernels'
import { classLabels, inputs, matrix, values } from '../util'
import { plattScaling, type PlattScaling } from 'aifn-compute/learning/calibration'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The curvature of a pair at or below 0 is replaced by this (LIBSVM's $\tau$), so a degenerate pair still moves. */
const TAU = 1e-12

/** The dual problem an SMO run solves. */
export interface SmoProblem {
  /** Inputs $n \times d$. */
  x: Tensor
  /** Labels $\pm 1$, $n$ of them. */
  y: Tensor
  /** The box constraint $C$ (default 1). */
  C?: number
  /** The kernel $k$ (default `rbf({ lengthscale: 1 })`). */
  kernel?: Kernel
  /**
   * Stop when the maximal KKT violation $m(\alphavec) - M(\alphavec)$ is at most `tolerance` (default 1e-3, LIBSVM's
   * and scikit-learn's).
   */
  tolerance?: number
  /** How the working pair is chosen (default `second-order`, as LIBSVM). */
  selection?: 'maximal-violating' | 'second-order'
}

/** One SMO state. */
export interface SmoState extends Status {
  /** Pair updates done. */
  t: number
  /** Dual variables $\alphavec$, $n$ of them. */
  alpha: Tensor
  /** Gradient of the dual objective, $\gvec = \Qmat\alphavec - \ones$, $n$ values. */
  grad: Tensor
  /**
   * Bias $b = -\rho$ of the decision function $f(\xvec) = \sum_t \alpha_t y_t k(\xvec_t, \xvec) + b$ at this
   * $\alphavec$.
   */
  bias: number
  /** Errors $E_t = f(\xvec_t) - y_t$, $n$ values. */
  errors: Tensor
  /** The working pair $(i, j)$ updated to reach this state ($[-1, -1]$ at the start). */
  pair: [number, number]
  /** The change in $\alpha_j$ the pair's closed-form step asked for, before clipping to the box. */
  step: number
  /** Whether clipping to the box changed the step. */
  clipped: boolean
  /** The maximal violation $m(\alphavec) - M(\alphavec)$ at this $\alphavec$ (the stopping measure). */
  gap: number
  /** The pair that attains `gap`, which the next step starts from ($-1$ entries when there is none). */
  violatingPair: [number, number]
  /** The dual objective to maximise, $\ones^\top\alphavec - \frac{1}{2} \alphavec^\top\Qmat\alphavec$. */
  dualObjective: number
  /** True when `gap` is at most the tolerance, or no violating pair is left. */
  converged: boolean
}

/**
 * The dual problem's data, computed once per run: the kernel matrix and labels, with the defaults filled in. Throws
 * `ShapeError` when the labels do not match the rows and `DomainError` when a label is not $\pm 1$.
 *
 * @param problem The problem as `smoSteps` was given it.
 * @returns $n$, the labels `y`, the Gram matrix `K` (row-major $n \times n$), `C`, the tolerance `tol` and the
 *   `selection` rule.
 */
function prepare(problem: SmoProblem) {
  const { n } = matrix(problem.x, 'smoSteps')
  const y = values(problem.y)
  if (y.length !== n) throw new ShapeError('smoSteps', `smoSteps: ${n} rows but ${y.length} labels`)
  for (const t of y) if (t !== 1 && t !== -1) throw new DomainError('smoSteps', 'smoSteps: labels must be ±1')
  const kernel = problem.kernel ?? rbf({ lengthscale: 1 })
  const K = Float64Array.from(values(gram(kernel, problem.x)))
  return { n, y, K, C: problem.C ?? 1, tol: problem.tolerance ?? 1e-3, selection: problem.selection ?? 'second-order' }
}

/** The data of a run, as `prepare` returns it. */
type Prepared = ReturnType<typeof prepare>

/**
 * Whether index $t$ is in $I_{\mathrm{up}}$: $\alpha_t$ can grow in the direction $y_t$ without leaving the box.
 *
 * @param a The dual variable $\alpha_t$.
 * @param y Its label $y_t = \pm 1$.
 * @param C The box constraint $C$.
 * @returns True for $\alpha_t < C$ when $y_t = 1$, and for $\alpha_t > 0$ when $y_t = -1$.
 */
const inUp = (a: number, y: number, C: number) => (y > 0 ? a < C : a > 0)
/**
 * Whether index $t$ is in $I_{\mathrm{low}}$: $\alpha_t$ can move against the direction $y_t$ without leaving the box.
 *
 * @param a The dual variable $\alpha_t$.
 * @param y Its label $y_t = \pm 1$.
 * @param C The box constraint $C$.
 * @returns True for $\alpha_t > 0$ when $y_t = 1$, and for $\alpha_t < C$ when $y_t = -1$.
 */
const inLow = (a: number, y: number, C: number) => (y > 0 ? a > 0 : a < C)

/**
 * The maximal violating pair ($i \in I_{\mathrm{up}}$ maximising $-y_t G_t$, $j \in I_{\mathrm{low}}$ minimising it)
 * and the gap $m - M$ between those two values (Keerthi et al., 2001).
 *
 * @param p The run's data.
 * @param alpha The dual variables $\alphavec$, $n$ values (read only).
 * @param G The gradient $\gvec = \Qmat\alphavec - \ones$, $n$ values (read only).
 * @returns The pair `i`, `j` ($-1$ where a set is empty) and `gap` (0 when either is).
 */
function violation(p: Prepared, alpha: Float64Array, G: Float64Array) {
  let i = -1
  let j = -1
  let m = -Infinity
  let M = Infinity
  for (let t = 0; t < p.n; t++) {
    const v = -p.y[t] * G[t]
    if (inUp(alpha[t], p.y[t], p.C) && v > m) {
      m = v
      i = t
    }
    if (inLow(alpha[t], p.y[t], p.C) && v < M) {
      M = v
      j = t
    }
  }
  return { i, j, gap: i < 0 || j < 0 ? 0 : m - M }
}

/**
 * LIBSVM's $\rho$: the mean of $y_t G_t$ over the free variables ($0 < \alpha_t < C$), or the midpoint of the feasible
 * interval the bounded ones leave when none is free.
 *
 * @param p The run's data.
 * @param alpha The dual variables $\alphavec$, $n$ values (read only).
 * @param G The gradient $\gvec = \Qmat\alphavec - \ones$, $n$ values (read only).
 * @returns $\rho$; the bias is $b = -\rho$.
 */
function rho(p: Prepared, alpha: Float64Array, G: Float64Array): number {
  let ub = Infinity
  let lb = -Infinity
  let free = 0
  let sum = 0
  for (let t = 0; t < p.n; t++) {
    const yG = p.y[t] * G[t]
    if (alpha[t] >= p.C) {
      if (p.y[t] < 0) ub = Math.min(ub, yG)
      else lb = Math.max(lb, yG)
    } else if (alpha[t] <= 0) {
      if (p.y[t] > 0) ub = Math.min(ub, yG)
      else lb = Math.max(lb, yG)
    } else {
      free++
      sum += yG
    }
  }
  return free > 0 ? sum / free : (ub + lb) / 2
}

/**
 * The SMO state at $\alphavec$: bias, errors, violating pair, gap and dual objective computed from the gradient.
 *
 * @param p The run's data.
 * @param alpha The dual variables $\alphavec$, $n$ values; kept by the state, so not to be written afterwards.
 * @param G The gradient $\gvec = \Qmat\alphavec - \ones$, $n$ values; kept by the state likewise.
 * @param extra The step count `t`, and any fields that override the computed ones (`pair`, `step`, `clipped`).
 * @returns The state.
 */
function stateAt(
  p: Prepared,
  alpha: Float64Array,
  G: Float64Array,
  extra: Partial<SmoState> & { t: number },
): SmoState {
  const v = violation(p, alpha, G)
  const b = -rho(p, alpha, G)
  // 1ᵀα − ½ αᵀQα, with Qα = G + 1.
  let dualObjective = 0
  for (let t = 0; t < p.n; t++) dualObjective += alpha[t] - 0.5 * alpha[t] * (G[t] + 1)
  return {
    alpha: fromData(alpha, [p.n]),
    grad: fromData(G, [p.n]),
    bias: b,
    errors: fromData(
      Float64Array.from(G, (g, t) => p.y[t] * g + b),
      [p.n],
    ),
    pair: [-1, -1],
    step: 0,
    clipped: false,
    gap: v.gap,
    violatingPair: [v.i, v.j],
    dualObjective,
    converged: v.gap <= p.tol,
    ...extra,
  }
}

/**
 * SMO for the kernel SVM's dual as a traceable algorithm. Each step picks a working pair $(i, j)$, moves $\alpha_i$ and
 * $\alpha_j$ along the line $y_i\alpha_i + y_j\alpha_j = \text{const}$ to the minimum of the dual on it, clips the move
 * to the box $[0, C]^2$, and updates the gradient. It has converged when the maximal KKT violation is at most
 * `tolerance`. `init` takes an optional feasible starting $\alphavec$ (default $\zeros$). Throws `ShapeError` or
 * `DomainError` for labels that do not match the rows or are not $\pm 1$.
 *
 * @param problem The inputs, the $\pm 1$ labels, $C$, the kernel, the tolerance and the pair selection rule.
 * @returns The algorithm: one pair update per step, with the state's `pair`, `step` and `clipped` describing it.
 *
 * @example Run to convergence on eight points on a line, two classes either side of a gap
 * const x = tensor([[0], [0.5], [1], [1.5], [2.5], [3], [3.5], [4]])
 * const y = tensor([-1, -1, -1, -1, 1, 1, 1, 1])
 * const s = run(smoSteps({ x, y, C: 10 }), {}, 100)
 * print('alpha =', s.alpha)
 * print('b =', s.bias, 'gap =', s.gap, 'converged:', s.converged, 'after', s.t, 'steps')
 *
 * @example The first steps: the pair each updates, and the dual objective rising
 * const x = tensor([[0], [0.5], [1], [1.5], [2.5], [3], [3.5], [4]])
 * const y = tensor([-1, -1, -1, -1, 1, 1, 1, 1])
 * const alg = smoSteps({ x, y, C: 10 })
 * for (let k = 1; k <= 3; k++) {
 *   const s = run(alg, {}, k)
 *   print(`step ${k}: pair`, s.pair, 'dual objective', s.dualObjective, 'gap', s.gap)
 * }
 */
export function smoSteps(problem: SmoProblem): Algorithm<{ alpha?: Tensor }, SmoState> {
  const p = prepare(problem)
  const { n, y, K, C } = p
  return {
    name: 'smo',
    init: ({ alpha: start } = {}) => {
      const alpha = start ? Float64Array.from(values(start)) : new Float64Array(n)
      const G = new Float64Array(n).fill(-1)
      for (let s = 0; s < n; s++) {
        if (alpha[s] === 0) continue
        for (let t = 0; t < n; t++) G[t] += y[t] * y[s] * K[t * n + s] * alpha[s]
      }
      return stateAt(p, alpha, G, { t: 0 })
    },
    step: (state) => {
      const alpha = Float64Array.from(state.alpha.data as Float64Array)
      const G = Float64Array.from(state.grad.data as Float64Array)
      let [i, j] = state.violatingPair
      if (i < 0 || j < 0) return { ...state, t: state.t + 1, converged: true }
      if (p.selection === 'second-order') {
        // WSS 3: keep i, and choose j among violators to maximise the guaranteed decrease b²/a.
        const gi = -y[i] * G[i]
        let best = Infinity
        for (let t = 0; t < n; t++) {
          if (!inLow(alpha[t], y[t], C)) continue
          const bit = gi + y[t] * G[t]
          if (bit <= 0) continue
          let a = K[i * n + i] + K[t * n + t] - 2 * K[i * n + t]
          if (a <= 0) a = TAU
          const gain = -(bit * bit) / a
          if (gain <= best) {
            if (gain < best || t < j) j = t
            best = gain
          }
        }
      }
      const ai = alpha[i]
      const aj = alpha[j]
      const Qij = y[i] * y[j] * K[i * n + j]
      let step: number
      if (y[i] !== y[j]) {
        let quad = K[i * n + i] + K[j * n + j] + 2 * Qij
        if (quad <= 0) quad = TAU
        step = (-G[i] - G[j]) / quad
        const diff = ai - aj
        alpha[i] += step
        alpha[j] += step
        if (diff > 0) {
          if (alpha[j] < 0) {
            alpha[j] = 0
            alpha[i] = diff
          }
        } else if (alpha[i] < 0) {
          alpha[i] = 0
          alpha[j] = -diff
        }
        if (diff > 0) {
          if (alpha[i] > C) {
            alpha[i] = C
            alpha[j] = C - diff
          }
        } else if (alpha[j] > C) {
          alpha[j] = C
          alpha[i] = C + diff
        }
      } else {
        let quad = K[i * n + i] + K[j * n + j] - 2 * Qij
        if (quad <= 0) quad = TAU
        step = (G[i] - G[j]) / quad
        const sum = ai + aj
        alpha[i] -= step
        alpha[j] += step
        if (sum > C) {
          if (alpha[i] > C) {
            alpha[i] = C
            alpha[j] = sum - C
          }
        } else if (alpha[j] < 0) {
          alpha[j] = 0
          alpha[i] = sum
        }
        if (sum > C) {
          if (alpha[j] > C) {
            alpha[j] = C
            alpha[i] = sum - C
          }
        } else if (alpha[i] < 0) {
          alpha[i] = 0
          alpha[j] = sum
        }
      }
      const di = alpha[i] - ai
      const dj = alpha[j] - aj
      for (let t = 0; t < n; t++) G[t] += y[t] * (y[i] * K[t * n + i] * di + y[j] * K[t * n + j] * dj)
      // Unclipped, αⱼ moves by `step` in both cases; the box may have shortened the move.
      return stateAt(p, alpha, G, {
        t: state.t + 1,
        pair: [i, j],
        step,
        clipped: Math.abs(dj - step) > 1e-12 * Math.max(1, Math.abs(step)),
      })
    },
  }
}

/**
 * The decision function $f(\xvec) = \sum_t \alpha_t y_t k(\xvec_t, \xvec) + b$ of dual variables $\alphavec$ (an SMO
 * state's, say) on a problem's data.
 *
 * @param problem The problem the dual variables solve: its inputs $\xvec_t$, its labels $y_t = \pm 1$ (used as they
 *   are) and its kernel (default `rbf({ lengthscale: 1 })`); `C`, `tolerance` and `selection` are not read.
 * @param alpha The dual variables $\alphavec$, one per row of `problem.x`.
 * @param bias The bias $b$, such as an SMO state's `bias`.
 * @returns A function from query rows ($m \times d$) to their $m$ decision values.
 *
 * @example The decision function of a converged SMO run, at the data and at two new points
 * const x = tensor([[0], [0.5], [1], [1.5], [2.5], [3], [3.5], [4]])
 * const problem = { x, y: tensor([-1, -1, -1, -1, 1, 1, 1, 1]) }
 * const s = run(smoSteps({ ...problem, C: 10 }), {}, 100)
 * const f = dualDecision(problem, s.alpha, s.bias)
 * print('f at the data:', f(problem.x))
 * print('f at 1.9 and 2.1:', f(tensor([[1.9], [2.1]])))
 */
export function dualDecision(problem: SmoProblem, alpha: Tensor, bias: number): (x: Tensor) => Tensor {
  const { n, d } = matrix(problem.x, 'dualDecision')
  const y = values(problem.y)
  const a = values(alpha)
  const kernel = problem.kernel ?? rbf({ lengthscale: 1 })
  return (q: Tensor) => {
    const { n: m } = inputs(q, d, 'dualDecision')
    const out = new Float64Array(m).fill(bias)
    const K = values(gram(kernel, problem.x, q))
    for (let t = 0; t < n; t++) if (a[t] !== 0) for (let i = 0; i < m; i++) out[i] += a[t] * y[t] * K[t * m + i]
    return fromData(out, [m])
  }
}

/** A fitted kernel SVM for labels 0/1. */
export interface SupportVectorMachineModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<SmoState> {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'support-vector-machine'
  /** The kernel it was fitted with. */
  readonly kernel: Kernel
  /** The box constraint $C$. */
  readonly C: number
  /** Dual variables $\alphavec$, one for every training row. */
  readonly alpha: Tensor
  /** Indices of the support vectors ($\alpha_t > 0$), int32. */
  readonly supportVectors: Tensor
  /** Indices of the free support vectors ($0 < \alpha_t < C$), which lie on the margin. */
  readonly marginVectors: Tensor
  /** $\alpha_t y_t$ for each support vector, in the order of `supportVectors` (scikit-learn's `dual_coef_`). */
  readonly dualCoefficients: Tensor
  /** The bias $b$ of the decision function. */
  readonly bias: number
  /** The weight vector $\sum_t \alpha_t y_t \xvec_t$ ($d$ values), for the linear kernel only (null otherwise). */
  readonly weights: Tensor | null
  /** SMO pair updates taken. */
  readonly steps: number
  /** Whether SMO reached the tolerance within `maxSteps`. */
  readonly converged: boolean
  /** The maximal KKT violation at the end (the stopping measure). */
  readonly gap: number
  /** With `probability`: Platt's sigmoid fitted to the training decision values (null otherwise). */
  readonly platt: PlattScaling | null
  /** With `probability`: the Bernoulli law of $y$ with $P(y = 1 \mid \xvec)$ from Platt's sigmoid of $f(\xvec)$. */
  readonly predictive?: (x: Tensor) => AnyUnivariate
}

/**
 * The soft-margin kernel SVM for labels 0/1 (mapped to $-1$/$+1$), solved by `smoSteps` from $\alphavec = \zeros$.
 * `score` and `forward` give the decision function $f(\xvec) = \sum_t \alpha_t y_t k(\xvec_t, \xvec) + b$ for each
 * query row; `decide` is 1 where $f(\xvec) > 0$. The run is kept in `training`. With `probability`, Platt scaling
 * (`plattScaling`) is fitted to the decision values of the training rows and `predictive` gives
 * $P(y = 1 \mid \xvec)$; LIBSVM fits it on cross-validated decision values instead, which are less optimistic than
 * training ones. As scikit-learn's `SVC`. Throws `DomainError` for $C \le 0$ (at once) and, at `fit`, for labels with
 * more than two classes.
 *
 * @param params The hyperparameters.
 * @param params.C The box constraint $C$: larger values penalise margin violations more (default 1).
 * @param params.kernel The kernel $k$ (default `rbf({ lengthscale: 1 })`). With the linear kernel the model also
 *   reports `weights`.
 * @param params.tolerance SMO stops when the maximal KKT violation is at most this (default 1e-3).
 * @param params.maxSteps The most SMO pair updates (default 100000).
 * @param params.selection The working-pair rule of `smoSteps` (default `second-order`).
 * @param params.probability Fit Platt scaling for `predictive` (default false).
 * @returns The estimator; its `fit` takes inputs `x` ($n \times d$) and labels `y` (0/1).
 *
 * @example The support vectors of two separable classes on a line
 * // The pair either side of the gap holds the margin; with the short RBF lengthscale, so do the two ends.
 * const x = tensor([[0], [0.5], [1], [1.5], [2.5], [3], [3.5], [4]])
 * const y = tensor([0, 0, 0, 0, 1, 1, 1, 1])
 * const model = supportVectorMachine({ C: 10 }).fit({ x, y })
 * print('support vectors:', model.supportVectors)
 * print('their alpha_t y_t:', model.dualCoefficients)
 * print('f at the data:', model.score(x))
 * print('decisions at 1.9 and 2.1:', model.decide(tensor([[1.9], [2.1]])))
 */
export function supportVectorMachine(
  params: {
    C?: number
    kernel?: Kernel
    tolerance?: number
    maxSteps?: number
    selection?: SmoProblem['selection']
    probability?: boolean
  } = {},
): Estimator<Supervised<Tensor, Tensor>, SupportVectorMachineModel> {
  const {
    C = 1,
    kernel = rbf({ lengthscale: 1 }),
    tolerance = 1e-3,
    maxSteps = 100000,
    selection,
    probability = false,
  } = params
  if (!(C > 0)) throw new DomainError('supportVectorMachine', 'supportVectorMachine: C must be positive')
  return {
    name: 'support-vector-machine',
    params: { C, kernel, tolerance, maxSteps, selection, probability },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'supportVectorMachine')
      const { y: labels, k } = classLabels(y, n, 'supportVectorMachine')
      if (k > 2)
        throw new DomainError(
          'supportVectorMachine',
          'supportVectorMachine: binary labels 0/1 only; use a multiclass reduction',
        )
      const signs = Float64Array.from(labels, (c) => (c === 1 ? 1 : -1))
      const alg = smoSteps({ x, y: fromData(signs, [n]), C, kernel, tolerance, selection })
      const training: Trace<SmoState> = trace(alg, {}, maxSteps, {
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          dualObjective: (s) => s.dualObjective,
          gap: (s) => s.gap,
          ...(options.trace?.record as Record<string, (s: SmoState, t: number) => number> | undefined),
        },
      })
      const final = training.final
      const alpha = final.alpha.data as Float64Array
      const sv: number[] = []
      const free: number[] = []
      for (let t = 0; t < n; t++) {
        if (alpha[t] > 0) sv.push(t)
        if (alpha[t] > 0 && alpha[t] < C) free.push(t)
      }
      const coef = Float64Array.from(sv, (t) => alpha[t] * signs[t])
      const svRows = new Float64Array(sv.length * d)
      sv.forEach((t, r) => svRows.set(v.subarray(t * d, (t + 1) * d), r * d))
      const svX = fromData(svRows, [sv.length, d])
      let weights: Tensor | null = null
      if (kernel.name === 'linear') {
        const w = new Float64Array(d)
        sv.forEach((_, r) => {
          for (let j = 0; j < d; j++) w[j] += coef[r] * svRows[r * d + j]
        })
        weights = fromData(w, [d])
      }
      const score = (q: Tensor) => {
        const { n: m } = inputs(q, d, 'supportVectorMachine')
        const out = new Float64Array(m).fill(final.bias)
        if (sv.length === 0) return fromData(out, [m])
        const Kq = values(gram(kernel, svX, q))
        for (let r = 0; r < sv.length; r++) for (let i = 0; i < m; i++) out[i] += coef[r] * Kq[r * m + i]
        return fromData(out, [m])
      }
      const platt = probability ? plattScaling(score(x), fromData(signs, [n])) : null
      return {
        kind: 'model',
        name: 'support-vector-machine',
        platt,
        ...(platt ? { predictive: (q: Tensor) => bernoulliPredictive(platt.probability(score(q))) } : {}),
        kernel,
        C,
        alpha: final.alpha,
        supportVectors: fromData(Int32Array.from(sv), [sv.length]),
        marginVectors: fromData(Int32Array.from(free), [free.length]),
        dualCoefficients: fromData(coef, [sv.length]),
        bias: final.bias,
        weights,
        steps: final.t,
        converged: final.converged,
        gap: final.gap,
        training,
        forward: score,
        score,
        decide: (q: Tensor) =>
          fromData(
            Int32Array.from(values(score(q)), (s) => (s > 0 ? 1 : 0)),
            [q.shape[0]],
          ),
      }
    },
  }
}

// ── Linear SVM ───────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The linear SVM problem: inputs, labels $\pm 1$ and $C$; the bias is learnt as the weight of a constant feature.
 * Below, $\tilde\xvec_i$ is row $i$ with that feature appended and $\tilde\wvec$ the weights with the bias appended.
 */
export interface LinearSvmProblem {
  /** Inputs $n \times d$. */
  x: Tensor
  /** Labels $\pm 1$, $n$ of them. */
  y: Tensor
  /** The box constraint $C$ (default 1). */
  C?: number
  /**
   * Learn a bias as the weight of an appended constant feature, regularised with $\wvec$ (LIBLINEAR's convention;
   * default true).
   */
  intercept?: boolean
  /** Dual coordinate descent: visit the examples in a fresh random order each epoch (default true, as LIBLINEAR). */
  shuffle?: boolean
}

/** One state of a linear SVM solver. */
export interface LinearSvmState extends Status {
  /** Weights $\wvec$, $d$ values. */
  weights: Tensor
  /** The bias $b$ (0 without an intercept). */
  bias: number
  /** Dual variables $\alphavec$, $n$ of them (dual coordinate descent only; zeros for Pegasos). */
  alpha: Tensor
  /**
   * The primal objective
   * $\frac{1}{2}\lVert \wvec \rVert^2 + \frac{1}{2}b^2 + C \sum_i \max(0, 1 - y_i \tilde\wvec^\top\tilde\xvec_i)$
   * (the $b^2$ term only with an intercept).
   */
  primalObjective: number
  /**
   * Dual coordinate descent: the dual objective
   * $\ones^\top\alphavec - \frac{1}{2}\lVert \sum_i \alpha_i y_i \tilde\xvec_i \rVert^2$ (NaN for Pegasos).
   */
  dualObjective: number
  /**
   * Dual coordinate descent: the spread of projected gradients in the last epoch (the stopping measure; $\infty$ at
   * the start, NaN for Pegasos).
   */
  violation: number
  /** Epochs (dual coordinate descent) or single-example steps (Pegasos) so far. */
  t: number
  /** Dual coordinate descent: whether `violation` is at most the tolerance. Always false for Pegasos. */
  converged: boolean
}

/**
 * The data of a linear SVM run: the rows with the constant feature appended when there is an intercept, and the
 * defaults filled in. Throws `ShapeError` when the labels do not match the rows.
 *
 * @param problem The problem.
 * @param where The caller's name, for error messages.
 * @returns $n$, $d$, the width $D$ ($d + 1$ with an intercept, else $d$), the rows `X` (row-major $n \times D$), the
 *   labels `y` (used as given, so $\pm 1$), `C` and `intercept`.
 */
function linearData(problem: LinearSvmProblem, where: string) {
  const { n, d, v } = matrix(problem.x, where)
  const y = values(problem.y)
  if (y.length !== n) throw new ShapeError(where, `${where}: ${n} rows but ${y.length} labels`)
  const intercept = problem.intercept ?? true
  const D = intercept ? d + 1 : d
  const X = new Float64Array(n * D)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) X[i * D + j] = v[i * d + j]
    if (intercept) X[i * D + d] = 1
  }
  return { n, d, D, X, y, C: problem.C ?? 1, intercept }
}

/**
 * The primal objective
 * $\frac{1}{2}\lVert \tilde\wvec \rVert^2 + C \sum_i \max(0, 1 - y_i \tilde\wvec^\top\tilde\xvec_i)$.
 *
 * @param data The run's data, as `linearData` returns it.
 * @param w The extended weights $\tilde\wvec$, $D$ values (the bias last with an intercept).
 * @returns The objective.
 */
function primal(data: ReturnType<typeof linearData>, w: Float64Array): number {
  let s = 0
  for (const u of w) s += 0.5 * u * u
  for (let i = 0; i < data.n; i++) {
    let f = 0
    for (let j = 0; j < data.D; j++) f += w[j] * data.X[i * data.D + j]
    s += data.C * Math.max(0, 1 - data.y[i] * f)
  }
  return s
}

/**
 * A linear SVM state from the extended weights and the dual variables, with the primal objective and
 * $\ones^\top\alphavec - \frac{1}{2}\lVert \tilde\wvec \rVert^2$ as the dual objective (the dual's value when
 * $\tilde\wvec = \sum_i \alpha_i y_i \tilde\xvec_i$).
 *
 * @param data The run's data, as `linearData` returns it.
 * @param w The extended weights $\tilde\wvec$, $D$ values (the bias last with an intercept); not kept.
 * @param alpha The dual variables, $n$ values; kept by the state, so not to be written afterwards.
 * @param rest The step count `t`, and any fields that override the defaults (`violation` $\infty$, `converged`
 *   false, or `dualObjective`).
 * @returns The state.
 */
function linearState(
  data: ReturnType<typeof linearData>,
  w: Float64Array,
  alpha: Float64Array,
  rest: Partial<LinearSvmState> & { t: number },
): LinearSvmState {
  let dual = 0
  for (const a of alpha) dual += a
  for (const u of w) dual -= 0.5 * u * u
  return {
    weights: fromData(w.slice(0, data.d), [data.d]),
    bias: data.intercept ? w[data.d] : 0,
    alpha: fromData(alpha, [data.n]),
    primalObjective: primal(data, w),
    dualObjective: dual,
    violation: Infinity,
    converged: false,
    ...rest,
  }
}

/**
 * Dual coordinate descent for the linear hinge-loss SVM (Hsieh et al., 2008, Algorithm 1): each step is one epoch
 * over the examples (in a random order drawn from the step's stream with `shuffle`, the default; else row order),
 * minimising the dual in one $\alpha_i$ at a time with $\tilde\wvec = \sum_i \alpha_i y_i \tilde\xvec_i$ kept up to
 * date. Converged when the projected gradients of an epoch span at most `tolerance`. No start:
 * $\alphavec = \zeros$. Throws `ShapeError` when the labels do not match the rows.
 *
 * @param problem The inputs, the $\pm 1$ labels, $C$, `intercept` and `shuffle`, and `tolerance`, the spread of
 *   projected gradients at which it has converged (default 1e-6).
 * @returns The algorithm, one epoch per step; its state's `t` counts epochs.
 *
 * @example Epochs to convergence on two separable clusters
 * const x = tensor([[0, 0], [1, 0], [0, 1], [3, 3], [4, 3], [3, 4]])
 * const y = tensor([-1, -1, -1, 1, 1, 1])
 * const s = run(dualCoordinateSteps({ x, y, C: 10 }), undefined, 200, { stream: stream(1) })
 * print('w =', s.weights, 'b =', s.bias)
 * print('alpha =', s.alpha)
 * print('primal', s.primalObjective, '= dual', s.dualObjective, 'after', s.t, 'epochs')
 */
export function dualCoordinateSteps(
  problem: LinearSvmProblem & { tolerance?: number },
): Algorithm<void, LinearSvmState> {
  const data = linearData(problem, 'dualCoordinateSteps')
  const { n, D, X, y, C } = data
  const tol = problem.tolerance ?? 1e-6
  const shuffle = problem.shuffle ?? true
  const Qd = new Float64Array(n)
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) Qd[i] += X[i * D + j] ** 2
  return {
    name: 'dual-coordinate-descent',
    init: () => linearState(data, new Float64Array(D), new Float64Array(n), { t: 0 }),
    step: (state, ctx) => {
      const alpha = Float64Array.from(state.alpha.data as Float64Array)
      const w = new Float64Array(D)
      for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) w[j] += alpha[i] * y[i] * X[i * D + j]
      const order = shuffle ? Array.from(permutation(ctx.stream, n).data) : Array.from({ length: n }, (_, i) => i)
      let lo = Infinity
      let hi = -Infinity
      for (const i of order) {
        let g = -1
        for (let j = 0; j < D; j++) g += y[i] * w[j] * X[i * D + j]
        // Projected gradient: zero where the bound blocks the descent direction.
        const pg = alpha[i] <= 0 ? Math.min(g, 0) : alpha[i] >= C ? Math.max(g, 0) : g
        lo = Math.min(lo, pg)
        hi = Math.max(hi, pg)
        if (pg !== 0 && Qd[i] > 0) {
          const old = alpha[i]
          alpha[i] = Math.min(Math.max(old - g / Qd[i], 0), C)
          const delta = (alpha[i] - old) * y[i]
          for (let j = 0; j < D; j++) w[j] += delta * X[i * D + j]
        }
      }
      const violation = hi - lo
      return linearState(data, w, alpha, {
        t: state.t + 1,
        violation,
        converged: violation <= tol,
      })
    },
  }
}

/**
 * Pegasos (Shalev-Shwartz et al., 2007): stochastic subgradient descent on
 * $\frac{\lambda}{2}\lVert \tilde\wvec \rVert^2 + \frac{1}{n} \sum_i \max(0, 1 - y_i \tilde\wvec^\top\tilde\xvec_i)$
 * with $\lambda = 1/(nC)$, the same minimiser as the $C$-form primal. Step $t$ draws one example $i$ from the step's
 * stream and sets $\tilde\wvec \leftarrow (1 - \eta\lambda) \tilde\wvec + \eta y_i \tilde\xvec_i \indicator[m_i < 1]$
 * with $\eta = 1/(\lambda t)$ and the margin $m_i = y_i \tilde\wvec^\top\tilde\xvec_i$ taken before the update. It
 * has no projection step and never declares convergence. No start: $\tilde\wvec = \zeros$. Throws `ShapeError` when
 * the labels do not match the rows.
 *
 * @param problem The inputs, the $\pm 1$ labels, $C$ and `intercept` (`shuffle` is not read).
 * @returns The algorithm, one example per step.
 *
 * @example Two thousand steps approach the dual coordinate descent solution
 * const x = tensor([[0, 0], [1, 0], [0, 1], [3, 3], [4, 3], [3, 4]])
 * const y = tensor([-1, -1, -1, 1, 1, 1])
 * const s = run(pegasosSteps({ x, y, C: 10 }), undefined, 2000, { stream: stream(1) })
 * print('Pegasos:', 'w =', s.weights, 'b =', s.bias, 'primal =', s.primalObjective)
 * const exact = run(dualCoordinateSteps({ x, y, C: 10 }), undefined, 200, { stream: stream(1) })
 * print('exact:  ', 'w =', exact.weights, 'b =', exact.bias, 'primal =', exact.primalObjective)
 */
export function pegasosSteps(problem: LinearSvmProblem): Algorithm<void, LinearSvmState> {
  const data = linearData(problem, 'pegasosSteps')
  const { n, D, X, y, C } = data
  const lambda = 1 / (n * C)
  return {
    name: 'pegasos',
    init: () =>
      linearState(data, new Float64Array(D), new Float64Array(n), {
        t: 0,
        dualObjective: NaN,
      }),
    step: (state, ctx) => {
      const t = state.t + 1
      const i = integers(ctx.stream, n)
      const w = new Float64Array(D)
      w.set(state.weights.data as Float64Array)
      if (data.intercept) w[data.d] = state.bias
      let f = 0
      for (let j = 0; j < D; j++) f += w[j] * X[i * D + j]
      const eta = 1 / (lambda * t)
      for (let j = 0; j < D; j++) w[j] *= 1 - eta * lambda
      if (y[i] * f < 1) for (let j = 0; j < D; j++) w[j] += eta * y[i] * X[i * D + j]
      return linearState(data, w, new Float64Array(n), {
        t,
        dualObjective: NaN,
        violation: NaN,
      })
    },
  }
}

/** A fitted linear SVM for labels 0/1. */
export interface LinearSvmModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<LinearSvmState> {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'linear-svm'
  /** The weights $\wvec$, $d$ values. */
  readonly weights: Tensor
  /** The bias $b$ (0 without an intercept). */
  readonly bias: number
  /** The solver that fitted it. */
  readonly method: 'dual-coordinate-descent' | 'pegasos'
  /** Whether dual coordinate descent reached the tolerance (always false for Pegasos). */
  readonly converged: boolean
}

/**
 * The linear hinge-loss SVM for labels 0/1 by dual coordinate descent (default; exact at convergence, the same
 * solution as scikit-learn's `LinearSVC(loss='hinge')`) or Pegasos. `score` is $\wvec^\top\xvec + b$ for each query
 * row and `decide` is 1 where it is positive. `fit` takes the stream for the random order or draws in its options.
 * Throws `DomainError` at `fit` for labels with more than two classes.
 *
 * @param params The hyperparameters.
 * @param params.C The box constraint $C$ (default 1).
 * @param params.method The solver: `dualCoordinateSteps` or `pegasosSteps` (default dual coordinate descent).
 * @param params.intercept Learn a bias as the weight of a constant feature, regularised with $\wvec$ (default true).
 * @param params.tolerance The stopping spread of projected gradients for dual coordinate descent (default 1e-6;
 *   Pegasos ignores it).
 * @param params.maxSteps The most steps: epochs for dual coordinate descent (default 1000), single-example steps for
 *   Pegasos (default 5000).
 * @returns The estimator; its `fit` takes inputs `x` ($n \times d$) and labels `y` (0/1).
 *
 * @example A separating line between two clusters
 * const x = tensor([[0, 0], [1, 0], [0, 1], [3, 3], [4, 3], [3, 4]])
 * const y = tensor([0, 0, 0, 1, 1, 1])
 * const model = linearSvm({ C: 10 }).fit({ x, y }, { stream: stream(1) })
 * print('w =', model.weights, 'b =', model.bias, 'converged:', model.converged)
 * print('scores:', model.score(x))
 * print('decisions at (1, 1) and (2.5, 2.5):', model.decide(tensor([[1, 1], [2.5, 2.5]])))
 */
export function linearSvm(
  params: {
    C?: number
    method?: 'dual-coordinate-descent' | 'pegasos'
    intercept?: boolean
    tolerance?: number
    maxSteps?: number
  } = {},
): Estimator<Supervised<Tensor, Tensor>, LinearSvmModel> {
  const { C = 1, method = 'dual-coordinate-descent', intercept = true, tolerance = 1e-6 } = params
  const maxSteps = params.maxSteps ?? (method === 'pegasos' ? 5000 : 1000)
  return {
    name: 'linear-svm',
    params: { C, method, intercept, tolerance, maxSteps },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'linearSvm')
      const { y: labels, k } = classLabels(y, n, 'linearSvm')
      if (k > 2) throw new DomainError('linearSvm', 'linearSvm: binary labels 0/1 only; use a multiclass reduction')
      const signs = fromData(
        Float64Array.from(labels, (c) => (c === 1 ? 1 : -1)),
        [n],
      )
      const problem = { x, y: signs, C, intercept, tolerance }
      const alg = method === 'pegasos' ? pegasosSteps(problem) : dualCoordinateSteps(problem)
      const training = trace(alg, undefined, maxSteps, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          primalObjective: (s) => s.primalObjective,
          ...(options.trace?.record as Record<string, (s: LinearSvmState, t: number) => number> | undefined),
        },
      })
      const final = training.final
      const w = final.weights.data as Float64Array
      const score = (q: Tensor) => {
        const { n: m, v } = inputs(q, d, 'linearSvm')
        const out = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let s = final.bias
          for (let j = 0; j < d; j++) s += w[j] * v[i * d + j]
          out[i] = s
        }
        return fromData(out, [m])
      }
      return {
        kind: 'model',
        name: 'linear-svm',
        weights: final.weights,
        bias: final.bias,
        method,
        converged: final.converged,
        training,
        forward: score,
        score,
        decide: (q: Tensor) =>
          fromData(
            Int32Array.from(values(score(q)), (s) => (s > 0 ? 1 : 0)),
            [q.shape[0]],
          ),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'supportVectorMachine',
    module: 'learning/kernel-methods',
    name: 'Support vector machine',
    summary: 'The soft-margin kernel SVM solved by SMO (kernel default: RBF with lengthscale 1).',
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({
      C: real(1e-3, 1e3, { default: 1, scale: 'log' }),
      tolerance: real(1e-8, 1e-1, { default: 1e-3, scale: 'log' }),
      maxSteps: int(1, 100000, { default: 100000 }),
      probability: bool(),
    }),
    notes: ['support-vector-machine', 'kernel-support-vector-machine', 'solving-support-vector-machines'],
    cite: ['cortes1995', 'platt1998'],
  },
  supportVectorMachine,
)

defineModel(
  {
    key: 'linearSvm',
    module: 'learning/kernel-methods',
    name: 'Linear SVM',
    summary: 'The soft-margin linear SVM by dual coordinate descent or Pegasos.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({
      C: real(1e-3, 1e3, { default: 1, scale: 'log' }),
      method: oneOf(['dual-coordinate-descent', 'pegasos']),
      intercept: bool({ default: true }),
      tolerance: real(1e-10, 1e-2, { default: 1e-6, scale: 'log' }),
    }),
    notes: ['support-vector-machine', 'solving-support-vector-machines'],
    cite: ['cortes1995'],
  },
  linearSvm,
)
