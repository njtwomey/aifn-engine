/**
 * Fitters of one GAM: each a traceable `Algorithm` that minimises a problem's penalised objective
 * $J(\betavec) = (D(\betavec) + \betavec^\top\Smat_\lambda\betavec)/(2n)$ at its fixed smoothing parameters
 * (`gamProblem`), from the common start $\betavec_0$, so their traces compare step by step and their limits agree.
 *
 * - `gamPirls`: penalised IRLS (Wood, 2017, "Generalized Additive Models", 2nd ed., §6.1.1): each step solves
 *   $(\Xmat^\top\Wmat\Xmat + \Smat_\lambda)\betavec = \Xmat^\top\Wmat\zvec$ for the working response
 *   $\zvec$ and weights $\Wmat$ at the current $\etavec$, with step halving. Newton's method on $J$ for a canonical
 *   link, Fisher scoring otherwise.
 * - `gamBackfitting`: backfitting with local scoring (Hastie and Tibshirani, 1990, §4.4 and §6.5): one sweep per step,
 *   each term's penalised smoother refitted to its partial working residual in turn.
 * - `gamGradientDescent`, `gamAdam`, `gamLbfgs`: compute's gradient descent, Adam (Kingma and Ba, 2015) and L-BFGS
 *   (Nocedal and Wright, 2006, §7.2) on $J$, with $\nabla J$ by reverse-mode autodiff through the family's deviance.
 * - `gamSgd`: stochastic gradient descent (with optional momentum) on minibatch estimates of $J$, drawn without
 *   replacement within each epoch (Robbins and Monro, 1951; Bottou, 2010).
 *
 * Every state carries $\betavec$, $J$, the deviance, the penalised deviance and $\lVert \nabla J \rVert$ (the
 * closed form on all the data), so one trace view serves every fitter; `gamModel(problem, beta)` gives the EDF, bands
 * and partial effects at any step. A fitter stops when its state is `converged` (or, for the first-order ones,
 * stalled).
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Status } from 'aifn-compute/foundation/contracts'
import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import { permutation, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { adam, gradientDescent, sgdRule, type FirstOrderState, type StepSize } from 'aifn-compute/optim/first-order'
import { lbfgs, type LbfgsState } from 'aifn-compute/optim/second-order'
import { backfitting, type BackfitState } from '../backfitting'
import { irls, type IrlsState } from '../irls'
import { gamProblem, type GamData, type GamProblem, type GamSpec } from './problem'
import { DomainError } from 'aifn-compute/foundation/errors'

type F64 = Float64Array
/**
 * A vector tensor holding a copy of some numbers.
 *
 * @param a The values; not modified.
 * @returns A float64 tensor of shape $[n]$ for $n$ values.
 */
const vec = (a: ArrayLike<number>) => fromData(Float64Array.from(a), [a.length])

/** The fitters `gamFitter` dispatches to. */
export type GamFitMethod = 'p-irls' | 'backfitting' | 'gradient-descent' | 'sgd' | 'adam' | 'lbfgs'

/** One state of any GAM fitter. */
export type GamFitState = Status & {
  /** Steps taken. */
  t: number
  /** $\betavec$ ($P$): the intercept, then each term's block. */
  coefficients: Tensor
  /** $J(\betavec) = (D + \betavec^\top\Smat_\lambda\betavec)/(2n)$. */
  objective: number
  /** The deviance $D(\betavec)$. */
  deviance: number
  /** $D + \betavec^\top\Smat_\lambda\betavec$. */
  penalisedDeviance: number
  /** $\lVert \nabla J(\betavec) \rVert$ on all the data. */
  gradNorm: number
  /** The step size of the last step (NaN for P-IRLS, backfitting and step 0). */
  stepSize: number
  /** P-IRLS: the working response $\zvec$ at this $\betavec$, which the next step regresses on; null otherwise. */
  working: Tensor | null
  /** P-IRLS: the working weights at this $\betavec$; null otherwise. */
  workingWeights: Tensor | null
  /** Rows of the minibatch that made the last step (SGD only; null otherwise). */
  batch: Tensor | null
  /** Passes over the data so far: steps for full-batch fitters, steps $\times$ batch size $/\, n$ for SGD. */
  epochs: number
  /** Whether the fitter's own convergence test passed (it then stops). */
  converged: boolean
  /** Whether the objective is no longer finite. */
  diverged: boolean
}

/** Options of the gradient-based fitters. */
export type GamGradientOptions = {
  /** Step size $\eta$ or a schedule (gradient descent and SGD: default `stepScale`$/L$; Adam: default 0.05). */
  stepSize?: StepSize
  /**
   * Gradient descent and SGD without `stepSize`: $\eta = \text{stepScale}/L$ for the curvature $L$ of $J$ at the
   * optimum, a step that is stable near it below 2 (default 1 for gradient descent, 0.5 for SGD).
   */
  stepScale?: number
  /** Converged when $\lVert \nabla J \rVert \le$ `tolerance` (default 1e-9). */
  tolerance?: number
}

/**
 * The state shared by every fitter at $\betavec$. A full-batch gradient method passes $J$ and
 * $\lVert \nabla J \rVert$ from its own evaluation (`known`), so the step costs no second pass over the data:
 * $D = 2nJ - \betavec^\top\Smat_\lambda\betavec$.
 *
 * @param problem The problem being fitted.
 * @param beta The coefficients $\betavec$ ($P$ values); copied into the state.
 * @param t The step number.
 * @param extra Fields that override the defaults (step size, working response, convergence, ...).
 * @param known $J$ and $\lVert \nabla J \rVert$ already computed at $\betavec$; ignored when $J$ is not finite, and
 *   then (or when left out) both are computed here.
 * @returns The state, `diverged` when $J$ is not finite unless `extra` says otherwise.
 */
function stateAt(
  problem: GamProblem,
  beta: ArrayLike<number>,
  t: number,
  extra: Partial<GamFitState> = {},
  known?: { objective: number; gradNorm: number },
): GamFitState {
  let objective: number
  let dev: number
  let pdev: number
  let gradNorm: number
  if (known && Number.isFinite(known.objective)) {
    const P = problem.design.P
    let pen = 0
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) pen += beta[a] * problem.penalty[a * P + b] * beta[b]
    objective = known.objective
    pdev = 2 * problem.design.n * objective
    dev = pdev - pen
    gradNorm = known.gradNorm
  } else {
    const e = problem.evaluate(beta)
    objective = e.objective
    dev = e.deviance
    pdev = e.penalisedDeviance
    gradNorm = Math.hypot(...problem.gradient(beta))
  }
  return {
    t,
    coefficients: vec(beta),
    objective,
    deviance: dev,
    penalisedDeviance: pdev,
    gradNorm,
    stepSize: NaN,
    working: null,
    workingWeights: null,
    batch: null,
    epochs: t,
    converged: false,
    diverged: !Number.isFinite(objective),
    ...extra,
  }
}

/**
 * $J$ and its gradient by reverse-mode autodiff, as an `ObjectiveFn`. An invalid mean, a non-finite value or a thrown
 * error gives $+\infty$ with a zero gradient, which line searches back away from.
 *
 * @param problem The problem whose `objective` is differentiated.
 * @returns A function of $\betavec$ ($P$) returning `value` and `grad`.
 */
function autodiffObjective(problem: GamProblem) {
  const vg = valueAndGrad((beta: Value) => problem.objective(beta))
  const P = problem.design.P
  return (x: Tensor) => {
    try {
      const { value, grad } = vg(x)
      const v = typeof value === 'number' ? value : toFlat(value as Tensor)[0]
      if (!Number.isFinite(v)) return { value: Infinity, grad: fromData(new Float64Array(P), [P]) }
      return { value: v, grad: grad as Tensor }
    } catch {
      return { value: Infinity, grad: fromData(new Float64Array(P), [P]) }
    }
  }
}

/**
 * Penalised IRLS from $\betavec_0$ (`aifn-methods/learning/glm`'s `irls` with the problem's penalty); each state
 * carries the working response and weights the next step regresses on. For a Gaussian identity-link problem one step
 * reaches the optimum.
 *
 * @param problem The problem, with its smoothing parameters chosen.
 * @param options `tolerance`, the convergence tolerance of `irls` (default 1e-12).
 * @returns The algorithm; its state's `inner` is the `irls` state.
 *
 * @example A logistic GAM converges in a few Newton steps
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [60, 1] })
 * const y = bernoulli(r, div(1, add(1, exp(mul(-2, sin(reshape(x, [60])))))))
 * const problem = gamProblem({ terms: [s(0)], family: 'binomial' }, { x, y })
 * for (const st of trace(gamPirls(problem), undefined, 10).steps) print('step', st.t, 'J =', st.objective)
 * print('optimum J =', problem.evaluate(problem.optimum.beta).objective)
 */
export function gamPirls(
  problem: GamProblem,
  options: { tolerance?: number } = {},
): Algorithm<void, GamFitState & { inner: IrlsState }> {
  const A = problem.design
  const inner = irls({
    design: fromData(A.X, [A.n, A.P]),
    y: problem.data.y,
    family: problem.family,
    link: problem.link,
    weights: problem.data.weights,
    offset: problem.data.offset,
    penalty: fromData(problem.penalty, [A.P, A.P]),
    tolerance: options.tolerance ?? 1e-12,
  })
  const wrap = (s: IrlsState) => ({
    ...stateAt(problem, toFlat(s.coefficients ?? vec(problem.start)), s.t, {
      working: s.working,
      workingWeights: s.workingWeights,
      converged: s.converged,
    }),
    diverged: s.diverged,
    inner: s,
  })
  return {
    name: 'gam-p-irls',
    init: (_start, stream) => wrap(inner.init({ coefficients: vec(problem.start) }, stream)),
    step: (s, ctx) => wrap(inner.step(s.inner, ctx)),
  }
}

/**
 * Backfitting with local scoring: each step refreshes the working response and weights, then refits every term's
 * penalised smoother to its partial residual in turn. Its fixed point is the P-IRLS optimum. Offsets are not supported:
 * a problem with one throws `DomainError`.
 *
 * @param problem The problem, with its smoothing parameters chosen; each term's smoother uses its diagonal block of
 *   the penalty.
 * @param options `tolerance`, the convergence tolerance of the backfitting sweeps (default 1e-10).
 * @returns The algorithm; its state's `inner` is the backfitting state.
 *
 * @example Two smooths, refitted in turn until they agree with P-IRLS
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 2] })
 * const y = add(tensor(toArray(x).map(([a, b]) => Math.sin(a) + 0.3 * b)), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0), s(1)] }, { x, y })
 * const alg = gamBackfitting(problem)
 * for (const k of [0, 1, 3, 10]) print('sweep', k, 'J =', run(alg, undefined, k).objective)
 * print('optimum J =', problem.evaluate(problem.optimum.beta).objective)
 */
export function gamBackfitting(
  problem: GamProblem,
  options: { tolerance?: number } = {},
): Algorithm<void, GamFitState & { inner: BackfitState }> {
  if (problem.data.offset) throw new DomainError('gamBackfitting', 'gamBackfitting: offsets are not supported')
  const A = problem.design
  const block = (M: F64, from: number, rows: number, cols = rows, rowStride = A.P) => {
    const out = new Float64Array(rows * cols)
    for (let i = 0; i < rows; i++) for (let c = 0; c < cols; c++) out[i * cols + c] = M[i * rowStride + from + c]
    return out
  }
  const designs = A.terms.map((t, j) => fromData(block(A.X, A.offsets[j], A.n, t.size), [A.n, t.size]))
  const penalties = A.terms.map((t, j) => {
    const o = A.offsets[j]
    const S = new Float64Array(t.size * t.size)
    for (let a = 0; a < t.size; a++)
      for (let b = 0; b < t.size; b++) S[a * t.size + b] = problem.penalty[(o + a) * A.P + o + b]
    return fromData(S, [t.size, t.size])
  })
  const inner = backfitting({
    designs,
    penalties,
    y: problem.data.y,
    weights: problem.data.weights,
    family: problem.family,
    link: problem.link,
    tolerance: options.tolerance ?? 1e-10,
  })
  const wrap = (s: BackfitState) => {
    const beta = new Float64Array(A.P)
    beta[0] = s.intercept
    s.coefficients.forEach((b, j) => beta.set(toFlat(b), A.offsets[j]))
    return { ...stateAt(problem, beta, s.t, { converged: s.converged }), inner: s }
  }
  return {
    name: 'gam-backfitting',
    init: (_start, stream) => wrap(inner.init(undefined, stream)),
    step: (s, ctx) => wrap(inner.step(s.inner, ctx)),
  }
}

/**
 * Gradient descent on $J$ with a fixed step (default $1/L$, $L$ the problem's curvature), a schedule, or a line
 * search, by compute's `gradientDescent`.
 *
 * @param problem The problem, with its smoothing parameters chosen.
 * @param options `stepSize`, `stepScale` and `tolerance` (see `GamGradientOptions`), and `lineSearch`, a line search
 *   that starts from the step size (none by default).
 * @returns The algorithm; its state's `inner` is the first-order state.
 *
 * @example Slow but steady: the fixed step 1/L, and a strong Wolfe line search
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * const fixed = gamGradientDescent(problem)
 * const wolfe = gamGradientDescent(problem, { lineSearch: 'strong-wolfe' })
 * for (const k of [10, 100]) {
 *   print('step', k, 'fixed: J =', run(fixed, undefined, k).objective)
 *   print('step', k, 'Wolfe: J =', run(wolfe, undefined, k).objective)
 * }
 * print('optimum J =', problem.evaluate(problem.optimum.beta).objective)
 */
export function gamGradientDescent(
  problem: GamProblem,
  options: GamGradientOptions & { lineSearch?: 'backtracking' | 'strong-wolfe' } = {},
): Algorithm<void, GamFitState & { inner: FirstOrderState }> {
  const { stepScale = 1, tolerance = 1e-9, lineSearch } = options
  const stepSize = options.stepSize ?? stepScale / problem.curvature
  const inner = gradientDescent(autodiffObjective(problem), { stepSize, tolerance, lineSearch })
  return wrapFirstOrder(problem, 'gam-gradient-descent', inner)
}

/**
 * Adam on $J$, full batch (default $\eta = 0.05$), by compute's `adam`.
 *
 * @param problem The problem, with its smoothing parameters chosen.
 * @param options `stepSize` (default 0.05) and `tolerance` (default 1e-9); `stepScale` is not read.
 * @returns The algorithm; its state's `inner` is the first-order state.
 *
 * @example Adam approaches the optimum
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * const alg = gamAdam(problem)
 * for (const k of [0, 10, 100, 500]) print('step', k, 'J =', run(alg, undefined, k).objective)
 * print('optimum J =', problem.evaluate(problem.optimum.beta).objective)
 */
export function gamAdam(
  problem: GamProblem,
  options: GamGradientOptions = {},
): Algorithm<void, GamFitState & { inner: FirstOrderState }> {
  const { stepSize = 0.05, tolerance = 1e-9 } = options
  return wrapFirstOrder(problem, 'gam-adam', adam(autodiffObjective(problem), { stepSize, tolerance }))
}

/**
 * A compute first-order algorithm on $J$ as a GAM fitter: started at $\betavec_0$, with its value and gradient norm
 * reused for the state, and stopped when it stalls.
 *
 * @param problem The problem being fitted.
 * @param name The fitter's name.
 * @param inner The first-order algorithm, built on `autodiffObjective(problem)`.
 * @returns The fitter; its state's `inner` is the first-order state.
 */
function wrapFirstOrder(
  problem: GamProblem,
  name: string,
  inner: Algorithm<{ x0: Tensor }, FirstOrderState>,
): Algorithm<void, GamFitState & { inner: FirstOrderState }> {
  const wrap = (s: FirstOrderState) => ({
    ...stateAt(
      problem,
      toFlat(s.x),
      s.t,
      { stepSize: s.stepSize, converged: s.converged },
      { objective: Number(s.value), gradNorm: s.gradNorm },
    ),
    diverged: !!s.diverged,
    stalled: s.stalled,
    inner: s,
  })
  return {
    name,
    init: (_start, stream) => wrap(inner.init({ x0: vec(problem.start) }, stream)),
    step: (s, ctx) => wrap(inner.step(s.inner, ctx)),
    done: (s) => s.stalled === true,
  }
}

/**
 * L-BFGS on $J$ with a strong Wolfe line search, by compute's `lbfgs`.
 *
 * @param problem The problem, with its smoothing parameters chosen.
 * @param options `memory`, the number of curvature pairs kept (default 10), and `tolerance` on
 *   $\lVert \nabla J \rVert$ (default 1e-9).
 * @returns The algorithm; its state's `inner` is the L-BFGS state.
 *
 * @example L-BFGS reaches the optimum in a few dozen steps
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * const alg = gamLbfgs(problem)
 * for (const k of [0, 5, 20]) print('step', k, 'J =', run(alg, undefined, k).objective)
 * print('optimum J =', problem.evaluate(problem.optimum.beta).objective)
 */
export function gamLbfgs(
  problem: GamProblem,
  options: { memory?: number; tolerance?: number } = {},
): Algorithm<void, GamFitState & { inner: LbfgsState }> {
  const { memory = 10, tolerance = 1e-9 } = options
  const inner = lbfgs(autodiffObjective(problem), { memory, tolerance })
  const wrap = (s: LbfgsState) => ({
    ...stateAt(
      problem,
      toFlat(s.x),
      s.t,
      { stepSize: s.stepSize, converged: s.converged },
      { objective: s.value, gradNorm: s.gradNorm },
    ),
    diverged: !!s.diverged,
    stalled: s.stalled,
    inner: s,
  })
  return {
    name: 'gam-lbfgs',
    init: (_start, stream) => wrap(inner.init({ x0: vec(problem.start) }, stream)),
    step: (s, ctx) => wrap(inner.step(s.inner, ctx)),
    done: (s) => s.stalled === true,
  }
}

/** Options of `gamSgd`. */
export type GamSgdOptions = GamGradientOptions & {
  /** Rows per minibatch (default 32), rounded and kept in $[1, n]$. */
  batchSize?: number
  /** Momentum $\mu \in [0, 1)$ (default 0). */
  momentum?: number
}

/** The SGD state's bookkeeping: the epoch's row order and the next position in it, and the rule's velocity. */
type SgdInner = { order: Int32Array; position: number; velocity: Tensor | null }

/**
 * Minibatch SGD on $J$: each step takes the next `batchSize` rows of a permutation drawn at the start of each epoch
 * (the rows left over when $n$ is not a multiple of the batch size wait for a later epoch), differentiates the
 * minibatch estimate of $J$ by autodiff and applies compute's `sgdRule` (default $\eta = 0.5/L$). It is converged
 * when the full-data $\lVert \nabla J \rVert$ is within `tolerance`. Random: run it with a stream.
 *
 * @param problem The problem, with its smoothing parameters chosen.
 * @param options Step size or scale, tolerance, batch size and momentum (see `GamSgdOptions`).
 * @returns The algorithm; its state's `inner` holds the epoch's row order, the position in it and the velocity.
 *
 * @example Minibatches of 10 rows: near the optimum after some epochs, then noisy around it
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * const alg = gamSgd(problem, { batchSize: 10 })
 * for (const k of [0, 10, 100, 400]) {
 *   const st = run(alg, undefined, k, { stream: stream(1) })
 *   print('epochs', st.epochs, 'J =', st.objective)
 * }
 * print('optimum J =', problem.evaluate(problem.optimum.beta).objective)
 */
export function gamSgd(
  problem: GamProblem,
  options: GamSgdOptions = {},
): Algorithm<void, GamFitState & { inner: SgdInner }> {
  const { stepScale = 0.5, momentum = 0, tolerance = 1e-9 } = options
  const stepSize = options.stepSize ?? stepScale / problem.curvature
  const n = problem.design.n
  const b = Math.max(1, Math.min(n, Math.round(options.batchSize ?? 32)))
  const rule = sgdRule({ stepSize, momentum })
  const vg = valueAndGrad((beta: Value, rows: Int32Array) => problem.objective(beta, rows))
  const shuffled = (stream: Parameters<Algorithm<void, GamFitState>['init']>[1]) =>
    Int32Array.from(toFlat(permutation(stream, n)))
  return {
    name: 'gam-sgd',
    init: (_start, stream) => {
      const s = stateAt(problem, problem.start, 0)
      return {
        ...s,
        epochs: 0,
        converged: s.gradNorm <= tolerance,
        inner: { order: shuffled(stream), position: 0, velocity: null },
      }
    },
    step: (s, ctx) => {
      let { order, position } = s.inner
      if (position + b > n) {
        order = shuffled(ctx.stream)
        position = 0
      }
      const rows = order.slice(position, position + b)
      const { grad } = vg(s.coefficients, rows)
      const slots: Record<string, Tensor> = s.inner.velocity ? { velocity: s.inner.velocity } : {}
      const out = rule.update(grad as Tensor, { t: s.t, slots })
      const update = toFlat(out.updates as Tensor)
      const beta = Float64Array.from(toFlat(s.coefficients), (v, i) => v + update[i])
      const next = stateAt(problem, beta, s.t + 1, {
        stepSize: typeof stepSize === 'function' ? Number(stepSize(s.t)) : Number(stepSize),
        batch: fromData(Int32Array.from(rows), [rows.length]),
        epochs: ((s.t + 1) * b) / n,
      })
      return {
        ...next,
        converged: next.gradNorm <= tolerance,
        inner: { order, position: position + b, velocity: (out.state.slots.velocity as Tensor | undefined) ?? null },
      }
    },
  }
}

/** Options of `gamFitter`: each fitter reads its own. */
export type GamFitterOptions = GamSgdOptions & {
  /** Gradient descent's line search (see `gamGradientDescent`). */
  lineSearch?: 'backtracking' | 'strong-wolfe'
  /** L-BFGS's memory (see `gamLbfgs`). */
  memory?: number
}

/**
 * The fitter for a method, with its options. An unknown method throws `DomainError`.
 *
 * @param problem The problem, with its smoothing parameters chosen.
 * @param method Which fitter: `'p-irls'`, `'backfitting'`, `'gradient-descent'`, `'sgd'`, `'adam'` or `'lbfgs'`.
 * @param options The options; each fitter reads its own.
 * @returns The fitter.
 *
 * @example Three fitters, 30 steps each, on one problem
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const problem = gamProblem({ terms: [s(0)] }, { x, y })
 * for (const method of ['p-irls', 'lbfgs', 'adam']) {
 *   print(method, 'J =', run(gamFitter(problem, method), undefined, 30).objective)
 * }
 */
export function gamFitter(
  problem: GamProblem,
  method: GamFitMethod,
  options: GamFitterOptions = {},
): Algorithm<void, GamFitState> {
  switch (method) {
    case 'p-irls':
      return gamPirls(problem, options)
    case 'backfitting':
      return gamBackfitting(problem, options)
    case 'gradient-descent':
      return gamGradientDescent(problem, options)
    case 'sgd':
      return gamSgd(problem, options)
    case 'adam':
      return gamAdam(problem, options)
    case 'lbfgs':
      return gamLbfgs(problem, options)
  }
  throw new DomainError('gamFitter', `gamFitter: unknown method "${method as string}"`)
}

// ── Runs as plain data ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What `gamProblem` chose ($\lambda$, active shape rows, the criterion), so the problem can be rebuilt without the
 * search.
 */
export type GamProblemChoices = {
  /** One $\lambda$ per penalty, in term order. */
  lambdas: number[]
  /** The active shape-constraint rows of each term. */
  active: number[][]
  /** The criterion, its value at the chosen $\lambda$ and how many fits the search took. */
  smoothing: GamProblem['smoothing']
}

/**
 * The choices of a GAM problem on data, as plain data (for a worker to send back): rebuild the problem without the
 * search with `gamProblem({ ...spec, ...choices }, data)`.
 *
 * @param spec The problem's specification.
 * @param data The data.
 * @returns The chosen $\lambda$, the active shape rows and the smoothing criterion.
 *
 * @example Choose once, rebuild without searching
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const spec = { terms: [s(0)] }
 * const choices = gamProblemChoices(spec, { x, y })
 * print('lambda =', choices.lambdas, 'after', choices.smoothing.evaluations, 'fits')
 * const again = gamProblem({ ...spec, ...choices }, { x, y })
 * print('rebuilt: lambda =', again.lambdas, 'after', again.smoothing.evaluations, 'fits')
 */
export function gamProblemChoices(spec: GamSpec, data: GamData): GamProblemChoices {
  const p = gamProblem(spec, data)
  return { lambdas: p.lambdas, active: p.active, smoothing: p.smoothing }
}

/**
 * One fitter's run in a `GamTrainingRun`: the `method` and its `options`, the most `steps` to take, and the `seed` of
 * its stream (default `'gam-training'`).
 */
export type GamRunRequest = { method: GamFitMethod; options?: GamFitterOptions; steps: number; seed?: string | number }

/** A GAM fitted step by step by one or more fitters, as plain data (for a worker or a log). */
export type GamTrainingRun = {
  /** The problem's choices and the facts every run is measured against. */
  choices: GamProblemChoices
  /** The curvature $L$ of $J$ at the optimum (see `GamProblem`). */
  curvature: number
  /** The deviance of the intercept-only model. */
  nullDeviance: number
  /** The P-IRLS optimum $\hat{\betavec}$ ($P$) and $J(\hat{\betavec})$. */
  optimum: { coefficients: Tensor; objective: number }
  /** Per request: every state from step 0 (each fitter's own bookkeeping dropped). */
  runs: { method: GamFitMethod; states: GamFitState[] }[]
}

/**
 * Build the problem once and run each requested fitter on it, keeping every state: the data a figure plays. With
 * `lambdas` (and `active`) in the spec the smoothing-parameter search is skipped.
 *
 * @param spec The problem's specification.
 * @param data The data.
 * @param requests The fitters to run, each with its options, step budget and seed.
 * @returns The problem's choices and reference optimum, and every state of each run (without the fitters' `inner`).
 *
 * @example P-IRLS and Adam on one problem
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const out = gamTrainingRun({ terms: [s(0)] }, { x, y }, [
 *   { method: 'p-irls', steps: 5 },
 *   { method: 'adam', steps: 50 },
 * ])
 * print('optimum J =', out.optimum.objective)
 * for (const run of out.runs) print(run.method, run.states.length, 'states, last J =', run.states.at(-1).objective)
 */
export function gamTrainingRun(spec: GamSpec, data: GamData, requests: readonly GamRunRequest[]): GamTrainingRun {
  const problem = gamProblem(spec, data)
  const runs = requests.map(({ method, options, steps, seed }) => {
    const tr = trace(gamFitter(problem, method, options), undefined, steps, { stream: stream(seed ?? 'gam-training') })
    const states = tr.steps.map((st) => {
      const { inner: _inner, ...plain } = st as GamFitState & { inner?: unknown }
      return plain as GamFitState
    })
    return { method, states }
  })
  return {
    choices: { lambdas: problem.lambdas, active: problem.active, smoothing: problem.smoothing },
    curvature: problem.curvature,
    nullDeviance: problem.nullDeviance,
    optimum: { coefficients: vec(problem.optimum.beta), objective: problem.evaluate(problem.optimum.beta).objective },
    runs,
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/generalised/gam')
/**
 * The state roles of a fitter's registry entry: `coefficients` is the iterate and `objective` the objective.
 *
 * @param stepSize Whether the state's `stepSize` is a role (the gradient-based fitters).
 * @param flags The status flags the fitter sets.
 * @returns The `state` of the registry entry.
 */
const roles = (stepSize: boolean, flags: AlgorithmInfo['state']['flags']) => ({
  iterate: 'coefficients',
  objective: 'objective',
  ...(stepSize ? { stepSize: 'stepSize' } : {}),
  flags,
})
const notes = ['generalised-additive-model']

algorithm(
  {
    key: 'gamPirls',
    name: 'GAM by penalised IRLS',
    summary: 'Penalised IRLS on a GAM problem: weighted ridge-type solves on the working response, with step halving.',
    problem: 'objective',
    state: roles(false, ['converged', 'diverged']),
    notes: ['generalised-additive-model', 'iteratively-reweighted-least-squares'],
    cite: ['wood2017'],
  },
  gamPirls,
)
algorithm(
  {
    key: 'gamBackfitting',
    name: 'GAM by backfitting',
    summary: 'Backfitting with local scoring: one sweep of penalised smoothers over the terms per step.',
    problem: 'objective',
    state: roles(false, ['converged', 'diverged']),
    notes,
    cite: ['hastie1990'],
  },
  gamBackfitting,
)
algorithm(
  {
    key: 'gamGradientDescent',
    name: 'GAM by gradient descent',
    summary: 'Gradient descent on the penalised deviance per observation, with the gradient by autodiff.',
    problem: 'objective',
    state: roles(true, ['converged', 'diverged', 'stalled']),
    notes: ['generalised-additive-model', 'gradient-descent'],
  },
  gamGradientDescent,
)
algorithm(
  {
    key: 'gamSgd',
    name: 'GAM by minibatch SGD',
    summary: 'Stochastic gradient descent on minibatch estimates of the penalised deviance, by autodiff.',
    problem: 'objective',
    state: roles(true, ['converged', 'diverged']),
    random: true,
    notes,
  },
  gamSgd,
)
algorithm(
  {
    key: 'gamAdam',
    name: 'GAM by Adam',
    summary: 'Adam on the penalised deviance per observation, full batch, with the gradient by autodiff.',
    problem: 'objective',
    state: roles(true, ['converged', 'diverged', 'stalled']),
    notes,
  },
  gamAdam,
)
algorithm(
  {
    key: 'gamLbfgs',
    name: 'GAM by L-BFGS',
    summary: 'L-BFGS with a strong Wolfe line search on the penalised deviance per observation, by autodiff.',
    problem: 'objective',
    state: roles(true, ['converged', 'diverged', 'stalled']),
    notes,
  },
  gamLbfgs,
)

/** The GAM fitters, keyed by factory name. */
export const gamFitters: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', {
    gamPirls,
    gamBackfitting,
    gamGradientDescent,
    gamSgd,
    gamAdam,
    gamLbfgs,
  }) as Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>>
