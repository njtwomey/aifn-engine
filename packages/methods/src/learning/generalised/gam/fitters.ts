/**
 * Fitters of one GAM: each a traceable `Algorithm` that minimises a problem's penalised objective
 * J(β) = (D(β) + βᵀS_λβ)/(2n) at its fixed smoothing parameters (`gamProblem`), from the common start β₀, so their
 * traces compare step by step and their limits agree.
 *
 * - `gamPirls`: penalised IRLS (Wood, 2017, "Generalized Additive Models", 2nd ed., §6.1.1): each step solves
 *   (XᵀWX + S_λ)β = XᵀWz for the working response z and weights W at the current η, with step halving. Newton's method
 *   on J for a canonical link, Fisher scoring otherwise.
 * - `gamBackfitting`: backfitting with local scoring (Hastie and Tibshirani, 1990, §4.4 and §6.5): one sweep per step,
 *   each term's penalised smoother refitted to its partial working residual in turn.
 * - `gamGradientDescent`, `gamAdam`, `gamLbfgs`: compute's gradient descent, Adam (Kingma and Ba, 2015) and L-BFGS
 *   (Nocedal and Wright, 2006, §7.2) on J, with ∇J by reverse-mode autodiff through the family's deviance.
 * - `gamSgd`: stochastic gradient descent (with optional momentum) on minibatch estimates of J, drawn without
 *   replacement within each epoch (Robbins and Monro, 1951; Bottou, 2010).
 *
 * Every state carries β, J, the deviance, the penalised deviance and ‖∇J‖ (the closed form on all the data), so one
 * trace view serves every fitter; `gamModel(problem, β)` gives the EDF, bands and partial effects at any step.
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
const vec = (a: ArrayLike<number>) => fromData(Float64Array.from(a), [a.length])

/** The fitters `gamFitter` dispatches to. */
export type GamFitMethod = 'p-irls' | 'backfitting' | 'gradient-descent' | 'sgd' | 'adam' | 'lbfgs'

/** One state of any GAM fitter. */
export type GamFitState = Status & {
  t: number
  /** β [P]: the intercept, then each term's block. */
  coefficients: Tensor
  /** J(β) = (D + βᵀS_λβ)/(2n). */
  objective: number
  deviance: number
  /** D + βᵀS_λβ. */
  penalisedDeviance: number
  /** ‖∇J(β)‖ on all the data. */
  gradNorm: number
  /** The step size of the last step (NaN for P-IRLS, backfitting and step 0). */
  stepSize: number
  /** P-IRLS: the working response z and weights W at this β, which the next step regresses on; null otherwise. */
  working: Tensor | null
  workingWeights: Tensor | null
  /** Rows of the minibatch that made the last step (SGD only; null otherwise). */
  batch: Tensor | null
  /** Passes over the data so far: steps for full-batch fitters, steps × batch / n for SGD. */
  epochs: number
  converged: boolean
  diverged: boolean
}

/** Options of the gradient-based fitters. */
export type GamGradientOptions = {
  /** Step size η or a schedule (gradient descent and SGD: default `stepScale`/L; Adam: default 0.05). */
  stepSize?: StepSize
  /**
   * Gradient descent and SGD without `stepSize`: η = stepScale/L for the curvature L of J at the optimum, a step that
   * is stable near it below 2 (default 1 for gradient descent, 0.5 for SGD).
   */
  stepScale?: number
  /** Converged when ‖∇J‖ ≤ tolerance (default 1e-9). */
  tolerance?: number
}

/**
 * The state shared by every fitter at β. A full-batch gradient method passes J and ‖∇J‖ from its own evaluation
 * (`known`), so the step costs no second pass over the data: D = 2nJ − βᵀS_λβ.
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

/** J and its gradient by reverse-mode autodiff, as an `ObjectiveFn` (an invalid mean gives +∞, for line searches). */
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

/** Penalised IRLS from β₀; each state carries the working response and weights the next step regresses on. */
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
 * penalised smoother to its partial residual in turn. Its fixed point is the P-IRLS optimum; offsets are not supported.
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

/** Gradient descent on J with a fixed step (default 1/L, L the problem's curvature), a schedule, or a line search. */
export function gamGradientDescent(
  problem: GamProblem,
  options: GamGradientOptions & { lineSearch?: 'backtracking' | 'strong-wolfe' } = {},
): Algorithm<void, GamFitState & { inner: FirstOrderState }> {
  const { stepScale = 1, tolerance = 1e-9, lineSearch } = options
  const stepSize = options.stepSize ?? stepScale / problem.curvature
  const inner = gradientDescent(autodiffObjective(problem), { stepSize, tolerance, lineSearch })
  return wrapFirstOrder(problem, 'gam-gradient-descent', inner)
}

/** Adam on J, full batch (default η = 0.05). */
export function gamAdam(
  problem: GamProblem,
  options: GamGradientOptions = {},
): Algorithm<void, GamFitState & { inner: FirstOrderState }> {
  const { stepSize = 0.05, tolerance = 1e-9 } = options
  return wrapFirstOrder(problem, 'gam-adam', adam(autodiffObjective(problem), { stepSize, tolerance }))
}

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

/** L-BFGS on J with a strong Wolfe line search (memory m, default 10). */
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
  /** Rows per minibatch (default 32, at most n). */
  batchSize?: number
  /** Momentum μ ∈ [0, 1) (default 0). */
  momentum?: number
}

/** The SGD state's bookkeeping: the epoch's row order and the next position in it, and the rule's velocity. */
type SgdInner = { order: Int32Array; position: number; velocity: Tensor | null }

/**
 * Minibatch SGD on J: each step takes the next `batchSize` rows of a permutation drawn at the start of each epoch,
 * differentiates the minibatch estimate of J by autodiff and applies compute's `sgdRule` (default η = 0.5/L).
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
  lineSearch?: 'backtracking' | 'strong-wolfe'
  memory?: number
}

/** The fitter for a method, with its options. */
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

/** What `gamProblem` chose (λ, active shape rows, the criterion), so the problem can be rebuilt without the search. */
export type GamProblemChoices = {
  lambdas: number[]
  active: number[][]
  smoothing: GamProblem['smoothing']
}

/** The choices of a GAM problem on data, as plain data: rebuild it with `gamProblem({ ...spec, ...choices }, data)`. */
export function gamProblemChoices(spec: GamSpec, data: GamData): GamProblemChoices {
  const p = gamProblem(spec, data)
  return { lambdas: p.lambdas, active: p.active, smoothing: p.smoothing }
}

/** One fitter's run in a `GamTrainingRun`. */
export type GamRunRequest = { method: GamFitMethod; options?: GamFitterOptions; steps: number; seed?: string | number }

/** A GAM fitted step by step by one or more fitters, as plain data (for a worker or a log). */
export type GamTrainingRun = {
  /** The problem's choices and the facts every run is measured against. */
  choices: GamProblemChoices
  curvature: number
  nullDeviance: number
  /** The P-IRLS optimum β̂ [P] and J(β̂). */
  optimum: { coefficients: Tensor; objective: number }
  /** Per request: every state from step 0 (each fitter's own bookkeeping dropped). */
  runs: { method: GamFitMethod; states: GamFitState[] }[]
}

/**
 * Build the problem once and run each requested fitter on it, keeping every state: the data a figure plays. With
 * `lambdas` (and `active`) in the spec the smoothing-parameter search is skipped.
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
