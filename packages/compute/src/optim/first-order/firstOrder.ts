/**
 * First-order methods as traceable algorithms: gradient descent (fixed, scheduled or line-searched step), heavy-ball
 * momentum, Nesterov's accelerated gradient, AdaGrad, RMSProp and Adam / AdamW. Each runs one of the pytree update
 * rules of `rules.ts` (the one definition of the method) on a single vector $\xvec$ with a fixed objective $f$,
 * evaluating $f$ and $\nabla f$ once per step. Each takes `{ x0 }` to `init`, stops (converged) when
 * $\lVert \nabla f(\xvec) \rVert_2 \le$ `tolerance`, and reports divergence in its state rather than throwing.
 *
 * The steps are written with primitives, so `unrolled` differentiates through them: with respect to $\xvec_0$, to a
 * traced step size (a learning-rate hypergradient), or to anything $f$ closes over, when $f$ is written with
 * primitives too (an `Objective` through `objectiveFn`, or an `ObjectiveFn` whose value and gradient are). Gradient
 * descent with a line search is the exception: its steps are taken on float64 arrays.
 *
 * Sources: Cauchy (1847), "Méthode générale pour la résolution des systèmes d'équations simultanées"; Polyak (1964);
 * Sutskever et al. (2013); Duchi, Hazan & Singer (2011); Tieleman & Hinton (2012); Kingma & Ba (2015); Loshchilov &
 * Hutter (2019) (full titles in `rules.ts`).
 */

import { DomainError, NotDifferentiableError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  add,
  dense,
  fromData,
  isTensor,
  isTraced,
  shapeOfValue,
  unwrap,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import type { IterateState, ObjectiveFn, Scalar, StoppingOptions } from 'aifn-compute/foundation/contracts'
import {
  backtrackingSearch,
  strongWolfeSearch,
  type BacktrackingOptions,
  type LineSearchResult,
  type StrongWolfeOptions,
} from 'aifn-compute/optim/line-search'
import { divergedAt, stopping, type StartOptions } from '../options'
import {
  adagradRule,
  adamRule,
  rmspropRule,
  sgdRule,
  stepSizeAt,
  type AdamRuleOptions,
  type AdaptiveRuleOptions,
  type RmspropRuleOptions,
  type StepSize,
  type UpdateRule,
} from './rules'

const { data, norm, sub, toF64, vec } = dense

/** The state of a first-order method. */
export type FirstOrderState = IterateState & {
  /** $\nabla f(\xvec)$. */
  grad: Vector
  /** $\lVert \nabla f(\xvec) \rVert_2$, the quantity the stopping test compares with `tolerance`. */
  gradNorm: Scalar
  /** The step applied last, $\xvec_t - \xvec_{t-1}$ (zeros at $t = 0$). */
  update: Vector
  /** The step size (or accepted line-search step) used on the last step; NaN at $t = 0$. */
  stepSize: Scalar
  /**
   * The update rule's running quantities (`RuleState.slots`): `velocity` (momentum, Nesterov), `sumSquares`
   * (AdaGrad), `meanSquare` (RMSProp), `firstMoment` and `secondMoment` (Adam, before bias correction). Empty for
   * gradient descent.
   */
  slots: Record<string, Vector>
  /** The last line search (gradient descent with `lineSearch` only), with its trial points; null otherwise. */
  lineSearch: LineSearchResult | null
  /**
   * True when the last line search could not lower $f$ ($\xvec$ unchanged); the run stops. Line-searched descent only.
   */
  stalled: boolean
}

/** Options every first-order method takes: the stopping options and the step size. */
export type FirstOrderOptions = StoppingOptions & {
  /**
   * Step size $\eta$, or a schedule $t \mapsto \eta_t$ (see `inverseTimeDecay`, `exponentialDecay`,
   * `inverseSqrtDecay`). The default depends on the method.
   */
  stepSize?: StepSize
}

/**
 * A number from a value that may be traced (its primal), for the flags and the reported norms.
 *
 * @param v A number, a tensor or a traced value; for a tensor, its first entry is taken.
 * @returns The value as a plain number.
 */
const primal = (v: Value): number => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : dense.toF64(raw, 'primal')[0]
}

/**
 * $f(\xvec)$ and $\nabla f(\xvec)$ at a point that may be traced, kept as values (so a traced $\xvec$ gives a traced
 * value and gradient), with the gradient checked for length. An objective that returns no gradient throws
 * `DomainError`; a plain-array gradient at a traced point throws `NotDifferentiableError`, since its dependence on
 * $\xvec$ is lost; a gradient of the wrong length throws `ShapeError`.
 *
 * @param f The objective, returning `{ value, grad }`.
 * @param x The point, a vector or a traced value.
 * @param n The length of `x`, which the gradient must match.
 * @param where The caller's name, for error messages.
 * @returns The value and the gradient, a plain-array gradient converted to a tensor.
 */
function evaluateValue(f: ObjectiveFn, x: Value, n: number, where: string): { value: Value; grad: Value } {
  const out = f(x as Vector) as { value: Value; grad?: Value | ArrayLike<number> } | number
  if (typeof out === 'number' || out.grad === undefined)
    throw new DomainError(where, `${where}: the objective must return { value, grad }; this method uses the gradient`)
  const g = out.grad
  // A plain array for a traced point was computed on raw values: the gradient's dependence on x is lost.
  if (isTraced(x) && typeof g === 'object' && !isTensor(g) && !isTraced(g))
    throw new NotDifferentiableError(
      where,
      `${where}: the objective returned a plain-array gradient for a traced point; write it with aifn primitives`,
    )
  const grad: Value =
    typeof g === 'number' || isTensor(g) || !('length' in g) ? (g as Value) : fromData(Float64Array.from(g), [g.length])
  const length = sizeOf(grad)
  if (length !== n) throw new ShapeError(where, `${where}: the gradient has length ${length}, but x has length ${n}`)
  return { value: out.value, grad }
}

/**
 * The number of elements of a value.
 *
 * @param v A number, a tensor or a traced value.
 * @returns The product of its shape (1 for a number).
 */
const sizeOf = (v: Value): number => shapeOfValue(v).reduce((a, b) => a * b, 1)

/**
 * $\lVert \vvec \rVert_2$ of the primal values.
 *
 * @param v A number, a tensor or a traced value.
 * @returns The Euclidean norm of its primal values (the absolute value for a number).
 */
const primalNorm = (v: Value): number => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? Math.abs(raw) : norm(dense.data(raw))
}

/**
 * Runs an update rule on one vector: evaluate at $\xvec_0$, then $\xvec \leftarrow \xvec + u(\nabla f(\xvec))$, with
 * $u$ the rule's update, and re-evaluate. Every quantity that carries $\xvec$ (the iterate, the value, the gradient,
 * the update and the rule's slots) is computed with primitives; the flags and norms read primal values. A traced
 * $\xvec_0$ stays traced; anything else is copied.
 *
 * @param name The algorithm's name, also used in error messages.
 * @param f The objective, returning `{ value, grad }`.
 * @param options The stopping options (`tolerance`, `divergeAbove`).
 * @param rule The update rule run on the iterate, which is passed to it as the parameters.
 * @param stepSize The rule's step size, read only to report `stepSize` in the state.
 * @returns The algorithm; `init` takes `{ x0 }`.
 */
function ruleAlgorithm(
  name: string,
  f: ObjectiveFn,
  options: StoppingOptions,
  rule: UpdateRule,
  stepSize: StepSize,
): Algorithm<StartOptions, FirstOrderState> {
  const { tolerance, divergeAbove } = stopping(options)
  return {
    name,
    init: ({ x0 }) => {
      // A traced x₀ (unrolled, differentiating with respect to the start) stays traced; anything else is copied.
      const x: Value = isTraced(x0) ? (x0 as unknown as Value) : vec(toF64(x0, name))
      const n = sizeOf(x)
      const { value, grad } = evaluateValue(f, x, n, name)
      const gradNorm = primalNorm(grad)
      const { slots } = rule.init(x as Tensor)
      return {
        t: 0,
        x: x as Vector,
        value: value as Scalar,
        grad: grad as Vector,
        gradNorm,
        update: vec(new Float64Array(n)),
        stepSize: NaN,
        slots: slots as Record<string, Vector>,
        lineSearch: null,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(primal(value), dense.data(unwrap(x) as Tensor), divergeAbove),
      }
    },
    step: (s) => {
      const out = rule.update(s.grad, { t: s.t, slots: s.slots }, s.x)
      const update = out.updates as Value
      const next = add(s.x, update)
      const { value, grad } = evaluateValue(f, next, sizeOf(next), name)
      const gradNorm = primalNorm(grad)
      return {
        t: s.t + 1,
        x: next as Vector,
        value: value as Scalar,
        grad: grad as Vector,
        gradNorm,
        update: update as Vector,
        stepSize: primal(stepSizeAt(stepSize, s.t)),
        slots: out.state.slots as Record<string, Vector>,
        lineSearch: null,
        stalled: false,
        evaluations: s.evaluations + 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(primal(value), dense.data(unwrap(next) as Tensor), divergeAbove),
      }
    },
    done: (s) => s.stalled,
  }
}

/** Options for `gradientDescent`. */
export type GradientDescentOptions = FirstOrderOptions & {
  /**
   * Choose each step by a line search along $-\nabla f$ starting from $\alpha_0$ = `stepSize` (at step $t$):
   * backtracking to the Armijo condition, or a strong Wolfe search. Off by default (fixed or scheduled step).
   */
  lineSearch?: 'backtracking' | 'strong-wolfe'
  /** Options for the line search. */
  lineSearchOptions?: BacktrackingOptions & StrongWolfeOptions
}

/**
 * Gradient descent, $\xvec \leftarrow \xvec - \eta_t\nabla f(\xvec)$ (Cauchy, 1847), with a fixed step, a schedule,
 * or a line search along the negative gradient (Nocedal & Wright, 2006, §3.1). Default $\eta = 0.01$ (1 as the first
 * trial of a line search). A line search that cannot lower $f$ sets `stalled` and ends the run.
 *
 * On a quadratic with Hessian eigenvalues in $[\mu, L]$, a fixed step $\eta < 2/L$ converges linearly, with rate
 * $\max(\lvert 1 - \eta\mu \rvert, \lvert 1 - \eta L \rvert)$ per step.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The step size or schedule, the optional line search and its options, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`, and each step evaluates $f$ once (or once per line-search trial).
 *
 * @example A few steps on a quadratic bowl
 * // f(x) = x₁² + 10x₂²: Hessian eigenvalues 2 and 20, so η = 0.08 contracts by
 * // max(|1 − 0.16|, |1 − 1.6|) = 0.84 a step.
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * for (const steps of [1, 5, 50]) {
 *   const s = run(gradientDescent(bowl, { stepSize: 0.08 }), { x0: [1, 1] }, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' f =', s.value)
 * }
 *
 * @example Too large a step diverges
 * // 2 / L = 0.1 for this bowl.
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * for (const stepSize of [0.09, 0.11]) {
 *   const s = run(gradientDescent(bowl, { stepSize }), { x0: [1, 1] }, 1000)
 *   print(`step size ${stepSize}: steps =`, s.t, ' converged =', s.converged, ' diverged =', s.diverged)
 * }
 *
 * @example With a backtracking line search
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const s = run(gradientDescent(bowl, { lineSearch: 'backtracking' }), { x0: [1, 1] }, 1000)
 * print('x =', s.x, ' converged =', s.converged)
 * print('steps =', s.t, ' evaluations =', s.evaluations)
 * print('last step length =', s.stepSize)
 */
export function gradientDescent(
  f: ObjectiveFn,
  options: GradientDescentOptions = {},
): Algorithm<StartOptions, FirstOrderState> {
  const { lineSearch } = options
  if (!lineSearch) {
    const stepSize = options.stepSize ?? 0.01
    return ruleAlgorithm('gradient-descent', f, options, sgdRule({ stepSize }), stepSize)
  }
  const stepSize = options.stepSize ?? 1
  const { tolerance, divergeAbove } = stopping(options)
  const base = ruleAlgorithm('gradient-descent', f, options, sgdRule({ stepSize }), stepSize)
  return {
    ...base,
    step: (s) => {
      const x = data(s.x)
      const g = data(s.grad)
      const p = dense.scale(-1, g)
      const alpha0 = primal(stepSizeAt(stepSize, s.t))
      const search =
        lineSearch === 'backtracking'
          ? backtrackingSearch(f, x, s.value, g, p, { ...options.lineSearchOptions, alpha0 })
          : strongWolfeSearch(f, x, s.value, g, p, { ...options.lineSearchOptions, alpha0 })
      const gradNorm = norm(search.grad)
      return {
        ...s,
        t: s.t + 1,
        x: vec(search.x),
        value: search.value,
        grad: vec(search.grad),
        gradNorm,
        update: vec(sub(search.x, x)),
        stepSize: search.result.alpha,
        lineSearch: search.result,
        stalled: search.result.alpha === 0,
        evaluations: s.evaluations + search.result.evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(search.value, search.x, divergeAbove),
      }
    },
  }
}

/** Options for `momentum` and `nesterov`. */
export type MomentumOptions = FirstOrderOptions & {
  /** Momentum coefficient $\mu$ in $[0, 1)$. Default 0.9. */
  momentum?: Scalar
}

/**
 * Heavy-ball momentum (Polyak, 1964) in PyTorch's form: $\vvec \leftarrow \mu\vvec + \nabla f(\xvec)$,
 * $\xvec \leftarrow \xvec - \eta\vvec$. With $\mu = 0$ it is gradient descent. Default $\eta = 0.01$, $\mu = 0.9$.
 * The velocity (a sum of gradients) is `slots.velocity`. See `sgdRule`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The step size $\eta$ or schedule, the momentum $\mu$, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Momentum against gradient descent on an ill-conditioned bowl
 * // f(x) = x₁² + 10x₂², 100 steps of size 0.01 from (1, 1).
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const gd = run(gradientDescent(bowl, { stepSize: 0.01 }), { x0: [1, 1] }, 100)
 * const heavy = run(momentum(bowl, { stepSize: 0.01 }), { x0: [1, 1] }, 100)
 * print('gradient descent: x =', gd.x, ' f =', gd.value)
 * print('momentum: x =', heavy.x, ' f =', heavy.value)
 * print('velocity =', heavy.slots.velocity)
 */
export function momentum(f: ObjectiveFn, options: MomentumOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.01
  const rule = sgdRule({ stepSize, momentum: options.momentum ?? 0.9 })
  return ruleAlgorithm('momentum', f, options, rule, stepSize)
}

/**
 * Nesterov's accelerated gradient in the form of Sutskever et al. (2013), as PyTorch writes it:
 * $\vvec \leftarrow \mu\vvec + \nabla f(\xvec)$, $\xvec \leftarrow \xvec - \eta(\nabla f(\xvec) + \mu\vvec)$. One
 * gradient per step; the look-ahead is folded into the update. Default $\eta = 0.01$, $\mu = 0.9$. The velocity is
 * `slots.velocity`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The step size $\eta$ or schedule, the momentum $\mu$, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Nesterov against heavy-ball momentum
 * // f(x) = x₁² + 10x₂², 100 steps of size 0.01 from (1, 1), μ = 0.9 for both.
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const heavy = run(momentum(bowl, { stepSize: 0.01 }), { x0: [1, 1] }, 100)
 * const ahead = run(nesterov(bowl, { stepSize: 0.01 }), { x0: [1, 1] }, 100)
 * print('momentum: x =', heavy.x, ' f =', heavy.value)
 * print('Nesterov: x =', ahead.x, ' f =', ahead.value)
 */
export function nesterov(f: ObjectiveFn, options: MomentumOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.01
  const rule = sgdRule({ stepSize, momentum: options.momentum ?? 0.9, nesterov: true })
  return ruleAlgorithm('nesterov', f, options, rule, stepSize)
}

/** Options for the adaptive methods (`adagrad`): the stopping options, the step size and `epsilon`. */
export type AdaptiveOptions = FirstOrderOptions & Omit<AdaptiveRuleOptions, 'stepSize'>

/**
 * AdaGrad (Duchi, Hazan & Singer, 2011): $\mathbf{G} \leftarrow \mathbf{G} + \gvec^2$,
 * $\xvec \leftarrow \xvec - \eta\gvec / (\sqrt{\mathbf{G}} + \varepsilon)$, elementwise, with
 * $\gvec = \nabla f(\xvec)$.
 * The accumulated squares are `slots.sumSquares`. Default $\eta = 0.1$. See `adagradRule`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The step size $\eta$ or schedule, $\varepsilon$, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Every coordinate moves at the same rate
 * // f(x) = x₁² + 10x₂²: the per-coordinate scaling makes the first step ±η whatever the gradient.
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * for (const steps of [1, 10, 100]) {
 *   const s = run(adagrad(bowl, { stepSize: 0.5 }), { x0: [1, 1] }, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' f =', s.value)
 * }
 */
export function adagrad(f: ObjectiveFn, options: AdaptiveOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.1
  return ruleAlgorithm('adagrad', f, options, adagradRule({ ...options, stepSize }), stepSize)
}

/** Options for `rmsprop`. */
export type RmspropOptions = FirstOrderOptions & Omit<RmspropRuleOptions, 'stepSize'>

/**
 * RMSProp (Tieleman & Hinton, 2012): $\svec \leftarrow \rho\svec + (1 - \rho)\gvec^2$,
 * $\xvec \leftarrow \xvec - \eta\gvec / (\sqrt{\svec} + \varepsilon)$, elementwise, with $\gvec = \nabla f(\xvec)$.
 * The mean square is `slots.meanSquare`. Default $\eta = 0.01$, $\rho = 0.9$. See `rmspropRule`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The step size $\eta$ or schedule, the decay $\rho$, $\varepsilon$, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example RMSProp on a quadratic bowl
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * for (const steps of [1, 10, 100]) {
 *   const s = run(rmsprop(bowl, { stepSize: 0.05 }), { x0: [1, 1] }, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' f =', s.value)
 * }
 */
export function rmsprop(f: ObjectiveFn, options: RmspropOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.01
  return ruleAlgorithm('rmsprop', f, options, rmspropRule({ ...options, stepSize }), stepSize)
}

/** Options for `adam` and `adamw`. */
export type AdamOptions = FirstOrderOptions & Omit<AdamRuleOptions, 'stepSize'>

/**
 * Adam (Kingma & Ba, 2015, Algorithm 1): $\mvec \leftarrow \beta_1\mvec + (1 - \beta_1)\gvec$,
 * $\vvec \leftarrow \beta_2\vvec + (1 - \beta_2)\gvec^2$, $\hat{\mvec} = \mvec/(1 - \beta_1^t)$,
 * $\hat{\vvec} = \vvec/(1 - \beta_2^t)$, $\xvec \leftarrow \xvec - \eta\hat{\mvec}/(\sqrt{\hat{\vvec}} + \varepsilon)$,
 * elementwise, with $\gvec = \nabla f(\xvec)$ and $t$ counting from 1. The raw moments are `slots.firstMoment` and
 * `slots.secondMoment`. Default $\eta = 0.001$. With `weightDecay` and `decoupled` it is AdamW (Loshchilov & Hutter,
 * 2019).
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The step size $\eta$ or schedule, $\beta_1$, $\beta_2$, $\varepsilon$, the weight decay and whether it
 *   is decoupled, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Adam on a quadratic bowl
 * // The first step is close to η in every coordinate, whatever the size of the gradient.
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * for (const steps of [1, 10, 100]) {
 *   const s = run(adam(bowl, { stepSize: 0.1 }), { x0: [1, 1] }, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' f =', s.value)
 * }
 */
export function adam(f: ObjectiveFn, options: AdamOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 1e-3
  const rule = adamRule({ ...options, stepSize })
  return ruleAlgorithm(rule.name, f, options, rule, stepSize)
}

/**
 * AdamW (Loshchilov & Hutter, 2019): Adam with decoupled weight decay, default $\lambda = 0.01$: the update adds
 * $-\eta\lambda\xvec$ to Adam's. See `adam`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options As for `adam`; `weightDecay` defaults to 0.01 and `decoupled` to true.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Weight decay pulls the iterate towards zero
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const plain = run(adam(bowl, { stepSize: 0.1 }), { x0: [1, 1] }, 1)
 * const decayed = run(adamw(bowl, { stepSize: 0.1 }), { x0: [1, 1] }, 1)
 * print('Adam, one step: x =', plain.x)
 * print('AdamW, one step: x =', decayed.x)
 */
export function adamw(f: ObjectiveFn, options: AdamOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  return adam(f, { weightDecay: 0.01, ...options, decoupled: options.decoupled ?? true })
}
