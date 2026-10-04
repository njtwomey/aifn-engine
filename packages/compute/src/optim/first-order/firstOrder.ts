/**
 * First-order methods as traceable algorithms: gradient descent (fixed, scheduled or line-searched step), heavy-ball
 * momentum, Nesterov's accelerated gradient, AdaGrad, RMSProp and Adam / AdamW. Each runs one of the pytree update
 * rules of `rules.ts` (the one definition of the method) on a single vector x with a fixed objective f, evaluating
 * f and ∇f once per step. The steps are written with primitives, so `unrolled` differentiates through them: with
 * respect to x₀, to a traced step size (a learning-rate hypergradient), or to anything f closes over, when f is written
 * with primitives too (an `Objective` through `objectiveFn`, or an `ObjectiveFn` whose value and gradient are).
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
  /** ∇f(x). */
  grad: Vector
  gradNorm: Scalar
  /** The step applied last, x_t − x_{t−1} (zeros at t = 0). */
  update: Vector
  /** The step size (or accepted line-search step) used on the last step; NaN at t = 0. */
  stepSize: Scalar
  /**
   * The update rule's running quantities (`RuleState.slots`): `velocity` (momentum, Nesterov), `sumSquares`
   * (AdaGrad), `meanSquare` (RMSProp), `firstMoment` and `secondMoment` (Adam, before bias correction). Empty for
   * gradient descent.
   */
  slots: Record<string, Vector>
  /** The last line search (gradient descent with `lineSearch` only), with its trial points; null otherwise. */
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower f (x unchanged); the run stops. Line-searched descent only. */
  stalled: boolean
}

/** Options every first-order method takes: the stopping options and the step size. */
export type FirstOrderOptions = StoppingOptions & {
  /** Step size η, or a schedule t ↦ η_t (see `inverseTimeDecay`, `exponentialDecay`, `inverseSqrtDecay`). */
  stepSize?: StepSize
}

/** A number from a value that may be traced (its primal), for the flags and the reported norms. */
const primal = (v: Value): number => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : dense.toF64(raw, 'primal')[0]
}

/**
 * f(x) and ∇f(x) at a point that may be traced, kept as values (so a traced x gives a traced value and gradient), with
 * the gradient checked for length.
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

/** The number of elements of a value. */
const sizeOf = (v: Value): number => shapeOfValue(v).reduce((a, b) => a * b, 1)

/** ‖v‖₂ of the primal values. */
const primalNorm = (v: Value): number => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? Math.abs(raw) : norm(dense.data(raw))
}

/**
 * Runs an update rule on one vector: evaluate at x₀, then x ← x + update(∇f(x)) and re-evaluate. Every quantity that
 * carries x (the iterate, the value, the gradient, the update and the rule's slots) is computed with primitives; the
 * flags and norms read primal values.
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
   * Choose each step by a line search along −∇f starting from α₀ = `stepSize` (at step t): backtracking to the Armijo
   * condition, or a strong Wolfe search. Off by default (fixed or scheduled step).
   */
  lineSearch?: 'backtracking' | 'strong-wolfe'
  /** Options for the line search. */
  lineSearchOptions?: BacktrackingOptions & StrongWolfeOptions
}

/**
 * Gradient descent, x ← x − η_t∇f(x) (Cauchy, 1847), with a fixed step, a schedule, or a line search along the
 * negative gradient (Nocedal & Wright, 2006, §3.1). `f` returns `{ value, grad }`; `init` takes `{ x0 }`. Default
 * η = 0.01 (1 as the first trial of a line search).
 *
 * On a quadratic with Hessian eigenvalues in [μ, L], a fixed step η < 2/L converges linearly, with rate
 * max(|1 − ημ|, |1 − ηL|) per step.
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
  /** Momentum coefficient μ in [0, 1). Default 0.9. */
  momentum?: Scalar
}

/**
 * Heavy-ball momentum (Polyak, 1964) in PyTorch's form: v ← μv + ∇f(x), x ← x − ηv. With μ = 0 it is gradient
 * descent. Default η = 0.01, μ = 0.9. The velocity (a sum of gradients) is `slots.velocity`. See `sgdRule`.
 */
export function momentum(f: ObjectiveFn, options: MomentumOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.01
  const rule = sgdRule({ stepSize, momentum: options.momentum ?? 0.9 })
  return ruleAlgorithm('momentum', f, options, rule, stepSize)
}

/**
 * Nesterov's accelerated gradient in the form of Sutskever et al. (2013), as PyTorch writes it: v ← μv + ∇f(x),
 * x ← x − η(∇f(x) + μv). One gradient per step; the look-ahead is folded into the update. Default η = 0.01, μ = 0.9.
 */
export function nesterov(f: ObjectiveFn, options: MomentumOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.01
  const rule = sgdRule({ stepSize, momentum: options.momentum ?? 0.9, nesterov: true })
  return ruleAlgorithm('nesterov', f, options, rule, stepSize)
}

/** Options for the adaptive methods. */
export type AdaptiveOptions = FirstOrderOptions & Omit<AdaptiveRuleOptions, 'stepSize'>

/**
 * AdaGrad (Duchi, Hazan & Singer, 2011): G ← G + g², x ← x − ηg / (√G + ε), elementwise. The accumulated squares are
 * `slots.sumSquares`. Default η = 0.1. See `adagradRule`.
 */
export function adagrad(f: ObjectiveFn, options: AdaptiveOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.1
  return ruleAlgorithm('adagrad', f, options, adagradRule({ ...options, stepSize }), stepSize)
}

/** Options for `rmsprop`. */
export type RmspropOptions = FirstOrderOptions & Omit<RmspropRuleOptions, 'stepSize'>

/**
 * RMSProp (Tieleman & Hinton, 2012): s ← ρs + (1 − ρ)g², x ← x − ηg / (√s + ε). The mean square is
 * `slots.meanSquare`. Default η = 0.01, ρ = 0.9. See `rmspropRule`.
 */
export function rmsprop(f: ObjectiveFn, options: RmspropOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 0.01
  return ruleAlgorithm('rmsprop', f, options, rmspropRule({ ...options, stepSize }), stepSize)
}

/** Options for `adam` and `adamw`. */
export type AdamOptions = FirstOrderOptions & Omit<AdamRuleOptions, 'stepSize'>

/**
 * Adam (Kingma & Ba, 2015, Algorithm 1): m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², m̂ = m/(1 − β₁ᵗ),
 * v̂ = v/(1 − β₂ᵗ), x ← x − ηm̂/(√v̂ + ε), with t counting from 1. The raw moments are `slots.firstMoment` and
 * `slots.secondMoment`. Default η = 0.001. With `weightDecay` and `decoupled` it is AdamW (Loshchilov & Hutter, 2019).
 */
export function adam(f: ObjectiveFn, options: AdamOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const stepSize = options.stepSize ?? 1e-3
  const rule = adamRule({ ...options, stepSize })
  return ruleAlgorithm(rule.name, f, options, rule, stepSize)
}

/** AdamW (Loshchilov & Hutter, 2019): Adam with decoupled weight decay, default λ = 0.01. See `adam`. */
export function adamw(f: ObjectiveFn, options: AdamOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  return adam(f, { weightDecay: 0.01, ...options, decoupled: options.decoupled ?? true })
}
