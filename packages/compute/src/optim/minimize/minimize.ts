/**
 * `minimize`: run any optimiser of `aifn-compute/optim` by name, as a `run` wrapper over its `Algorithm` (after scipy's
 * `optimize.minimize`; Virtanen et al., 2020, "SciPy 1.0", Nature Methods 17).
 *
 * The objective is given once, in whichever form the caller has: an `Objective` (its value written with primitives,
 * differentiated by reverse-mode autodiff), a bare function returning `{ value, grad }`, or, for the derivative-free
 * methods, one returning the value alone. Each method is handed the form it needs. Non-convergence and divergence are
 * reported in the result (`converged`, `diverged`), not thrown; an unknown method name throws `DomainError`.
 */

import type { Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { conjugateGradient, type ConjugateGradientOptions } from 'aifn-compute/optim/first-order'
import { coordinateDescent, type CoordinateDescentOptions } from 'aifn-compute/optim/first-order'
import {
  adagrad,
  adam,
  adamw,
  gradientDescent,
  momentum,
  nesterov,
  rmsprop,
  type AdamOptions,
  type AdaptiveOptions,
  type GradientDescentOptions,
  type MomentumOptions,
  type RmspropOptions,
} from 'aifn-compute/optim/first-order'
import { objectiveFn, valueFunction, type RunOptions, type StartOptions } from '../options'
import { nelderMead, type NelderMeadOptions } from 'aifn-compute/optim/derivative-free'
import { newton, trustRegion, type NewtonOptions, type TrustRegionOptions } from 'aifn-compute/optim/second-order'
import {
  bfgs,
  lbfgs,
  owlqn,
  type LbfgsOptions,
  type OwlqnOptions,
  type QuasiNewtonOptions,
} from 'aifn-compute/optim/second-order'
import {
  cmaEs,
  simulatedAnnealing,
  type CmaEsOptions,
  type SimulatedAnnealingOptions,
} from 'aifn-compute/optim/derivative-free'
import type {
  IterateState,
  Objective,
  ObjectiveFn,
  Scalar,
  Size,
  ValueFunction,
  VectorLike,
} from 'aifn-compute/foundation/contracts'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of each method `minimize` can run, by name. */
export type MethodOptions = {
  'gradient-descent': GradientDescentOptions
  momentum: MomentumOptions
  nesterov: MomentumOptions
  adagrad: AdaptiveOptions
  rmsprop: RmspropOptions
  adam: AdamOptions
  adamw: AdamOptions
  newton: NewtonOptions
  'trust-region': TrustRegionOptions
  bfgs: QuasiNewtonOptions
  lbfgs: LbfgsOptions
  owlqn: OwlqnOptions
  'conjugate-gradient': ConjugateGradientOptions
  'coordinate-descent': CoordinateDescentOptions
  'nelder-mead': NelderMeadOptions
  'simulated-annealing': SimulatedAnnealingOptions
  'cma-es': CmaEsOptions
}

/** A method name for `minimize`. */
export type Method = keyof MethodOptions

/** The result of `minimize`. */
export type MinimizeResult<S extends IterateState = IterateState> = {
  /** The method that ran. */
  method: Method
  /** The final iterate (for simulated annealing, the best point seen). */
  x: Vector
  /** $f$ at `x`. */
  value: Scalar
  /** The method's stopping test passed before the step budget ran out. */
  converged: boolean
  /** The value or iterate became non-finite, or the value passed `divergeAbove`. */
  diverged: boolean
  /** The steps taken (the state's step counter `t`). */
  steps: Size
  /** Calls of the objective, the initial one included. */
  evaluations: Size
  /** The final state of the algorithm, with all its internals. */
  state: S
}

/**
 * The step-through `Algorithm` of the named method on $f$, the one table `minimize` runs: gradient methods get the
 * value-and-gradient function of $f$ (`Objective`s differentiated by autodiff), derivative-free ones its value. Use it
 * to drive a method step by step (e.g. `aifn-compute/nn/training`'s `fullBatchTraining`) rather than run it to the end.
 * An unknown method name throws `DomainError`.
 *
 * @param method The method's name, one of the keys of `MethodOptions`.
 * @param f The objective: an `Objective`, a function returning `{ value, grad }`, or (for `'nelder-mead'`,
 *   `'simulated-annealing'` and `'cma-es'` only) a function returning the value.
 * @param options The named method's own options, passed to it unchanged.
 * @returns The method's `Algorithm`, started from `{ x0 }`.
 *
 * @example Step gradient descent through its first iterations
 * // f(x) = x₁² + 10x₂²: with step size 0.05, x₂ reaches 0 in one step and x₁ shrinks by 0.9 a step.
 * const bowl = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const alg = methodAlgorithm('gradient-descent', bowl, { stepSize: 0.05 })
 * for (const steps of [1, 2, 3, 10]) {
 *   const s = run(alg, { x0: [1, 1] }, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' f =', s.value)
 * }
 */
export function methodAlgorithm<M extends Method>(
  method: M,
  f: Objective<Tensor> | ObjectiveFn | ValueFunction,
  options: MethodOptions[M] = {} as MethodOptions[M],
): Algorithm<StartOptions, IterateState> {
  const gradient = objectiveFn(f as Objective<Tensor> | ObjectiveFn)
  const value = valueFunction(f)
  const o = options as never
  const algorithms: Record<Method, () => Algorithm<StartOptions, IterateState>> = {
    'gradient-descent': () => gradientDescent(gradient, o),
    momentum: () => momentum(gradient, o),
    nesterov: () => nesterov(gradient, o),
    adagrad: () => adagrad(gradient, o),
    rmsprop: () => rmsprop(gradient, o),
    adam: () => adam(gradient, o),
    adamw: () => adamw(gradient, o),
    newton: () => newton(gradient, o),
    'trust-region': () => trustRegion(gradient, o),
    bfgs: () => bfgs(gradient, o),
    lbfgs: () => lbfgs(gradient, o),
    owlqn: () => owlqn(gradient, o),
    'conjugate-gradient': () => conjugateGradient(gradient, o),
    'coordinate-descent': () => coordinateDescent(gradient, o),
    'nelder-mead': () => nelderMead(value, o),
    'simulated-annealing': () => simulatedAnnealing(value, o),
    'cma-es': () => cmaEs(value, o),
  }
  if (!Object.hasOwn(algorithms, method)) throw new DomainError('minimize', `minimize: unknown method "${method}"`)
  return algorithms[method]()
}

/**
 * Minimises $f$ from `x0` with the named method (default `'lbfgs'`) for at most `maxSteps` steps (default 1000),
 * returning the final iterate and state. The run stops early when the method converges or diverges, which the result
 * reports. `'newton'` and `'trust-region'` need a `hessian` option. Stochastic methods draw from `stream`.
 *
 * @param f The objective: an `Objective` (its value written with primitives; gradients by autodiff) or a bare
 *   function, returning `{ value, grad }` for gradient methods or a plain number for `'nelder-mead'`,
 *   `'simulated-annealing'` and `'cma-es'`.
 * @param x0 The starting point $\xvec_0$.
 * @param options `method` (default `'lbfgs'`), the runner's `maxSteps` and `stream`, and the named method's own
 *   options.
 * @returns The final iterate and its value, whether the method converged or diverged, the steps and evaluations used,
 *   and the method's final state.
 *
 * @example Rosenbrock's function by L-BFGS
 * // (1 − a)² + 100(b − a²)², least at (1, 1); started from (−1.2, 1).
 * const rosenbrock = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     value: (1 - a) ** 2 + 100 * (b - a * a) ** 2,
 *     grad: [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)],
 *   }
 * }
 * const r = minimize(rosenbrock, [-1.2, 1])
 * print('x =', r.x, ' f =', r.value)
 * print('converged =', r.converged, ' steps =', r.steps, ' evaluations =', r.evaluations)
 *
 * @example Without a gradient, by Nelder–Mead
 * const rosenbrock = (x) => {
 *   const [a, b] = x.data
 *   return (1 - a) ** 2 + 100 * (b - a * a) ** 2
 * }
 * const r = minimize(rosenbrock, [-1.2, 1], { method: 'nelder-mead', maxSteps: 2000 })
 * print('x =', r.x, ' f =', r.value)
 * print('converged =', r.converged, ' steps =', r.steps)
 *
 * @example An `Objective` is differentiated for the method
 * // x₁² + 10x₂² written with primitives; gradient descent gets its gradient by autodiff.
 * const bowl = { kind: 'objective', name: 'bowl', dim: 2, value: (x) => sum(mul(tensor([1, 10]), mul(x, x))) }
 * const r = minimize(bowl, [1, 1], { method: 'gradient-descent', stepSize: 0.05 })
 * print('x =', r.x, ' f =', r.value)
 * print('steps =', r.steps)
 */
export function minimize<M extends Method = 'lbfgs'>(
  f: Objective<Tensor> | ObjectiveFn | ValueFunction,
  x0: VectorLike,
  options: { method?: M } & RunOptions & MethodOptions[M] = {} as MethodOptions[M],
): MinimizeResult {
  const method = (options.method ?? 'lbfgs') as Method
  const { maxSteps = 1000, stream } = options
  const state = run(methodAlgorithm(method, f, options as MethodOptions[Method]), { x0 }, maxSteps, { stream })
  const annealing = method === 'simulated-annealing'
  const s = state as IterateState & { best?: Vector; bestValue?: number }
  return {
    method,
    x: annealing && s.best ? s.best : state.x,
    value: annealing && s.bestValue !== undefined ? s.bestValue : state.value,
    converged: state.converged,
    diverged: state.diverged,
    steps: state.t,
    evaluations: state.evaluations,
    state,
  }
}
