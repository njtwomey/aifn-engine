/**
 * Full-batch training of a parameter tree by any vector method of `aifn-compute/optim` (L-BFGS by default): the tree is read
 * as one flat vector θ (`ravel`, JAX's `ravel_pytree`), and the objective is θ ↦ loss(unravel(θ), the whole training
 * set), its gradient the per-leaf gradients (reverse-mode autodiff over the tree) raveled in the same order. Quasi-Newton
 * methods need exactly this: their curvature pairs y = ∇f(θ_{k+1}) − ∇f(θ_k) and their line search compare values and
 * gradients of one deterministic function, which minibatch noise would corrupt (Liu & Nocedal, 1989; Le et al., 2011,
 * "On optimization methods for deep learning", ICML).
 */

import type { IterateState, ObjectiveFn, StepContext } from 'aifn-compute/foundation/contracts'
import { valueAndGrad, type ValueAndGrad } from 'aifn-compute/foundation/autodiff'
import { ravel, type Params } from 'aifn-compute/foundation/pytree'
import { toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { methodAlgorithm, type Method, type MethodOptions } from 'aifn-compute/optim/minimize'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `fullBatchTraining`. */
export type FullBatchOptions<P extends Params, B, M extends Method = 'lbfgs'> = {
  /** The loss of the parameters on the whole training set, a number or rank-0 value written with primitives. */
  loss: (params: P, data: B) => Value
  /** The training set, passed whole to every evaluation. */
  data: B
  /** The vector method of `aifn-compute/optim/minimize` (default `'lbfgs'`). */
  method?: M
  /** The method's options (L-BFGS: `memory`, `tolerance`, `lineSearchOptions`; gradient descent: `stepSize`, …). */
  options?: MethodOptions[M]
}

/** The state of `fullBatchTraining`: the method's own state (θ as `x`, its internals) and θ as a parameter tree. */
export type FullBatchState<P extends Params, S extends IterateState = IterateState> = S & {
  /** The parameters, `unravel(x)`. */
  readonly params: P
}

/**
 * The flat objective of a parameter tree: `unravel` rebuilds the tree from θ, and the gradient is the tree of per-leaf
 * gradients raveled in leaf order (the same order as θ). Every call evaluates the loss and its gradient on the whole set
 * once, so a method's `evaluations` counts full-data gradient evaluations.
 */
export function treeObjective<P extends Params, B>(
  loss: (params: P, data: B) => Value,
  data: B,
  unravel: (x: ArrayLike<number>) => P,
): ObjectiveFn {
  const lossAndGrad: (params: P) => ValueAndGrad<Value, unknown> = valueAndGrad((params: P) => loss(params, data))
  return (x) => {
    const { value, grad } = lossAndGrad(unravel(toFlat(x)))
    const raw = unwrap(value)
    return { value: typeof raw === 'number' ? raw : toFlat(raw)[0], grad: ravel(grad).vector }
  }
}

/**
 * Train a parameter tree on the whole training set with a vector method of `aifn-compute/optim` (default L-BFGS), as a
 * step-through algorithm: `init` takes `{ params }` (e.g. `model.init(stream)`); each step is one step of the method
 * (for L-BFGS, a two-loop direction and a strong Wolfe line search, which may evaluate the loss several times). The
 * state is the method's (`x`, `value`, `evaluations`, and for L-BFGS `stepSize`, `lineSearch`, `pairs`, `skipped`, …)
 * with the parameters as a tree in `params`. It stops when the method does (converged, diverged or stalled).
 */
export function fullBatchTraining<P extends Params, B, M extends Method = 'lbfgs'>(
  options: FullBatchOptions<P, B, M>,
): Algorithm<{ params: P }, FullBatchState<P>> {
  const { loss, data } = options
  const method = (options.method ?? 'lbfgs') as M
  const methodOptions = (options.options ?? {}) as MethodOptions[M]
  // The tree's structure fixes `unravel`; the method is rebuilt from it, so every step is a pure function of its state.
  const methodFor = (params: P) => {
    const { vector, unravel } = ravel(params)
    const alg = methodAlgorithm(method, treeObjective(loss, data, unravel), methodOptions)
    return { vector, unravel, alg }
  }
  // The stopping rule reads only the state's flags, so it comes from the method built on any objective.
  const unused: ObjectiveFn = () => {
    throw new DomainError('fullBatchTraining', 'fullBatchTraining: no objective here')
  }
  const stops = methodAlgorithm(method, unused, methodOptions)
  return {
    name: `full-batch-${method}`,
    init: ({ params }, s) => {
      const { vector, unravel, alg } = methodFor(params)
      const state = alg.init({ x0: vector }, s)
      return { ...state, params: unravel(toFlat(state.x)) }
    },
    step: (state, ctx: StepContext) => {
      const { unravel, alg } = methodFor(state.params)
      const next = alg.step(state, ctx)
      return { ...next, params: unravel(toFlat(next.x)) }
    },
    done: (state) => stops.done?.(state) ?? false,
  }
}
