/**
 * Full-batch training of a parameter tree by any vector method of `aifn-compute/optim/minimize` (L-BFGS by default).
 *
 * The tree is read as one flat vector $\thetavec$ (`ravel`, JAX's `ravel_pytree`), and the objective is
 * $\thetavec \mapsto$ the loss of the rebuilt tree on the whole training set, its gradient the per-leaf gradients
 * (reverse-mode autodiff over the tree) raveled in the same order. Quasi-Newton methods need exactly this: their
 * curvature pairs $\yvec_k = \nabla f(\thetavec_{k+1}) - \nabla f(\thetavec_k)$ and their line search compare values
 * and gradients of one deterministic function, which minibatch noise would corrupt (Liu & Nocedal, 1989; Le et al.,
 * 2011, "On optimization methods for deep learning", ICML).
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
  /** The method's options (L-BFGS: `memory`, `tolerance`, `lineSearchOptions`; gradient descent: `stepSize`, ...). */
  options?: MethodOptions[M]
}

/**
 * The state of `fullBatchTraining`: the method's own state ($\thetavec$ as `x`, its internals) and $\thetavec$ as a
 * parameter tree.
 */
export type FullBatchState<P extends Params, S extends IterateState = IterateState> = S & {
  /** The parameters, `unravel(x)`. */
  readonly params: P
}

/**
 * The flat objective of a parameter tree: `unravel` rebuilds the tree from $\thetavec$, and the gradient is the tree of
 * per-leaf gradients raveled in leaf order (the same order as $\thetavec$). Every call evaluates the loss and its
 * gradient on the whole set once, so a method's `evaluations` counts full-data gradient evaluations.
 *
 * @param loss The loss of a parameter tree on the data, a number or rank-0 value written with primitives.
 * @param data The training set, passed whole to every evaluation.
 * @param unravel Rebuilds the parameter tree from the entries of $\thetavec$ (as `ravel` returns it); its leaf order
 *   must be the one `ravel` uses, so that the gradient lines up with $\thetavec$.
 * @returns The objective: $\thetavec$ to the loss and its gradient as a flat vector.
 *
 * @example The objective $\lVert \wvec - \tvec \rVert^2$ of a one-leaf tree, with gradient $2(\wvec - \tvec)$
 * const loss = (p, d) => sum(square(sub(p.w, d.target)))
 * const unravel = (x) => ({ w: tensor([x[0], x[1]]) })
 * const f = treeObjective(loss, { target: tensor([1, 2]) }, unravel)
 * print(f(tensor([0, 0])))
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
 * Train a parameter tree on the whole training set with a vector method of `aifn-compute/optim/minimize` (default
 * L-BFGS), as a step-through algorithm: `init` takes `{ params }` (e.g. `model.init(stream)`); each step is one step of
 * the method (for L-BFGS, a two-loop direction and a strong Wolfe line search, which may evaluate the loss several
 * times). The state is the method's (`x`, `value`, `evaluations`, and for L-BFGS `stepSize`, `lineSearch`, `pairs`,
 * `skipped`, ...) with the parameters as a tree in `params`. It stops when the method does (converged, diverged or
 * stalled).
 *
 * @param options The loss, the data, the method and the method's options.
 * @returns The algorithm, to run with `run` or `trace` from `{ params }`.
 *
 * @example L-BFGS fits $y = 2x + 1$ in a few steps
 * const data = { x: tensor([[0], [1], [2], [3]]), y: tensor([[1], [3], [5], [7]]) }
 * const loss = (p, d) => mean(square(sub(add(matmul(d.x, p.w), p.b), d.y)))
 * const alg = fullBatchTraining({ loss, data })
 * const tr = trace(alg, { params: { w: zeros([1, 1]), b: zeros([1]) } }, 20, { record: { loss: (s) => s.value } })
 * print('loss by step:', tr.series.loss)
 * const final = run(alg, { params: { w: zeros([1, 1]), b: zeros([1]) } }, 20)
 * print('w =', final.params.w, ' b =', final.params.b, ' evaluations:', final.evaluations)
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
