/**
 * The ODE block (Chen, Rubanova, Bettencourt & Duvenaud, 2018, "Neural ordinary differential equations", NeurIPS): a
 * layer whose output is the solution $\xvec(t_1)$ of $\xvec' = f_\theta(t, \xvec)$ from its input
 * $\xvec(t_0) = \xvec$, with $f_\theta$ any layer mapping `[B, d]` (or `[B, d + 1]` with the time appended) to
 * `[B, d]`. Solved by `odeFlow` of `aifn-compute/dynamics/ode` (fixed-step RK4 by default), and differentiated by
 * backpropagation through the solver or by the adjoint method.
 */

import {
  augmentedDynamics,
  odeFlow,
  type AugmentedDynamicsOptions,
  type AugmentedParts,
  type OdeFlowOptions,
} from 'aifn-compute/dynamics/ode'
import type { Stream } from 'aifn-compute/foundation/random'
import { treeFlatten, treeUnflatten, type Params } from 'aifn-compute/foundation/pytree'
import {
  avalOf,
  concat,
  get,
  mul,
  ones,
  reshape,
  shapeOfValue,
  slice,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Scalar } from 'aifn-compute/foundation/contracts'
import { childContext, tap, type Context, type Layer } from './layers'

/** Options of `OdeBlock`: the solver and gradient options of `odeFlow`, and the interval. */
export type OdeBlockOptions = OdeFlowOptions & {
  /** The interval $[t_0, t_1]$. Default `[0, 1]`. */
  interval?: readonly [Scalar, Scalar]
  /**
   * Append $t$ as a last input column of the field: $f(t, \xvec)$ rather than the autonomous $f(\xvec)$. Default
   * false.
   */
  timeDependent?: boolean
}

/** An ODE block: a layer, its field and its flow at any times. */
export interface OdeBlockLayer<P extends Params> extends Layer<P> {
  /** The vector field $f_\theta(t, \xvec)$ on a batch `[B, d]`; the inner layer runs at path `field`. */
  field(params: P, t: Scalar, x: Value, ctx?: Context): Value
  /**
   * The states $[\xvec(t_0), \xvec(t_1), \dots]$ at `times`, starting from `x` at the first time (with the block's
   * options, overridden by `options`).
   */
  flow(params: P, x: Value, times: readonly Scalar[], options?: OdeFlowOptions): Value[]
  /**
   * The flow with integrals appended (`augmentedDynamics`): the change in log density of a continuous normalising
   * flow, and the kinetic-energy and Jacobian-Frobenius regularisers of RNODE, at every time (each from 0 at
   * `times[0]`).
   */
  flowAugmented(
    params: P,
    x: Value,
    times: readonly Scalar[],
    augment: Omit<AugmentedDynamicsOptions, 'dim'>,
    options?: OdeFlowOptions,
  ): AugmentedParts[]
}

/**
 * Parameters as one vector and the map back, both written with primitives, so a gradient with respect to the vector
 * reaches every leaf (the adjoint differentiates one parameter tensor).
 *
 * @param params The parameter tree; its leaves are flattened in `treeFlatten` order and concatenated.
 * @returns `vector`, the concatenated leaves, and `unpack`, which slices a vector of that length back into a tree of
 *   the same structure and shapes (number leaves come back as rank-0 values).
 */
function packed<P>(params: P): { vector: Value; unpack: (v: Value) => P } {
  const { leaves, treedef } = treeFlatten(params)
  const shapes = leaves.map((leaf) => avalOf(leaf).shape)
  const sizes = shapes.map((s) => s.reduce((a, b) => a * b, 1))
  const vector = concat(leaves.map((leaf, i) => reshape(leaf, [sizes[i]])))
  const unpack = (v: Value): P => {
    let at = 0
    const rebuilt = leaves.map((leaf, i) => {
      const part = slice(v, [at, at + sizes[i]])
      at += sizes[i]
      return typeof leaf === 'number' ? get(part, 0) : reshape(part, shapes[i])
    })
    return treeUnflatten<P>(treedef, rebuilt)
  }
  return { vector, unpack }
}

/**
 * A layer $\xvec \mapsto \xvec(t_1)$ where $\xvec' = f_\theta(t, \xvec)$, $\xvec(t_0) = \xvec$. Its parameters are the
 * field's, and it is initialised as the field. With `gradient: 'adjoint'` the parameter tree is packed into one vector
 * for the adjoint method; with `'backprop'` (the default) the solver's steps are recorded. The field's input is
 * $\xvec$, or $[\xvec, t]$ with `timeDependent`. The layer also exposes the field, the flow at any times, and the flow
 * with the integrals of `augmentedDynamics` appended.
 *
 * @param f The field $f_\theta$: a layer from `[B, d]` (or `[B, d + 1]` with `timeDependent`) to `[B, d]`.
 * @param options The interval, whether the field sees the time, and the solver and gradient options of `odeFlow`.
 * @returns The block, an `OdeBlockLayer` whose `apply` gives $\xvec(t_1)$, `[B, d]`.
 *
 * @example The field $f(\xvec) = -\xvec$ decays the input by $e^{-1}$ over $[0, 1]$
 * const block = OdeBlock(ActivationLayer((v) => mul(-1, v)))
 * const x = tensor([[1, 2]])
 * print('x(1):', block.apply({}, x))
 * print('x e^-1:', mul(Math.exp(-1), x))
 * print('flow at 0, 0.5, 1:', block.flow({}, x, [0, 0.5, 1]))
 *
 * @example A learned linear field on a batch of three
 * const block = OdeBlock(Linear(2, 2), { stepSize: 0.25 })
 * const p = block.init(stream(0))
 * print(block.label, ' output shape:', block.apply(p, normals(stream(1), [3, 2])).shape)
 */
export function OdeBlock<P extends Params>(f: Layer<P>, options: OdeBlockOptions = {}): OdeBlockLayer<P> {
  const { interval = [0, 1], timeDependent = false, ...solver } = options
  const field = (params: P, t: Scalar, x: Value, ctx?: Context): Value => {
    const input = timeDependent ? concat([x, mul(t, ones([shapeOfValue(x)[0], 1]))], 1) : x
    return f.apply(params, input, childContext(ctx, 'field'))
  }
  /** Solve x′ = rhs(t, x, θ) with θ the field's parameters, packing them into one vector for the adjoint. */
  const solve = (
    params: P,
    rhs: (t: Scalar, y: Value, p: P) => Value,
    x: Value,
    times: readonly Scalar[],
    opts: OdeFlowOptions,
  ): Value[] => {
    if ((opts.gradient ?? 'backprop') === 'adjoint') {
      const { vector, unpack } = packed(params)
      return odeFlow((t, y, v) => rhs(t, y, unpack(v)), times, opts)(x, vector)
    }
    // Backprop closes over the parameter tree; the solver records its steps.
    return odeFlow((t, y) => rhs(t, y, params), times, opts)(x, 0)
  }
  const flow = (params: P, x: Value, times: readonly Scalar[], override: OdeFlowOptions = {}): Value[] =>
    solve(params, (t, y, p) => field(p, t, y), x, times, { ...solver, ...override })
  const flowAugmented = (
    params: P,
    x: Value,
    times: readonly Scalar[],
    augment: Omit<AugmentedDynamicsOptions, 'dim'>,
    override: OdeFlowOptions = {},
  ): AugmentedParts[] => {
    const aug = augmentedDynamics((t, y, p) => field(p as unknown as P, t, y), { ...augment, dim: shapeOfValue(x)[1] })
    const rhs = (t: Scalar, y: Value, p: P) => aug.rhs(t, y, p as unknown as Value)
    return solve(params, rhs, aug.pack(x), times, { ...solver, ...override }).map(aug.unpack)
  }
  return {
    kind: 'OdeBlock',
    label: `OdeBlock(${f.label}${timeDependent ? ', t' : ''})`,
    init: (s: Stream) => f.init(s),
    apply: (params, x, ctx) => tap(ctx, flow(params, x, interval).at(-1)!),
    field,
    flow,
    flowAugmented,
  }
}
