/**
 * Looking inside a network: every layer's activation (its output, as tapped by the layer) and the gradient of a loss
 * with respect to each activation and each parameter, from one forward and one reverse pass. The activation gradients
 * come from zero "probes" added at each tap: the gradient with respect to a probe is the gradient with respect to the
 * activation it was added to (reverse-mode differentiation; Baydin, Pearlmutter, Radul and Siskind, 2018, "Automatic
 * differentiation in machine learning: a survey", JMLR 18).
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { add, toFlat, unwrap, zeros, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Context, Layer } from 'aifn-compute/nn/layers'
import type { Params } from 'aifn-compute/foundation/pytree'

const raw = (v: Value): Tensor | number => unwrap(v)

/** The output and every tapped activation of a forward pass, keyed by layer path (e.g. `0`, `2`, `1.weights`). */
export type Activations = { output: Tensor | number; activations: Record<string, Tensor | number> }

/**
 * Run any forward pass written on a `Context` (a layer, or a model that is not one, such as a language model) and
 * record every activation it taps, in the order tapped. `ctx`'s own tap, if any, still sees each value.
 */
export function recordActivations(forward: (ctx: Context) => Value, ctx: Context = {}): Activations {
  const seen: Record<string, Tensor | number> = {}
  const output = forward({
    ...ctx,
    tap: (path, v) => {
      seen[path] = raw(v)
      return ctx.tap ? ctx.tap(path, v) : v
    },
  })
  return { output: raw(output), activations: seen }
}

/** Run a layer and record every activation it taps. */
export function activations<P extends Params>(layer: Layer<P>, params: P, x: Value, ctx: Context = {}): Activations {
  return recordActivations((c) => layer.apply(params, x, c), ctx)
}

/** The result of `inspect`. */
export type Inspection<P> = Activations & {
  loss: number
  /** ∂loss/∂activation for each tapped path. */
  activationGrads: Record<string, Tensor | number>
  /** ∂loss/∂θ, with the structure of the parameters. */
  paramGrads: P
}

/**
 * Activations, the loss `lossOf(output)` and the gradients of the loss with respect to every activation and every
 * parameter, for figures of gradient flow (vanishing and exploding gradients, saliency). Attention weights are
 * tapped before they weight the values, so they get their gradient too.
 */
export function inspect<P extends Params>(
  layer: Layer<P>,
  params: P,
  x: Value,
  lossOf: (output: Value) => Value,
  ctx: Context = {},
): Inspection<P> {
  const first = activations(layer, params, x, ctx)
  const probes: Record<string, Tensor | number> = {}
  for (const [path, v] of Object.entries(first.activations)) probes[path] = typeof v === 'number' ? 0 : zeros(v.shape)
  const f = (p: P, pr: Record<string, Value>) =>
    lossOf(layer.apply(p, x, { ...ctx, tap: (path, v) => (path in pr ? add(v, pr[path]) : v) }))
  const { value, grad } = valueAndGrad(f, { argnums: [0, 1] })(params, probes)
  const [paramGrads, activationGrads] = grad as [P, Record<string, Tensor | number>]
  const loss = unwrap(value)
  return {
    ...first,
    loss: typeof loss === 'number' ? loss : toFlat(loss)[0],
    activationGrads,
    paramGrads,
  }
}
