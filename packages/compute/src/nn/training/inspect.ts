/**
 * Looking inside a network: every layer's activation (its output, as tapped by the layer) and the gradient of a loss
 * with respect to each activation and each parameter, from a recording forward pass and then one forward and one
 * reverse pass. The activation gradients
 * come from zero "probes" added at each tap: the gradient with respect to a probe is the gradient with respect to the
 * activation it was added to (reverse-mode differentiation; Baydin, Pearlmutter, Radul and Siskind, 2018, "Automatic
 * differentiation in machine learning: a survey", JMLR 18).
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { add, toFlat, unwrap, zeros, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Context, Layer } from 'aifn-compute/nn/layers'
import type { Params } from 'aifn-compute/foundation/pytree'

/**
 * A value as plain data, with any tracing stripped.
 *
 * @param v A number, tensor or traced value.
 * @returns Its primal value, a tensor or a number.
 */
const raw = (v: Value): Tensor | number => unwrap(v)

/** The output and every tapped activation of a forward pass, keyed by layer path (e.g. `0`, `2`, `1.weights`). */
export type Activations = { output: Tensor | number; activations: Record<string, Tensor | number> }

/**
 * Run any forward pass written on a `Context` (a layer, or a model that is not one, such as a language model) and
 * record every activation it taps, in the order tapped. `ctx`'s own tap, if any, still sees each value. A path tapped
 * twice keeps its last value.
 *
 * @param forward The forward pass, given the context with the recording tap; returns the output.
 * @param ctx The context to run in (training mode, stream, buffers, and a tap of its own).
 * @returns The output and the tapped activations by path, as plain data.
 *
 * @example Record the intermediate values of a forward pass
 * const x = tensor([1, 2, 3])
 * const forward = (ctx) => {
 *   const h = ctx.tap('hidden', mul(2, x))
 *   return ctx.tap('total', sum(h))
 * }
 * print(recordActivations(forward))
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

/**
 * Run a layer and record every activation it taps; `recordActivations` of `layer.apply`.
 *
 * @param layer The layer (or model) to run.
 * @param params Its parameters.
 * @param x The input.
 * @param ctx The context to apply it in.
 * @returns The output and the tapped activations by path.
 *
 * @example The activations of a two-stage layer
 * const layer = {
 *   kind: 'Toy',
 *   label: 'Toy',
 *   init: () => ({ w: tensor([1, -2]) }),
 *   apply: (p, x, ctx = {}) => {
 *     const t = ctx.tap ?? ((path, v) => v)
 *     const h = t('0', mul(p.w, x))
 *     return t('1', sum(square(h)))
 *   },
 * }
 * print(activations(layer, layer.init(), tensor([3, 1])))
 */
export function activations<P extends Params>(layer: Layer<P>, params: P, x: Value, ctx: Context = {}): Activations {
  return recordActivations((c) => layer.apply(params, x, c), ctx)
}

/** The result of `inspect`. */
export type Inspection<P> = Activations & {
  /** The loss, `lossOf` of the output. */
  loss: number
  /** The gradient of the loss with respect to each tapped activation, by path. */
  activationGrads: Record<string, Tensor | number>
  /** The gradient of the loss with respect to the parameters $\thetavec$, with the structure of the parameters. */
  paramGrads: P
}

/**
 * Activations, the loss `lossOf(output)` and the gradients of the loss with respect to every activation and every
 * parameter, for figures of gradient flow (vanishing and exploding gradients, saliency). Attention weights are
 * tapped before they weight the values, so they get their gradient too. The layer runs twice: once to find its taps,
 * then under `valueAndGrad` with a zero probe added at each, which replaces `ctx`'s own tap.
 *
 * @param layer The layer (or model) to inspect.
 * @param params Its parameters, differentiated.
 * @param x The input, held fixed.
 * @param lossOf The loss of the output, a number or rank-0 value.
 * @param ctx The context to apply it in.
 * @returns The activations, the loss, and its gradients with respect to the activations and the parameters.
 *
 * @example Gradients of $\sum_i h_i^2$, $\hvec = \wvec \odot \xvec$: $2\hvec$, and $2\hvec \odot \xvec$ in $\wvec$
 * const layer = {
 *   kind: 'Toy',
 *   label: 'Toy',
 *   init: () => ({ w: tensor([1, -2]) }),
 *   apply: (p, x, ctx = {}) => {
 *     const t = ctx.tap ?? ((path, v) => v)
 *     const h = t('0', mul(p.w, x))
 *     return t('1', sum(square(h)))
 *   },
 * }
 * const r = inspect(layer, layer.init(), tensor([3, 1]), (y) => y)
 * print('loss:', r.loss, ' activations:', r.activations)
 * print('d loss / d activation:', r.activationGrads)
 * print('d loss / d w:', r.paramGrads)
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
