/**
 * Grad-CAM of a network built from `aifn-compute/nn/layers` (Selvaraju et al., 2017, "Grad-CAM: Visual Explanations
 * from Deep Networks via Gradient-based Localization", §3): the output $A$ of one tapped layer and the gradient of a
 * class score with respect to it, from `inspect`, turned into a map by the channel weights of
 * `aifn-compute/learning/explain` (`gradCamWeights`, or `gradCamPlusPlusWeights` for Grad-CAM++).
 */

import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Params } from 'aifn-compute/foundation/pytree'
import { mul, shapeOfValue, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  featureMaps,
  gradCamPlusPlusWeights,
  gradCamWeights,
  weightedMapSum,
  type FeatureMaps,
} from 'aifn-compute/learning/explain'
import type { Context, Layer } from 'aifn-compute/nn/layers'
import { inspect } from './inspect'

/** What `gradCam` returns. */
export type GradCamResult = {
  /** The map $\operatorname{ReLU}(\sum_k \alpha_k A_k)$, $h \times w$, row-major. */
  map: Float64Array
  /** The channel weights $\alpha_k$. */
  weights: Float64Array
  /** The tapped maps $A$. */
  maps: FeatureMaps
  /** The gradients $\partial y_c / \partial A$. */
  grads: FeatureMaps
  /** The network's output for the example, flat. */
  output: Float64Array
}

/** Options of `gradCam`. */
export type GradCamOptions = {
  /** The channel weights: `grad-cam` (the spatial mean of the gradient, default) or `grad-cam++`. */
  weighting?: 'grad-cam' | 'grad-cam++'
  /** The context to apply the network in. */
  ctx?: Context
}

/**
 * Grad-CAM of one example at the layer whose tap path is `path` (for a `Sequential`, the layer's index as a
 * string): the layer's output $A$ and the gradient of output `classIndex` with respect to it, by `inspect`, then the
 * channel weights and the ReLU of the weighted sum of the maps. The network's output must be `[1, classes]` or
 * `[classes]`; only its shape is read inside the loss, so it may be traced. Throws `AifnError` when no tensor was
 * tapped at `path`, and `ShapeError` when the tapped value is not one example's `[1, K, h, w]` or `[K, h, w]`.
 *
 * @param layer The network.
 * @param params Its parameters.
 * @param x One input, with the batch axis the network expects.
 * @param classIndex The class $c$ whose score $y_c$ is explained.
 * @param path The tap path of the layer whose output is mapped.
 * @param options The weighting (Grad-CAM or Grad-CAM++) and the context.
 * @returns The map, the weights, the maps and gradients, and the network's output.
 *
 * @example A small convolutional network: the map for each class
 * import { ActivationLayer, AvgPool2d, Conv2d, Flatten, Linear, Sequential } from 'aifn-compute/nn/layers'
 * const conv = Conv2d(1, 2, 3, { padding: 1 })
 * const net = Sequential(conv, ActivationLayer('relu'), AvgPool2d(4), Flatten(1), Linear(2, 2))
 * const params = net.init(stream(0))
 * const x = tensor([[[[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]]])
 * for (const c of [0, 1]) {
 *   const g = gradCam(net, params, x, c, '1')
 *   print(`class ${c}: weights`, g.weights, ', map', g.map)
 * }
 */
export function gradCam<P extends Params>(
  layer: Layer<P>,
  params: P,
  x: Value,
  classIndex: number,
  path: string,
  options: GradCamOptions = {},
): GradCamResult {
  const r = inspect(
    layer,
    params,
    x,
    (out) => {
      // The output may be traced here, so only its shape is read.
      const shape = shapeOfValue(out)
      const onehot = new Array<number>(shape[shape.length - 1]).fill(0)
      onehot[classIndex] = 1
      return sum(mul(out, tensor(shape.length === 2 ? [onehot] : onehot)))
    },
    options.ctx,
  )
  const a = r.activations[path]
  const g = r.activationGrads[path]
  if (a === undefined || g === undefined || typeof a === 'number' || typeof g === 'number')
    throw new AifnError('gradCam', `gradCam: no tensor activation at path ${path}`)
  const maps = featureMaps(a as Tensor)
  const grads = featureMaps(g as Tensor)
  if (maps.data.length !== grads.data.length)
    throw new ShapeError('gradCam', 'gradCam: the activation and its gradient differ in shape')
  const weights = options.weighting === 'grad-cam++' ? gradCamPlusPlusWeights(maps, grads) : gradCamWeights(grads)
  return {
    map: weightedMapSum(maps, weights, true),
    weights,
    maps,
    grads,
    output: Float64Array.from(toFlat(r.output as Tensor)),
  }
}
