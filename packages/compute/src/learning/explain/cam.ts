/**
 * Class activation maps of a convolutional network: the CAM of a network with global average pooling and a linear
 * head, the channel weights of Grad-CAM and Grad-CAM++, and the weighted sums that turn feature maps into one map.
 *
 * A CAM (Zhou, Khosla, Lapedriza, Oliva and Torralba, 2016, "Learning Deep Features for Discriminative Localization",
 * §2) needs a network whose last convolutional feature maps $A_k$ ($K$ maps of $h \times w$) are globally average
 * pooled and fed to a linear layer; the map of class $c$ is $M_c = \sum_k w_{kc} A_k$, with $w_{kc}$ that layer's
 * weight from feature $k$ to class $c$, and the class score is the mean of $M_c$ plus the class's bias. Grad-CAM
 * (Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", §3,
 * eqs. 1 and 2) works for any differentiable head:
 *
 * $$
 * \alpha_{kc} = \frac{1}{hw} \sum_{ij} \frac{\partial y_c}{\partial A_{k,ij}}, \qquad
 * M_c = \operatorname{ReLU}\Big(\sum_k \alpha_{kc} A_k\Big).
 * $$
 *
 * Grad-CAM++ (Chattopadhyay, Sarkar, Howlader and Balasubramanian, 2018, eq. 19, under their exponential-output
 * assumption) weights the positive gradients pixel by pixel, $w_{kc} = \sum_{ij} a_{ij} \operatorname{ReLU}(g_{ij})$
 * with $a_{ij} = g_{ij}^2 / (2g_{ij}^2 + \sum_{ab} A_{ab} g_{ij}^3)$. Maps are `FeatureMaps` records and results
 * plain `Float64Array`s, as the rest of this module. `gradCam` of `aifn-compute/nn/training` taps a layer and its
 * gradient and applies these weights; `upsampleBilinear` of `aifn-compute/signal/image` lays a map over its image.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** $K$ feature maps of $h \times w$ as one flat array, channel-major: value $(k, i, j)$ at $khw + iw + j$. */
export type FeatureMaps = {
  /** The values, $Khw$ of them, channel-major. */
  data: Float64Array
  /** The number of maps $K$. */
  channels: number
  /** Rows $h$ of each map. */
  height: number
  /** Columns $w$ of each map. */
  width: number
}

/**
 * Feature maps from the activations of one example, a tensor of shape `[1, K, h, w]` or `[K, h, w]`. Throws
 * `ShapeError` for another shape, or a batch of more than one.
 *
 * @param t The activations of one example.
 * @returns The maps, with a copy of the values.
 *
 * @example Two 2 x 2 maps
 * const m = featureMaps(tensor([[[[1, 2], [3, 4]], [[0, 0], [1, 1]]]]))
 * print('channels, height, width =', m.channels, m.height, m.width)
 * print('data =', m.data)
 */
export function featureMaps(t: Tensor): FeatureMaps {
  const s = t.shape
  if (s.length !== 3 && s.length !== 4)
    throw new ShapeError('featureMaps', 'featureMaps: shape must be [1, K, h, w] or [K, h, w]', [s])
  if (s.length === 4 && s[0] !== 1) throw new ShapeError('featureMaps', 'featureMaps: one example only', [s])
  const [channels, height, width] = s.length === 4 ? s.slice(1) : s
  return { data: Float64Array.from(toFlat(t)), channels, height, width }
}

/**
 * The weighted sum $\sum_k \lambda_k A_k$ of the maps, optionally through ReLU. Throws `ShapeError` unless there is
 * one weight per map.
 *
 * @param maps The $K$ maps.
 * @param weights One weight $\lambda_k$ per map ($K$ values).
 * @param relu Whether to clip negative values to zero.
 * @returns The $h \times w$ map, row-major.
 *
 * @example The second map minus the first, with and without ReLU
 * const m = { data: Float64Array.of(1, 0, 0, 1, 0, 1, 1, 0), channels: 2, height: 2, width: 2 }
 * print('sum  =', weightedMapSum(m, [-1, 1]))
 * print('relu =', weightedMapSum(m, [-1, 1], true))
 */
export function weightedMapSum(maps: FeatureMaps, weights: ArrayLike<number>, relu = false): Float64Array {
  const { data, channels: K, height, width } = maps
  if (weights.length !== K)
    throw new ShapeError('weightedMapSum', `weightedMapSum: ${weights.length} weights for ${K} maps`)
  const hw = height * width
  const out = new Float64Array(hw)
  for (let k = 0; k < K; k++) for (let p = 0; p < hw; p++) out[p] += weights[k] * data[k * hw + p]
  if (relu) for (let p = 0; p < hw; p++) out[p] = Math.max(0, out[p])
  return out
}

/**
 * The class activation map $M_c = \sum_k w_{kc} A_k$ (no ReLU) of Zhou et al. (2016): the classifier's weights for
 * one class applied at every position. The class score is the mean of $M_c$ over positions plus the class's bias.
 *
 * @param maps The last convolutional layer's maps, as fed to global average pooling.
 * @param classWeights The linear layer's weights from each feature to the class ($K$ values).
 * @returns The $h \times w$ map, row-major.
 *
 * @example Two 3 x 3 maps and a class that likes the first and dislikes the second
 * const data = Float64Array.of(2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2)
 * const m = { data, channels: 2, height: 3, width: 3 }
 * const cam = classActivationMap(m, [1, -1])
 * print('CAM =', cam)
 * print('score - bias = mean =', cam.reduce((a, b) => a + b) / 9)
 */
export function classActivationMap(maps: FeatureMaps, classWeights: ArrayLike<number>): Float64Array {
  return weightedMapSum(maps, classWeights, false)
}

/**
 * Grad-CAM's channel weights $\alpha_k$: the spatial mean of the gradient of the class score with respect to each
 * map (Selvaraju et al., 2017, eq. 1). For a global-average-pooling and linear head they are the classifier's
 * weights divided by $hw$.
 *
 * @param grads The gradients $\partial y_c / \partial A$, shaped like the maps.
 * @returns One weight per map ($K$ values).
 *
 * @example Constant gradients give their value back
 * print(gradCamWeights({ data: Float64Array.of(1, 1, 1, 1, -2, -2, -2, -2), channels: 2, height: 2, width: 2 }))
 */
export function gradCamWeights(grads: FeatureMaps): Float64Array {
  const { data, channels: K, height, width } = grads
  const hw = height * width
  const out = new Float64Array(K)
  for (let k = 0; k < K; k++) {
    let s = 0
    for (let p = 0; p < hw; p++) s += data[k * hw + p]
    out[k] = s / hw
  }
  return out
}

/**
 * Grad-CAM++'s channel weights under the exponential-score assumption (Chattopadhyay et al., 2018, eq. 19): the
 * positive gradients weighted per position by $a_{ij} = g_{ij}^2 / (2g_{ij}^2 + \sum_{ab} A_{ab} g_{ij}^3)$ (0 where
 * the denominator is 0).
 *
 * @param maps The maps $A$.
 * @param grads The gradients $\partial y_c / \partial A$, shaped like the maps.
 * @returns One weight per map ($K$ values).
 *
 * @example Positive gradients count, negative ones do not
 * const maps = { data: Float64Array.of(1, 1, 0, 0, 1, 1, 0, 0), channels: 2, height: 2, width: 2 }
 * const grads = { data: Float64Array.of(1, 1, 1, 1, -1, -1, -1, -1), channels: 2, height: 2, width: 2 }
 * print(gradCamPlusPlusWeights(maps, grads))
 */
export function gradCamPlusPlusWeights(maps: FeatureMaps, grads: FeatureMaps): Float64Array {
  const { channels: K, height, width } = maps
  const hw = height * width
  const out = new Float64Array(K)
  for (let k = 0; k < K; k++) {
    let sumA = 0
    for (let p = 0; p < hw; p++) sumA += maps.data[k * hw + p]
    let w = 0
    for (let p = 0; p < hw; p++) {
      const g = grads.data[k * hw + p]
      const denom = 2 * g * g + sumA * g * g * g
      const a = denom !== 0 ? (g * g) / denom : 0
      w += a * Math.max(0, g)
    }
    out[k] = w
  }
  return out
}

/**
 * A map divided by its largest absolute value, so it spans at most $[-1, 1]$ (all zeros when the map is zero).
 *
 * @param map The map, any array of numbers; not modified.
 * @returns The scaled copy.
 *
 * @example Scaled by the largest magnitude
 * print(normaliseMap([2, -4, 1]))
 */
export function normaliseMap(map: ArrayLike<number>): Float64Array {
  let m = 0
  for (let i = 0; i < map.length; i++) m = Math.max(m, Math.abs(map[i]))
  const out = Float64Array.from(map)
  if (m > 0) for (let i = 0; i < out.length; i++) out[i] /= m
  return out
}
