/**
 * The mixture density network (Bishop, 1994, "Mixture density networks", NCRG/94/004; Bishop, 2006, "Pattern
 * Recognition and Machine Learning", §5.6): an MLP whose outputs parametrise a mixture of $K$ Gaussians over the
 * target, $p(\yvec \mid \xvec) = \sum_k \pi_k(\xvec) \Gauss(\yvec; \muvec_k(\xvec), \diag \sigmavec_k(\xvec)^2)$,
 * fitted by maximum likelihood with `mixtureDensityNll` of `aifn-compute/learning/losses`. Where $\yvec$ given
 * $\xvec$ is multimodal (an inverse problem), the mixture puts a component on each branch and the mixing weights
 * switch components on and off along $\xvec$.
 *
 * The same body with $D$ outputs and the squared error (`objective: 'squared-error'`) is the usual regression network,
 * whose population minimiser is the conditional mean $\expect[\yvec \mid \xvec]$: on an inverse problem it averages
 * the branches and can predict a $\yvec$ that solves nothing.
 *
 * Inputs are $n \times d$ matrices (a vector is one column, by `inputMatrix`) and targets $n$ values or $n \times D$.
 * The network sees standardised inputs, $(x_j - \text{shift}_j) / \text{scale}_j$ per column.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { div, fromData, reshape, sub, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  meanSquaredErrorLoss,
  mixtureDensityHead,
  mixtureDensityNll,
  mixtureHeadSize,
  type MixtureDensity,
  type ScaleLink,
} from 'aifn-compute/learning/losses'
import type { Activation } from 'aifn-compute/nn/functional'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import type { Params } from 'aifn-compute/foundation/pytree'
import type { Stream } from 'aifn-compute/foundation/random'

/** What the network is trained for: the mixture's likelihood, or the squared error of a point prediction. */
export type MdnObjective = 'mixture' | 'squared-error'

/** A mixture density network's structure. */
export type MdnConfig = {
  /** Input width $d$. */
  inputs: Size
  /** Target dimension $D$ (default 1). */
  outputs?: Size
  /** Mixture components $K$ (default 3; ignored for the squared error). */
  components?: Size
  /** Hidden layer widths (default [20], Bishop's single layer of 20 units). */
  hidden?: readonly Size[]
  /** Hidden activation (default tanh). */
  activation?: Activation
  /** What the network is trained for (default `'mixture'`). */
  objective?: MdnObjective
  /**
   * How a raw output $s$ becomes a standard deviation: $\sigma = \text{floor} + e^s$ (`'exp'`, default) or
   * $\text{floor} + \operatorname{softplus}(s)$.
   */
  scale?: ScaleLink
  /** The floor on every $\sigma$ (default 1e-3). */
  floor?: number
  /**
   * Inputs enter the network as $(x_j - \text{shift}_j) / \text{scale}_j$, per column (default no shift).
   * `inputStandardisation` gives the training data's mean and standard deviation: uncentred inputs leave tanh units
   * nearly linear and the squared-error network stuck on the linear fit.
   */
  inputShift?: readonly number[]
  /** The divisor of each input column, as `inputShift` (default 1). */
  inputScale?: readonly number[]
}

/** A config with every default filled in. */
export type MdnSpec = Required<MdnConfig>

/**
 * Fill in the defaults of a config: one output, 3 components, one hidden layer of 20 tanh units, the mixture
 * objective, the `'exp'` scale link with floor 1e-3, and unstandardised inputs.
 *
 * @param config The structure; its fields override the defaults.
 * @returns The config with every field set.
 *
 * @example The defaults for two inputs
 * print(mdnSpec({ inputs: 2 }))
 */
export function mdnSpec(config: MdnConfig): MdnSpec {
  return {
    outputs: 1,
    components: 3,
    hidden: [20],
    activation: 'tanh',
    objective: 'mixture',
    scale: 'exp',
    floor: 1e-3,
    inputShift: new Array<number>(config.inputs).fill(0),
    inputScale: new Array<number>(config.inputs).fill(1),
    ...config,
  }
}

/**
 * The mean and standard deviation (divided by $n$) of each input column, for `inputShift` and `inputScale`; a
 * constant column gets scale 1.
 *
 * @param x The training inputs, $n \times d$, or a vector of $n$ values (one column).
 * @returns `inputShift`, the $d$ column means, and `inputScale`, the $d$ standard deviations.
 *
 * @example A varying column and a constant one
 * print(inputStandardisation(tensor([[1, 10], [3, 10]])))
 */
export function inputStandardisation(x: Tensor): { inputShift: number[]; inputScale: number[] } {
  const m = inputMatrix(x)
  const [n, d] = m.shape
  const v = toFlat(m)
  const inputShift: number[] = []
  const inputScale: number[] = []
  for (let j = 0; j < d; j++) {
    let mean = 0
    for (let i = 0; i < n; i++) mean += v[i * d + j] / n
    let ss = 0
    for (let i = 0; i < n; i++) ss += (v[i * d + j] - mean) ** 2
    inputShift.push(mean)
    inputScale.push(Math.sqrt(ss / n) || 1)
  }
  return { inputShift, inputScale }
}

/**
 * Inputs as the network sees them: $(x_j - \text{shift}_j) / \text{scale}_j$ per column. The shift and scale are
 * constants, never differentiated; `x` may be traced.
 *
 * @param spec The network's spec, whose `inputShift` and `inputScale` are used.
 * @param x The inputs, $n \times d$.
 * @returns The standardised inputs, $n \times d$.
 *
 * @example Shift 2, scale 0.5
 * const spec = mdnSpec({ inputs: 1, inputShift: [2], inputScale: [0.5] })
 * print(standardisedInputs(spec, tensor([[1], [2], [3]])))
 */
export function standardisedInputs(spec: MdnSpec, x: Value): Value {
  const shift = fromData(Float64Array.from(spec.inputShift), [1, spec.inputs])
  const scale = fromData(Float64Array.from(spec.inputScale), [1, spec.inputs])
  return div(sub(x, shift), scale)
}

/** A built network: the MLP and its spec. */
export type MdnModel = {
  /** The config the network was built from, with every default filled in. */
  readonly spec: MdnSpec
  /** The MLP, from the standardised inputs to the raw head (or the point prediction). */
  readonly net: Layer<Params[]>
  /** Fresh parameters (Xavier-uniform weights) drawn from the stream `s`. */
  init(s: Stream): Params[]
}

/**
 * Build the network of a config: an MLP with layer widths $d$, the hidden widths, then $K(1 + 2D)$ for the mixture
 * (a weight, $D$ means and $D$ scales per component) or $D$ for the squared error, with Xavier-uniform weights.
 *
 * @param config The structure; missing fields take the defaults of `mdnSpec`.
 * @returns The network, its spec and its initialiser.
 *
 * @example An untrained two-component network: its raw head for one input
 * const model = mdnModel({ inputs: 1, components: 2, hidden: [4] })
 * const params = model.init(stream(0))
 * print('head width =', model.net.apply(params, tensor([[1]])).shape[1])
 * print('raw head =', model.net.apply(params, tensor([[1]])))
 */
export function mdnModel(config: MdnConfig): MdnModel {
  const spec = mdnSpec(config)
  const width = spec.objective === 'mixture' ? mixtureHeadSize(spec.components, spec.outputs) : spec.outputs
  const net = Mlp([spec.inputs, ...spec.hidden, width], { activation: spec.activation, init: xavierUniform() })
  return { spec, net, init: (s) => net.init(s) }
}

/**
 * The options of the mixture head of a spec, for `mixtureDensityNll` and `mixtureDensityHead`.
 *
 * @param spec The network's spec.
 * @returns Its `components`, `dims` (the target dimension), `scale` link and `floor`.
 */
const headOptions = (spec: MdnSpec) => ({
  components: spec.components,
  dims: spec.outputs,
  scale: spec.scale,
  floor: spec.floor,
})

/**
 * The training loss: the mixture's negative log-likelihood averaged over rows, or, for the squared-error network, the
 * squared error averaged over rows and coordinates. Differentiable in `params`.
 *
 * @param model The network.
 * @param params Its parameters, possibly traced.
 * @param x The inputs, $n \times d$ (not yet standardised: the network does it).
 * @param y The targets, $n$ values or $n \times D$.
 * @returns The scalar loss.
 *
 * @example The losses of two untrained networks
 * const x = tensor([[0], [0.5], [1]])
 * const y = tensor([0, 1, 2])
 * const mixture = mdnModel({ inputs: 1, components: 2, hidden: [4] })
 * print('mixture NLL =', mdnLoss(mixture, mixture.init(stream(0)), x, y))
 * const mean = mdnModel({ inputs: 1, hidden: [4], objective: 'squared-error' })
 * print('squared error =', mdnLoss(mean, mean.init(stream(0)), x, y))
 */
export function mdnLoss(model: MdnModel, params: Params[], x: Value, y: Tensor): Value {
  const out = model.net.apply(params, standardisedInputs(model.spec, x))
  if (model.spec.objective === 'mixture') return mixtureDensityNll(out, y, headOptions(model.spec))
  return meanSquaredErrorLoss(out, reshape(y, [y.shape[0], model.spec.outputs]))
}

/** A prediction on a batch, as numbers. */
export type MdnPrediction = {
  /**
   * $\expect[\yvec \mid \xvec]$ per row, $n \times D$ row-major: the mixture's mean, or the squared-error network's
   * output.
   */
  readonly mean: Float64Array
  /** The mixture per row (null for the squared-error network). */
  readonly mixture: MixtureDensity | null
}

/**
 * Evaluate the network on a batch, as plain numbers (not differentiable).
 *
 * @param model The network.
 * @param params Its parameters.
 * @param x The inputs, $n \times d$ (not yet standardised).
 * @returns The conditional mean of each row, and the mixture (null for the squared-error network).
 *
 * @example An untrained network's mixtures for two inputs
 * const model = mdnModel({ inputs: 1, components: 2, hidden: [4] })
 * const prediction = mdnPredict(model, model.init(stream(0)), tensor([[0], [1]]))
 * print('mean =', prediction.mean)
 * print('weights =', prediction.mixture.weights)
 * print('scales =', prediction.mixture.scales)
 */
export function mdnPredict(model: MdnModel, params: Params[], x: Tensor): MdnPrediction {
  const out = model.net.apply(params, standardisedInputs(model.spec, x))
  if (model.spec.objective === 'squared-error') {
    const r = unwrap(out) as Tensor
    return { mean: Float64Array.from(toFlat(r)), mixture: null }
  }
  const mixture = mixtureDensityHead(out, headOptions(model.spec))
  return { mean: mixture.mean(), mixture }
}

/**
 * The average log-likelihood per row of targets under a prediction: the mixture's, or, for the squared-error network,
 * a Gaussian around its output with the variance $\sigma^2$ of its residuals on the same data (the maximum-likelihood
 * unimodal model with one shared $\sigma$ per coordinate, floored at $\sigma^2 = 10^{-12}$).
 *
 * @param model The network that made the prediction (its target dimension $D$ is read).
 * @param prediction The prediction on the rows of `y`, from `mdnPredict`.
 * @param y The targets, $n$ values or $n \times D$.
 * @returns The mean of $\log p(\yvec_i \mid \xvec_i)$ over the rows.
 *
 * @example For the mixture, minus the training loss
 * const model = mdnModel({ inputs: 1, components: 2, hidden: [4] })
 * const params = model.init(stream(0))
 * const x = tensor([[0], [0.5], [1]])
 * const y = tensor([0, 1, 2])
 * print('log-likelihood per row =', mdnLogLikelihood(model, mdnPredict(model, params, x), y))
 * print('minus the loss =', mul(mdnLoss(model, params, x, y), -1))
 */
export function mdnLogLikelihood(model: MdnModel, prediction: MdnPrediction, y: Tensor): number {
  const D = model.spec.outputs
  const ys = Float64Array.from(toFlat(y))
  const n = ys.length / D
  if (prediction.mixture) {
    let acc = 0
    for (let i = 0; i < n; i++) acc += prediction.mixture.logDensity(i, ys.subarray(i * D, (i + 1) * D))
    return acc / n
  }
  let acc = 0
  for (let j = 0; j < D; j++) {
    let ss = 0
    for (let i = 0; i < n; i++) ss += (ys[i * D + j] - prediction.mean[i * D + j]) ** 2
    const v = Math.max(1e-12, ss / n)
    acc += -0.5 * Math.log(2 * Math.PI * v) - 0.5
  }
  return acc
}

/**
 * The mean squared error of a prediction's mean against targets, averaged over rows and coordinates.
 *
 * @param prediction The prediction, whose `mean` is compared.
 * @param y The targets, as many values as the prediction's mean, in the same row-major order.
 * @returns The mean of the squared differences.
 *
 * @example Errors 0, 0 and 2
 * const prediction = { mean: Float64Array.of(1, 2, 3), mixture: null }
 * print(mdnMeanSquaredError(prediction, tensor([1, 2, 5])))
 */
export function mdnMeanSquaredError(prediction: MdnPrediction, y: Tensor): number {
  const ys = toFlat(y)
  let ss = 0
  for (let i = 0; i < ys.length; i++) ss += (ys[i] - prediction.mean[i]) ** 2
  return ss / ys.length
}

/**
 * Inputs as a matrix: a vector of $n$ values becomes an $n \times 1$ float64 matrix; anything else is returned as it
 * is.
 *
 * @param x The inputs, $n$ values or $n \times d$.
 * @returns The $n \times d$ inputs.
 *
 * @example A vector becomes a column
 * print(inputMatrix(tensor([1, 2, 3])))
 */
export function inputMatrix(x: Tensor): Tensor {
  return x.shape.length === 1 ? fromData(Float64Array.from(toFlat(x)), [x.shape[0], 1]) : x
}
