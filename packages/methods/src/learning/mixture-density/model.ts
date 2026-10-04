/**
 * The mixture density network (Bishop, 1994, "Mixture density networks", NCRG/94/004; Bishop, 2006, "Pattern
 * Recognition and Machine Learning", §5.6): an MLP whose outputs parametrise a mixture of K Gaussians over the target,
 * p(y | x) = Σₖ πₖ(x) N(y; μₖ(x), diag σₖ(x)²), fitted by maximum likelihood with `mixtureDensityNll` of
 * `aifn-compute/learning/losses`. Where y given x is multimodal (an inverse problem), the mixture puts a component on each
 * branch and the mixing weights switch components on and off along x.
 *
 * The same body with D outputs and the squared error (`objective: 'squared-error'`) is the usual regression network,
 * whose population minimiser is the conditional mean E[y | x]: on an inverse problem it averages the branches and can
 * predict a y that solves nothing.
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
  /** Input width d. */
  inputs: Size
  /** Target dimension D (default 1). */
  outputs?: Size
  /** Mixture components K (default 3; ignored for the squared error). */
  components?: Size
  /** Hidden layer widths (default [20], Bishop's single layer of 20 units). */
  hidden?: readonly Size[]
  /** Hidden activation (default tanh). */
  activation?: Activation
  /** Default `mixture`. */
  objective?: MdnObjective
  /** σ = floor + exp(s) (default) or floor + softplus(s). */
  scale?: ScaleLink
  /** The floor on every σ (default 1e-3). */
  floor?: number
  /**
   * Inputs enter the network as (x − shift)/scale, per column (default none). `inputStandardisation` gives the
   * training data's mean and sd: uncentred inputs leave tanh units nearly linear and the squared-error network stuck on
   * the linear fit.
   */
  inputShift?: readonly number[]
  inputScale?: readonly number[]
}

/** A config with every default filled in. */
export type MdnSpec = Required<MdnConfig>

/** Fill in the defaults of a config. */
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

/** The mean and sd of each input column (sd 1 for a constant column), for `inputShift` and `inputScale`. */
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

/** Inputs as the network sees them: (x − shift)/scale per column (constants: never differentiated). */
export function standardisedInputs(spec: MdnSpec, x: Value): Value {
  const shift = fromData(Float64Array.from(spec.inputShift), [1, spec.inputs])
  const scale = fromData(Float64Array.from(spec.inputScale), [1, spec.inputs])
  return div(sub(x, shift), scale)
}

/** A built network: the MLP, its spec and the mixture head's options. */
export type MdnModel = {
  readonly spec: MdnSpec
  readonly net: Layer<Params[]>
  init(s: Stream): Params[]
}

/** Build the network of a config: an MLP [d, hidden…, K(1 + 2D)] (mixture) or [d, hidden…, D] (squared error). */
export function mdnModel(config: MdnConfig): MdnModel {
  const spec = mdnSpec(config)
  const width = spec.objective === 'mixture' ? mixtureHeadSize(spec.components, spec.outputs) : spec.outputs
  const net = Mlp([spec.inputs, ...spec.hidden, width], { activation: spec.activation, init: xavierUniform() })
  return { spec, net, init: (s) => net.init(s) }
}

const headOptions = (spec: MdnSpec) => ({
  components: spec.components,
  dims: spec.outputs,
  scale: spec.scale,
  floor: spec.floor,
})

/** The training loss on (x [n, d], y [n] or [n, D]): the mixture NLL, or the squared error averaged over coordinates. */
export function mdnLoss(model: MdnModel, params: Params[], x: Value, y: Tensor): Value {
  const out = model.net.apply(params, standardisedInputs(model.spec, x))
  if (model.spec.objective === 'mixture') return mixtureDensityNll(out, y, headOptions(model.spec))
  return meanSquaredErrorLoss(out, reshape(y, [y.shape[0], model.spec.outputs]))
}

/** A prediction on a batch, as numbers. */
export type MdnPrediction = {
  /** E[y | x] per row, row-major [n, D]: the mixture's mean, or the squared-error network's output. */
  readonly mean: Float64Array
  /** The mixture per row (null for the squared-error network). */
  readonly mixture: MixtureDensity | null
}

/** Evaluate the network on x [n, d]. */
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
 * a Gaussian around its output with the variance σ² of its residuals on the same data (the maximum-likelihood
 * unimodal model with one shared σ per coordinate).
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

/** The mean squared error per coordinate of a prediction's mean against targets. */
export function mdnMeanSquaredError(prediction: MdnPrediction, y: Tensor): number {
  const ys = toFlat(y)
  let ss = 0
  for (let i = 0; i < ys.length; i++) ss += (ys[i] - prediction.mean[i]) ** 2
  return ss / ys.length
}

/** Inputs as a float64 matrix [n, d] (a vector becomes one column). */
export function inputMatrix(x: Tensor): Tensor {
  return x.shape.length === 1 ? fromData(Float64Array.from(toFlat(x)), [x.shape[0], 1]) : x
}
