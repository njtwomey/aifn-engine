/**
 * A small learned noise predictor: an MLP on the noised point and features of its noise level, trained with the
 * simple DDPM objective $\expect \lVert \epsilonvec - \hat\epsilonvec_{\thetavec}(\xvec_t, \bar\alpha_t) \rVert^2$,
 * with $\xvec_t = \sqrt{\bar\alpha_t}\,\xvec_0 + \sqrt{1 - \bar\alpha_t}\,\epsilonvec$ and $t$ uniform on
 * $1, \dots, T$ (Ho et al., 2020, eq. 14 and Algorithm 1). Enough to learn 1-d and 2-d toy data in the browser; the
 * network and its training loop come from `aifn-compute/nn`, the loss from `aifn-compute/learning/losses`.
 */

import { meanSquaredErrorLoss } from 'aifn-compute/learning/losses'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { type Activation } from 'aifn-compute/nn/functional'
import { type Params } from 'aifn-compute/foundation/pytree'
import { normals, child, integers } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { NoisePredictor } from './predictor'
import { alphaBarAt, type NoiseSchedule } from './schedules'

/** A noise-prediction network: an MLP from the point and its noise-level features to $\hat\epsilonvec$. */
export type Denoiser = {
  /** The MLP, from $d + 1 + 2K$ inputs to $d$ outputs; its parameters are kept apart (`layer.init`). */
  readonly layer: Layer<Params[]>
  /** The data's dimension $d$. */
  readonly dimension: number
  /** Frequencies $K$ of the noise-level features ($1 + 2K$ features). */
  readonly frequencies: number
}

/** Options of `denoiser`. */
export type DenoiserOptions = {
  /** Hidden layer widths (default `[64, 64]`). */
  hidden?: readonly number[]
  /** Frequencies $K$ of the noise-level features (default 4). */
  frequencies?: number
  /** Hidden activation (default SiLU, as in DDPM's U-Net). */
  activation?: Activation
}

/**
 * A noise-prediction MLP for $d$-dimensional data, without parameters (`net.layer.init(stream)` draws them).
 *
 * @param dimension The data's dimension $d$, the network's output width.
 * @param options The network's shape.
 * @param options.hidden The hidden layers' widths.
 * @param options.frequencies The number $K$ of sinusoid frequencies in the noise-level features.
 * @param options.activation The hidden layers' activation.
 * @returns The network.
 *
 * @example A network with one hidden layer of 16 for 2-d data: $2 + 1 + 2 \cdot 3 = 9$ inputs
 * const net = denoiser(2, { hidden: [16], frequencies: 3 })
 * const params = net.layer.init(stream(1))
 * print('weight shapes:', params[0].weight.shape, params[2].weight.shape)
 */
export function denoiser(
  dimension: number,
  { hidden = [64, 64], frequencies = 4, activation = 'silu' }: DenoiserOptions = {},
): Denoiser {
  const layer = Mlp([dimension + 1 + 2 * frequencies, ...hidden, dimension], { activation })
  return { layer, dimension, frequencies }
}

/**
 * The features of a noise level: the noise scale $\sigma = \sqrt{1 - \bar\alpha}$ and $\sin(k\pi\sigma)$,
 * $\cos(k\pi\sigma)$ for $k = 1, \dots, K$, a sinusoidal embedding (as Vaswani et al., 2017, for positions) of a
 * bounded quantity that orders the levels.
 *
 * @param alphaBar The signal level $\bar\alpha \in [0, 1]$.
 * @param frequencies The number of frequencies $K$.
 * @returns The $1 + 2K$ features, in the order $\sigma, \sin \pi\sigma, \cos \pi\sigma, \sin 2\pi\sigma, \dots$.
 *
 * @example At $\bar\alpha = 0.75$, $\sigma = 0.5$
 * print(noiseLevelFeatures(0.75, 2))
 */
export function noiseLevelFeatures(alphaBar: number, frequencies: number): Tensor {
  const f = levelFeatures(alphaBar, frequencies)
  return fromData(Float64Array.from(f), [f.length])
}

/**
 * The features of `noiseLevelFeatures` as a plain array.
 *
 * @param alphaBar The signal level $\bar\alpha \in [0, 1]$.
 * @param frequencies The number of frequencies $K$.
 * @returns The $1 + 2K$ features.
 */
function levelFeatures(alphaBar: number, frequencies: number): number[] {
  const sigma = Math.sqrt(1 - alphaBar)
  const out = [sigma]
  for (let k = 1; k <= frequencies; k++) out.push(Math.sin(k * Math.PI * sigma), Math.cos(k * Math.PI * sigma))
  return out
}

/**
 * The network's input: each point followed by the features of its noise level, $[\xvec_i, \phi(\bar\alpha_i)]$.
 *
 * @param net The network, for its $K$.
 * @param x The noised points, $[n, d]$.
 * @param alphaBars The signal level of each row ($n$ values), or one number for every row.
 * @returns The input, $[n, d + 1 + 2K]$.
 *
 * @example Two points at different noise levels, with $K = 1$
 * const net = denoiser(2, { frequencies: 1 })
 * print(denoiserInput(net, tensor([[1, 2], [3, 4]]), [0.75, 0]))
 */
export function denoiserInput(net: Denoiser, x: Tensor, alphaBars: number | ArrayLike<number>): Tensor {
  const [n, d] = x.shape
  const width = d + 1 + 2 * net.frequencies
  const xs = toFlat(x)
  const out = new Float64Array(n * width)
  const shared = typeof alphaBars === 'number' ? levelFeatures(alphaBars, net.frequencies) : null
  for (let i = 0; i < n; i++) {
    const f = shared ?? levelFeatures((alphaBars as ArrayLike<number>)[i], net.frequencies)
    for (let a = 0; a < d; a++) out[i * width + a] = xs[i * d + a]
    for (let j = 0; j < f.length; j++) out[i * width + d + j] = f[j]
  }
  return fromData(out, [n, width])
}

/**
 * The network, under given parameters, as a noise predictor for the samplers.
 *
 * @param net The network.
 * @param params Its parameters, as `net.layer.init` or `denoiserTraining` give them.
 * @returns The predictor: points $[n, d]$ and one level $\bar\alpha$ to $\hat\epsilonvec$, $[n, d]$.
 *
 * @example An untrained network's prediction for two points
 * const net = denoiser(2, { hidden: [16] })
 * const predictor = networkNoisePredictor(net, net.layer.init(stream(1)))
 * print(predictor(tensor([[1, 2], [0, 0]]), 0.5))
 */
export function networkNoisePredictor(net: Denoiser, params: Params[]): NoisePredictor {
  return (x, alphaBar) => unwrap(net.layer.apply(params, denoiserInput(net, x, alphaBar))) as Tensor
}

/** Options of `denoiserTraining`. */
export type DenoiserTrainingOptions = {
  /** Training points $\xvec_0$, shape $[n, d]$. */
  data: Tensor
  /** The noise schedule whose levels $\bar\alpha_t$ are drawn. */
  schedule: NoiseSchedule
  /** The network trained. */
  net: Denoiser
  /** Points per step (default 128). */
  batchSize?: number
  /** The update rule. Default `adamRule({ stepSize: 3e-3 })`. */
  optimizer?: UpdateRule<unknown>
}

/**
 * Training of a noise predictor as a traceable algorithm (`aifn-compute/nn`'s `trainingLoop`): each step takes a
 * minibatch of data, a level $t \sim \mathcal{U}\{1, \dots, T\}$ and noise $\epsilonvec \sim \Gauss(\zeros, \Imat)$ per
 * point, all from the step's stream, and takes one optimiser step on the mean squared error between
 * $\hat\epsilonvec$ and $\epsilonvec$. `init` takes `{ params: net.layer.init(stream) }`.
 *
 * @param options The data, the schedule, the network, the batch size and the update rule.
 * @returns The algorithm; its state's `params` go to `networkNoisePredictor`, and `loss` is the minibatch loss.
 *
 * @example A few steps on two clusters: the loss falls
 * const s = stream(1)
 * const data = sampleMixture(s, gaussianMixtureData([1, 1], [[-2, 0], [2, 0]], [0.3, 0.3]), 256)
 * const net = denoiser(2, { hidden: [16] })
 * const training = denoiserTraining({ data, schedule: linearSchedule(100), net, batchSize: 64 })
 * const start = { params: net.layer.init(s) }
 * for (const steps of [0, 20, 100]) print('after', steps, 'steps: loss', run(training, start, steps).loss)
 */
export function denoiserTraining(
  options: DenoiserTrainingOptions,
): Algorithm<{ params: Params[] }, TrainingState<Params[]>> {
  const { data, schedule, net, batchSize = 128, optimizer = adamRule({ stepSize: 3e-3 }) } = options
  const T = schedule.steps
  return trainingLoop({
    data: { x0: data },
    batchSize,
    optimizer,
    loss: (params: Params[], batch: { x0: Tensor }, ctx) => {
      const x0 = batch.x0
      const n = x0.shape[0]
      const s = ctx.stream!
      const levels = child(s, 'levels')
      const alphaBars = Float64Array.from({ length: n }, () => alphaBarAt(schedule, 1 + integers(levels, T)))
      const eps = normals(child(s, 'noise'), x0.shape)
      const x0s = toFlat(x0)
      const es = toFlat(eps)
      const d = x0.shape[1]
      const xt = new Float64Array(n * d)
      for (let i = 0; i < n; i++)
        for (let a = 0; a < d; a++) {
          const k = i * d + a
          xt[k] = Math.sqrt(alphaBars[i]) * x0s[k] + Math.sqrt(1 - alphaBars[i]) * es[k]
        }
      const input = denoiserInput(net, fromData(xt, [n, d]), alphaBars)
      return meanSquaredErrorLoss(net.layer.apply(params, input), eps)
    },
  })
}
