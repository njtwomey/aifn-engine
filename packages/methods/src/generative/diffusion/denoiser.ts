/**
 * A small learned noise predictor: an MLP on the noised point and features of its noise level, trained with the
 * "simple" DDPM objective E‖ε − ε̂_θ(√ᾱₜx₀ + √(1 − ᾱₜ)ε, ᾱₜ)‖² with t uniform on 1 … T (Ho et al., 2020, eq. 14 and
 * Algorithm 1). Enough to learn 1-D and 2-D toy data in the browser; the network and its training loop come from
 * `aifn-compute/nn`, the loss from `aifn-compute/learning/losses`.
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

/** A noise-prediction network: an MLP from [x, noise-level features] to ε̂. */
export type Denoiser = {
  readonly layer: Layer<Params[]>
  /** The data's dimension d. */
  readonly dimension: number
  /** Frequencies K of the noise-level features (1 + 2K features). */
  readonly frequencies: number
}

/** Options of `denoiser`. */
export type DenoiserOptions = {
  /** Hidden layer widths (default [64, 64]). */
  hidden?: readonly number[]
  /** Frequencies K of the noise-level features (default 4). */
  frequencies?: number
  /** Hidden activation (default SiLU, as in DDPM's U-Net). */
  activation?: Activation
}

/** A noise-prediction MLP for d-dimensional data. */
export function denoiser(
  dimension: number,
  { hidden = [64, 64], frequencies = 4, activation = 'silu' }: DenoiserOptions = {},
): Denoiser {
  const layer = Mlp([dimension + 1 + 2 * frequencies, ...hidden, dimension], { activation })
  return { layer, dimension, frequencies }
}

/**
 * The features of a noise level: the noise scale σ = √(1 − ᾱ) and sin(kπσ), cos(kπσ) for k = 1 … K, a sinusoidal
 * embedding (as Vaswani et al., 2017, for positions) of a bounded quantity that orders the levels.
 */
export function noiseLevelFeatures(alphaBar: number, frequencies: number): Tensor {
  const f = levelFeatures(alphaBar, frequencies)
  return fromData(Float64Array.from(f), [f.length])
}

function levelFeatures(alphaBar: number, frequencies: number): number[] {
  const sigma = Math.sqrt(1 - alphaBar)
  const out = [sigma]
  for (let k = 1; k <= frequencies; k++) out.push(Math.sin(k * Math.PI * sigma), Math.cos(k * Math.PI * sigma))
  return out
}

/** The network input [x, features(ᾱᵢ)] for points x [n, d] and one level ᾱ per row (or one for all). */
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

/** The trained network as a noise predictor. */
export function networkNoisePredictor(net: Denoiser, params: Params[]): NoisePredictor {
  return (x, alphaBar) => unwrap(net.layer.apply(params, denoiserInput(net, x, alphaBar))) as Tensor
}

/** Options of `denoiserTraining`. */
export type DenoiserTrainingOptions = {
  /** Training points x₀, shape [n, d]. */
  data: Tensor
  schedule: NoiseSchedule
  net: Denoiser
  /** Points per step (default 128). */
  batchSize?: number
  /** The update rule. Default `adamRule({ stepSize: 3e-3 })`. */
  optimizer?: UpdateRule<unknown>
}

/**
 * Training of a noise predictor as a traceable algorithm (`aifn-compute/nn`'s `trainingLoop`): each step takes a minibatch of
 * data, a level t ~ U{1 … T} and noise ε ~ N(0, I) per point, all from the step's stream, and takes one optimiser step
 * on the mean squared error between ε̂ and ε. `init` takes `{ params: net.layer.init(stream) }`.
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
