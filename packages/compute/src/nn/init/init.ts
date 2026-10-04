/**
 * Weight initialisers. Each draws from a stream, so the same stream gives the same weights. An `Initialiser` takes the
 * stream, the parameter's shape and its fans (the number of inputs and outputs each unit sees), which the layer
 * supplies: a linear layer's weight [in, out] has fanIn = in and fanOut = out; a convolution's kernel [O, C, KH, KW]
 * has fanIn = C·KH·KW and fanOut = O·KH·KW.
 */

import { normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { zeros, type Tensor } from 'aifn-compute/foundation/tensor'

/** Fan-in and fan-out of a parameter. */
export type Fans = { fanIn: number; fanOut: number }

/** Draws a parameter tensor of `shape` from `s`. */
export type Initialiser = (s: Stream, shape: readonly number[], fans: Fans) => Tensor

const uniformOf = (s: Stream, shape: readonly number[], bound: number): Tensor =>
  uniform(s, -bound, bound, { shape: [...shape] })
const normalOf = (s: Stream, shape: readonly number[], sd: number): Tensor => normal(s, 0, sd, { shape: [...shape] })

/**
 * Glorot (Xavier) uniform (Glorot & Bengio, 2010, eq. 16): U(−a, a) with a = gain·√(6/(fanIn + fanOut)), so the
 * variance of activations and of gradients is kept across layers of tanh-like units.
 */
export function xavierUniform({ gain = 1 }: { gain?: number } = {}): Initialiser {
  return (s, shape, { fanIn, fanOut }) => uniformOf(s, shape, gain * Math.sqrt(6 / (fanIn + fanOut)))
}

/** Glorot (Xavier) normal: N(0, σ²) with σ = gain·√(2/(fanIn + fanOut)). */
export function xavierNormal({ gain = 1 }: { gain?: number } = {}): Initialiser {
  return (s, shape, { fanIn, fanOut }) => normalOf(s, shape, gain * Math.sqrt(2 / (fanIn + fanOut)))
}

/** Options of the He initialisers. */
export type HeOptions = {
  /** The negative slope of a leaky ReLU (0 for ReLU): the gain is √(2/(1 + slope²)). */
  negativeSlope?: number
  /** Scale by fanIn (default; preserves the forward variance) or fanOut (the backward variance). */
  mode?: 'fanIn' | 'fanOut'
}

const heGain = (slope: number) => Math.sqrt(2 / (1 + slope * slope))

/**
 * He (Kaiming) uniform (He et al., 2015, §2.2): U(−a, a) with a = gain·√(3/fan), gain = √2 for ReLU, so the variance
 * of ReLU activations is kept across layers.
 */
export function heUniform({ negativeSlope = 0, mode = 'fanIn' }: HeOptions = {}): Initialiser {
  return (s, shape, fans) => uniformOf(s, shape, heGain(negativeSlope) * Math.sqrt(3 / fans[mode]))
}

/** He (Kaiming) normal: N(0, σ²) with σ = gain/√fan. */
export function heNormal({ negativeSlope = 0, mode = 'fanIn' }: HeOptions = {}): Initialiser {
  return (s, shape, fans) => normalOf(s, shape, heGain(negativeSlope) / Math.sqrt(fans[mode]))
}

/** LeCun uniform U(−1/√fanIn, 1/√fanIn): PyTorch's default for linear and convolution weights. */
export function lecunUniform(): Initialiser {
  return (s, shape, { fanIn }) => uniformOf(s, shape, 1 / Math.sqrt(fanIn))
}

/** N(0, σ²) regardless of the fans, e.g. σ = 0.02 for transformer embeddings. */
export function normalInit(sd: number): Initialiser {
  return (s, shape) => normalOf(s, shape, sd)
}

/** Zeros (biases, and the last layer of a residual branch). */
export function zerosInit(): Initialiser {
  return (_s, shape) => zeros(shape)
}
