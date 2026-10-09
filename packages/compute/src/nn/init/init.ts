/**
 * Weight initialisers, after torch.nn.init. Each function here returns an `Initialiser`, which draws from a stream, so
 * the same stream gives the same weights.
 *
 * An `Initialiser` takes the stream, the parameter's shape and its fans (the number of inputs and outputs each unit
 * sees), which the layer supplies: a linear layer's weight `[in, out]` has `fanIn` $= \text{in}$ and `fanOut`
 * $= \text{out}$; a convolution's kernel `[O, C, KH, KW]` has `fanIn` $= C K_H K_W$ and `fanOut` $= O K_H K_W$. The
 * scaled initialisers keep the variance of activations (Glorot & Bengio, 2010; He et al., 2015) or of gradients
 * steady from layer to layer.
 */

import { normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { zeros, type Tensor } from 'aifn-compute/foundation/tensor'

/**
 * Fan-in and fan-out of a parameter: `fanIn` is the number of inputs each output unit sums over, `fanOut` the number
 * of outputs each input feeds.
 */
export type Fans = { fanIn: number; fanOut: number }

/**
 * Draws a parameter tensor of `shape` from the stream `s`, scaled by the parameter's `fans` where the scheme uses
 * them.
 */
export type Initialiser = (s: Stream, shape: readonly number[], fans: Fans) => Tensor

/**
 * Uniform draws on $[-a, a]$.
 *
 * @param s The stream to draw from.
 * @param shape The shape of the tensor drawn.
 * @param bound The half-width $a$.
 * @returns A tensor of `shape` with entries from $\Unif(-a, a)$.
 */
const uniformOf = (s: Stream, shape: readonly number[], bound: number): Tensor =>
  uniform(s, -bound, bound, { shape: [...shape] })
/**
 * Normal draws with mean 0.
 *
 * @param s The stream to draw from.
 * @param shape The shape of the tensor drawn.
 * @param sd The standard deviation $\sigma$.
 * @returns A tensor of `shape` with entries from $\Gauss(0, \sigma^2)$.
 */
const normalOf = (s: Stream, shape: readonly number[], sd: number): Tensor => normal(s, 0, sd, { shape: [...shape] })

/**
 * Glorot (Xavier) uniform (Glorot & Bengio, 2010, eq. 16): $\Unif(-a, a)$ with
 * $a = g\sqrt{6/(n_\text{in} + n_\text{out})}$, $g$ the gain and $n_\text{in}$, $n_\text{out}$ the fans, so the
 * variance of activations and of gradients is kept across layers of tanh-like units.
 *
 * @param options The scheme's settings.
 * @param options.gain The gain $g$ that scales the bound (as `torch.nn.init.calculate_gain` gives for an activation).
 * @returns An initialiser drawing from $\Unif(-a, a)$.
 *
 * @example A $2 \times 3$ weight lies within $\pm\sqrt{6/5}$
 * const W = xavierUniform()(stream(0), [2, 3], { fanIn: 2, fanOut: 3 })
 * print('W =', W)
 * print('bound =', Math.sqrt(6 / 5), ' largest |w| =', max(abs(W)))
 */
export function xavierUniform({ gain = 1 }: { gain?: number } = {}): Initialiser {
  return (s, shape, { fanIn, fanOut }) => uniformOf(s, shape, gain * Math.sqrt(6 / (fanIn + fanOut)))
}

/**
 * Glorot (Xavier) normal (Glorot & Bengio, 2010): $\Gauss(0, \sigma^2)$ with
 * $\sigma = g\sqrt{2/(n_\text{in} + n_\text{out})}$, the variance of `xavierUniform`.
 *
 * @param options The scheme's settings.
 * @param options.gain The gain $g$ that scales the standard deviation.
 * @returns An initialiser drawing from $\Gauss(0, \sigma^2)$.
 *
 * @example The sample standard deviation is close to $\sqrt{2/(n_\text{in} + n_\text{out})}$
 * const W = xavierNormal()(stream(0), [100, 100], { fanIn: 100, fanOut: 100 })
 * print('sigma =', Math.sqrt(2 / 200), ' sample sd =', std(W))
 */
export function xavierNormal({ gain = 1 }: { gain?: number } = {}): Initialiser {
  return (s, shape, { fanIn, fanOut }) => normalOf(s, shape, gain * Math.sqrt(2 / (fanIn + fanOut)))
}

/** Options of the He initialisers. */
export type HeOptions = {
  /** The negative slope $\alpha$ of a leaky ReLU (0 for ReLU, the default): the gain is $\sqrt{2/(1 + \alpha^2)}$. */
  negativeSlope?: number
  /**
   * Scale by `fanIn` (the default; preserves the variance of activations going forward) or by `fanOut` (the variance
   * of gradients going backward).
   */
  mode?: 'fanIn' | 'fanOut'
}

/**
 * The He gain for a leaky ReLU.
 *
 * @param slope Its negative slope $\alpha$ (0 for ReLU).
 * @returns $\sqrt{2/(1 + \alpha^2)}$: $\sqrt{2}$ for ReLU.
 */
const heGain = (slope: number) => Math.sqrt(2 / (1 + slope * slope))

/**
 * He (Kaiming) uniform (He et al., 2015, §2.2): $\Unif(-a, a)$ with $a = g\sqrt{3/n}$, $n$ the fan chosen by `mode`
 * and $g = \sqrt{2/(1 + \alpha^2)}$ ($\sqrt{2}$ for ReLU), so the variance of ReLU activations is kept across
 * layers.
 *
 * @param options The scheme's settings.
 * @param options.negativeSlope The negative slope $\alpha$ of the leaky ReLU that follows (0 for ReLU).
 * @param options.mode Which fan $n$ scales the bound: `'fanIn'` keeps the forward variance, `'fanOut'` the backward.
 * @returns An initialiser drawing from $\Unif(-a, a)$.
 *
 * @example The bound is $\sqrt{6/n_\text{in}}$ for ReLU
 * const W = heUniform()(stream(0), [3, 2], { fanIn: 3, fanOut: 2 })
 * print('W =', W)
 * print('bound =', Math.sqrt(6 / 3), ' largest |w| =', max(abs(W)))
 */
export function heUniform({ negativeSlope = 0, mode = 'fanIn' }: HeOptions = {}): Initialiser {
  return (s, shape, fans) => uniformOf(s, shape, heGain(negativeSlope) * Math.sqrt(3 / fans[mode]))
}

/**
 * He (Kaiming) normal (He et al., 2015, §2.2): $\Gauss(0, \sigma^2)$ with $\sigma = g/\sqrt{n}$, $n$ the fan chosen
 * by `mode` and $g = \sqrt{2/(1 + \alpha^2)}$.
 *
 * @param options The scheme's settings.
 * @param options.negativeSlope The negative slope $\alpha$ of the leaky ReLU that follows (0 for ReLU).
 * @param options.mode Which fan $n$ scales the deviation: `'fanIn'` keeps the forward variance, `'fanOut'` the
 *   backward.
 * @returns An initialiser drawing from $\Gauss(0, \sigma^2)$.
 *
 * @example The sample standard deviation is close to $\sqrt{2/n_\text{in}}$
 * const W = heNormal()(stream(0), [50, 200], { fanIn: 50, fanOut: 200 })
 * print('sigma =', Math.sqrt(2 / 50), ' sample sd =', std(W))
 * const V = heNormal({ mode: 'fanOut' })(stream(0), [50, 200], { fanIn: 50, fanOut: 200 })
 * print('by fanOut: sigma =', Math.sqrt(2 / 200), ' sample sd =', std(V))
 */
export function heNormal({ negativeSlope = 0, mode = 'fanIn' }: HeOptions = {}): Initialiser {
  return (s, shape, fans) => normalOf(s, shape, heGain(negativeSlope) / Math.sqrt(fans[mode]))
}

/**
 * LeCun uniform $\Unif(-1/\sqrt{n_\text{in}}, 1/\sqrt{n_\text{in}})$: PyTorch's default for linear and convolution
 * weights.
 *
 * @returns An initialiser drawing from that uniform; it ignores `fanOut`.
 *
 * @example With $n_\text{in} = 4$ the weights lie within $\pm 1/2$
 * print('W =', lecunUniform()(stream(0), [4, 2], { fanIn: 4, fanOut: 2 }))
 */
export function lecunUniform(): Initialiser {
  return (s, shape, { fanIn }) => uniformOf(s, shape, 1 / Math.sqrt(fanIn))
}

/**
 * $\Gauss(0, \sigma^2)$ regardless of the fans, e.g. $\sigma = 0.02$ for transformer embeddings.
 *
 * @param sd The standard deviation $\sigma$.
 * @returns An initialiser drawing from $\Gauss(0, \sigma^2)$; it ignores the fans.
 *
 * @example Small embedding weights
 * print('E =', normalInit(0.02)(stream(0), [2, 4], { fanIn: 2, fanOut: 4 }))
 */
export function normalInit(sd: number): Initialiser {
  return (s, shape) => normalOf(s, shape, sd)
}

/**
 * Zeros (biases, and the last layer of a residual branch).
 *
 * @returns An initialiser giving a zero tensor of the shape; it ignores the stream and the fans.
 *
 * @example A zero bias
 * print('b =', zerosInit()(stream(0), [3], { fanIn: 2, fanOut: 3 }))
 */
export function zerosInit(): Initialiser {
  return (_s, shape) => zeros(shape)
}
