/**
 * A generative adversarial network on low-dimensional data (Goodfellow et al., 2014): a generator $G$ maps latent noise
 * $\zvec \sim \Gauss(\zeros, \Imat)$ to points, a discriminator (critic) $D$ scores points, and the two are trained
 * against each other. Both are small MLPs from `aifn-compute/nn`; the games (minimax, non-saturating, Wasserstein with
 * gradient penalty, hinge) are the losses of `aifn-compute/learning/losses`, and the alternating updates are
 * `aifn-compute/nn/training`'s `adversarialTraining`.
 *
 * Points are rows: $n$ latents are $[n, L]$ for the latent width $L$, and $n$ points $[n, d]$ for the data dimension
 * $d$. Each network's parameters are the `Params[]` of its MLP, as `init` returns them.
 */

import type { Activation } from 'aifn-compute/nn/functional'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { adversarialTraining, type AdversarialTrainingState } from 'aifn-compute/nn/training'
import { stopGradient } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, integers, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  mul,
  reshape,
  shapeOfValue,
  take,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  discriminatorLoss,
  generatorLoss,
  gradientPenalty,
  GRADIENT_PENALTY_WEIGHT,
  type AdversarialGame,
} from 'aifn-compute/learning/losses'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'

/** A generator and a discriminator for $d$-dimensional data. */
export type Gan = {
  /** The generator $G$: latents $\zvec$ $[n, L]$ to points $\xvec$ $[n, d]$. */
  readonly generator: Layer<Params[]>
  /** The discriminator $D$: points $\xvec$ $[n, d]$ to scores $[n, 1]$ (logits, or critic values). */
  readonly discriminator: Layer<Params[]>
  /** The latent width $L$. */
  readonly latent: number
  /** The data dimension $d$. */
  readonly dimension: number
}

/** Options of `gan`. */
export type GanOptions = {
  /** The data dimension $d$. Default 2. */
  dimension?: number
  /** The latent width $L$. Default 2. */
  latent?: number
  /** Hidden widths of both networks. Default `[64, 64]`. */
  hidden?: readonly number[]
  /** Generator activation. Default ReLU. */
  generatorActivation?: Activation
  /** Discriminator activation. Default leaky ReLU (Radford et al., 2016). */
  discriminatorActivation?: Activation
}

/**
 * The two networks of a GAN: a generator MLP of widths $[L, \dots, d]$ and a discriminator MLP of widths
 * $[d, \dots, 1]$, with the same hidden widths, a linear output and their own hidden activations. They come without
 * parameters: `net.generator.init(s)` and `net.discriminator.init(s)` draw them.
 *
 * @param options The data dimension, latent width, hidden widths and the two activations; each has a default.
 * @returns The networks, with $L$ and $d$.
 *
 * @example A small GAN for 1-d data, and the shapes of its generator's weights
 * const net = gan({ dimension: 1, latent: 1, hidden: [8] })
 * print('latent:', net.latent, ' dimension:', net.dimension)
 * const params = net.generator.init(stream(1))
 * print('generator weights:', params.filter((p) => p.weight).map((p) => p.weight.shape))
 */
export function gan(options: GanOptions = {}): Gan {
  const { dimension = 2, latent = 2, hidden = [64, 64] } = options
  const { generatorActivation = 'relu', discriminatorActivation = 'leakyRelu' } = options
  return {
    generator: Mlp([latent, ...hidden, dimension], { activation: generatorActivation }),
    discriminator: Mlp([dimension, ...hidden, 1], { activation: discriminatorActivation }),
    latent,
    dimension,
  }
}

/**
 * $n$ latent draws $\zvec \sim \Gauss(\zeros, \Imat)$, as rows $[n, L]$.
 *
 * @param net The GAN, whose latent width $L$ is used.
 * @param s The stream the standard normals are drawn from; advanced by the draw.
 * @param n The number of latents.
 * @returns The latents, a tensor $[n, L]$.
 *
 * @example Three latents for a GAN with $L = 2$
 * print(latents(gan(), stream(1), 3))
 */
export function latents(net: Gan, s: Stream, n: number): Tensor {
  return normals(s, [n, net.latent])
}

/**
 * $G(\zvec)$ for latents $\zvec$: the generator applied to each row. Traced (differentiable) when the parameters or
 * the latents are, which is how the generator's loss reaches its parameters.
 *
 * @param net The GAN.
 * @param params The generator's parameters, as `net.generator.init` returns them.
 * @param z The latents $[n, L]$.
 * @returns The points $[n, d]$.
 *
 * @example An untrained generator on three latents
 * const net = gan({ dimension: 1, latent: 1, hidden: [8] })
 * const params = net.generator.init(stream(1))
 * print('G(z):', generatePoints(net, params, tensor([[-1], [0], [1]])))
 */
export function generatePoints(net: Gan, params: Params[], z: Value): Value {
  return net.generator.apply(params, z)
}

/**
 * $D(\xvec)$ for points $\xvec$, as a flat vector of $n$ scores: logits for the minimax and non-saturating games,
 * critic values for the Wasserstein and hinge games. Traced (differentiable) when the parameters or points are.
 *
 * @param net The GAN.
 * @param params The discriminator's parameters, as `net.discriminator.init` returns them.
 * @param x The points $[n, d]$.
 * @returns The scores, a vector of length $n$.
 *
 * @example An untrained discriminator on three 1-d points
 * const net = gan({ dimension: 1, latent: 1, hidden: [8] })
 * const params = net.discriminator.init(stream(1))
 * print('D(x):', discriminate(net, params, tensor([[-1], [0], [1]])))
 */
export function discriminate(net: Gan, params: Params[], x: Value): Value {
  const out = net.discriminator.apply(params, x)
  return reshape(out, [shapeOfValue(x)[0]])
}

/** Options of `ganTraining`. */
export type GanTrainingOptions = {
  /** The GAN whose networks are trained. */
  net: Gan
  /** Training points $[N, d]$. */
  data: Tensor
  /** The game. Default `non-saturating`. */
  game?: AdversarialGame
  /** Points per batch, real and generated alike. Default 128. */
  batchSize?: number
  /** Discriminator updates per generator update. Default 1 (5 is usual for the Wasserstein game). */
  criticSteps?: number
  /** The generator's update rule. Default Adam with step 1e-3 and $\beta_1 = 0.5$ (Radford et al., 2016). */
  generatorOptimizer?: UpdateRule<unknown>
  /** The discriminator's update rule. Default Adam with step 1e-3 and $\beta_1 = 0.5$, as for the generator. */
  criticOptimizer?: UpdateRule<unknown>
  /** Gradient-penalty weight $\lambda$ (Gulrajani et al., 2017). Default 10 for `wasserstein`, else 0. */
  penalty?: number
}

/**
 * A minibatch of real points drawn with replacement from `data`.
 *
 * @param data The training points $[N, d]$.
 * @param s The stream the row indices are drawn from; advanced by $n$ draws.
 * @param n The batch size.
 * @returns The rows drawn, $[n, d]$.
 */
function realBatch(data: Tensor, s: Stream, n: number): Tensor {
  const N = data.shape[0]
  const ids = Array.from({ length: n }, () => integers(s, N))
  return unwrap(take(data, ids)) as Tensor
}

/**
 * GAN training as a traceable algorithm (`adversarialTraining` with the chosen game). Each critic update draws a real
 * batch (`batch`), latents (`latent`) and, with a penalty, the mixing weights (`mix`) from its stream, and minimises
 * `discriminatorLoss` plus $\lambda$ times the gradient penalty; each generator update draws its latents and minimises
 * `generatorLoss` with the critic held fixed. At the equilibrium of the minimax and non-saturating games, where
 * $D(\xvec) = \tfrac12$ everywhere, the critic's loss is $\log 4$ and the non-saturating generator's $\log 2$.
 * `init` takes `{ generator: net.generator.init(s1), critic: net.discriminator.init(s2) }`.
 *
 * @param options The network, the data, the game, the batch size, the critic updates per generator update, the two
 *   update rules and the penalty weight; all but `net` and `data` have defaults.
 * @returns The algorithm, to run with `run`, `trace` or `live`; its state holds both networks' parameters and the last
 *   two losses.
 *
 * @example The generator learns to put its points at 3, on 1-d data from $\Gauss(3, 0.5^2)$
 * const data = normal(stream(1), 3, 0.5, { shape: [256, 1] })
 * const net = gan({ dimension: 1, latent: 1, hidden: [8] })
 * const start = { generator: net.generator.init(stream(2)), critic: net.discriminator.init(stream(3)) }
 * const alg = ganTraining({
 *   net,
 *   data,
 *   batchSize: 32,
 *   generatorOptimizer: optimizerOf({ name: 'adam', stepSize: 0.01 }),
 *   criticOptimizer: optimizerOf({ name: 'adam', stepSize: 0.01 }),
 * })
 * const z = latents(net, stream(4), 200)
 * print('mean of G(z) before:', mean(sampleGenerator(net, start.generator, z)))
 * const state = run(alg, start, 100)
 * print('mean of G(z) after 100 steps:', mean(sampleGenerator(net, state.generator, z)))
 * print('critic loss:', state.criticLoss, ' log 4:', Math.log(4))
 * print('generator loss:', state.generatorLoss, ' log 2:', Math.log(2))
 */
export function ganTraining(
  options: GanTrainingOptions,
): Algorithm<{ generator: Params[]; critic: Params[] }, AdversarialTrainingState<Params[], Params[]>> {
  const { net, data, game = 'non-saturating', batchSize = 128, criticSteps = 1 } = options
  const penalty = options.penalty ?? (game === 'wasserstein' ? GRADIENT_PENALTY_WEIGHT : 0)
  const rule = () => adamRule({ stepSize: 1e-3, beta1: 0.5 }) as UpdateRule<unknown>
  return adversarialTraining<Params[], Params[]>({
    criticSteps,
    generatorOptimizer: options.generatorOptimizer ?? rule(),
    criticOptimizer: options.criticOptimizer ?? rule(),
    criticLoss: (critic, generator, s) => {
      const real = realBatch(data, child(s, 'batch'), batchSize)
      const fake = unwrap(generatePoints(net, generator, latents(net, child(s, 'latent'), batchSize))) as Tensor
      const loss = discriminatorLoss(discriminate(net, critic, real), discriminate(net, critic, fake), game)
      if (!(penalty > 0)) return loss
      const mix = uniform(child(s, 'mix'), 0, 1, { shape: [batchSize] }) as Tensor
      const gp = gradientPenalty((x: Value) => discriminate(net, critic, x), real, fake, mix)
      return add(loss, mul(penalty, gp))
    },
    generatorLoss: (generator, critic, s) => {
      const fake = generatePoints(net, generator, latents(net, child(s, 'latent'), batchSize))
      return generatorLoss(discriminate(net, stopGradient(critic), fake), game)
    },
  })
}

/**
 * Points of a trained generator from given latents, as a plain (untraced) tensor: `generatePoints` for reading out
 * rather than differentiating.
 *
 * @param net The GAN.
 * @param params The generator's parameters.
 * @param z The latents $[n, L]$; fixed latents show how the same points move as training goes on.
 * @returns The points $[n, d]$.
 *
 * @example Four points of an untrained generator
 * const net = gan({ dimension: 1, latent: 1, hidden: [8] })
 * const params = net.generator.init(stream(1))
 * const x = sampleGenerator(net, params, latents(net, stream(2), 4))
 * print('shape:', x.shape, ' points:', x)
 */
export function sampleGenerator(net: Gan, params: Params[], z: Tensor): Tensor {
  return unwrap(generatePoints(net, params, z)) as Tensor
}

/**
 * Discriminator scores at points, as a plain array: `discriminate` for reading out, such as on a grid for a figure.
 *
 * @param net The GAN.
 * @param params The discriminator's parameters.
 * @param x The points $[n, d]$.
 * @returns The $n$ scores (logits, or critic values).
 *
 * @example Logits, and the probabilities $D(\xvec) = \sigma(\text{logit})$ they stand for
 * const net = gan({ dimension: 1, latent: 1, hidden: [8] })
 * const params = net.discriminator.init(stream(1))
 * const logits = scoresAt(net, params, tensor([[-1], [0], [1]]))
 * print('logits:', logits)
 * print('D(x):', logits.map((v) => 1 / (1 + Math.exp(-v))))
 */
export function scoresAt(net: Gan, params: Params[], x: Tensor): Float64Array {
  return Float64Array.from(toFlat(unwrap(discriminate(net, params, x)) as Tensor))
}
