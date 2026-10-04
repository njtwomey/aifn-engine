/**
 * A generative adversarial network on low-dimensional data (Goodfellow et al., 2014): a generator G maps latent noise
 * z ~ N(0, I) to points, a discriminator (critic) D scores points, and the two are trained against each other. Both are
 * small MLPs from `aifn-compute/nn`; the games (minimax, non-saturating, Wasserstein with gradient penalty, hinge) are the
 * losses of `aifn-compute/learning/losses`, and the alternating updates are `aifn-compute/nn/training`'s `adversarialTraining`.
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

/** A generator and a discriminator for d-dimensional data. */
export type Gan = {
  /** z [n, latent] → x [n, d]. */
  readonly generator: Layer<Params[]>
  /** x [n, d] → score [n, 1] (a logit, or a critic value). */
  readonly discriminator: Layer<Params[]>
  readonly latent: number
  readonly dimension: number
}

/** Options of `gan`. */
export type GanOptions = {
  /** Data dimension. Default 2. */
  dimension?: number
  /** Latent dimension. Default 2. */
  latent?: number
  /** Hidden widths of both networks. Default [64, 64]. */
  hidden?: readonly number[]
  /** Generator activation. Default ReLU. */
  generatorActivation?: Activation
  /** Discriminator activation. Default leaky ReLU (Radford et al., 2016). */
  discriminatorActivation?: Activation
}

/** The two networks of a GAN. */
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

/** n latent draws z ~ N(0, I), [n, latent]. */
export function latents(net: Gan, s: Stream, n: number): Tensor {
  return normals(s, [n, net.latent])
}

/** G(z) for latents z [n, latent]: points [n, d]. Traced when the parameters are. */
export function generatePoints(net: Gan, params: Params[], z: Value): Value {
  return net.generator.apply(params, z)
}

/** D(x) as a flat [n] vector of scores (logits, or critic values). */
export function discriminate(net: Gan, params: Params[], x: Value): Value {
  const out = net.discriminator.apply(params, x)
  return reshape(out, [shapeOfValue(x)[0]])
}

/** Options of `ganTraining`. */
export type GanTrainingOptions = {
  net: Gan
  /** Training points [N, d]. */
  data: Tensor
  /** The game. Default `non-saturating`. */
  game?: AdversarialGame
  /** Points per batch, real and generated alike. Default 128. */
  batchSize?: number
  /** Discriminator updates per generator update. Default 1 (5 is usual for the Wasserstein game). */
  criticSteps?: number
  /** Update rules. Default Adam with step 1e-3 and β₁ = 0.5 (Radford et al., 2016) for both. */
  generatorOptimizer?: UpdateRule<unknown>
  criticOptimizer?: UpdateRule<unknown>
  /** Gradient-penalty weight λ (Gulrajani et al., 2017). Default 10 for `wasserstein`, else 0. */
  penalty?: number
}

/** A minibatch of real points drawn with replacement from `data`. */
function realBatch(data: Tensor, s: Stream, n: number): Tensor {
  const N = data.shape[0]
  const ids = Array.from({ length: n }, () => integers(s, N))
  return unwrap(take(data, ids)) as Tensor
}

/**
 * GAN training as a traceable algorithm (`adversarialTraining` with the chosen game). Each critic update draws a real
 * batch (`batch`), latents (`latent`) and, with a penalty, the mixing weights (`mix`) from its stream; each generator
 * update draws its latents. `init` takes `{ generator: net.generator.init(s1), critic: net.discriminator.init(s2) }`.
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

/** Points of a trained generator from given latents, as a plain tensor [n, d]. */
export function sampleGenerator(net: Gan, params: Params[], z: Tensor): Tensor {
  return unwrap(generatePoints(net, params, z)) as Tensor
}

/** Discriminator scores at points x [n, d], as a plain Float64Array. */
export function scoresAt(net: Gan, params: Params[], x: Tensor): Float64Array {
  return Float64Array.from(toFlat(unwrap(discriminate(net, params, x)) as Tensor))
}
