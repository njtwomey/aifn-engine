/**
 * Adversarial losses: the two players' objectives of a generative adversarial network (Goodfellow et al., 2014) in its
 * four common forms, from the discriminator's outputs on real and generated points, and the gradient penalty of
 * WGAN-GP (Gulrajani et al., 2017).
 *
 * - `minimax`: the original game, min_G max_D E log D(x) + E log(1 − D(G(z))), with D = σ(logit). The discriminator
 *   minimises the binary cross-entropy of real against fake; the generator minimises E log(1 − D(G(z))) =
 *   −E softplus(logit), whose gradient vanishes when D rejects the fakes confidently.
 * - `non-saturating`: the same discriminator; the generator minimises −E log D(G(z)) = E softplus(−logit) instead
 *   (Goodfellow et al., 2014, §3), which has the same fixed point and strong gradients early in training.
 * - `wasserstein`: a critic of unbounded scores (Arjovsky, Chintala & Bottou, 2017); the critic minimises
 *   E f(G(z)) − E f(x), the generator −E f(G(z)). The critic must be kept near 1-Lipschitz, here by the gradient penalty.
 * - `hinge`: the critic minimises E max(0, 1 − f(x)) + E max(0, 1 + f(G(z))) (Lim & Ye, 2017; Miyato et al., 2018),
 *   the generator −E f(G(z)).
 *
 * Every loss is a composition of tensor primitives, so it is differentiable in the scores to any order.
 */

import { grad } from 'aifn-compute/foundation/autodiff'
import {
  add,
  fromData,
  maximum,
  mean,
  neg,
  sqrt,
  square,
  sub,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { softplus } from 'aifn-compute/numerics/special'
import { defineLoss } from './core'

/** The four adversarial games. */
export type AdversarialGame = 'minimax' | 'non-saturating' | 'wasserstein' | 'hinge'

/** The games, for pickers. */
export const adversarialGames: readonly AdversarialGame[] = ['minimax', 'non-saturating', 'wasserstein', 'hinge']

/**
 * The discriminator's (critic's) loss from its scores on a batch of real points and a batch of generated ones. For
 * `minimax` and `non-saturating` the scores are logits and the loss is E softplus(−real) + E softplus(fake), the binary
 * cross-entropy of real (label 1) against fake (label 0); for `wasserstein` it is E fake − E real; for `hinge`,
 * E max(0, 1 − real) + E max(0, 1 + fake).
 */
export const discriminatorLoss = defineLoss(
  {
    key: 'discriminatorLoss',
    name: 'Adversarial discriminator loss',
    family: 'adversarial',
    inputs: 'logits',
    notes: ['generative-adversarial-network'],
    cite: ['goodfellow2014', 'arjovsky2017', 'miyato2018'],
    target: 'the log density ratio log p_data/p_g (minimax), or a Lipschitz critic',
  },
  (real: Value, fake: Value, game: AdversarialGame = 'non-saturating'): Value => {
    switch (game) {
      case 'minimax':
      case 'non-saturating':
        return add(mean(softplus(neg(real))), mean(softplus(fake)))
      case 'wasserstein':
        return sub(mean(fake), mean(real))
      case 'hinge':
        return add(mean(maximum(0, sub(1, real))), mean(maximum(0, add(1, fake))))
    }
  },
)

/**
 * The generator's loss from the discriminator's scores on generated points: −E softplus(fake) for `minimax`
 * (E log(1 − D)), E softplus(−fake) for `non-saturating` (−E log D), and −E fake for `wasserstein` and `hinge`.
 */
export const generatorLoss = defineLoss(
  {
    key: 'generatorLoss',
    name: 'Adversarial generator loss',
    family: 'adversarial',
    inputs: 'logits',
    notes: ['generative-adversarial-network'],
    cite: ['goodfellow2014', 'arjovsky2017'],
  },
  (fake: Value, game: AdversarialGame = 'non-saturating'): Value => {
    switch (game) {
      case 'minimax':
        return neg(mean(softplus(fake)))
      case 'non-saturating':
        return mean(softplus(neg(fake)))
      case 'wasserstein':
      case 'hinge':
        return neg(mean(fake))
    }
  },
)

/**
 * The gradient penalty of WGAN-GP (Gulrajani et al., 2017, eq. 3): E (‖∇ₓ f(x̂)‖₂ − 1)² over the points x̂ = εx +
 * (1 − ε)x̃ on segments between real points x and generated points x̃ (rows of [n, d] tensors), with ε ∈ [0, 1] per row
 * (`mix`, shape [n], usually uniform draws). `critic` maps a batch [n, d] to scores [n] or [n, 1] and may close over
 * traced parameters: the penalty is then differentiable in them (a gradient of a gradient). The norm is smoothed as
 * √(‖g‖² + 10⁻¹²), so its derivative is finite where g = 0.
 */
export const gradientPenalty = defineLoss(
  {
    key: 'gradientPenalty',
    name: 'Gradient penalty (WGAN-GP)',
    family: 'adversarial',
    inputs: 'values',
    notes: ['generative-adversarial-network'],
    cite: ['gulrajani2017'],
  },
  (critic: (x: Value) => Value, real: Tensor, fake: Tensor, mix: Tensor, target = 1): Value => {
    const [n, d] = real.shape
    const e = toFlat(mix)
    const r = toFlat(real)
    const f = toFlat(fake)
    const between = new Float64Array(n * d)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++) between[i * d + j] = e[i] * r[i * d + j] + (1 - e[i]) * f[i * d + j]
    const xHat = fromData(between, [n, d])
    const g = grad((x: Value) => sum(critic(x)))(xHat) as Value
    const norms = sqrt(add(sum(square(g), -1), 1e-12))
    return mean(square(sub(norms, target)))
  },
)

/** The penalty's weight λ in the WGAN-GP critic loss, as Gulrajani et al. (2017) use it. */
export const GRADIENT_PENALTY_WEIGHT = 10
