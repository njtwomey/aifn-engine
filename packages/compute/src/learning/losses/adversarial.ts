/**
 * Adversarial losses: the two players' objectives of a generative adversarial network (Goodfellow et al., 2014) in its
 * four common forms, from the discriminator's outputs on real and generated points, and the gradient penalty of
 * WGAN-GP (Gulrajani et al., 2017).
 *
 * - `minimax`: the original game, $\min_G \max_D \expect \log D(\xvec) + \expect \log(1 - D(G(\zvec)))$, with
 *   $D = \sigma(\text{logit})$. The discriminator minimises the binary cross-entropy of real against fake; the
 *   generator minimises $\expect \log(1 - D(G(\zvec))) = -\expect \operatorname{softplus}(\text{logit})$, whose
 *   gradient vanishes when $D$ rejects the fakes confidently.
 * - `non-saturating`: the same discriminator; the generator minimises
 *   $-\expect \log D(G(\zvec)) = \expect \operatorname{softplus}(-\text{logit})$ instead (Goodfellow et al., 2014,
 *   §3), which has the same fixed point and strong gradients early in training.
 * - `wasserstein`: a critic of unbounded scores (Arjovsky, Chintala & Bottou, 2017); the critic minimises
 *   $\expect f(G(\zvec)) - \expect f(\xvec)$, the generator $-\expect f(G(\zvec))$. The critic must be kept near
 *   1-Lipschitz, here by the gradient penalty.
 * - `hinge`: the critic minimises $\expect \max(0, 1 - f(\xvec)) + \expect \max(0, 1 + f(G(\zvec)))$ (Lim & Ye,
 *   2017; Miyato et al., 2018), the generator $-\expect f(G(\zvec))$.
 *
 * The expectations are means over the batch of scores given.
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
 * `minimax` and `non-saturating` the scores are logits $r$ (real) and $f$ (fake) and the loss is
 * $\expect \operatorname{softplus}(-r) + \expect \operatorname{softplus}(f)$, the binary cross-entropy of real
 * (label 1) against fake (label 0); for `wasserstein` it is $\expect f - \expect r$; for `hinge`,
 * $\expect \max(0, 1 - r) + \expect \max(0, 1 + f)$. Each mean is over its own batch, so the two may differ in size.
 *
 * @param real The discriminator's scores $r$ on real points, of any shape.
 * @param fake Its scores $f$ on generated points, of any shape.
 * @param game Which of the four games.
 * @returns The loss, a number (or a traced scalar).
 *
 * @example The same scores under each game
 * const real = tensor([2, 0.5])
 * const fake = tensor([-2, 0])
 * for (const game of adversarialGames) print(`${game}:`, discriminatorLoss(real, fake, game))
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
 * The generator's loss from the discriminator's scores $f$ on generated points:
 * $-\expect \operatorname{softplus}(f)$ for `minimax` (that is $\expect \log(1 - D)$),
 * $\expect \operatorname{softplus}(-f)$ for `non-saturating` ($-\expect \log D$), and $-\expect f$ for `wasserstein`
 * and `hinge`.
 *
 * @param fake The discriminator's scores $f$ on generated points, of any shape.
 * @param game Which of the four games.
 * @returns The loss, a number (or a traced scalar).
 *
 * @example The minimax generator's gradient vanishes when the fakes are rejected confidently
 * const z = tensor([-6])
 * print('minimax:', grad((f) => generatorLoss(f, 'minimax'))(z))
 * print('non-saturating:', grad((f) => generatorLoss(f, 'non-saturating'))(z))
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
 * The gradient penalty of WGAN-GP (Gulrajani et al., 2017, eq. 3):
 * $\expect (\norm{\nabla_{\xvec} f(\hat{\xvec})}_2 - 1)^2$ over the points
 * $\hat{\xvec} = \varepsilon \xvec + (1 - \varepsilon) \tilde{\xvec}$ on segments between real points $\xvec$ and
 * generated points $\tilde{\xvec}$ (rows of `[n, d]` tensors), with $\varepsilon \in [0, 1]$ per row (`mix`, shape
 * `[n]`, usually uniform draws). `critic` maps a batch `[n, d]` to scores `[n]` or `[n, 1]` and may close over traced
 * parameters: the penalty is then differentiable in them (a gradient of a gradient). The points themselves are read as
 * constants. The norm is smoothed as $\sqrt{\norm{\gvec}^2 + 10^{-12}}$, so its derivative is finite where
 * $\gvec = \zeros$.
 *
 * @param critic The critic $f$: a batch of points `[n, d]` to scores `[n]` or `[n, 1]`.
 * @param real The real points $\xvec$, one per row, `[n, d]`.
 * @param fake The generated points $\tilde{\xvec}$, `[n, d]`.
 * @param mix The weight $\varepsilon$ of the real point on each row's segment, `[n]`.
 * @param target The gradient norm the penalty pulls towards (default 1, the 1-Lipschitz bound).
 * @returns The penalty, the mean over rows of the squared gap; weight it by `GRADIENT_PENALTY_WEIGHT` in the critic's
 *   loss.
 *
 * @example A linear critic has gradient norm 5 everywhere
 * const critic = (x) => matmul(x, tensor([3, 4]))
 * const real = tensor([[0, 0], [1, 1]])
 * const fake = tensor([[2, 0], [0, 3]])
 * print('penalty =', gradientPenalty(critic, real, fake, tensor([0.5, 0.25])), ' (5 - 1)^2 =', 16)
 *
 * @example Differentiable in the critic's parameters
 * // (|w| - 1)^2 has gradient 2 (|w| - 1) w / |w| = (4.8, 6.4) at w = (3, 4).
 * const real = tensor([[0, 0], [1, 1]])
 * const fake = tensor([[2, 0], [0, 3]])
 * print(grad((w) => gradientPenalty((x) => matmul(x, w), real, fake, tensor([0.5, 0.5])))(tensor([3, 4])))
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

/** The penalty's weight $\lambda$ in the WGAN-GP critic loss, as Gulrajani et al. (2017) use it: 10. */
export const GRADIENT_PENALTY_WEIGHT = 10
