/**
 * Losses for energy-based models p_θ(x) = exp(−E_θ(x))/Z(θ). The gradient of the negative log-likelihood is
 * ∇θ E_θ(x) − E_{x′~p_θ}[∇θ E_θ(x′)] (Hinton, 2002; LeCun et al., 2006): it pushes the energy down on data and up on
 * the model's own samples, and needs no Z. With the expectation replaced by samples x′ from a sampler run on the
 * current model (contrastive divergence, or persistent chains as in Tieleman, 2008), the surrogate whose gradient this
 * is reads mean E(x) − mean E(x′), with the samples held constant.
 */

import { add, mean, mul, square, sub, type Value } from 'aifn-compute/foundation/tensor'
import { defineLoss } from './core'

/** Options of `contrastiveDivergenceLoss`. */
export type ContrastiveDivergenceOptions = {
  /**
   * Weight α of the energy-magnitude penalty α(mean E(x)² + mean E(x′)²) of Du & Mordatch (2019, §3.3), which keeps
   * the energies near zero; the surrogate leaves them free up to a constant. Default 0.
   */
  regularisation?: number
}

/**
 * The contrastive-divergence surrogate mean E(x) − mean E(x′) from the energies of data points (`positive`) and of
 * samples from the model (`negative`), each of shape [n] (or [n, 1]); its gradient in θ is the maximum-likelihood
 * gradient's Monte Carlo estimate when the negatives are exact model samples. The negatives' positions must be
 * constants (sampled, not traced); their energies are traced in θ.
 */
export const contrastiveDivergenceLoss = defineLoss(
  {
    key: 'contrastiveDivergenceLoss',
    name: 'Contrastive divergence',
    family: 'energy',
    inputs: 'values',
    notes: ['energy-based-models'],
    cite: ['tieleman2008', 'du2019', 'lecun2006'],
    target: 'the energy −log p(x) up to a constant',
  },
  (positive: Value, negative: Value, { regularisation = 0 }: ContrastiveDivergenceOptions = {}): Value => {
    const gap = sub(mean(positive), mean(negative))
    if (regularisation === 0) return gap
    return add(gap, mul(regularisation, add(mean(square(positive)), mean(square(negative)))))
  },
)
