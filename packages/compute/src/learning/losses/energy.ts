/**
 * Losses for energy-based models $p_\theta(\xvec) = \exp(-E_\theta(\xvec)) / Z(\theta)$. The gradient of the
 * negative log-likelihood is
 * $\nabla_\theta E_\theta(\xvec) - \expect_{\xvec' \sim p_\theta}[\nabla_\theta E_\theta(\xvec')]$
 * (Hinton, 2002; LeCun et al., 2006): it pushes the energy down on data and up on the model's own samples, and needs
 * no $Z$. With the expectation replaced by samples $\xvec'$ from a sampler run on the current model (contrastive
 * divergence, or persistent chains as in Tieleman, 2008), the surrogate whose gradient this is reads
 * $\operatorname{mean} E(\xvec) - \operatorname{mean} E(\xvec')$, with the samples held constant.
 */

import { add, mean, mul, square, sub, type Value } from 'aifn-compute/foundation/tensor'
import { defineLoss } from './core'

/** Options of `contrastiveDivergenceLoss`. */
export type ContrastiveDivergenceOptions = {
  /**
   * Weight $\alpha$ of the energy-magnitude penalty
   * $\alpha(\operatorname{mean} E(\xvec)^2 + \operatorname{mean} E(\xvec')^2)$ of Du & Mordatch (2019, §3.3), which
   * keeps the energies near zero; the surrogate leaves them free up to a constant. Default 0.
   */
  regularisation?: number
}

/**
 * The contrastive-divergence surrogate $\operatorname{mean} E(\xvec) - \operatorname{mean} E(\xvec')$ from the
 * energies of data points (`positive`) and of samples from the model (`negative`), each of shape `[n]` (or `[n, 1]`);
 * its gradient in $\theta$ is the maximum-likelihood gradient's Monte Carlo estimate when the negatives are exact model
 * samples. The negatives' positions must be constants (sampled, not traced); their energies are traced in $\theta$.
 * There is no `reduction`: the means are the loss.
 *
 * @param positive The energies $E_\theta(\xvec)$ of the data points.
 * @param negative The energies $E_\theta(\xvec')$ of the model's samples; the two batches may differ in size.
 * @param options The weight $\alpha$ of the energy-magnitude penalty.
 * @returns The surrogate, a number (or a traced scalar).
 *
 * @example Low energy on data, high on samples, with and without the penalty
 * const data = tensor([-1, -2])
 * const samples = tensor([1, 0])
 * print('loss =', contrastiveDivergenceLoss(data, samples), ' -1.5 - 0.5 =', -2)
 * const penalised = contrastiveDivergenceLoss(data, samples, { regularisation: 0.1 })
 * print('alpha = 0.1:', penalised, ' -2 + 0.1 (2.5 + 0.5) =', -1.7)
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
