/**
 * `aifn-methods/inference/conjugate-models`: coordinate-ascent variational inference for the normal-gamma model, beside
 * its exact posterior.
 *
 * - The exact answer: `normalGammaPosterior` gives the conjugate posterior of a Gaussian's mean $\mu$ and precision
 *   $\tau$ under a `NormalGammaPrior` (`defaultNormalGammaPrior` by default), its log evidence and the marginal
 *   moments.
 * - The approximation: `caviNormalGamma` steps the mean-field $q(\mu)q(\tau)$ by coordinate ascent, reporting the
 *   ELBO and its exact gap $\KL(q \,\Vert\, p)$ to the posterior at every sweep.
 * - `conjugateModelAlgorithms` and `conjugateModelFunctions` register them with the notes they serve.
 */

export {
  caviNormalGamma,
  defaultNormalGammaPrior,
  normalGammaPosterior,
  type CaviNormalGammaState,
  type NormalGammaPosterior,
  type NormalGammaPrior,
} from './cavi'
export { conjugateModelAlgorithms, conjugateModelFunctions } from './registry'
