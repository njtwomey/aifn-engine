/**
 * `aifn-methods/inference/conjugate-models`: coordinate-ascent variational inference for the normal–gamma model.
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
