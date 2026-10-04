/**
 * `aifn-methods/inference/sequence-models`: sequence models: the hidden Markov model (`Hmm`, `hmm`, its chain of
 * potentials `hmmChain` for the generic engines of `aifn-compute/inference/exact`, the occasionally dishonest casino,
 * `sampleHmm`, `hmmModel` in the model language), and the linear-chain
 * CRF with its structure (`crfStructure`) and factor graph (`crfFactorGraph`), both chain-shaped; and the linear-chain
 * CRF over CRF++-style feature templates (`templateCrf`, `crfTraining` by L-BFGS, OWL-QN, SGD or Adam, `crfTrainingRun`).
 */

export {
  crfFactorGraph,
  crfGradient,
  crfLogLikelihood,
  crfMarginals,
  crfPotentials,
  crfScore,
  crfStructure,
  crfViterbi,
  linearChainCrf,
  type CrfGradient,
  type LinearChainCrf,
} from './crf'
export {
  activeWeights,
  crfLogPotentials,
  crfNegLogLikelihood,
  crfObjective,
  crfppRegularisation,
  crfProblem,
  crfTraining,
  crfTrainingRun,
  crfWeightCount,
  encodeLabelled,
  firingFeatures,
  labelSet,
  templateCrf,
  templateCrfMarginals,
  templateCrfPosterior,
  templateCrfPotentials,
  templateCrfScore,
  templateCrfViterbi,
  topFeatures,
  transitionWeights,
  type CrfOptimizer,
  type CrfProblem,
  type CrfSnapshot,
  type CrfTrainingOptions,
  type CrfTrainingState,
  type EncodedLabelled,
  type FiringFeature,
  type FitTemplateCrfOptions,
  type LabelledSequence,
  type TemplateCrf,
  type WeightedString,
} from './template-crf'
export { posRows, TOY_POS_TAGS, TOY_POS_TEMPLATES, toyPosCorpus } from './toy-tagging'
export { dishonestCasino, hmm, hmmChain, hmmModel, sampleHmm, type Hmm } from './hmm'
export { sequenceModelAlgorithms, sequenceModelFunctions } from './registry'
