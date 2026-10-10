/**
 * `aifn-methods/inference/sequence-models`: hidden Markov models and linear-chain conditional random fields, the
 * models that label each position of a sequence.
 *
 * - Hidden Markov models: `hmm` builds an `Hmm` from $\pivec$, $\Amat$ and $\Bmat$, and `dishonestCasino` is the
 *   textbook one; `sampleHmm` draws a path and its observations; `hmmChain` gives the chain of potentials that the
 *   engines of `aifn-compute/inference/exact` run on (`forwardBackward`, `viterbi`), and `hmmModel` the same HMM in
 *   the model language, for `infer`.
 * - The linear-chain CRF over dense feature vectors: `linearChainCrf`, its structure (`crfStructure`) and factor graph
 *   (`crfFactorGraph`), both chain-shaped; `crfPotentials`, `crfMarginals` (forward–backward), `crfViterbi`, and for
 *   learning `crfScore`, `crfLogLikelihood` and `crfGradient` (observed minus expected counts).
 * - The CRF over CRF++-style feature templates: `crfProblem` indexes and encodes the data (with `labelSet`,
 *   `encodeLabelled` and `crfWeightCount`) and `templateCrf` holds the weights; decode with `templateCrfViterbi` (the
 *   most probable labelling) or `templateCrfPosterior` (the most probable label at each position), from
 *   `templateCrfMarginals`, `templateCrfPotentials`, `templateCrfScore` and `crfLogPotentials`.
 * - Training the template CRF: `crfTraining`, a step-through algorithm by L-BFGS, OWL-QN (L1 and L2), SGD or Adam on
 *   `crfObjective` (`crfNegLogLikelihood` plus the L2 penalty), with `crfppRegularisation` translating CRF++'s `-c`;
 *   `crfTrainingRun` streams a whole run from token rows.
 * - Reading a trained template CRF: `firingFeatures` at a position, `topFeatures` per label, `transitionWeights` and
 *   `activeWeights`.
 * - A toy part-of-speech tagging task: `toyPosCorpus`, with `TOY_POS_TAGS` and `TOY_POS_TEMPLATES`, and `posRows`,
 *   which turns a sentence into token rows.
 * - `sequenceModelFunctions` and `sequenceModelAlgorithms` are the module's registry entries.
 *
 * States, symbols and the labels of the dense CRF are integer ids from 0; the template CRF's labels are strings, with
 * ids by position in its label set. Inference is exact on the chain, in log space for the CRFs. Bad input throws
 * `ShapeError` or `DomainError`; none of the module is differentiable by `grad` (the CRFs give their gradients).
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
