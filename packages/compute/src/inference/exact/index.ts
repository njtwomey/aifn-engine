/**
 * `aifn-compute/inference/exact`: exact inference: enumeration and variable elimination on discrete factor graphs, and the
 * generic chain engines: on a chain of potentials (`ChainPotentials`: `forwardBackward`, `viterbi`, their steps,
 * `sampleHiddenPath` by forward filtering backward sampling), on log-potentials (`chainForwardBackward`,
 * `chainViterbi`, which a linear-chain CRF runs), and on chain-shaped factor graphs (`factorChain`, `chainSumProduct`,
 * which `infer` picks by shape). Named chain models (the HMM, the CRF) are in `aifn-methods/inference/sequence-models`.
 */

export {
  eliminationResult,
  enumerate,
  enumerationSteps,
  jointDistribution,
  variableElimination,
  variableEliminationSteps,
  type EliminationEvent,
  type EliminationOptions,
  type EliminationOrder,
  type EliminationResult,
  type EliminationState,
  type EnumerationOptions,
  type EnumerationState,
  type ExactResult,
} from './exact'
export {
  chainForwardBackward,
  chainSumProduct,
  chainViterbi,
  factorChain,
  forwardBackward,
  forwardBackwardSteps,
  posteriorDecode,
  sampleHiddenPath,
  viterbi,
  viterbiSteps,
  type ChainMarginals,
  type ChainPotentials,
  type ChainSumProductState,
  type FactorChain,
  type ForwardBackwardResult,
  type ForwardBackwardState,
  type PosteriorDecoding,
  type ViterbiResult,
  type ViterbiState,
} from './chain'
export { exactAlgorithms, exactFunctions } from './registry'
