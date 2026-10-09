/**
 * `aifn-compute/inference/exact`: exact inference on discrete factor graphs, and the generic chain engines.
 *
 * - Any small factor graph: `enumerate` (marginals, $\log Z$ and the MAP by visiting every assignment; exponential in
 *   the number of variables, the reference for other engines), `jointDistribution` (the whole normalised table), and
 *   `variableElimination` (a query marginal given evidence, by sum or max; exponential only in the width of the order,
 *   fixed, min-degree or min-fill). Stepped: `enumerationSteps`, `variableEliminationSteps` with `eliminationResult`.
 * - A chain of potentials (`ChainPotentials`: $N \times K$ node potentials $\psi$, a shared $K \times K$ transition
 *   $\Amat$): `forwardBackward` (scaled; filtered, smoothed and pairwise marginals and $\log Z$), `viterbi` (the most
 *   probable path), `sampleHiddenPath` (an exact draw by forward filtering, backward sampling), and the stepped
 *   `forwardBackwardSteps` and `viterbiSteps`.
 * - A chain of log-potentials, which a linear-chain CRF runs: `chainForwardBackward`, `chainViterbi` (the pairwise
 *   term shared or one per step), and `posteriorDecode` (each position's most probable label).
 * - A chain-shaped factor graph: `factorChain` reads it as chain log-potentials, and `chainSumProduct` steps
 *   sum-product along it; `infer` picks it by shape.
 *
 * Potentials are non-negative tables (not logs) unless a name says log; marginals are float64 and paths int32, and
 * ties go to the smaller state. None of it is differentiable. Named chain models (the HMM, the CRF) are in
 * `aifn-methods/inference/sequence-models`.
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
