/**
 * `aifn-compute/nn/decoding`: decoding a language model over any `logits(prefix)` function, one token per step. Greedy
 * decoding, sampling with temperature, top-k, top-p and a repetition penalty, beam search with a length penalty (and
 * its search tree), and speculative decoding with a draft model and an acceptance trace; the logit processors on
 * their own, and the next-token distribution before and after them.
 */

export {
  beamSearch,
  expectedTokensPerCall,
  greedyDecoding,
  lengthPenalty,
  samplingDecoding,
  speculativeDecoding,
  type BeamNode,
  type BeamOptions,
  type BeamState,
  type DecodingOptions,
  type DecodingState,
  type Hypothesis,
  type SpeculativeOptions,
  type SpeculativeRound,
  type SpeculativeState,
} from './decoders'
export {
  applyRepetitionPenalty,
  applyTemperature,
  applyTopK,
  applyTopP,
  logitsOf,
  nextTokenDistribution,
  softmaxOf,
  type LogitsFn,
  type NextTokenDistribution,
  type SamplingOptions,
} from './processors'
export { decodingAlgorithms, decodingFunctions } from './registry'
