/**
 * `aifn-compute/nn/decoding`: decoding a language model over any `logits(prefix)` function, one token per step.
 *
 * - Decoders, each a step-through algorithm: `greedyDecoding` (the most probable token, deterministic),
 *   `samplingDecoding` (draws from the processed distribution with the step's stream), `beamSearch` (the $B$ best
 *   partial sequences with GNMT's `lengthPenalty`, and its search tree for drawing) and `speculativeDecoding` (a
 *   draft model verified by the target, with an acceptance trace; `expectedTokensPerCall` gives its speed-up).
 * - Logit processors, which return logits with $-\infty$ on removed tokens so they compose: `applyRepetitionPenalty`,
 *   `applyTemperature`, `applyTopK`, `applyTopP`; `nextTokenDistribution` applies them in Hugging Face's order and
 *   reports the distribution before and after.
 * - Helpers: `logitsOf` (a model's output as numbers) and `softmaxOf`.
 *
 * A model is a `LogitsFn` from the prompt and the tokens so far to the next-token logits `[V]`. Decoding works on
 * concrete numbers and is not differentiated; decoder states are plain data, and randomness comes from the runner's
 * per-step stream, so a decoding can be replayed exactly. `decodingAlgorithms` and `decodingFunctions` are the
 * registry entries.
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
