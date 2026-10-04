/**
 * `aifn-compute/inference/engines`: `infer(model, bindings, options)` picks an engine by the shape of the model's structured
 * graph (forward–backward on a chain, exact BP on a tree, loopy BP otherwise, Gibbs for continuous latents) from a
 * static engine table; applications pass extra engines with `withEngines` (LDA in
 * `aifn-methods/inference/topic-models`).
 */

export {
  builtInEngines,
  infer,
  withEngines,
  type EngineRegistration,
  type EngineTable,
  type InferOptions,
  type Inference,
  type InferenceContext,
} from './engines'
export { enginesFunctions } from './registry'
