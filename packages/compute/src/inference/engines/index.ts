/**
 * `aifn-compute/inference/engines`: automatic inference, choosing an engine by the shape of the model.
 *
 * - `infer(model, bindings, options)` picks the first engine of a table that matches the model: forward–backward on a
 *   chain, exact belief propagation on a tree and loopy belief propagation on any other discrete model, expectation
 *   propagation on a linear-Gaussian model with interval or Gaussian evidence, and Gibbs sampling otherwise. An
 *   engine can also be named (`variable-elimination` and `enumeration` run only when named).
 * - `builtInEngines` is that table, as static data; applications put their own engines in front of it with
 *   `withEngines` (LDA in `aifn-methods/inference/topic-models`), never by registering into a global.
 * - The result is an algorithm with no start, run with the runners of `aifn-compute/foundation/trace`. No engine
 *   matching, or no engine of the name asked for, throws `DomainError`.
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
