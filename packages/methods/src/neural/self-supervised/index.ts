/**
 * `aifn-methods/neural/self-supervised`: learning representations without labels, on problems small enough to see
 * the whole objective.
 *
 * - Self-distillation: `selfDistillationTraining`, DINO's student and EMA teacher (`parameterEma` of
 *   `aifn-compute/nn/training`) with centring and sharpening, reporting the collapse diagnostics (the teacher's
 *   conditional and marginal entropies, and the teacher-to-student KL divergence).
 * - The objective families for a one-number linear encoder $z = \wvec^\top \xvec$: `linearCodeObjective`
 *   (reconstruction, InfoNCE, view agreement and VICReg's variance hinge), `linearCodeLandscape` (the objective on a
 *   grid of encoders), `inputCovariance` and `thresholdProbeAccuracy` (a linear probe of the code).
 * - Contrastive predictive coding's optimum for a Gaussian series: `linearPredictiveCode` (the window code most
 *   informative about the value $k$ steps ahead, with $I = -\tfrac12 \log(1 - \rho^2)$), `codeCorrelation` and
 *   `applyPredictiveCode`.
 * - `selfSupervisedAlgorithms` and `selfSupervisedFunctions`: the module's registry entries.
 *
 * Arrays are row-major `Float64Array`s; every draw comes from the step's stream, so a run is deterministic in its seed.
 */

export {
  inputCovariance,
  linearCodeLandscape,
  linearCodeObjective,
  thresholdProbeAccuracy,
  type LinearCodeObjective,
  type LinearCodeProblem,
} from './linear-codes'
export { applyPredictiveCode, codeCorrelation, linearPredictiveCode, type PredictiveCode } from './predictive-code'
export { selfDistillationTraining, type SelfDistillationOptions, type SelfDistillationState } from './self-distillation'
export { selfSupervisedAlgorithms, selfSupervisedFunctions } from './registry'
