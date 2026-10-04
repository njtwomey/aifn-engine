/**
 * `aifn-methods/neural/privacy`: differentially private training of a small MLP by compute DP-SGD (`privateTraining`),
 * one run per noise multiplier (`privateTrainingStudy`, a generator of snapshots for a worker), tracing test accuracy
 * against the ε spent.
 */

export {
  privateStudyModel,
  privateTrainingStudy,
  type PrivateRun,
  type PrivateStudyOptions,
  type PrivateStudySnapshot,
} from './study'
export { privacyStudyFunctions } from './registry'
