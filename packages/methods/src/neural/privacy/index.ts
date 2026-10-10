/**
 * `aifn-methods/neural/privacy`: differentially private training of a small MLP by DP-SGD, traced as test accuracy
 * against the privacy spent.
 *
 * - The network: `privateStudyModel`, a one-hidden-layer tanh MLP and the map from its flat parameters back to it.
 * - The study: `privateTrainingStudy` trains it by `aifn-compute/nn/training`'s `privateTraining` once per noise
 *   multiplier $\sigma$ (with $\sigma = 0$, clipping only, as the baseline), recording the loss, the test accuracy
 *   and the $\varepsilon$ spent at $\delta$, as a generator of snapshots for a worker.
 *
 * Every run starts from the same weights and is deterministic from the seed. `privacyStudyFunctions` is the module's
 * registry table.
 */

export {
  privateStudyModel,
  privateTrainingStudy,
  type PrivateRun,
  type PrivateStudyOptions,
  type PrivateStudySnapshot,
} from './study'
export { privacyStudyFunctions } from './registry'
