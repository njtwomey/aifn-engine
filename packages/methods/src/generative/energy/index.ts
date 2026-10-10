/**
 * `aifn-methods/generative/energy`: energy-based models on toy data, centred on JEM (Grathwohl et al., 2019): a
 * softmax classifier's logits read as an energy $E(\xvec) = -\operatorname{logsumexp}_y f(\xvec)_y$, trained with
 * cross-entropy plus the contrastive-divergence term of $\log p(\xvec)$.
 *
 * - The classifier as an energy model: `classifier` builds the network, `classifierLogits` gives $f(\xvec)$,
 *   `classifierEnergy` the energy of $p(\xvec)$ and `classEnergy` that of $p(\xvec \mid y)$; `classifierScore` is
 *   $\nabla_{\xvec} \log p(\xvec)$ or $\nabla_{\xvec} \log p(\xvec \mid y)$, for Langevin sampling.
 * - Training: `jemTraining` is JEM, or plain cross-entropy on the same minibatches, as a traceable algorithm;
 *   `uniformBox` draws the replay buffer's restarts.
 * - The free shift: `logitShift` gives a $c(\xvec)$ which, added to every logit, leaves $p(y \mid \xvec)$ unchanged
 *   and reshapes $p(\xvec)$.
 * - A streamed run for a worker: `jemRun` (both models side by side, with fields, samples, calibration and
 *   out-of-distribution scores at checkpoints).
 *
 * Points are the rows of an $n \times d$ batch (2-d in `jemRun`); the energy and the score are differentiable by
 * autodiff, and every random draw comes from a `Stream`.
 */

export {
  classEnergy,
  classifier,
  classifierEnergy,
  classifierScore,
  jemTraining,
  classifierLogits,
  logitShift,
  uniformBox,
  type Classifier,
  type ClassifierOptions,
  type JemTrainingOptions,
  type LogitShift,
} from './jem'
export { jemRun, type JemCheckpoint, type JemData, type JemRun, type JemRunOptions, type JemTrack } from './run'
export { energyAlgorithms, energyFunctions } from './registry'
