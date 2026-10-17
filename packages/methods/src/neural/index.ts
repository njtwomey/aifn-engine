/**
 * `aifn-methods/neural`: neural-network applications built from `aifn-compute/nn`, each small enough to train in a
 * browser.
 *
 * - Language models: `aifn-methods/neural/language-models`, a tiny character-level GPT and an interpolated Kneser–Ney
 *   n-gram model on a toy corpus. This index re-exports them (`charCorpus`, `charGpt`, `Gpt`, `gptLogits`,
 *   `kneserNey`) and collects the registered estimators among them in `neuralModelRegistry`.
 * - Representation learning: `aifn-methods/neural/contrastive`, a tiny CLIP aligning two views by the symmetric
 *   InfoNCE loss; `aifn-methods/neural/graph`, semi-supervised node classification with a GCN, GAT or GraphSAGE;
 *   `aifn-methods/neural/self-supervised`, self-distillation (DINO) and the self-supervised objectives of a linear
 *   code and of contrastive predictive coding.
 * - Few labels: `aifn-methods/neural/semi-supervised`, consistency regularisation ($\Pi$-model, temporal ensembling,
 *   mean teacher, VAT, UDA, MixMatch, FixMatch and its adaptive thresholds) of a small classifier.
 * - Training dynamics: `aifn-methods/neural/full-batch`, full-batch L-BFGS against gradient descent, Adam and SGD on a
 *   small MLP; `aifn-methods/neural/grokking`, delayed generalisation on modular arithmetic;
 *   `aifn-methods/neural/privacy`, differentially private training by DP-SGD against the privacy spent.
 * - Continuous depth: `aifn-methods/neural/ode`, neural ODEs, a continuous normalising flow and a latent ODE;
 *   `aifn-methods/neural/ode-mixtures`, neural ODEs with stochastic vector field mixtures.
 * - Physics: `aifn-methods/neural/pinn`, physics-informed networks that solve Burgers' and the heat equation from the
 *   equation alone, against their exact solutions, and their variants (hard conditions, inverse problems, loss
 *   balancing, resampling, causal weights); `aifn-methods/neural/operator`, operator learning (DeepONet, the Fourier
 *   neural operator, mesh message passing) against closed-form linear baselines.
 * - Efficient sequence models: `aifn-methods/neural/sequence`, associative recall by state size, parity at finite
 *   precision, and low-rank and hashing approximations of the attention matrix.
 * - Convolutional networks: `aifn-methods/neural/convolutional`, worked forms of Winograd and fast convolution with
 *   their cost model, deformable convolution at one position, and capsule routing by agreement.
 * - Adapting and aligning a trained model: `aifn-methods/neural/post-training` (SFT, DPO and GRPO on toy policies, with
 *   GRPO's advantages and pass@$k$), `aifn-methods/neural/adaptation` (fitting a weight change with LoRA or PiSSA),
 *   `aifn-methods/neural/reward-models` (Bradley–Terry reward models and best-of-$n$ over-optimisation) and
 *   `aifn-methods/neural/quantisation` (the memory and compute of serving and of fine-tuning, with LoRA and QLoRA).
 * - Tuning: `aifn-methods/neural/tuning`, a learning-rate sweep of a small network over rates and seeds, run in a
 *   worker.
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { ModelEntry } from 'aifn-compute/learning/estimators'
import * as languageModels from './language-models'

export { charCorpus, charGpt, Gpt, gptLogits, kneserNey } from './language-models'

/** Every registered estimator factory of `aifn-methods/neural`, keyed by `info.key` (kind `model`). */
export const neuralModelRegistry = entries('model', languageModels) as Readonly<Record<string, ModelEntry>>
