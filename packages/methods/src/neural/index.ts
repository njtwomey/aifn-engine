/**
 * `aifn-methods/neural`: neural-network applications built from `aifn-compute/nn`, each small enough to train in a
 * browser.
 *
 * - Language models: `aifn-methods/neural/language-models`, a tiny character-level GPT and an interpolated Kneser–Ney
 *   n-gram model on a toy corpus. This index re-exports them (`charCorpus`, `charGpt`, `Gpt`, `gptLogits`,
 *   `kneserNey`) and collects the registered estimators among them in `neuralModelRegistry`.
 * - Representation learning: `aifn-methods/neural/contrastive`, a tiny CLIP aligning two views by the symmetric
 *   InfoNCE loss; `aifn-methods/neural/graph`, semi-supervised node classification with a GCN, GAT or GraphSAGE.
 * - Training dynamics: `aifn-methods/neural/full-batch`, full-batch L-BFGS against gradient descent, Adam and SGD on a
 *   small MLP; `aifn-methods/neural/grokking`, delayed generalisation on modular arithmetic;
 *   `aifn-methods/neural/privacy`, differentially private training by DP-SGD against the privacy spent.
 * - Continuous depth: `aifn-methods/neural/ode`, neural ODEs, a continuous normalising flow and a latent ODE;
 *   `aifn-methods/neural/ode-mixtures`, neural ODEs with stochastic vector field mixtures.
 * - Adapting and aligning a trained model: `aifn-methods/neural/post-training` (SFT, DPO and GRPO on toy policies, with
 *   GRPO's advantages and pass@$k$), `aifn-methods/neural/adaptation` (fitting a weight change with LoRA or PiSSA),
 *   `aifn-methods/neural/reward-models` (Bradley–Terry reward models and best-of-$n$ over-optimisation) and
 *   `aifn-methods/neural/quantisation` (the memory and compute of serving and of fine-tuning, with LoRA and QLoRA).
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { ModelEntry } from 'aifn-compute/learning/estimators'
import * as languageModels from './language-models'

export { charCorpus, charGpt, Gpt, gptLogits, kneserNey } from './language-models'

/** Every registered estimator factory of `aifn-methods/neural`, keyed by `info.key` (kind `model`). */
export const neuralModelRegistry = entries('model', languageModels) as Readonly<Record<string, ModelEntry>>
