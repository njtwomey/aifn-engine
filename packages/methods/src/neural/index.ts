/**
 * `aifn-methods/neural`: neural-network applications built from `aifn-compute/nn`: language models (a tiny
 * character-level GPT and a Kneser–Ney n-gram model, on a toy corpus) and contrastive alignment of two views (a tiny
 * CLIP, in `aifn-methods/neural/contrastive`). Adapting and aligning a trained model has its own modules:
 * `aifn-methods/neural/post-training` (SFT, DPO and GRPO on toy policies, with GRPO's advantages and pass@$k$),
 * `aifn-methods/neural/adaptation` (fitting a weight change with LoRA or PiSSA), `aifn-methods/neural/reward-models`
 * (Bradley–Terry reward models and best-of-$n$ over-optimisation) and `aifn-methods/neural/quantisation` (the memory
 * and compute of serving and of fine-tuning, with LoRA and QLoRA).
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { ModelEntry } from 'aifn-compute/learning/estimators'
import * as languageModels from './language-models'

export { charCorpus, charGpt, Gpt, gptLogits, kneserNey } from './language-models'

/** Every registered estimator factory of `aifn-methods/neural`, keyed by `info.key` (kind `model`). */
export const neuralModelRegistry = entries('model', languageModels) as Readonly<Record<string, ModelEntry>>
