/**
 * `aifn-methods/neural/quantisation`: what precision costs and buys, from back-of-envelope sizes of large transformers
 * to the measured accuracy of a small quantised network.
 *
 * - Serving: `servingMemory` (weights and key–value cache at a given precision, and whether they fit a device) and
 *   `decodeThroughput` (the roofline: small-batch decoding is bound by memory bandwidth, large-batch by arithmetic).
 * - Fine-tuning: `fineTuningMemory` (model state, activations and logits of full fine-tuning, LoRA and QLoRA) and
 *   `fineTuningCompute` (FLOPs, time and price), with the counts `linearParameters` (what QLoRA stores in NF4) and
 *   `adapterParameters` (LoRA's trainable parameters), and the shapes `LLAMA_3_1_8B` and `LLAMA_3_1_70B`.
 * - Post-training quantisation: `quantisationStudy` trains a small ReLU MLP (`studyModel`) and rounds its weights per
 *   tensor, per channel, by GPTQ and by AWQ (`QUANTISATION_METHODS`), reporting accuracy and SQNR at each bit width.
 * - Quantisation-aware training: `quantisationAwareTraining` compares rounding a trained network once (PTQ) with
 *   fine-tuning it through fake-quantised weights and the straight-through estimator (QAT).
 *
 * Sizes are in bytes and a GB is $10^9$ bytes. The two studies are generators of snapshots, for a worker to stream to
 * a page, deterministic from their seed. `quantisationFunctions` is the module's registry table.
 */

export {
  adapterParameters,
  decodeThroughput,
  fineTuningCompute,
  fineTuningMemory,
  linearParameters,
  LLAMA_3_1_70B,
  LLAMA_3_1_8B,
  servingMemory,
  type AdapterTarget,
  type DecodeThroughput,
  type Device,
  type FineTuningCompute,
  type FineTuningMemory,
  type FineTuningRun,
  type FineTuningSetup,
  type ServingMemory,
  type ServingSetup,
  type TransformerShape,
  type TransformerTrainShape,
} from './cost'
export {
  QUANTISATION_METHODS,
  quantisationStudy,
  studyModel,
  type BitResult,
  type QuantisationMethod,
  type QuantisationSnapshot,
  type QuantisationStudyOptions,
} from './study'
export { quantisationAwareTraining, type QatBitResult, type QatOptions, type QatSnapshot } from './qat'
export { quantisationFunctions } from './registry'
