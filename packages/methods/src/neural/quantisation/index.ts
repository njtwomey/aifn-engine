/**
 * `aifn-methods/neural/quantisation`: the memory and decoding throughput of a served transformer at a given
 * precision (`servingMemory`, `decodeThroughput`, the roofline argument); the memory, compute and price of
 * fine-tuning one fully, with LoRA or with QLoRA (`fineTuningMemory`, `fineTuningCompute`, `adapterParameters`,
 * `linearParameters`, with the shapes `LLAMA_3_1_8B` and `LLAMA_3_1_70B`); and a streamed study that trains a small MLP
 * and quantises its weights by round-to-nearest (per tensor, per channel), GPTQ and AWQ (`quantisationStudy`), and
 * quantisation-aware training against post-training quantisation (`quantisationAwareTraining`).
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
