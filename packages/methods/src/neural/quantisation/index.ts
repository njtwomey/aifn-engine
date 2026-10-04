/**
 * `aifn-methods/neural/quantisation`: the memory and decoding throughput of a served transformer at a given
 * precision (`servingMemory`, `decodeThroughput`, the roofline argument), and a streamed study that trains a small MLP
 * and quantises its weights by round-to-nearest (per tensor, per channel), GPTQ and AWQ (`quantisationStudy`), and
 * quantisation-aware training against post-training quantisation (`quantisationAwareTraining`).
 */

export {
  decodeThroughput,
  servingMemory,
  type DecodeThroughput,
  type Device,
  type ServingMemory,
  type ServingSetup,
  type TransformerShape,
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
