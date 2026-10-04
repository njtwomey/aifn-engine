/**
 * `aifn-compute/text/cooccurrence`: windowed word–context counts, PMI, PPMI with context smoothing and shifting, the
 * truncated SVD, word vectors from it, and the coherence of topics' top words (NPMI, UMass).
 */

export {
  cooccurrence,
  pmi,
  ppmi,
  wordVectors,
  type Cooccurrence,
  type CooccurrenceOptions,
  type PmiOptions,
} from './cooccurrence'
export { topicCoherence, type Coherence, type CoherenceMeasure, type CoherenceOptions } from './coherence'
export { truncatedSvd, type TruncatedSvd } from './svd'
export { cooccurrenceFunctions } from './registry'
