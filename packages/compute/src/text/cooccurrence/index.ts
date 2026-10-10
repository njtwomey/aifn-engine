/**
 * `aifn-compute/text/cooccurrence`: word–context co-occurrence counts, PMI and the vectors and scores built on them.
 *
 * - Counting: `cooccurrence` counts context words in a window (symmetric or asymmetric, uniform or weighted by
 *   distance as GloVe, word2vec or HAL) into a word $\times$ context matrix with its vocabularies.
 * - Association: `pmi`, and `ppmi` with context-distribution smoothing ($\alpha = 0.75$ as word2vec) and shifting by
 *   $\log k$ (Levy and Goldberg's link to skip-gram with $k$ negative samples).
 * - Vectors: `truncatedSvd` (the $k$ largest singular triplets, signed as scikit-learn's, with energy shares) and
 *   `wordVectors`, the rows of $\Umat_d \Sigmamat_d^p$.
 * - Topic quality: `topicCoherence`, NPMI or UMass coherence of topics' top words on a reference corpus, as gensim.
 *
 * Matrices are dense float64 tensors; probabilities are count shares. Invalid input (negative counts, empty matrices,
 * bad window sizes or ranks) throws `DomainError`.
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
