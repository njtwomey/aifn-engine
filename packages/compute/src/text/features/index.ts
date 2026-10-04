/**
 * `aifn-compute/text/features`: text as numbers. Word and character n-grams, the bag of words (counts or presence), TF-IDF
 * with SMART's term-frequency, document-frequency and normalisation variants, BM25 and BM25+, and signed feature
 * hashing with scikit-learn's MurmurHash3 conventions; one-hot encoding; character and word shingles with Jaccard
 * similarity, MinHash signatures and LSH banding; CRF++-style feature templates (`%x[r,c]` macros over token
 * rows) expanded into indexed feature strings for sequence labellers.
 */

export { characterNgrams, wordNgrams, type CharacterNgramOptions, type NgramRange } from './ngrams'
export { bagOfWords, type BagOfWords, type BagOfWordsOptions } from './bag'
export {
  bm25,
  bm25Weights,
  documentFrequency,
  inverseDocumentFrequency,
  smartWeighting,
  termFrequency,
  tfidf,
  type Bm25Options,
  type IdfScheme,
  type NormScheme,
  type TfidfOptions,
  type TfScheme,
} from './weighting'
export { featureHash, hashColumn, hashedFeatures, murmurHash3, type HashingOptions } from './hashing'
export { oneHotTokens } from './onehot'
export { characterShingles, jaccardSimilarity, wordShingles } from './shingles'
export {
  minHashSignature,
  minHashSignatures,
  minHashSimilarity,
  minHashStandardError,
  type MinHashOptions,
} from './minhash'
export {
  encodeTemplateRows,
  expandTemplate,
  expandTemplates,
  featureIndex,
  MAX_TEMPLATE_OFFSET,
  parseTemplates,
  templateCell,
  TemplateSyntaxError,
  type EncodedSequence,
  type ExpandedSequence,
  type FeatureIndex,
  type FeatureIndexOptions,
  type FeatureTemplate,
  type FeatureTemplates,
  type TemplateMacro,
  type TokenRows,
} from './templates'
export { featuresFunctions } from './registry'
