/**
 * `aifn-compute/text/features`: text as numbers, from n-grams to weighted document–term matrices, hashed features,
 * shingles and sequence-labelling templates.
 *
 * - N-grams: `wordNgrams` and `characterNgrams` (across the text, or inside padded words as scikit-learn's `char_wb`).
 * - Counting: `bagOfWords`, the document–term matrix of counts or presence with its vocabulary (minimum and maximum
 *   document frequency, maximum size, as `CountVectorizer`); `oneHotTokens`, a token sequence as one-hot columns.
 * - Weighting: `tfidf` with SMART's term-frequency, document-frequency and normalisation variants (`smartWeighting`
 *   reads a code such as `ltc`; the defaults equal scikit-learn's `TfidfVectorizer`), its parts `termFrequency`,
 *   `documentFrequency` and `inverseDocumentFrequency`, and `bm25Weights` and `bm25` (BM25 and BM25+) for ranking.
 * - Hashing: `featureHash` (dense rows) and `hashedFeatures` (one sparse row) with scikit-learn's signed
 *   `HashingVectorizer` conventions, built on `murmurHash3` and `hashColumn`.
 * - Near duplicates: `characterShingles` and `wordShingles` with `jaccardSimilarity`; `minHashSignature`,
 *   `minHashSignatures`, `minHashSimilarity` and `minHashStandardError` estimate it. The LSH banding of signatures is
 *   in `aifn-compute/numerics/neighbours`.
 * - Sequence features: CRF++ templates (`%x[r,c]` macros over token rows) parsed by `parseTemplates` (throwing
 *   `TemplateSyntaxError`), read by `templateCell`, `expandTemplate` and `expandTemplates`, indexed over training data
 *   by `featureIndex` and turned into ids by `encodeTemplateRows`.
 *
 * Documents are token lists (tokenise first); matrices are dense float64 tensors, one row per document. Invalid input
 * throws `DomainError`.
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
