/**
 * `aifn-compute/text`: text processing, from characters to features, every step keeping offsets into the original
 * text. The equivalents elsewhere are NLTK, scikit-learn's `feature_extraction.text`, gensim and Hugging Face
 * `tokenizers`.
 *
 * - `normalise`: Unicode normal forms, full case folding, accent stripping and white space, composed by `normalise`.
 * - `tokenise`: regular-expression tokenisers (words, scikit-learn's default, the GPT-2, cl100k, o200k and BERT
 *   pre-tokenisers), characters, NLTK's Treebank and casual (tweet) tokenisers, Punkt-style sentence splitting and
 *   detokenisation, each token with its offsets.
 * - `stem`: the Porter stemmer of 1980 with a rule-by-rule trace, and NLTK's and scikit-learn's English stop words.
 * - `vocabulary`: token counts and the map between tokens and ids, with special tokens, minimum counts and maximum
 *   sizes.
 * - `subword`: subword tokenisers (byte-pair encoding, WordPiece and the unigram language model) trained as
 *   step-through algorithms, one merge or pruning round per step, and encoding with offsets.
 * - `pipeline`: tokenisers as Hugging Face composes them, normaliser, pre-tokeniser, model, post-processor and decoder,
 *   with truncation, padding and training through the pipeline.
 * - `statistics`: how tokenisers compare on a corpus: fertility, bytes and characters per token, unknown rate, word
 *   coverage and vocabulary usage.
 * - `features`: n-grams, the bag of words, TF-IDF and BM25, feature hashing, one-hot encoding, shingles and MinHash,
 *   and CRF++ feature templates for sequence labellers.
 * - `cooccurrence`: windowed word–context counts, PMI and PPMI, the truncated SVD and word vectors from it, and topic
 *   coherence.
 * - `representations`: term–document and term–term matrices and their weightings, latent semantic analysis, cosine
 *   neighbours, analogies and random indexing.
 *
 * The shared layer, exported here, is aligned text (`aligned`, `alignedReplace`, `alignedMap`, `alignedPrepend`,
 * `alignedSlice`, `alignedConcat`, `originalSpan`): strings that keep, for each code unit, the range of the original
 * it came from through every rewrite. The most used functions of the children are re-exported here too: `normalise`,
 * `tokenise` and `detokenise`, `buildVocabulary`, `encodeTokens` and `decodeTokens`, `bagOfWords`, `tfidf` and `bm25`.
 */

export {
  aligned,
  alignedConcat,
  alignedMap,
  alignedPrepend,
  alignedReplace,
  alignedSlice,
  originalSpan,
  type AlignedText,
  type AlignmentUnit,
} from './aligned'
export { normalise } from './normalise'
export { tokenise, detokenise, type Tokenisation } from './tokenise'
export { buildVocabulary, encodeTokens, decodeTokens, type Vocabulary } from './vocabulary'
export { bagOfWords, tfidf, bm25 } from './features'
