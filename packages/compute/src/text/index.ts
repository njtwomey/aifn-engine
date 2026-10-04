/**
 * `aifn-compute/text`: text processing, from characters to features. Children: normalise (Unicode forms, case folding,
 * accents), tokenise (regular-expression, Treebank and casual tokenisers with offsets; sentence splitting), stem
 * (Porter, stop words), vocabulary (token ↔ id), subword (BPE, WordPiece and unigram-LM training as step-through
 * algorithms, and encoding), pipeline (tokenisers as normaliser → pre-tokeniser → model → post-processor → decoder),
 * statistics (fertility, compression, coverage), features (n-grams, bag of words, TF-IDF, BM25, feature hashing,
 * one-hot, shingles, MinHash and LSH), cooccurrence (windowed counts, PMI, PPMI, truncated SVD, SVD word vectors) and
 * representations (term–document and term–term matrices, LSA, cosine neighbours, analogies, random indexing). The shared layer is aligned text: strings that keep
 * offsets into the original through every rewrite.
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
