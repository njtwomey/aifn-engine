/** The functions of `aifn-compute/text/features`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bag from './bag'
import * as hashing from './hashing'
import * as minhash from './minhash'
import * as ngrams from './ngrams'
import * as onehot from './onehot'
import * as shingles from './shingles'
import * as templates from './templates'
import * as weighting from './weighting'

const fn = definer<FunctionInfo>('function', 'text/features')
const TFIDF = ['term-frequency-inverse-document-frequency', 'term-weighting-variants']

fn(
  {
    key: 'wordNgrams',
    name: 'Word n-grams',
    role: 'transform',
    summary: 'Runs of n consecutive tokens, for n in a range.',
    notes: ['word-n-grams'],
    cite: ['jurafsky2025'],
  },
  ngrams.wordNgrams,
)
fn(
  {
    key: 'characterNgrams',
    name: 'Character n-grams',
    role: 'transform',
    summary: 'Runs of n characters across the text, or inside space-padded words.',
    notes: ['character-n-grams-and-shingles'],
    cite: ['bojanowski2017'],
  },
  ngrams.characterNgrams,
)
fn(
  {
    key: 'bagOfWords',
    name: 'Bag of words',
    role: 'transform',
    returns: 'bag-of-words',
    summary: 'The document–term matrix of counts or presence over a vocabulary.',
    notes: ['bag-of-words'],
    cite: ['salton1975', 'manning2008'],
  },
  bag.bagOfWords,
)
fn(
  { key: 'documentFrequency', name: 'Document frequency', tex: '\\mathrm{df}_t', role: 'estimator', notes: TFIDF },
  weighting.documentFrequency,
)
fn(
  {
    key: 'inverseDocumentFrequency',
    name: 'Inverse document frequency',
    tex: '\\mathrm{idf}_t',
    role: 'estimator',
    summary: 'Standard, smoothed, probabilistic, Robertson–Spärck Jones and non-negative IDF.',
    notes: TFIDF,
    cite: ['sparckjones1972', 'robertson2009', 'pedregosa2011'],
  },
  weighting.inverseDocumentFrequency,
)
fn(
  {
    key: 'termFrequency',
    name: 'Term-frequency weighting',
    role: 'transform',
    summary: 'Raw, binary, logarithmic, augmented and log-average term frequency.',
    notes: ['term-weighting-variants'],
    cite: ['salton1988', 'manning2008'],
  },
  weighting.termFrequency,
)
fn(
  {
    key: 'tfidf',
    name: 'TF-IDF',
    role: 'transform',
    summary: 'Term frequency × inverse document frequency, each document normalised.',
    notes: TFIDF,
    cite: ['salton1988', 'sparckjones1972', 'pedregosa2011'],
  },
  weighting.tfidf,
)
fn(
  {
    key: 'smartWeighting',
    name: 'SMART weighting code',
    role: 'construction',
    summary: 'TF-IDF options from a SMART code such as ltc.',
    notes: ['term-weighting-variants'],
    cite: ['salton1988', 'manning2008'],
  },
  weighting.smartWeighting,
)
fn(
  {
    key: 'bm25Weights',
    name: 'BM25 term weights',
    role: 'transform',
    summary: 'IDF times saturated, length-normalised term frequency, per document and term; BM25+ with δ.',
    notes: ['term-frequency-inverse-document-frequency'],
    cite: ['robertson2009'],
  },
  weighting.bm25Weights,
)
fn(
  {
    key: 'bm25',
    name: 'BM25 scores',
    role: 'transform',
    summary: "Each document's BM25 (or BM25+) score for a query.",
    notes: ['term-frequency-inverse-document-frequency'],
    cite: ['robertson2009'],
  },
  weighting.bm25,
)
fn(
  {
    key: 'murmurHash3',
    name: 'MurmurHash3 (x86, 32-bit)',
    role: 'transform',
    summary: 'The 32-bit MurmurHash3 of a string’s UTF-8 bytes, as scikit-learn hashes features.',
    notes: ['feature-hashing-for-text'],
    cite: ['pedregosa2011'],
  },
  hashing.murmurHash3,
)
fn(
  { key: 'hashColumn', name: 'Hashed column and sign', role: 'transform', notes: ['feature-hashing-for-text'] },
  hashing.hashColumn,
)
fn(
  {
    key: 'hashedFeatures',
    name: 'Hashed features of a document (sparse)',
    role: 'transform',
    summary: 'The non-zero columns and signed counts of one hashed document.',
    notes: ['feature-hashing-for-text'],
    cite: ['weinberger2009'],
  },
  hashing.hashedFeatures,
)
fn(
  {
    key: 'featureHash',
    name: 'Feature hashing',
    role: 'transform',
    summary: 'The signed hashed document–feature matrix, as scikit-learn’s HashingVectorizer.',
    notes: ['feature-hashing-for-text'],
    cite: ['weinberger2009', 'pedregosa2011'],
  },
  hashing.featureHash,
)

const SHINGLES = ['character-n-grams-and-shingles']
const LSH = ['locality-sensitive-hashing']
fn(
  {
    key: 'oneHotTokens',
    name: 'One-hot encoding',
    role: 'transform',
    summary:
      'A token sequence as the vocabulary × positions matrix of indicator vectors; distinct words are orthogonal.',
    notes: ['one-hot-encoding-of-words'],
    cite: ['jurafsky2025'],
  },
  onehot.oneHotTokens,
)
fn(
  {
    key: 'characterShingles',
    name: 'Character shingles',
    role: 'transform',
    summary: 'The set of contiguous k-character substrings of a text.',
    notes: SHINGLES,
    cite: ['broder1997'],
  },
  shingles.characterShingles,
)
fn(
  {
    key: 'wordShingles',
    name: 'Word shingles',
    role: 'transform',
    summary: 'The set of runs of w consecutive tokens.',
    notes: SHINGLES,
    cite: ['broder1997'],
  },
  shingles.wordShingles,
)
fn(
  {
    key: 'jaccardSimilarity',
    name: 'Jaccard similarity of sets',
    tex: '\\frac{|A \\cap B|}{|A \\cup B|}',
    role: 'property',
    summary: 'The resemblance |A ∩ B| / |A ∪ B| of two shingle sets.',
    notes: [...SHINGLES, 'hamming-jaccard-and-exact-match'],
    cite: ['jaccard1912', 'broder1997'],
  },
  shingles.jaccardSimilarity,
)
fn(
  {
    key: 'minHashSignature',
    name: 'MinHash signature',
    role: 'transform',
    summary: 'The minima of k seeded hash functions over a set; agreeing minima estimate the Jaccard similarity.',
    notes: [...SHINGLES, ...LSH],
    cite: ['broder1997', 'broder2000minwise'],
  },
  minhash.minHashSignature,
)
fn(
  {
    key: 'minHashSignatures',
    name: 'MinHash signatures of several sets',
    role: 'transform',
    notes: [...SHINGLES, ...LSH],
    cite: ['broder1997'],
  },
  minhash.minHashSignatures,
)
fn(
  {
    key: 'minHashSimilarity',
    name: 'MinHash estimate of Jaccard similarity',
    role: 'estimator',
    summary: 'The share of signature positions where two sets agree; unbiased for J with variance J(1 − J)/k.',
    notes: [...SHINGLES, ...LSH],
    cite: ['broder1997', 'broder2000minwise'],
  },
  minhash.minHashSimilarity,
)
fn(
  {
    key: 'minHashStandardError',
    name: 'Standard error of the MinHash estimate',
    tex: '\\sqrt{J(1 - J)/k}',
    role: 'property',
    notes: SHINGLES,
    cite: ['broder1997'],
  },
  minhash.minHashStandardError,
)
const TEMPLATES = ['conditional-random-field', 'sequence-labelling']

fn(
  {
    key: 'parseTemplates',
    name: 'CRF++ feature templates',
    role: 'construction',
    summary: 'Parse CRF++ templates: U (unigram) and B (bigram) lines with %x[row,column] macros over token rows.',
    notes: TEMPLATES,
  },
  templates.parseTemplates,
)
fn(
  {
    key: 'expandTemplates',
    name: 'Expand feature templates',
    role: 'transform',
    summary: 'The feature strings of every template at every position, with _B-k and _B+k past the ends.',
    notes: TEMPLATES,
  },
  templates.expandTemplates,
)
fn(
  {
    key: 'featureIndex',
    name: 'Feature index of templates',
    role: 'construction',
    summary:
      'Expanded template strings of training data with ids and counts, dropping those below a minimum frequency.',
    notes: TEMPLATES,
  },
  templates.featureIndex,
)
fn(
  {
    key: 'encodeTemplateRows',
    name: 'Encode a sequence by a feature index',
    role: 'transform',
    summary: 'The ids of the indexed feature strings that fire at each position, in compressed rows.',
    notes: TEMPLATES,
  },
  templates.encodeTemplateRows,
)

/** The functions of the module, keyed by name. */
export const featuresFunctions = entries<FunctionInfo>(
  'function',
  ngrams,
  bag,
  weighting,
  hashing,
  onehot,
  shingles,
  minhash,
  templates,
) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
