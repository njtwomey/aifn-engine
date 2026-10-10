/**
 * `aifn-compute/text/subword`: subword tokenisers, trained by step-through algorithms and encoding with offsets into
 * the source text.
 *
 * - Byte-pair encoding (Sennrich et al. 2016), at character or byte level: `bpeSteps` (one merge of the most frequent
 *   pair per step), `bpe` and `bpeModel` to get a tokeniser, `bpeSegment` (with BPE-dropout), `bpeEncode` and
 *   `bpeDecode`; `bpePairCounts` is the count each step reads.
 * - GPT-2's byte-level symbols: `byteAlphabet`, `byteSymbols` (a word's UTF-8 bytes) and `textFromByteSymbols`.
 * - WordPiece: `wordPieceSteps` (merges ranked by $c_{ab} / (c_a c_b)$ or by likelihood gain), `wordPiece` and
 *   `wordPieceModel` (also from a given vocabulary, such as BERT's), `wordPieceSegment` and `wordPieceEncode` (greedy
 *   longest match first); `wordPieceCounts`.
 * - The unigram language model (Kudo 2018): `unigramLmSteps` (EM and a pruning round per step), `unigramLm` and
 *   `unigramLmModel` (also from given probabilities), `unigramLmViterbi` and `unigramLmSegment` (best segmentation),
 *   `unigramLmMarginal` and `unigramLmSegmentations` (every segmentation), `unigramLmSample` (subword
 *   regularisation), `unigramLmEncode`; `unigramLmLosses` and `unigramLmCorpus` for the pruning criterion.
 * - The registry: `subwordAlgorithms` and `subwordFunctions`.
 *
 * Trainers take a list of words or a table of word counts, and break ties as the reference tokenisers do (code-point
 * order, first occurrence). Segmenters take one pre-tokenised word and return pieces with their ranges in it;
 * encoders pre-tokenise text by the model's pattern and return a `Tokenisation` with offsets.
 */

export {
  bpe,
  bpeDecode,
  bpeEncode,
  bpeModel,
  bpePairCounts,
  bpeSegment,
  bpeSteps,
  type BpeMerge,
  type BpeModel,
  type BpeOptions,
  type BpeState,
} from './bpe'
export { byteAlphabet, byteSymbols, textFromByteSymbols } from './bytes'
export {
  wordPiece,
  wordPieceCounts,
  wordPieceEncode,
  wordPieceModel,
  wordPieceSegment,
  wordPieceSteps,
  type WordPieceMerge,
  type WordPieceModel,
  type WordPieceOptions,
  type WordPieceState,
} from './wordpiece'
export {
  unigramLm,
  unigramLmCorpus,
  unigramLmEncode,
  unigramLmLosses,
  unigramLmMarginal,
  unigramLmModel,
  unigramLmSample,
  unigramLmSegment,
  unigramLmSegmentations,
  unigramLmSteps,
  unigramLmViterbi,
  type PrunedPiece,
  type UnigramLmCorpus,
  type UnigramLmModel,
  type UnigramLmOptions,
  type UnigramLmState,
} from './unigram'
export type { Piece, WordCountsLike } from './words'
export { subwordAlgorithms, subwordFunctions } from './registry'
