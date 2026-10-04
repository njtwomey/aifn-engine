/**
 * `aifn-compute/text/subword`: subword tokenisers. Byte-pair encoding (character or byte level), WordPiece and the unigram
 * language model, each trained by a step-through algorithm (one merge or one pruning round per step) and encoded with
 * offsets into the source text.
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
