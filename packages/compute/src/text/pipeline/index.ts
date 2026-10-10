/**
 * `aifn-compute/text/pipeline`: tokenisers as pipelines of composable stages, as Hugging Face `tokenizers`.
 *
 * - Assemble and run: `tokeniser` from its stages, `withStages` to swap some, `encodeText` (one text or a pair) and
 *   `encodeBatch` (padded to the longest), `decodeIds`, `vocabularySize`, `encodingTokenisation` (the first sequence
 *   as a `Tokenisation`) and `byteTokeniser` (ByT5, ready-made).
 * - Normalisers that keep offsets: `unicodeNormaliser`, `lowercaseNormaliser`, `caseFoldNormaliser`,
 *   `stripAccentsNormaliser`, `replaceNormaliser`, `prependNormaliser`, `stripNormaliser`,
 *   `collapseWhitespaceNormaliser`, `normaliserSequence`; `applyNormaliser` runs one on aligned text.
 * - Pre-tokenisers: `whitespacePreTokeniser`, `whitespaceSplitPreTokeniser`, `bertPreTokeniser`,
 *   `punctuationPreTokeniser`, `digitsPreTokeniser`, `splitPreTokeniser` (any pattern, including GPT-2's, cl100k and
 *   o200k), `metaspacePreTokeniser` (SentencePiece), `byteLevelPreTokeniser` (GPT-2), `treebankPreTokeniser`,
 *   `casualPreTokeniser`, `preTokeniserSequence`; `applyPreTokeniser` and `splitAligned` run them.
 * - Models: `bpeStage` (byte fallback, BPE-dropout), `wordPieceStage`, `unigramStage`, `wordLevelStage`,
 *   `characterStage` and `byteStage`, from a trained state or a given vocabulary (`vocabularyWithIds`, with the ids of
 *   a `tokenizer.json`); `modelSegment` segments one pre-token; `byteToken`, `byteOfToken` and `BYTE_TOKENS` name the
 *   `<0xNN>` byte tokens.
 * - Post-processing: `templateProcessor` and `bertProcessor` add special tokens with type ids; `truncation` (stride,
 *   overflowing windows) and `padding` (fixed, to the longest, to a multiple); `truncationWindows`, `pairLengths` and
 *   `addedTokens` are the arithmetic behind them.
 * - Decoders: `byteLevelDecoder`, `metaspaceDecoder`, `wordPieceDecoder`, `byteFallbackDecoder`, `fuseDecoder`,
 *   `stripDecoder`, `replaceDecoder`, `endOfWordDecoder`, `decoderSequence`; `applyDecoder` runs one.
 * - Training through the pipeline: `trainTokeniser` trains the model on the pre-tokens the pipeline cuts
 *   (`preTokenCounts`); `trainingSteps` and `trainedModel` step through it. The registry: `pipelineAlgorithms` and
 *   `pipelineFunctions`.
 *
 * Every stage is plain data (a tagged object), so a tokeniser can be stored, sent to a worker and rebuilt. Offsets are
 * in UTF-16 code units of the text as typed, through every normalisation. Failures (a token with no id, a bad
 * template or stride) throw `DomainError`.
 */

export {
  applyNormaliser,
  caseFoldNormaliser,
  collapseWhitespaceNormaliser,
  lowercaseNormaliser,
  normaliserSequence,
  prependNormaliser,
  replaceNormaliser,
  stripAccentsNormaliser,
  stripNormaliser,
  unicodeNormaliser,
  type NormalForm,
  type Normaliser,
} from './normalisers'
export {
  applyPreTokeniser,
  bertPreTokeniser,
  byteLevelPreTokeniser,
  casualPreTokeniser,
  digitsPreTokeniser,
  metaspacePreTokeniser,
  preTokeniserSequence,
  punctuationPreTokeniser,
  splitAligned,
  splitPreTokeniser,
  treebankPreTokeniser,
  whitespacePreTokeniser,
  whitespaceSplitPreTokeniser,
  type PreTokeniser,
  type SplitBehaviour,
} from './pre-tokenisers'
export {
  BYTE_TOKENS,
  bpeStage,
  byteOfToken,
  byteStage,
  byteToken,
  characterStage,
  modelSegment,
  unigramStage,
  vocabularyWithIds,
  wordLevelStage,
  wordPieceStage,
  type BpeStage,
  type ByteStage,
  type CharacterStage,
  type ModelOptions,
  type ModelToken,
  type SegmentOptions,
  type TokeniserModel,
  type UnigramStage,
  type WordLevelStage,
  type WordPieceStage,
} from './models'
export {
  addedTokens,
  bertProcessor,
  padding,
  pairLengths,
  templateProcessor,
  truncation,
  truncationWindows,
  type Padding,
  type PostProcessor,
  type TemplateItem,
  type Truncation,
} from './post-processors'
export {
  applyDecoder,
  byteFallbackDecoder,
  byteLevelDecoder,
  decoderSequence,
  endOfWordDecoder,
  fuseDecoder,
  metaspaceDecoder,
  replaceDecoder,
  stripDecoder,
  wordPieceDecoder,
  type Decoder,
} from './decoders'
export {
  byteTokeniser,
  decodeIds,
  encodeText,
  encodeBatch,
  encodingTokenisation,
  preTokenCounts,
  tokeniser,
  vocabularySize,
  withStages,
  type DecodeOptions,
  type Encoding,
  type EncodeOptions,
  type Tokeniser,
  type TokeniserParts,
  type UntrainedTokeniser,
} from './tokeniser'
export { trainTokeniser, trainedModel, trainingSteps, type Trainer, type TrainerState } from './trainers'
export { pipelineAlgorithms, pipelineFunctions } from './registry'
