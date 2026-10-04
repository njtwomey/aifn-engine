/** The stages and functions of `aifn-compute/text/pipeline`, registered with the notes that define them. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as normalisers from './normalisers'
import * as preTokenisers from './pre-tokenisers'
import * as models from './models'
import * as post from './post-processors'
import * as decoders from './decoders'
import * as tok from './tokeniser'
import * as trainers from './trainers'

const fn = definer<FunctionInfo>('function', 'text/pipeline')
const algorithm = definer<AlgorithmInfo>('algorithm', 'text/pipeline')

fn(
  {
    key: 'unicodeNormaliser',
    name: 'Unicode normal-form normaliser',
    role: 'construction',
    summary: 'NFC, NFD, NFKC or NFKD per grapheme cluster, keeping offsets.',
    notes: ['text-normalisation', 'tokenisation'],
    cite: ['whistler2026uax15'],
  },
  normalisers.unicodeNormaliser,
)
fn(
  {
    key: 'lowercaseNormaliser',
    name: 'Lower-case normaliser',
    role: 'construction',
    notes: ['text-normalisation', 'tokenisation'],
  },
  normalisers.lowercaseNormaliser,
)
fn(
  {
    key: 'caseFoldNormaliser',
    name: 'Case-folding normaliser',
    role: 'construction',
    notes: ['text-normalisation', 'tokenisation'],
    cite: ['unicode2026casefolding'],
  },
  normalisers.caseFoldNormaliser,
)
fn(
  {
    key: 'stripAccentsNormaliser',
    name: 'Accent-stripping normaliser',
    role: 'construction',
    summary: 'Drop non-spacing marks (after NFD).',
    notes: ['text-normalisation', 'tokenisation'],
  },
  normalisers.stripAccentsNormaliser,
)
fn(
  {
    key: 'replaceNormaliser',
    name: 'Replace normaliser',
    role: 'construction',
    notes: ['text-normalisation', 'tokenisation'],
  },
  normalisers.replaceNormaliser,
)
fn(
  {
    key: 'prependNormaliser',
    name: 'Prepend normaliser',
    role: 'construction',
    summary: 'A prefix such as SentencePiece’s ▁ before non-empty text.',
    notes: ['text-normalisation', 'tokenisation'],
    cite: ['kudo2018b'],
  },
  normalisers.prependNormaliser,
)
fn(
  {
    key: 'stripNormaliser',
    name: 'Strip normaliser',
    role: 'construction',
    notes: ['text-normalisation', 'tokenisation'],
  },
  normalisers.stripNormaliser,
)
fn(
  {
    key: 'collapseWhitespaceNormaliser',
    name: 'White-space collapsing normaliser',
    role: 'construction',
    notes: ['text-normalisation', 'tokenisation'],
  },
  normalisers.collapseWhitespaceNormaliser,
)
fn(
  {
    key: 'normaliserSequence',
    name: 'Normaliser sequence',
    role: 'construction',
    notes: ['text-normalisation', 'tokenisation'],
  },
  normalisers.normaliserSequence,
)
fn(
  {
    key: 'applyNormaliser',
    name: 'Apply a normaliser',
    role: 'transform',
    summary: 'Normalise aligned text, each output character keeping its source range.',
    notes: ['text-normalisation', 'tokenisation'],
  },
  normalisers.applyNormaliser,
)
fn(
  {
    key: 'whitespacePreTokeniser',
    name: 'White-space pre-tokeniser',
    role: 'construction',
    summary: 'Runs of word characters or of other non-space characters.',
    notes: ['tokenisation'],
  },
  preTokenisers.whitespacePreTokeniser,
)
fn(
  {
    key: 'whitespaceSplitPreTokeniser',
    name: 'White-space split pre-tokeniser',
    role: 'construction',
    notes: ['tokenisation'],
  },
  preTokenisers.whitespaceSplitPreTokeniser,
)
fn(
  {
    key: 'bertPreTokeniser',
    name: 'BERT pre-tokeniser',
    role: 'construction',
    summary: 'Split on white space and isolate every punctuation character.',
    notes: ['wordpiece-and-unigram-tokenisation', 'tokenisation'],
    cite: ['devlin2019'],
  },
  preTokenisers.bertPreTokeniser,
)
fn(
  { key: 'punctuationPreTokeniser', name: 'Punctuation pre-tokeniser', role: 'construction', notes: ['tokenisation'] },
  preTokenisers.punctuationPreTokeniser,
)
fn(
  {
    key: 'digitsPreTokeniser',
    name: 'Digit-splitting pre-tokeniser',
    role: 'construction',
    summary: 'Each digit its own pre-token (LLaMA) or runs of digits.',
    notes: ['tokenisation'],
    cite: ['touvron2023'],
  },
  preTokenisers.digitsPreTokeniser,
)
fn(
  {
    key: 'splitPreTokeniser',
    name: 'Regular-expression split pre-tokeniser',
    role: 'construction',
    summary: 'Split by a pattern (GPT-2, cl100k, o200k, …) with a delimiter behaviour.',
    notes: ['tokenisation'],
    cite: ['radford2019'],
  },
  preTokenisers.splitPreTokeniser,
)
fn(
  {
    key: 'metaspacePreTokeniser',
    name: 'Metaspace pre-tokeniser',
    role: 'construction',
    summary: 'Spaces become ▁, a ▁ is prepended, and the text splits before each ▁ (SentencePiece).',
    notes: ['wordpiece-and-unigram-tokenisation', 'tokenisation'],
    cite: ['kudo2018b'],
  },
  preTokenisers.metaspacePreTokeniser,
)
fn(
  {
    key: 'byteLevelPreTokeniser',
    name: 'Byte-level pre-tokeniser',
    role: 'construction',
    summary: 'A regex split, then every character as its UTF-8 byte symbols (GPT-2).',
    notes: ['byte-pair-encoding', 'tokenisation'],
    cite: ['radford2019'],
  },
  preTokenisers.byteLevelPreTokeniser,
)
fn(
  {
    key: 'treebankPreTokeniser',
    name: 'Treebank pre-tokeniser',
    role: 'construction',
    notes: ['tokenisation'],
    cite: ['bird2009'],
  },
  preTokenisers.treebankPreTokeniser,
)
fn(
  {
    key: 'casualPreTokeniser',
    name: 'Casual (tweet) pre-tokeniser',
    role: 'construction',
    notes: ['tokenisation'],
    cite: ['bird2009'],
  },
  preTokenisers.casualPreTokeniser,
)
fn(
  { key: 'preTokeniserSequence', name: 'Pre-tokeniser sequence', role: 'construction', notes: ['tokenisation'] },
  preTokenisers.preTokeniserSequence,
)
fn(
  { key: 'applyPreTokeniser', name: 'Apply a pre-tokeniser', role: 'transform', notes: ['tokenisation'] },
  preTokenisers.applyPreTokeniser,
)
fn(
  {
    key: 'splitAligned',
    name: 'Split aligned text',
    role: 'transform',
    summary: 'Removed, isolated, merged-with-previous/next or contiguous delimiters.',
    notes: ['tokenisation'],
  },
  preTokenisers.splitAligned,
)
fn(
  {
    key: 'bpeStage',
    name: 'BPE model stage',
    role: 'construction',
    summary: 'Merges and a vocabulary; byte fallback, BPE-dropout, end-of-word symbol.',
    notes: ['byte-pair-encoding', 'tokenisation'],
    cite: ['sennrich2016'],
  },
  models.bpeStage,
)
fn(
  {
    key: 'wordPieceStage',
    name: 'WordPiece model stage',
    role: 'construction',
    notes: ['wordpiece-and-unigram-tokenisation', 'tokenisation'],
    cite: ['devlin2019'],
  },
  models.wordPieceStage,
)
fn(
  {
    key: 'unigramStage',
    name: 'Unigram model stage',
    role: 'construction',
    summary: 'Pieces with log-probabilities, Viterbi segmentation, byte fallback.',
    notes: ['wordpiece-and-unigram-tokenisation', 'tokenisation'],
    cite: ['kudo2018'],
  },
  models.unigramStage,
)
fn(
  {
    key: 'wordLevelStage',
    name: 'Word-level model stage',
    role: 'construction',
    summary: 'One id per word, the unknown token for the rest.',
    notes: ['tokenisation'],
  },
  models.wordLevelStage,
)
fn(
  { key: 'characterStage', name: 'Character model stage', role: 'construction', notes: ['tokenisation'] },
  models.characterStage,
)
fn(
  {
    key: 'byteStage',
    name: 'Byte model stage',
    role: 'construction',
    summary: '256 byte ids after the specials (ByT5).',
    notes: ['tokenisation'],
  },
  models.byteStage,
)
fn(
  {
    key: 'vocabularyWithIds',
    name: 'Vocabulary with given ids',
    role: 'construction',
    returns: 'vocabulary',
    notes: ['tokenisation'],
  },
  models.vocabularyWithIds,
)
fn(
  {
    key: 'modelSegment',
    name: 'Segment a pre-token',
    role: 'transform',
    summary: 'Tokens with ids and ranges, unknowns as the unknown token or as <0xNN> bytes.',
    notes: ['tokenisation'],
  },
  models.modelSegment,
)
fn(
  {
    key: 'templateProcessor',
    name: 'Template post-processor',
    role: 'construction',
    summary: 'Special tokens and type ids around one sequence or a pair: [CLS] $A [SEP] $B:1 [SEP]:1.',
    notes: ['tokenisation'],
    cite: ['devlin2019'],
  },
  post.templateProcessor,
)
fn(
  {
    key: 'bertProcessor',
    name: 'BERT post-processor',
    role: 'construction',
    notes: ['tokenisation'],
    cite: ['devlin2019'],
  },
  post.bertProcessor,
)
fn(
  {
    key: 'truncation',
    name: 'Truncation settings',
    role: 'construction',
    summary: 'Maximum length, stride and strategy (longest first, only first, only second).',
    notes: ['tokenisation'],
  },
  post.truncation,
)
fn({ key: 'padding', name: 'Padding settings', role: 'construction', notes: ['tokenisation'] }, post.padding)
fn(
  {
    key: 'truncationWindows',
    name: 'Truncation windows',
    role: 'transform',
    summary: 'Overlapping windows of at most max tokens, stride tokens shared.',
    notes: ['tokenisation'],
  },
  post.truncationWindows,
)
fn(
  { key: 'pairLengths', name: 'Longest-first pair lengths', role: 'transform', notes: ['tokenisation'] },
  post.pairLengths,
)
fn(
  {
    key: 'byteLevelDecoder',
    name: 'Byte-level decoder',
    role: 'construction',
    notes: ['byte-pair-encoding', 'tokenisation'],
    cite: ['radford2019'],
  },
  decoders.byteLevelDecoder,
)
fn(
  {
    key: 'metaspaceDecoder',
    name: 'Metaspace decoder',
    role: 'construction',
    notes: ['wordpiece-and-unigram-tokenisation', 'tokenisation'],
    cite: ['kudo2018b'],
  },
  decoders.metaspaceDecoder,
)
fn(
  {
    key: 'wordPieceDecoder',
    name: 'WordPiece decoder',
    role: 'construction',
    notes: ['wordpiece-and-unigram-tokenisation', 'tokenisation'],
  },
  decoders.wordPieceDecoder,
)
fn(
  {
    key: 'byteFallbackDecoder',
    name: 'Byte-fallback decoder',
    role: 'construction',
    summary: 'Runs of <0xNN> tokens back to UTF-8 characters.',
    notes: ['tokenisation'],
  },
  decoders.byteFallbackDecoder,
)
fn({ key: 'fuseDecoder', name: 'Fuse decoder', role: 'construction', notes: ['tokenisation'] }, decoders.fuseDecoder)
fn({ key: 'stripDecoder', name: 'Strip decoder', role: 'construction', notes: ['tokenisation'] }, decoders.stripDecoder)
fn(
  { key: 'replaceDecoder', name: 'Replace decoder', role: 'construction', notes: ['tokenisation'] },
  decoders.replaceDecoder,
)
fn(
  {
    key: 'endOfWordDecoder',
    name: 'End-of-word decoder',
    role: 'construction',
    notes: ['byte-pair-encoding', 'tokenisation'],
    cite: ['sennrich2016'],
  },
  decoders.endOfWordDecoder,
)
fn(
  { key: 'decoderSequence', name: 'Decoder sequence', role: 'construction', notes: ['tokenisation'] },
  decoders.decoderSequence,
)
fn({ key: 'applyDecoder', name: 'Apply a decoder', role: 'transform', notes: ['tokenisation'] }, decoders.applyDecoder)
fn(
  {
    key: 'tokeniser',
    name: 'Tokeniser pipeline',
    role: 'construction',
    returns: 'tokeniser',
    summary: 'Normaliser → pre-tokeniser → model → post-processor, and a decoder back.',
    notes: ['tokenisation'],
  },
  tok.tokeniser,
)
fn(
  {
    key: 'withStages',
    name: 'Replace tokeniser stages',
    role: 'construction',
    returns: 'tokeniser',
    notes: ['tokenisation'],
  },
  tok.withStages,
)
fn(
  {
    key: 'byteTokeniser',
    name: 'Byte tokeniser (ByT5)',
    role: 'construction',
    returns: 'tokeniser',
    summary: 'One token per UTF-8 byte, ids after <pad>, </s>, <unk>.',
    notes: ['tokenisation'],
  },
  tok.byteTokeniser,
)
fn(
  {
    key: 'encodeText',
    name: 'Encode with a tokeniser',
    role: 'transform',
    returns: 'encoding',
    summary: 'Ids, tokens, offsets into the original text, word ids, masks; truncation with overflow.',
    notes: ['tokenisation'],
  },
  tok.encodeText,
)
fn(
  {
    key: 'encodeBatch',
    name: 'Encode a batch',
    role: 'transform',
    summary: 'Encodings padded to the longest.',
    notes: ['tokenisation'],
  },
  tok.encodeBatch,
)
fn(
  {
    key: 'decodeIds',
    name: 'Decode ids',
    role: 'transform',
    summary: 'Ids back to text through the decoder.',
    notes: ['tokenisation'],
  },
  tok.decodeIds,
)
fn(
  { key: 'vocabularySize', name: 'Tokeniser vocabulary size', role: 'property', notes: ['tokenisation'] },
  tok.vocabularySize,
)
fn(
  {
    key: 'encodingTokenisation',
    name: 'Encoding as a tokenisation',
    role: 'transform',
    returns: 'tokens',
    notes: ['tokenisation'],
  },
  tok.encodingTokenisation,
)
fn(
  {
    key: 'preTokenCounts',
    name: 'Pre-token counts',
    role: 'estimator',
    summary: 'The word table a trainer starts from, cut by the pipeline itself.',
    notes: ['tokenisation'],
  },
  tok.preTokenCounts,
)
fn(
  {
    key: 'trainTokeniser',
    name: 'Train a tokeniser',
    role: 'fit',
    returns: 'tokeniser',
    summary: 'Train the model through the pipeline’s normaliser and pre-tokeniser.',
    notes: ['tokenisation'],
    cite: ['sennrich2016', 'kudo2018', 'schuster2012'],
  },
  trainers.trainTokeniser,
)
fn(
  { key: 'trainedModel', name: 'Model at a training state', role: 'construction', notes: ['tokenisation'] },
  trainers.trainedModel,
)
algorithm(
  {
    key: 'trainingSteps',
    name: 'Tokeniser training through a pipeline',
    summary:
      'BPE or WordPiece merges, or unigram pruning rounds, on the pre-tokens the pipeline cuts; a merge or round per step.',
    problem: 'corpus',
    state: { iterate: 'vocabulary', objective: 'symbols', flags: [] },
    notes: ['byte-pair-encoding', 'wordpiece-and-unigram-tokenisation', 'tokenisation'],
    cite: ['sennrich2016', 'schuster2012', 'kudo2018'],
  },
  trainers.trainingSteps,
)

/** The training algorithm of the module, keyed by factory name. */
export const pipelineAlgorithms = entries<AlgorithmInfo>('algorithm', trainers) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

/** The functions of the module, keyed by name. */
export const pipelineFunctions = entries<FunctionInfo>(
  'function',
  normalisers,
  preTokenisers,
  models,
  post,
  decoders,
  tok,
  trainers,
) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
