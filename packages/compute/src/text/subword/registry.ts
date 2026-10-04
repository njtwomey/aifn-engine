/**
 * The subword tokenisers of `aifn-compute/text/subword`: the three trainers as traceable algorithms (each merge or pruning round
 * a step) beside the one-call functions that run them, and the encoders.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bpe from './bpe'
import * as bytes from './bytes'
import * as unigram from './unigram'
import * as wordpiece from './wordpiece'

const algorithm = definer<AlgorithmInfo>('algorithm', 'text/subword')
const fn = definer<FunctionInfo>('function', 'text/subword')

const BPE = ['byte-pair-encoding', 'tokenisation']
const WP = ['wordpiece-and-unigram-tokenisation', 'tokenisation']

// ── Byte-pair encoding ───────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'bpeSteps',
    name: 'Byte-pair encoding training',
    summary: 'Merge the most frequent adjacent symbol pair, one merge per step, recording the merge and its count.',
    problem: 'corpus',
    state: { iterate: 'vocabulary', objective: 'symbols', flags: [] },
    notes: BPE,
    cite: ['sennrich2016', 'gage1994'],
  },
  bpe.bpeSteps,
)
fn(
  {
    key: 'bpe',
    name: 'Train a byte-pair encoding tokeniser',
    role: 'fit',
    returns: 'bpe',
    notes: BPE,
    cite: ['sennrich2016'],
  },
  bpe.bpe,
)
fn(
  { key: 'bpeModel', name: 'BPE tokeniser of a training state', role: 'construction', returns: 'bpe', notes: BPE },
  bpe.bpeModel,
)
fn(
  {
    key: 'bpePairCounts',
    name: 'Adjacent pair counts',
    role: 'estimator',
    summary: 'Every adjacent symbol pair with its count weighted by word counts, in order of first occurrence.',
    notes: BPE,
  },
  bpe.bpePairCounts,
)
fn(
  {
    key: 'bpeSegment',
    name: 'BPE segmentation of a word',
    role: 'transform',
    summary: 'Apply the learned merges in order (lowest rank first) to one word.',
    notes: BPE,
  },
  bpe.bpeSegment,
)
fn(
  { key: 'bpeEncode', name: 'BPE encoding', role: 'transform', returns: 'tokens', notes: BPE, cite: ['sennrich2016'] },
  bpe.bpeEncode,
)
fn({ key: 'bpeDecode', name: 'BPE decoding', role: 'transform', notes: BPE }, bpe.bpeDecode)
fn(
  {
    key: 'byteSymbols',
    name: 'Byte-level symbols',
    role: 'transform',
    summary: "A word's UTF-8 bytes as GPT-2's 256 printable byte symbols.",
    notes: ['byte-pair-encoding'],
    cite: ['radford2019'],
  },
  bytes.byteSymbols,
)
fn(
  { key: 'textFromByteSymbols', name: 'Text from byte symbols', role: 'transform', notes: ['byte-pair-encoding'] },
  bytes.textFromByteSymbols,
)
fn(
  {
    key: 'byteAlphabet',
    name: 'Byte-level alphabet',
    role: 'construction',
    notes: ['byte-pair-encoding'],
    cite: ['radford2019'],
  },
  bytes.byteAlphabet,
)

// ── WordPiece ────────────────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'wordPieceSteps',
    name: 'WordPiece training',
    summary: 'Merge the pair with the highest score c_ab / (c_a c_b) (or likelihood gain), one merge per step.',
    problem: 'corpus',
    state: { iterate: 'vocabulary', objective: 'logLikelihood', flags: [] },
    notes: WP,
    cite: ['schuster2012', 'huggingface2024wordpiece'],
  },
  wordpiece.wordPieceSteps,
)
fn(
  {
    key: 'wordPiece',
    name: 'Train a WordPiece tokeniser',
    role: 'fit',
    returns: 'wordpiece',
    notes: WP,
    cite: ['schuster2012', 'huggingface2024wordpiece'],
  },
  wordpiece.wordPiece,
)
fn(
  { key: 'wordPieceModel', name: 'WordPiece tokeniser', role: 'construction', returns: 'wordpiece', notes: WP },
  wordpiece.wordPieceModel,
)
fn(
  {
    key: 'wordPieceCounts',
    name: 'Symbol and pair counts',
    role: 'estimator',
    summary: 'Symbol counts and adjacent pair counts, weighted by word counts.',
    notes: WP,
  },
  wordpiece.wordPieceCounts,
)
fn(
  {
    key: 'wordPieceSegment',
    name: 'WordPiece segmentation of a word',
    role: 'transform',
    summary: 'Greedy longest-match-first with a continuation prefix; the unknown token when a match fails.',
    notes: WP,
    cite: ['devlin2019'],
  },
  wordpiece.wordPieceSegment,
)
fn(
  {
    key: 'wordPieceEncode',
    name: 'WordPiece encoding',
    role: 'transform',
    returns: 'tokens',
    notes: WP,
    cite: ['devlin2019'],
  },
  wordpiece.wordPieceEncode,
)

// ── Unigram language model ───────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'unigramLmSteps',
    name: 'Unigram language-model tokeniser training',
    summary:
      'EM on piece probabilities, then prune the pieces whose removal lowers the likelihood least; a round per step.',
    problem: 'corpus',
    state: { iterate: 'pieces', objective: 'logLikelihood', flags: [] },
    notes: WP,
    cite: ['kudo2018', 'kudo2018b'],
  },
  unigram.unigramLmSteps,
)
fn(
  {
    key: 'unigramLm',
    name: 'Train a unigram language-model tokeniser',
    role: 'fit',
    returns: 'unigram-lm',
    notes: WP,
    cite: ['kudo2018'],
  },
  unigram.unigramLm,
)
fn(
  { key: 'unigramLmModel', name: 'Unigram tokeniser', role: 'construction', returns: 'unigram-lm', notes: WP },
  unigram.unigramLmModel,
)
fn(
  {
    key: 'unigramLmLosses',
    name: 'Unigram piece losses',
    role: 'estimator',
    summary: 'The drop in corpus log marginal likelihood when each piece is removed.',
    notes: WP,
    cite: ['kudo2018'],
  },
  unigram.unigramLmLosses,
)
fn(
  {
    key: 'unigramLmViterbi',
    name: 'Unigram Viterbi segmentation',
    role: 'inference',
    summary: 'The most probable segmentation of a word under Π p(x_i), by Viterbi on its lattice.',
    notes: WP,
    cite: ['kudo2018'],
  },
  unigram.unigramLmViterbi,
)
fn(
  { key: 'unigramLmSegment', name: 'Unigram segmentation of a word', role: 'inference', notes: WP },
  unigram.unigramLmSegment,
)
fn(
  {
    key: 'unigramLmMarginal',
    name: 'Unigram marginal likelihood of a word',
    role: 'inference',
    summary: 'log Σ_x P(x) over all segmentations, by the forward algorithm.',
    notes: WP,
  },
  unigram.unigramLmMarginal,
)
fn(
  {
    key: 'unigramLmSegmentations',
    name: 'All unigram segmentations',
    role: 'inference',
    summary: 'Every segmentation of a word with its probability and posterior.',
    notes: WP,
  },
  unigram.unigramLmSegmentations,
)
fn(
  {
    key: 'unigramLmSample',
    name: 'Subword regularisation sample',
    role: 'simulation',
    summary: 'A segmentation drawn with probability ∝ P(x)^α by forward filtering, backward sampling.',
    notes: WP,
    cite: ['kudo2018'],
    random: true,
  },
  unigram.unigramLmSample,
)
fn(
  {
    key: 'unigramLmEncode',
    name: 'Unigram encoding',
    role: 'transform',
    returns: 'tokens',
    notes: WP,
    cite: ['kudo2018'],
  },
  unigram.unigramLmEncode,
)
fn(
  { key: 'unigramLmCorpus', name: 'Unigram training corpus', role: 'construction', notes: WP },
  unigram.unigramLmCorpus,
)

/** The trainers of the module as traceable algorithms, keyed by factory name. */
export const subwordAlgorithms = entries<AlgorithmInfo>('algorithm', bpe, wordpiece, unigram) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

/** The functions of the module, keyed by name. */
export const subwordFunctions = entries<FunctionInfo>('function', bpe, bytes, wordpiece, unigram) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
