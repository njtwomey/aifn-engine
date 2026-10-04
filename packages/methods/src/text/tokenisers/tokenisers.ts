/**
 * A suite of tokenisers to compare side by side, each assembled from `aifn-compute/text/pipeline` stages as its namesake is:
 * characters, raw bytes (ByT5), Penn Treebank words with a word-level vocabulary, character-level BPE with "</w>"
 * (Sennrich et al. 2016), byte-level BPE (GPT-2, or with the cl100k / o200k split), WordPiece (BERT), a unigram
 * language model with Metaspace, and SentencePiece-style BPE with "▁" and byte fallback (LLaMA). All are trained on one
 * small corpus, so their vocabularies and their cuts can be compared on the same text.
 */

import { definer, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import {
  bertPreTokeniser,
  bertProcessor,
  byteFallbackDecoder,
  byteLevelDecoder,
  byteLevelPreTokeniser,
  byteTokeniser,
  decoderSequence,
  digitsPreTokeniser,
  endOfWordDecoder,
  fuseDecoder,
  lowercaseNormaliser,
  metaspaceDecoder,
  metaspacePreTokeniser,
  normaliserSequence,
  preTokeniserSequence,
  stripAccentsNormaliser,
  trainTokeniser,
  treebankPreTokeniser,
  unicodeNormaliser,
  whitespacePreTokeniser,
  wordPieceDecoder,
  type Tokeniser,
  type Trainer,
  type UntrainedTokeniser,
} from 'aifn-compute/text/pipeline'

/** The tokenisers of the suite, from finest to coarsest units. */
export const TOKENISER_KINDS = [
  'character',
  'byte',
  'word',
  'bpe',
  'byteLevelBpe',
  'wordPiece',
  'unigram',
  'sentencePiece',
] as const

/** A tokeniser of the suite. */
export type TokeniserKind = (typeof TOKENISER_KINDS)[number]

/** Display names. */
export const TOKENISER_LABELS: Readonly<Record<TokeniserKind, string>> = {
  character: 'characters',
  byte: 'bytes (ByT5)',
  word: 'words (Treebank)',
  bpe: 'BPE (</w>)',
  byteLevelBpe: 'byte-level BPE (GPT-2)',
  wordPiece: 'WordPiece (BERT)',
  unigram: 'unigram LM (▁)',
  sentencePiece: 'SentencePiece BPE (▁, bytes)',
}

/** The kinds whose model is trained by a step-through subword trainer. */
export const SUBWORD_KINDS = ['bpe', 'byteLevelBpe', 'wordPiece', 'unigram', 'sentencePiece'] as const

/** Options of the suite. */
export interface SuiteOptions {
  /**
   * The vocabulary size of the subword models (default 400). Byte-level BPE and SentencePiece BPE hold their 256 byte
   * tokens on top of it, so every model gets the same number of learned pieces and specials.
   */
  vocabularySize?: number
  /** The split of byte-level BPE before bytes: GPT-2's (default), cl100k's or o200k's. */
  pattern?: 'gpt2' | 'cl100k' | 'o200k'
  /** Split digits one by one before the model (LLaMA), in the BPE-family tokenisers. */
  splitDigits?: boolean
  /** Keep a word in the word-level vocabulary only if seen at least this often (default 1). */
  wordMinCount?: number
  /** Lower-case text before the Treebank word tokeniser, in training and encoding alike (default false). */
  lowercaseWords?: boolean
}

/**
 * A short corpus about tokenisation (original prose), with contractions, numbers, hyphens and some accented words, for
 * training the suite in a browser in well under a second.
 */
export const TOKENISER_CORPUS: readonly string[] = [
  'A tokeniser turns text into a sequence of tokens, and a vocabulary turns each token into an integer id.',
  'Language models read ids, not letters, so the tokeniser decides what the model can see.',
  'Word tokenisers split on spaces and punctuation; they cannot handle words they have never seen.',
  "Unknown words become a single unknown token, and the model can't tell them apart.",
  'Character tokenisers never meet an unknown word, but their sequences are long and slow to read.',
  'Subword tokenisers sit between the two: frequent words stay whole and rare words break into pieces.',
  'Byte-pair encoding starts from characters and merges the most frequent adjacent pair, again and again.',
  'Each merge adds one symbol to the vocabulary and makes the training corpus a little shorter.',
  'The ordered list of merges is the tokeniser: encoding a new word replays the merges in order.',
  'Byte-level BPE starts from the 256 bytes of UTF-8, so any text at all can be encoded without an unknown token.',
  'WordPiece merges the pair that most raises the likelihood of the corpus, not simply the most frequent pair.',
  'BERT marks the pieces that continue a word with two hashes, as in token, ##iser and ##s.',
  'The unigram model starts from a large vocabulary and prunes the pieces whose removal costs the least likelihood.',
  'SentencePiece treats the space as a symbol of its own, written as a low line, so decoding restores the text.',
  'Byte fallback spells a character that is missing from the vocabulary as the bytes of its UTF-8 encoding.',
  'Numbers such as 1234, 2024 and 3.14159 are cut very differently by different tokenisers.',
  'Some tokenisers split every digit on its own, which helps arithmetic but makes numbers longer.',
  "Contractions like don't, won't, it's and they'll are split by the Penn Treebank rules.",
  'The newest, lowest and widest words share the ending -est, which a good tokeniser learns as one piece.',
  'Tokenisation, tokenise, tokenised, tokeniser and tokenisers share a stem and differ in their endings.',
  'Offsets map every token back to the characters of the original text, even after normalisation.',
  'Normalisation folds case, removes accents and composes characters: café, naïve and résumé become plain.',
  'Fertility counts tokens per word, and compression counts bytes per token.',
  'A good tokeniser for English uses about one and a third tokens per word.',
  'The same tokeniser may need three or four times as many tokens for a language it was not trained on.',
  'Padding, truncation and special tokens prepare encodings for a batch of model inputs.',
  'The quick brown fox jumps over the lazy dog, and the lazy dog sleeps on.',
  'Lower, lowest, newer, newest, wider and widest are the classic examples of subword units.',
]

const specials = (kind: TokeniserKind): string[] =>
  kind === 'wordPiece' ? ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] : ['<unk>', '<s>', '</s>']

/** The stages of a suite tokeniser before training (its model is trained by {@link tokeniserTrainer}). */
export function untrainedTokeniser(kind: TokeniserKind, options: SuiteOptions = {}): UntrainedTokeniser {
  const digits = options.splitDigits ? [digitsPreTokeniser(true)] : []
  const seq = (...ps: Parameters<typeof preTokeniserSequence>) =>
    ps.length === 1 ? ps[0] : preTokeniserSequence(...ps)
  switch (kind) {
    case 'character':
      return { decoder: decoderSequence(byteFallbackDecoder(), fuseDecoder()) }
    case 'byte':
      return { ...byteTokeniser({ eos: false }), model: undefined }
    case 'word':
      return {
        normaliser: options.lowercaseWords ? lowercaseNormaliser() : null,
        preTokeniser: treebankPreTokeniser(),
      }
    case 'bpe':
      return { preTokeniser: seq(whitespacePreTokeniser(), ...digits), decoder: endOfWordDecoder() }
    case 'byteLevelBpe':
      return {
        preTokeniser: seq(...digits, byteLevelPreTokeniser({ pattern: options.pattern ?? 'gpt2' })),
        decoder: byteLevelDecoder(),
      }
    case 'wordPiece':
      return {
        normaliser: normaliserSequence(unicodeNormaliser('NFD'), lowercaseNormaliser(), stripAccentsNormaliser()),
        preTokeniser: bertPreTokeniser(),
        postProcessor: bertProcessor(),
        decoder: wordPieceDecoder(),
      }
    case 'unigram':
      return {
        normaliser: unicodeNormaliser('NFKC'),
        preTokeniser: metaspacePreTokeniser(),
        decoder: metaspaceDecoder(),
      }
    case 'sentencePiece':
      return {
        preTokeniser: seq(metaspacePreTokeniser({ prependScheme: 'first' }), ...digits),
        decoder: decoderSequence(metaspaceDecoder({ prependScheme: 'first' }), byteFallbackDecoder(), fuseDecoder()),
      }
  }
}

/** The trainer of a suite tokeniser at a vocabulary size. */
export function tokeniserTrainer(kind: TokeniserKind, vocabularySize = 400, options: SuiteOptions = {}): Trainer {
  switch (kind) {
    case 'character':
      return { type: 'character', specials: ['<unk>'], byteFallback: true }
    case 'byte':
    case 'word':
      // No size cap: every pre-token seen `wordMinCount` times is kept, so unknowns are only words never seen.
      return { type: 'wordLevel', specials: ['<unk>'], unknown: '<unk>', minCount: options.wordMinCount ?? 1 }
    case 'bpe':
      return { type: 'bpe', vocabularySize, specials: ['<unk>'], unknown: '<unk>', endOfWord: '</w>', minCount: 2 }
    case 'byteLevelBpe':
      return { type: 'bpe', vocabularySize: vocabularySize + 256, specials: ['<|endoftext|>'], byteAlphabet: true }
    case 'wordPiece':
      return { type: 'wordPiece', vocabularySize, specials: specials('wordPiece') }
    case 'unigram':
      return { type: 'unigram', vocabularySize, specials: ['<unk>'] }
    case 'sentencePiece':
      return { type: 'bpe', vocabularySize: vocabularySize + 256, specials: specials(kind), byteFallback: true }
  }
}

/** One tokeniser of the suite, trained on `corpus`. */
export function suiteTokeniser(kind: TokeniserKind, corpus: readonly string[], options: SuiteOptions = {}): Tokeniser {
  if (kind === 'byte') return byteTokeniser({ eos: false })
  return trainTokeniser(
    untrainedTokeniser(kind, options),
    corpus,
    tokeniserTrainer(kind, options.vocabularySize ?? 400, options),
  )
}

/** Every tokeniser of the suite, trained on `corpus` (default {@link TOKENISER_CORPUS}). */
export function tokeniserSuite(
  corpus: readonly string[] = TOKENISER_CORPUS,
  options: SuiteOptions = {},
): Record<TokeniserKind, Tokeniser> {
  return Object.fromEntries(TOKENISER_KINDS.map((k) => [k, suiteTokeniser(k, corpus, options)])) as Record<
    TokeniserKind,
    Tokeniser
  >
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

const fn = definer<FunctionInfo>('function', 'text/tokenisers')
const notes = ['tokenisation', 'byte-pair-encoding', 'wordpiece-and-unigram-tokenisation']

fn(
  {
    key: 'tokeniserSuite',
    name: 'Tokeniser suite',
    role: 'fit',
    summary:
      'Characters, bytes, Treebank words, BPE, byte-level BPE, WordPiece, unigram and SentencePiece BPE, trained.',
    notes,
    cite: ['sennrich2016', 'radford2019', 'devlin2019', 'kudo2018', 'kudo2018b'],
  },
  tokeniserSuite,
)
fn(
  { key: 'suiteTokeniser', name: 'One tokeniser of the suite', role: 'fit', returns: 'tokeniser', notes },
  suiteTokeniser,
)
fn({ key: 'untrainedTokeniser', name: 'Suite tokeniser stages', role: 'construction', notes }, untrainedTokeniser)
fn({ key: 'tokeniserTrainer', name: 'Suite tokeniser trainer', role: 'construction', notes }, tokeniserTrainer)

/** The functions of the module, keyed by name. */
export const tokeniserFunctions = {
  tokeniserSuite,
  suiteTokeniser,
  untrainedTokeniser,
  tokeniserTrainer,
} as unknown as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
