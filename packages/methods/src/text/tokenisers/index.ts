/**
 * `aifn-methods/text/tokenisers`: a suite of named tokenisers, built from `aifn-compute/text/pipeline` and trained on
 * one corpus so that their vocabularies and cuts can be compared.
 *
 * - The suite: `tokeniserSuite` trains every kind on one corpus (by default `TOKENISER_CORPUS`), `suiteTokeniser` one
 *   kind. `TOKENISER_KINDS` lists the kinds (characters, bytes, Treebank words, BPE, byte-level BPE, WordPiece,
 *   unigram, SentencePiece BPE with byte fallback), `TOKENISER_LABELS` their display names and `SUBWORD_KINDS` the
 *   five with a subword model.
 * - One kind in parts: `untrainedTokeniser` (its stages before training) and `tokeniserTrainer` (its trainer).
 * - `SuiteOptions` are shared by every kind: the subword vocabulary size (byte-level BPE and SentencePiece BPE hold
 *   their 256 byte tokens on top of it), the byte-level split, digit splitting, and the word level's minimum count and
 *   lower-casing.
 * - `tokeniserFunctions`: the registered functions, keyed by name.
 *
 * A trained tokeniser encodes and decodes text with `encodeText` and `decodeIds` of `aifn-compute/text/pipeline`.
 */

export {
  SUBWORD_KINDS,
  suiteTokeniser,
  TOKENISER_CORPUS,
  TOKENISER_KINDS,
  TOKENISER_LABELS,
  tokeniserFunctions,
  tokeniserSuite,
  tokeniserTrainer,
  untrainedTokeniser,
  type SuiteOptions,
  type TokeniserKind,
} from './tokenisers'
