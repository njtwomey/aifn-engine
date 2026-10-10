/**
 * `aifn-methods/text`: applications of text processing on `aifn-compute/text`, with the corpora they run on.
 *
 * - `aifn-methods/text/corpora`: the hand-worked corpora of the notes and seeded generators of sentences and documents
 *   with known topics, for tokenisation, word representations and topic models.
 * - `aifn-methods/text/tokenisers`: a suite of named tokenisers (characters, bytes, Treebank words, BPE, byte-level
 *   BPE, WordPiece, unigram, SentencePiece BPE) trained on one corpus, to compare their vocabularies and cuts.
 * - `aifn-methods/text/hyphenation`: four hyphenators (Liang's patterns, a window MLP, a bidirectional LSTM and a
 *   template CRF) learned from dictionary points and scored on held-out words.
 *
 * The root re-exports the corpora `namedCorpus` and `toyCorpus` (and the `Corpus` type), and the suite `tokeniserSuite`
 * with its `TOKENISER_KINDS`.
 */

export { namedCorpus, toyCorpus, type Corpus } from './corpora'
export { tokeniserSuite, TOKENISER_KINDS, type TokeniserKind } from './tokenisers'
