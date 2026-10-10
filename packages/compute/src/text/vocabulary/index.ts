/**
 * `aifn-compute/text/vocabulary`: the map between tokens and integer ids, built from a corpus or a fixed list.
 *
 * - Counting: `tokenCounts` gives each distinct token's corpus count and document count, in order of first appearance.
 * - Building: `buildVocabulary` from tokenised documents (special tokens first, then corpus tokens filtered by a
 *   minimum count, capped at a maximum size and ordered by frequency, alphabetically or by appearance, as torchtext or
 *   scikit-learn order them); `vocabularyOf` over a fixed list.
 * - Encoding: `tokenId` for one token, `encodeTokens` to an int32 tensor of ids (unknown tokens to the unknown id, or
 *   an error, or skipped), and `decodeTokens` back to strings.
 *
 * A `Vocabulary` is plain data; the lookup table is built on first use and cached. Errors are `DomainError`s.
 */

export {
  buildVocabulary,
  decodeTokens,
  encodeTokens,
  tokenCounts,
  tokenId,
  vocabularyOf,
  type TokenDocuments,
  type Vocabulary,
  type VocabularyOptions,
} from './vocabulary'
export { vocabularyFunctions } from './registry'
