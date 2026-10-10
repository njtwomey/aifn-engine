/**
 * `aifn-compute/text/tokenise`: words, characters and sentences, each token with its offsets into the text.
 *
 * - Regular-expression tokenisers: `tokenise` with a pattern of `TOKEN_PATTERNS` (words, words and punctuation, white
 *   space, scikit-learn's default, the GPT-2, cl100k and o200k pre-tokenisers, BERT's) or one of your own, and
 *   `whitespaceTokenise`.
 * - Characters: `characterTokenise`, by code point or by grapheme cluster.
 * - Word tokenisers of NLTK: `treebankTokens` and `treebankTokenise` (the Penn Treebank rules, without and with
 *   offsets) and `casualTokenise` (`TweetTokenizer`: emoticons, handles, hashtags, URLs, HTML entities).
 * - Sentences: `sentenceSplit`, Punkt's decision rules with NLTK's English abbreviations and sentence starters
 *   (`ENGLISH_ABBREVIATIONS`, `ENGLISH_SENTENCE_STARTERS`).
 * - Back to text: `detokenise`, exact for a `Tokenisation`, by English spacing rules for a token list.
 *
 * Every tokeniser returns a `Tokenisation`: the tokens and an int32 [n, 2] tensor of [start, end) offsets in UTF-16
 * code units, so that a token can be traced to the text it came from even when it was rewritten (decoded entities,
 * converted quotes). The tokenisers of a trainable pipeline are in `aifn-compute/text/pipeline`.
 */

export {
  characterTokenise,
  detokenise,
  tokenise,
  TOKEN_PATTERNS,
  whitespaceTokenise,
  type CharacterOptions,
  type Tokenisation,
  type TokeniseOptions,
  type TokenPattern,
} from './tokenise'
export { treebankTokenise, treebankTokens, type TreebankOptions } from './treebank'
export { casualTokenise, type CasualOptions } from './casual'
export { ENGLISH_ABBREVIATIONS, ENGLISH_SENTENCE_STARTERS, sentenceSplit, type SentenceOptions } from './sentences'
export { tokeniseFunctions } from './registry'
