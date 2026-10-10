/**
 * `aifn-methods/evaluation/text`: text metrics for translation, summarisation, speech recognition, question answering
 * and text embeddings.
 *
 * - Translation: `bleu` and `bleuScore` (corpus BLEU with its parts, `BleuOptions` and `BleuSmoothing` for single
 *   sentences, `BleuResult`), `chrF` and `chrFScore` (character $n$-grams), and `translationEditRate` with
 *   `translationEdits` (edits including block shifts).
 * - Summarisation: `rougeN` and `rougeNScores` ($n$-gram overlap), `rougeL` and `rougeLScores` (longest common
 *   subsequence, by `longestCommonSubsequence`), as `Prf` triples.
 * - Speech recognition: `editAlignment` (an `Alignment` of `EditOperation`s), and from its pooled counts
 *   `wordErrorRate`, `characterErrorRate`, `matchErrorRate` and `wordInformationLost`.
 * - Question answering: `squadExactMatch` and `squadF1`, after `normaliseAnswer`.
 * - Embeddings: `bertScore` from given token embeddings, and `backretrieval` of a text embedding through images.
 * - Tokens: the `Tokeniser`s `whitespaceTokens` (the default) and `words` (lowercased, punctuation removed), and
 *   `ngrams`.
 *
 * Every metric takes the reference first; a corpus metric takes arrays, one `References` entry (a `Text` or several)
 * per candidate, and pools its counts. The metrics are collected in `evaluationMetricRegistry` of
 * `aifn-methods/evaluation`, and `textEvaluationFunctions` registers the other functions.
 */

export {
  backretrieval,
  bertScore,
  bleu,
  bleuScore,
  characterErrorRate,
  chrF,
  chrFScore,
  editAlignment,
  longestCommonSubsequence,
  matchErrorRate,
  ngrams,
  normaliseAnswer,
  rougeL,
  rougeLScores,
  rougeN,
  rougeNScores,
  squadExactMatch,
  squadF1,
  translationEditRate,
  translationEdits,
  whitespaceTokens,
  wordErrorRate,
  wordInformationLost,
  words,
  type Alignment,
  type BleuOptions,
  type BleuResult,
  type BleuSmoothing,
  type EditOperation,
  type Prf,
  type References,
  type Text,
  type Tokeniser,
} from './text'
export { textEvaluationFunctions } from './registry'
