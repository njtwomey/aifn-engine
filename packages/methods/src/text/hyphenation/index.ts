/**
 * `aifn-methods/text/hyphenation`: four ways to hyphenate English words, learned from dictionary points and compared
 * on held-out words.
 *
 * - Liang's patterns: `liangLearningRun` learns them with PATGEN one pass at a time (on
 *   `aifn-compute/text/hyphenation`), counting hits, false alarms and misses after each pass.
 * - Neural taggers: `WindowTagger`, a NETtalk-style MLP on a 7-character window, and `BiRnnTagger`, a small
 *   bidirectional LSTM (or GRU) over `.word.`, trained side by side by `taggerTrainingRun`. Their inputs are
 *   `letterWindow` and `dottedIds` (ids into `HYPHEN_ALPHABET`), batched for training by `windowExamples` and
 *   `rowExamples`; their outputs are `windowProbabilities` and `rnnProbabilities`, and the occlusion saliency of each
 *   character, `windowSaliency` and `rnnSaliency`.
 * - A linear-chain CRF over CRF++ templates: `crfHyphenationRun` trains it and scores it as it goes, `crfHyphenate`
 *   hyphenates a word by Viterbi or posterior decoding. `HYPHENATION_TEMPLATES` holds four template sets, and
 *   `hyphenationRows` and `hyphenationSequence` turn a word into rows labelled with `HYPHEN_LABELS`.
 * - Scores: `hyphenScores` of predicted hyphens and `thresholdScores` of probabilities at a threshold, against the
 *   dictionary labels of `gapLabels`, with precision, recall, $F_1$ and the precision-first $F_{0.5}$.
 *
 * Words are lower case a–z, and a hyphen point is the gap after letter $i$ (from 0), as in `HyphenatedWord`. The runs
 * are generators of snapshots, so the lab's worker can stream them to the page.
 */

export {
  liangLearningRun,
  rowExamples,
  taggerTrainingRun,
  windowExamples,
  type HyphenationSplit,
  type LiangLearningOptions,
  type LiangSnapshot,
  type TaggerCheckpoint,
  type TaggerSnapshot,
  type TaggerTrainingOptions,
} from './runs'
export {
  crfHyphenate,
  crfHyphenationRun,
  HYPHEN_LABELS,
  HYPHENATION_TEMPLATES,
  hyphenationRows,
  hyphenationSequence,
  type CrfDecision,
  type CrfHyphenationOptions,
  type CrfHyphenationSnapshot,
} from './crf'
export { gapLabels, hyphenScores, thresholdScores, type HyphenScores } from './scores'
export {
  dottedIds,
  HYPHEN_ALPHABET,
  letterWindow,
  BiRnnTagger,
  rnnProbabilities,
  rnnSaliency,
  WindowTagger,
  windowProbabilities,
  windowSaliency,
  type BiRnnTaggerConfig,
  type BiRnnTaggerParams,
  type WindowTaggerConfig,
  type WindowTaggerParams,
} from './taggers'
