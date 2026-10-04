/**
 * `aifn-methods/text/hyphenation`: three ways to hyphenate English words, compared on dictionary points. Liang's
 * patterns learned by PATGEN (`liangLearningRun`, on `aifn-compute/text/hyphenation`), a NETtalk-style 7-character window MLP
 * (`WindowTagger`) and a small bidirectional LSTM tagger (`BiRnnTagger`), trained side by side
 * (`taggerTrainingRun`); a linear-chain CRF over CRF++ templates (`crfHyphenationRun`, `HYPHENATION_TEMPLATES`); and
 * precision-first scores of hyphen points (`hyphenScores`, `thresholdScores`).
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
