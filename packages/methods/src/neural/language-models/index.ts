/**
 * `aifn-methods/neural/language-models`: language models as users of `aifn-compute/nn/attention` and `aifn-compute/nn/decoding`. A
 * tiny character-level GPT (`Gpt`, its training loop, `gptLogits` for decoding, and the registered estimator
 * `charGpt`), the same GPT trained on prompt–answer tasks (`taskTrainingRun`), and the interpolated Kneser–Ney n-gram model (`kneserNey`), on a toy corpus of nursery rhymes.
 */

export { charCorpus, decodeChars, encodeChars, NURSERY_RHYMES, type CharCorpus } from './corpus'
export {
  Gpt,
  gptLogits,
  gptTraining,
  gptTrainingRun,
  nextTokenLoss,
  nextTokenWindows,
  type CharGptModel,
  type CharGptOptions,
  type GptConfig,
  type GptParams,
  type GptPosition,
  type GptTrainingRunOptions,
  type GptTrainingSnapshot,
  type GptTrainingOptions,
} from './gpt'
export { type KneserNeyModel, type KneserNeyOptions, type TokenCorpus } from './ngram'
export { charGpt, kneserNey } from './registry'
export {
  taskAccuracy,
  taskLoss,
  taskTraining,
  taskTrainingRun,
  type TaskAccuracy,
  type TaskCheckpoint,
  type TaskExamples,
  type TaskSnapshot,
  type TaskTrainingOptions,
  type TaskTrainingRunOptions,
} from './tasks'
