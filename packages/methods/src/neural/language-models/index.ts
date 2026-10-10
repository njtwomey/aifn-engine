/**
 * `aifn-methods/neural/language-models`: small language models trained in the browser, as users of
 * `aifn-compute/nn/attention` and `aifn-compute/nn/decoding`.
 *
 * - A toy corpus: `NURSERY_RHYMES`, encoded character by character by `charCorpus` over its own alphabet, with
 *   `encodeChars` and `decodeChars` between text and ids.
 * - A tiny character-level GPT: `Gpt` (a decoder-only transformer with learned, sinusoidal, RoPE or ALiBi positions
 *   and a tied output layer), `nextTokenWindows` and `nextTokenLoss` for next-token training, `gptTraining` as a
 *   traceable training loop, `gptTrainingRun` as a generator of snapshots for a worker, and `gptLogits` to decode
 *   from it.
 * - The same GPT on prompt–answer tasks: `taskLoss` (cross-entropy at the answer positions only), `taskTraining`,
 *   `taskAccuracy` (teacher-forced, by token and by example) and `taskTrainingRun`.
 * - Registered estimators: `charGpt` (the GPT fitted to a token corpus) and `kneserNey` (the interpolated, optionally
 *   modified, Kneser–Ney n-gram model, fitted by counting).
 *
 * Every model exposes its next-token logits as a `LogitsFn`, so any decoder of `aifn-compute/nn/decoding` samples
 * from it. Training is deterministic from its stream or seed.
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
