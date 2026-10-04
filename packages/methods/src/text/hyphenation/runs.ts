/**
 * The streamed runs of the hyphenation showcase: PATGEN pattern learning one pass at a time (`liangLearningRun`), and
 * the two neural taggers (a window MLP and a bidirectional LSTM) trained side by side by Adam (`taggerTrainingRun`). Both are generators, so the lab's worker
 * can stream each snapshot to the page, which plays them back.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import { div, fromData, mul, sum, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import {
  hyphenCounts,
  PATGEN_LEVELS,
  patgenSteps,
  type HyphenatedWord,
  type HyphenCounts,
  type PatgenLevel,
  type PatgenPass,
} from 'aifn-compute/text/hyphenation'
import {
  dottedIds,
  letterWindow,
  BiRnnTagger,
  rnnProbabilities,
  WindowTagger,
  windowProbabilities,
  type BiRnnTaggerConfig,
  type BiRnnTaggerParams,
  type WindowTaggerConfig,
  type WindowTaggerParams,
} from './taggers'

/** The words a run learns from and is scored on (the shape of `mobyHyphenation`'s parts). */
export type HyphenationSplit = {
  readonly train: { readonly words: readonly HyphenatedWord[] }
  readonly test: { readonly words: readonly HyphenatedWord[] }
}

// ── Liang ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A snapshot of `liangLearningRun` after `step` passes. */
export type LiangSnapshot = {
  readonly step: Size
  /** The passes of the whole schedule. */
  readonly steps: Size
  readonly patterns: readonly string[]
  readonly passes: readonly PatgenPass[]
  readonly train: HyphenCounts
  readonly test: HyphenCounts
  /** Both splits' counts after every pass so far (index: the pass, 0 for no patterns). */
  readonly history: readonly { readonly train: HyphenCounts; readonly test: HyphenCounts }[]
  readonly done: boolean
}

/** Options of `liangLearningRun`. */
export type LiangLearningOptions = {
  levels?: readonly PatgenLevel[]
  maxPatterns?: Size
  /** Margins (default 1 and 1: every dictionary point counts). */
  leftMin?: Size
  rightMin?: Size
}

/** Learn patterns from the training words with `patgenSteps`, yielding the patterns and both splits' counts per pass. */
export function* liangLearningRun(
  data: HyphenationSplit,
  options: LiangLearningOptions = {},
): Generator<LiangSnapshot> {
  const margins = { leftMin: options.leftMin ?? 1, rightMin: options.rightMin ?? 1 }
  const alg = patgenSteps(data.train.words, { ...margins, levels: options.levels, maxPatterns: options.maxPatterns })
  const steps = (options.levels ?? PATGEN_LEVELS).reduce((a, l) => a + l.lengths[1] - l.lengths[0] + 1, 0)
  let s = alg.init(undefined, stream('patgen'))
  const history: { train: HyphenCounts; test: HyphenCounts }[] = []
  const snap = (): LiangSnapshot => {
    const test = hyphenCounts(s.patterns, data.test.words, margins)
    history.push({ train: s.counts, test })
    return {
      step: s.t,
      // A run cut short by the pattern budget ends at its last pass.
      steps: s.done ? s.t : steps,
      patterns: s.patterns.patterns,
      passes: s.passes,
      train: s.counts,
      test,
      history: [...history],
      done: s.done,
    }
  }
  yield snap()
  while (!s.done) {
    s = alg.step(s, { t: s.t, stream: stream('patgen') })
    yield snap()
  }
}

// ── Neural taggers ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Training windows [N, 2r + 1] and labels [N] of every letter but the last of every word. */
export function windowExamples(words: readonly HyphenatedWord[], radius: Size): { x: Tensor; y: Tensor } {
  const ids: number[] = []
  const y: number[] = []
  for (const w of words) {
    const at = new Set(w.hyphens)
    for (let i = 0; i + 1 < w.word.length; i++) {
      ids.push(...letterWindow(w.word, i, radius))
      y.push(at.has(i) ? 1 : 0)
    }
  }
  return {
    x: fromData(Int32Array.from(ids), [y.length, 2 * radius + 1]),
    y: fromData(Float64Array.from(y), [y.length]),
  }
}

/** Training rows of `.word.` ids [N, W], per-position labels [N, W] and weights [N, W] (1 at letters but the last). */
export function rowExamples(words: readonly HyphenatedWord[], width: Size): { x: Tensor; y: Tensor; w: Tensor } {
  const n = words.length
  const ids = new Int32Array(n * width)
  const y = new Float64Array(n * width)
  const w = new Float64Array(n * width)
  words.forEach((word, r) => {
    ids.set(dottedIds(word.word, width), r * width)
    for (let i = 0; i + 1 < word.word.length; i++) w[r * width + i + 1] = 1
    for (const i of word.hyphens) y[r * width + i + 1] = 1
  })
  return { x: fromData(ids, [n, width]), y: fromData(y, [n, width]), w: fromData(w, [n, width]) }
}

/** Mean binary cross-entropy of logits against labels, over the positions with weight 1. */
function weightedBce(logits: Value, y: Tensor, w: Tensor): Value {
  const each = binaryCrossEntropyWithLogits(logits, y, { reduction: 'none' })
  return div(
    sum(mul(each, w)),
    Math.max(
      1,
      toFlat(w).reduce((a, b) => a + b, 0),
    ),
  )
}

/** A checkpoint of `taggerTrainingRun`: both models' parameters and their test probabilities, every gap in order. */
export type TaggerCheckpoint = {
  readonly step: Size
  readonly window: WindowTaggerParams
  readonly rnn: BiRnnTaggerParams
  /** P(hyphen) at every gap of every test word, in the order of `gapLabels(test.words)`. */
  readonly windowTest: Float64Array
  readonly rnnTest: Float64Array
}

/** A snapshot of `taggerTrainingRun`. */
export type TaggerSnapshot = {
  readonly step: Size
  readonly steps: Size
  /** Each model's minibatch loss at every step. */
  readonly windowLosses: readonly number[]
  readonly rnnLosses: readonly number[]
  readonly windowConfig: Required<WindowTaggerConfig>
  readonly rnnConfig: Required<BiRnnTaggerConfig>
  readonly checkpoints: readonly TaggerCheckpoint[]
}

/** Options of `taggerTrainingRun`. */
export type TaggerTrainingOptions = {
  /** Adam steps of each model (default 600). */
  steps?: Size
  /** Checkpoint every this many steps (default 40), and at step 0 and the end. */
  every?: Size
  window?: WindowTaggerConfig
  rnn?: Omit<BiRnnTaggerConfig, 'width'>
  /** Window examples per step (default 256) and words per recurrent-tagger step (default 32). */
  windowBatch?: Size
  rnnBatch?: Size
  /** Adam step sizes (defaults 0.01 and 0.01). */
  windowStepSize?: number
  rnnStepSize?: number
  /** The root stream's seed (default 'hyphenation-taggers'). */
  seed?: string | number
}

/**
 * Train a NETtalk-style window MLP and a bidirectional LSTM tagger on the training words, one Adam step of each per
 * step, yielding a snapshot at every checkpoint (step 0 first) with both models' test probabilities.
 */
export function* taggerTrainingRun(
  data: HyphenationSplit,
  options: TaggerTrainingOptions = {},
): Generator<TaggerSnapshot> {
  const {
    steps = 600,
    every = 40,
    windowBatch = 256,
    rnnBatch = 32,
    windowStepSize = 0.01,
    rnnStepSize = 0.01,
    seed = 'hyphenation-taggers',
  } = options
  const longest = Math.max(...data.train.words.map((w) => w.word.length), ...data.test.words.map((w) => w.word.length))
  const windowModel = WindowTagger(options.window)
  const rnnModel = BiRnnTagger({ ...options.rnn, width: longest + 2 })
  const root = stream(seed)
  const wd = windowExamples(data.train.words, windowModel.config.radius)
  const td = rowExamples(data.train.words, rnnModel.config.width)
  const windowAlg = trainingLoop<WindowTaggerParams, { x: Tensor; y: Tensor }>({
    loss: (p, b, ctx) => binaryCrossEntropyWithLogits(windowModel.apply(p, b.x, ctx), b.y),
    data: wd,
    batchSize: Math.min(windowBatch, wd.y.shape[0]),
    optimizer: adamRule({ stepSize: windowStepSize }) as never,
    clipNorm: 1,
  })
  const rnnAlg = trainingLoop<BiRnnTaggerParams, { x: Tensor; y: Tensor; w: Tensor }>({
    loss: (p, b, ctx) => weightedBce(rnnModel.apply(p, b.x, ctx), b.y, b.w),
    data: td,
    batchSize: Math.min(rnnBatch, td.x.shape[0]),
    optimizer: adamRule({ stepSize: rnnStepSize }) as never,
    clipNorm: 1,
  })
  const ws = child(root, 'window')
  const rs = child(root, 'rnn')
  let wState = windowAlg.init({ params: windowModel.init(child(ws, 'init')) }, child(ws, 'init'))
  let rState = rnnAlg.init({ params: rnnModel.init(child(rs, 'init')) }, child(rs, 'init'))
  const windowLosses = [wState.loss]
  const rnnLosses = [rState.loss]
  const testWords = data.test.words.map((w) => w.word)
  const checkpoints: TaggerCheckpoint[] = []
  const checkpoint = (t: Size) =>
    checkpoints.push({
      step: t,
      window: wState.params,
      rnn: rState.params,
      windowTest: Float64Array.from(windowProbabilities(windowModel, wState.params, testWords).flat()),
      rnnTest: Float64Array.from(rnnProbabilities(rnnModel, rState.params, testWords).flat()),
    })
  const snapshot = (t: Size): TaggerSnapshot => ({
    step: t,
    steps,
    windowLosses: [...windowLosses],
    rnnLosses: [...rnnLosses],
    windowConfig: windowModel.config,
    rnnConfig: rnnModel.config,
    checkpoints: [...checkpoints],
  })
  checkpoint(0)
  yield snapshot(0)
  for (let t = 0; t < steps; t++) {
    wState = windowAlg.step(wState, { t, stream: child(ws, 'step', t) })
    rState = rnnAlg.step(rState, { t, stream: child(rs, 'step', t) })
    windowLosses.push(wState.loss)
    rnnLosses.push(rState.loss)
    if ((t + 1) % every === 0 || t + 1 === steps) {
      checkpoint(t + 1)
      yield snapshot(t + 1)
    }
  }
}
