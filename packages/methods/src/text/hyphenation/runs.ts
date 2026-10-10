/**
 * The streamed runs of the hyphenation showcase: PATGEN pattern learning one pass at a time (`liangLearningRun`), and
 * the two neural taggers (a window MLP and a bidirectional LSTM) trained side by side by Adam (`taggerTrainingRun`).
 * Both are generators, so the lab's worker can stream each snapshot to the page, which plays them back.
 *
 * Both learn from the training words of a `HyphenationSplit` and report on its held-out words: Liang's patterns as
 * hit, false-alarm and miss counts after each pass, the taggers as the probability of a hyphen at every held-out gap
 * at each checkpoint, for `thresholdScores`.
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
  /** The words to learn from, with their dictionary hyphens. */
  readonly train: { readonly words: readonly HyphenatedWord[] }
  /** The held-out words to score on. */
  readonly test: { readonly words: readonly HyphenatedWord[] }
}

// ── Liang ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A snapshot of `liangLearningRun` after `step` passes. */
export type LiangSnapshot = {
  /** The passes made so far (0 before the first). */
  readonly step: Size
  /**
   * The passes of the whole schedule, one per pattern length of each level; the passes made, once a run cut short by
   * the pattern budget is done.
   */
  readonly steps: Size
  /** The patterns learned so far, in TeX notation (`1te`, `t1t`). */
  readonly patterns: readonly string[]
  /** What each pass so far did. */
  readonly passes: readonly PatgenPass[]
  /** Hits, false alarms and misses of the patterns on the training words. */
  readonly train: HyphenCounts
  /** Hits, false alarms and misses of the patterns on the held-out words. */
  readonly test: HyphenCounts
  /** Both splits' counts after every pass so far (index: the pass, 0 for no patterns). */
  readonly history: readonly { readonly train: HyphenCounts; readonly test: HyphenCounts }[]
  /** Whether this is the last snapshot. */
  readonly done: boolean
}

/** Options of `liangLearningRun`. */
export type LiangLearningOptions = {
  /** The selection rule and pattern lengths of each level (default `PATGEN_LEVELS`). */
  levels?: readonly PatgenLevel[]
  /** Stop adding patterns once the set holds this many (default no limit). */
  maxPatterns?: Size
  /** The fewest letters before a hyphen (default 1: every dictionary point counts). */
  leftMin?: Size
  /** The fewest letters after a hyphen (default 1). */
  rightMin?: Size
}

/**
 * Learn Liang's patterns from the training words with `patgenSteps` (Liang 1983), yielding the patterns and both
 * splits' counts before the first pass and after each one.
 *
 * @param data The training words, which the patterns are learned from, and the held-out words, which they are
 *   counted on.
 * @param options The levels, the pattern budget and the hyphenation margins (applied to learning and counting
 *   alike).
 * @returns A generator of snapshots: one before the first pass (no patterns), then one after each pass.
 *
 * @example One permissive level learns the double-consonant split
 * const dict = (s) => s.split(' ').map((h) => ({ word: h.replaceAll('-', ''), hyphens: [h.indexOf('-') - 1] }))
 * const train = dict('let-ter but-ter bet-ter lad-der sum-mer din-ner pep-per rab-bit kit-ten hap-pen tab-let win-ter')
 * const test = dict('lit-ter mat-ter sup-per')
 * const levels = [{ goodWeight: 1, badWeight: 1, threshold: 2, lengths: [2, 3] }]
 * const last = [...liangLearningRun({ train: { words: train }, test: { words: test } }, { levels })].at(-1)
 * print('passes', last.step, 'of', last.steps, '; patterns', last.patterns.join(' '))
 * print('train', last.train)
 * print('test', last.test)
 */
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

/**
 * The window tagger's training data: the window of every letter but the last of every word, and whether a hyphen
 * follows that letter.
 *
 * @param words The hyphenated words.
 * @param radius The window's radius $r$: letters each side of the centre.
 * @returns `x`, the symbol ids of the windows (int32, $N \times (2r + 1)$, one row per gap), and `y`, the labels
 *   (float64, $N$; 1 for a hyphen).
 *
 * @example The two windows of "cat"
 * print(windowExamples([{ word: 'cat', hyphens: [] }], 2))
 */
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

/**
 * The recurrent tagger's training data: each word as a row of `.word.` ids, with a label and a weight at each
 * position. Letter $i$ sits at position $i + 1$, after the leading `.`.
 *
 * @param words The hyphenated words. Throws `DomainError` for a word longer than `width - 2` or not in a–z.
 * @param width The row width $W$, at least the longest word's length plus 2.
 * @returns `x`, the ids (int32, $N \times W$), `y`, the labels (1 where a hyphen follows the letter at that
 *   position), and `w`, the weights (1 at every letter but the last, 0 at the dots, the last letter and padding),
 *   both float64 $N \times W$.
 *
 * @example hy-phen in a row of eight
 * print(rowExamples([{ word: 'hyphen', hyphens: [1] }], 8))
 */
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

/**
 * Mean binary cross-entropy of logits against labels, over the positions with weight 1.
 *
 * @param logits The recurrent tagger's logits ($N \times W$).
 * @param y The labels, with the shape of `logits`.
 * @param w The weights, 1 at the positions that count and 0 elsewhere.
 * @returns The weighted sum of the losses divided by the sum of the weights (or by 1, if that is 0).
 */
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
  /** The Adam steps taken. */
  readonly step: Size
  /** The window tagger's parameters. */
  readonly window: WindowTaggerParams
  /** The recurrent tagger's parameters. */
  readonly rnn: BiRnnTaggerParams
  /** The window tagger's probability of a hyphen at every gap of every test word, in the order of `gapLabels`. */
  readonly windowTest: Float64Array
  /** The recurrent tagger's probabilities, in the same order. */
  readonly rnnTest: Float64Array
}

/** A snapshot of `taggerTrainingRun`. */
export type TaggerSnapshot = {
  /** The Adam steps taken. */
  readonly step: Size
  /** The steps of the whole run. */
  readonly steps: Size
  /** The window tagger's minibatch loss at every step so far, from step 0. */
  readonly windowLosses: readonly number[]
  /** The recurrent tagger's minibatch loss at every step so far, from step 0. */
  readonly rnnLosses: readonly number[]
  /** The window tagger's architecture, defaults filled in. */
  readonly windowConfig: Required<WindowTaggerConfig>
  /** The recurrent tagger's architecture, defaults filled in. */
  readonly rnnConfig: Required<BiRnnTaggerConfig>
  /** Every checkpoint so far, step 0 first. */
  readonly checkpoints: readonly TaggerCheckpoint[]
}

/** Options of `taggerTrainingRun`. */
export type TaggerTrainingOptions = {
  /** Adam steps of each model (default 600). */
  steps?: Size
  /** Checkpoint every this many steps (default 40), and at step 0 and the end. */
  every?: Size
  /** The window tagger's architecture (default `WindowTagger`'s). */
  window?: WindowTaggerConfig
  /** The recurrent tagger's architecture; its width is the longest training or test word plus 2. */
  rnn?: Omit<BiRnnTaggerConfig, 'width'>
  /** Window examples per step (default 256, or all of them if fewer). */
  windowBatch?: Size
  /** Words per recurrent-tagger step (default 32, or all of them if fewer). */
  rnnBatch?: Size
  /** The window tagger's Adam step size (default 0.01). */
  windowStepSize?: number
  /** The recurrent tagger's Adam step size (default 0.01). */
  rnnStepSize?: number
  /** The root stream's seed (default 'hyphenation-taggers'). */
  seed?: string | number
}

/**
 * Train a NETtalk-style window MLP and a bidirectional LSTM tagger on the training words, one Adam step of each per
 * step (minibatches, gradient norm clipped at 1), yielding a snapshot at every checkpoint (step 0 first) with both
 * models' test probabilities. The window tagger minimises the mean binary cross-entropy of its windows, the recurrent
 * one that of every letter but the last.
 *
 * @param data The training words and the held-out words; every word must be in a–z.
 * @param options The steps, the checkpoint interval, each model's architecture, batch size and step size, and the
 *   seed.
 * @returns A generator of snapshots, at step 0, every `every` steps and at the end.
 *
 * @example A few steps on a few words
 * const dict = (s) => s.split(' ').map((h) => ({ word: h.replaceAll('-', ''), hyphens: [h.indexOf('-') - 1] }))
 * const train = dict('let-ter but-ter lad-der sum-mer din-ner pep-per')
 * const data = { train: { words: train }, test: { words: dict('lit-ter') } }
 * const options = { steps: 6, every: 3, window: { hidden: 8 }, rnn: { hidden: 4 }, windowStepSize: 0.05 }
 * const runs = [...taggerTrainingRun(data, options)]
 * const last = runs.at(-1)
 * print('snapshots at', runs.map((s) => s.step))
 * print('window loss', last.windowLosses[0], '→', last.windowLosses.at(-1))
 * print('LSTM loss', last.rnnLosses[0], '→', last.rnnLosses.at(-1))
 * print('window P(hyphen) at the gaps of lit|ter', last.checkpoints.at(-1).windowTest)
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
