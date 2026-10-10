/**
 * Hyphenation as sequence labelling with a linear-chain CRF over CRF++ templates (the fourth hyphenator beside Liang's
 * patterns, the window MLP and the BiLSTM). Each letter of a word is a row with two columns, the letter and its class
 * (`v` for a vowel a e i o u y, `c` for a consonant), and its label is `HYPH` when the dictionary puts a hyphen after
 * it, else `O`. The classic templates read a window of letters around the gap after the current letter and the
 * character n-grams that span it, so a feature such as `U12:%x[0,0]/%x[1,0]` at "hy|phen" is `U12:y/p`.
 *
 * Training is `crfTrainingRun` of `aifn-methods/inference/sequence-models`, and decoding is its Viterbi path or
 * posterior (max-marginal) labelling, scored against the dictionary like the other hyphenators.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { toFlat } from 'aifn-compute/foundation/tensor'
import type { HyphenatedWord } from 'aifn-compute/text/hyphenation'
import { parseTemplates, type TokenRows } from 'aifn-compute/text/features'
import {
  crfTrainingRun,
  templateCrfMarginals,
  templateCrfPosterior,
  templateCrfViterbi,
  type CrfOptimizer,
  type CrfSnapshot,
  type LabelledSequence,
  type TemplateCrf,
} from 'aifn-methods/inference/sequence-models'
import type { HyphenationSplit } from './runs'
import { hyphenScores, type HyphenScores } from './scores'

/** The two labels, in weight order: `O`, no hyphen after the letter, and `HYPH`, a hyphen after it. */
export const HYPHEN_LABELS = ['O', 'HYPH'] as const

/** The letters of class `v`. */
const VOWELS = new Set('aeiouy')

/**
 * The token rows of a word for the templates: one per letter, columns [letter, class (`v` for a e i o u y, `c` for any
 * other character)].
 *
 * @param word The word, in lower case.
 * @returns One row of two columns per letter.
 *
 * @example The rows of "hyphen"
 * print(hyphenationRows('hyphen'))
 */
export function hyphenationRows(word: string): string[][] {
  return [...word].map((ch) => [ch, VOWELS.has(ch) ? 'v' : 'c'])
}

/**
 * A dictionary word as a labelled sequence for CRF training: its rows, and the label `HYPH` at each letter a hyphen
 * follows, `O` elsewhere.
 *
 * @param w The hyphenated word.
 * @returns The rows of `hyphenationRows` and one label per letter.
 *
 * @example hy-phen
 * print(hyphenationSequence({ word: 'hyphen', hyphens: [1] }))
 */
export function hyphenationSequence(w: HyphenatedWord): LabelledSequence {
  const at = new Set(w.hyphens)
  return { rows: hyphenationRows(w.word), labels: [...w.word].map((_, i) => (at.has(i) ? 'HYPH' : 'O')) }
}

/**
 * CRF++ template sets for hyphenation, from one letter to the classic window with n-grams across the gap: `minimal`,
 * `unigram window`, `classic` (the default) and `combined` (letter pairs with the vowel/consonant classes).
 */
export const HYPHENATION_TEMPLATES: Readonly<Record<string, string>> = {
  minimal: ['# The letter itself, and label transitions.', 'U00:%x[0,0]', 'B'].join('\n'),
  'unigram window': [
    '# Each letter in a window of ±2, on its own.',
    'U00:%x[-2,0]',
    'U01:%x[-1,0]',
    'U02:%x[0,0]',
    'U03:%x[1,0]',
    'U04:%x[2,0]',
    'B',
  ].join('\n'),
  classic: [
    '# Letters −2 … +3 around the gap after the current letter.',
    'U00:%x[-2,0]',
    'U01:%x[-1,0]',
    'U02:%x[0,0]',
    'U03:%x[1,0]',
    'U04:%x[2,0]',
    'U05:%x[3,0]',
    '# Character bigrams and trigrams, most of them across the gap.',
    'U10:%x[-1,0]/%x[0,0]',
    'U11:%x[0,0]/%x[1,0]',
    'U12:%x[1,0]/%x[2,0]',
    'U20:%x[-1,0]/%x[0,0]/%x[1,0]',
    'U21:%x[0,0]/%x[1,0]/%x[2,0]',
    'U22:%x[-2,0]/%x[-1,0]/%x[0,0]',
    '# Four letters across the gap, and the vowel/consonant pattern there.',
    'U30:%x[-1,0]/%x[0,0]/%x[1,0]/%x[2,0]',
    'U40:%x[-1,1]/%x[0,1]/%x[1,1]/%x[2,1]',
    'U41:%x[0,1]/%x[1,1]',
    'B',
  ].join('\n'),
  combined: [
    '# Letter pairs across the gap, conjoined with the vowel/consonant classes.',
    'U00:%x[0,0]/%x[1,0]',
    'U01:%x[0,1]/%x[1,1]',
    'U02:%x[0,0]/%x[1,1]',
    'U03:%x[0,1]/%x[1,0]',
    '# Transitions that depend on the class of the current letter.',
    'B',
    'B01:%x[0,1]',
  ].join('\n'),
}

/** Options of `crfHyphenationRun`. */
export interface CrfHyphenationOptions {
  /** CRF++ template source (default: the classic set of `HYPHENATION_TEMPLATES`). */
  templates?: string
  /** Train on the first `trainWords` training words (default: all). */
  trainWords?: Size
  /** The optimiser: `'lbfgs'` (default), `'owlqn'`, `'sgd'` or `'adam'`. */
  optimizer?: CrfOptimizer
  /** L1 strength $c_1$ (OWL-QN only; default 0). */
  c1?: number
  /** L2 strength $c_2$ (CRFsuite convention, default 0.01; CRF++'s `-c C` is $c_2 = 1/(2C)$). */
  c2?: number
  /** Keep a feature string only if seen at least this often, CRF++'s `-f` (default 1). */
  minFrequency?: Size
  /** Most steps (default 100): an L-BFGS / OWL-QN iteration, or an SGD / Adam epoch. */
  maxSteps?: Size
  /** SGD / Adam step size (defaults 0.1 for SGD and 0.05 for Adam). */
  stepSize?: number
  /** SGD / Adam words per minibatch (default 16; 0 for all of them). */
  batchSize?: Size
  /** Score the held-out words every this many steps (default 5) and at the end. */
  every?: Size
}

/** A snapshot of `crfHyphenationRun`: the CRF run, and the held-out scores when they were last computed. */
export interface CrfHyphenationSnapshot extends CrfSnapshot {
  /**
   * The marginal probability of a hyphen at every gap of every test word, in the order of `gapLabels(test.words)`
   * (null before scoring).
   */
  readonly testProbabilities: Float64Array | null
  /** Held-out scores of the Viterbi hyphens. */
  readonly testScores: HyphenScores | null
  /** Held-out scores of posterior (max-marginal) decoding. */
  readonly testScoresPosterior: HyphenScores | null
  /** The step the test figures are from (0 before scoring). */
  readonly scoredAt: Size
}

/** How a labelling is decided: the Viterbi path (MAP sequence) or posterior (max-marginal) decoding. */
export type CrfDecision = 'viterbi' | 'posterior'

/**
 * Hyphenate a word with a trained CRF: the marginal probability of `HYPH` at each gap (after letters $0, \dots, n - 2$
 * of a word of $n$ letters), and the hyphens of the labelling chosen by `decision`.
 *
 * @param crf The trained CRF, whose labels include `HYPH` (as `crfHyphenationRun`'s snapshots hold it).
 * @param word The word, in lower case.
 * @param decision `viterbi` (default) for the most probable labelling, `posterior` for the most probable label at
 *   each letter.
 * @returns `probabilities`, one per gap, and `hyphens`, the gaps labelled `HYPH` (a label on the last letter is
 *   dropped).
 *
 * @example Train on a few words, then hyphenate new ones
 * const dict = (s) => s.split(' ').map((h) => ({ word: h.replaceAll('-', ''), hyphens: [h.indexOf('-') - 1] }))
 * const words = dict('let-ter but-ter lad-der sum-mer din-ner pep-per rab-bit kit-ten')
 * const crf = [...crfHyphenationRun({ train: { words }, test: { words } }, { maxSteps: 20 })].at(-1).crf
 * print('matter', crfHyphenate(crf, 'matter'))
 * print('hammer', crfHyphenate(crf, 'hammer', 'posterior'))
 */
export function crfHyphenate(
  crf: TemplateCrf,
  word: string,
  decision: CrfDecision = 'viterbi',
): { probabilities: number[]; hyphens: number[] } {
  const rows: TokenRows = hyphenationRows(word)
  const h = crf.labels.indexOf('HYPH')
  const K = crf.labels.length
  const m = toFlat(templateCrfMarginals(crf, rows).marginals)
  const path = toFlat(
    decision === 'viterbi' ? templateCrfViterbi(crf, rows).path : templateCrfPosterior(crf, rows).path,
  )
  const probabilities = Array.from({ length: Math.max(0, word.length - 1) }, (_, i) => m[i * K + h])
  const hyphens = probabilities.flatMap((_, i) => (path[i] === h ? [i] : []))
  return { probabilities, hyphens }
}

/**
 * Train a template CRF on the training words (`crfTrainingRun`, seeded `'crf-hyphenation'`) and score its Viterbi and
 * posterior hyphens on the held-out words every `every` steps and at the last, yielding a snapshot after every step
 * (step 0 first). A snapshot between scorings repeats the last scores, with `scoredAt` saying when they were taken.
 *
 * @param data The training words and the held-out words.
 * @param options The templates, how many training words, the optimiser and its settings, and how often to score.
 * @returns A generator of snapshots, ending when training converges, stalls or reaches `maxSteps`.
 *
 * @example A CRF learns to split double consonants
 * const dict = (s) => s.split(' ').map((h) => ({ word: h.replaceAll('-', ''), hyphens: [h.indexOf('-') - 1] }))
 * const train = dict('let-ter but-ter bet-ter lad-der sum-mer din-ner pep-per rab-bit kit-ten hap-pen')
 * const test = dict('lit-ter mat-ter sup-per')
 * const last = [...crfHyphenationRun({ train: { words: train }, test: { words: test } }, { maxSteps: 20 })].at(-1)
 * print('steps', last.step, 'converged', last.converged)
 * print('held-out scores', last.testScores)
 */
export function* crfHyphenationRun(
  data: HyphenationSplit,
  options: CrfHyphenationOptions = {},
): Generator<CrfHyphenationSnapshot> {
  const { every = 5, maxSteps = 100 } = options
  const templates = parseTemplates(options.templates ?? HYPHENATION_TEMPLATES.classic)
  const words = data.train.words.slice(0, options.trainWords ?? data.train.words.length)
  const test = data.test.words
  type Scored = { at: Size; p: Float64Array; s: HyphenScores; q: HyphenScores }
  let scored: Scored | null = null
  const score = (crf: TemplateCrf, at: Size) => {
    const p: number[] = []
    const hy = test.map((w) => {
      const r = crfHyphenate(crf, w.word)
      p.push(...r.probabilities)
      return r.hyphens
    })
    const hp = test.map((w) => crfHyphenate(crf, w.word, 'posterior').hyphens)
    scored = { at, p: Float64Array.from(p), s: hyphenScores(test, hy), q: hyphenScores(test, hp) }
  }
  for (const snap of crfTrainingRun(templates, words.map(hyphenationSequence), {
    labels: HYPHEN_LABELS,
    optimizer: options.optimizer,
    c1: options.c1,
    c2: options.c2,
    minFrequency: options.minFrequency,
    maxSteps,
    stepSize: options.stepSize,
    batchSize: options.batchSize,
    seed: 'crf-hyphenation',
  })) {
    if (snap.done || snap.step % every === 0) score(snap.crf, snap.step)
    const s = scored as Scored | null
    yield {
      ...snap,
      testProbabilities: s?.p ?? null,
      testScores: s?.s ?? null,
      testScoresPosterior: s?.q ?? null,
      scoredAt: s?.at ?? 0,
    }
  }
}
