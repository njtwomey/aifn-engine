/**
 * Hyphenation as sequence labelling with a linear-chain CRF over CRF++ templates (the fourth hyphenator beside Liang's
 * patterns, the window MLP and the BiLSTM). Each letter of a word is a row with two columns, the letter and its class
 * (`v` for a vowel a e i o u y, `c` for a consonant), and its label is `HYPH` when the dictionary puts a hyphen after
 * it, else `O`. The classic templates read a window of letters around the gap after the current letter and the
 * character n-grams that span it, so a feature such as `U12:%x[0,0]/%x[1,0]` at "hy|phen" is `U12:y/p`.
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

/** The two labels: no hyphen after the letter, a hyphen after it. */
export const HYPHEN_LABELS = ['O', 'HYPH'] as const

const VOWELS = new Set('aeiouy')

/** The rows of a word: one per letter, columns [letter, class (`v` or `c`)]. */
export function hyphenationRows(word: string): string[][] {
  return [...word].map((ch) => [ch, VOWELS.has(ch) ? 'v' : 'c'])
}

/** A dictionary word as a labelled sequence (`HYPH` after each hyphen point). */
export function hyphenationSequence(w: HyphenatedWord): LabelledSequence {
  const at = new Set(w.hyphens)
  return { rows: hyphenationRows(w.word), labels: [...w.word].map((_, i) => (at.has(i) ? 'HYPH' : 'O')) }
}

/** Template sets for hyphenation, from one letter to the classic window with n-grams across the gap. */
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
  /** CRF++ template source (default: the classic set). */
  templates?: string
  /** Train on the first `trainWords` training words (default: all). */
  trainWords?: Size
  optimizer?: CrfOptimizer
  /** L1 (OWL-QN only) and L2 strengths (CRFsuite convention; CRF++'s -c C is c₂ = 1/(2C)). */
  c1?: number
  c2?: number
  /** CRF++'s -f (default 1). */
  minFrequency?: Size
  /** Most steps (default 100 for L-BFGS / OWL-QN; an SGD / Adam step is an epoch). */
  maxSteps?: Size
  stepSize?: number
  batchSize?: Size
  /** Score the held-out words every this many steps (default 5) and at the end. */
  every?: Size
}

/** A snapshot of `crfHyphenationRun`: the CRF run, and the held-out scores when they were last computed. */
export interface CrfHyphenationSnapshot extends CrfSnapshot {
  /** P(hyphen) at every gap of every test word, in the order of `gapLabels(test.words)` (null before scoring). */
  readonly testProbabilities: Float64Array | null
  /** Held-out scores of the Viterbi hyphens. */
  readonly testScores: HyphenScores | null
  /** Held-out scores of posterior (max-marginal) decoding. */
  readonly testScoresPosterior: HyphenScores | null
  /** The step the test figures are from. */
  readonly scoredAt: Size
}

/** How a labelling is decided: the Viterbi path (MAP sequence) or posterior (max-marginal) decoding. */
export type CrfDecision = 'viterbi' | 'posterior'

/** P(HYPH) at each gap of a word (letters 0 … N − 2) and its hyphens by `decision` (default Viterbi). */
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
 * Train a template CRF on the training words (`crfTrainingRun`) and score its Viterbi hyphens on the held-out words
 * every `every` steps, yielding a snapshot after every step (step 0 first).
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
