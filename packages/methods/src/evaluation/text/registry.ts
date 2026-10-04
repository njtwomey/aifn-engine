/** The functions of `aifn-methods/evaluation/text` besides its metrics: the scores with their parts, and alignments. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as text from './text'

const fn = definer<FunctionInfo>('function', 'evaluation/text')

fn(
  {
    key: 'bleuScore',
    name: 'BLEU with its parts',
    summary: 'Clipped n-gram precisions, their geometric mean and the brevity penalty.',
    role: 'estimator',
    notes: ['bleu', 'machine-translation'],
    cite: ['papineni2002'],
  },
  text.bleuScore,
)
fn(
  { key: 'chrFScore', name: 'chrF with its parts', role: 'estimator', notes: ['chrf'], cite: ['popovic2015'] },
  text.chrFScore,
)
fn(
  {
    key: 'rougeNScores',
    name: 'ROUGE-N precision, recall and F',
    role: 'estimator',
    notes: ['rouge'],
    cite: ['lin2004'],
  },
  text.rougeNScores,
)
fn(
  {
    key: 'rougeLScores',
    name: 'ROUGE-L precision, recall and F',
    role: 'estimator',
    notes: ['rouge'],
    cite: ['lin2004'],
  },
  text.rougeLScores,
)
fn(
  { key: 'bertScore', name: 'BERTScore with its parts', role: 'estimator', notes: ['bertscore'], cite: ['zhang2020'] },
  text.bertScore,
)
fn(
  {
    key: 'translationEdits',
    name: 'Translation edits (TER)',
    role: 'estimator',
    notes: ['translation-edit-rate'],
    cite: ['snover2006'],
  },
  text.translationEdits,
)
fn(
  {
    key: 'editAlignment',
    name: 'Edit alignment',
    summary: 'The Levenshtein alignment of two token sequences: substitutions, insertions and deletions.',
    role: 'solver',
    notes: ['word-and-character-error-rates'],
  },
  text.editAlignment,
)
fn(
  { key: 'longestCommonSubsequence', name: 'Longest common subsequence', role: 'solver', notes: ['rouge'] },
  text.longestCommonSubsequence,
)
fn(
  { key: 'ngrams', name: 'n-grams of a token sequence', role: 'transform', notes: ['bleu', 'word-n-grams'] },
  text.ngrams,
)

/** The functions of the module that are not metrics, keyed by name. */
export const textEvaluationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', text) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
