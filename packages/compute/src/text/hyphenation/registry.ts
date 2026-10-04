/** The algorithms and functions of `aifn-compute/text/hyphenation`. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as liang from './liang'
import * as patgen from './patgen'

const algorithm = definer<AlgorithmInfo>('algorithm', 'text/hyphenation')
const fn = definer<FunctionInfo>('function', 'text/hyphenation')

algorithm(
  {
    key: 'liangSteps',
    name: "Liang's pattern matching",
    summary:
      'Hyphenate a word by patterns: each step applies the patterns starting at one position of .word., keeping the largest digit per gap; odd digits hyphenate.',
    problem: 'sequence',
    state: { iterate: 'slots', flags: [] },
  },
  liang.liangSteps,
)
algorithm(
  {
    key: 'patgenSteps',
    name: 'PATGEN pattern learning',
    summary:
      'Learn hyphenation patterns level by level from hyphenated words: each step is one pass (a level and a pattern length) keeping candidates with good·w₊ − bad·w₋ ≥ threshold.',
    problem: 'corpus',
    state: { iterate: 'patterns', flags: [] },
  },
  patgen.patgenSteps,
)
fn(
  {
    key: 'liangHyphenate',
    name: 'Hyphenate a word by patterns',
    role: 'transform',
    summary: 'The gap values, hyphens and matching patterns of one word under a pattern set.',
  },
  liang.liangHyphenate,
)
fn({ key: 'liangResult', name: 'Hyphens of a matching state', role: 'property' }, liang.liangResult)
fn(
  {
    key: 'hyphenationPatterns',
    name: 'Hyphenation pattern set',
    role: 'construction',
    summary: "A pattern set from patterns in TeX's notation (e.g. hen5at), merging patterns with the same letters.",
  },
  liang.hyphenationPatterns,
)
fn({ key: 'parsePattern', name: 'Parse a TeX pattern', role: 'transform' }, liang.parsePattern)
fn({ key: 'formatPattern', name: 'Write a TeX pattern', role: 'transform' }, liang.formatPattern)
fn({ key: 'patternSlots', name: 'Pattern values of a dotted word', role: 'transform' }, liang.patternSlots)
fn({ key: 'markHyphens', name: 'Mark hyphens in a word', role: 'transform' }, liang.markHyphens)
fn({ key: 'parseHyphenated', name: 'Read a hyphenated word', role: 'transform' }, liang.parseHyphenated)
fn(
  {
    key: 'hyphenCounts',
    name: 'Hyphens found, wrong and missed',
    role: 'estimator',
    summary: 'True-positive, false-positive and false-negative hyphen counts of a pattern set over hyphenated words.',
  },
  patgen.hyphenCounts,
)

/** The step-through algorithms of the module, keyed by factory name. */
export const hyphenationAlgorithms = entries<AlgorithmInfo>('algorithm', liang, patgen) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

/** The functions of the module, keyed by name. */
export const hyphenationFunctions = entries<FunctionInfo>('function', liang, patgen) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
