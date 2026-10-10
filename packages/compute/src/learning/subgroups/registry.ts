/**
 * The registry of `aifn-compute/learning/subgroups`: subgroup discovery as a step-through algorithm
 * (`problem: 'table'`), and the description language, quality measures, SD-Map and the exceptional-model classes as
 * functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as emm from './emm'
import * as language from './language'
import * as quality from './quality'
import * as sdmap from './sdmap'
import * as subgroups from './subgroups'

const MODULE = 'learning/subgroups'
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)
const fn = definer<FunctionInfo>('function', MODULE)

const SD_NOTES = ['intrinsically-interpretable-models', 'interpretability']

algorithm(
  {
    key: 'subgroupDiscoverySteps',
    name: 'Subgroup discovery',
    summary:
      'Search a table’s conjunctive descriptions for the top k subgroups by a quality measure: beam, best-first or exhaustive, with branch and bound by optimistic estimates.',
    problem: 'table',
    state: { iterate: 'frontier', objective: 'best', flags: ['terminated'] },
    notes: SD_NOTES,
  },
  subgroups.subgroupDiscoverySteps,
)
fn(
  {
    key: 'subgroupDiscovery',
    name: 'Subgroup discovery, run to the end',
    summary: 'The top k subgroups of a table by a quality measure, with their covers.',
    role: 'solver',
    notes: SD_NOTES,
  },
  subgroups.subgroupDiscovery,
)
fn(
  {
    key: 'sdMap',
    name: 'SD-Map',
    summary:
      'Exhaustive subgroup discovery for a binary target by FP-growth over selectors, counting positives per tree node.',
    role: 'solver',
    notes: SD_NOTES,
  },
  sdmap.sdMap,
)
fn(
  {
    key: 'subgroupSpace',
    name: 'Subgroup search space',
    summary: 'The refinement search space of subgroup discovery: canonical refinements, quality, optimistic estimate.',
    role: 'construction',
  },
  subgroups.subgroupSpace,
)
fn(
  {
    key: 'subgroupOf',
    name: 'Subgroup of a description',
    summary: 'The cover, size and quality of one description, e.g. one refined by hand.',
    role: 'property',
  },
  subgroups.subgroupOf,
)
fn(
  {
    key: 'subgroupRedundancy',
    name: 'Subgroup redundancy test',
    summary: 'Cover-based (Jaccard index of covers) or description-based (one contains the other) redundancy.',
    role: 'construction',
  },
  subgroups.subgroupRedundancy,
)
fn(
  {
    key: 'selectorLanguage',
    name: 'Selector language',
    summary:
      'Nominal (=, ≠) and numeric (≤, ≥) selectors over a table, with equal-frequency, equal-width or on-the-fly cut points; covers as bitsets.',
    role: 'construction',
  },
  language.selectorLanguage,
)
fn(
  {
    key: 'cutPoints',
    name: 'Cut points (equal-frequency, equal-width)',
    summary: 'Cut points that discretise a numeric column into b intervals of equal count or equal width.',
    role: 'transform',
    notes: ['histogram-density-estimation'],
  },
  language.cutPoints,
)
fn(
  {
    key: 'standardQuality',
    name: 'Klösgen quality family',
    tex: '(n/N)^a\\,(p - p_0)',
    summary: 'Size to the power a times the rate difference; a = 1 is WRAcc, a = ½ the binomial-test quality.',
    role: 'estimator',
  },
  quality.standardQuality,
)
fn(
  {
    key: 'wraccQuality',
    name: 'Weighted relative accuracy',
    tex: '\\tfrac{n}{N}\\,(p - p_0)',
    summary: 'Coverage times the rate difference; optimistic estimate (tp/N)(1 − p₀).',
    role: 'estimator',
  },
  quality.wraccQuality,
)
fn(
  {
    key: 'binomialQuality',
    name: 'Binomial-test quality',
    tex: '\\sqrt{n}\\,(p - p_0)/\\sqrt{p_0(1-p_0)}',
    summary: 'The one-sample z statistic of the subgroup’s positive rate against the population’s.',
    role: 'estimator',
    notes: ['binomial-test', 'z-test'],
  },
  quality.binomialQuality,
)
fn(
  {
    key: 'liftQuality',
    name: 'Lift',
    tex: 'p/p_0',
    summary: 'The subgroup’s positive rate over the population’s; bounded by min(1, tp/m)/p₀ under minimum support m.',
    role: 'estimator',
    notes: ['relative-lift'],
  },
  quality.liftQuality,
)
fn(
  { key: 'coverageQuality', name: 'Coverage', tex: 'n/N', summary: 'The share of rows covered.', role: 'estimator' },
  quality.coverageQuality,
)
fn(
  {
    key: 'chiSquareQuality',
    name: 'χ² quality',
    summary:
      'The χ² statistic of subgroup × target; optimistic estimate the larger χ² of the positives or negatives alone.',
    role: 'estimator',
    notes: ['chi-squared-distribution', 'nominal-association'],
  },
  quality.chiSquareQuality,
)
fn(
  {
    key: 'meanShiftQuality',
    name: 'Mean shift (z-score)',
    tex: 'n^a\\,(\\mu - \\mu_0)/\\sigma_0',
    summary: 'For a numeric target; a = ½ is the z-score of the subgroup mean, bounded by the best top-j prefix.',
    role: 'estimator',
    notes: ['z-test'],
  },
  quality.meanShiftQuality,
)
fn(
  {
    key: 'correlationModel',
    name: 'Exceptional model: correlation',
    summary:
      'Subgroups where the correlation of two numeric targets differs from the complement’s (Fisher z, absolute, entropy).',
    role: 'estimator',
    notes: ['pearson-correlation', 'simpsons-paradox'],
  },
  emm.correlationModel,
)
fn(
  {
    key: 'regressionModel',
    name: 'Exceptional model: regression slope',
    summary:
      'Subgroups whose regression slope differs: the t statistic of equal slopes, entropy-weighted difference, or Cook’s distance.',
    role: 'estimator',
    notes: ['leverage-and-influence', 'linear-regression'],
  },
  emm.regressionModel,
)
fn(
  {
    key: 'logisticModel',
    name: 'Exceptional model: logistic classifier',
    summary:
      'Subgroups where a logistic regression’s slope differs from the complement’s (Wald statistic of the interaction).',
    role: 'estimator',
    notes: ['logistic-regression'],
  },
  emm.logisticModel,
)
fn(
  {
    key: 'associationModel',
    name: 'Exceptional model: association',
    summary: 'Subgroups where the association (Yule’s Q) of two binary targets differs from the complement’s.',
    role: 'estimator',
    notes: ['nominal-association', 'bayesian-network'],
  },
  emm.associationModel,
)

/** The entries of one kind, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const all = { ...subgroups, ...sdmap, ...language, ...quality, ...emm }

/** The algorithm of the module, keyed by factory name. */
export const subgroupAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', all) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const subgroupFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', all) as Table<FunctionInfo>
