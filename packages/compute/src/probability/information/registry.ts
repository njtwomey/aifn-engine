/**
 * The functions of `aifn-compute/probability/information`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bounds from './bounds'
import * as continuous from './continuous'
import * as measures from './measures'

const fn = definer<FunctionInfo>('function', 'probability/information')
const F = ['f-divergences-and-jensen-shannon']

fn(
  {
    key: 'entropy',
    name: 'Entropy',
    tex: 'H(p)',
    role: 'estimator',
    notes: ['entropy', 'source-coding-theorem'],
    cite: ['shannon1948', 'cover2006'],
  },
  measures.entropy,
)
fn(
  {
    key: 'jointEntropy',
    name: 'Joint entropy',
    role: 'estimator',
    notes: ['joint-and-conditional-entropy'],
    cite: ['cover2006'],
  },
  measures.jointEntropy,
)
fn(
  {
    key: 'conditionalEntropy',
    name: 'Conditional entropy',
    role: 'estimator',
    notes: ['joint-and-conditional-entropy'],
    cite: ['cover2006'],
  },
  measures.conditionalEntropy,
)
fn(
  { key: 'crossEntropy', name: 'Cross-entropy', role: 'estimator', notes: ['cross-entropy-and-perplexity'] },
  measures.crossEntropy,
)
fn(
  {
    key: 'klDivergence',
    name: 'Kullback–Leibler divergence',
    tex: 'D_{KL}(p \\| q)',
    role: 'estimator',
    notes: ['kullback-leibler-divergence'],
    cite: ['kullback1951'],
  },
  measures.klDivergence,
)
fn(
  { key: 'jensenShannonDivergence', name: 'Jensen–Shannon divergence', role: 'estimator', notes: F, cite: ['lin1991'] },
  measures.jensenShannonDivergence,
)
fn(
  { key: 'jensenShannonDistance', name: 'Jensen–Shannon distance', role: 'estimator', notes: F, cite: ['lin1991'] },
  measures.jensenShannonDistance,
)
fn({ key: 'totalVariation', name: 'Total variation distance', role: 'estimator', notes: F }, measures.totalVariation)
fn({ key: 'hellingerDistance', name: 'Hellinger distance', role: 'estimator', notes: F }, measures.hellingerDistance)
fn({ key: 'fDivergence', name: 'f-divergence', role: 'estimator', notes: F, cite: ['ali1966'] }, measures.fDivergence)
fn(
  {
    key: 'mutualInformation',
    name: 'Mutual information',
    tex: 'I(X; Y)',
    role: 'estimator',
    notes: ['mutual-information'],
    cite: ['cover2006'],
  },
  measures.mutualInformation,
)
fn(
  {
    key: 'pointwiseMutualInformation',
    name: 'Pointwise mutual information',
    role: 'estimator',
    notes: ['mutual-information'],
  },
  measures.pointwiseMutualInformation,
)
fn(
  { key: 'differentialEntropy', name: 'Differential entropy', role: 'property', notes: ['differential-entropy'] },
  measures.differentialEntropy,
)
fn(
  {
    key: 'gaussianMutualInformation',
    name: 'Gaussian mutual information',
    summary: '−½ log det of the correlation structure between blocks of a Gaussian.',
    role: 'estimator',
    notes: ['mutual-information', 'differential-entropy'],
  },
  continuous.gaussianMutualInformation,
)
fn(
  {
    key: 'ksgMutualInformation',
    name: 'KSG mutual information estimator',
    summary: 'The Kraskov–Stögbauer–Grassberger k-nearest-neighbour estimator of mutual information.',
    role: 'estimator',
    notes: ['mutual-information'],
  },
  continuous.ksgMutualInformation,
)

// Variational bounds on mutual information from critic scores (Poole et al. 2019).
const MI_BOUNDS = ['mutual-information-maximisation']

fn(
  {
    key: 'infoNceBound',
    name: 'InfoNCE bound',
    tex: '\\tfrac1n \\textstyle\\sum_i \\big[T_{ii} - \\log \\tfrac1n \\sum_j e^{T_{ij}}\\big] \\le \\log n',
    summary: 'A lower bound on mutual information from a batch of critic scores; it never exceeds log n.',
    role: 'estimator',
    notes: MI_BOUNDS,
    cite: ['oord2018', 'poole2019'],
  },
  bounds.infoNceBound,
)
fn(
  {
    key: 'nwjBound',
    name: 'NWJ bound',
    tex: '\\tfrac1n \\textstyle\\sum_i T_{ii} - e^{-1} \\operatorname{mean}_{i \\ne j} e^{T_{ij}}',
    summary: 'The Nguyen–Wainwright–Jordan lower bound on mutual information: unbiased, variance growing as e^I.',
    role: 'estimator',
    notes: MI_BOUNDS,
    cite: ['nguyen2010', 'poole2019'],
  },
  bounds.nwjBound,
)
fn(
  {
    key: 'donskerVaradhanBound',
    name: 'Donsker–Varadhan bound',
    tex: '\\tfrac1n \\textstyle\\sum_i T_{ii} - \\log \\operatorname{mean}_{i \\ne j} e^{T_{ij}}',
    summary: 'The Donsker–Varadhan lower bound on mutual information that MINE maximises.',
    role: 'estimator',
    notes: MI_BOUNDS,
    cite: ['poole2019'],
  },
  bounds.donskerVaradhanBound,
)
fn(
  {
    key: 'correlationForMutualInformation',
    name: 'Gaussian correlation for a mutual information',
    tex: '\\rho = \\sqrt{1 - e^{-2I/d}}',
    role: 'property',
    notes: MI_BOUNDS,
  },
  bounds.correlationForMutualInformation,
)

/** The functions of the module, keyed by name. */
export const informationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', measures, continuous, bounds) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
