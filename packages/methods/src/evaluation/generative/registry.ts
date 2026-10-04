/** The functions of `aifn-methods/evaluation/generative` besides its metrics. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as generative from './generative'

const fn = definer<FunctionInfo>('function', 'evaluation/generative')

fn(
  {
    key: 'frechetDistance',
    name: 'Fréchet distance between Gaussians',
    tex: '\\|\\mu_1 - \\mu_2\\|^2 + \\operatorname{tr}(\\Sigma_1 + \\Sigma_2 - 2(\\Sigma_1\\Sigma_2)^{1/2})',
    role: 'estimator',
    notes: ['frechet-inception-distance', 'wasserstein-distances'],
    cite: ['heusel2017'],
  },
  generative.frechetDistance,
)
fn(
  {
    key: 'kidSubsets',
    name: 'KID over subsets',
    role: 'estimator',
    notes: ['kernel-inception-distance'],
    cite: ['binkowski2018'],
  },
  generative.kidSubsets,
)
fn(
  {
    key: 'inceptionScoreSplits',
    name: 'Inception score over splits',
    role: 'estimator',
    notes: ['inception-score'],
    cite: ['salimans2016'],
  },
  generative.inceptionScoreSplits,
)
fn(
  {
    key: 'generativePrecisionRecall',
    name: 'Precision and recall for generative models',
    role: 'estimator',
    notes: ['precision-and-recall-for-generative-models'],
    cite: ['kynkaanniemi2019'],
  },
  generative.generativePrecisionRecall,
)
fn(
  {
    key: 'densityCoverage',
    name: 'Density and coverage',
    role: 'estimator',
    notes: ['precision-and-recall-for-generative-models'],
    cite: ['naeem2020'],
  },
  generative.densityCoverage,
)

/** The functions of the module that are not metrics, keyed by name. */
export const generativeEvaluationFunctions: Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
> = entries<FunctionInfo>('function', generative) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
