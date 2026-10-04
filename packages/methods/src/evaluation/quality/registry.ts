/** The functions of `aifn-methods/evaluation/quality` besides its metrics. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as signal from './signal'

const fn = definer<FunctionInfo>('function', 'evaluation/quality')

fn(
  { key: 'ssimMap', name: 'SSIM map', role: 'estimator', notes: ['structural-similarity-index'], cite: ['wang2004'] },
  signal.ssimMap,
)
fn(
  {
    key: 'permutationInvariantScore',
    name: 'Permutation-invariant score',
    summary: 'The best score over assignments of estimated to reference sources.',
    role: 'estimator',
    notes: ['permutation-invariant-training'],
    cite: ['yu2017'],
  },
  signal.permutationInvariantScore,
)

/** The functions of the module that are not metrics, keyed by name. */
export const qualityFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', signal) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
