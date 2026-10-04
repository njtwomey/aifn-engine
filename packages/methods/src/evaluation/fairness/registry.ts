/** The functions of `aifn-methods/evaluation/fairness` besides its metrics. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as fairness from './fairness'

definer<FunctionInfo>('function', 'evaluation/fairness')(
  {
    key: 'groupRates',
    name: 'Rates per group',
    summary: 'Selection, true-positive and false-positive rates per protected group.',
    role: 'estimator',
    notes: ['group-fairness-metrics'],
    cite: ['hardt2016'],
  },
  fairness.groupRates,
)

/** The functions of the module that are not metrics, keyed by name. */
export const fairnessFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', fairness) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
