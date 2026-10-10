/** Registry entries of `aifn-methods/learning/explanation`: studies of explanation methods. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as valuation from './valuation'

const fn = definer<FunctionInfo>('function', 'learning/explanation')

fn(
  {
    key: 'dataValuationStudy',
    name: 'Data valuation study',
    summary:
      'Influence, self-influence, TracIn, KNN-Shapley and TMC data Shapley of one logistic regression, ranked against planted label noise.',
    role: 'estimator',
    notes: ['detecting-and-cleaning-label-errors', 'leverage-and-influence', 'interpretability'],
  },
  valuation.dataValuationStudy,
)

/** The functions of the module, keyed by name. */
export const explanationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', valuation) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
