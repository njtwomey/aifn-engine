/** The registry of `aifn-methods/theory/shrinkage`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as shrinkage from './shrinkage'

const fn = definer<FunctionInfo>('function', 'theory/shrinkage')

fn(
  {
    key: 'jamesStein',
    name: 'James-Stein estimator',
    summary:
      'Shrink a p-dimensional normal observation towards zero, a target or its grand mean; optionally positive part.',
    role: 'estimator',
    cite: ['james1961', 'stein1956', 'efron1973'],
  },
  shrinkage.jamesStein,
)
fn(
  {
    key: 'jamesSteinRisk',
    name: "Stein's paradox by simulation",
    summary: 'Risk of the MLE, James-Stein and its positive part against the distance of the mean from the target.',
    role: 'simulation',
    random: true,
    cite: ['james1961', 'efron1975'],
  },
  shrinkage.jamesSteinRisk,
)
fn(
  {
    key: 'jamesSteinExactRisk',
    name: 'Exact James-Stein risk',
    summary: 'The risk p - (p - 2)^2 E[1 / chi-squared_p(lambda)] of James-Stein, as a Poisson mixture.',
    role: 'property',
    cite: ['james1961', 'efron1975'],
  },
  shrinkage.jamesSteinExactRisk,
)

/** The functions of the module, keyed by name. */
export const shrinkageFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', shrinkage) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
