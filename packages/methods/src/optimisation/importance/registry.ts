/**
 * Registry entries of `aifn-methods/optimisation/importance`: the exact functional ANOVA of a tabulated surface and
 * its reconstruction.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as fanova from './fanova'

const fn = definer<FunctionInfo>('function', 'optimisation/importance')

fn(
  {
    key: 'fanova',
    name: 'Functional ANOVA of a tabulated surface',
    summary:
      'Exact main effects, pairwise and higher interactions of a surface on a full grid, with each term’s fraction of the variance.',
    role: 'estimator',
    cite: ['hutter2014'],
  },
  fanova.fanova,
)

fn(
  {
    key: 'fanovaReconstruct',
    name: 'Functional ANOVA reconstruction',
    summary: 'The table rebuilt from the mean and the terms up to an order: the best additive fit at order 1.',
    role: 'transform',
    cite: ['hutter2014'],
  },
  fanova.fanovaReconstruct,
)

/** The functions of the module, keyed by name. */
export const importanceFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', fanova) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
