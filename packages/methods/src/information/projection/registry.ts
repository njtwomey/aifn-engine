/** The functions of `aifn-methods/information/projection`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as projection from './projection'

definer<FunctionInfo>('function', 'information/projection')(
  {
    key: 'normalProjection',
    name: 'Moment-matching projection onto a Gaussian',
    summary: 'The Gaussian minimising KL(p ‖ q): the one with the mean and variance of p.',
    role: 'transform',
    notes: ['kullback-leibler-divergence', 'expectation-propagation', 'assumed-density-filtering'],
  },
  projection.normalProjection,
)

/** The functions of the module, keyed by name. */
export const projectionFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', projection) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
