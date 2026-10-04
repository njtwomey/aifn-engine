/** The functions of `aifn-methods/learning/generalised/ordinal`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as decomposition from './decomposition'
import * as thresholds from './thresholds'

const fn = definer<FunctionInfo>('function', 'learning/generalised/ordinal')

fn(
  {
    key: 'thresholdClasses',
    name: 'Classes from thresholds',
    role: 'transform',
    notes: ['cumulative-link-model', 'ordinal-regression'],
  },
  thresholds.thresholdClasses,
)
fn(
  {
    key: 'thresholdPenalty',
    name: 'Threshold ordering penalty',
    role: 'construction',
    notes: ['ordering-constraints-on-ordinal-thresholds'],
  },
  thresholds.thresholdPenalty,
)
fn(
  {
    key: 'differenceExceedance',
    name: 'Exceedance probabilities to class probabilities',
    role: 'transform',
    notes: ['binary-decomposition-for-ordinal-regression'],
  },
  decomposition.differenceExceedance,
)

/** The functions of the module, keyed by name. */
export const ordinalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', thresholds, decomposition) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
