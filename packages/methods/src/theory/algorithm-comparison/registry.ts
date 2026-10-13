/** The registry of `aifn-methods/theory/algorithm-comparison`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as typeI from './type-i-error'

const fn = definer<FunctionInfo>('function', 'theory/algorithm-comparison')
const cite = ['dietterich1998', 'nadeau2003', 'alpaydin1999']

fn(
  {
    key: 'typeIErrorStudy',
    name: 'Type I error of comparison tests',
    summary: 'Two equally good learners on many simulated data sets: how often each comparison test rejects.',
    role: 'simulation',
    random: true,
    cite,
  },
  typeI.typeIErrorStudy,
)
fn(
  {
    key: 'typeIErrorStudySteps',
    name: 'Type I error study, streamed',
    summary: 'The Type I error study one data set at a time, yielding the rates so far.',
    role: 'simulation',
    random: true,
    cite,
  },
  typeI.typeIErrorStudySteps,
)
fn(
  {
    key: 'typeIErrorReplicate',
    name: 'Comparison tests on one data set',
    summary: 'The p-values of six tests comparing two equally good learners on one simulated data set.',
    role: 'simulation',
    random: true,
    cite,
  },
  typeI.typeIErrorReplicate,
)

/** The functions of the module, keyed by name. */
export const algorithmComparisonFunctions: Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
> = entries<FunctionInfo>('function', typeI) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
