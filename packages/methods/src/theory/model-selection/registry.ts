/** The registry of `aifn-methods/theory/model-selection`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as ms from './selection'

const fn = definer<FunctionInfo>('function', 'theory/model-selection')

fn(
  {
    key: 'selectionStudy',
    name: 'Selection and assessment over many data sets',
    summary: 'Tune a penalty by CV on many data sets; best CV, nested CV and test estimates against the true error.',
    role: 'simulation',
    random: true,
    cite: ['varma2006', 'cawley2010', 'krstajic2014', 'hastie2009', 'eilers1996'],
  },
  ms.selectionStudy,
)

/** The functions of the module, keyed by name. */
export const modelSelectionFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', ms) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
