/**
 * The functions of `aifn-compute/optim/minimize`.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as minimize from './minimize'

definer<FunctionInfo>('function', 'optim/minimize')(
  {
    key: 'minimize',
    name: 'Minimise',
    summary: 'One entry point over the registered optimisers, as scipy.optimize.minimize.',
    role: 'solver',
    notes: ['gradient-descent', 'quasi-newton-methods', 'nelder-mead'],
  },
  minimize.minimize,
)

/** The functions of the module, keyed by name. */
export const minimizeFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', minimize) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
