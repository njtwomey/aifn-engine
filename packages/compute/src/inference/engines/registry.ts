/**
 * The functions of `aifn-compute/inference/engines`.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as engines from './engines'

definer<FunctionInfo>('function', 'inference/engines')(
  {
    key: 'infer',
    name: 'Infer',
    summary: 'Pick an inference engine by the shape of the model: forward–backward, belief propagation, EP or Gibbs.',
    role: 'inference',
    notes: ['probabilistic-programming', 'infer-net', 'model-based-machine-learning'],
    cite: ['koller2009'],
  },
  engines.infer,
)

/** The functions of the module, keyed by name. */
export const enginesFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', engines) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
