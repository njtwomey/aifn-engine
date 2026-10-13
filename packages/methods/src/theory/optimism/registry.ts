/** The registry of `aifn-methods/theory/optimism`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as op from './optimism'

const fn = definer<FunctionInfo>('function', 'theory/optimism')

fn(
  {
    key: 'optimismStudy',
    name: 'Optimism of the training error',
    summary: 'Refit one model to many noisy samples; training, in-sample and extra-sample errors, optimism and df.',
    role: 'simulation',
    random: true,
    cite: ['efron1986', 'efron2004covariance', 'ye1998', 'hastie2009'],
  },
  op.optimismStudy,
)
fn(
  {
    key: 'optimismSweep',
    name: 'Optimism across complexity',
    summary: 'The optimism study for each polynomial degree, neighbour count, ridge penalty or number of candidates.',
    role: 'simulation',
    random: true,
    cite: ['hastie2009', 'ye1998'],
  },
  op.optimismSweep,
)

/** The functions of the module, keyed by name. */
export const optimismFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', op) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
