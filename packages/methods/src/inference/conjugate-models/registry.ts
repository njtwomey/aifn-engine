/**
 * The registry of `aifn-methods/inference/conjugate-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cavi from './cavi'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/conjugate-models')
const fn = definer<FunctionInfo>('function', 'inference/conjugate-models')

algorithm(
  {
    key: 'caviNormalGamma',
    name: 'CAVI for a Gaussian with unknown mean and precision',
    summary: 'Coordinate ascent on q(μ)q(τ); each update raises the ELBO.',
    problem: 'gaussian-model',
    state: { objective: 'elbo', flags: ['converged', 'diverged'] },
    notes: [
      'coordinate-ascent-variational-inference-for-a-gaussian',
      'mean-field-variational-inference',
      'evidence-lower-bound',
    ],
    cite: ['bishop2006'],
  },
  cavi.caviNormalGamma,
)
fn(
  {
    key: 'normalGammaPosterior',
    name: 'Normal–gamma posterior',
    role: 'inference',
    notes: ['conjugate-priors', 'coordinate-ascent-variational-inference-for-a-gaussian'],
    cite: ['bishop2006'],
  },
  cavi.normalGammaPosterior,
)

/** The algorithms of the module. */
export const conjugateModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  cavi,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const conjugateModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  cavi,
) as Table<FunctionInfo>
