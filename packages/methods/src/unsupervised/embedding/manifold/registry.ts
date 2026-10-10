/**
 * The registry of `aifn-methods/unsupervised/embedding/manifold`: the neighbourhood graph and the SOM grid as
 * functions with the notes they serve, and the batch SOM as a step-through algorithm (the roles of its state's fields
 * in `state`). The estimators (`isomap`, `laplacianEigenmaps`, `locallyLinearEmbedding`, `diffusionMap`,
 * `selfOrganisingMap`) are registered as models beside their code.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as diffusion from './diffusion'
import * as manifold from './manifold'
import * as som from './som'

/** A registry table of the module: entries keyed by name, each a function with its `info`. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
/** Registers a function of the module under `unsupervised/embedding/manifold`. */
const fn = definer<FunctionInfo>('function', 'unsupervised/embedding/manifold')

fn(
  {
    key: 'neighbourGraph',
    name: 'Neighbourhood graph',
    role: 'construction',
    notes: ['isomap', 'laplacian-eigenmaps', 'locally-linear-embedding'],
  },
  manifold.neighbourGraph,
)
fn({ key: 'somGrid', name: 'SOM grid positions', role: 'construction', notes: ['self-organising-maps'] }, som.somGrid)
definer<AlgorithmInfo>('algorithm', 'unsupervised/embedding/manifold')(
  {
    key: 'selfOrganisingMapSteps',
    name: 'Batch self-organising map',
    summary: 'Best-matching units, then neighbourhood-weighted means of the rows under a shrinking radius.',
    problem: 'objective',
    state: { iterate: 'weights', objective: 'quantisationError', flags: ['diverged'] },
    random: true,
    notes: ['self-organising-maps'],
    cite: ['kohonen1982', 'kohonen2013'],
  },
  som.selfOrganisingMapSteps,
)

/**
 * The step-through algorithms of the module, keyed by factory name: `selfOrganisingMapSteps`, with its `info`.
 *
 * @example The algorithm and what its state reports
 * for (const [key, entry] of Object.entries(manifoldAlgorithms)) print(key, entry.info.state)
 */
export const manifoldAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', som) as Table<AlgorithmInfo>
/**
 * The functions of the module, keyed by name: `neighbourGraph` and `somGrid`, each with its `info`.
 *
 * @example The functions and the notes they serve
 * for (const [key, entry] of Object.entries(manifoldFunctions)) print(key, entry.info.notes)
 */
export const manifoldFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  manifold,
  som,
  diffusion,
) as Table<FunctionInfo>
