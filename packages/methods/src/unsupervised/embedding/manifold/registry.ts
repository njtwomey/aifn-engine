/** The registry of `aifn-methods/unsupervised/embedding/manifold`: neighbourhood graphs and the batch SOM. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as diffusion from './diffusion'
import * as manifold from './manifold'
import * as som from './som'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
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

/** The algorithms of the module, keyed by factory name. */
export const manifoldAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', som) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const manifoldFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  manifold,
  som,
  diffusion,
) as Table<FunctionInfo>
