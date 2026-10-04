/** The functions of `aifn-compute/graph/matrices`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as matrices from './matrices'

const fn = definer<FunctionInfo>('function', 'graph/matrices')
const SPECTRAL = ['spectral-clustering', 'laplacian-eigenmaps']

fn(
  { key: 'adjacencyMatrix', name: 'Adjacency matrix', role: 'construction', notes: ['graph-convolutional-network'] },
  matrices.adjacencyMatrix,
)
fn({ key: 'degreeMatrix', name: 'Degree matrix', role: 'construction', notes: SPECTRAL }, matrices.degreeMatrix)
fn({ key: 'degrees', name: 'Degrees', role: 'property' }, matrices.degrees)
fn({ key: 'incidenceMatrix', name: 'Incidence matrix', role: 'construction' }, matrices.incidenceMatrix)
fn(
  {
    key: 'laplacian',
    name: 'Graph Laplacian',
    tex: 'L = D - A',
    summary: 'The combinatorial, symmetric-normalised or random-walk Laplacian.',
    role: 'construction',
    notes: [...SPECTRAL, 'graph-convolutional-network', 'diffusion-maps'],
  },
  matrices.laplacian,
)

/** The functions of the module, keyed by name. */
export const matricesFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', matrices) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
