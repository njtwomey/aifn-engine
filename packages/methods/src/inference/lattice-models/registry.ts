/**
 * The registry of `aifn-methods/inference/lattice-models`.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as ising from './ising'

const fn = definer<FunctionInfo>('function', 'inference/lattice-models')
const notes = ['ising-model', 'markov-random-field']

fn({ key: 'isingModel', name: 'Ising model', role: 'construction', notes, cite: ['ising1925'] }, ising.isingModel)
fn(
  { key: 'isingLattice', name: 'Ising lattice factor graph', role: 'construction', notes: [...notes, 'factor-graph'] },
  ising.isingLattice,
)
fn({ key: 'isingShape', name: 'Shape of an Ising graph', role: 'property', notes }, ising.isingShape)
fn(
  {
    key: 'isingInference',
    name: 'Ising inference by shape',
    summary:
      'Exact chain sum-product, tree belief propagation, or loopy belief propagation, chosen by the graph shape.',
    role: 'inference',
    notes: [...notes, 'belief-propagation', 'loopy-belief-propagation'],
    cite: ['murphy1999'],
  },
  ising.isingInference,
)

/** The functions of the module. */
export const latticeModelFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', ising) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
