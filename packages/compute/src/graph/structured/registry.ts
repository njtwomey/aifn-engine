/**
 * The functions of `aifn-compute/graph/structured` (intentional graphs with roles, plates and templates), registered
 * with their display names, roles and the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as build from './build'
import * as diagram from './diagram'
import * as shape from './shape'
import * as unroll from './unroll'

const fn = definer<FunctionInfo>('function', 'graph/structured')
const PGM = ['bayesian-network', 'factor-graph', 'markov-random-field']

fn(
  {
    key: 'structured',
    name: 'Structured graph',
    summary: 'A graph with node roles (observed, latent, factor, parameter), plates and templates.',
    role: 'construction',
    returns: 'graph',
    notes: PGM,
  },
  build.structured,
)
fn(
  { key: 'structuredGraph', name: 'Structured graph from parts', role: 'construction', returns: 'graph', notes: PGM },
  build.structuredGraph,
)
fn(
  { key: 'chainTemplate', name: 'Chain template', role: 'construction', notes: ['hidden-markov-model', ...PGM] },
  build.chainTemplate,
)
fn(
  {
    key: 'latticeTemplate',
    name: 'Lattice template',
    role: 'construction',
    notes: ['markov-random-field', 'ising-model'],
  },
  build.latticeTemplate,
)
fn(
  { key: 'treeTemplate', name: 'Tree template', role: 'construction', notes: ['belief-propagation'] },
  build.treeTemplate,
)
fn(
  { key: 'unroll', name: 'Unroll templates and plates', role: 'transform', returns: 'graph', notes: PGM },
  unroll.unroll,
)
fn(
  {
    key: 'shape',
    name: 'Graph shape',
    summary: 'Chain, tree, lattice or general, so algorithms take fast paths.',
    role: 'property',
    notes: ['belief-propagation', 'junction-tree-algorithm'],
  },
  shape.shape,
)
fn(
  {
    key: 'markovBlanket',
    name: 'Markov blanket',
    role: 'property',
    notes: ['bayesian-network', 'markov-random-field'],
  },
  shape.markovBlanket,
)
fn({ key: 'toDiagram', name: 'Graph diagram', role: 'transform', notes: PGM }, diagram.toDiagram)

/** The functions of the module, keyed by name. */
export const structuredFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', build, unroll, shape, diagram) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
