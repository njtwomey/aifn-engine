/**
 * The algorithms of `aifn-compute/graph/traversal`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as components from './components'
import * as order from './order'
import * as traversal from './traversal'

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/traversal')

algorithm(
  { key: 'breadthFirstSteps', name: 'Breadth-first search', problem: 'graph', state: { iterate: 'order', flags: [] } },
  traversal.breadthFirstSteps,
)
algorithm(
  { key: 'depthFirstSteps', name: 'Depth-first search', problem: 'graph', state: { iterate: 'preorder', flags: [] } },
  traversal.depthFirstSteps,
)
algorithm(
  { key: 'iterativeDeepeningSteps', name: 'Iterative deepening', problem: 'graph', state: { flags: [] } },
  traversal.iterativeDeepeningSteps,
)
algorithm(
  { key: 'kahnSteps', name: 'Kahn topological sort', problem: 'graph', state: { iterate: 'order', flags: [] } },
  order.kahnSteps,
)
algorithm(
  {
    key: 'tarjanSteps',
    name: 'Tarjan strongly connected components',
    problem: 'graph',
    state: { iterate: 'component', flags: [] },
  },
  components.tarjanSteps,
)
algorithm(
  {
    key: 'kosarajuSteps',
    name: 'Kosaraju strongly connected components',
    problem: 'graph',
    state: { iterate: 'component', flags: [] },
  },
  components.kosarajuSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const traversalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', components, order, traversal) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'graph/traversal')

fn({ key: 'breadthFirstSearch', name: 'Breadth-first search', role: 'solver' }, traversal.breadthFirstSearch)
fn({ key: 'depthFirstSearch', name: 'Depth-first search', role: 'solver' }, traversal.depthFirstSearch)
fn({ key: 'iterativeDeepening', name: 'Iterative deepening', role: 'solver' }, traversal.iterativeDeepening)
fn(
  { key: 'unweightedShortestPaths', name: 'Unweighted shortest paths', role: 'solver' },
  traversal.unweightedShortestPaths,
)
fn(
  { key: 'topologicalSort', name: 'Topological sort', role: 'solver', notes: ['bayesian-network'] },
  order.topologicalSort,
)
fn({ key: 'isDag', name: 'Is a DAG', role: 'property', notes: ['bayesian-network'] }, order.isDag)
fn({ key: 'findCycle', name: 'Find a cycle', role: 'solver' }, order.findCycle)
fn(
  {
    key: 'connectedComponents',
    name: 'Connected components',
    role: 'solver',
    notes: ['density-based-spatial-clustering'],
  },
  components.connectedComponents,
)
fn(
  { key: 'stronglyConnectedComponents', name: 'Strongly connected components', role: 'solver' },
  components.stronglyConnectedComponents,
)
fn({ key: 'condensation', name: 'Condensation', role: 'transform' }, components.condensation)
fn({ key: 'bipartite', name: 'Bipartition', role: 'property' }, components.bipartite)

/** The functions of the module, keyed by name. */
export const traversalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', traversal, order, components) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
