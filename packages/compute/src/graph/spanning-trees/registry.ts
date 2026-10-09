/**
 * The algorithms of `aifn-compute/graph/spanning-trees`, registered with what each factory takes (`problem`) and the
 * roles of its state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a
 * generic trace view picks default series and a worker can address an algorithm by key (design S §2.3); and its
 * function, `minimumSpanningTree`, registered with its role.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as trees from './trees'

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/spanning-trees')

algorithm(
  {
    key: 'kruskalSteps',
    name: 'Kruskal',
    problem: 'graph',
    state: { iterate: 'tree', objective: 'weight', flags: [] },
    cite: ['kruskal1956'],
  },
  trees.kruskalSteps,
)
algorithm(
  {
    key: 'primSteps',
    name: 'Prim',
    problem: 'graph',
    state: { iterate: 'tree', objective: 'weight', flags: [] },
    cite: ['prim1957'],
  },
  trees.primSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const spanningTreesAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', trees) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

definer<FunctionInfo>('function', 'graph/spanning-trees')(
  { key: 'minimumSpanningTree', name: 'Minimum spanning tree', role: 'solver', notes: ['hierarchical-clustering'] },
  trees.minimumSpanningTree,
)

/** The functions of the module, keyed by name. */
export const spanningTreesFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', trees) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
