/**
 * The algorithms of `aifn-compute/graph/shortest-paths`, registered with what each factory takes (`problem`) and the
 * roles of its state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a
 * generic trace view picks default series and a worker can address an algorithm by key (design S §2.3). The module's
 * functions are registered too, with their role and the notes that use them.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as paths from './paths'

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/shortest-paths')

algorithm(
  { key: 'dijkstraSteps', name: 'Dijkstra', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.dijkstraSteps,
)
algorithm(
  { key: 'aStarSteps', name: 'A*', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.aStarSteps,
)
algorithm(
  { key: 'bellmanFordSteps', name: 'Bellman–Ford', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.bellmanFordSteps,
)
algorithm(
  { key: 'floydWarshallSteps', name: 'Floyd–Warshall', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.floydWarshallSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const shortestPathsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', paths) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'graph/shortest-paths')

fn({ key: 'dijkstra', name: "Dijkstra's shortest paths", role: 'solver' }, paths.dijkstra)
fn({ key: 'aStar', name: 'A* search', role: 'solver' }, paths.aStar)
fn({ key: 'bellmanFord', name: 'Bellman–Ford', role: 'solver' }, paths.bellmanFord)
fn({ key: 'floydWarshall', name: 'Floyd–Warshall', role: 'solver', notes: ['isomap'] }, paths.floydWarshall)
fn({ key: 'shortestPath', name: 'Shortest path', role: 'solver' }, paths.shortestPath)

/** The functions of the module, keyed by name. */
export const shortestPathsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', paths) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
