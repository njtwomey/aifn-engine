/**
 * The algorithms of `aifn-compute/graph/flows`, registered with what each factory takes (`problem`) and the roles of
 * its state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as flows from './flows'

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/flows')

algorithm(
  {
    key: 'edmondsKarpSteps',
    name: 'Edmonds–Karp',
    summary: 'Maximum flow by shortest augmenting paths.',
    problem: 'flow-network',
    state: { iterate: 'flow', objective: 'value', flags: [] },
  },
  flows.edmondsKarpSteps,
)
algorithm(
  {
    key: 'minCostFlowSteps',
    name: 'Minimum-cost flow',
    summary: 'Minimum-cost flow by successive shortest paths with potentials.',
    problem: 'flow-network',
    state: { iterate: 'flow', objective: 'cost', flags: [] },
  },
  flows.minCostFlowSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const flowsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', flows) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'graph/flows')

fn({ key: 'maxFlow', name: 'Maximum flow', role: 'solver', notes: ['optimal-transport'] }, flows.maxFlow)
fn(
  {
    key: 'minCostFlow',
    name: 'Minimum-cost flow',
    role: 'solver',
    notes: ['optimal-transport', 'earth-movers-distance'],
  },
  flows.minCostFlow,
)

/** The functions of the module, keyed by name. */
export const flowsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', flows) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
