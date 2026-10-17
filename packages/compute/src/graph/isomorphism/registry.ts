/** The functions and the stepped algorithm of `aifn-compute/graph/isomorphism`, registered with their notes. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as subgraphs from './subgraphs'
import * as wl from './weisfeiler-leman'

const fn = definer<FunctionInfo>('function', 'graph/isomorphism')
const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/isomorphism')
const WL = [
  'weisfeiler-leman-test',
  'graph-isomorphism-network',
  'expressive-power-of-message-passing',
  'higher-order-and-subgraph-graph-neural-networks',
]

fn(
  {
    key: 'weisfeilerLeman',
    name: 'Weisfeiler-Leman colour refinement',
    tex: 'c^{(t+1)}_v = \\mathrm{hash}\\big(c^{(t)}_v, \\{\\!\\{ c^{(t)}_u : u \\in N(v) \\}\\!\\}\\big)',
    summary: 'Refine node colours by the multiset of neighbour colours until stable; differing histograms separate.',
    role: 'property',
    notes: WL,
    cite: ['weisfeiler1968', 'shervashidze2011', 'morris2019weisfeiler'],
  },
  wl.weisfeilerLeman,
)
fn(
  {
    key: 'folkloreWeisfeilerLeman2',
    name: 'Folklore 2-WL test',
    summary: 'Refine the colours of ordered node pairs through every third node: as strong as 3-WL.',
    role: 'property',
    notes: ['higher-order-and-subgraph-graph-neural-networks'],
    cite: ['cai1992', 'morris2019weisfeiler'],
  },
  wl.folkloreWeisfeilerLeman2,
)
fn(
  {
    key: 'subgraphBag',
    name: 'Subgraph selection policy',
    summary: 'The bag of node-deleted, edge-deleted or ego-network subgraphs of a graph.',
    role: 'construction',
    notes: ['higher-order-and-subgraph-graph-neural-networks'],
    cite: ['bevilacqua2022esan'],
  },
  subgraphs.subgraphBag,
)
fn(
  {
    key: 'subgraphWeisfeilerLeman',
    name: 'DS-WL test',
    summary: 'Colour refinement on every subgraph of a bag, comparing the bags of stable histograms.',
    role: 'property',
    notes: ['higher-order-and-subgraph-graph-neural-networks'],
    cite: ['bevilacqua2022esan'],
  },
  subgraphs.subgraphWeisfeilerLeman,
)
algorithm(
  {
    key: 'weisfeilerLemanSteps',
    name: 'Weisfeiler-Leman colour refinement',
    problem: 'graph',
    state: { iterate: 'colours', flags: [] },
    notes: WL,
    cite: ['weisfeiler1968', 'shervashidze2011'],
  },
  wl.weisfeilerLemanSteps,
)

/** The functions of the module, keyed by name. */
export const isomorphismFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', wl, subgraphs) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

/** The stepped algorithms of the module, keyed by factory name. */
export const isomorphismAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', wl) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
