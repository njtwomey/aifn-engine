/** The functions of `aifn-compute/graph/measures`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cuts from './cuts'
import * as homophily from './homophily'
import * as links from './links'

const fn = definer<FunctionInfo>('function', 'graph/measures')
const HETEROPHILY = ['graph-neural-networks-under-heterophily', 'graph-learning-benchmarks']

fn(
  {
    key: 'edgeHomophily',
    name: 'Edge homophily',
    tex: 'h_{\\text{edge}} = \\frac{|\\{(u, v) \\in E : y_u = y_v\\}|}{|E|}',
    summary: 'The share of edges whose two ends have the same label.',
    role: 'property',
    notes: HETEROPHILY,
    cite: ['zhu2020h2gcn'],
  },
  homophily.edgeHomophily,
)
fn(
  {
    key: 'nodeHomophily',
    name: 'Node homophily',
    summary: 'The mean, over nodes, of the share of their neighbours that have their label.',
    role: 'property',
    notes: HETEROPHILY,
    cite: ['pei2020geomgcn'],
  },
  homophily.nodeHomophily,
)
fn(
  {
    key: 'adjustedHomophily',
    name: 'Adjusted homophily',
    tex: 'h_{\\text{adj}} = \\frac{h_{\\text{edge}} - \\sum_k \\bar p_k^2}{1 - \\sum_k \\bar p_k^2}',
    summary: 'Edge homophily corrected for class sizes and degrees: 0 when edges ignore the labels.',
    role: 'property',
    notes: HETEROPHILY,
    cite: ['platonov2023characterizing'],
  },
  homophily.adjustedHomophily,
)
fn(
  {
    key: 'cheegerConstant',
    name: 'Cheeger constant',
    tex: 'h_G = \\min_S \\frac{|\\partial S|}{\\min(\\mathrm{vol}\\, S, \\mathrm{vol}\\, \\bar S)}',
    summary: 'The exact size of the worst bottleneck of a small graph, by enumerating its cuts.',
    role: 'property',
    notes: ['oversquashing-and-graph-rewiring'],
    cite: ['topping2022curvature'],
  },
  cuts.cheegerConstant,
)
fn(
  {
    key: 'pairHeuristics',
    name: 'Common neighbours and Adamic-Adar',
    tex: '\\mathrm{AA}(u, v) = \\sum_{w \\in N(u) \\cap N(v)} \\frac{1}{\\log \\deg w}',
    summary: 'Neighbourhood scores of node pairs: the classical link-prediction baselines.',
    role: 'property',
    notes: ['link-prediction-with-graph-neural-networks'],
    cite: ['libennowell2007'],
  },
  links.pairHeuristics,
)

/** The functions of the module, keyed by name. */
export const measuresFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', cuts, homophily, links) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
