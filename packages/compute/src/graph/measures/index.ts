/**
 * `aifn-compute/graph/measures`: numbers that describe a graph, its labels or its node pairs.
 *
 * - Homophily of a labelled graph: `edgeHomophily` (the share of edges inside a class), `nodeHomophily` (the mean
 *   share of each node's neighbours in its class) and `adjustedHomophily` (corrected for class sizes and degrees, 0
 *   when edges ignore the labels).
 * - Bottlenecks: `cheegerConstant`, the exact Cheeger constant $h_G$ by enumeration with a minimising set, for graphs
 *   of up to `CHEEGER_MAX_NODES` nodes.
 * - Node pairs: `pairHeuristics`, the common-neighbour and Adamic-Adar scores of link prediction.
 *
 * Graphs are read as undirected (the Cheeger constant throws on a directed one); self-loops are ignored by the
 * homophily and pair scores. An undefined value (homophily without edges) is NaN, not an error.
 */

export { adjustedHomophily, edgeHomophily, nodeHomophily } from './homophily'
export { CHEEGER_MAX_NODES, cheegerConstant, type CheegerResult } from './cuts'
export { pairHeuristics, type NodePair, type PairHeuristics } from './links'
export { measuresFunctions } from './registry'
