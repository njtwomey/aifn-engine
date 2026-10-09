/**
 * Queries on a structured graph's dependence structure: the interaction scopes among its free variables (with the
 * observed ones conditioned on), its shape (chain, tree, lattice, DAG or general), the order of a chain, and Markov
 * blankets (Pearl 1988, "Probabilistic Reasoning in Intelligent Systems", §3.2; Koller & Friedman 2009, §3.2.2 and
 * §4.3). The shape picks an inference path: forward–backward on a chain, exact belief propagation on a tree.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { Index } from 'aifn-compute/foundation/contracts'
import { isDag } from 'aifn-compute/graph/traversal'
import { unionFind, unite } from '../heap'
import { groupChain } from './build'
import { unroll, type SizeBindings } from './unroll'
import type { GraphShape, StructuredGraph } from './types'

/**
 * Whether node `i` is a free variable: latent or deterministic, so not conditioned on.
 *
 * @param g The graph.
 * @param i The node's index.
 * @returns True for a latent or deterministic node.
 */
const isFree = (g: StructuredGraph, i: Index) => {
  const r = g.attributes[i].role
  return r === 'latent' || r === 'deterministic'
}
/**
 * Whether node `i` is a variable (observed, latent or deterministic) rather than a factor or a parameter.
 *
 * @param g The graph.
 * @param i The node's index.
 * @returns True unless the node is a factor or a parameter.
 */
const isVariable = (g: StructuredGraph, i: Index) =>
  g.attributes[i].role !== 'factor' && g.attributes[i].role !== 'parameter'

/**
 * The interactions among the free (latent and deterministic) variables of an explicit graph, with observed variables
 * and parameters conditioned on: one scope per factor node (its free neighbours), per variable with directed parents
 * (the variable and its parents: the moral graph's cliques), and per undirected edge between variables. Scopes of
 * fewer than two variables are dropped (they do not change the shape); each scope is ascending node indices. Two
 * interactions over the same variables stay two scopes, as two factors do in a factor graph (a loop for BP). An
 * observed child keeps its free parents in one scope, as moralisation does.
 *
 * @param graph An explicit graph (unrolled, or with no groups): lags are not followed.
 * @returns The scopes: those of undirected edges between variables, in edge order, then one per factor, then one
 *   per child with parents (each of those two in the order of its first edge).
 *
 * @example The sprinkler network: a v-structure gives one three-variable scope
 * const sprinkler = structuredGraph({
 *   nodes: ['cloudy', 'sprinkler', 'rain', 'wet'].map((name) => ({ name, role: 'latent', group: null })),
 *   edges: [
 *     { from: 'cloudy', to: 'sprinkler' }, { from: 'cloudy', to: 'rain' },
 *     { from: 'sprinkler', to: 'wet' }, { from: 'rain', to: 'wet' },
 *   ],
 * })
 * print('scopes:', interactionScopes(sprinkler))
 */
export function interactionScopes(graph: StructuredGraph): Index[][] {
  const g = graph
  const scopes: Index[][] = []
  const add = (vars: Iterable<Index>) => {
    const s = [...new Set(vars)].filter((i) => isFree(g, i)).sort((a, b) => a - b)
    if (s.length >= 2) scopes.push(s)
  }
  const factorScope = new Map<Index, Index[]>()
  const parents = new Map<Index, Index[]>()
  for (const e of g.edges) {
    const [a, b] = [g.attributes[e.from].role, g.attributes[e.to].role]
    if (a === 'factor' || b === 'factor') {
      const [f, v] = a === 'factor' ? [e.from, e.to] : [e.to, e.from]
      factorScope.set(f, [...(factorScope.get(f) ?? []), v])
    } else if (!isVariable(g, e.from) || !isVariable(g, e.to)) continue
    else if (e.directed) parents.set(e.to, [...(parents.get(e.to) ?? []), e.from])
    else add([e.from, e.to])
  }
  for (const vs of factorScope.values()) add(vs)
  for (const [child, ps] of parents) add([child, ...ps])
  return scopes
}

/**
 * Fill unbound named sizes with 3, the smallest size at which a template shows its shape.
 *
 * @param graph The compact graph whose named sizes (`graph.sizes`) are bound.
 * @param sizes The caller's bindings, if any; a function is returned as it is.
 * @returns Bindings for every named size: the caller's value where given, else 3.
 */
function defaultSizes(graph: StructuredGraph, sizes: SizeBindings | undefined): SizeBindings {
  if (typeof sizes === 'function') return sizes
  const out: Record<string, number | readonly number[]> = {}
  for (const s of graph.sizes) out[s] = sizes?.[s] ?? 3
  return out
}

/**
 * The shape of the graph's free variables with the observed ones conditioned on:
 * - `chain` when their interactions are pairwise and form one path (an HMM, a linear-chain CRF; a single variable);
 * - `tree` when the factor graph of their interactions has no cycle (a forest; higher-order factors allowed);
 * - `lattice` when every free variable lies in a lattice template;
 * - `dag` when every edge is directed and the directed graph is acyclic (a Bayesian network with loops in its moral
 *   graph);
 * - `general` otherwise.
 *
 * The tests are tried in this order, so an acyclic lattice (a single row) is a chain or a tree. A compact graph is
 * unrolled first against `sizes` (unbound sizes read as 3).
 *
 * @param graph The graph, compact or unrolled.
 * @param sizes The values of named sizes to unroll a compact graph with; any left out are 3.
 * @returns The shape.
 *
 * @example The templates and a few small graphs
 * print('HMM:', shape(chainTemplate('T', { observed: 'x' })))
 * print('binary tree:', shape(treeTemplate(2, 'D')))
 * print('Ising:', shape(latticeTemplate(3, 3)))
 * const latent = (name) => ({ name, role: 'latent', group: null })
 * const diamond = structuredGraph({
 *   nodes: ['a', 'b', 'c', 'd'].map(latent),
 *   edges: [{ from: 'a', to: 'b' }, { from: 'a', to: 'c' }, { from: 'b', to: 'd' }, { from: 'c', to: 'd' }],
 * })
 * print('diamond:', shape(diamond))
 * const triangle = structuredGraph({
 *   nodes: ['a', 'b', 'c'].map(latent),
 *   edges: [['a', 'b'], ['b', 'c'], ['c', 'a']].map(([from, to]) => ({ from, to, directed: false })),
 * })
 * print('triangle MRF:', shape(triangle))
 */
export function shape(graph: StructuredGraph, sizes?: SizeBindings): GraphShape {
  const g = graph.groups.length && !graph.unrolled ? unroll(graph, defaultSizes(graph, sizes)) : graph
  const scopes = interactionScopes(g)
  const free = g.attributes.flatMap((_, i) => (isFree(g, i) ? [i] : []))
  // Acyclic factor graph: variables and scopes as a bipartite graph without cycles.
  const uf = unionFind(g.nodes + scopes.length)
  let acyclic = true
  scopes.forEach((s, k) => {
    for (const v of s) if (!unite(uf, v, g.nodes + k)) acyclic = false
  })
  if (acyclic) {
    const degree = new Map<Index, number>()
    for (const s of scopes) for (const v of s) degree.set(v, (degree.get(v) ?? 0) + 1)
    const pairwise = scopes.every((s) => s.length === 2)
    // Connected pieces among the free variables (an isolated variable is a piece of its own).
    const joined = unionFind(g.nodes)
    for (const s of scopes) for (const v of s.slice(1)) unite(joined, s[0], v)
    const pieces = joined.count - (g.nodes - free.length)
    if (pairwise && [...degree.values()].every((d) => d <= 2) && pieces <= 1) return 'chain'
    return 'tree'
  }
  const inLattice = (i: Index) => groupChain(g, g.attributes[i].group).some((q) => q.kind === 'lattice')
  if (free.length && free.every(inLattice)) return 'lattice'
  if (g.edges.every((e) => e.directed) && g.attributes.every((n) => n.role !== 'factor') && isDag(g)) return 'dag'
  return 'general'
}

/**
 * The free variables of a chain-shaped graph in path order (node indices of the explicit graph, starting at the end
 * with the smaller index), or null when the graph is not a chain. A compact graph is unrolled against `sizes` first,
 * so the indices refer to `unroll(graph, sizes)`; every named size must then be bound (`AifnError` otherwise).
 *
 * @param graph The graph, compact or unrolled.
 * @param sizes The values of the named sizes of a compact graph.
 * @returns Node indices of the explicit graph along the chain (empty when it has no free variable), or null.
 *
 * @example The latent chain of an HMM
 * const hmm = chainTemplate('T', { observed: 'x' })
 * const order = chainOrder(hmm, { T: 3 })
 * print('order:', order)
 * print('names:', order.map((i) => unroll(hmm, { T: 3 }).attributes[i].name))
 * print('a tree of depth 1, a path:', chainOrder(treeTemplate(2, 1)))
 * print('a tree of depth 2:', chainOrder(treeTemplate(2, 2)))
 */
export function chainOrder(graph: StructuredGraph, sizes: SizeBindings = {}): Index[] | null {
  const g = graph.groups.length && !graph.unrolled ? unroll(graph, sizes) : graph
  if (shape(g) !== 'chain') return null
  const free = g.attributes.flatMap((_, i) => (isFree(g, i) ? [i] : []))
  const next = new Map<Index, Index[]>(free.map((v) => [v, []]))
  for (const [a, b] of interactionScopes(g)) {
    next.get(a)!.push(b)
    next.get(b)!.push(a)
  }
  if (free.length === 0) return []
  const start = free.find((v) => next.get(v)!.length <= 1)!
  const order = [start]
  for (let prev = -1, v = start; ;) {
    const w = next.get(v)!.find((u) => u !== prev)
    if (w === undefined) break
    order.push(w)
    ;[prev, v] = [v, w]
  }
  return order
}

/** The Markov blanket of a node of a structured graph, by name. */
export interface Blanket {
  /** Directed parents. */
  parents: string[]
  /** Directed children. */
  children: string[]
  /** The children's other parents. */
  coParents: string[]
  /** Undirected neighbours, and the other variables of adjacent factors. */
  neighbours: string[]
  /** All of the above, without repeats. */
  blanket: string[]
}

/**
 * The Markov blanket of node `name` in an explicit graph: parents, children and co-parents along directed edges,
 * neighbours along undirected edges, and the variables sharing a factor with it. Given its blanket, the node is
 * independent of every other variable (Pearl 1988). Parents that are parameters are included; factors are not. A
 * compact graph is unrolled against `sizes` first (every named size must be bound). Throws `DomainError` when no
 * node has the name.
 *
 * @param graph The graph, compact or unrolled.
 * @param name The node's name in the explicit graph (a copy's name, such as `z[1]`, for a compact one).
 * @param sizes The values of the named sizes of a compact graph.
 * @returns The blanket by kind and in all, as node names.
 *
 * @example A node of the sprinkler network
 * const sprinkler = structuredGraph({
 *   nodes: ['cloudy', 'sprinkler', 'rain', 'wet'].map((name) => ({ name, role: 'latent', group: null })),
 *   edges: [
 *     { from: 'cloudy', to: 'sprinkler' }, { from: 'cloudy', to: 'rain' },
 *     { from: 'sprinkler', to: 'wet' }, { from: 'rain', to: 'wet' },
 *   ],
 * })
 * print(markovBlanket(sprinkler, 'sprinkler'))
 *
 * @example A latent step of an HMM: its neighbours in time and its observation
 * print(markovBlanket(chainTemplate('T', { observed: 'x' }), 'z[1]', { T: 3 }).blanket)
 */
export function markovBlanket(graph: StructuredGraph, name: string, sizes: SizeBindings = {}): Blanket {
  const g = graph.groups.length && !graph.unrolled ? unroll(graph, sizes) : graph
  const v = g.attributes.findIndex((n) => n.name === name)
  if (v < 0) throw new DomainError('markovBlanket', `markovBlanket: no node named ${name}`)
  const names = (xs: Iterable<Index>) => [...new Set(xs)].filter((i) => i !== v).map((i) => g.attributes[i].name)
  const role = (i: Index) => g.attributes[i].role
  const parentsOf = (c: Index) =>
    g.edges.filter((e) => e.directed && e.to === c && role(e.from) !== 'factor').map((e) => e.from)
  const parents = parentsOf(v)
  const children = g.edges.filter((e) => e.directed && e.from === v).map((e) => e.to)
  const coParents = children.flatMap(parentsOf)
  const neighbours: Index[] = []
  for (const e of g.edges) {
    if (e.directed || (e.from !== v && e.to !== v)) continue
    const u = e.from === v ? e.to : e.from
    if (role(u) !== 'factor') neighbours.push(u)
    else
      for (const f of g.edges)
        if (!f.directed && (f.from === u || f.to === u)) neighbours.push(f.from === u ? f.to : f.from)
  }
  const out = {
    parents: names(parents),
    children: names(children),
    coParents: names(coParents),
    neighbours: names(neighbours),
  }
  return { ...out, blanket: [...new Set([...out.parents, ...out.children, ...out.coParents, ...out.neighbours])] }
}

/**
 * The number of copies each group of a compact graph makes at the given sizes (for display and cost estimates): a
 * nested group counts its copies across every copy of the groups outside it. A group with no node inside it counts 0.
 *
 * @param graph The compact graph.
 * @param sizes The values of its named sizes (every one must be bound).
 * @returns The count per group name.
 *
 * @example Three words in each of two documents
 * const lda = structured('nested plates', (b) => {
 *   const docs = b.plate('docs', 2)
 *   docs.plate('words', 3).observed('w')
 * })
 * print(groupCounts(lda))
 * print(groupCounts(latticeTemplate('R', 'C'), { R: 2, C: 3 }))
 */
export function groupCounts(graph: StructuredGraph, sizes: SizeBindings = {}): Record<string, number> {
  const g = unroll(graph, sizes)
  const out: Record<string, number> = {}
  for (const grp of graph.groups) {
    const members = g.attributes.filter((n) => groupChain(graph, n.group).some((q) => q.name === grp.name))
    const axes = groupChain(graph, grp.name).reduce((a, q) => a + (q.kind === 'lattice' ? 2 : 1), 0)
    out[grp.name] = new Set(members.map((n) => (n.index ?? []).slice(0, axes).join(','))).size
  }
  return out
}
