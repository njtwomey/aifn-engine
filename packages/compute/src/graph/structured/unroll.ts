/**
 * Unrolling: a compact structured graph expanded into one node per copy (Buntine 1994 for plates; Koller & Friedman
 * 2009, §6.2 for templates). Each plate or template multiplies the nodes inside it; an edge joins every pair of copies
 * that agree on the groups both ends share, and a lagged edge joins neighbouring copies of a template.
 */

import type { Index, Size } from 'aifn-compute/foundation/contracts'
import { AifnError } from 'aifn-compute/foundation/errors'
import { groupChain, groupSizes, structuredGraph, type EdgeSpec } from './build'
import type { Group, StructuredGraph, StructuredNode } from './types'

/**
 * Values of named sizes: a number, or for a group nested in another, one size per index of the enclosing group's
 * last axis (a ragged size). Or a function giving the size of a group's axis at the index of its enclosing copies,
 * called for every axis whose size is a name.
 */
export type SizeBindings =
  Readonly<Record<string, Size | readonly Size[]>> | ((group: Group, axis: Index, outer: readonly Index[]) => Size)

/**
 * Throw the module's error, an `AifnError` from `unroll`.
 *
 * @param message What is wrong, without the function's name (it is prefixed).
 * @returns Never: it always throws.
 */
const fail = (message: string): never => {
  throw new AifnError('unroll', `unroll: ${message}`)
}

/**
 * The size of one axis of a group at the index of its enclosing copies. Throws `AifnError` when the size is a name
 * that `sizes` does not bind, or a ragged size with no entry for the enclosing index.
 *
 * @param group The group.
 * @param axis Which of the group's sizes: 0, or 1 for a lattice's columns.
 * @param outer The index of the enclosing copies, one entry per axis of the groups outside `group`, outermost first.
 *   A ragged size is read at its last entry (0 when it is empty).
 * @param sizes The bindings of the named sizes.
 * @returns The number of copies along the axis (for a tree, its depth).
 */
function axisSize(group: Group, axis: Index, outer: readonly Index[], sizes: SizeBindings): Size {
  const spec = groupSizes(group)[axis]
  if (typeof spec === 'number') return spec
  if (typeof sizes === 'function') return sizes(group, axis, outer)
  const s = sizes[spec] ?? fail(`size ${spec} of group ${group.name} is not bound`)
  if (typeof s === 'number') return s
  return s[outer[outer.length - 1] ?? 0] ?? fail(`size ${spec} has no entry for index ${outer.join(',')}`)
}

/**
 * The number of nodes of a complete $k$-ary tree of depth $d$ (the root alone at depth 0):
 * $(k^{d+1} - 1)/(k - 1)$, or $d + 1$ when $k = 1$.
 *
 * @param arity The number of children $k$ of each inner node.
 * @param depth The depth $d$.
 * @returns The node count.
 */
export const treeSize = (arity: Size, depth: Size): Size =>
  arity === 1 ? depth + 1 : (Math.pow(arity, depth + 1) - 1) / (arity - 1)

/**
 * The number of index axes a group adds (a lattice two, every other kind one).
 *
 * @param g The group.
 * @returns 2 for a lattice, else 1.
 */
const axesOf = (g: Group): Size => (g.kind === 'lattice' ? 2 : 1)

/**
 * The copies of a group at the index of its enclosing copies: its axis sizes (a tree as one axis of all its nodes).
 *
 * @param group The group.
 * @param outer The index of the enclosing copies, outermost first.
 * @param sizes The bindings of the named sizes.
 * @returns One count per axis: two for a lattice (rows, cols), one otherwise.
 */
function extents(group: Group, outer: readonly Index[], sizes: SizeBindings): Size[] {
  if (group.kind === 'lattice') return [axisSize(group, 0, outer, sizes), axisSize(group, 1, outer, sizes)]
  const n = axisSize(group, 0, outer, sizes)
  return [group.kind === 'tree' ? treeSize(group.arity!, n) : n]
}

/**
 * Every index of a node inside the given groups, in row-major order.
 *
 * @param chain The groups holding the node, outermost first (as `groupChain` returns them).
 * @param sizes The bindings of the named sizes.
 * @returns One index per copy, each with one entry per axis of the groups; a single empty index for no groups.
 */
function copies(chain: readonly Group[], sizes: SizeBindings): Index[][] {
  const out: Index[][] = []
  const visit = (depth: number, index: Index[]) => {
    if (depth === chain.length) return void out.push(index)
    const ext = extents(chain[depth], index, sizes)
    if (ext.length === 1) for (let i = 0; i < ext[0]; i++) visit(depth + 1, [...index, i])
    else for (let i = 0; i < ext[0]; i++) for (let j = 0; j < ext[1]; j++) visit(depth + 1, [...index, i, j])
  }
  visit(0, [])
  return out
}

/**
 * The name of a copy: `z[2,5]`, or the compact name for a node in no group.
 *
 * @param name The compact node's name.
 * @param index The copy's index along each axis of its groups, outermost first.
 * @returns The name `unroll` gives the copy.
 *
 * @example Copy names
 * print(copyName('z', [2, 5]))
 * print(copyName('mu', []))
 */
export const copyName = (name: string, index: readonly Index[]): string =>
  index.length ? `${name}[${index.join(',')}]` : name

/**
 * The TeX label of a copy: the compact label with the index as a subscript, `z_{2,5}`.
 *
 * @param label The compact node's TeX label.
 * @param index The copy's index along each axis of its groups, outermost first; empty leaves the label as it is.
 * @returns The label `unroll` gives the copy.
 */
const copyLabel = (label: string, index: readonly Index[]): string =>
  index.length ? `${label}_{${index.join(',')}}` : label

/**
 * Expand a structured graph into one node per copy. Nodes come in the compact graph's order, each node's copies in
 * row-major index order, so a topological order of the compact graph (ignoring lags) stays one. A copy is named
 * `z[t]` (`z[i,j]` on a lattice, `θ[d]` in a plate), labelled with the index as a TeX subscript, carries `source` and
 * `index`, and keeps its group so a diagram can draw the plate around the copies. An edge without a lag joins every
 * pair of copies whose indices agree on the groups both ends share; a lagged edge joins copy $t - k$ to $t$ on a chain
 * (mod $T$ when periodic), site $(i - d_i, j - d_j)$ to $(i, j)$ on a lattice (wrapped on a torus), and a tree node's
 * parent to it, skipping a copy that would link to itself. An unrolled graph, or one with no groups, is returned as it
 * is. Throws `AifnError` when a named size is not bound.
 *
 * @param graph The compact graph.
 * @param sizes The values of its named sizes (default none: every size must then be a number).
 * @returns The explicit graph, with `unrolled: true` and no named sizes.
 *
 * @example A ragged plate: two documents of 2 and 1 words
 * const lda = structured('ragged plates', (b) => {
 *   const docs = b.plate('docs', 2)
 *   const theta = docs.latent('theta')
 *   b.edge(theta, docs.plate('words', 'N').observed('w'))
 * })
 * const explicit = unroll(lda, { N: [2, 1] })
 * const name = (i) => explicit.attributes[i].name
 * print('nodes:', explicit.attributes.map((n) => n.name))
 * print('edges:', explicit.edges.map((e) => `${name(e.from)} -> ${name(e.to)}`))
 *
 * @example A periodic chain closes into a ring
 * const ring = unroll(chainTemplate(3, { periodic: true }))
 * const name = (i) => ring.attributes[i].name
 * print('edges:', ring.edges.map((e) => `${name(e.from)} -> ${name(e.to)}`))
 *
 * @example An unbound size throws
 * try {
 *   unroll(chainTemplate('T'))
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function unroll<D>(graph: StructuredGraph<D>, sizes: SizeBindings = {}): StructuredGraph<D> {
  if (graph.unrolled || graph.groups.length === 0) return graph
  const chains = graph.attributes.map((n) => groupChain(graph, n.group))
  const perNode = chains.map((chain) => copies(chain, sizes))
  const nodes: StructuredNode<D>[] = []
  graph.attributes.forEach((n, i) =>
    perNode[i].forEach((index) =>
      nodes.push({
        ...n,
        name: copyName(n.name, index),
        label: copyLabel(n.label ?? n.name, index),
        source: n.name,
        index,
      }),
    ),
  )
  const edges: EdgeSpec[] = []
  graph.edges.forEach((e) => {
    const [cu, cv] = [chains[e.from], chains[e.to]]
    let shared = 0
    while (shared < cu.length && shared < cv.length && cu[shared].name === cv[shared].name) shared++
    const prefix = cu.slice(0, shared).reduce((a, g) => a + axesOf(g), 0)
    const base = { directed: e.directed, ...(e.weight === undefined ? {} : { weight: e.weight }) }
    const [nu, nv] = [graph.attributes[e.from].name, graph.attributes[e.to].name]
    if (e.lag === undefined) {
      const byPrefix = new Map<string, Index[][]>()
      for (const iu of perNode[e.from]) {
        const key = iu.slice(0, prefix).join(',')
        byPrefix.set(key, [...(byPrefix.get(key) ?? []), iu])
      }
      for (const iv of perNode[e.to])
        for (const iu of byPrefix.get(iv.slice(0, prefix).join(',')) ?? [])
          edges.push({ from: copyName(nu, iu), to: copyName(nv, iv), ...base })
      return
    }
    // A lag couples copies along the innermost group, which both ends share.
    const group = cv[cv.length - 1]
    const at = prefix - axesOf(group)
    const known = new Set(perNode[e.from].map((iu) => iu.join(',')))
    for (const iv of perNode[e.to]) {
      const outer = iv.slice(0, at)
      const own = iv.slice(at)
      const ext = extents(group, outer, sizes)
      let from: Index[] | null = null
      if (group.kind === 'tree') from = own[0] >= 1 ? [Math.floor((own[0] - 1) / group.arity!)] : null
      else {
        const lag = typeof e.lag === 'number' ? [e.lag] : (e.lag as readonly number[])
        from = own.map((x, a) => x - lag[a])
        if (group.periodic) from = from.map((x, a) => ((x % ext[a]) + ext[a]) % ext[a])
        if (from.some((x, a) => x < 0 || x >= ext[a])) from = null
      }
      if (from === null) continue
      const iu = [...outer, ...from]
      if (!known.has(iu.join(','))) continue
      // A periodic chain of length 1 or 2 would link a copy to itself or twice; skip self-links.
      if (e.from === e.to && iu.join(',') === iv.join(',')) continue
      edges.push({ from: copyName(nu, iu), to: copyName(nv, iv), ...base })
    }
  })
  return {
    ...structuredGraph<D>({
      ...(graph.name === undefined ? {} : { name: graph.name }),
      nodes,
      edges,
      groups: graph.groups,
    }),
    sizes: [],
    unrolled: true,
  }
}

/**
 * The indices of the copies of compact node `name` in an unrolled graph (the node itself in a compact graph). Empty
 * when there is no such node.
 *
 * @param graph The graph, unrolled or compact.
 * @param name The compact node's name.
 * @returns Ascending node indices.
 *
 * @example The copies of the observation in an unrolled HMM
 * const hmm = unroll(chainTemplate('T', { observed: 'x' }), { T: 3 })
 * print('copies of x:', copiesOf(hmm, 'x'))
 * print('names:', copiesOf(hmm, 'x').map((i) => hmm.attributes[i].name))
 */
export function copiesOf(graph: StructuredGraph, name: string): Index[] {
  return graph.attributes.flatMap((n, i) => ((n.source ?? n.name) === name ? [i] : []))
}
