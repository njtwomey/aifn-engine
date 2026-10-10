/**
 * Building structured graphs: from a plain specification (`structuredGraph`), with a builder (`structured`), and
 * from templates: a chain (a hidden Markov model or a linear-chain CRF), a lattice (an Ising or Potts model), a tree,
 * and repeated slices (a dynamic Bayesian network; Koller & Friedman 2009, §6.2; Murphy 2002, "Dynamic Bayesian
 * Networks", PhD thesis, UC Berkeley).
 *
 * Every constructor ends in `structuredGraph`, which checks the specification and throws `AifnError` on the first
 * problem it finds; nodes and groups are referred to by name, and node indices follow declaration order.
 */

import type { Index, Size } from 'aifn-compute/foundation/contracts'
import { AifnError } from 'aifn-compute/foundation/errors'
import type {
  Group,
  GroupKind,
  Lag,
  NodeRole,
  SizeSpec,
  StructuredEdge,
  StructuredGraph,
  StructuredNode,
} from './types'

/** An edge of a specification, between node names. */
export interface EdgeSpec extends Partial<StructuredEdge> {
  /** The name of the node the edge leaves. */
  from: string
  /** The name of the node the edge enters. */
  to: string
  /** The edge's weight, carried to the graph's edge (none when left out). */
  weight?: number
}

/** A plain specification of a structured graph: nodes, edges by name, and groups. */
export interface StructuredSpec<D = unknown> {
  /** The model's name. */
  name?: string
  /** The nodes, in the order that fixes their indices. */
  nodes: readonly StructuredNode<D>[]
  /** The edges, between node names (default none). */
  edges?: readonly EdgeSpec[]
  /** The plates and templates the nodes sit in (default none). */
  groups?: readonly Group[]
  /** Named sizes; those used by groups are added automatically. */
  sizes?: readonly string[]
}

/**
 * Throw the module's error, an `AifnError` from `structuredGraph`.
 *
 * @param message What is wrong, without the function's name (it is prefixed).
 * @returns Never: it always throws.
 */
const fail = (message: string): never => {
  throw new AifnError('structuredGraph', `structuredGraph: ${message}`)
}

/**
 * The sizes of a group as a list, one per axis: two for a lattice (rows, cols), one for a plate or a chain, and one
 * for a tree (its depth).
 *
 * @param g The group.
 * @returns Its sizes, each a number or a named size.
 *
 * @example A lattice has two sizes, a chain one
 * print('lattice:', groupSizes(latticeTemplate(2, 3).groups[0]))
 * print('chain:', groupSizes(chainTemplate('T').groups[0]))
 */
export const groupSizes = (g: Group): readonly SizeSpec[] => (Array.isArray(g.size) ? g.size : [g.size as SizeSpec])

/**
 * The groups holding `group`, outermost first, ending with it (empty for null). Throws `AifnError` when a name on the
 * way is not a group of the graph, or when the parents lead back to a group already on the way (a nesting cycle).
 *
 * @param graph The graph whose groups are searched (only `groups` is read).
 * @param group The name of the innermost group, or null for a node outside every group.
 * @returns The chain of groups from the outermost to `group`.
 *
 * @example Words nested in documents
 * const lda = structured('nested plates', (b) => {
 *   const docs = b.plate('docs', 2)
 *   docs.plate('words', 3).observed('w')
 * })
 * print('words:', groupChain(lda, 'words').map((g) => g.name))
 * print('none:', groupChain(lda, null))
 */
export function groupChain(graph: Pick<StructuredGraph, 'groups'>, group: string | null): Group[] {
  const chain: Group[] = []
  for (let name = group; name !== null;) {
    const g = graph.groups.find((q) => q.name === name)
    if (!g) return fail(`unknown group ${name}`)
    if (chain.includes(g)) return fail(`groups nest in a cycle at ${g.name}`)
    chain.unshift(g)
    name = g.parent
  }
  return chain
}

/**
 * The index of the node named `name`; throws `AifnError` when there is none.
 *
 * @param graph The graph whose nodes are searched (only `attributes` is read).
 * @param name The node's name (a copy's name, such as `z[2]`, in an unrolled graph).
 * @returns The node's index in `graph.attributes`.
 *
 * @example Nodes are indexed in declaration order
 * const hmm = chainTemplate('T', { observed: 'x' })
 * print('z:', nodeIndex(hmm, 'z'), 'x:', nodeIndex(hmm, 'x'))
 */
export function nodeIndex(graph: Pick<StructuredGraph, 'attributes'>, name: string): Index {
  const i = graph.attributes.findIndex((n) => n.name === name)
  if (i < 0) return fail(`no node named ${name}`)
  return i
}

/**
 * The indices of the nodes with one of the given roles (every node without `roles`), in node order.
 *
 * @param graph The graph whose nodes are searched (only `attributes` is read).
 * @param roles The roles to keep; none keeps every node.
 * @returns Ascending node indices.
 *
 * @example The latent and observed nodes of an unrolled chain
 * const hmm = unroll(chainTemplate('T', { observed: 'x' }), { T: 2 })
 * print('names:', hmm.attributes.map((n) => n.name))
 * print('latent:', nodesWithRole(hmm, 'latent'))
 * print('observed:', nodesWithRole(hmm, 'observed'))
 */
export function nodesWithRole(graph: Pick<StructuredGraph, 'attributes'>, ...roles: NodeRole[]): Index[] {
  return graph.attributes.flatMap((n, i) => (roles.length === 0 || roles.includes(n.role) ? [i] : []))
}

/**
 * Throws `AifnError` unless `lag` fits the kind of `group`: a positive integer on a chain, a pair of integers (not
 * both 0) on a lattice, `'parent'` on a tree, and nothing on a plate.
 *
 * @param lag The lag of an edge.
 * @param group The template group both ends of the edge sit in.
 * @param where The edge, as named in the error message.
 */
function checkLag(lag: Lag, group: Group, where: string): void {
  const kinds: Record<GroupKind, (l: Lag) => boolean> = {
    plate: () => false,
    chain: (l) => typeof l === 'number' && Number.isInteger(l) && l >= 1,
    lattice: (l) => Array.isArray(l) && l.length === 2 && l.every((x) => Number.isInteger(x)) && (l[0] || l[1]) !== 0,
    tree: (l) => l === 'parent',
  }
  if (!kinds[group.kind](lag)) fail(`${where}: lag ${JSON.stringify(lag)} does not fit a ${group.kind} group`)
}

/**
 * A structured graph from a specification. Checks that node and group names are unique, each group has one index
 * symbol per size (a tree, an `arity` of at least 1), parents and groups named exist, edge ends exist, every lagged
 * edge joins two nodes of one group with a lag of that group's kind, no edge without a lag joins a node to itself, and
 * group nesting forms no cycle; the first failure throws `AifnError`. Edges touching a factor default to undirected,
 * others to directed; the graph is `directed` when every edge is. The named sizes are those of the specification plus
 * any a group uses.
 *
 * @param spec The nodes, the edges between their names, the groups and the named sizes.
 * @returns The structured graph, compact (its groups not expanded).
 *
 * @example A latent cause, an observation and a factor
 * const g = structuredGraph({
 *   nodes: [
 *     { name: 'a', role: 'latent', group: null },
 *     { name: 'b', role: 'observed', group: null },
 *     { name: 'f', role: 'factor', group: null },
 *   ],
 *   edges: [{ from: 'a', to: 'b' }, { from: 'f', to: 'a' }],
 * })
 * print('labels:', g.labels)
 * print('directed per edge:', g.edges.map((e) => e.directed))
 *
 * @example An edge to an unknown node throws
 * try {
 *   structuredGraph({ nodes: [{ name: 'a', role: 'latent', group: null }], edges: [{ from: 'a', to: 'c' }] })
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function structuredGraph<D = unknown>(spec: StructuredSpec<D>): StructuredGraph<D> {
  const groups = [...(spec.groups ?? [])]
  const groupNames = new Set<string>()
  for (const g of groups) {
    if (groupNames.has(g.name)) fail(`duplicate group ${g.name}`)
    groupNames.add(g.name)
    if (g.index.length !== groupSizes(g).length && g.kind !== 'tree')
      fail(`group ${g.name} has ${g.index.length} index symbols for ${groupSizes(g).length} sizes`)
    if (g.kind === 'tree' && !(Number.isInteger(g.arity) && g.arity! >= 1)) fail(`tree group ${g.name} needs an arity`)
  }
  for (const g of groups) {
    if (g.parent !== null && !groupNames.has(g.parent)) fail(`group ${g.name} has unknown parent ${g.parent}`)
    groupChain({ groups }, g.name)
  }
  const attributes = spec.nodes.map((n) => ({ ...n }))
  const index = new Map<string, number>()
  attributes.forEach((n, i) => {
    if (index.has(n.name)) fail(`duplicate node ${n.name}`)
    if (n.group !== null && !groupNames.has(n.group)) fail(`node ${n.name} is in unknown group ${n.group}`)
    index.set(n.name, i)
  })
  const edges = (spec.edges ?? []).map((e, k) => {
    const from = index.get(e.from) ?? fail(`edge ${k} starts at unknown node ${e.from}`)
    const to = index.get(e.to) ?? fail(`edge ${k} ends at unknown node ${e.to}`)
    const touchesFactor = attributes[from].role === 'factor' || attributes[to].role === 'factor'
    const directed = e.directed ?? !touchesFactor
    if (e.lag !== undefined) {
      const ga = attributes[from].group
      if (ga === null || ga !== attributes[to].group)
        fail(`edge ${e.from} → ${e.to}: a lag needs both ends in one group`)
      checkLag(
        e.lag,
        groups.find((g) => g.name === ga)!,
        `edge ${e.from} → ${e.to}`,
      )
    } else if (from === to) fail(`edge ${e.from} → ${e.to}: a self-edge needs a lag`)
    return {
      from,
      to,
      directed,
      ...(e.weight === undefined ? {} : { weight: e.weight }),
      ...(e.lag === undefined ? {} : { lag: e.lag }),
      ...(e.label === undefined ? {} : { label: e.label }),
    }
  })
  const sizes = [...(spec.sizes ?? [])]
  for (const g of groups) for (const s of groupSizes(g)) if (typeof s === 'string' && !sizes.includes(s)) sizes.push(s)
  return {
    kind: 'graph',
    nodes: attributes.length,
    edges,
    directed: edges.every((e) => e.directed),
    labels: attributes.map((n) => n.label ?? n.name),
    attributes,
    groups,
    sizes,
    ...(spec.name === undefined ? {} : { name: spec.name }),
  }
}

// ── Builder ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of a node declared with the builder. */
export interface NodeSpec<D = unknown> {
  /** TeX for display (default: the name). */
  label?: string
  /** The caller's payload: a distribution, a table. */
  data?: D
}

/** Options of a group declared with the builder. */
export interface GroupSpec {
  /** Index symbol(s); default `n` (plate), `t` (chain), `i`, `j` (lattice), `v` (tree). */
  index?: string | readonly string[]
  /** TeX for the group's label (default: its size). */
  label?: string
  /** A chain that wraps into a cycle, or a lattice that wraps into a torus. */
  periodic?: boolean
}

/** A place to declare nodes: the graph itself or a group. Each declaration returns the node's name. */
export interface StructuredScope<D = unknown> {
  /** A latent variable in this scope. */
  latent(name: string, options?: NodeSpec<D>): string
  /** An observed variable in this scope. */
  observed(name: string, options?: NodeSpec<D>): string
  /** A factor in this scope. */
  factor(name: string, options?: NodeSpec<D>): string
  /** A deterministic function of its parents in this scope. */
  deterministic(name: string, options?: NodeSpec<D>): string
  /** A parameter in this scope. */
  parameter(name: string, options?: NodeSpec<D>): string
  /** A plate nested here: exchangeable copies. */
  plate(name: string, size: SizeSpec, options?: GroupSpec): StructuredScope<D>
  /** A chain template nested here: copies $0, \dots, \text{length} - 1$ in order. */
  chain(name: string, length: SizeSpec, options?: GroupSpec): StructuredScope<D>
  /** A lattice template nested here: $\text{rows} \times \text{cols}$ sites (4 neighbours unless given 8). */
  lattice(
    name: string,
    rows: SizeSpec,
    cols: SizeSpec,
    options?: GroupSpec & { neighbourhood?: 4 | 8 },
  ): StructuredScope<D>
  /** A tree template nested here: a complete `arity`-ary tree of the given depth. */
  tree(name: string, arity: Size, depth: SizeSpec, options?: GroupSpec): StructuredScope<D>
}

/** The builder passed to {@link structured}: a scope, plus named sizes and edges. */
export interface StructuredBuilder<D = unknown> extends StructuredScope<D> {
  /** Declare a named size (bound by `unroll`); returns the name. */
  size(name: string): string
  /** An edge between declared nodes (see `structuredGraph` for the default direction). */
  edge(from: string, to: string, options?: Partial<StructuredEdge> & { weight?: number }): void
}

/**
 * Describe a structured graph with a builder: `build` declares nodes in the graph or in nested plates and templates,
 * named sizes and edges, and the result is checked by `structuredGraph` (which throws `AifnError` on a bad
 * declaration). Groups take default index symbols (`n` for a plate, `t` for a chain, `i`, `j` for a lattice, `v` for a
 * tree), and a lattice 4 neighbours.
 *
 * @param name The model's name, kept as the graph's `name`.
 * @param build Called once with the builder; what it declares, in order, becomes the graph.
 * @returns The compact structured graph.
 *
 * @example A hidden Markov model
 * const hmm = structured('hidden Markov model', (b) => {
 *   const time = b.chain('time', b.size('T'))
 *   const z = time.latent('z')
 *   const x = time.observed('x')
 *   b.edge(z, z, { lag: 1 })
 *   b.edge(z, x)
 * })
 * print('nodes:', hmm.attributes.map((n) => `${n.name} (${n.role})`))
 * print('groups:', hmm.groups.map((g) => `${g.name}: ${g.kind} of length ${g.size}`))
 * print('shape:', shape(hmm))
 */
export function structured<D = unknown>(name: string, build: (b: StructuredBuilder<D>) => void): StructuredGraph<D> {
  const nodes: StructuredNode<D>[] = []
  const edges: EdgeSpec[] = []
  const groups: Group[] = []
  const sizes: string[] = []
  const scope = (group: string | null): StructuredScope<D> => {
    const node =
      (role: NodeRole) =>
      (n: string, o: NodeSpec<D> = {}): string => {
        nodes.push({ name: n, role, group, ...o })
        return n
      }
    const nest = (g: Omit<Group, 'parent'>): StructuredScope<D> => {
      groups.push({ ...g, parent: group })
      return scope(g.name)
    }
    const symbols = (o: GroupSpec, fallback: readonly string[]) =>
      o.index === undefined ? fallback : typeof o.index === 'string' ? [o.index] : o.index
    const extra = (o: GroupSpec) => ({
      ...(o.label === undefined ? {} : { label: o.label }),
      ...(o.periodic === undefined ? {} : { periodic: o.periodic }),
    })
    return {
      latent: node('latent'),
      observed: node('observed'),
      factor: node('factor'),
      deterministic: node('deterministic'),
      parameter: node('parameter'),
      plate: (n, size, o = {}) => nest({ name: n, kind: 'plate', size, index: symbols(o, ['n']), ...extra(o) }),
      chain: (n, length, o = {}) =>
        nest({ name: n, kind: 'chain', size: length, index: symbols(o, ['t']), ...extra(o) }),
      lattice: (n, rows, cols, o = {}) =>
        nest({
          name: n,
          kind: 'lattice',
          size: [rows, cols],
          index: symbols(o, ['i', 'j']),
          neighbourhood: o.neighbourhood ?? 4,
          ...extra(o),
        }),
      tree: (n, arity, depth, o = {}) =>
        nest({ name: n, kind: 'tree', size: depth, index: symbols(o, ['v']), arity, ...extra(o) }),
    }
  }
  build({
    ...scope(null),
    size: (n) => (sizes.includes(n) || sizes.push(n), n),
    edge: (from, to, o = {}) => void edges.push({ from, to, ...o }),
  })
  return structuredGraph({ name, nodes, edges, groups, sizes })
}

// ── Templates ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the templates: node names and labels, and whether each copy has an observed child. */
export interface TemplateOptions {
  /** The latent node's name (default `z`; `x` for a lattice). */
  name?: string
  /** Add an observed child per copy, with this name (default none). */
  observed?: string
  /** Directed (a Bayesian network, default for chains and trees) or undirected (an MRF or a CRF). */
  directed?: boolean
  /** The template group's name (default `chain`, `lattice`, `tree`). */
  group?: string
  /** Chain: a cycle; lattice: a torus. Not used by `treeTemplate`. */
  periodic?: boolean
}

/**
 * Add an observed child of `z` to `scope` when the options name one, joined by an edge directed unless
 * `o.directed` is false.
 *
 * @param b The builder, which receives the edge.
 * @param scope The template group to declare the observed node in.
 * @param z The name of the template's latent node.
 * @param o The template's options: `observed` names the child (none when left out), `directed` the edge's direction.
 */
function withObservation(b: StructuredBuilder, scope: StructuredScope, z: string, o: TemplateOptions): void {
  if (o.observed === undefined) return
  const x = scope.observed(o.observed)
  b.edge(z, x, { directed: o.directed ?? true })
}

/**
 * A chain template of length $T$: latent $z_t$ with $z_{t-1} \to z_t$ (lag 1), and an observed child $x_t$ with
 * `observed: 'x'` (the hidden Markov model's structure; `directed: false` gives a linear-chain CRF's).
 *
 * @param length The chain's length $T$: a number, or a name bound by `unroll` (added to the graph's sizes).
 * @param options The latent node's name (default `z`), the observed child's name, the direction of the edges
 *   (default directed), the group's name (default `chain`) and `periodic`.
 * @returns The compact graph, named `chain`.
 *
 * @example A hidden Markov model, compact and unrolled
 * const hmm = chainTemplate('T', { observed: 'x' })
 * print('compact:', hmm.attributes.map((n) => n.name), 'sizes:', hmm.sizes)
 * const explicit = unroll(hmm, { T: 3 })
 * print('unrolled:', explicit.attributes.map((n) => n.name))
 * const name = (i) => explicit.attributes[i].name
 * print('edges:', explicit.edges.map((e) => `${name(e.from)} -> ${name(e.to)}`))
 */
export function chainTemplate(length: SizeSpec, options: TemplateOptions = {}): StructuredGraph {
  return structured('chain', (b) => {
    if (typeof length === 'string') b.size(length)
    const time = b.chain(options.group ?? 'chain', length, { periodic: options.periodic })
    const z = time.latent(options.name ?? 'z')
    b.edge(z, z, { lag: 1, directed: options.directed ?? true })
    withObservation(b, time, z, options)
  })
}

/**
 * A lattice template of $\text{rows} \times \text{cols}$ sites: latent $x_{ij}$ linked to its right and lower
 * neighbours (4 neighbours) and the two lower diagonals too (8), undirected by default (an Ising or Potts model); a
 * torus with `periodic`.
 *
 * @param rows The number of rows: a number, or a name bound by `unroll`.
 * @param cols The number of columns: a number, or a name bound by `unroll`.
 * @param options The latent node's name (default `x`), the observed child's name, the direction of the edges
 *   (default undirected), the group's name (default `lattice`), `periodic`, and `neighbourhood` (4, the default, or 8).
 * @returns The compact graph, named `lattice`.
 *
 * @example A 2 by 2 Ising lattice
 * const ising = unroll(latticeTemplate(2, 2))
 * print('sites:', ising.attributes.map((n) => n.name))
 * print('links:', ising.edges.map((e) => `${ising.attributes[e.from].name} - ${ising.attributes[e.to].name}`))
 */
export function latticeTemplate(
  rows: SizeSpec,
  cols: SizeSpec,
  options: TemplateOptions & { neighbourhood?: 4 | 8 } = {},
): StructuredGraph {
  const neighbourhood = options.neighbourhood ?? 4
  return structured('lattice', (b) => {
    for (const s of [rows, cols]) if (typeof s === 'string') b.size(s)
    const grid = b.lattice(options.group ?? 'lattice', rows, cols, { neighbourhood, periodic: options.periodic })
    const x = grid.latent(options.name ?? 'x')
    const lags: [number, number][] = [
      [0, 1],
      [1, 0],
    ]
    if (neighbourhood === 8) lags.push([1, 1], [1, -1])
    for (const lag of lags) b.edge(x, x, { lag, directed: options.directed ?? false })
    withObservation(b, grid, x, { ...options, directed: options.directed ?? false })
  })
}

/**
 * A tree template: latent $z_v$ at each node of a complete `arity`-ary tree of the given depth, with edges from parent
 * to child (directed by default).
 *
 * @param arity The number of children of each inner node.
 * @param depth The tree's depth (0 for the root alone): a number, or a name bound by `unroll`.
 * @param options The latent node's name (default `z`), the observed child's name, the direction of the edges
 *   (default directed) and the group's name (default `tree`).
 * @returns The compact graph, named `tree`.
 *
 * @example A binary tree of depth 2 has 7 nodes
 * const tree = unroll(treeTemplate(2, 2))
 * print('nodes:', tree.attributes.map((n) => n.name))
 * print('edges:', tree.edges.map((e) => `${tree.attributes[e.from].name} -> ${tree.attributes[e.to].name}`))
 */
export function treeTemplate(arity: Size, depth: SizeSpec, options: TemplateOptions = {}): StructuredGraph {
  return structured('tree', (b) => {
    if (typeof depth === 'string') b.size(depth)
    const tree = b.tree(options.group ?? 'tree', arity, depth)
    const z = tree.latent(options.name ?? 'z')
    b.edge(z, z, { lag: 'parent', directed: options.directed ?? true })
    withObservation(b, tree, z, options)
  })
}

/**
 * Repeated slices (a dynamic Bayesian network, or a factorial HMM): every node and group of `slice` inside a new chain
 * group of the given length, plus a directed lag-1 edge for each `[from, to]` in `transitions` (from slice $t - 1$ to
 * slice $t$). The slice's own nodes must sit at its top level (not inside its groups) to be linked across slices.
 * Throws `AifnError` when the slice already has a group of the new group's name.
 *
 * @param slice One time slice: a compact structured graph, whose edges are kept inside every slice.
 * @param length The number of slices: a number, or a name bound by `unroll`.
 * @param transitions Pairs of node names `[from, to]`, each an edge from `from` in one slice to `to` in the next.
 * @param options The new chain group's name (default `slices`) and index symbol (default `t`).
 * @returns The compact graph, with the slice's name.
 *
 * @example A factorial HMM: two latent chains, one observation per step
 * const slice = structured('factorial HMM', (b) => {
 *   b.edge(b.latent('a'), b.observed('y'))
 *   b.edge(b.latent('b'), 'y')
 * })
 * const fhmm = repeatedSlices(slice, 3, [['a', 'a'], ['b', 'b']])
 * const explicit = unroll(fhmm)
 * print('nodes:', explicit.attributes.map((n) => n.name))
 * print('shape:', shape(fhmm))
 */
export function repeatedSlices<D>(
  slice: StructuredGraph<D>,
  length: SizeSpec,
  transitions: readonly (readonly [string, string])[],
  options: { group?: string; index?: string } = {},
): StructuredGraph<D> {
  const group = options.group ?? 'slices'
  if (slice.groups.some((g) => g.name === group)) throw new AifnError('repeated', `repeated: group ${group} exists`)
  const name = (i: number) => slice.attributes[i].name
  return structuredGraph<D>({
    ...(slice.name === undefined ? {} : { name: slice.name }),
    nodes: slice.attributes.map((n) => ({ ...n, group: n.group ?? group })),
    groups: [
      { name: group, kind: 'chain', size: length, index: [options.index ?? 't'], parent: null },
      ...slice.groups.map((g) => ({ ...g, parent: g.parent ?? group })),
    ],
    edges: [
      ...slice.edges.map((e) => ({ ...e, from: name(e.from), to: name(e.to) })),
      ...transitions.map(([from, to]) => ({ from, to, lag: 1, directed: true })),
    ],
    sizes: [...slice.sizes, ...(typeof length === 'string' && !slice.sizes.includes(length) ? [length] : [])],
  })
}
