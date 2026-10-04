/**
 * Building structured graphs: from a plain specification (`structuredGraph`), with a builder (`structured`), and
 * from templates: a chain (a hidden Markov model or a linear-chain CRF), a lattice (an Ising or Potts model), a tree,
 * and repeated slices (a dynamic Bayesian network; Koller & Friedman 2009, §6.2; Murphy 2002, "Dynamic Bayesian
 * Networks", PhD thesis, UC Berkeley).
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
  from: string
  to: string
  weight?: number
}

/** A plain specification of a structured graph: nodes, edges by name, and groups. */
export interface StructuredSpec<D = unknown> {
  name?: string
  nodes: readonly StructuredNode<D>[]
  edges?: readonly EdgeSpec[]
  groups?: readonly Group[]
  /** Named sizes; those used by groups are added automatically. */
  sizes?: readonly string[]
}

const fail = (message: string): never => {
  throw new AifnError('structuredGraph', `structuredGraph: ${message}`)
}

/** The sizes of a group as a list (one per axis; a tree has one, its depth). */
export const groupSizes = (g: Group): readonly SizeSpec[] => (Array.isArray(g.size) ? g.size : [g.size as SizeSpec])

/** The groups holding `group`, outermost first, ending with it (empty for null). */
export function groupChain(graph: Pick<StructuredGraph, 'groups'>, group: string | null): Group[] {
  const chain: Group[] = []
  for (let name = group; name !== null;) {
    const g = graph.groups.find((q) => q.name === name)
    if (!g) return fail(`unknown group ${name}`)
    chain.unshift(g)
    name = g.parent
  }
  return chain
}

/** The index of the node named `name`; throws when there is none. */
export function nodeIndex(graph: Pick<StructuredGraph, 'attributes'>, name: string): Index {
  const i = graph.attributes.findIndex((n) => n.name === name)
  if (i < 0) return fail(`no node named ${name}`)
  return i
}

/** The indices of the nodes with one of the given roles (every node without `roles`), in node order. */
export function nodesWithRole(graph: Pick<StructuredGraph, 'attributes'>, ...roles: NodeRole[]): Index[] {
  return graph.attributes.flatMap((n, i) => (roles.length === 0 || roles.includes(n.role) ? [i] : []))
}

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
 * A structured graph from a specification. Checks that names are unique, groups exist and nest without cycles, edge
 * ends exist, and every lagged edge joins two nodes of one template group with a lag of the template's kind. Edges
 * touching a factor default to undirected, others to directed.
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
    if (groupChain({ groups }, g.name).length > groups.length) fail(`groups nest in a cycle at ${g.name}`)
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
  label?: string
  data?: D
}

/** Options of a group declared with the builder. */
export interface GroupSpec {
  /** Index symbol(s); default `n` (plate), `t` (chain), `i`, `j` (lattice), `v` (tree). */
  index?: string | readonly string[]
  label?: string
  periodic?: boolean
}

/** A place to declare nodes: the graph itself or a group. Each declaration returns the node's name. */
export interface StructuredScope<D = unknown> {
  latent(name: string, options?: NodeSpec<D>): string
  observed(name: string, options?: NodeSpec<D>): string
  factor(name: string, options?: NodeSpec<D>): string
  deterministic(name: string, options?: NodeSpec<D>): string
  parameter(name: string, options?: NodeSpec<D>): string
  /** A plate nested here: exchangeable copies. */
  plate(name: string, size: SizeSpec, options?: GroupSpec): StructuredScope<D>
  /** A chain template nested here: copies 0 … length − 1 in order. */
  chain(name: string, length: SizeSpec, options?: GroupSpec): StructuredScope<D>
  /** A lattice template nested here: rows × cols sites. */
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
 * Describe a structured graph with a builder.
 *
 * ```ts
 * const hmm = structured('hidden Markov model', (b) => {
 *   const time = b.chain('time', b.size('T'))
 *   const z = time.latent('z'), x = time.observed('x')
 *   b.edge(z, z, { lag: 1 })
 *   b.edge(z, x)
 * })
 * ```
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
  periodic?: boolean
}

function withObservation(b: StructuredBuilder, scope: StructuredScope, z: string, o: TemplateOptions): void {
  if (o.observed === undefined) return
  const x = scope.observed(o.observed)
  b.edge(z, x, { directed: o.directed ?? true })
}

/**
 * A chain template of length T: latent z_t with z_{t−1} → z_t (lag 1), and an observed child x_t with
 * `observed: 'x'` (the hidden Markov model's structure; `directed: false` gives a linear-chain CRF's).
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
 * A lattice template of rows × cols sites: latent x_{ij} linked to its right and lower neighbours (4 neighbours) and
 * the two diagonals (8), undirected by default (an Ising or Potts model); a torus with `periodic`.
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

/** A tree template: latent z_v at each node of a complete `arity`-ary tree of the given depth, parent → child. */
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
 * group of the given length, plus a lag-1 edge for each `[from, to]` in `transitions` (from slice t − 1 to slice t).
 * The slice's own nodes must sit at its top level (not inside its groups) to be linked across slices.
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
    sizes: [...slice.sizes, ...(typeof length === 'string' ? [length] : [])],
  })
}
