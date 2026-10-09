/**
 * A small typed language for describing probabilistic models (plan §7.1), built on the structured graphs of
 * `aifn-compute/graph/structured`: a model is a structured graph whose nodes carry their conditional distribution
 * (from `aifn-compute/probability/distributions`) or deterministic link, whose plates are groups (nested, with fixed
 * or ragged sizes), and whose chains are chain templates (a variable with a first and a next conditional, as in a
 * hidden Markov model). Parameters (constants) are nodes with role `parameter`. `expandModel` unrolls the plates and
 * chains against sizes and data into instances, from which the factor graph, the Markov blankets, the log joint,
 * ancestral samples and the inference engines are built (Koller & Friedman 2009, "Probabilistic Graphical Models",
 * ch. 3 and 6).
 *
 * ```ts
 * const lda = model('Latent Dirichlet allocation', (m) => {
 *   const K = m.size('K'), V = m.size('V')
 *   const topics = m.plate('topics', K), docs = m.plate('documents', 'D'), words = docs.plate('words', 'N')
 *   const alpha = m.constant('α'), beta = m.constant('β')
 *   const phi = topics.variable('φ', dist.Dirichlet(beta, V))
 *   const theta = docs.variable('θ', dist.Dirichlet(alpha, K))
 *   const z = words.variable('z', dist.Categorical(theta))
 *   words.observed('w', dist.Categorical(phi.at(z)))
 * })
 * const hmm = model('hidden Markov model', (m) => {
 *   const pi = m.constant('π'), A = m.constant('A'), B = m.constant('B')
 *   const time = m.chain('time', 'T')
 *   const z = time.variable('z', dist.Categorical(pi), { next: (previous) => dist.Categorical(A.at(previous)) })
 *   time.observed('x', dist.Categorical(B.at(z)))
 * })
 * ```
 *
 * `ref.at(selector)` indexes by the value of a discrete node: it picks one instance of a plate the referring node is
 * not in ($\phi_z$ above), or one entry (row) of a vector- or matrix-valued node (the entry $\mu_z$ of a vector of
 * means, the row $\Amat_{z_{t-1}}$ of a transition matrix above).
 *
 * Errors in a model (a duplicate name, a reference to an undeclared node, a size or value missing at expansion) throw
 * `DomainError`.
 */

import {
  Bernoulli,
  Beta,
  Binomial,
  Categorical,
  Dirichlet,
  Gamma,
  Normal,
  Poisson,
  type Distribution,
} from 'aifn-compute/probability/distributions'
import type { Index, Size } from 'aifn-compute/foundation/contracts'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { normalCdf, sigmoid } from 'aifn-compute/numerics/special'
import {
  add,
  exp,
  fromData,
  full,
  isTensor,
  mul,
  sub,
  sum,
  tensor,
  toArray,
  toFlat,
  type NestedArray,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import type { Raw as NodeValue } from 'aifn-compute/foundation/contracts'
import {
  groupChain,
  groupSizes,
  structuredGraph,
  unroll,
  type Blanket,
  type EdgeSpec,
  type Group,
  type SizeBindings,
  type StructuredGraph,
  type StructuredNode,
} from 'aifn-compute/graph/structured'
import { DomainError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Raw as NodeValue } from 'aifn-compute/foundation/contracts'

/**
 * Families the language supports, each realised by the `aifn-compute/probability/distributions` constructor of that
 * name.
 */
export type Family = 'Normal' | 'Bernoulli' | 'Categorical' | 'Binomial' | 'Poisson' | 'Beta' | 'Gamma' | 'Dirichlet'

/**
 * A reference to another node. `lag: 1` reads the previous copy along the enclosing chain (inside a chain variable's
 * `next`). `select` (with `selectLag`) indexes by the value of a discrete node (see `at`).
 */
export interface NodeRef {
  /** Marks the argument as a reference. */
  readonly kind: 'ref'
  /** The name of the node referred to. */
  readonly node: string
  /** How many copies back along the enclosing chain to read (left out: the copy at the same index). */
  readonly lag?: number
  /** The name of the discrete node whose value picks the instance or entry (set by `at`). */
  readonly select?: string
  /** The lag of the selector, when it is the previous copy along the chain. */
  readonly selectLag?: number
}

/** A named size, bound when the model is expanded. */
export interface SizeRef {
  /** Marks the argument as a size. */
  readonly kind: 'size'
  /** The size's name, a key of `Bindings.sizes`. */
  readonly name: string
}

/** An argument of a distribution or deterministic node. */
export type Arg = number | readonly number[] | Tensor | NodeRef | SizeRef

/** A conditional distribution: a family and its arguments, in the constructor's order. */
export interface DistSpec {
  /** The distribution family. */
  family: Family
  /** The family's arguments, in the order of its constructor in `dist`. */
  args: readonly Arg[]
}

/**
 * Deterministic links: `sum` ($\sum_i a_i$ of the args), `difference` ($a - b$), `product` ($\prod_i a_i$), `linear`
 * ($\wvec^\top \xvec + b$ for a constant weight vector $\wvec$, a vector node $\xvec$ and an optional bias $b$),
 * `probit` ($\Phi(a)$), `logistic` ($\sigma(a)$), `exp`, `index` (the entry $T_{ij\dots}$ of a table: a conditional
 * probability table indexed by the values of discrete parents), and `interval` ($\indicator(l < x < u)$ for args
 * $[x, l, u]$; either bound may be $\pm\infty$).
 *
 * An interval states a constraint (a truncation) when an observed `Bernoulli` of it is 1: the node then contributes
 * $\log \indicator(l < x < u)$ to the joint density, and expectation propagation over the model
 * (`modelExpectationPropagation` of `aifn-compute/inference/expectation-propagation`) treats it as an interval factor
 * with truncated-normal moments. An observed 0 states the complement, which is an interval only when one bound is
 * infinite.
 */
export type DeterministicOp =
  'sum' | 'difference' | 'product' | 'linear' | 'probit' | 'logistic' | 'exp' | 'index' | 'interval'

/**
 * What a model node carries: its conditional distribution (`dist`; for a chain variable, the one at $t = 0$, with
 * `next` the one at $t \ge 1$), or its deterministic link (`op`, `args`), or a parameter's default value.
 */
export interface ModelNodeData {
  /** A stochastic node's conditional distribution (at $t = 0$ for a chain variable with `next`). */
  dist?: DistSpec
  /** A chain variable's conditional at $t \ge 1$, which may read the previous copy. */
  next?: DistSpec
  /** A deterministic node's link. */
  op?: DeterministicOp
  /** A deterministic node's arguments, in the order its link takes them. */
  args?: readonly Arg[]
  /** A parameter's default value (nested by plate when the parameter is in one), overridden by `Bindings.constants`. */
  value?: NodeValue | NestedArray
}

/** One node of a model: a structured node (role `parameter`, `latent`, `observed` or `deterministic`) with its data. */
export type ModelNode = StructuredNode<ModelNodeData>

/** A model: a structured graph of model nodes, with plates and chains as its groups. Plain, serialisable data. */
export type Model = StructuredGraph<ModelNodeData> & { name: string }

/** A node handle in the builder: a reference, with `at` for indexing by a discrete node's value. */
export interface NodeHandle extends NodeRef {
  /**
   * A reference to this node indexed by the value of the discrete node `selector`: the instance of this node's plate
   * it picks, or the entry (row) of this node's value.
   */
  at(selector: NodeRef): NodeRef
}

/** Options for a node: its TeX label, and for a variable in a chain its conditional given the previous copy. */
export interface NodeOptions {
  /** The node's TeX label in diagrams. */
  label?: string
  /** Chain variables: the conditional at $t \ge 1$, given a handle on the previous copy (`lag: 1`). */
  next?: (previous: NodeHandle) => DistSpec
}

/** Options of a plate or chain: its TeX label and index symbol. */
export interface GroupOptions {
  /** The group's TeX label in diagrams (default: its size). */
  label?: string
  /** The symbol of its index (default `t` for a chain, the first letter of the name for a plate). */
  index?: string
}

/** A place to declare nodes: the model itself, a plate or a chain. */
export interface ModelScope {
  /** A parameter (role `parameter`), with its default value, which `Bindings.constants` may override. */
  constant(name: string, value?: NodeValue | NestedArray, options?: NodeOptions): NodeHandle
  /** A latent random variable with its conditional distribution. */
  variable(name: string, distribution: DistSpec, options?: NodeOptions): NodeHandle
  /** An observed random variable with its conditional distribution; its values come from `Bindings.data`. */
  observed(name: string, distribution: DistSpec, options?: NodeOptions): NodeHandle
  /** A deterministic node: the link `op` applied to `args`. */
  deterministic(name: string, op: DeterministicOp, args: readonly Arg[], options?: NodeOptions): NodeHandle
  /** A plate nested here, of a fixed size or a named size (declared on first use). */
  plate(name: string, size: Size | string | SizeRef, options?: GroupOptions): ModelScope
  /** A chain nested here: copies $t = 0, \dots, T - 1$ in order; its variables may take a `next` conditional. */
  chain(name: string, size: Size | string | SizeRef, options?: GroupOptions): ModelScope
}

/** The builder passed to `model`. */
export interface ModelBuilder extends ModelScope {
  /** Declare a named size, bound when the model is expanded, to pass as an argument (a dimension) or a plate size. */
  size(name: string): SizeRef
}

/**
 * Distribution constructors for the language; each records its family and arguments, which may be numbers, vectors,
 * tensors, node handles or sizes.
 *
 * @example A conditional distribution is plain data
 * print(dist.Normal(0, 1))
 * print(dist.Dirichlet(1, 3))
 */
export const dist = {
  /** Normal by mean and standard deviation. */
  Normal: (mean: Arg, sd: Arg): DistSpec => ({ family: 'Normal', args: [mean, sd] }),
  /** Bernoulli over $\{0, 1\}$ with probability `p` of 1. */
  Bernoulli: (p: Arg): DistSpec => ({ family: 'Bernoulli', args: [p] }),
  /** Categorical over $0, \dots, K - 1$ with probabilities `probs` (a vector, or a node holding one). */
  Categorical: (probs: Arg): DistSpec => ({ family: 'Categorical', args: [probs] }),
  /** Binomial: the number of successes in `n` trials of probability `p`. */
  Binomial: (n: Arg, p: Arg): DistSpec => ({ family: 'Binomial', args: [n, p] }),
  /** Poisson with mean `rate`. */
  Poisson: (rate: Arg): DistSpec => ({ family: 'Poisson', args: [rate] }),
  /** Beta with shapes `a` and `b`. */
  Beta: (a: Arg, b: Arg): DistSpec => ({ family: 'Beta', args: [a, b] }),
  /** Gamma by shape and rate. */
  Gamma: (shape: Arg, rate: Arg): DistSpec => ({ family: 'Gamma', args: [shape, rate] }),
  /** Dirichlet with a concentration vector, or a symmetric concentration and a dimension. */
  Dirichlet: (concentration: Arg, dimension?: Arg): DistSpec => ({
    family: 'Dirichlet',
    args: dimension === undefined ? [concentration] : [concentration, dimension],
  }),
}

/**
 * Whether an argument is a reference to a node.
 *
 * @param a The argument.
 */
const isRef = (a: Arg): a is NodeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'ref'
/**
 * Whether an argument is a named size.
 *
 * @param a The argument.
 */
const isSize = (a: Arg): a is SizeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'size'

/**
 * Every argument of a node: its distribution's, its next conditional's and its link's.
 *
 * @param n The model node.
 * @returns The arguments of `dist`, then of `next`, then the link's `args` (empty for a parameter).
 *
 * @example The arguments of a chain variable's two conditionals
 * const hmm = model('chain', (m) => {
 *   const A = m.constant('A', [[0.9, 0.1], [0.2, 0.8]])
 *   m.chain('time', 3).variable('z', dist.Categorical([0.5, 0.5]), { next: (prev) => dist.Categorical(A.at(prev)) })
 * })
 * print(nodeArgs(hmm.attributes[1]))
 */
export const nodeArgs = (n: ModelNode): readonly Arg[] => [
  ...(n.data?.dist?.args ?? []),
  ...(n.data?.next?.args ?? []),
  ...(n.data?.args ?? []),
]

/**
 * The nodes an argument list refers to (including selectors).
 *
 * @param args The arguments; constants and sizes among them are skipped. Left out, none.
 * @returns The referred-to node names in argument order, each reference's node followed by its selector, with any
 *   repeats kept.
 *
 * @example A mean picked by a selector, and a constant scale
 * const mix = model('mixture', (m) => {
 *   const mu = m.plate('clusters', 2).variable('mu', dist.Normal(0, 10))
 *   const z = m.variable('z', dist.Categorical([0.5, 0.5]))
 *   m.observed('x', dist.Normal(mu.at(z), 1))
 * })
 * print(argRefs(nodeArgs(mix.attributes[2])))
 */
export function argRefs(args: readonly Arg[] = []): string[] {
  return args.flatMap((a) => (isRef(a) ? (a.select ? [a.node, a.select] : [a.node]) : []))
}

/**
 * The edges a node's arguments make: from each parent (and selector) to the node, directed, carrying the lag for
 * references to a previous copy. A parent referred to twice with the same lag gives one edge.
 *
 * @param n The model node whose incoming edges are made.
 * @returns The edges, by node name.
 */
function refEdges(n: ModelNode): EdgeSpec[] {
  const out = new Map<string, EdgeSpec>()
  const add = (from: string, lag: number | undefined) => {
    const e: EdgeSpec = { from, to: n.name, directed: true, ...(lag ? { lag } : {}) }
    out.set(`${from}|${lag ?? 0}`, e)
  }
  for (const a of nodeArgs(n)) {
    if (!isRef(a)) continue
    add(a.node, a.lag)
    if (a.select) add(a.select, a.selectLag)
  }
  return [...out.values()]
}

/**
 * Describe a model. Node names must be unique; nodes may refer only to nodes declared before them. The build function
 * declares the model's nodes, plates and chains on the builder; the result is a structured graph with an edge from
 * each node's parents (lagged for a chain variable's previous copy). Throws `DomainError` for a duplicate node or
 * group name, a reference to an undeclared node, a `next` conditional outside a chain, or a previous copy read from
 * outside its chain.
 *
 * @param name The model's name.
 * @param build Called once with the builder, on which it declares the model.
 * @returns The model: plain, serialisable data.
 *
 * @example A coin of unknown bias, flipped $N$ times
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * print('nodes:', coins.attributes.map((n) => `${n.name} (${n.role}, in ${n.group})`))
 * print('plates:', coins.groups.map((g) => `${g.name} of size ${g.size}`))
 * print('edges:', coins.edges.map((e) => `${coins.attributes[e.from].name} -> ${coins.attributes[e.to].name}`))
 *
 * @example A chain whose next state reads the previous one
 * const hmm = model('hidden Markov model', (m) => {
 *   const A = m.constant('A', [[0.9, 0.1], [0.2, 0.8]])
 *   const time = m.chain('time', 'T')
 *   const z = time.variable('z', dist.Categorical([0.5, 0.5]), { next: (prev) => dist.Categorical(A.at(prev)) })
 *   time.observed('x', dist.Normal(z, 1))
 * })
 * const name = (i) => hmm.attributes[i].name
 * print('edges:', hmm.edges.map((e) => `${name(e.from)} -> ${name(e.to)} (lag ${e.lag ?? 0})`))
 *
 * @example A reference to an undeclared node throws
 * try {
 *   model('bad', (m) => m.variable('y', dist.Normal({ kind: 'ref', node: 'mu' }, 1)))
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function model(name: string, build: (m: ModelBuilder) => void): Model {
  const nodes: ModelNode[] = []
  const groups: Group[] = []
  const sizes: string[] = []
  const names = new Set<string>()
  // A handle is stored as a plain reference wherever it is passed as an argument, so `at` is not enumerable: the
  // model stays plain data (structuredClone and JSON copy the reference and drop the method).
  const handle = (node: string, lag?: number): NodeHandle => {
    const ref: NodeRef = { kind: 'ref', node, ...(lag ? { lag } : {}) }
    const at = (selector: NodeRef): NodeRef => ({
      ...ref,
      select: selector.node,
      ...(selector.lag ? { selectLag: selector.lag } : {}),
    })
    return Object.defineProperty(ref, 'at', { value: at, enumerable: false }) as NodeHandle
  }
  const add = (node: ModelNode, options: NodeOptions): NodeHandle => {
    if (names.has(node.name)) throw new DomainError('model', `model: duplicate node ${node.name}`)
    if (options.next) {
      const g = groups.find((q) => q.name === node.group)
      if (g?.kind !== 'chain')
        throw new DomainError('model', `model: ${node.name} has a next conditional outside a chain`)
      node = { ...node, data: { ...node.data, next: options.next(handle(node.name, 1)) } }
    }
    for (const a of nodeArgs(node)) {
      if (!isRef(a)) continue
      for (const [r, lag] of [
        [a.node, a.lag],
        [a.select, a.selectLag],
      ] as const) {
        if (r === undefined) continue
        if (lag) {
          if (r !== node.name && !names.has(r))
            throw new DomainError('model', `model: ${node.name} refers to undeclared node ${r}`)
          const rg = r === node.name ? node.group : nodes.find((q) => q.name === r)!.group
          if (rg !== node.group)
            throw new DomainError('model', `model: ${node.name} reads the previous ${r} outside its chain`)
        } else if (!names.has(r)) throw new DomainError('model', `model: ${node.name} refers to undeclared node ${r}`)
      }
    }
    names.add(node.name)
    nodes.push(node)
    return handle(node.name)
  }
  const sizeName = (s: string) => {
    if (!sizes.includes(s)) sizes.push(s)
  }
  const label = (o: NodeOptions) => (o.label === undefined ? {} : { label: o.label })
  const scope = (group: string | null): ModelScope => {
    const nest = (kind: 'plate' | 'chain', n: string, size: Size | string | SizeRef, o: GroupOptions): ModelScope => {
      if (groups.some((p) => p.name === n)) throw new DomainError('model', `model: duplicate plate ${n}`)
      const s = typeof size === 'object' ? size.name : size
      if (typeof s === 'string') sizeName(s)
      groups.push({
        name: n,
        kind,
        size: s,
        index: [o.index ?? (kind === 'chain' ? 't' : n[0])],
        parent: group,
        ...(o.label === undefined ? {} : { label: o.label }),
      })
      return scope(n)
    }
    return {
      constant: (n, value, o = {}) =>
        add({ name: n, role: 'parameter', group, ...label(o), ...(value === undefined ? {} : { data: { value } }) }, o),
      variable: (n, d, o = {}) => add({ name: n, role: 'latent', group, ...label(o), data: { dist: d } }, o),
      observed: (n, d, o = {}) => add({ name: n, role: 'observed', group, ...label(o), data: { dist: d } }, o),
      deterministic: (n, op, args, o = {}) =>
        add({ name: n, role: 'deterministic', group, ...label(o), data: { op, args } }, o),
      plate: (n, size, o = {}) => nest('plate', n, size, o),
      chain: (n, size, o = {}) => nest('chain', n, size, o),
    }
  }
  const root = scope(null)
  build({ ...root, size: (n) => (sizeName(n), { kind: 'size', name: n }) })
  return { ...structuredGraph<ModelNodeData>({ name, nodes, groups, sizes, edges: nodes.flatMap(refEdges) }), name }
}

/**
 * The names of the plates and chains holding a node's group, outermost first, ending with the group itself. Throws
 * when a name on the way is not a group of the model.
 *
 * @param m The model, or anything with its `groups`.
 * @param group The name of the group a node is in, or null for a node outside every plate.
 * @returns The group names, outermost first (empty for null).
 *
 * @example Words nested in documents
 * const docs = model('nested plates', (m) => {
 *   m.plate('documents', 2).plate('words', 3).observed('w', dist.Bernoulli(0.5))
 * })
 * print('words:', plateChain(docs, 'words'))
 * print('documents:', plateChain(docs, 'documents'))
 * print('none:', plateChain(docs, null))
 */
export function plateChain(m: Pick<Model, 'groups'>, group: string | null): string[] {
  return groupChain(m, group).map((g) => g.name)
}

// ── Expansion ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Nested data: a value per plate index (ragged arrays allowed), leaves numbers or vectors. */
export type Nested = NodeValue | NestedArray | readonly Nested[]

/** What a model is expanded against. */
export interface Bindings {
  /** Named sizes: a number, or for a nested plate one size per index of its parent plate. */
  sizes?: Readonly<Record<string, Size | readonly Size[]>>
  /** Values of constants (nested by plate), overriding their defaults. */
  constants?: Readonly<Record<string, Nested>>
  /** Values of observed nodes (nested by plate). */
  data?: Readonly<Record<string, Nested>>
}

/** One copy of a node: its key (`z[2,5]`), node, plate indices (outermost first) and plates. */
export interface Instance {
  /** The copy's key, `name[i,j]`, or the node's name when it is in no plate. */
  key: string
  /** The model node it is a copy of. */
  node: ModelNode
  /** Its index in each enclosing plate or chain, outermost first. */
  index: Index[]
  /** The names of the enclosing plates and chains, outermost first. */
  plates: string[]
}

/** A model unrolled against bindings: the explicit graph, its instances and the fixed values. */
export interface ExpandedModel {
  /** The model that was expanded. */
  model: Model
  /** The bindings it was expanded against. */
  bindings: Bindings
  /** The unrolled structured graph: one node per instance, in the order of `instances`. */
  graph: StructuredGraph<ModelNodeData>
  /** Instances in declaration order (a topological order). */
  instances: Instance[]
  /** Every instance by its key. */
  byKey: Map<string, Instance>
  /** The instances of each model node, by the node's name, in the order of `instances`. */
  byNode: Map<string, Instance[]>
  /** Values of constants and observed instances, by key. */
  fixed: Map<string, NodeValue>
}

/**
 * The key of a copy: `z[2,5]`, or the name of an unplated node.
 *
 * @param name The model node's name.
 * @param index The copy's index in each enclosing plate, outermost first (empty for an unplated node).
 * @returns The key.
 *
 * @example A copy in two plates, and an unplated node
 * print(instanceKey('z', [2, 5]))
 * print(instanceKey('p', []))
 */
export const instanceKey = (name: string, index: readonly Index[]): string =>
  index.length ? `${name}[${index.join(',')}]` : name

/**
 * A nested value as a node value: a number or tensor as it is, nested arrays as a tensor.
 *
 * @param v The value.
 */
function toValue(v: Nested): NodeValue {
  if (typeof v === 'number' || isTensor(v)) return v as NodeValue
  return tensor(v as NestedArray)
}

/**
 * Pick a nested value by plate index: one level of nesting per index, a tensor read as nested arrays. Throws
 * `DomainError` when the value is not nested deeply enough or has no entry at the index.
 *
 * @param v The nested values of a node (from the bindings or a default), or undefined.
 * @param index The plate indices of the instance, outermost first.
 * @param what The node's name, for error messages.
 * @returns The value at the index, or undefined when `v` is.
 */
function pick(v: Nested | undefined, index: readonly Index[], what: string): NodeValue | undefined {
  if (v === undefined) return undefined
  let cur: Nested = isTensor(v) && index.length ? (toArray(v as Tensor) as Nested) : v
  for (const i of index) {
    if (!Array.isArray(cur)) throw new DomainError('expandModel', `expandModel: ${what} is not nested deeply enough`)
    cur = (cur as readonly Nested[])[i]
    if (cur === undefined)
      throw new DomainError('expandModel', `expandModel: ${what} has no entry at index ${index.join(',')}`)
  }
  return toValue(cur)
}

/**
 * The sizes of a model's groups against bindings: a bound size (per index of the enclosing group when ragged), or one
 * read from the data of an observed node inside the group. The function returned throws `DomainError` for a named
 * size that is neither bound nor readable from data.
 *
 * @param m The model.
 * @param b The bindings: named sizes and data.
 * @returns The size lookup `unroll` calls for each axis of each group, given the indices of the enclosing groups.
 */
function modelSizes(m: Model, b: Bindings): SizeBindings {
  return (group, axis, outer) => {
    const spec = groupSizes(group)[axis]
    if (typeof spec === 'number') return spec
    const s = b.sizes?.[spec]
    if (s !== undefined) return typeof s === 'number' ? s : s[outer[outer.length - 1] ?? 0]
    const inside = m.attributes.find(
      (n) => n.role === 'observed' && b.data?.[n.name] !== undefined && plateChain(m, n.group).includes(group.name),
    )
    if (inside) {
      let cur = b.data![inside.name] as Nested
      for (const i of outer)
        cur = isTensor(cur) ? (toArray(cur as Tensor) as Nested[])[i] : (cur as readonly Nested[])[i]
      if (Array.isArray(cur)) return cur.length
      if (isTensor(cur)) return (cur as Tensor).shape[0]
    }
    throw new DomainError('expandModel', `expandModel: size ${spec} is not bound`)
  }
}

/**
 * Unroll a model's plates and chains against sizes, constants and data. A named size is taken from `bindings.sizes`,
 * or else from the length of the data of an observed node inside the plate. Throws `DomainError` for a size that is
 * not bound, a constant with no value, or data not nested deeply enough.
 *
 * @param m The model.
 * @param bindings Named sizes, constant values and data. Observed nodes without data stay free (they are sampled by
 *   `sampleModel`).
 * @returns The expanded model: its instances, in declaration order, and the fixed values of constants and data.
 *
 * @example Three flips of a coin, the size read from the data
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const em = expandModel(coins, { data: { x: [1, 0, 1] } })
 * print('instances:', em.instances.map((i) => i.key))
 * print('fixed:', [...em.fixed])
 */
export function expandModel(m: Model, bindings: Bindings = {}): ExpandedModel {
  const graph = unroll(m, modelSizes(m, bindings))
  const byName = new Map(m.attributes.map((n) => [n.name, n]))
  const instances: Instance[] = []
  const byKey = new Map<string, Instance>()
  const byNode = new Map<string, Instance[]>(m.attributes.map((n) => [n.name, []]))
  const fixed = new Map<string, NodeValue>()
  for (const copy of graph.attributes) {
    const node = byName.get(copy.source ?? copy.name)!
    const index = [...(copy.index ?? [])]
    const inst: Instance = { key: copy.name, node, index, plates: plateChain(m, node.group) }
    instances.push(inst)
    byKey.set(inst.key, inst)
    byNode.get(node.name)!.push(inst)
    if (node.role === 'parameter') {
      const v =
        pick(bindings.constants?.[node.name], index, node.name) ?? pick(node.data?.value as Nested, index, node.name)
      if (v === undefined) throw new DomainError('expandModel', `expandModel: constant ${node.name} has no value`)
      fixed.set(inst.key, v)
    } else if (node.role === 'observed') {
      const v = pick(bindings.data?.[node.name], index, node.name)
      if (v !== undefined) fixed.set(inst.key, v)
    }
  }
  return { model: m, bindings, graph, instances, byKey, byNode, fixed }
}

// ── Resolving arguments ─────────────────────────────────────────────────────────────────────────────────────────────

/** A lookup of instance values by key (latent values, then fixed ones). */
export type Env = (key: string) => NodeValue

/**
 * The conditional of a stochastic instance: `next` at $t \ge 1$ of a chain, else `dist`. Throws `DomainError` for an
 * instance with no distribution (a parameter or a deterministic node).
 *
 * @param inst The instance; its last plate index is its position $t$ along the chain.
 * @returns The distribution spec, with its arguments unresolved.
 *
 * @example The first and a later step of a chain
 * const hmm = model('chain', (m) => {
 *   const A = m.constant('A', [[0.9, 0.1], [0.2, 0.8]])
 *   m.chain('time', 3).variable('z', dist.Categorical([0.5, 0.5]), { next: (prev) => dist.Categorical(A.at(prev)) })
 * })
 * const em = expandModel(hmm)
 * print('z[0]:', distOf(em.byKey.get('z[0]')))
 * print('z[2]:', distOf(em.byKey.get('z[2]')))
 */
export function distOf(inst: Instance): DistSpec {
  const d = inst.node.data
  if (!d?.dist) throw new DomainError('model', `model: ${inst.key} has no distribution`)
  return d.next && inst.index[inst.index.length - 1] >= 1 ? d.next : d.dist
}

/**
 * The key of the copy of `node` that `inst` reads `lag` steps back along its chain. Throws `DomainError` when that
 * would be before the first copy.
 *
 * @param inst The reading instance; its last plate index is its position along the chain.
 * @param node The name of the node read.
 * @param lag How many steps back.
 * @returns The key of the copy read.
 */
function laggedKey(inst: Instance, node: string, lag: number): string {
  const index = [...inst.index]
  index[index.length - 1] -= lag
  if (index[index.length - 1] < 0) throw new DomainError('model', `model: ${inst.key} has no previous ${node}`)
  return instanceKey(node, index)
}

/**
 * The instance keys a reference could point to from `inst`, and a function choosing one given the values. A lagged
 * reference reads the previous copy along the chain; a reference to a node in the same plates (or fewer) reads the copy
 * at the same indices; a reference with a selector to a node in one more plate has every copy of that plate as a
 * candidate. Throws `DomainError` for a reference across plates without a selector.
 *
 * @param em The expanded model.
 * @param inst The instance holding the reference.
 * @param ref The reference, as it appears in the node's arguments.
 * @returns `candidates`, the keys it may point to; `selector`, the key of the selector's instance, or null;
 *   `indexesValue`, true when the selector picks an entry of the one candidate's value rather than a candidate; and
 *   `choose`, which given an environment returns the key pointed to.
 *
 * @example A mean picked from a plate of clusters, and an entry picked from a vector
 * const mix = model('mixture', (m) => {
 *   const mu = m.plate('clusters', 2).variable('mu', dist.Normal(0, 10))
 *   const shift = m.constant('shift', [-1, 1])
 *   const points = m.plate('points', 'N')
 *   const z = points.variable('z', dist.Categorical([0.5, 0.5]))
 *   points.observed('x', dist.Normal(mu.at(z), 1))
 *   points.observed('y', dist.Normal(shift.at(z), 1))
 * })
 * const em = expandModel(mix, { sizes: { N: 2 } })
 * const fromX = resolveRef(em, em.byKey.get('x[1]'), distOf(em.byKey.get('x[1]')).args[0])
 * print('x[1]:', fromX.candidates, 'selected by', fromX.selector)
 * print('when z[1] = 1:', fromX.choose(environment(em, { 'z[1]': 1 })))
 * const fromY = resolveRef(em, em.byKey.get('y[1]'), distOf(em.byKey.get('y[1]')).args[0])
 * print('y[1]:', fromY.candidates, 'indexes the value:', fromY.indexesValue)
 */
export function resolveRef(
  em: ExpandedModel,
  inst: Instance,
  ref: NodeRef,
): { candidates: string[]; selector: string | null; indexesValue: boolean; choose: (env: Env) => string } {
  const parent = em.model.attributes.find((n) => n.name === ref.node)!
  const pc = plateChain(em.model, parent.group)
  const prefixOf = (chain: string[], of: string[]) => chain.every((p, i) => of[i] === p) && chain.length <= of.length
  const selector = ref.select
    ? resolveRef(em, inst, { kind: 'ref', node: ref.select, ...(ref.selectLag ? { lag: ref.selectLag } : {}) })
        .candidates[0]
    : null
  if (ref.lag) {
    const key = laggedKey(inst, ref.node, ref.lag)
    return { candidates: [key], selector, indexesValue: selector !== null, choose: () => key }
  }
  if (prefixOf(pc, inst.plates)) {
    const key = instanceKey(parent.name, inst.index.slice(0, pc.length))
    return { candidates: [key], selector, indexesValue: selector !== null, choose: () => key }
  }
  const outer = pc.slice(0, -1)
  if (selector !== null && prefixOf(outer, inst.plates)) {
    const prefix = inst.index.slice(0, outer.length)
    const candidates = em.byNode
      .get(parent.name)!
      .filter((c) => c.index.slice(0, outer.length).every((v, i) => v === prefix[i]))
      .map((c) => c.key)
    return {
      candidates,
      selector,
      indexesValue: false,
      choose: (env) => candidates[Math.round(env(selector) as number)],
    }
  }
  throw new DomainError(
    'model',
    `model: ${inst.node.name} cannot refer to ${ref.node} across plates without .at(selector)`,
  )
}

/**
 * Index the first axis of a value: an entry of a vector, a row of a matrix (e.g. a CPT row for a Categorical). Throws
 * `DomainError` for a number.
 *
 * @param v The vector, matrix or higher tensor.
 * @param i The index along its first axis (not range-checked).
 * @returns The entry (a number) or the sub-tensor at `i`.
 */
function indexValue(v: NodeValue, i: number): NodeValue {
  if (typeof v === 'number') throw new DomainError('model', 'model: .at on a scalar node')
  const flat = toFlat(v)
  if (v.shape.length === 1) return flat[i]
  const rest = v.shape.slice(1)
  const n = rest.reduce((a, b) => a * b, 1)
  return fromData(
    Float64Array.from({ length: n }, (_, j) => flat[i * n + j]),
    rest,
  )
}

/**
 * The value of an argument for an instance: a number as it is, a size from the bindings (which must be a single
 * number), a reference resolved with `resolveRef` and read from `env` (and indexed by the selector's value when it
 * picks an entry), a tensor as it is, and an array as a tensor.
 *
 * @param em The expanded model, whose bindings give the sizes.
 * @param inst The instance whose argument it is.
 * @param arg The argument.
 * @param env The values of the instances.
 * @returns The argument's value.
 *
 * @example A mean picked by a cluster assignment
 * const mix = model('mixture', (m) => {
 *   const mu = m.plate('clusters', 2).variable('mu', dist.Normal(0, 10))
 *   const z = m.variable('z', dist.Categorical([0.5, 0.5]))
 *   m.observed('x', dist.Normal(mu.at(z), 1))
 * })
 * const em = expandModel(mix)
 * const x = em.byKey.get('x')
 * const [mean, sd] = distOf(x).args
 * const env = environment(em, { 'mu[0]': -2, 'mu[1]': 3, z: 1 })
 * print('mean:', argValue(em, x, mean, env))
 * print('sd:', argValue(em, x, sd, env))
 */
export function argValue(em: ExpandedModel, inst: Instance, arg: Arg, env: Env): NodeValue {
  if (typeof arg === 'number') return arg
  if (isSize(arg)) {
    const s = em.bindings.sizes?.[arg.name]
    if (typeof s !== 'number') throw new DomainError('model', `model: size ${arg.name} must be a number here`)
    return s
  }
  if (isRef(arg)) {
    const r = resolveRef(em, inst, arg)
    const v = env(r.choose(env))
    return r.indexesValue ? indexValue(v, Math.round(env(r.selector!) as number)) : v
  }
  if (isTensor(arg)) return arg as Tensor
  return tensor(arg as number[])
}

/**
 * Build the `aifn-compute/probability/distributions` object of a stochastic node from its argument values. A
 * Dirichlet with two values has the symmetric concentration `values[0]` repeated `values[1]` times.
 *
 * @param spec The conditional; only its family is read.
 * @param values The values of its arguments, in the constructor's order (as `argValue` gives them).
 * @returns The distribution.
 *
 * @example A normal, and a symmetric Dirichlet over three categories
 * const normal = realise(dist.Normal(0, 2), [0, 2])
 * print('normal:', normal.name, 'mean', normal.mean(), 'sd', normal.stddev())
 * print('Dirichlet:', realise(dist.Dirichlet(1, 3), [1, 3]).params)
 */
export function realise(spec: DistSpec, values: readonly NodeValue[]): Distribution {
  const [a, b] = values
  switch (spec.family) {
    case 'Normal':
      return Normal(a, b)
    case 'Bernoulli':
      return Bernoulli(a)
    case 'Categorical':
      return Categorical(a)
    case 'Binomial':
      return Binomial(a, b)
    case 'Poisson':
      return Poisson(a)
    case 'Beta':
      return Beta(a, b)
    case 'Gamma':
      return Gamma(a, b)
    case 'Dirichlet':
      return Dirichlet(b === undefined ? a : full([b as number], a as number))
  }
}

/**
 * The value of a deterministic node from its argument values (see `DeterministicOp` for the links). `index` takes the
 * table then one index per leading axis; indexing fewer axes than the table has gives a sub-table. Throws
 * `DomainError` for an `index` whose table is a number, or an `interval` with a non-scalar argument.
 *
 * @param op The link.
 * @param values The values of its arguments, in the order the link takes them.
 * @returns The node's value.
 *
 * @example Sums, a linear predictor and an interval
 * print('sum:', evaluateOp('sum', [1, 2, 3]))
 * print('linear:', evaluateOp('linear', [tensor([1, 2]), tensor([3, 4]), 0.5]))
 * print('0 < 0.5 < 1:', evaluateOp('interval', [0.5, 0, 1]))
 * print('0 < 2 < 1:', evaluateOp('interval', [2, 0, 1]))
 *
 * @example A row and an entry of a conditional probability table
 * const table = tensor([[0.1, 0.9], [0.7, 0.3]])
 * print('row 1:', evaluateOp('index', [table, 1]))
 * print('entry (1, 0):', evaluateOp('index', [table, 1, 0]))
 */
export function evaluateOp(op: DeterministicOp, values: readonly NodeValue[]): NodeValue {
  switch (op) {
    case 'sum':
      return values.reduce((s, v) => add(s, v) as NodeValue)
    case 'difference':
      return sub(values[0], values[1]) as NodeValue
    case 'product':
      return values.reduce((s, v) => mul(s, v) as NodeValue)
    case 'linear': {
      const dot = sum(mul(values[0], values[1])) as number
      return values.length > 2 ? dot + (values[2] as number) : dot
    }
    case 'probit':
      return normalCdf(values[0]) as NodeValue
    case 'logistic':
      return sigmoid(values[0]) as NodeValue
    case 'exp':
      return exp(values[0]) as NodeValue
    case 'interval': {
      const [x, lower, upper] = values.map((v) => {
        if (typeof v === 'number') return v
        if (v.shape.reduce((a, b) => a * b, 1) !== 1)
          throw new DomainError('model', 'model: interval takes scalar arguments')
        return toFlat(v)[0]
      })
      return lower < x && x < upper ? 1 : 0
    }
    case 'index': {
      const table = values[0]
      if (typeof table === 'number') throw new DomainError('model', 'model: index needs a table')
      const flat = toFlat(table)
      const k = values.length - 1
      let offset = 0
      let stride = 1
      for (let i = table.shape.length - 1; i >= 0; i--) {
        if (i < k) offset += Math.round(values[i + 1] as number) * stride
        stride *= table.shape[i]
      }
      // Indexing fewer axes than the table has leaves a sub-table (e.g. a row of probabilities for a Categorical).
      const rest = table.shape.slice(k)
      const n = rest.reduce((a, b) => a * b, 1)
      const start = offset
      return rest.length === 0
        ? flat[start]
        : fromData(
            Float64Array.from({ length: n }, (_, j) => flat[start + j]),
            rest,
          )
    }
  }
}

/**
 * An environment over the expanded model: `values` for latent (and any overridden) instances, `fixed` for constants
 * and data, deterministic instances computed on demand. The lookup throws `DomainError` for an unknown key, or for a
 * stochastic instance with no value.
 *
 * @param em The expanded model.
 * @param values Values by instance key, as a map or a record; read at each lookup, so later changes to a map are seen.
 *   They take precedence over the model's fixed values.
 * @returns The lookup from instance key to value.
 *
 * @example A latent value, a datum and a deterministic node
 * const wet = model('wet grass', (m) => {
 *   const rain = m.variable('rain', dist.Bernoulli(0.2))
 *   const pWet = m.deterministic('pWet', 'index', [[0.1, 0.9], rain])
 *   m.observed('wet', dist.Bernoulli(pWet))
 * })
 * const env = environment(expandModel(wet, { data: { wet: 1 } }), { rain: 1 })
 * print('rain:', env('rain'), 'wet:', env('wet'), 'pWet:', env('pWet'))
 */
export function environment(
  em: ExpandedModel,
  values: ReadonlyMap<string, NodeValue> | Readonly<Record<string, NodeValue>>,
): Env {
  const get =
    values instanceof Map ? (k: string) => values.get(k) : (k: string) => (values as Record<string, NodeValue>)[k]
  const env: Env = (key) => {
    const v = get(key) ?? em.fixed.get(key)
    if (v !== undefined) return v
    const inst = em.byKey.get(key)
    if (!inst) throw new DomainError('model', `model: unknown instance ${key}`)
    if (inst.node.role === 'deterministic')
      return evaluateOp(
        inst.node.data!.op!,
        inst.node.data!.args!.map((a) => argValue(em, inst, a, env)),
      )
    throw new DomainError('model', `model: no value for ${key}`)
  }
  return env
}

/**
 * The distribution of a stochastic instance given the values of everything else: its conditional (`distOf`) with its
 * arguments read from `env`.
 *
 * @param em The expanded model.
 * @param inst The stochastic instance.
 * @param env The values of its parents.
 * @returns The distribution.
 *
 * @example A flip given the coin's bias
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const em = expandModel(coins, { data: { x: [1, 0, 1] } })
 * const d = conditionalOf(em, em.byKey.get('x[0]'), environment(em, { p: 0.7 }))
 * print(d.name, d.params)
 */
export function conditionalOf(em: ExpandedModel, inst: Instance, env: Env): Distribution {
  const d = distOf(inst)
  return realise(
    d,
    d.args.map((a) => argValue(em, inst, a, env)),
  )
}

/**
 * A number, or the sum of a tensor's entries (a log-density over the entries of a vector event).
 *
 * @param v A number or a tensor.
 */
const asNumber = (v: unknown): number => (typeof v === 'number' ? v : (sum(v as Tensor) as number))

/**
 * $\log p(x_i \mid \text{parents})$, the log conditional density of the value of `inst` given its parents' values
 * under `env` (summed over the entries if the log-density is a tensor).
 *
 * @param em The expanded model.
 * @param inst The stochastic instance.
 * @param env The values of the instance and its parents.
 * @returns The log density (or log mass, for a discrete node).
 *
 * @example One flip, $\log 0.7$ for heads
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const em = expandModel(coins, { data: { x: [1, 0, 1] } })
 * const env = environment(em, { p: 0.7 })
 * print('x[0] = 1:', instanceLogDensity(em, em.byKey.get('x[0]'), env), 'log 0.7:', Math.log(0.7))
 * print('x[1] = 0:', instanceLogDensity(em, em.byKey.get('x[1]'), env), 'log 0.3:', Math.log(0.3))
 */
export function instanceLogDensity(em: ExpandedModel, inst: Instance, env: Env): number {
  return asNumber(conditionalOf(em, inst, env).logProb(env(inst.key)))
}

/**
 * Whether an instance is stochastic (latent or observed), not a parameter or a deterministic node.
 *
 * @param inst The instance.
 */
const stochastic = (inst: Instance) => inst.node.role === 'latent' || inst.node.role === 'observed'

/**
 * $\log p(\text{latent}, \text{data})$: the sum of every stochastic instance's log conditional density.
 *
 * @param m The model, or a model already expanded (then `bindings` is ignored).
 * @param values The value of every latent instance (and of any observed one without data), by key.
 * @param bindings What to expand a model against: sizes, constants and data.
 * @returns The log joint density.
 *
 * @example A Beta(2, 2) coin with $p = 0.5$ and flips 1, 0, 1: $\log 1.5 + 3 \log 0.5$
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * print('log joint:', logJoint(coins, { p: 0.5 }, { data: { x: [1, 0, 1] } }))
 * print('by hand:', Math.log(1.5) + 3 * Math.log(0.5))
 */
export function logJoint(
  m: Model | ExpandedModel,
  values: ReadonlyMap<string, NodeValue> | Readonly<Record<string, NodeValue>>,
  bindings: Bindings = {},
): number {
  const em = 'instances' in m ? m : expandModel(m, bindings)
  const env = environment(em, values)
  let total = 0
  for (const inst of em.instances) if (stochastic(inst)) total += instanceLogDensity(em, inst, env)
  return total
}

// ── Structure ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The stochastic instances an instance depends on directly, looking through deterministic nodes (all candidates of an
 * `at`, and its selector). Parameters are left out.
 *
 * @param em The expanded model.
 * @param inst The instance: stochastic, or deterministic to list what it is computed from.
 * @returns The keys of the stochastic parents, without repeats.
 *
 * @example Through a deterministic node, and through an `at`
 * const mix = model('mixture', (m) => {
 *   const mu = m.plate('clusters', 2).variable('mu', dist.Normal(0, 10))
 *   const z = m.variable('z', dist.Categorical([0.5, 0.5]))
 *   const mean = m.deterministic('mean', 'sum', [mu.at(z), 1])
 *   m.observed('x', dist.Normal(mean, 1))
 * })
 * const em = expandModel(mix)
 * print('parents of x:', stochasticParents(em, em.byKey.get('x')))
 */
export function stochasticParents(em: ExpandedModel, inst: Instance): string[] {
  const out = new Set<string>()
  const visit = (i: Instance) => {
    const args = i.node.role === 'deterministic' ? (i.node.data?.args ?? []) : distOf(i).args
    for (const a of args) {
      if (!isRef(a)) continue
      const r = resolveRef(em, i, a)
      for (const k of [...r.candidates, ...(r.selector ? [r.selector] : [])]) {
        const p = em.byKey.get(k)!
        if (p.node.role === 'deterministic') visit(p)
        else if (p.node.role !== 'parameter') out.add(k)
      }
    }
  }
  visit(inst)
  return [...out]
}

/**
 * Parents and children of every stochastic instance (looking through deterministic nodes), as `stochasticParents`
 * gives them.
 *
 * @param em The expanded model.
 * @returns `parents` and `children`, each mapping every stochastic instance's key to a list of keys.
 *
 * @example The coin's bias is the parent of every flip
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const { parents, children } = dependencyMaps(expandModel(coins, { sizes: { N: 2 } }))
 * print('parents:', [...parents])
 * print('children:', [...children])
 */
export function dependencyMaps(em: ExpandedModel): { parents: Map<string, string[]>; children: Map<string, string[]> } {
  const parents = new Map<string, string[]>()
  const children = new Map<string, string[]>()
  for (const inst of em.instances) {
    if (!stochastic(inst)) continue
    const ps = stochasticParents(em, inst)
    parents.set(inst.key, ps)
    if (!children.has(inst.key)) children.set(inst.key, [])
    for (const p of ps) {
      if (!children.has(p)) children.set(p, [])
      children.get(p)!.push(inst.key)
    }
  }
  return { parents, children }
}

/**
 * The Markov blanket of a stochastic instance (`z[0,3]`, or a node name for an unplated node) in a model expanded
 * against `bindings`: its parents, children and the children's other parents, looking through deterministic nodes
 * (Pearl 1988). For the blanket in the graph itself, deterministic nodes included, use `markovBlanket` of
 * `aifn-compute/graph/structured` on `expandModel(m, bindings).graph`. Throws `DomainError` when `key` is not a
 * stochastic instance.
 *
 * @param m The model, or a model already expanded (then `bindings` is ignored).
 * @param key The instance's key.
 * @param bindings What to expand a model against.
 * @returns The blanket: `parents`, `children`, `coParents` and their union `blanket` (`neighbours` is empty, as the
 *   model is directed).
 *
 * @example In a chain $z_0 \to z_1 \to z_2$ with an observation of each, the middle state's blanket
 * const hmm = model('chain', (m) => {
 *   const A = m.constant('A', [[0.9, 0.1], [0.2, 0.8]])
 *   const time = m.chain('time', 3)
 *   const z = time.variable('z', dist.Categorical([0.5, 0.5]), { next: (prev) => dist.Categorical(A.at(prev)) })
 *   time.observed('x', dist.Normal(z, 1))
 * })
 * print(modelMarkovBlanket(hmm, 'z[1]'))
 */
export function modelMarkovBlanket(m: Model | ExpandedModel, key: string, bindings: Bindings = {}): Blanket {
  const em = 'instances' in m ? m : expandModel(m, bindings)
  const { parents, children } = dependencyMaps(em)
  if (!parents.has(key)) throw new DomainError('markovBlanket', `markovBlanket: ${key} is not a stochastic instance`)
  const ch = children.get(key) ?? []
  const co = [...new Set(ch.flatMap((c) => parents.get(c)!).filter((p) => p !== key))]
  const ps = parents.get(key)!
  return { parents: ps, children: ch, coParents: co, neighbours: [], blanket: [...new Set([...ps, ...ch, ...co])] }
}

/**
 * The number of values of a discrete node (Bernoulli 2, Categorical $K$, Binomial $n + 1$), or null. A Categorical's
 * $K$ is read from its probabilities: a vector given inline, a parameter's value, the table of an `index` node, or the
 * dimension of a Dirichlet node; otherwise, as for a Binomial whose $n$ is not a number and for every other family,
 * the result is null.
 *
 * @param em The expanded model, whose fixed values and bindings give the sizes read.
 * @param inst The instance; only its node's `dist` is read (not `next`).
 * @returns The number of values, or null when the node is not discrete with a known number of values.
 *
 * @example A Bernoulli, a Categorical over a parameter's probabilities, and a normal
 * const m = model('mixed', (b) => {
 *   const probs = b.constant('probs', [0.2, 0.3, 0.5])
 *   b.variable('coin', dist.Bernoulli(0.5))
 *   b.variable('die', dist.Categorical(probs))
 *   b.variable('height', dist.Normal(170, 10))
 * })
 * const em = expandModel(m)
 * print('coin:', cardinalityOf(em, em.byKey.get('coin')))
 * print('die:', cardinalityOf(em, em.byKey.get('die')))
 * print('height:', cardinalityOf(em, em.byKey.get('height')))
 */
export function cardinalityOf(em: ExpandedModel, inst: Instance): number | null {
  const d = inst.node.data?.dist
  if (!d) return null
  if (d.family === 'Bernoulli') return 2
  if (d.family === 'Binomial') {
    const n = d.args[0]
    return typeof n === 'number' ? n + 1 : null
  }
  if (d.family !== 'Categorical') return null
  const probs = d.args[0]
  if (Array.isArray(probs)) return probs.length
  if (isTensor(probs)) return (probs as Tensor).shape[(probs as Tensor).shape.length - 1]
  if (!isRef(probs)) return null
  const target = em.byNode.get(probs.node)![0]
  if (target.node.role === 'parameter') {
    const v = em.fixed.get(target.key)!
    return typeof v === 'number' ? null : v.shape[v.shape.length - 1]
  }
  if (target.node.data?.op === 'index') {
    const t = target.node.data.args![0]
    const table = isRef(t)
      ? em.fixed.get(em.byNode.get(t.node)![0].key)
      : isTensor(t)
        ? (t as Tensor)
        : tensor(t as number[])
    return table === undefined || typeof table === 'number' ? null : table.shape[table.shape.length - 1]
  }
  const pd = target.node.data?.dist
  if (pd?.family === 'Dirichlet') {
    const dim = pd.args[1]
    if (typeof dim === 'number') return dim
    if (dim !== undefined && isSize(dim)) return em.bindings.sizes?.[dim.name] as number
    const c = pd.args[0]
    if (Array.isArray(c)) return c.length
    if (isTensor(c)) return (c as Tensor).shape[0]
  }
  return null
}

/**
 * Ancestral sampling: a value for every latent and unobserved instance, in declaration order. Each instance draws
 * from its own child stream of `s`, named by its key, so a draw does not depend on how many others came before it.
 *
 * @param s The random stream.
 * @param m The model.
 * @param bindings What to expand the model against (an observed node with data is not sampled), and `given`: values,
 *   nested by plate, to use instead of drawing for the named nodes.
 * @returns The value of every sampled instance, by key, in declaration order.
 *
 * @example Simulate four flips of a coin, with the bias drawn and then fixed
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * print('drawn:', [...sampleModel(stream(0), coins, { sizes: { N: 4 } })])
 * print('given p:', [...sampleModel(stream(0), coins, { sizes: { N: 4 }, given: { p: 0.99 } })])
 */
export function sampleModel(
  s: Stream,
  m: Model,
  bindings: Bindings & { given?: Readonly<Record<string, Nested>> } = {},
): Map<string, NodeValue> {
  const em = expandModel(m, bindings)
  const values = new Map<string, NodeValue>()
  const env = environment(em, values)
  for (const inst of em.instances) {
    if (!stochastic(inst) || em.fixed.has(inst.key)) continue
    const given = pick(bindings.given?.[inst.node.name], inst.index, inst.node.name)
    values.set(inst.key, given ?? (conditionalOf(em, inst, env).sample(child(s, inst.key)) as NodeValue))
  }
  return values
}

/**
 * Collect the values of one node's instances into nested arrays by plate index (e.g. sampled data), in the form
 * `Bindings.data` takes. Values missing from `values` are read from the model's fixed values.
 *
 * @param em The expanded model.
 * @param node The model node's name.
 * @param values Values by instance key, such as `sampleModel` returns.
 * @returns One level of nesting per plate (a tensor value as nested arrays), or the value itself for an unplated node.
 *
 * @example Sampled flips as data for the model
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const em = expandModel(coins, { sizes: { N: 4 } })
 * const draws = sampleModel(stream(0), coins, { sizes: { N: 4 } })
 * print('x:', nestedValues(em, 'x', draws))
 * print('p:', nestedValues(em, 'p', draws))
 */
export function nestedValues(em: ExpandedModel, node: string, values: ReadonlyMap<string, NodeValue>): Nested {
  const out: Nested[] = []
  const insts = em.byNode.get(node)!
  if (insts.length === 1 && insts[0].index.length === 0) return values.get(insts[0].key) ?? em.fixed.get(insts[0].key)!
  for (const inst of insts) {
    let cur = out
    inst.index.slice(0, -1).forEach((i) => {
      cur[i] ??= []
      cur = cur[i] as Nested[]
    })
    const v = values.get(inst.key) ?? em.fixed.get(inst.key)!
    cur[inst.index[inst.index.length - 1]] = typeof v === 'number' ? v : (toArray(v) as Nested)
  }
  return out
}
