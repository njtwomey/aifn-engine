/**
 * A small typed language for describing probabilistic models (plan §7.1), built on the structured graphs of
 * `aifn-compute/graph/structured`: a model is a structured graph whose nodes carry their conditional distribution (from
 * `aifn-compute/probability/distributions`) or deterministic link, whose plates are groups (nested, with fixed or ragged
 * sizes), and whose chains are chain templates (a variable with a first and a next conditional, as in a hidden Markov
 * model). Parameters (constants) are nodes with role `parameter`. `expandModel` unrolls the plates and chains against
 * sizes and data into instances, from which the factor graph, the Markov blankets, the log joint, ancestral samples and
 * the inference engines are built (Koller & Friedman 2009, "Probabilistic Graphical Models", ch. 3 and 6).
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
 * not in (φ_{z}), or one entry (row) of a vector- or matrix-valued node (means[z], A[z_{t−1}]).
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

/** Families the language supports, each realised by the `aifn-compute/probability/distributions` constructor of that name. */
export type Family = 'Normal' | 'Bernoulli' | 'Categorical' | 'Binomial' | 'Poisson' | 'Beta' | 'Gamma' | 'Dirichlet'

/**
 * A reference to another node. `lag: 1` reads the previous copy along the enclosing chain (inside a chain variable's
 * `next`). `select` (with `selectLag`) indexes by the value of a discrete node (see `at`).
 */
export interface NodeRef {
  readonly kind: 'ref'
  readonly node: string
  readonly lag?: number
  readonly select?: string
  readonly selectLag?: number
}

/** A named size, bound when the model is expanded. */
export interface SizeRef {
  readonly kind: 'size'
  readonly name: string
}

/** An argument of a distribution or deterministic node. */
export type Arg = number | readonly number[] | Tensor | NodeRef | SizeRef

/** A conditional distribution: a family and its arguments, in the constructor's order. */
export interface DistSpec {
  family: Family
  args: readonly Arg[]
}

/**
 * Deterministic links: `sum` (Σ args), `difference` (a − b), `product` (Π args), `linear` (w · x + b for a constant
 * weight vector w, a vector node x and an optional bias), `probit` (Φ(a)), `logistic` (σ(a)), `exp`, `index`
 * (table[i, j, …]: a conditional probability table indexed by the values of discrete parents), and `interval`
 * (𝟙(lower < x < upper) for args [x, lower, upper]; either bound may be ±∞).
 *
 * An interval states a constraint (a truncation) when an observed `Bernoulli` of it is 1: the node then contributes
 * log 𝟙(lower < x < upper) to the joint density, and expectation propagation over the model
 * (`modelExpectationPropagation` of `aifn-compute/inference/expectation-propagation`) treats it as an interval factor with
 * truncated-normal moments. An observed 0 states the complement, which is an interval only when one bound is infinite.
 */
export type DeterministicOp =
  'sum' | 'difference' | 'product' | 'linear' | 'probit' | 'logistic' | 'exp' | 'index' | 'interval'

/**
 * What a model node carries: its conditional distribution (`dist`; for a chain variable, the one at t = 0, with `next`
 * the one at t ≥ 1), or its deterministic link (`op`, `args`), or a parameter's default value.
 */
export interface ModelNodeData {
  dist?: DistSpec
  next?: DistSpec
  op?: DeterministicOp
  args?: readonly Arg[]
  value?: NodeValue | NestedArray
}

/** One node of a model: a structured node (role `parameter`, `latent`, `observed` or `deterministic`) with its data. */
export type ModelNode = StructuredNode<ModelNodeData>

/** A model: a structured graph of model nodes, with plates and chains as its groups. Plain, serialisable data. */
export type Model = StructuredGraph<ModelNodeData> & { name: string }

/** A node handle in the builder: a reference, with `at` for indexing by a discrete node's value. */
export interface NodeHandle extends NodeRef {
  at(selector: NodeRef): NodeRef
}

/** Options for a node: its TeX label, and for a variable in a chain its conditional given the previous copy. */
export interface NodeOptions {
  label?: string
  /** Chain variables: the conditional at t ≥ 1, given a handle on the previous copy (`lag: 1`). */
  next?: (previous: NodeHandle) => DistSpec
}

/** Options of a plate or chain: its TeX label and index symbol. */
export interface GroupOptions {
  label?: string
  index?: string
}

/** A place to declare nodes: the model itself, a plate or a chain. */
export interface ModelScope {
  constant(name: string, value?: NodeValue | NestedArray, options?: NodeOptions): NodeHandle
  variable(name: string, distribution: DistSpec, options?: NodeOptions): NodeHandle
  observed(name: string, distribution: DistSpec, options?: NodeOptions): NodeHandle
  deterministic(name: string, op: DeterministicOp, args: readonly Arg[], options?: NodeOptions): NodeHandle
  /** A plate nested here, of a fixed size or a named size (declared on first use). */
  plate(name: string, size: Size | string | SizeRef, options?: GroupOptions): ModelScope
  /** A chain nested here: copies t = 0 … T − 1 in order; its variables may take a `next` conditional. */
  chain(name: string, size: Size | string | SizeRef, options?: GroupOptions): ModelScope
}

/** The builder passed to `model`. */
export interface ModelBuilder extends ModelScope {
  size(name: string): SizeRef
}

/** Distribution constructors for the language; each records its family and arguments. */
export const dist = {
  Normal: (mean: Arg, sd: Arg): DistSpec => ({ family: 'Normal', args: [mean, sd] }),
  Bernoulli: (p: Arg): DistSpec => ({ family: 'Bernoulli', args: [p] }),
  /** Categorical over 0 … K − 1 with probabilities `probs` (a vector, or a node holding one). */
  Categorical: (probs: Arg): DistSpec => ({ family: 'Categorical', args: [probs] }),
  Binomial: (n: Arg, p: Arg): DistSpec => ({ family: 'Binomial', args: [n, p] }),
  Poisson: (rate: Arg): DistSpec => ({ family: 'Poisson', args: [rate] }),
  Beta: (a: Arg, b: Arg): DistSpec => ({ family: 'Beta', args: [a, b] }),
  /** Gamma by shape and rate. */
  Gamma: (shape: Arg, rate: Arg): DistSpec => ({ family: 'Gamma', args: [shape, rate] }),
  /** Dirichlet with a concentration vector, or a symmetric concentration and a dimension. */
  Dirichlet: (concentration: Arg, dimension?: Arg): DistSpec => ({
    family: 'Dirichlet',
    args: dimension === undefined ? [concentration] : [concentration, dimension],
  }),
}

const isRef = (a: Arg): a is NodeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'ref'
const isSize = (a: Arg): a is SizeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'size'

/** Every argument of a node: its distribution's, its next conditional's and its link's. */
export const nodeArgs = (n: ModelNode): readonly Arg[] => [
  ...(n.data?.dist?.args ?? []),
  ...(n.data?.next?.args ?? []),
  ...(n.data?.args ?? []),
]

/** The nodes an argument list refers to (including selectors). */
export function argRefs(args: readonly Arg[] = []): string[] {
  return args.flatMap((a) => (isRef(a) ? (a.select ? [a.node, a.select] : [a.node]) : []))
}

/** The edges a node's arguments make: parent → node, lagged for references to the previous copy. */
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

/** Describe a model. Node names must be unique; nodes may refer only to nodes declared before them. */
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

/** The names of the plates and chains holding a node's group, outermost first. */
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
  key: string
  node: ModelNode
  index: Index[]
  plates: string[]
}

/** A model unrolled against bindings: the explicit graph, its instances and the fixed values. */
export interface ExpandedModel {
  model: Model
  bindings: Bindings
  /** The unrolled structured graph: one node per instance, in the order of `instances`. */
  graph: StructuredGraph<ModelNodeData>
  /** Instances in declaration order (a topological order). */
  instances: Instance[]
  byKey: Map<string, Instance>
  byNode: Map<string, Instance[]>
  /** Values of constants and observed instances, by key. */
  fixed: Map<string, NodeValue>
}

/** The key of a copy: `z[2,5]`, or the name of an unplated node. */
export const instanceKey = (name: string, index: readonly Index[]): string =>
  index.length ? `${name}[${index.join(',')}]` : name

function toValue(v: Nested): NodeValue {
  if (typeof v === 'number' || isTensor(v)) return v as NodeValue
  return tensor(v as NestedArray)
}

/** Pick a nested value by plate index. */
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
 * read from the data of an observed node inside the group.
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

/** Unroll a model's plates and chains against sizes, constants and data. */
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

/** The conditional of a stochastic instance: `next` at t ≥ 1 of a chain, else `dist`. */
export function distOf(inst: Instance): DistSpec {
  const d = inst.node.data
  if (!d?.dist) throw new DomainError('model', `model: ${inst.key} has no distribution`)
  return d.next && inst.index[inst.index.length - 1] >= 1 ? d.next : d.dist
}

/** The key of the copy of `node` that `inst` reads `lag` steps back along its chain. */
function laggedKey(inst: Instance, node: string, lag: number): string {
  const index = [...inst.index]
  index[index.length - 1] -= lag
  if (index[index.length - 1] < 0) throw new DomainError('model', `model: ${inst.key} has no previous ${node}`)
  return instanceKey(node, index)
}

/** The instance keys a reference could point to from `inst`, and a function choosing one given the values. */
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

/** Index the first axis of a value: an entry of a vector, a row of a matrix (e.g. a CPT row for a Categorical). */
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

/** The value of an argument for an instance. */
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

/** Build the `aifn-compute/probability/distributions` object of a stochastic node from its argument values. */
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

/** The value of a deterministic node from its argument values. */
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
 * and data, deterministic instances computed on demand.
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

/** The distribution of a stochastic instance given the values of everything else. */
export function conditionalOf(em: ExpandedModel, inst: Instance, env: Env): Distribution {
  const d = distOf(inst)
  return realise(
    d,
    d.args.map((a) => argValue(em, inst, a, env)),
  )
}

const asNumber = (v: unknown): number => (typeof v === 'number' ? v : (sum(v as Tensor) as number))

/** log p(value of `inst` | its parents) under `env`. */
export function instanceLogDensity(em: ExpandedModel, inst: Instance, env: Env): number {
  return asNumber(conditionalOf(em, inst, env).logProb(env(inst.key)))
}

const stochastic = (inst: Instance) => inst.node.role === 'latent' || inst.node.role === 'observed'

/** log p(latent, data): the sum of every stochastic instance's log conditional density. */
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

/** The stochastic instances an instance depends on directly, looking through deterministic nodes (all candidates of an `at`). */
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

/** Parents and children of every stochastic instance (looking through deterministic nodes). */
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
 * `aifn-compute/graph/structured` on `expandModel(m, bindings).graph`.
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

/** The number of values of a discrete node (Bernoulli 2, Categorical K, Binomial n + 1), or null. */
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

/** Ancestral sampling: a value for every latent and unobserved instance, in declaration order. */
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

/** Collect the values of one node's instances into nested arrays by plate index (e.g. sampled data). */
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
