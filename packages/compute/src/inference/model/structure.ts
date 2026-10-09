/**
 * From a model to its structures, all structured graphs of `aifn-compute/graph/structured`: the factor graph (one
 * factor per stochastic instance, over the instance and its stochastic parents; Kschischang, Frey & Loeliger 2001),
 * the discrete factor graph the exact and message-passing engines run on (observed values clamped), and diagram data
 * for plate notation and factor graphs (`toDiagram`).
 */

import type { Index } from 'aifn-compute/foundation/contracts'
import {
  structuredGraph,
  toDiagram,
  type DiagramData,
  type DiagramOptions,
  type EdgeSpec,
  type StructuredGraph,
  type StructuredNode,
} from 'aifn-compute/graph/structured'
import {
  bipartiteGraph,
  discreteFactor,
  discreteFactorGraph,
  type DiscreteFactor,
  type DiscreteFactorGraph,
} from './factors'
import {
  cardinalityOf,
  dependencyMaps,
  environment,
  expandModel,
  instanceLogDensity,
  type Bindings,
  type ExpandedModel,
  type Model,
  type NodeValue,
} from './model'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * What a node of a model's factor graph carries: the instance key and model node of a variable (with its number of
 * values when discrete), or for a factor the key of the instance whose conditional it is, its variables (node
 * indices, the child first) and the child.
 */
export interface FactorNodeData {
  /** A variable's instance key, or for a factor the key of the instance whose conditional it is. */
  key: string
  /** The name of the model node the variable (or the factor's child) is a copy of. */
  node: string
  /** Variables: number of values for a discrete node, null otherwise. */
  cardinality?: number | null
  /** Factors: the variables (node indices), the child first, then its stochastic parents. */
  scope?: readonly Index[]
  /** Factors: the variable whose conditional this is. */
  child?: Index
}

/**
 * A model expanded into its factor graph: variables are nodes $0, \dots, V - 1$ for $V$ the count `variables` (role
 * `latent` or `observed`), factors the nodes after them (role `factor`). Edges run from each parent to the factor and
 * from the factor to the child (undirected), so a layered layout reads them left to right; nodes keep their plate so a
 * diagram draws plates around the copies.
 */
export interface ModelFactorGraph extends StructuredGraph<FactorNodeData> {
  /** The expanded model the graph was built from. */
  expanded: ExpandedModel
  /** The number of variable nodes (the factors follow them). */
  variables: number
}

/**
 * A model expanded against bindings, or an expanded model as it is.
 *
 * @param m The model, or a model already expanded (then `b` is ignored).
 * @param b What to expand a model against.
 */
const expand = (m: Model | ExpandedModel, b: Bindings): ExpandedModel => ('instances' in m ? m : expandModel(m, b))

/**
 * A node's TeX label with an instance's index appended as a subscript. A label that already has a subscript is grouped
 * first, so `\theta_d` at index 0 reads `{\theta_d}_{0}` (KaTeX rejects the double subscript `\theta_d_{0}`); a label
 * between dollar signs is handled inside the delimiters.
 *
 * @param label The model node's label, bare TeX or between dollar signs.
 * @param index The instance's plate indices, outermost first, joined by commas in the subscript.
 * @returns The label with the subscript, or `label` itself for an empty index.
 */
export function indexedLabel(label: string, index: readonly (string | number)[]): string {
  if (index.length === 0) return label
  const math = label.length > 1 && label.startsWith('$') && label.endsWith('$')
  const tex = math ? label.slice(1, -1) : label
  const base = tex.includes('_') ? `{${tex}}` : tex
  const out = `${base}_{${index.join(',')}}`
  return math ? `$${out}$` : out
}

/**
 * Expand a model into its factor graph: one variable per stochastic instance, and one factor per variable, its
 * conditional, over the variable and its stochastic parents. Deterministic nodes are folded into the factors of their
 * stochastic children; an `at` reference makes the factor depend on every instance it could select, and on the
 * selector. A variable is `observed` when the bindings give its value, and `latent` otherwise. Parameters do not
 * appear.
 *
 * @param m The model, or a model already expanded (then `bindings` is ignored).
 * @param bindings What to expand a model against: sizes, constants and data.
 * @returns The factor graph; factor $k$ is node `variables + k` and is the conditional of variable $k$.
 *
 * @example A coin and two flips: three variables, three factors
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const fg = toFactorGraph(coins, { data: { x: [1, 0] } })
 * print('variables:', fg.variables)
 * print('nodes:', fg.attributes.map((n) => `${n.name} (${n.role})`))
 * print('edges:', fg.edges.map((e) => `${fg.attributes[e.from].name} - ${fg.attributes[e.to].name}`))
 */
export function toFactorGraph(m: Model | ExpandedModel, bindings: Bindings = {}): ModelFactorGraph {
  const em = expand(m, bindings)
  const { parents } = dependencyMaps(em)
  const nodes: StructuredNode<FactorNodeData>[] = []
  const index = new Map<string, number>()
  for (const inst of em.instances) {
    if (inst.node.role !== 'latent' && inst.node.role !== 'observed') continue
    index.set(inst.key, nodes.length)
    nodes.push({
      name: inst.key,
      role: em.fixed.has(inst.key) ? 'observed' : 'latent',
      group: inst.node.group,
      label: inst.node.label ? indexedLabel(inst.node.label, inst.index) : inst.key,
      source: inst.node.name,
      index: inst.index,
      data: { key: inst.key, node: inst.node.name, cardinality: cardinalityOf(em, inst) },
    })
  }
  const variables = nodes.length
  const edges: EdgeSpec[] = []
  for (let child = 0; child < variables; child++) {
    const v = nodes[child]
    const scope = [child, ...parents.get(v.name)!.map((p) => index.get(p)!)]
    const name = `p(${v.name})`
    nodes.push({
      name,
      role: 'factor',
      group: v.group,
      data: { key: v.name, node: v.data!.node, scope, child },
    })
    for (const p of scope.slice(1)) edges.push({ from: nodes[p].name, to: name, directed: false })
    edges.push({ from: name, to: v.name, directed: false })
  }
  const graph = structuredGraph<FactorNodeData>({ name: em.model.name, nodes, edges, groups: em.model.groups })
  return { ...graph, unrolled: true, expanded: em, variables }
}

/**
 * The factors of a model's factor graph: for each, its node index and its data (scope, child).
 *
 * @param fg The factor graph, from `toFactorGraph`.
 * @returns One record per factor, in node order: `node`, its index in the graph; `scope`, its variables (the child
 *   first); `child`, the variable whose conditional it is; and `key`, the child's instance key.
 *
 * @example The conditionals of a coin and two flips
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const fg = toFactorGraph(coins, { data: { x: [1, 0] } })
 * print(factorsOf(fg).map((f) => `node ${f.node}: p(${f.key}), scope ${f.scope}`))
 */
export function factorsOf(fg: ModelFactorGraph): { node: Index; scope: readonly Index[]; child: Index; key: string }[] {
  return fg.attributes.slice(fg.variables).map((n, k) => ({
    node: fg.variables + k,
    scope: n.data!.scope!,
    child: n.data!.child!,
    key: n.data!.key,
  }))
}

/** A discrete factor graph over a model's latent variables, with the data clamped. */
export interface ModelDiscreteGraph {
  /** The factor graph over the latent variables; its variable names are their instance keys. */
  graph: DiscreteFactorGraph
  /** The instance key of each variable of `graph`. */
  keys: string[]
  /** The sum of the log-densities of factors with no latent variable (observed given constants): add to $\log Z$. */
  logConstant: number
}

/**
 * Tabulate a model with discrete latent variables as a discrete factor graph: each factor
 * $p(\text{child} \mid \text{parents})$ becomes a table over its latent variables, with observed values fixed. Its
 * name is `p(<key>)` for the child's key. $\log Z$ of the result plus `logConstant` is $\log p(\text{data})$. Throws
 * `DomainError` when a latent variable is not discrete with a finite number of values (see `cardinalityOf`).
 *
 * @param m The model, or a model already expanded (then `bindings` is ignored).
 * @param bindings What to expand a model against; the data fixes the observed variables.
 * @returns The graph, the instance key of each of its variables, and `logConstant`.
 *
 * @example Rain makes the grass wet: $p(\text{wet} = 1) = 0.8 \times 0.1 + 0.2 \times 0.9 = 0.26$
 * const wet = model('wet grass', (m) => {
 *   const rain = m.variable('rain', dist.Bernoulli(0.2))
 *   const pWet = m.deterministic('pWet', 'index', [[0.1, 0.9], rain])
 *   m.observed('wet', dist.Bernoulli(pWet))
 *   m.observed('coin', dist.Bernoulli(0.5))
 * })
 * const { graph, keys, logConstant } = toDiscreteFactorGraph(wet, { data: { wet: 1, coin: 1 } })
 * print('variables:', keys)
 * print(graph.factors[0].name, 'over rain:', graph.factors[0].table)
 * print(graph.factors[1].name, 'over rain:', graph.factors[1].table)
 * print('log constant (the coin, log 0.5):', logConstant)
 * const Z = factorMarginalise(factorProductAll(graph.factors, graph.cardinalities), [0]).table
 * print('Z:', Z, 'log p(data):', Math.log(toFlat(Z)[0]) + logConstant, 'log(0.26 × 0.5):', Math.log(0.13))
 *
 * @example A continuous latent variable throws
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * try {
 *   toDiscreteFactorGraph(coins, { data: { x: [1, 0] } })
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function toDiscreteFactorGraph(m: Model | ExpandedModel, bindings: Bindings = {}): ModelDiscreteGraph {
  const fg = toFactorGraph(m, bindings)
  const em = fg.expanded
  const vars = fg.attributes.slice(0, fg.variables)
  const latent = vars.map((v, i) => (v.role === 'observed' ? -1 : i)).filter((i) => i >= 0)
  for (const i of latent)
    if (vars[i].data!.cardinality == null)
      throw new DomainError(
        'toDiscreteFactorGraph',
        `toDiscreteFactorGraph: ${vars[i].name} is not discrete with finitely many values`,
      )
  const position = new Map(latent.map((v, i) => [v, i]))
  const cards = latent.map((i) => vars[i].data!.cardinality!)
  const factors: DiscreteFactor[] = []
  let logConstant = 0
  for (const f of factorsOf(fg)) {
    const scope = f.scope.filter((v) => position.has(v))
    const child = em.byKey.get(f.key)!
    if (scope.length === 0) {
      logConstant += instanceLogDensity(em, child, environment(em, new Map()))
      continue
    }
    const values = new Map<string, NodeValue>()
    const env = environment(em, values)
    factors.push(
      discreteFactor(
        scope.map((v) => position.get(v)!),
        cards,
        (a) => {
          scope.forEach((v, i) => values.set(vars[v].name, a[i]))
          return Math.exp(instanceLogDensity(em, child, env))
        },
        `p(${f.key})`,
      ),
    )
  }
  const keys = latent.map((i) => vars[i].name)
  return { graph: discreteFactorGraph(cards, factors, keys), keys, logConstant }
}

// ── Diagrams ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Plate notation for a model: `toDiagram` of the compact model. Latent variables are circles, observed ones shaded,
 * deterministic ones dashed, parameters small; links run from each node's parents; plates and chains are groups
 * labelled with their size; a chain's dependence on the previous copy is a loop labelled $t-1$.
 *
 * @param m The model (compact, not expanded).
 * @param options Diagram options of `toDiagram`: positions, labels, a node to highlight.
 * @returns The diagram data, for the lab's diagram component to draw.
 *
 * @example A chain with a parameter, and the loop to the previous state
 * const hmm = model('hidden Markov model', (m) => {
 *   const A = m.constant('A', [[0.9, 0.1], [0.2, 0.8]])
 *   const time = m.chain('time', 'T')
 *   const z = time.variable('z', dist.Categorical([0.5, 0.5]), { next: (prev) => dist.Categorical(A.at(prev)) })
 *   time.observed('x', dist.Normal(z, 1))
 * })
 * const d = toPlateDiagram(hmm)
 * print('nodes:', d.nodes.map((n) => `${n.id}${n.filled ? ' (shaded)' : ''}${n.small ? ' (small)' : ''}`))
 * print('edges:', d.edges.map((e) => `${e.from} -> ${e.to}${e.label ? ` labelled ${e.label}` : ''}`))
 * print('plates:', d.groups.map((g) => `${g.label} around ${g.around}`))
 */
export function toPlateDiagram(m: Model, options: DiagramOptions = {}): DiagramData {
  return toDiagram(m, options)
}

/** Options of {@link toFactorDiagram}: diagram options, and the bindings to expand a model against. */
export interface FactorDiagramOptions extends DiagramOptions {
  /** What to expand a model against (ignored for an expanded model or a discrete factor graph). */
  bindings?: Bindings
}

/**
 * A factor-graph diagram: variables as circles (observed ones shaded), factors as small squares, undirected links.
 * For a model, node ids are instance keys and `p(<key>)` for factors, and plates are drawn around the copies; for a
 * discrete factor graph, ids are `x<i>` and `f<k>` (see `bipartiteGraph`). `highlight` marks a variable, its Markov
 * blanket and the factors joining them.
 *
 * @param target A model (expanded against `options.bindings`), an expanded model, or a discrete factor graph.
 * @param options The diagram options of `toDiagram`, and `bindings` for a model.
 * @returns The diagram data, for the lab's diagram component to draw.
 *
 * @example A coin and two flips, highlighting the first flip
 * const coins = model('coin flips', (m) => {
 *   const p = m.variable('p', dist.Beta(2, 2))
 *   m.plate('flips', 'N').observed('x', dist.Bernoulli(p))
 * })
 * const d = toFactorDiagram(coins, { bindings: { data: { x: [1, 0] } }, highlight: 'x[0]' })
 * print('nodes:', d.nodes.map((n) => `${n.id} (${n.shape}, ${n.state})`))
 * print('plates:', d.groups.map((g) => `${g.label} around ${g.around}`))
 *
 * @example A discrete factor graph
 * const c = [2, 2]
 * const g = discreteFactorGraph(c, [discreteFactor([0], c, [0.6, 0.4]), discreteFactor([0, 1], c, [1, 2, 2, 1])])
 * print('nodes:', toFactorDiagram(g).nodes.map((n) => `${n.id} (${n.shape})`))
 */
export function toFactorDiagram(
  target: Model | ExpandedModel | DiscreteFactorGraph,
  options: FactorDiagramOptions = {},
): DiagramData {
  const { bindings = {}, ...rest } = options
  const graph = 'cardinalities' in target ? bipartiteGraph(target) : toFactorGraph(target, bindings)
  return toDiagram(graph, rest)
}
