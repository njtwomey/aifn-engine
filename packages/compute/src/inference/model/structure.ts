/**
 * From a model to its structures, all structured graphs of `aifn-compute/graph/structured`: the factor graph (one factor per
 * stochastic instance, over the instance and its stochastic parents; Kschischang, Frey & Loeliger 2001), the discrete
 * factor graph the exact and message-passing engines run on (observed values clamped), and diagram data for plate
 * notation and factor graphs (`toDiagram`).
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
  key: string
  node: string
  /** Variables: number of values for a discrete node, null otherwise. */
  cardinality?: number | null
  /** Factors: the variables (node indices), the child first, then its stochastic parents. */
  scope?: readonly Index[]
  /** Factors: the variable whose conditional this is. */
  child?: Index
}

/**
 * A model expanded into its factor graph: variables are nodes 0 … variables − 1 (role `latent` or `observed`), factors
 * the nodes after them (role `factor`). Edges run parent → factor → child (undirected), so a layered layout reads
 * them left to right; nodes keep their plate so a diagram draws plates around the copies.
 */
export interface ModelFactorGraph extends StructuredGraph<FactorNodeData> {
  expanded: ExpandedModel
  /** The number of variable nodes (the factors follow them). */
  variables: number
}

const expand = (m: Model | ExpandedModel, b: Bindings): ExpandedModel => ('instances' in m ? m : expandModel(m, b))

/**
 * A node's TeX label with an instance's index appended as a subscript. A label that already has a subscript is grouped
 * first, so `\theta_d` at index 0 reads `{\theta_d}_{0}` (KaTeX rejects the double subscript `\theta_d_{0}`); a label
 * in `$…$` is handled inside the delimiters.
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
 * Expand a model into its factor graph. Deterministic nodes are folded into the factors of their stochastic
 * children; an `at` reference makes the factor depend on every instance it could select, and on the selector.
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

/** The factors of a model's factor graph: for each, its node index and its data (scope, child). */
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
  graph: DiscreteFactorGraph
  /** The instance key of each variable of `graph`. */
  keys: string[]
  /** Σ of the log-densities of factors with no latent variable (observed given constants): add to log Z. */
  logConstant: number
}

/**
 * Tabulate a model with discrete latent variables as a discrete factor graph: each factor p(child | parents) becomes
 * a table over its latent variables, with observed values fixed. log Z of the result plus `logConstant` is
 * log p(data). Throws when a latent variable is not discrete with a finite number of values.
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
 * labelled with their size; a chain's dependence on the previous copy is a loop labelled `t-1`.
 */
export function toPlateDiagram(m: Model, options: DiagramOptions = {}): DiagramData {
  return toDiagram(m, options)
}

/** Options of {@link toFactorDiagram}: diagram options, and the bindings to expand a model against. */
export interface FactorDiagramOptions extends DiagramOptions {
  bindings?: Bindings
}

/**
 * A factor-graph diagram: variables as circles (observed ones shaded), factors as small squares, undirected links.
 * For a model, node ids are instance keys and `p(<key>)` for factors, and plates are drawn around the copies; for a
 * discrete factor graph, ids are `x<i>` and `f<k>` (see `bipartiteGraph`). `highlight` marks a variable, its Markov
 * blanket and the factors joining them.
 */
export function toFactorDiagram(
  target: Model | ExpandedModel | DiscreteFactorGraph,
  options: FactorDiagramOptions = {},
): DiagramData {
  const { bindings = {}, ...rest } = options
  const graph = 'cardinalities' in target ? bipartiteGraph(target) : toFactorGraph(target, bindings)
  return toDiagram(graph, rest)
}
