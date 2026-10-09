/**
 * Gibbs sampling (Geman & Geman 1984; Gelfand & Smith 1990), built from the Markov blanket: each variable is redrawn
 * from its full conditional given the others.
 *
 * - On a discrete factor graph, the conditional of $x_v$ is the normalised product of the factors that mention it.
 * - On a model description, a discrete latent variable's conditional is enumerated over its values (its own density
 *   times its children's), and a continuous one's comes from a conjugate update: Beta with Bernoulli or Binomial
 *   children, Dirichlet with Categorical children (through `at` selections too), Normal (unknown mean) with Normal
 *   children of known sd, and Gamma with Poisson children (Bishop 2006, "Pattern Recognition and Machine Learning",
 *   §2.1–2.3). A latent variable without such a conditional is an error at `init`.
 */

import { beta as betaDraw, dirichlet, gammaVariate } from 'aifn-compute/probability/samplers'
import type { Index, Size, Status } from 'aifn-compute/foundation/contracts'
import { categorical, child, integers, normal, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { stridesOf, valuesOf, type DiscreteFactorGraph } from 'aifn-compute/inference/model'
import {
  argValue,
  cardinalityOf,
  dependencyMaps,
  distOf,
  environment,
  expandModel,
  instanceLogDensity,
  logJoint,
  resolveRef,
  sampleModel,
  type Bindings,
  type Env,
  type ExpandedModel,
  type Instance,
  type Model,
  type NodeRef,
  type NodeValue,
} from 'aifn-compute/inference/model'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── Gibbs on a discrete factor graph ────────────────────────────────────────────────────────────────────────────────

/** Options of {@link factorGraphGibbs}. */
export interface FactorGraphGibbsOptions {
  /** Starting values (default: drawn uniformly). */
  initial?: ArrayLike<number>
  /** Observed values, held fixed. */
  evidence?: Readonly<Record<number, number>>
  /** `variable`: one variable per step; `sweep`: every free variable once per step (default). */
  granularity?: 'variable' | 'sweep'
  /** The visiting order (default $0, \dots, V - 1$); observed variables are dropped from it. */
  order?: readonly number[]
}

/** The state of Gibbs sampling on a discrete factor graph. */
export interface FactorGraphGibbsState extends Status {
  /** The factor graph sampled. */
  graph: DiscreteFactorGraph
  /** Whether a step updates one variable or completes a sweep. */
  granularity: 'variable' | 'sweep'
  /** The free variables in visiting order (the evidence removed). */
  order: Index[]
  /** The current values (int32), evidence included. */
  assignment: Tensor
  /** Completed sweeps. */
  sweep: Size
  /** The position in `order` of the next variable to update. */
  position: Index
  /** The variable updated last ($-1$ before the first draw). */
  variable: Index
  /** The conditional the last variable was drawn from (empty before the first draw). */
  conditional: Tensor
  /** Per variable, how often each value was held at the end of a sweep; divide by `sweep` for marginal estimates. */
  counts: Tensor[]
}

/**
 * $p(x_v = \cdot \mid \text{rest})$ from the factors mentioning $v$: the product of their entries at the current
 * values of the other variables, normalised (NaN when every value has zero potential).
 *
 * @param g The factor graph.
 * @param a The current assignment, one value per variable; only the values of $v$'s neighbours are read.
 * @param v The variable whose conditional is wanted.
 * @returns The conditional distribution of $x_v$, $K_v$ probabilities.
 */
function discreteConditional(g: DiscreteFactorGraph, a: Int32Array, v: number): Float64Array {
  const p = new Float64Array(g.cardinalities[v]).fill(1)
  for (const f of g.factors) {
    const i = f.scope.indexOf(v)
    if (i < 0) continue
    const strides = stridesOf(f.table.shape)
    const t = valuesOf(f.table)
    let base = 0
    f.scope.forEach((u, j) => j !== i && (base += a[u] * strides[j]))
    for (let x = 0; x < p.length; x++) p[x] *= t[base + x * strides[i]]
  }
  let z = 0
  for (const x of p) z += x
  return p.map((x) => x / z)
}

/**
 * Gibbs sampling on a discrete factor graph as a traceable algorithm. Starting values are drawn uniformly from the
 * init stream (`child(s, v)` for variable $v$) unless given; the draw of variable $v$ in a step uses
 * `child(ctx.stream, v)`, so runs are reproducible and `seek` agrees with `run`. The value counts behind
 * `gibbsMarginals` are taken at the end of each sweep.
 *
 * @param graph The discrete factor graph, its potentials non-negative.
 * @param o The `initial` values, the `evidence` held fixed, the `granularity` of a step and the visiting `order`.
 * @returns The sampler as an algorithm, run with no start: `run(factorGraphGibbs(graph), undefined, sweeps)`.
 *
 * @example Rain given wet grass
 * // p(rain) = 0.2, and the grass is wet with probability 0.1 without rain and 0.8 with it.
 * const graph = {
 *   cardinalities: [2, 2],
 *   names: ['rain', 'wet'],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const s = run(factorGraphGibbs(graph, { evidence: { 1: 1 } }), undefined, 300)
 * print('p(rain | wet) by Gibbs =', gibbsMarginals(s)[0])
 * print('exact =', [0.08 / 0.24, 0.16 / 0.24])
 *
 * @example One variable at a time
 * // p(rain) = 0.2, and the grass is wet with probability 0.1 without rain and 0.8 with it.
 * const graph = {
 *   cardinalities: [2, 2],
 *   names: ['rain', 'wet'],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const s = run(factorGraphGibbs(graph, { granularity: 'variable', initial: [1, 0] }), undefined, 1)
 * print('updated variable', s.variable, 'from', s.conditional, 'now', s.assignment)
 */
export function factorGraphGibbs(
  graph: DiscreteFactorGraph,
  o: FactorGraphGibbsOptions = {},
): Algorithm<void, FactorGraphGibbsState> {
  return {
    name: 'factor-graph-gibbs',
    init: (_, stream) => {
      const V = graph.cardinalities.length
      const a = new Int32Array(V)
      for (let v = 0; v < V; v++) a[v] = o.initial ? o.initial[v] : integers(child(stream, v), graph.cardinalities[v])
      const evidence = o.evidence ?? {}
      for (const [k, x] of Object.entries(evidence)) a[Number(k)] = x
      const order = (o.order ?? a.map((_, v) => v)).filter((v) => evidence[v] === undefined)
      return {
        t: 0,
        graph,
        granularity: o.granularity ?? 'sweep',
        order: Array.from(order),
        assignment: fromData(a, [V]),
        sweep: 0,
        position: 0,
        variable: -1,
        conditional: fromData(new Float64Array(0), [0]),
        counts: graph.cardinalities.map((k) => fromData(new Float64Array(k), [k])),
      }
    },
    step: (s, ctx) => {
      const a = Int32Array.from(s.assignment.data)
      const n = s.granularity === 'sweep' ? s.order.length - s.position : 1
      let position = s.position
      let variable = s.variable
      let conditional = s.conditional
      for (let i = 0; i < n; i++) {
        variable = s.order[position]
        const p = discreteConditional(s.graph, a, variable)
        a[variable] = categorical(child(ctx.stream, variable), p)
        conditional = fromData(p, [p.length])
        position++
      }
      let { sweep, counts } = s
      if (position >= s.order.length) {
        position = 0
        sweep += 1
        counts = counts.map((c, v) => {
          const d = Float64Array.from(c.data)
          d[a[v]] += 1
          return fromData(d, c.shape)
        })
      }
      return { ...s, t: s.t + 1, assignment: fromData(a, [a.length]), sweep, position, variable, conditional, counts }
    },
  }
}

/**
 * Marginal estimates from a Gibbs state: the fraction of completed sweeps each variable spent at each value.
 *
 * @param s A state of `factorGraphGibbs`.
 * @returns One vector of $K_v$ frequencies per variable (zeros before the first sweep; an observed variable's sits at
 *   its value).
 *
 * @example Both marginals of rain and wet grass
 * // p(rain) = 0.2, and the grass is wet with probability 0.1 without rain and 0.8 with it.
 * const graph = {
 *   cardinalities: [2, 2],
 *   names: ['rain', 'wet'],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const marginals = gibbsMarginals(run(factorGraphGibbs(graph), undefined, 400))
 * print('p(rain) =', marginals[0], 'exact [0.8, 0.2]')
 * print('p(wet) =', marginals[1], 'exact [0.76, 0.24]')
 */
export function gibbsMarginals(s: FactorGraphGibbsState): Tensor[] {
  return s.counts.map((c) =>
    fromData(
      c.data.map((x) => x / Math.max(s.sweep, 1)),
      c.shape,
    ),
  )
}

// ── Gibbs on a model description ────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link modelGibbs}. */
export interface ModelGibbsOptions {
  /** Starting values by instance key (default: an ancestral draw from the prior with the data held fixed). */
  initial?: Readonly<Record<string, NodeValue>>
  /** `variable`: one latent instance per step; `sweep`: all of them (default). */
  granularity?: 'variable' | 'sweep'
}

/** The state of Gibbs sampling on a model. */
export interface ModelGibbsState extends Status {
  /** The model expanded against its bindings: one instance per plate index. */
  expanded: ExpandedModel
  /** Whether a step updates one instance or completes a sweep. */
  granularity: 'variable' | 'sweep'
  /** Instance keys in visiting (declaration) order: the latent ones and the observed ones without data. */
  order: string[]
  /** The kind of conditional of each instance in `order`. */
  kinds: Record<string, ConditionalKind>
  /** Current values of the instances in `order`, by key. */
  values: Readonly<Record<string, NodeValue>>
  /** Completed sweeps. */
  sweep: Size
  /** The position in `order` of the next instance to update. */
  position: Index
  /** The instance updated last. */
  updated: string | null
  /** log p(latent, data) at the end of the last step. */
  logJoint: number
}

/** How a latent instance is redrawn. */
export type ConditionalKind = 'enumerate' | 'beta' | 'dirichlet' | 'normal' | 'gamma'

/**
 * The children of `key` with the arguments of each that are direct references able to select `key` (not ones that
 * index into its value).
 *
 * @param em The expanded model.
 * @param key The instance whose children are examined.
 * @param children The keys of its children.
 * @returns For each child, its instance and `hits`: the index `i` and resolved reference `r` of each such argument.
 */
function directChildren(em: ExpandedModel, key: string, children: readonly string[]) {
  return children.map((c) => {
    const inst = em.byKey.get(c)!
    const hits = distOf(inst).args.flatMap((a, i) => {
      if (typeof a !== 'object' || a === null || !('kind' in a) || a.kind !== 'ref') return []
      const r = resolveRef(em, inst, a as NodeRef)
      return r.candidates.includes(key) && !r.indexesValue ? [{ i, r }] : []
    })
    return { inst, hits }
  })
}

/** For each conjugate prior family, its child families and the argument of the child the prior must fill. */
const CONJUGATE: Record<string, Partial<Record<string, number>>> = {
  // prior family → child family → which child argument must be the prior's variable
  Beta: { Bernoulli: 0, Binomial: 1 },
  Dirichlet: { Categorical: 0 },
  Normal: { Normal: 0 },
  Gamma: { Poisson: 0 },
}
/** The conditional kind of each conjugate prior family. */
const KIND: Record<string, ConditionalKind> = { Beta: 'beta', Dirichlet: 'dirichlet', Normal: 'normal', Gamma: 'gamma' }

/**
 * How an instance is redrawn: by enumeration when it is discrete, otherwise by the conjugate update of its family.
 * Throws `DomainError` for a continuous family with no conjugate update, or a child that is not a conjugate child
 * (another family, or the instance in another argument or in more than one).
 *
 * @param em The expanded model.
 * @param inst The instance to classify.
 * @param children The keys of its children.
 * @returns Its conditional kind.
 */
function classify(em: ExpandedModel, inst: Instance, children: readonly string[]): ConditionalKind {
  if (cardinalityOf(em, inst) !== null) return 'enumerate'
  const family = distOf(inst).family
  const rule = CONJUGATE[family]
  if (!rule) throw new DomainError('gibbs', `gibbs: no conjugate conditional for ${inst.key} (${family})`)
  for (const { inst: child, hits } of directChildren(em, inst.key, children)) {
    const want = rule[distOf(child).family]
    if (want === undefined || hits.length !== 1 || hits[0].i !== want)
      throw new DomainError('gibbs', `gibbs: ${child.key} is not a conjugate child of ${inst.key} (${family})`)
  }
  return KIND[family]
}

/**
 * A value as a number: the number itself, or a tensor's first entry.
 *
 * @param v The value.
 * @returns The number.
 */
const num = (v: NodeValue): number => (typeof v === 'number' ? v : toFlat(v)[0])

/**
 * Draw a new value of `inst` from its full conditional: enumerated over its values (its own density times its
 * children's), or by the conjugate update of `kind` from the children that currently select it.
 *
 * @param em The expanded model.
 * @param inst The instance to redraw.
 * @param kind Its conditional kind, from `classify`.
 * @param children The keys of its children.
 * @param values The current values of every instance; read, and during enumeration `inst`'s entry is overwritten
 *   with each candidate value in turn (the caller sets the new value after).
 * @param s The stream of this draw.
 * @returns The new value.
 */
function redraw(
  em: ExpandedModel,
  inst: Instance,
  kind: ConditionalKind,
  children: readonly string[],
  values: Map<string, NodeValue>,
  s: Stream,
): NodeValue {
  const env: Env = environment(em, values)
  const args = distOf(inst).args.map((a) => argValue(em, inst, a, env))
  const active = directChildren(em, inst.key, children).filter(({ hits }) =>
    hits.every(({ r }) => r.choose(env) === inst.key),
  )
  const xs = active.map(({ inst: c }) => ({ c, x: env(c.key) }))
  switch (kind) {
    case 'enumerate': {
      const K = cardinalityOf(em, inst)!
      const logs = new Float64Array(K)
      const kids = children.map((c) => em.byKey.get(c)!)
      for (let k = 0; k < K; k++) {
        values.set(inst.key, k)
        logs[k] = instanceLogDensity(em, inst, env) + kids.reduce((t, c) => t + instanceLogDensity(em, c, env), 0)
      }
      const top = Math.max(...logs)
      return categorical(
        s,
        logs.map((l) => Math.exp(l - top)),
      )
    }
    case 'beta': {
      let [a, b] = [num(args[0]), num(args[1])]
      for (const { c, x } of xs) {
        const n = distOf(c).family === 'Binomial' ? num(argValue(em, c, distOf(c).args[0], env)) : 1
        a += num(x)
        b += n - num(x)
      }
      return betaDraw(s, a, b)
    }
    case 'dirichlet': {
      const conc = isTensor(args[0])
        ? Float64Array.from(toFlat(args[0] as Tensor))
        : new Float64Array(num(args[1])).fill(num(args[0]))
      for (const { x } of xs) conc[Math.round(num(x))] += 1
      return dirichlet(s, conc)
    }
    case 'normal': {
      let precision = 1 / num(args[1]) ** 2
      let shift = num(args[0]) * precision
      for (const { c, x } of xs) {
        const sd = num(argValue(em, c, distOf(c).args[1], env))
        precision += 1 / (sd * sd)
        shift += num(x) / (sd * sd)
      }
      return normal(s, shift / precision, Math.sqrt(1 / precision))
    }
    case 'gamma': {
      let [shape, rate] = [num(args[0]), num(args[1])]
      for (const { x } of xs) {
        shape += num(x)
        rate += 1
      }
      return gammaVariate(s, shape, 1 / rate)
    }
  }
}

/**
 * Gibbs sampling on a model description against `bindings` as a traceable algorithm. Latent instances, and observed
 * ones the bindings give no data for, are visited in declaration order; the start is an ancestral draw from the init
 * stream (data held fixed) unless given, and the draw for instance `key` in a step uses `child(ctx.stream, key)`.
 * `init` throws `DomainError` for an instance with no enumerable or conjugate conditional.
 *
 * @param model The model description, as `model(...)` builds it.
 * @param bindings The data on observed nodes and the sizes of plates (default none).
 * @param o The `initial` values by instance key, and the `granularity` of a step.
 * @returns The sampler as an algorithm, run with no start: `run(modelGibbs(model, bindings), undefined, sweeps)`.
 *
 * @example A Beta–Bernoulli coin
 * // p ~ Beta(2, 2) and ten flips with eight heads: the posterior mean is (2 + 8) / (4 + 10).
 * const p = { kind: 'ref', node: 'p' }
 * const coin = {
 *   kind: 'graph', name: 'coin', directed: true, nodes: 2, labels: ['p', 'x'],
 *   edges: [{ from: 0, to: 1, directed: true }],
 *   attributes: [
 *     { name: 'p', role: 'latent', group: null, data: { dist: { family: 'Beta', args: [2, 2] } } },
 *     { name: 'x', role: 'observed', group: 'flips', data: { dist: { family: 'Bernoulli', args: [p] } } },
 *   ],
 *   groups: [{ name: 'flips', kind: 'plate', size: 'n', index: ['f'], parent: null }],
 *   sizes: ['n'],
 * }
 * const x = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
 * const tr = trace(modelGibbs(coin, { data: { x } }), undefined, 300, { record: { p: (s) => s.values.p } })
 * const draws = toFlat(tr.series.p).slice(1)
 * print('conditional of p:', tr.final.kinds.p)
 * print('posterior mean =', draws.reduce((a, b) => a + b) / draws.length, 'exact', 10 / 14)
 *
 * @example Two means with mixture indicators: enumeration and conjugacy together
 * const muZ = { kind: 'ref', node: 'mu', select: 'z' }
 * const mixture = {
 *   kind: 'graph', name: 'two means', directed: true, nodes: 3, labels: ['mu', 'z', 'x'],
 *   edges: [{ from: 0, to: 2, directed: true }, { from: 1, to: 2, directed: true }],
 *   attributes: [
 *     { name: 'mu', role: 'latent', group: 'components', data: { dist: { family: 'Normal', args: [0, 10] } } },
 *     { name: 'z', role: 'latent', group: 'points', data: { dist: { family: 'Categorical', args: [[0.5, 0.5]] } } },
 *     { name: 'x', role: 'observed', group: 'points', data: { dist: { family: 'Normal', args: [muZ, 1] } } },
 *   ],
 *   groups: [
 *     { name: 'components', kind: 'plate', size: 2, index: ['c'], parent: null },
 *     { name: 'points', kind: 'plate', size: 'n', index: ['p'], parent: null },
 *   ],
 *   sizes: ['n'],
 * }
 * const s = run(modelGibbs(mixture, { data: { x: [-5.1, -4.8, -5.3, 4.9, 5.2, 5.0] } }), undefined, 100)
 * print('kinds =', s.kinds)
 * print('means =', s.values['mu[0]'], s.values['mu[1]'])
 * print('indicators =', [0, 1, 2, 3, 4, 5].map((i) => s.values[`z[${i}]`]))
 */
export function modelGibbs(
  model: Model,
  bindings: Bindings = {},
  o: ModelGibbsOptions = {},
): Algorithm<void, ModelGibbsState> {
  return {
    name: 'model-gibbs',
    init: (_, s) => {
      const em = expandModel(model, bindings)
      const { children } = dependencyMaps(em)
      const order = em.instances
        .filter((i) => i.node.role === 'latent' || (i.node.role === 'observed' && !em.fixed.has(i.key)))
        .map((i) => i.key)
      const kinds: Record<string, ConditionalKind> = {}
      for (const key of order) kinds[key] = classify(em, em.byKey.get(key)!, children.get(key) ?? [])
      const drawn = sampleModel(s, model, bindings)
      const values: Record<string, NodeValue> = {}
      for (const key of order) values[key] = o.initial?.[key] ?? drawn.get(key)!
      return {
        t: 0,
        expanded: em,
        granularity: o.granularity ?? 'sweep',
        order,
        kinds,
        values,
        sweep: 0,
        position: 0,
        updated: null,
        logJoint: logJoint(em, values),
      }
    },
    step: (s, ctx) => {
      const em = s.expanded
      const { children } = childrenCache(em)
      const values = new Map(Object.entries(s.values))
      const n = s.granularity === 'sweep' ? s.order.length - s.position : 1
      let position = s.position
      let updated = s.updated
      for (let i = 0; i < n; i++) {
        const key = s.order[position]
        const v = redraw(em, em.byKey.get(key)!, s.kinds[key], children.get(key) ?? [], values, child(ctx.stream, key))
        values.set(key, v)
        updated = key
        position++
      }
      let sweep = s.sweep
      if (position >= s.order.length) {
        position = 0
        sweep++
      }
      const out = Object.fromEntries(values)
      return { ...s, t: s.t + 1, values: out, sweep, position, updated, logJoint: logJoint(em, out) }
    },
  }
}

/** The dependency maps of each expanded model, computed once per model. */
const cache = new WeakMap<ExpandedModel, ReturnType<typeof dependencyMaps>>()
/**
 * The dependency maps of an expanded model, from the cache or computed and stored.
 *
 * @param em The expanded model, the cache's key.
 * @returns Its dependency maps (`children` among them).
 */
function childrenCache(em: ExpandedModel) {
  let c = cache.get(em)
  if (!c) cache.set(em, (c = dependencyMaps(em)))
  return c
}
