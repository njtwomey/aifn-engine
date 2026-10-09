/**
 * Belief propagation on discrete factor graphs: sum-product and max-product (Pearl 1988; Kschischang, Frey &
 * Loeliger 2001, "Factor graphs and the sum-product algorithm", IEEE Trans. Inf. Theory 47(2)), exact on trees in one
 * inward and one outward pass, and loopy BP with flooding or sequential schedules and damping (Murphy, Weiss & Jordan
 * 1999, "Loopy belief propagation for approximate inference: an empirical study"). The Bethe free energy (Yedidia,
 * Freeman & Weiss 2005) gives $\log Z$, exactly on trees.
 *
 * Messages run along the edges of the variable-factor graph, indexed as `factorGraphEdges` lists them: $\mu_{x \to f}$
 * from a variable to a factor and $\mu_{f \to x}$ back, each a float64 vector over the variable's values. Messages are
 * normalised to sum to one, and start uniform. Evidence enters as an indicator on the observed variable. Every step
 * records which messages it updated, so a figure can step through the schedule message by message.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { breadthFirstSearch } from 'aifn-compute/graph/traversal'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import {
  bipartiteGraph,
  isTree,
  factorGraphEdges,
  forEachAssignment,
  valuesOf,
  type DiscreteFactorGraph,
  type FactorGraphEdge,
} from 'aifn-compute/inference/model'
import { DomainError } from 'aifn-compute/foundation/errors'

/** One message update: along edge `edge` (into `factorGraphEdges(graph)`), towards the factor or the variable. */
export interface MessageUpdate {
  /** The edge's index into `factorGraphEdges(graph)`. */
  edge: number
  /** The message's direction: `'factor'` for $\mu_{x \to f}$, `'variable'` for $\mu_{f \to x}$. */
  to: 'factor' | 'variable'
}

/**
 * A message schedule. `tree`: leaves to a root and back (exact on trees and forests, in $2E$ messages for $E$ edges);
 * `flooding`: every factor-to-variable message in parallel, then every variable-to-factor message; `sequential`:
 * variable by variable, its incoming then its outgoing messages; or an explicit list of updates, repeated each sweep.
 */
export type Schedule = 'tree' | 'flooding' | 'sequential' | readonly MessageUpdate[]

/** Options of {@link beliefPropagationSteps}. */
export interface BeliefPropagationOptions {
  /** `'sum'` for marginals, `'max'` for max-marginals (max-product). Default `'sum'`. */
  mode?: 'sum' | 'max'
  /** The message schedule. Default `'tree'` when the graph is a tree or forest, else `'flooding'`. */
  schedule?: Schedule
  /**
   * Weight $\lambda$ of the old message in each update, in $[0, 1)$: the new message is
   * $(1 - \lambda)\,\mu_\text{new} + \lambda\,\mu_\text{old}$. Default 0; outside $[0, 1)$ throws `DomainError`.
   */
  damping?: number
  /** A sweep whose largest message change is below this has converged. Default 1e-9. */
  tolerance?: number
  /**
   * `'message'` (default): one update per step; `'sweep'`: a whole pass of the schedule per step. Flooding is always
   * per sweep.
   */
  granularity?: 'message' | 'sweep'
  /** Observed values, as variable index to value, applied as indicator factors. */
  evidence?: Readonly<Record<number, number>>
}

/**
 * The state of belief propagation. Messages are indexed like `edges`. `t` counts steps (messages or sweeps, by
 * `granularity`); `converged` is set once a whole sweep changes no message by more than `tolerance`.
 */
export interface BeliefPropagationState extends Status {
  /** The factor graph, as given (without the evidence). */
  graph: DiscreteFactorGraph
  /** The edges of the variable-factor graph, as `factorGraphEdges` lists them. */
  edges: FactorGraphEdge[]
  /** Sum-product (`'sum'`) or max-product (`'max'`). */
  mode: 'sum' | 'max'
  /** The weight of the old message in each update. */
  damping: number
  /** The largest message change of a sweep that counts as converged. */
  tolerance: number
  /** Whether a step is one message update or one sweep (always a sweep for flooding). */
  granularity: 'message' | 'sweep'
  /** The updates of one sweep, in order (flooding: all factor-to-variable, then all variable-to-factor). */
  schedule: MessageUpdate[]
  /** Whether the schedule is flooding, whose sweep updates every message from the previous ones. */
  flooding: boolean
  /** $\mu_{x \to f}$, one float64 vector per edge over the variable's values. */
  toFactor: Tensor[]
  /** $\mu_{f \to x}$, one float64 vector per edge over the variable's values. */
  toVariable: Tensor[]
  /** Normalised beliefs $b(x) \propto e(x) \prod_f \mu_{f \to x}(x)$, one per variable, with $e$ the evidence. */
  beliefs: Tensor[]
  /** Position in the schedule (0 at the start of a sweep). */
  position: number
  /** Completed sweeps. */
  sweep: number
  /** The updates made by the last step. */
  updated: MessageUpdate[]
  /** Largest absolute change of a message in the last step. */
  change: number
  /** Largest absolute change of a message in the current sweep so far. */
  sweepChange: number
  /** Largest absolute change of a message in the last full sweep ($\infty$ before the first). */
  lastSweepChange: number
  /** Whether a whole sweep changed no message by more than `tolerance`, or a tree schedule finished its sweep. */
  converged: boolean
  /** True when a message summed to zero (contradictory evidence); it was left unnormalised. */
  degenerate: boolean
  /** Evidence indicator per variable (all ones when unobserved). */
  evidence: Tensor[]
}

/**
 * The uniform message over $k$ values.
 *
 * @param k The number of values of the variable.
 * @returns A float64 vector of $k$ entries $1/k$.
 */
const uniform = (k: number) => fromData(new Float64Array(k).fill(1 / k), [k])

/**
 * Leaves-to-root-and-back order over a forest, from breadth-first search on the bipartite graph: every message
 * towards the root, deepest first, then every message away from it.
 *
 * @param graph The factor graph, a tree or forest.
 * @param edges Its edges, as `factorGraphEdges` lists them.
 * @returns The schedule of $2E$ updates for $E$ edges.
 */
function treeSchedule(graph: DiscreteFactorGraph, edges: FactorGraphEdge[]): MessageUpdate[] {
  const V = graph.cardinalities.length
  const b = bipartiteGraph(graph)
  const bfs = breadthFirstSearch(b)
  const edgeOf = new Map<string, number>()
  edges.forEach((e, k) => edgeOf.set(`${e.variable}:${e.factor}`, k))
  // The message from node u to its parent p in the BFS tree, and back.
  const between = (u: number, p: number): MessageUpdate => {
    const [v, f] = u < V ? [u, p - V] : [p, u - V]
    return { edge: edgeOf.get(`${v}:${f}`)!, to: u < V ? 'factor' : 'variable' }
  }
  const order = Array.from(bfs.order.data)
  const up: MessageUpdate[] = []
  for (let i = order.length - 1; i >= 0; i--) {
    const u = order[i]
    const p = bfs.parent.data[u]
    if (p >= 0) up.push(between(u, p))
  }
  const down: MessageUpdate[] = []
  for (const u of order) {
    const p = bfs.parent.data[u]
    if (p >= 0) {
      const m = between(p, u)
      down.push(m)
    }
  }
  return [...up, ...down]
}

/**
 * The updates of one sweep of a schedule.
 *
 * @param graph The factor graph.
 * @param edges Its edges, as `factorGraphEdges` lists them.
 * @param schedule `'tree'`, `'flooding'`, `'sequential'`, or an explicit list of updates (copied).
 * @returns The updates in order: for flooding every factor-to-variable message then every variable-to-factor one, for
 *   sequential each variable's incoming then outgoing messages, variable by variable.
 */
function makeSchedule(graph: DiscreteFactorGraph, edges: FactorGraphEdge[], schedule: Schedule): MessageUpdate[] {
  if (Array.isArray(schedule)) return [...(schedule as readonly MessageUpdate[])]
  if (schedule === 'tree') return treeSchedule(graph, edges)
  if (schedule === 'flooding')
    return [
      ...edges.map((_, edge) => ({ edge, to: 'variable' as const })),
      ...edges.map((_, edge) => ({ edge, to: 'factor' as const })),
    ]
  const out: MessageUpdate[] = []
  for (let v = 0; v < graph.cardinalities.length; v++) {
    edges.forEach((e, edge) => e.variable === v && out.push({ edge, to: 'variable' }))
    edges.forEach((e, edge) => e.variable === v && out.push({ edge, to: 'factor' }))
  }
  return out
}

/**
 * Normalise in place; returns false when the vector sums to zero.
 *
 * @param m The non-negative vector, divided by its sum in place (left as it is when the sum is not positive).
 * @returns Whether it was normalised.
 */
function normaliseInPlace(m: Float64Array): boolean {
  let z = 0
  for (const x of m) z += x
  if (!(z > 0)) return false
  for (let i = 0; i < m.length; i++) m[i] /= z
  return true
}

/**
 * The variable-to-factor message $\mu_{x \to f}(x) \propto e(x) \prod_{g \ni x, g \ne f} \mu_{g \to x}(x)$, with $e$
 * the evidence indicator, unnormalised.
 *
 * @param s The state, for the edges and the evidence.
 * @param toVariable The factor-to-variable messages to read (the state's own, or newer ones).
 * @param edge The index of the edge between $x$ and $f$.
 * @returns The new message, over the variable's values.
 */
function variableToFactor(s: BeliefPropagationState, toVariable: readonly Tensor[], edge: number): Float64Array {
  const { variable, factor } = s.edges[edge]
  const out = Float64Array.from(s.evidence[variable].data)
  s.edges.forEach((e, k) => {
    if (e.variable !== variable || e.factor === factor) return
    const m = toVariable[k].data
    for (let i = 0; i < out.length; i++) out[i] *= m[i]
  })
  return out
}

/**
 * The factor-to-variable message
 * $\mu_{f \to x}(x) \propto \sum_{\xvec_f \setminus x} f(\xvec_f) \prod_{y \in f, y \ne x} \mu_{y \to f}(y)$, the sum
 * over the factor's other variables (a max for max-product), unnormalised.
 *
 * @param s The state, for the graph, the edges and the mode.
 * @param toFactor The variable-to-factor messages to read (the state's own, or newer ones).
 * @param edge The index of the edge between $f$ and $x$.
 * @returns The new message, over the values of $x$.
 */
function factorToVariable(s: BeliefPropagationState, toFactor: readonly Tensor[], edge: number): Float64Array {
  const { variable, factor, position } = s.edges[edge]
  const f = s.graph.factors[factor]
  const table = valuesOf(f.table)
  const incoming = f.scope.map((_, i) => {
    if (i === position) return null
    const k = s.edges.findIndex((e) => e.factor === factor && e.position === i)
    return toFactor[k].data
  })
  const out = new Float64Array(s.graph.cardinalities[variable]).fill(s.mode === 'sum' ? 0 : -Infinity)
  forEachAssignment(f.table.shape, (a, flat) => {
    let w = table[flat]
    for (let i = 0; i < a.length; i++) if (incoming[i]) w *= incoming[i]![a[i]]
    const x = a[position]
    out[x] = s.mode === 'sum' ? out[x] + w : Math.max(out[x], w)
  })
  return out
}

/**
 * The normalised beliefs $b(x) \propto e(x) \prod_f \mu_{f \to x}(x)$ of every variable.
 *
 * @param s The state, for the graph, the edges and the evidence.
 * @param toVariable The factor-to-variable messages to combine.
 * @returns One float64 vector per variable (left unnormalised when it sums to zero).
 */
function beliefsOf(s: BeliefPropagationState, toVariable: readonly Tensor[]): Tensor[] {
  return s.graph.cardinalities.map((k, v) => {
    const b = Float64Array.from(s.evidence[v].data)
    s.edges.forEach((e, j) => {
      if (e.variable !== v) return
      const m = toVariable[j].data
      for (let i = 0; i < k; i++) b[i] *= m[i]
    })
    normaliseInPlace(b)
    return fromData(b, [k])
  })
}

/**
 * The largest absolute difference between two vectors of the same length.
 *
 * @param a The first vector.
 * @param b The second vector, at least as long as `a`.
 * @returns $\max_i \lvert a_i - b_i \rvert$ (0 for empty vectors).
 */
function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]))
  return d
}

/**
 * Apply a batch of updates computed from the same (old) messages, with damping; returns the new message arrays.
 *
 * @param s The state whose messages every update reads; not modified.
 * @param updates The messages to recompute; each is normalised, then damped towards its old value.
 * @returns The new `toFactor` and `toVariable` arrays (copies), the largest change of a message, and whether any
 *   message summed to zero (`degenerate`, kept once set).
 */
function applyUpdates(s: BeliefPropagationState, updates: readonly MessageUpdate[]) {
  const toFactor = [...s.toFactor]
  const toVariable = [...s.toVariable]
  let change = 0
  let degenerate = s.degenerate
  const computed = updates.map((u) =>
    u.to === 'factor' ? variableToFactor(s, s.toVariable, u.edge) : factorToVariable(s, s.toFactor, u.edge),
  )
  updates.forEach((u, i) => {
    const m = computed[i]
    if (!normaliseInPlace(m)) degenerate = true
    const old = (u.to === 'factor' ? s.toFactor : s.toVariable)[u.edge].data
    if (s.damping > 0) for (let j = 0; j < m.length; j++) m[j] = (1 - s.damping) * m[j] + s.damping * old[j]
    change = Math.max(change, maxDiff(m, old))
    const t = fromData(m, [m.length])
    if (u.to === 'factor') toFactor[u.edge] = t
    else toVariable[u.edge] = t
  })
  return { toFactor, toVariable, change, degenerate }
}

/**
 * Belief propagation on `graph` as a traceable algorithm. Messages start uniform. A step applies one message update
 * (or, with `granularity: 'sweep'` or a flooding schedule, one sweep). Updates in a sequential, tree or explicit
 * schedule use the newest messages; a flooding sweep computes all factor-to-variable messages from the old
 * variable-to-factor messages, then all variable-to-factor messages from the new ones. The run stops as `converged`
 * once a whole sweep changes no message by more than `tolerance` (a tree schedule converges after its first sweep).
 * Throws `DomainError` at `init` when `damping` is not in $[0, 1)$.
 *
 * @param graph The discrete factor graph; read, not modified.
 * @param o The mode, schedule, damping, tolerance, granularity and evidence (see `BeliefPropagationOptions`).
 * @returns The algorithm, to run with `run(alg, undefined, steps)`.
 *
 * @example Message by message on a tree
 * // Rain (x0) and wet grass (x1): p(rain) = 0.2, p(wet | no rain) = 0.1, p(wet | rain) = 0.8. The tree schedule
 * // sends the three edges' messages in and back out, six in all.
 * const graph = {
 *   cardinalities: [2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const alg = beliefPropagationSteps(graph)
 * for (let steps = 1; steps <= 6; steps++) {
 *   const s = run(alg, undefined, steps)
 *   print(`step ${steps}: edge ${s.updated[0].edge} to ${s.updated[0].to}, p(wet) =`, s.beliefs[1], s.converged)
 * }
 *
 * @example Flooding on a loop, with damping
 * // A triangle of agreement factors with a bias on x0. Each step is a sweep; the change shrinks to the tolerance.
 * const agree = tensor([[2, 1], [1, 2]])
 * const graph = {
 *   cardinalities: [2, 2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([3, 1]) },
 *     { scope: [0, 1], table: agree },
 *     { scope: [1, 2], table: agree },
 *     { scope: [0, 2], table: agree },
 *   ],
 * }
 * const tr = trace(beliefPropagationSteps(graph, { damping: 0.5 }), undefined, 100)
 * print('changes =', tr.steps.slice(1, 6).map((s) => s.change))
 * const last = tr.steps[tr.steps.length - 1]
 * print('converged after', last.sweep, 'sweeps:', last.converged)
 * print('beliefs =', last.beliefs)
 */
export function beliefPropagationSteps(
  graph: DiscreteFactorGraph,
  o: BeliefPropagationOptions = {},
): Algorithm<void, BeliefPropagationState> {
  return {
    name: 'belief-propagation',
    init: () => {
      const edges = factorGraphEdges(graph)
      const damping = o.damping ?? 0
      if (!(damping >= 0 && damping < 1))
        throw new DomainError('beliefPropagation', 'beliefPropagation: damping must be in [0, 1)')
      const tree = isTree(graph)
      const scheduleName = o.schedule ?? (tree ? 'tree' : 'flooding')
      const evidence = graph.cardinalities.map((k, v) => {
        const e = new Float64Array(k).fill(1)
        const seen = o.evidence?.[v]
        if (seen !== undefined) {
          e.fill(0)
          e[seen] = 1
        }
        return fromData(e, [k])
      })
      const base = {
        t: 0,
        graph,
        edges,
        mode: o.mode ?? 'sum',
        damping,
        tolerance: o.tolerance ?? 1e-9,
        granularity: scheduleName === 'flooding' ? ('sweep' as const) : (o.granularity ?? 'message'),
        schedule: makeSchedule(graph, edges, scheduleName),
        flooding: scheduleName === 'flooding',
        toFactor: edges.map((e) => uniform(graph.cardinalities[e.variable])),
        toVariable: edges.map((e) => uniform(graph.cardinalities[e.variable])),
        position: 0,
        sweep: 0,
        updated: [],
        change: 0,
        sweepChange: 0,
        lastSweepChange: Infinity,
        converged: false,
        degenerate: false,
        evidence,
      } satisfies Omit<BeliefPropagationState, 'beliefs'>
      return { ...base, beliefs: beliefsOf(base as unknown as BeliefPropagationState, base.toVariable) }
    },
    step: (s) => {
      const n = s.schedule.length
      let state = s
      if (s.flooding) {
        const half = n / 2
        const first = applyUpdates(s, s.schedule.slice(0, half))
        const mid = { ...s, toVariable: first.toVariable }
        const second = applyUpdates(mid, s.schedule.slice(half))
        const change = Math.max(first.change, second.change)
        return {
          ...s,
          t: s.t + 1,
          toVariable: first.toVariable,
          toFactor: second.toFactor,
          beliefs: beliefsOf(s, first.toVariable),
          sweep: s.sweep + 1,
          updated: s.schedule,
          change,
          sweepChange: 0,
          lastSweepChange: change,
          converged: change < s.tolerance,
          degenerate: first.degenerate || second.degenerate,
        }
      }
      const count = s.granularity === 'sweep' ? n - s.position : 1
      const updated: MessageUpdate[] = []
      let change = 0
      for (let i = 0; i < count; i++) {
        const u = s.schedule[state.position]
        const r = applyUpdates(state, [u])
        updated.push(u)
        change = Math.max(change, r.change)
        state = {
          ...state,
          toFactor: r.toFactor,
          toVariable: r.toVariable,
          degenerate: r.degenerate,
          position: state.position + 1,
          sweepChange: Math.max(state.sweepChange, r.change),
        }
      }
      let { position, sweep, sweepChange, lastSweepChange, converged } = state
      if (position >= n) {
        // A tree schedule is exact after one sweep; otherwise converge when a whole sweep changed nothing.
        converged = sweepChange < s.tolerance || (sweep === 0 && isTreeSchedule(s))
        lastSweepChange = sweepChange
        position = 0
        sweep += 1
        sweepChange = 0
      }
      return {
        ...state,
        t: s.t + 1,
        beliefs: beliefsOf(state, state.toVariable),
        updated,
        change,
        position,
        sweep,
        sweepChange,
        lastSweepChange,
        converged,
      }
    },
  }
}

const treeScheduleFlag = new WeakMap<object, boolean>()
/**
 * Whether the state's schedule is the tree schedule of a tree-shaped graph, so one sweep of it is exact. Cached per
 * schedule array.
 *
 * @param s The state, for its graph, edges and schedule.
 * @returns True when the graph is a tree and the schedule is exactly its tree schedule.
 */
function isTreeSchedule(s: BeliefPropagationState): boolean {
  let flag = treeScheduleFlag.get(s.schedule)
  if (flag === undefined) {
    const tree = treeSchedule(s.graph, s.edges)
    flag =
      isTree(s.graph) &&
      tree.length === s.schedule.length &&
      tree.every((u, i) => u.edge === s.schedule[i].edge && u.to === s.schedule[i].to)
    treeScheduleFlag.set(s.schedule, flag)
  }
  return flag
}

/** The result of {@link beliefPropagation}. */
export interface BeliefPropagationResult {
  /** Beliefs (approximate marginals; exact on trees), one per variable. */
  marginals: Tensor[]
  /** The Bethe approximation to $\log Z$ (exact on trees); NaN in max mode. */
  logZ: number
  /** The sweeps run. */
  sweeps: number
  /** Whether the messages converged within `maxSteps` sweeps. */
  converged: boolean
  /** The final state, for `factorBeliefs`, `betheLogZ` and `decodeBeliefs`. */
  state: BeliefPropagationState
}

/**
 * Run belief propagation for at most `maxSteps` sweeps (default 200) and report beliefs and the Bethe $\log Z$. Exact
 * on trees (in one sweep of the default tree schedule); on a graph with loops, loopy BP, whose beliefs approximate
 * the marginals when it converges. Throws `DomainError` when `damping` is not in $[0, 1)$.
 *
 * @param graph The discrete factor graph; read, not modified.
 * @param options The options of `beliefPropagationSteps` (its `granularity` is ignored: each step is a sweep), and
 *   `maxSteps`, the most sweeps to run (default 200).
 * @returns The beliefs, the Bethe $\log Z$ (NaN in max mode), the sweeps run, whether they converged, and the final
 *   state.
 *
 * @example Exact on a tree, with and without evidence
 * // p(rain) = 0.2, p(wet | no rain) = 0.1, p(wet | rain) = 0.8.
 * const graph = {
 *   cardinalities: [2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const prior = beliefPropagation(graph)
 * print('p(wet) =', prior.marginals[1], 'in', prior.sweeps, 'sweep')
 * const wet = beliefPropagation(graph, { evidence: { 1: 1 } })
 * print('p(rain | wet) =', wet.marginals[0])
 * print('p(wet) from log Z =', Math.exp(wet.logZ))
 *
 * @example Approximate on a loop
 * // A triangle of agreement factors with a bias on x0, against the exact marginal by summing the eight assignments.
 * const agree = tensor([[2, 1], [1, 2]])
 * const graph = {
 *   cardinalities: [2, 2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([3, 1]) },
 *     { scope: [0, 1], table: agree },
 *     { scope: [1, 2], table: agree },
 *     { scope: [0, 2], table: agree },
 *   ],
 * }
 * const r = beliefPropagation(graph)
 * print('converged after', r.sweeps, 'sweeps:', r.converged)
 * print('BP p(x1) =', r.marginals[1])
 * const f = (a, b) => (a === b ? 2 : 1)
 * const p = [0, 0]
 * for (const a of [0, 1])
 *   for (const b of [0, 1])
 *     for (const c of [0, 1]) p[b] += (a ? 1 : 3) * f(a, b) * f(b, c) * f(a, c)
 * print('exact p(x1) =', p.map((x) => x / (p[0] + p[1])))
 */
export function beliefPropagation(
  graph: DiscreteFactorGraph,
  options: BeliefPropagationOptions & { maxSteps?: Size } = {},
): BeliefPropagationResult {
  const { maxSteps = 200, ...rest } = options
  const s = run(beliefPropagationSteps(graph, { ...rest, granularity: 'sweep' }), undefined, maxSteps)
  return {
    marginals: s.beliefs,
    logZ: s.mode === 'sum' ? betheLogZ(s) : NaN,
    sweeps: s.sweep,
    converged: s.converged,
    state: s,
  }
}

/**
 * Factor beliefs $b_f(\xvec_f) \propto f(\xvec_f) \prod_{x \in f} \mu_{x \to f}(x)$, one normalised table per factor
 * (the exact joint marginals of each factor's variables on a tree, once converged). A factor whose beliefs sum to
 * zero (contradictory evidence) gives NaN.
 *
 * @param s A state of `beliefPropagationSteps` (or the `state` of `beliefPropagation`); its messages are read.
 * @returns One float64 table per factor, of the factor's shape.
 *
 * @example The pairwise marginal of rain and wet grass
 * const graph = {
 *   cardinalities: [2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const { state } = beliefPropagation(graph)
 * print('p(rain, wet) =', factorBeliefs(state)[1])
 * print('the prior factor =', factorBeliefs(state)[0])
 */
export function factorBeliefs(s: BeliefPropagationState): Tensor[] {
  return s.graph.factors.map((f, k) => {
    const table = valuesOf(f.table)
    const incoming = f.scope.map(
      (_, i) => s.toFactor[s.edges.findIndex((e) => e.factor === k && e.position === i)].data,
    )
    const out = new Float64Array(table.length)
    let z = 0
    forEachAssignment(f.table.shape, (a, flat) => {
      let w = table[flat]
      for (let i = 0; i < a.length; i++) w *= incoming[i][a[i]]
      out[flat] = w
      z += w
    })
    return fromData(
      out.map((x) => x / z),
      f.table.shape,
    )
  })
}

/**
 * $-F_\text{Bethe}$, the Bethe approximation to $\log Z$ at the current messages (Yedidia, Freeman & Weiss 2005):
 * $\sum_f \sum_{\xvec_f} b_f(\xvec_f) \log [f(\xvec_f) / b_f(\xvec_f)] + \sum_x (d_x - 1) \sum_k b_x(k) \log b_x(k)$,
 * with $d_x$ the number of factors on $x$, $b_f$ the factor beliefs and $b_x$ the beliefs. Exact on trees at
 * convergence. Evidence enters as unary indicators, so with evidence it approximates the log of $Z$ restricted to the
 * observed values. Meaningful for sum-product beliefs only.
 *
 * @param s A state of `beliefPropagationSteps` (or the `state` of `beliefPropagation`), in sum mode.
 * @returns The approximation to $\log Z$.
 *
 * @example Exact on a tree, close on a loop
 * // Three binary variables in a triangle of factors [[2, 1], [1, 2]]: the two agreeing assignments weigh 8 and the
 * // six others 2, so Z = 2 · 8 + 6 · 2.
 * const agree = tensor([[2, 1], [1, 2]])
 * const loop = {
 *   cardinalities: [2, 2, 2],
 *   factors: [
 *     { scope: [0, 1], table: agree },
 *     { scope: [1, 2], table: agree },
 *     { scope: [0, 2], table: agree },
 *   ],
 * }
 * print('Bethe log Z =', betheLogZ(beliefPropagation(loop).state))
 * print('exact log Z =', Math.log(2 * 8 + 6 * 2))
 * const chain = { ...loop, factors: loop.factors.slice(0, 2) }
 * print('chain: Bethe', betheLogZ(beliefPropagation(chain).state), 'exact', Math.log(2 * 3 * 3))
 */
export function betheLogZ(s: BeliefPropagationState): number {
  const fb = factorBeliefs(s)
  let value = 0
  s.graph.factors.forEach((f, k) => {
    const table = valuesOf(f.table)
    const b = fb[k].data
    forEachAssignment(f.table.shape, (a, flat) => {
      if (b[flat] <= 0) return
      // Evidence indicators are part of the model: a belief on an excluded value has no mass anyway.
      let ev = 1
      for (let i = 0; i < a.length; i++) ev *= s.evidence[f.scope[i]].data[a[i]]
      value += b[flat] * (Math.log(table[flat] * ev) - Math.log(b[flat]))
    })
  })
  s.beliefs.forEach((b, v) => {
    const d = s.edges.filter((e) => e.variable === v).length
    let h = 0
    for (const p of b.data) if (p > 0) h += p * Math.log(p)
    value += (d - 1) * h
  })
  return value
}

/**
 * The value of each variable maximising its (max-product) belief; the MAP assignment on a tree without ties. Ties go
 * to the smaller value. On sum-product beliefs it gives the most probable value of each variable on its own instead.
 *
 * @param s A state of `beliefPropagationSteps` (or the `state` of `beliefPropagation`), usually run with
 *   `mode: 'max'`.
 * @returns The chosen value of each variable (int32, one entry per variable).
 *
 * @example Max-product finds the MAP; sum-product beliefs need not
 * // p(0, 0) = 0.4, p(0, 1) = 0.3, p(1, 1) = 0.3: the best pair is (0, 0), but x1 = 1 is more probable on its own.
 * const graph = { cardinalities: [2, 2], factors: [{ scope: [0, 1], table: tensor([[0.4, 0.3], [0, 0.3]]) }] }
 * print('max-product:', decodeBeliefs(beliefPropagation(graph, { mode: 'max' }).state))
 * print('sum-product:', decodeBeliefs(beliefPropagation(graph).state))
 */
export function decodeBeliefs(s: BeliefPropagationState): Tensor {
  const out = Int32Array.from(s.beliefs, (b) => {
    let best = 0
    for (let i = 1; i < b.data.length; i++) if (b.data[i] > b.data[best]) best = i
    return best
  })
  return fromData(out, [out.length])
}
