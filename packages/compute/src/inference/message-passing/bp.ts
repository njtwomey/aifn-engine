/**
 * Belief propagation on discrete factor graphs: sum-product and max-product (Pearl 1988; Kschischang, Frey &
 * Loeliger 2001, "Factor graphs and the sum-product algorithm", IEEE Trans. Inf. Theory 47(2)), exact on trees in one
 * inward and one outward pass, and loopy BP with flooding or sequential schedules and damping (Murphy, Weiss & Jordan
 * 1999, "Loopy belief propagation for approximate inference: an empirical study"). The Bethe free energy (Yedidia,
 * Freeman & Weiss 2005) gives log Z, exactly on trees.
 *
 * Messages are normalised to sum to one. Every step records which messages it updated, so a figure can step
 * through the schedule message by message.
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
  edge: number
  to: 'factor' | 'variable'
}

/**
 * A message schedule. `tree`: leaves to a root and back (exact on trees and forests, in 2 × edges messages);
 * `flooding`: every factor-to-variable message in parallel, then every variable-to-factor message; `sequential`:
 * variable by variable, its incoming then its outgoing messages; or an explicit list of updates, repeated each sweep.
 */
export type Schedule = 'tree' | 'flooding' | 'sequential' | readonly MessageUpdate[]

/** Options of {@link beliefPropagationSteps}. */
export interface BeliefPropagationOptions {
  /** `sum` for marginals, `max` for max-marginals (max-product). Default `sum`. */
  mode?: 'sum' | 'max'
  /** Default `tree` when the graph is a tree or forest, else `flooding`. */
  schedule?: Schedule
  /** Weight of the old message in each update, in [0, 1): new ← (1 − λ) new + λ old. Default 0. */
  damping?: number
  /** A sweep whose largest message change is below this has converged. Default 1e-9. */
  tolerance?: number
  /** `message`: one update per step; `sweep`: a whole pass of the schedule per step. Flooding is always per sweep. */
  granularity?: 'message' | 'sweep'
  /** Observed values: variable → value, applied as indicator factors. */
  evidence?: Readonly<Record<number, number>>
}

/**
 * The state of belief propagation. Messages are indexed like `edges`. `t` counts steps (messages or sweeps, by
 * `granularity`); `converged` is set once a whole sweep changes no message by more than `tolerance`.
 */
export interface BeliefPropagationState extends Status {
  graph: DiscreteFactorGraph
  edges: FactorGraphEdge[]
  mode: 'sum' | 'max'
  damping: number
  tolerance: number
  granularity: 'message' | 'sweep'
  /** The updates of one sweep, in order (flooding: all factor-to-variable, then all variable-to-factor). */
  schedule: MessageUpdate[]
  flooding: boolean
  /** μ_{x→f} and μ_{f→x}, one float64 vector per edge over the variable's values. */
  toFactor: Tensor[]
  toVariable: Tensor[]
  /** Normalised beliefs b(x) ∝ (evidence) Π_f μ_{f→x}, one per variable. */
  beliefs: Tensor[]
  /** Position in the schedule (0 at the start of a sweep) and completed sweeps. */
  position: number
  sweep: number
  /** The updates made by the last step. */
  updated: MessageUpdate[]
  /** Largest absolute change of a message in the last step, in the current sweep, and in the last full sweep. */
  change: number
  sweepChange: number
  lastSweepChange: number
  converged: boolean
  /** True when a message summed to zero (contradictory evidence); it was left unnormalised. */
  degenerate: boolean
  /** Evidence indicator per variable (all ones when unobserved). */
  evidence: Tensor[]
}

const uniform = (k: number) => fromData(new Float64Array(k).fill(1 / k), [k])

/** Leaves-to-root-and-back order over a forest, from breadth-first search on the bipartite graph. */
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

/** Normalise in place; returns false when the vector sums to zero. */
function normaliseInPlace(m: Float64Array): boolean {
  let z = 0
  for (const x of m) z += x
  if (!(z > 0)) return false
  for (let i = 0; i < m.length; i++) m[i] /= z
  return true
}

/** μ_{x→f}(x) ∝ evidence(x) Π_{g ∋ x, g ≠ f} μ_{g→x}(x). */
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

/** μ_{f→x}(x) ∝ Σ (or max) over the other variables of f(x_f) Π_{y ∈ f, y ≠ x} μ_{y→f}(y). */
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

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]))
  return d
}

/** Apply a batch of updates computed from the same (old) messages, with damping; returns the new message arrays. */
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
 * Belief propagation on `graph` as a traceable algorithm. Messages start uniform. A step applies one message update (or, with
 * `granularity: 'sweep'` or a flooding schedule, one sweep). Updates in a sequential schedule use the newest
 * messages; a flooding sweep computes all factor-to-variable messages from the old variable-to-factor messages, then
 * all variable-to-factor messages from the new ones. The run is `done` once a whole sweep changes no message by more
 * than `tolerance` (a tree schedule converges after its first sweep).
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
  /** The Bethe approximation to log Z (exact on trees); NaN in max mode. */
  logZ: number
  sweeps: number
  converged: boolean
  state: BeliefPropagationState
}

/** Run belief propagation for at most `maxSteps` sweeps (default 200) and report beliefs and the Bethe log Z. */
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
 * Factor beliefs b_f(x_f) ∝ f(x_f) Π_{x ∈ f} μ_{x→f}(x), one normalised table per factor (the pairwise marginals on
 * a tree).
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
 * −F_Bethe, the Bethe approximation to log Z at the current messages: Σ_f Σ b_f log(f / b_f) + Σ_x (d_x − 1) Σ b_x
 * log b_x, with d_x the number of factors on x. Exact on trees at convergence. Evidence enters as unary indicators.
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

/** The value of each variable maximising its (max-product) belief; the MAP assignment on a tree without ties. */
export function decodeBeliefs(s: BeliefPropagationState): Tensor {
  const out = Int32Array.from(s.beliefs, (b) => {
    let best = 0
    for (let i = 1; i < b.data.length; i++) if (b.data[i] > b.data[best]) best = i
    return best
  })
  return fromData(out, [out.length])
}
