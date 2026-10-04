/**
 * Flows: maximum flow and minimum cut by Edmonds–Karp (Edmonds and Karp, 1972, "Theoretical improvements in
 * algorithmic efficiency for network flow problems", JACM 19(2); the max-flow min-cut theorem of Ford and Fulkerson,
 * 1956), and minimum-cost flow by successive shortest paths with potentials (Ahuja, Magnanti and Orlin, 1993, "Network
 * Flows", §9.7). Both are traceable.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { floats, floatsOf, ints, type Graph } from '../graph'
import { directedArcs, type DirectedArc } from 'aifn-compute/graph/shortest-paths'

const EPS = 1e-12

// ---------------------------------------------------------------------------------------------------------------------
// Edmonds–Karp.

/** Options of Edmonds–Karp: edge weights are capacities (an undirected edge has that capacity in each direction). */
export interface MaxFlowOptions {
  source: number
  sink: number
}

/** One state of Edmonds–Karp. */
export interface EdmondsKarpState extends Status {
  /** The directed arcs (an undirected edge gives two), with `weight` the capacity and `edge` the edge index. */
  arcs: readonly DirectedArc[]
  /** Residual moves at each node, in arc order: k ≥ 0 uses arc k forwards, −k−1 uses arc k backwards (cancelling). */
  residual: readonly (readonly number[])[]
  source: number
  sink: number
  /** Flow on each arc; float64, one per arc. */
  flow: Tensor
  /** The flow value (net flow out of the source). */
  value: number
  /** The augmenting path of the last step as residual moves (see `residual`), source to sink; int32. */
  path: Tensor
  /** The same path as nodes, source to sink; int32. */
  pathNodes: Tensor
  /** Flow sent along the last path. */
  bottleneck: number
  /**
   * Nodes the last breadth-first search reached in the residual graph (0/1; int32). Once done, these are the source
   * side of a minimum cut.
   */
  reached: Tensor
  iteration: number
  done: boolean
}

/** The init and step of `edmondsKarpSteps` on the whole problem; `t` is added by the factory. */
const edmondsKarp = {
  init: ({ graph, source, sink }: MaxFlowOptions & { graph: Graph }): Omit<EdmondsKarpState, 't'> => {
    if (source === sink) throw new ShapeError('edmondsKarp', 'edmondsKarp: source and sink must differ')
    const arcs = directedArcs(graph)
    if (arcs.some((a) => a.weight < 0))
      throw new DomainError('edmondsKarp', 'edmondsKarp: capacities must be non-negative')
    const residual: number[][] = Array.from({ length: graph.nodes }, () => [])
    arcs.forEach((a, k) => {
      residual[a.from].push(k)
      residual[a.to].push(-k - 1)
    })
    const reached = new Int32Array(graph.nodes)
    reached[source] = 1
    return {
      arcs,
      residual,
      source,
      sink,
      flow: floats(new Float64Array(arcs.length)),
      value: 0,
      path: ints([]),
      pathNodes: ints([]),
      bottleneck: 0,
      reached: ints(reached),
      iteration: 0,
      done: false,
    }
  },
  step: (s: EdmondsKarpState): Omit<EdmondsKarpState, 't'> => {
    if (s.done) return s
    const flow = floatsOf(s.flow)
    const V = s.residual.length
    const via = new Int32Array(V).fill(0)
    const reached = new Int32Array(V)
    reached[s.source] = 1
    const queue = [s.source]
    const spare = (m: number) => (m >= 0 ? s.arcs[m].weight - flow[m] : flow[-m - 1])
    const head = (m: number) => (m >= 0 ? s.arcs[m].to : s.arcs[-m - 1].from)
    while (queue.length && !reached[s.sink]) {
      const u = queue.shift()!
      for (const m of s.residual[u]) {
        const w = head(m)
        if (!reached[w] && spare(m) > EPS) {
          reached[w] = 1
          via[w] = m
          queue.push(w)
        }
      }
    }
    if (!reached[s.sink])
      return { ...s, reached: ints(reached), path: ints([]), pathNodes: ints([]), bottleneck: 0, done: true }
    const moves: number[] = []
    const nodes = [s.sink]
    let bottleneck = Infinity
    for (let v = s.sink; v !== s.source;) {
      const m = via[v]
      moves.push(m)
      bottleneck = Math.min(bottleneck, spare(m))
      v = m >= 0 ? s.arcs[m].from : s.arcs[-m - 1].to
      nodes.push(v)
    }
    for (const m of moves) {
      if (m >= 0) flow[m] += bottleneck
      else flow[-m - 1] -= bottleneck
    }
    return {
      ...s,
      flow: floats(flow),
      value: s.value + bottleneck,
      path: ints(moves.reverse()),
      pathNodes: ints(nodes.reverse()),
      bottleneck,
      reached: ints(reached),
      iteration: s.iteration + 1,
      done: bottleneck === Infinity,
    }
  },
}

/**
 * Edmonds–Karp as a traceable algorithm. Options: `{ source, sink }` with non-negative capacities as edge
 * weights. Each step runs breadth-first search in the residual graph from the source; if it reaches the sink, the
 * shortest augmenting path is saturated by its bottleneck. When it cannot, the flow is maximum and the reached nodes
 * form the source side of a minimum cut.
 */
export function edmondsKarpSteps(graph: Graph, options: MaxFlowOptions): Algorithm<void, EdmondsKarpState> {
  const problem: MaxFlowOptions & { graph: Graph } = { graph, ...options }
  return {
    name: 'edmonds-karp',
    init: () => ({ ...edmondsKarp.init(problem), t: 0 }),
    step: (s) => ({ ...edmondsKarp.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/** A maximum flow with a minimum cut. */
export interface MaxFlowResult {
  /** The flow value, equal to `cutCapacity`. */
  value: number
  /** Flow on each edge of `graph.edges` (for an undirected edge, the net flow from `from` to `to`, maybe negative). */
  flow: Tensor
  /** 1 for nodes on the source side of the minimum cut; int32, length V. */
  sourceSide: Tensor
  /** Edges crossing the cut from the source side to the sink side (indices into `graph.edges`); int32. */
  cut: Tensor
  /** Total capacity of `cut`. */
  cutCapacity: number
  /** Augmenting paths used. */
  iterations: number
}

/**
 * Maximum flow from `source` to `sink` by Edmonds–Karp, with the minimum cut read off the last residual search.
 * Unbounded when a path of infinite-capacity edges joins them (`value` is then Infinity).
 */
export function maxFlow(graph: Graph, source: number, sink: number): MaxFlowResult {
  const s = run(edmondsKarpSteps(graph, { source, sink }), undefined, Infinity)
  const flow = new Float64Array(graph.edges.length)
  s.arcs.forEach((a, k) => {
    const f = s.flow.data[k]
    flow[a.edge] += a.from === graph.edges[a.edge].from ? f : -f
  })
  const side = s.reached.data
  const cut: number[] = []
  let cutCapacity = 0
  const counted = new Set<number>()
  for (const a of s.arcs) {
    if (side[a.from] && !side[a.to] && !counted.has(a.edge)) {
      counted.add(a.edge)
      cut.push(a.edge)
      cutCapacity += a.weight
    }
  }
  return {
    value: s.value,
    flow: floats(flow),
    sourceSide: s.reached,
    cut: ints(cut),
    cutCapacity,
    iterations: s.iteration,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Minimum-cost flow.

/** An arc of a flow network. */
export interface FlowArc {
  from: number
  to: number
  capacity: number
  cost: number
}

/** A min-cost flow problem: send `supply[v]` (positive at sources, negative at sinks, summing to 0) at least cost. */
export interface FlowNetwork {
  nodes: number
  arcs: readonly FlowArc[]
  supply: readonly number[]
}

/** One state of successive shortest paths. */
export interface MinCostFlowState extends Status {
  /** Flow on each arc, length E. */
  flow: Tensor
  /** Node potentials π, which keep reduced costs c − π(from) + π(to) non-negative on residual arcs. */
  potential: Tensor
  /** Supply not yet sent (positive) or demand not yet met (negative), per node. */
  excess: Tensor
  /** The augmenting path of the last step as arc indices (a negative index −k−1 is arc k used backwards); int32. */
  path: Tensor
  /** Flow sent along the last path. */
  sent: number
  /** Total cost Σ cost · flow. */
  cost: number
  status: 'running' | 'optimal' | 'infeasible'
  network: FlowNetwork
}

/** The init and step of `minCostFlowSteps` on the whole problem; `t` is added by the factory. */
const successiveShortestPaths = {
  init: (network: FlowNetwork): Omit<MinCostFlowState, 't'> => {
    const V = network.nodes
    const total = network.supply.reduce((a, b) => a + b, 0)
    if (network.supply.length !== V)
      throw new ShapeError('minCostFlow', 'minCostFlow: supply must have one entry per node')
    if (Math.abs(total) > 1e-9) throw new DomainError('minCostFlow', 'minCostFlow: supplies must sum to zero')
    // Potentials: shortest distances from a virtual root over arcs with capacity (Bellman–Ford handles negative costs).
    const pi = new Float64Array(V)
    for (let pass = 0; pass < V; pass++)
      for (const a of network.arcs) if (a.capacity > 0 && pi[a.from] + a.cost < pi[a.to]) pi[a.to] = pi[a.from] + a.cost
    return {
      flow: floats(new Float64Array(network.arcs.length)),
      potential: floats(pi),
      excess: floats(network.supply),
      path: ints([]),
      sent: 0,
      cost: 0,
      status: network.supply.some((v) => v !== 0) ? 'running' : 'optimal',
      network,
    }
  },
  step: (s: MinCostFlowState): Omit<MinCostFlowState, 't'> => {
    if (s.status !== 'running') return s
    const { network } = s
    const V = network.nodes
    const flow = floatsOf(s.flow)
    const pi = floatsOf(s.potential)
    const excess = floatsOf(s.excess)
    const source = excess.findIndex((v) => v > EPS)
    // Dijkstra on reduced costs over residual arcs: forward with spare capacity, backward with positive flow.
    const d = new Float64Array(V).fill(Infinity)
    const via = new Int32Array(V).fill(0)
    const prev = new Int32Array(V).fill(-1)
    const settled = new Uint8Array(V)
    d[source] = 0
    for (;;) {
      let u = -1
      for (let v = 0; v < V; v++) if (!settled[v] && d[v] < Infinity && (u < 0 || d[v] < d[u])) u = v
      if (u < 0) break
      settled[u] = 1
      network.arcs.forEach((a, k) => {
        if (a.from === u && flow[k] < a.capacity - EPS) {
          const rc = a.cost + pi[u] - pi[a.to]
          if (d[u] + rc < d[a.to] - EPS) {
            d[a.to] = d[u] + rc
            via[a.to] = k
            prev[a.to] = u
          }
        }
        if (a.to === u && flow[k] > EPS) {
          const rc = -a.cost + pi[u] - pi[a.from]
          if (d[u] + rc < d[a.from] - EPS) {
            d[a.from] = d[u] + rc
            via[a.from] = -k - 1
            prev[a.from] = u
          }
        }
      })
    }
    // The nearest node with unmet demand.
    let sink = -1
    for (let v = 0; v < V; v++) if (excess[v] < -EPS && d[v] < Infinity && (sink < 0 || d[v] < d[sink])) sink = v
    if (sink < 0) return { ...s, status: 'infeasible' as const, path: ints([]), sent: 0 }
    // Update potentials so reduced costs stay non-negative; unreachable nodes take the largest finite distance, which
    // keeps arcs from them into the reachable set non-negative.
    let far = 0
    for (let v = 0; v < V; v++) if (d[v] < Infinity) far = Math.max(far, d[v])
    for (let v = 0; v < V; v++) pi[v] += d[v] < Infinity ? d[v] : far
    const path: number[] = []
    let amount = Math.min(excess[source], -excess[sink])
    for (let v = sink; v !== source; v = prev[v]) {
      const k = via[v]
      path.push(k)
      amount = Math.min(amount, k >= 0 ? network.arcs[k].capacity - flow[k] : flow[-k - 1])
    }
    for (const k of path) {
      if (k >= 0) flow[k] += amount
      else flow[-k - 1] -= amount
    }
    excess[source] -= amount
    excess[sink] += amount
    let cost = 0
    network.arcs.forEach((a, k) => (cost += a.cost * flow[k]))
    const remaining = excess.some((v) => Math.abs(v) > EPS)
    return {
      ...s,
      flow: floats(flow),
      potential: floats(pi),
      excess: floats(excess),
      path: ints(path.reverse()),
      sent: amount,
      cost,
      status: remaining ? 'running' : 'optimal',
    }
  },
}

/**
 * Minimum-cost flow by successive shortest paths as a traceable algorithm. Options: a `FlowNetwork` whose arcs have no
 * negative-cost cycle. Initial potentials come from Bellman–Ford (so negative costs are allowed); each step runs
 * Dijkstra on reduced costs in the residual network from a node with excess to a node with demand and augments along
 * the path by as much as capacities, excess and demand allow. `infeasible` when some demand cannot be reached.
 */
export function minCostFlowSteps(network: FlowNetwork): Algorithm<void, MinCostFlowState> {
  const problem: FlowNetwork = network
  return {
    name: 'min-cost-flow',
    init: () => ({ ...successiveShortestPaths.init(problem), t: 0 }),
    step: (s) => ({ ...successiveShortestPaths.step(s), t: s.t + 1 }),
    done: (s) => s.status !== 'running',
  }
}

/**
 * Solve a min-cost flow problem by successive shortest paths within `maxSteps` augmentations (default 100 000); returns
 * the final state (flow, cost, status; `t` is the number of augmenting steps).
 */
export function minCostFlow(network: FlowNetwork, options: { maxSteps?: number } = {}): MinCostFlowState {
  return run(minCostFlowSteps(network), undefined, options.maxSteps ?? 100_000)
}
