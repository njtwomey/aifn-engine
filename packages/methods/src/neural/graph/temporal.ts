/**
 * The memory module of a temporal graph network on a stream of timed interactions $(u, v, t, \evec)$ (Rossi et al.
 * 2020, "Temporal graph networks for deep learning on dynamic graphs", §3.1). Each node keeps a memory $\svec_i$,
 * starting at zero. An event updates both endpoints from their memories just before it,
 *
 * $$\svec_u \leftarrow \mathrm{mem}(\svec_u, \svec_v, \evec, \Delta t_u), \qquad
 * \svec_v \leftarrow \mathrm{mem}(\svec_v, \svec_u, \evec, \Delta t_v),$$
 *
 * with $\Delta t$ the time since that node's previous update. `memoryTrace` replays the stream one event at a time
 * with any update function (a GRU cell in TGN, a plain average for hand calculation) and keeps every intermediate
 * state, ready for a figure that steps through it. `timeEncoding` is the functional time encoding
 * $[\cos(\omega_k \Delta t + b_k)]_k$ of Xu et al. (2020), "Inductive representation learning on temporal graphs",
 * ICLR; `timeProjection` is JODIE's $(1 + \Delta t\,\wvec) \odot \svec$ (Kumar, Zhang and Leskovec 2019, "Predicting
 * dynamic embedding trajectory in temporal interaction networks", KDD, §3.2). `timeRespectingReach` marks the nodes
 * that information from one node can have reached along paths whose event times increase, and `staticReach` the nodes
 * joined to it by any events so far, in any order.
 *
 * Differences from TGN: one event per step (no batching, so no message aggregator), and the update receives the raw
 * inputs instead of a concatenated message. Events are taken in the order given; equal times count as ordered.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { unionFind, unionFindRoot, unite } from 'aifn-compute/graph'

/** One interaction: two node indices, a time and an edge feature vector. */
export interface TimedEvent {
  /** The source node $u$. */
  source: number
  /** The destination node $v$, distinct from the source. */
  destination: number
  /** The time $t$ of the event. */
  time: number
  /** The edge features $\evec$. */
  feature: readonly number[]
}

/**
 * A memory update: from the node's memory, the other endpoint's memory, the edge features and the time since the
 * node's previous update, to the node's new memory.
 */
export type MemoryUpdate = (
  self: readonly number[],
  other: readonly number[],
  feature: readonly number[],
  dt: number,
) => number[]

/** The replay of a stream by `memoryTrace`. */
export interface MemoryTrace {
  /** `memory[k][i]`: node $i$'s memory after the first $k$ events ($k = 0$: all zero); one more entry than events. */
  memory: number[][][]
  /** `lastUpdate[k][i]`: the time of node $i$'s last update after $k$ events (0 before any). */
  lastUpdate: number[][]
  /** `gaps[k]`: the time since the previous update of the source and of the destination of event $k + 1$. */
  gaps: [number, number][]
}

/**
 * Check that every event joins two distinct nodes of the graph. Throws `DomainError` otherwise.
 *
 * @param nodes The number of nodes.
 * @param events The events.
 * @param where The caller's name, for error messages.
 */
function checkEvents(nodes: number, events: readonly TimedEvent[], where: string): void {
  events.forEach((e, k) => {
    for (const v of [e.source, e.destination])
      if (!(Number.isInteger(v) && v >= 0 && v < nodes))
        throw new DomainError(where, `${where}: event ${k} has node ${v}, outside 0…${nodes - 1}`)
    if (e.source === e.destination)
      throw new DomainError(where, `${where}: event ${k} joins node ${e.source} to itself`)
  })
}

/**
 * Replay a stream of events in order, updating the memories of both endpoints of each event from the memories just
 * before it (see the file comment). Throws `DomainError` for an event outside the nodes or from a node to itself (whose
 * second update would read the first one's stale input).
 *
 * @param nodes The number of nodes.
 * @param dim The memory width.
 * @param events The events, in time order.
 * @param update The memory update.
 * @returns Every intermediate memory, the last-update times, and the gaps $\Delta t$ the updates saw.
 *
 * @example Two events with the average update: each memory becomes (own + other + rating) / 2
 * const events = [
 *   { source: 0, destination: 2, time: 1, feature: [4] },
 *   { source: 1, destination: 2, time: 3, feature: [0] },
 * ]
 * const average = (self, other, feature) => [0.5 * (self[0] + other[0] + feature[0])]
 * const t = memoryTrace(3, 1, events, average)
 * print('after event 1:', t.memory[1], ' after event 2:', t.memory[2])
 * print('gaps:', t.gaps)
 */
export function memoryTrace(
  nodes: number,
  dim: number,
  events: readonly TimedEvent[],
  update: MemoryUpdate,
): MemoryTrace {
  checkEvents(nodes, events, 'memoryTrace')
  let memory = Array.from({ length: nodes }, () => new Array<number>(dim).fill(0))
  let last = new Array<number>(nodes).fill(0)
  const trace: MemoryTrace = { memory: [memory], lastUpdate: [last], gaps: [] }
  for (const e of events) {
    const { source: u, destination: v, time } = e
    const dtU = time - last[u]
    const dtV = time - last[v]
    const next = memory.slice()
    next[u] = update(memory[u], memory[v], e.feature, dtU)
    next[v] = update(memory[v], memory[u], e.feature, dtV)
    const nextLast = last.slice()
    nextLast[u] = time
    nextLast[v] = time
    memory = next
    last = nextLast
    trace.memory.push(memory)
    trace.lastUpdate.push(last)
    trace.gaps.push([dtU, dtV])
  }
  return trace
}

/**
 * The functional time encoding
 * $\phi(\Delta t) = [\cos(\omega_1 \Delta t + b_1), \dots, \cos(\omega_d \Delta t + b_d)]$.
 *
 * @param dt The elapsed time $\Delta t$.
 * @param omega The frequencies $\omega_k$.
 * @param phase The phases $b_k$ (default zero).
 * @returns The encoding, one entry per frequency.
 *
 * @example Slow and fast frequencies
 * print(timeEncoding(10, [1, 0.1]))
 */
export function timeEncoding(dt: number, omega: readonly number[], phase?: readonly number[]): number[] {
  return omega.map((w, k) => Math.cos(w * dt + (phase ? phase[k] : 0)))
}

/**
 * JODIE's time projection $\hat\svec(t + \Delta) = (1 + \Delta\,\wvec) \odot \svec(t)$ of a memory to a later time.
 *
 * @param memory The memory $\svec(t)$.
 * @param dt The elapsed time $\Delta$.
 * @param w The learned vector $\wvec$, one entry per memory entry.
 * @returns The projected memory.
 *
 * @example A decay of 0.01 per unit of time halves the memory after 50
 * print(timeProjection([2, -4], 50, [-0.01, -0.01]))
 */
export function timeProjection(memory: readonly number[], dt: number, w: readonly number[]): number[] {
  return memory.map((s, k) => (1 + dt * w[k]) * s)
}

/**
 * The nodes that information from `origin` can have reached after each prefix of the stream: an event passes it on
 * when either endpoint already holds it, so it travels only along paths with increasing event times. Throws as
 * `memoryTrace` does.
 *
 * @param nodes The number of nodes.
 * @param events The events, in time order.
 * @param origin The node the information starts at.
 * @returns `reach[k][i]`: whether node $i$ holds it after the first $k$ events; one more entry than events.
 *
 * @example Information from node 0 cannot travel back in time to node 2
 * const events = [
 *   { source: 2, destination: 1, time: 1, feature: [] },
 *   { source: 0, destination: 1, time: 2, feature: [] },
 * ]
 * print(timeRespectingReach(3, events, 0).at(-1))
 */
export function timeRespectingReach(nodes: number, events: readonly TimedEvent[], origin: number): boolean[][] {
  checkEvents(nodes, events, 'timeRespectingReach')
  let held = new Array<boolean>(nodes).fill(false)
  held[origin] = true
  const out = [held]
  for (const e of events) {
    held = held.slice()
    if (held[e.source] || held[e.destination]) held[e.source] = held[e.destination] = true
    out.push(held)
  }
  return out
}

/**
 * The nodes joined to `origin` by any path of the events so far, in any time order: its component in the static
 * graph of the events, by union-find. Throws as `memoryTrace` does.
 *
 * @param nodes The number of nodes.
 * @param events The events.
 * @param origin The node.
 * @returns `reach[k][i]`: whether node $i$ is joined to `origin` by the first $k$ events; one more entry than events.
 *
 * @example Ignoring time, node 2 is joined to node 0
 * const events = [
 *   { source: 2, destination: 1, time: 1, feature: [] },
 *   { source: 0, destination: 1, time: 2, feature: [] },
 * ]
 * print(staticReach(3, events, 0).at(-1))
 */
export function staticReach(nodes: number, events: readonly TimedEvent[], origin: number): boolean[][] {
  checkEvents(nodes, events, 'staticReach')
  const uf = unionFind(nodes)
  const snapshot = () => {
    const root = unionFindRoot(uf, origin)
    return Array.from({ length: nodes }, (_, i) => unionFindRoot(uf, i) === root)
  }
  const out = [snapshot()]
  for (const e of events) {
    unite(uf, e.source, e.destination)
    out.push(snapshot())
  }
  return out
}
