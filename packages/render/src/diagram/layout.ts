/**
 * Automatic layered layout for diagrams generated from data (Sugiyama, Tagawa and Toda, 1981): nodes go into layers
 * by topological depth (cycles broken at depth-first back edges and the order from Kahn's algorithm, both from
 * `aifn-compute/graph`), long edges are split by virtual nodes, each layer is ordered by barycentre sweeps to reduce
 * crossings (with adjacent transpositions), and positions within a layer are pulled towards their neighbours while
 * keeping the order. Spacing comes from each node's actual size (after label fitting, when the caller supplies it), so
 * nodes never overlap. Right-angled edges bend only in the gaps between layers, each on its own track, so they leave a
 * node's downstream side, enter the next node's upstream side and never cross a node.
 */
import { depthFirstSearch, topologicalSort } from 'aifn-compute/graph/traversal'
import { fromEdges } from 'aifn-compute/graph'
import { parseEnd } from './ends'
import { nodeSize } from './size'
import type { DiagramEdge, DiagramNode, DiagramSpec, PlacedNode, Side } from './types'

const endId = (ref: string) => parseEnd(ref).id

/** Crossings between two adjacent layers, given each edge as (position above, position below). */
function crossings(pairs: [number, number][]): number {
  let count = 0
  for (let i = 0; i < pairs.length; i++)
    for (let j = i + 1; j < pairs.length; j++) {
      const [a1, b1] = pairs[i]
      const [a2, b2] = pairs[j]
      if ((a1 - a2) * (b1 - b2) < 0) count++
    }
  return count
}

/** Clearance between neighbouring nodes in a layer, and between a node and a passing edge, in grid units. */
const CLEAR_ACROSS = 0.5
const CLEAR_VIRTUAL = 0.35
/** Least room between layers for the edges' bends and arrowheads, and the room each extra track needs. */
const CLEAR_ALONG = 0.9
const TRACK = 0.4
/** Room a note or outside label takes beside a node. */
const NOTE = 0.6

/**
 * Least-squares positions for an ordered row: minimise Σ wᵢ (yᵢ − dᵢ)² subject to yᵢ₊₁ − yᵢ ≥ gapᵢ. With
 * zᵢ = dᵢ − oᵢ, where oᵢ is the sum of the gaps before i, this is isotonic regression of z, solved by pooling adjacent
 * violators.
 */
export function spaced(desired: number[], gaps: number[], weights: number[] = desired.map(() => 1)): number[] {
  const offset = desired.map(() => 0)
  for (let i = 1; i < desired.length; i++) offset[i] = offset[i - 1] + gaps[i - 1]
  const blocks: { sum: number; w: number; n: number }[] = []
  desired.forEach((d, i) => {
    blocks.push({ sum: weights[i] * (d - offset[i]), w: weights[i], n: 1 })
    while (blocks.length > 1) {
      const b = blocks[blocks.length - 1]
      const a = blocks[blocks.length - 2]
      if (a.sum / a.w <= b.sum / b.w) break
      blocks.splice(blocks.length - 2, 2, { sum: a.sum + b.sum, w: a.w + b.w, n: a.n + b.n })
    }
  })
  const out: number[] = []
  for (const b of blocks) for (let k = 0; k < b.n; k++) out.push(b.sum / b.w + offset[out.length])
  return out
}

/**
 * Place the nodes of a spec by the layered layout (see `LayeredOptions`), returning a spec with every node's `x` and
 * `y` set and waypoints on edges that bend or span several layers. Edges without a direction still count from `from`
 * to `to`; cycles are broken by ignoring the edges that close them. `sizeOf` gives each node's size in grid units
 * (default: its `w` and `h`, else its shape's); `Diagram` passes the sizes its labels grew to.
 */
export function layeredLayout(
  spec: DiagramSpec,
  sizeOf: (node: DiagramNode) => [number, number] = nodeSize,
): Omit<DiagramSpec, 'nodes'> & { nodes: PlacedNode[] } {
  const { direction = 'right', layerGap = 2, nodeGap = 1.6, sweeps = 8, compactSources = true } = spec.layered ?? {}
  const ids = spec.nodes.map((n) => n.id)
  const index = new Map(ids.map((id, i) => [id, i]))
  const edges = (spec.edges ?? [])
    .map((e, k) => ({ k, a: index.get(endId(e.from)), b: index.get(endId(e.to)) }))
    .filter((e): e is { k: number; a: number; b: number } => {
      if (e.a === undefined || e.b === undefined) throw new Error(`diagram edge ${spec.edges![e.k].from}: unknown node`)
      return e.a !== e.b
    })

  // Break cycles: a depth-first back edge (to a node still on the stack) closes a cycle and is left out of the layering.
  const graph = fromEdges(
    ids.length,
    edges.map((e) => [e.a, e.b] as const),
  )
  const { edgeClass } = depthFirstSearch(graph)
  const dag = edges.filter((_, i) => edgeClass[i] !== 'back')
  const preds: number[][] = ids.map(() => [])
  const succs: number[][] = ids.map(() => [])
  for (const e of dag) {
    preds[e.b].push(e.a)
    succs[e.a].push(e.b)
  }

  // Layers: longest path from the sources, visiting nodes in Kahn's topological order, unless pinned.
  const layer = ids.map(() => 0)
  const { order: kahnOrder } = topologicalSort(
    fromEdges(
      ids.length,
      dag.map((e) => [e.a, e.b] as const),
    ),
  )
  const order = Array.from(kahnOrder.data)
  for (const v of order) {
    const pinned = spec.nodes[v].layer
    if (pinned !== undefined) layer[v] = pinned
    for (const w of succs[v]) layer[w] = Math.max(layer[w], layer[v] + 1)
  }
  if (compactSources)
    for (const v of order)
      if (preds[v].length === 0 && succs[v].length > 0 && spec.nodes[v].layer === undefined)
        layer[v] = Math.max(layer[v], Math.min(...succs[v].map((w) => layer[w])) - 1)

  // Virtual nodes split every edge that spans more than one layer, so it can be ordered and routed layer by layer.
  const real = ids.length
  const slotLayer = [...layer]
  const links: [number, number][] = []
  const chains: { edge: number; path: number[]; flipped: boolean }[] = []
  for (const e of edges) {
    const flipped = layer[e.a] > layer[e.b]
    const [lo, hi] = flipped ? [e.b, e.a] : [e.a, e.b]
    if (layer[hi] === layer[lo]) continue
    const path = [lo]
    for (let l = layer[lo] + 1; l < layer[hi]; l++) {
      const s = slotLayer.length
      slotLayer.push(l)
      path.push(s)
    }
    path.push(hi)
    for (let i = 0; i + 1 < path.length; i++) links.push([path[i], path[i + 1]])
    chains.push({ edge: e.k, path, flipped })
  }
  const slots = slotLayer.length
  const layers: number[][] = []
  slotLayer.forEach((l, s) => (layers[l] ??= []).push(s))
  for (let l = 0; l < layers.length; l++) layers[l] ??= []
  const up: number[][] = Array.from({ length: slots }, () => [])
  const down: number[][] = Array.from({ length: slots }, () => [])
  for (const [a, b] of links) {
    down[a].push(b)
    up[b].push(a)
  }

  // Order each layer by barycentre sweeps and adjacent transpositions, keeping the order with the fewest crossings.
  const pos = new Map<number, number>()
  const setPositions = (rows: number[][]) => rows.forEach((row) => row.forEach((s, i) => pos.set(s, i)))
  const total = (rows: number[][]) => {
    setPositions(rows)
    let c = 0
    for (let l = 0; l + 1 < rows.length; l++)
      c += crossings(rows[l].flatMap((a) => down[a].map((b): [number, number] => [pos.get(a)!, pos.get(b)!])))
    return c
  }
  /** Crossings among the edges of u and v (u placed before v) towards one neighbouring layer. */
  const pairCrossings = (u: number, v: number, ns: number[][]) => {
    let c = 0
    for (const a of ns[u]) for (const b of ns[v]) if (pos.get(a)! > pos.get(b)!) c++
    return c
  }
  const transpose = (rows: number[][]) => {
    setPositions(rows)
    for (let pass = 0, improved = true; improved && pass < 4; pass++) {
      improved = false
      for (const row of rows)
        for (let i = 0; i + 1 < row.length; i++) {
          const [u, v] = [row[i], row[i + 1]]
          const now = pairCrossings(u, v, up) + pairCrossings(u, v, down)
          const swapped = pairCrossings(v, u, up) + pairCrossings(v, u, down)
          if (swapped < now) {
            row[i] = v
            row[i + 1] = u
            pos.set(v, i)
            pos.set(u, i + 1)
            improved = true
          }
        }
    }
  }
  let rows = layers.map((r) => [...r])
  let best = rows.map((r) => [...r])
  let bestCount = total(rows)
  const reorder = (row: number[], neighbours: number[][]) => {
    const bary = new Map(
      row.map((s, i) => {
        const ns = neighbours[s]
        return [s, ns.length ? ns.reduce((acc, n) => acc + pos.get(n)!, 0) / ns.length : i] as const
      }),
    )
    return [...row].sort((a, b) => bary.get(a)! - bary.get(b)!)
  }
  for (let it = 0; it < sweeps && bestCount > 0; it++) {
    const downward = it % 2 === 0
    const ls = rows.map((_, l) => l)
    for (const l of downward ? ls.slice(1) : ls.slice(0, -1).reverse()) {
      setPositions(rows)
      rows[l] = reorder(rows[l], downward ? up : down)
    }
    transpose(rows)
    const count = total(rows)
    if (count < bestCount) {
      bestCount = count
      best = rows.map((r) => [...r])
    }
  }
  rows = best

  // Sizes along the flow (a layer's thickness) and across it (a node's extent within its layer); virtual nodes are
  // points.
  const sizes = spec.nodes.map(sizeOf)
  // A note or outside label on a side takes room there: across the layer it widens the gap on that side only.
  const noted = (s: number, side: Side) =>
    s < real &&
    (spec.nodes[s].notes?.[side] !== undefined ||
      (spec.nodes[s].label !== undefined &&
        (spec.nodes[s].labelSide ?? (['factor', 'dot'].includes(spec.nodes[s].shape ?? '') ? 'n' : undefined)) ===
          side))
  const [before, after, ends]: [Side, Side, Side[]] =
    direction === 'right' ? ['n', 's', ['e', 'w']] : ['w', 'e', ['n', 's']]
  const alongSize = (s: number) =>
    s < real ? sizes[s][direction === 'right' ? 0 : 1] + (ends.some((e) => noted(s, e)) ? 2 * NOTE : 0) : 0
  const acrossHalf = (s: number, side: Side) =>
    s < real ? sizes[s][direction === 'right' ? 1 : 0] / 2 + (noted(s, side) ? NOTE : 0) : 0
  const gapBetween = (a: number, b: number) =>
    a < real && b < real
      ? Math.max(nodeGap, acrossHalf(a, after) + acrossHalf(b, before) + CLEAR_ACROSS)
      : Math.max(nodeGap / 2, acrossHalf(a, after) + acrossHalf(b, before) + CLEAR_VIRTUAL)

  // Positions across each layer: start packed and centred, then pull each node towards its neighbours' mean, keeping
  // the order and each pair's gap.
  const across = new Map<number, number>()
  const rowGaps = rows.map((row) => row.slice(1).map((s, i) => gapBetween(row[i], s)))
  rows.forEach((row, l) => {
    const width = rowGaps[l].reduce((a, b) => a + b, 0)
    let at = -width / 2
    row.forEach((s, i) => {
      across.set(s, at)
      at += rowGaps[l][i] ?? 0
    })
  })
  for (let it = 0; it < 8; it++) {
    const ls = rows.map((_, l) => l)
    for (const l of it % 2 === 0 ? ls : ls.reverse()) {
      const row = rows[l]
      const desired = row.map((s) => {
        const ns = [...up[s], ...down[s]]
        return ns.length ? ns.reduce((acc, n) => acc + across.get(n)!, 0) / ns.length : across.get(s)!
      })
      spaced(desired, rowGaps[l]).forEach((y, i) => across.set(row[i], y))
    }
  }
  // Then align: each node claims the line of its median neighbour on the side the sweep comes from, unless an earlier
  // node in its layer claimed that neighbour (Brandes and Köpf's vertical alignment, simplified). Claimed lines weigh
  // heavily in the spacing, so chains run straight and only the unclaimed nodes bend around them.
  for (const downward of [true, false, true]) {
    const ls = rows.map((_, l) => l)
    for (const l of downward ? ls.slice(1) : ls.slice(0, -1).reverse()) {
      const row = rows[l]
      const claimed = new Set<number>()
      let last = -Infinity
      const weights: number[] = []
      const desired = row.map((s) => {
        const ns = [...(downward ? up : down)[s]].sort((p, q) => across.get(p)! - across.get(q)!)
        const m = ns[(ns.length - 1) >> 1]
        if (m === undefined || claimed.has(m) || across.get(m)! <= last) {
          weights.push(1)
          return across.get(s)!
        }
        claimed.add(m)
        last = across.get(m)!
        weights.push(100)
        return last
      })
      spaced(desired, rowGaps[l], weights).forEach((y, i) => across.set(row[i], y))
    }
  }
  const values = [...across.values()]
  const mid = values.length ? (Math.min(...values) + Math.max(...values)) / 2 : 0
  // Rounded, so nodes snapped onto one line share it exactly and their edges need no hairline jog.
  for (const [s, a] of across) across.set(s, Math.round((a - mid) * 1e6) / 1e6)

  // Right-angled edges without ports or waypoints of their own are routed by the layout: each link bends on a track in
  // the gap between its two layers. Links of one gap that overlap get separate tracks unless they share an end.
  const routed = new Set(
    chains
      .filter(({ edge }) => {
        const e = spec.edges![edge]
        return !e.via && (e.route ?? 'ortho') === 'ortho' && !parseEnd(e.from).side && !parseEnd(e.to).side
      })
      .map((c) => c.edge),
  )
  type Segment = { a: number; b: number; lo: number; hi: number; track: number }
  const gaps: Segment[][] = rows.map(() => [])
  const segmentOf = new Map<string, Segment>()
  for (const c of chains) {
    if (!routed.has(c.edge)) continue
    for (let i = 0; i + 1 < c.path.length; i++) {
      const [a, b] = [c.path[i], c.path[i + 1]]
      const key = `${a}>${b}`
      const [ya, yb] = [across.get(a)!, across.get(b)!]
      if (segmentOf.has(key) || Math.abs(ya - yb) < 1e-6) continue
      const seg = { a, b, lo: Math.min(ya, yb), hi: Math.max(ya, yb), track: 0 }
      segmentOf.set(key, seg)
      gaps[slotLayer[a]].push(seg)
    }
  }
  const trackCount = gaps.map((segs) => {
    const tracks: Segment[][] = []
    for (const seg of [...segs].sort((p, q) => p.lo - q.lo || p.hi - q.hi)) {
      const fits = (t: Segment[]) =>
        t.every((o) => o.a === seg.a || o.b === seg.b || o.hi < seg.lo - 0.05 || seg.hi < o.lo - 0.05)
      let t = tracks.findIndex(fits)
      if (t < 0) t = tracks.push([]) - 1
      tracks[t].push(seg)
      seg.track = t
    }
    return tracks.length
  })

  // Positions along the flow: each layer as thick as its thickest node, with room between layers for the tracks.
  const half = rows.map((row) => Math.max(0, ...row.map((s) => alongSize(s) / 2)))
  const along: number[] = [0]
  for (let l = 1; l < rows.length; l++)
    along[l] =
      along[l - 1] + Math.max(layerGap, half[l - 1] + half[l] + Math.max(CLEAR_ALONG, TRACK * (trackCount[l - 1] + 1)))
  const trackAt = (l: number, t: number) => {
    const start = along[l] + half[l]
    const end = along[l + 1] - half[l + 1]
    return start + ((t + 1) * (end - start)) / (trackCount[l] + 1)
  }

  const toXY = (al: number, ac: number): [number, number] => (direction === 'right' ? [al, ac] : [ac, al])
  const place = (s: number) => toXY(along[slotLayer[s]], across.get(s)!)

  const via = new Map<number, [number, number][]>()
  for (const c of chains) {
    let pts: [number, number][]
    if (routed.has(c.edge)) {
      pts = []
      for (let i = 0; i + 1 < c.path.length; i++) {
        const seg = segmentOf.get(`${c.path[i]}>${c.path[i + 1]}`)
        if (!seg) continue
        const at = trackAt(slotLayer[c.path[i]], seg.track)
        pts.push(toXY(at, across.get(c.path[i])!), toXY(at, across.get(c.path[i + 1])!))
      }
    } else pts = c.path.slice(1, -1).map(place)
    if (pts.length) via.set(c.edge, c.flipped ? pts.reverse() : pts)
  }
  return {
    ...spec,
    layout: 'manual',
    nodes: spec.nodes.map((n, i) => {
      const [x, y] = place(i)
      return { ...n, x, y }
    }),
    edges: spec.edges?.map((e, k): DiagramEdge => (e.via || !via.has(k) ? e : { ...e, via: via.get(k) })),
  }
}
