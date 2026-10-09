/**
 * Diagram data for a structured graph: plain data in the shape of the lab's `DiagramSpec`
 * (`packages/render/src/diagram/types.ts`), with no React. Variables are circles (observed shaded, deterministic
 * dashed, parameters small), factors small squares, groups drawn as plates labelled bottom right (Buntine 1994),
 * directed edges with arrows and undirected ones without. aifn emits the data; the lab draws it.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { Index } from 'aifn-compute/foundation/contracts'
import { groupChain, groupSizes } from './build'
import { markovBlanket } from './shape'
import type { Lag, StructuredGraph } from './types'

/** A diagram node, in the shape of the lab's `DiagramNode`. */
export interface DiagramNodeData {
  /** The node's name in the graph. */
  id: string
  /** Horizontal centre in grid units; left out when the lab's layered layout places the node. */
  x?: number
  /** Vertical centre in grid units (growing downwards); left out with `x`. */
  y?: number
  /** A circle for a variable, a small square for a factor, or bare text. */
  shape: 'circle' | 'factor' | 'text'
  /** Text with TeX between dollar signs. */
  label?: string
  /** A palette slot, or a neutral tone. */
  tone?: number | 'neutral' | 'ink'
  /** Shaded: an observed variable. */
  filled?: boolean
  /** Dashed outline: a deterministic node. */
  dashed?: boolean
  /** Small label text: a parameter. */
  small?: boolean
  /** Width in grid units (default: the shape's own). */
  w?: number
  /** Height in grid units (default: the shape's own). */
  h?: number
  /** Which side of the shape the label sits outside (a factor's sits above, `n`). */
  labelSide?: 'n' | 's' | 'e' | 'w'
  /** Drawn in the accent colour: the highlighted node. */
  highlight?: boolean
  /** With a highlight: `active` for the node, `done` for its blanket, `idle` for the rest. */
  state?: 'idle' | 'active' | 'done'
}

/** A diagram edge, in the shape of the lab's `DiagramEdge` (a self-edge is a loop). */
export interface DiagramEdgeData {
  /** The name of the node the edge leaves. */
  from: string
  /** The name of the node the edge enters. */
  to: string
  /** A direct line, as is usual in graphical models. */
  route: 'straight'
  /** An arrowhead at `to` for a directed edge, none for an undirected one, or one halfway. */
  arrow: 'end' | 'none' | 'mid'
  /** The edge's label, or a lagged edge's lag (TeX between dollar signs). */
  label?: string
  /** Drawn in the accent colour. */
  highlight?: boolean
  /** With a highlight: `done` when both ends are in the highlighted node's blanket or are it, `idle` otherwise. */
  state?: 'idle' | 'active' | 'done'
}

/** A group (a plate or a template), in the shape of the lab's `DiagramGroup`. */
export interface DiagramGroupData {
  /** `group ` followed by the group's name. */
  id: string
  /** The group's label, or its sizes joined by $\times$, as TeX. */
  label?: string
  /** Drawn in ink. */
  tone?: 'ink'
  /** The names of the nodes the plate is drawn around: every node inside the group or a group nested in it. */
  around: string[]
  /** Margin around the nodes in grid units, larger for a group with groups nested inside it. */
  pad: number
  /** Where the label sits: bottom right, as in plate notation. */
  labelAt: 'bottom-right'
}

/** A diagram spec (the lab's `DiagramSpec`). */
export interface DiagramData {
  /** One per node of the graph, in node order. */
  nodes: DiagramNodeData[]
  /** One per edge of the graph, in edge order. */
  edges: DiagramEdgeData[]
  /** One per group that holds at least one node. */
  groups: DiagramGroupData[]
  /** `manual` when every node has a position, `layered` when the lab places them. */
  layout?: 'manual' | 'layered'
  /** The layered layout's flow direction and gaps (grid units), with `layout: 'layered'`. */
  layered?: { direction?: 'right' | 'down'; layerGap?: number; nodeGap?: number }
}

/** Options of {@link toDiagram}. */
export interface DiagramOptions {
  /** Fixed centres in grid units, by node name. Factors without one sit at the mean of their placed neighbours. */
  positions?: Readonly<Record<string, readonly [number, number]>>
  /** Labels (TeX allowed) by node name, overriding the nodes' own. */
  labels?: Readonly<Record<string, string>>
  /** Mark this node active, its Markov blanket and the factors joining them done, and everything else idle. */
  highlight?: string
  /** Distance between layers, in grid units (default 1.8). */
  layerGap?: number
  /** Least distance between nodes in a layer, in grid units (default 1.6). */
  nodeGap?: number
}

/**
 * A label as TeX: `s` between dollar signs, unless it already holds one (then it is mixed text and TeX, kept as is).
 *
 * @param s The label.
 * @returns The label ready for the lab.
 */
const tex = (s: string) => (s.includes('$') ? s : `$${s}$`)

/**
 * A lag as the label of a template's self-loop, in TeX: `$t-1$`, `$(i-1, j)$`, or `parent` as plain text.
 *
 * @param lag The edge's lag.
 * @param index The index symbols of the template group, one per axis.
 * @returns The label.
 */
function lagLabel(lag: Lag, index: readonly string[]): string {
  if (lag === 'parent') return 'parent'
  const ls = typeof lag === 'number' ? [lag] : lag
  const at = ls.map((l, a) => (l === 0 ? index[a] : `${index[a]}${l > 0 ? '-' : '+'}${Math.abs(l)}`))
  return tex(at.length === 1 ? at[0] : `(${at.join(', ')})`)
}

/**
 * The diagram of a structured graph. When every edge is directed, nodes are placed in columns by depth along the
 * unlagged edges (the longest path from a source; parameters one column before their earliest child) and ordered in
 * each column by the mean height of their parents; `positions` overrides any of them. In a graph with undirected
 * edges only nodes given `positions` are placed, and a factor without one sits at the mean of its placed neighbours
 * (offset up and left when there is one). The layout is `manual` when every node ends up placed, and otherwise the
 * lab's layered layout, reading each edge from `from` to `to`. Each group is drawn around its nodes (outer groups
 * with more padding), labelled with its label or size; a lagged edge of a compact template is a labelled loop or
 * link. Throws `DomainError` when `highlight` names no node.
 *
 * @param graph The graph, compact or unrolled.
 * @param options Fixed positions and labels by node name, a node to highlight with its Markov blanket, and the gaps
 *   of the automatic placement.
 * @returns Plain data for the lab's diagram.
 *
 * @example The compact HMM: a loop labelled with its lag, inside a plate
 * const d = toDiagram(chainTemplate('T', { observed: 'x' }))
 * print('nodes:', d.nodes.map((n) => `${n.id} at (${n.x}, ${n.y})${n.filled ? ', shaded' : ''}`))
 * print('edges:', d.edges.map((e) => `${e.from} -> ${e.to} ${e.label ?? ''}`))
 * print('groups:', d.groups.map((g) => `${g.label} around ${g.around.join(', ')}`), 'layout:', d.layout)
 *
 * @example Highlighting a node's Markov blanket
 * const d = toDiagram(unroll(chainTemplate(3, { observed: 'x' })), { highlight: 'z[1]' })
 * print(d.nodes.map((n) => `${n.id}: ${n.state}`))
 */
export function toDiagram(graph: StructuredGraph, options: DiagramOptions = {}): DiagramData {
  const { layerGap = 1.8, nodeGap = 1.6 } = options
  const g = graph
  const at = g.attributes
  const role = (i: Index) => at[i].role
  // Highlight: the node, its blanket, and the factors between them.
  let state: ((i: Index) => 'idle' | 'active' | 'done') | null = null
  if (options.highlight !== undefined) {
    const centre = at.findIndex((n) => n.name === options.highlight)
    if (centre < 0) throw new DomainError('toDiagram', `toDiagram: no node named ${options.highlight}`)
    const blanket = new Set(markovBlanket({ ...g, unrolled: true }, options.highlight).blanket)
    const touching = new Set(
      g.edges.flatMap((e) =>
        e.from === centre && role(e.to) === 'factor'
          ? [e.to]
          : e.to === centre && role(e.from) === 'factor'
            ? [e.from]
            : [],
      ),
    )
    state = (i) => (i === centre ? 'active' : blanket.has(at[i].name) || touching.has(i) ? 'done' : 'idle')
  }
  // Placement along directed, unlagged edges.
  const links = g.edges.filter((e) => e.directed && e.lag === undefined && e.from !== e.to).map((e) => [e.from, e.to])
  const placed = new Map<Index, [number, number]>()
  const directedOnly = g.edges.every((e) => e.directed)
  if (directedOnly) {
    const layer = new Array<number>(g.nodes).fill(0)
    // Longest path from a source (declaration order need not be topological, so relax until stable).
    for (let pass = 0, changed = true; changed && pass <= g.nodes; pass++) {
      changed = false
      for (const [a, b] of links)
        if (layer[b] < layer[a] + 1) {
          layer[b] = layer[a] + 1
          changed = true
        }
    }
    at.forEach((n, i) => {
      if (n.role !== 'parameter') return
      const kids = links.filter(([a]) => a === i).map(([, b]) => layer[b])
      if (kids.length) layer[i] = Math.min(...kids) - 1
    })
    const y = new Array<number>(g.nodes).fill(0)
    for (const L of [...new Set(layer)].sort((a, b) => a - b)) {
      const members = at.map((_, i) => i).filter((i) => layer[i] === L)
      const want = members.map((i, k) => {
        const ps = links.filter(([a, b]) => b === i && layer[a] < L).map(([a]) => a)
        return ps.length ? ps.reduce((s, a) => s + y[a], 0) / ps.length : k * nodeGap
      })
      let last = -Infinity
      for (const { i, w } of members.map((i, k) => ({ i, w: want[k] })).sort((a, b) => a.w - b.w)) {
        y[i] = Math.max(w, last + nodeGap)
        last = y[i]
      }
    }
    at.forEach((_, i) => placed.set(i, [layer[i] * layerGap, y[i]]))
  }
  const pos = options.positions
  if (pos) at.forEach((n, i) => pos[n.name] && placed.set(i, [pos[n.name][0], pos[n.name][1]]))
  if (pos && !directedOnly)
    at.forEach((n, i) => {
      if (n.role !== 'factor' || placed.has(i)) return
      const around = g.edges
        .flatMap((e) => (e.from === i ? [e.to] : e.to === i ? [e.from] : []))
        .map((j) => placed.get(j))
        .filter((p): p is [number, number] => p !== undefined)
      if (!around.length) return
      const mean: [number, number] = [
        around.reduce((s, p) => s + p[0], 0) / around.length,
        around.reduce((s, p) => s + p[1], 0) / around.length,
      ]
      placed.set(i, around.length === 1 ? [mean[0] - 0.7, mean[1] - 0.7] : mean)
    })
  const nodes: DiagramNodeData[] = at.map((n, i) => {
    const p = placed.get(i)
    const s = state?.(i)
    const label = options.labels?.[n.name] ?? tex(n.label ?? n.name)
    const common = {
      id: n.name,
      ...(p ? { x: p[0], y: p[1] } : {}),
      ...(s ? { state: s } : {}),
      ...(s === 'active' ? { highlight: true } : {}),
    }
    if (n.role === 'factor') return { ...common, shape: 'factor', labelSide: 'n' }
    return {
      ...common,
      shape: 'circle',
      tone: 'ink',
      label,
      ...(n.role === 'observed' ? { filled: true } : {}),
      ...(n.role === 'deterministic' ? { dashed: true } : {}),
      ...(n.role === 'parameter' ? { small: true, w: 0.7, h: 0.7 } : {}),
    }
  })
  const edges: DiagramEdgeData[] = g.edges.map((e) => {
    const group = g.groups.find((q) => q.name === at[e.to].group)
    const label = e.label ?? (e.lag !== undefined && group ? lagLabel(e.lag, group.index) : undefined)
    const s = state ? (state(e.from) !== 'idle' && state(e.to) !== 'idle' ? 'done' : 'idle') : undefined
    return {
      from: at[e.from].name,
      to: at[e.to].name,
      route: 'straight',
      arrow: e.directed ? 'end' : 'none',
      ...(label === undefined ? {} : { label }),
      ...(s ? { state: s } : {}),
    }
  })
  const depthBelow = (name: string): number => {
    const kids = g.groups.filter((q) => q.parent === name)
    return kids.length ? 1 + Math.max(...kids.map((q) => depthBelow(q.name))) : 0
  }
  const groups: DiagramGroupData[] = g.groups
    .map((q) => ({
      id: `group ${q.name}`,
      label: tex(q.label ?? groupSizes(q).map(String).join(' \\times ')),
      tone: 'ink' as const,
      around: at.filter((n) => groupChain(g, n.group).some((c) => c.name === q.name)).map((n) => n.name),
      pad: 0.3 + 0.4 * depthBelow(q.name),
      labelAt: 'bottom-right' as const,
    }))
    .filter((q) => q.around.length > 0)
  const manual = nodes.every((n) => n.x !== undefined)
  return {
    nodes,
    edges,
    groups,
    ...(manual
      ? { layout: 'manual' as const }
      : { layout: 'layered' as const, layered: { direction: 'right' as const, layerGap: 3, nodeGap: 1 } }),
  }
}
