/**
 * Diagram data for a structured graph: plain data in the shape of the lab's `DiagramSpec`
 * (`packages/render/src/diagram/types.ts`), with no React. Variables are circles (observed shaded, deterministic dashed,
 * parameters small), factors small squares, groups drawn as plates labelled bottom right (Buntine 1994), directed
 * edges with arrows and undirected ones without. aifn emits the data; the lab draws it.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { Index } from 'aifn-compute/foundation/contracts'
import { groupChain, groupSizes } from './build'
import { markovBlanket } from './shape'
import type { Lag, StructuredGraph } from './types'

/** A diagram node, in the shape of the lab's `DiagramNode`. */
export interface DiagramNodeData {
  id: string
  x?: number
  y?: number
  shape: 'circle' | 'factor' | 'text'
  label?: string
  tone?: number | 'neutral' | 'ink'
  filled?: boolean
  dashed?: boolean
  small?: boolean
  w?: number
  h?: number
  labelSide?: 'n' | 's' | 'e' | 'w'
  highlight?: boolean
  state?: 'idle' | 'active' | 'done'
}

/** A diagram edge, in the shape of the lab's `DiagramEdge` (a self-edge is a loop). */
export interface DiagramEdgeData {
  from: string
  to: string
  route: 'straight'
  arrow: 'end' | 'none' | 'mid'
  label?: string
  highlight?: boolean
  state?: 'idle' | 'active' | 'done'
}

/** A group (a plate or a template), in the shape of the lab's `DiagramGroup`. */
export interface DiagramGroupData {
  id: string
  label?: string
  tone?: 'ink'
  around: string[]
  pad: number
  labelAt: 'bottom-right'
}

/** A diagram spec (the lab's `DiagramSpec`). */
export interface DiagramData {
  nodes: DiagramNodeData[]
  edges: DiagramEdgeData[]
  groups: DiagramGroupData[]
  layout?: 'manual' | 'layered'
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
  /** Distance between layers and between nodes in a layer, in grid units (default 1.8, 1.6). */
  layerGap?: number
  nodeGap?: number
}

const tex = (s: string) => (s.includes('$') ? s : `$${s}$`)

/** A lag as the label of a template's self-loop: `t−1`, `(i−1, j)`, `parent`. */
function lagLabel(lag: Lag, index: readonly string[]): string {
  if (lag === 'parent') return 'parent'
  const ls = typeof lag === 'number' ? [lag] : lag
  const at = ls.map((l, a) => (l === 0 ? index[a] : `${index[a]}${l > 0 ? '-' : '+'}${Math.abs(l)}`))
  return tex(at.length === 1 ? at[0] : `(${at.join(', ')})`)
}

/**
 * The diagram of a structured graph. Nodes are placed in columns by depth along directed edges (the longest path
 * from a source; parameters just before their first child) and ordered in each column by the mean height of their
 * parents; `positions` overrides any of them. A graph with undirected edges and no positions gets the lab's layered
 * layout instead, reading each edge from `from` to `to`. Each group is drawn around its nodes (outer groups with more
 * padding), labelled with its label or size; a lagged edge of a compact template is a labelled loop or link.
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
