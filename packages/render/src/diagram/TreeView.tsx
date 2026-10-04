import { pathToRoot, type Tree } from 'aifn-compute/graph'
import { useMemo, useState, type KeyboardEvent } from 'react'
import { Diagram } from './Diagram'
import { treeLayout } from './tree'
import type { DiagramEdge, DiagramNode, DiagramSpec, ElementState, Shape, Side, Tone } from './types'

/** A value per item: an array indexed by id, or a function of the id. */
export type PerItem<T> = readonly (T | undefined)[] | ((i: number) => T | undefined)

const at = <T,>(p: PerItem<T> | undefined, i: number): T | undefined =>
  p === undefined ? undefined : typeof p === 'function' ? p(i) : p[i]

/** Rough width in grid units of a label at the diagram's label size: maths counts by its visible characters. */
function textWidth(text: string): number {
  const plain = text
    .replace(/\$([^$]*)\$/g, (_, m: string) =>
      m
        .replace(/\\[a-zA-Z]+/g, 'x')
        .replace(/[{}^_\\ ]/g, '')
        .replace(/./g, 'x'),
    )
    .split('\n')
    .reduce((a, b) => (a.length >= b.length ? a : b), '')
  return (plain.length * 13 * 0.55 + 14) / 40
}

export type TreeViewProps = {
  /** An `aifn-compute/graph` `Tree`. Node labels come from `node.label`, else the id. */
  tree: Tree
  ariaLabel?: string
  /** `down` (root at the top, default) or `right` (root at the left). */
  orientation?: 'down' | 'right'
  /**
   * Place nodes by their `height` (a dendrogram: leaves at the bottom, merges at their distance) with elbow edges,
   * instead of by depth with straight edges. A number sets grid units per unit of height.
   */
  heightAxis?: boolean | number
  /** Gap between neighbouring nodes, and between levels, in grid units. */
  siblingGap?: number
  levelGap?: number
  /** Node labels (TeX in `$…$`), overriding `node.label`; an empty string draws an unlabelled node. */
  nodeLabels?: PerItem<string>
  /**
   * A note beside each node, e.g. a probability: a string goes below a leaf and beside an internal node (outside the
   * edges), or give the sides explicitly.
   */
  nodeNotes?: PerItem<string | Partial<Record<Side, string>>>
  /** Edge labels by child id (the edge parent → child), overriding `tree.edges[child].label`. */
  edgeLabels?: PerItem<string>
  /** Step state of each node, and of each edge by child id (`idle` dimmed, `active` emphasised, `done` plain). */
  nodeState?: PerItem<ElementState>
  edgeState?: PerItem<ElementState>
  nodeTone?: PerItem<Tone>
  edgeTone?: PerItem<Tone>
  /** Nodes to highlight; an edge is highlighted when both its ends are, so a path lights up as a path. */
  highlight?: Iterable<number>
  /** Nodes not drawn (with their edges), keeping the layout of the whole tree, e.g. nodes not yet created. */
  hidden?: PerItem<boolean>
  /** Node shape; default circles, or boxes when a label is wider than a circle. Leaves may differ from the rest. */
  shape?: Shape | ((id: number) => Shape)
  /** Clicking an internal node collapses or expands it; collapsed nodes show how many nodes they hide. */
  collapsible?: boolean
  /** Nodes collapsed at first (with `collapsible`), or every node at this depth. */
  defaultCollapsed?: readonly number[] | { depth: number }
  /** Further overrides of each node's or edge's (by child id) diagram spec. */
  node?: (id: number) => Partial<DiagramNode>
  edge?: (child: number) => Partial<DiagramEdge>
  /** Height in pixels, or `fill`; by default the Figure frame's height. */
  height?: number | 'fill'
  onNodeClick?: (id: number) => void
  onNodeHover?: (id: number | null) => void
  /**
   * The chosen node: ringed, with its path from the root emphasised. With `onSelect`, nodes are clickable (and
   * focusable: Tab and Enter), and the arrow keys move the choice (up to the parent, down to the first child, left and
   * right to the neighbouring sibling).
   */
  selected?: number | null
  onSelect?: (id: number) => void
  /**
   * Nodes cut back to leaves (pruning): drawn as leaves, with their descendants hidden and the layout of the whole tree
   * kept, so the tree shrinks in place.
   */
  cut?: readonly number[]
  /** A path of node ids (e.g. a point's decision path): its edges emphasised and every other node dimmed. */
  path?: readonly number[]
  /**
   * What each node holds, drawn in it: class shares as a stacked bar in the class colours (thin under an internal
   * node's label; a leaf's label is the majority share), or a mean; with `n` beneath leaves. By default read from a
   * decision tree's nodes (`value`, `count`), so any `aifn-compute` decision tree shows its class proportions; `false` turns
   * it off. Large trees fall back to a bar alone in small leaves.
   */
  nodeSummary?: false | ((id: number) => NodeSummary | undefined)
}

/** What a node holds: class shares (in palette slots, by class) or a mean, and its number of rows. */
export type NodeSummary = { shares?: readonly number[]; mean?: number; n?: number }

type SummaryNode = { value?: unknown; count?: unknown }

/** A decision tree's node summaries (duck-typed on `task`, `value` and `count`), or undefined for other trees. */
function decisionSummary(tree: Tree): ((id: number) => NodeSummary) | undefined {
  const task = (tree as { task?: unknown }).task
  const first = tree.nodes[0] as SummaryNode | undefined
  if ((task !== 'classification' && task !== 'regression') || !first || !Array.isArray(first.value)) return undefined
  return (id) => {
    const node = tree.nodes[id] as SummaryNode
    const value = node.value as number[]
    const n = typeof node.count === 'number' ? node.count : undefined
    return task === 'classification' ? { shares: value, n } : { mean: value[0], n }
  }
}

/** Leaves beyond which leaf labels and notes give way to a bar alone. */
const COMPACT_LEAVES = 14
/** Leaves beyond which internal nodes also show a bar alone. */
const TINY_LEAVES = 28
const fmt = (v: number) => Number(v.toPrecision(2)).toString()

const nid = (v: number) => `t${v}`

/**
 * An `aifn-compute/graph` `Tree` drawn with the diagram system and the tidy `treeLayout`: parents centred over their
 * children, binary trees keeping left and right, or a dendrogram by node height with elbow edges. Nodes carry KaTeX
 * labels and a note (e.g. a probability), edges a label (e.g. 0/1); nodes and edges take step states, tones and a
 * highlighted path; hidden nodes keep their place; internal nodes can collapse. It sizes from the `Figure` frame.
 */
export function TreeView({
  tree,
  ariaLabel = 'A tree',
  orientation = 'down',
  heightAxis = false,
  siblingGap,
  levelGap,
  nodeLabels,
  nodeNotes,
  edgeLabels,
  nodeState,
  edgeState,
  nodeTone,
  edgeTone,
  highlight,
  hidden,
  shape,
  collapsible = false,
  defaultCollapsed,
  node,
  edge,
  height,
  onNodeClick,
  onNodeHover,
  selected = null,
  onSelect,
  path,
  cut,
  nodeSummary,
}: TreeViewProps) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<number>>(() => {
    if (!collapsible || !defaultCollapsed) return new Set()
    if ('depth' in defaultCollapsed) {
      const out = new Set<number>()
      const walk = (v: number, d: number) => {
        if (d === defaultCollapsed.depth && tree.nodes[v].children.length) out.add(v)
        else tree.nodes[v].children.forEach((c) => walk(c, d + 1))
      }
      walk(tree.root, 0)
      return out
    }
    return new Set(defaultCollapsed)
  })
  // A collapse set from another tree is dropped rather than applied to the wrong nodes.
  const live = useMemo(
    () => new Set([...collapsed].filter((v) => v < tree.nodes.length && tree.nodes[v].children.length)),
    [collapsed, tree],
  )

  const summaryOf = useMemo(
    () => (nodeSummary === false ? undefined : (nodeSummary ?? decisionSummary(tree))),
    [nodeSummary, tree],
  )
  const emphasis = useMemo(() => {
    const onPath = new Set(
      path ?? (selected !== null && selected < tree.nodes.length ? pathToRoot(tree, selected) : []),
    )
    return { onPath, dim: path !== undefined && path.length > 0 }
  }, [path, selected, tree])

  const pruned = useMemo(() => {
    const cutSet = new Set(cut ?? [])
    const below = new Set<number>()
    const stack = [...cutSet].flatMap((v) => tree.nodes[v]?.children ?? [])
    while (stack.length) {
      const w = stack.pop()!
      below.add(w)
      stack.push(...tree.nodes[w].children)
    }
    return { cutSet, below }
  }, [cut, tree])

  const spec = useMemo((): DiagramSpec => {
    const down = orientation === 'down'
    const isLeaf = (v: number) => tree.nodes[v].children.length === 0 || pruned.cutSet.has(v)
    const leafCount = tree.nodes.filter((n) => isLeaf(n.id) && !pruned.below.has(n.id)).length
    const compact = leafCount > COMPACT_LEAVES
    // Very large trees: internal nodes give way to their bar too (the rule is in the node's accessible name).
    const tiny = leafCount > TINY_LEAVES
    const summaries = tree.nodes.map((n) => summaryOf?.(n.id))
    const leaf = (v: number) => isLeaf(v) || live.has(v)
    const summaryLabel = (v: number): string | undefined => {
      const sm = summaries[v]
      if (!sm || !isLeaf(v)) return undefined
      if (compact) return ''
      if (sm.mean !== undefined) return `$${fmt(sm.mean)}$`
      if (!sm.shares) return undefined
      const total = sm.shares.reduce((a, b) => a + b, 0) || 1
      const top = sm.shares.reduce((b, v2, k) => (v2 > sm.shares![b] ? k : b), 0)
      return `${fmt(sm.shares[top] / total)} class ${top}`
    }
    const labelOf = (v: number) =>
      at(nodeLabels, v) ?? summaryLabel(v) ?? (tiny && summaries[v]?.shares ? '' : (tree.nodes[v].label ?? String(v)))
    const labels = tree.nodes.map((n) => labelOf(n.id))
    const shapeOf = (v: number): Shape => {
      if (typeof shape === 'function') return shape(v)
      if (shape) return shape
      return textWidth(labels[v]) > 0.95 ? 'box' : 'circle'
    }
    // Hidden descendants of each collapsed node.
    const hiddenCount = new Map<number, number>()
    for (const v of live) {
      let count = 0
      const stack = [...tree.nodes[v].children]
      while (stack.length) {
        const w = stack.pop()!
        count++
        stack.push(...tree.nodes[w].children)
      }
      hiddenCount.set(v, count)
    }
    const notesOf = (v: number): Partial<Record<Side, string>> | undefined => {
      const sm = summaries[v]
      const note =
        at(nodeNotes, v) ?? (sm?.n !== undefined && !compact && isLeaf(v) && !live.has(v) ? `n = ${sm.n}` : undefined)
      const extra = hiddenCount.has(v) ? `+${hiddenCount.get(v)}` : undefined
      if (note === undefined) return extra ? { [down ? 's' : 'e']: extra } : undefined
      if (typeof note !== 'string') return extra ? { ...note, [down ? 's' : 'e']: extra } : note
      const side: Side = leaf(v) ? (down ? 's' : 'e') : down ? 'e' : 'n'
      return extra && side === (down ? 's' : 'e')
        ? { [side]: `${note} (${extra})` }
        : { [side]: note, ...(extra && { [down ? 's' : 'e']: extra }) }
    }
    const notes = tree.nodes.map((n) => notesOf(n.id))
    // Sizes: a circle grows to fit its label; a box is as wide as its label. Across the levels a node also needs room
    // for notes above or below it (down) or beside it (right).
    const dims = tree.nodes.map((n) => {
      const s = shapeOf(n.id)
      const w = textWidth(labels[n.id])
      if (s === 'circle') {
        const d = Math.max(0.9, w)
        return [d, d]
      }
      if (s === 'dot') return [0.14, 0.14]
      const sm = summaries[n.id]
      if (sm?.shares && labels[n.id] === '') return [0.5, 0.36]
      return [Math.max(0.9, w), sm?.shares ? 0.85 : 0.7]
    })
    const acrossSize = (v: number) => {
      const [w, h] = dims[v]
      const ns = notes[v] ?? {}
      if (down) {
        const below = Math.max(ns.s ? textWidth(ns.s) : 0, ns.n ? textWidth(ns.n) : 0)
        const beside = (ns.e ? textWidth(ns.e) + 0.1 : 0) + (ns.w ? textWidth(ns.w) + 0.1 : 0)
        return Math.max(w, below) + 2 * beside
      }
      return h + (ns.n ? 0.5 : 0) + (ns.s ? 0.5 : 0)
    }
    const maxAlong = Math.max(
      ...tree.nodes.map((n) => {
        const [w, h] = dims[n.id]
        const ns = notes[n.id] ?? {}
        return down ? h + (ns.s ? 0.5 : 0) + (ns.n ? 0.5 : 0) : w + (ns.e ? textWidth(ns.e) : 0)
      }),
    )
    const labelled = tree.nodes.some((n) => n.parent !== null && (at(edgeLabels, n.id) ?? tree.edges[n.id]?.label))
    const positions = treeLayout(tree, {
      orientation,
      siblingGap: siblingGap ?? 0.35,
      // Labelled edges need room between levels for their labels beside the notes above and below.
      levelGap: levelGap ?? maxAlong + (down ? 0.8 : 1.2) + (labelled ? 0.5 : 0),
      size: acrossSize,
      heightAxis,
      collapsed: live,
    })
    const lit = new Set(highlight ?? [])
    const visible = (v: number) => positions[v] !== null && !at(hidden, v) && !pruned.below.has(v)
    const nodes = tree.nodes
      .filter((n) => visible(n.id))
      .map((n): DiagramNode => {
        const p = positions[n.id]!
        const [w, h] = dims[n.id]
        return {
          id: nid(n.id),
          shape: shapeOf(n.id),
          x: p.x,
          y: p.y,
          w,
          h,
          label: labels[n.id] || undefined,
          ariaLabel: tree.nodes[n.id].label ?? `node ${n.id}`,
          tone: at(nodeTone, n.id) ?? 'ink',
          state: at(nodeState, n.id) ?? (emphasis.dim && !emphasis.onPath.has(n.id) ? 'idle' : undefined),
          highlight: lit.has(n.id) || undefined,
          ...(summaries[n.id]?.shares && {
            bar: summaries[n.id]!.shares,
            barHeight: leaf(n.id) ? 5 : 3,
          }),
          ...(onSelect && { selected: n.id === selected }),
          notes: notes[n.id],
          dashed: live.has(n.id) || undefined,
          ...node?.(n.id),
        }
      })
    const edges = tree.nodes
      .filter((n) => n.parent !== null && visible(n.id) && visible(n.parent))
      .map((n): DiagramEdge => {
        const c = n.id
        const p = n.parent!
        const pc = positions[c]!
        const pp = positions[p]!
        const label = at(edgeLabels, c) ?? tree.edges[c]?.label
        // A dendrogram's edge leaves its parent sideways and drops to the child at a right angle.
        const elbow = heightAxis !== false
        let from = nid(p)
        if (elbow) {
          const offset = down ? pc.x - pp.x : pc.y - pp.y
          from +=
            Math.abs(offset) < 1e-9 ? (down ? ':s' : ':e') : offset < 0 ? (down ? ':w' : ':n') : down ? ':e' : ':s'
        }
        return {
          from,
          to: elbow ? `${nid(c)}:${down ? 'n' : 'w'}` : nid(c),
          route: elbow ? 'ortho' : 'straight',
          arrow: 'none',
          label,
          labelRotate: false,
          // Labels sit outside the fork: left of a left child's edge, right of a right child's.
          labelSide: (down ? pc.x < pp.x : pc.y > pp.y) ? 'right' : 'left',
          ...(label !== undefined && !elbow && { labelPos: 0.45 }),
          tone: at(edgeTone, c),
          state:
            at(edgeState, c) ??
            (emphasis.onPath.has(c) && emphasis.onPath.has(p) ? 'active' : emphasis.dim ? 'idle' : undefined),
          highlight: (lit.has(c) && lit.has(p)) || undefined,
          ...edge?.(c),
        }
      })
    // Hidden nodes keep their room: the view frames the whole tree, so it keeps its scale as nodes appear or go.
    const placed = tree.nodes.flatMap((n) => {
      const q = positions[n.id]
      // Pruned nodes give their room back, so a pruned tree grows to fill the view.
      if (!q || pruned.below.has(n.id)) return []
      const [w, h] = dims[n.id]
      return [{ x0: q.x - w / 2, y0: q.y - h / 2, x1: q.x + w / 2, y1: q.y + h / 2 + 0.5 }]
    })
    const frame = placed.length
      ? {
          x0: Math.min(...placed.map((b) => b.x0)),
          y0: Math.min(...placed.map((b) => b.y0)),
          x1: Math.max(...placed.map((b) => b.x1)),
          y1: Math.max(...placed.map((b) => b.y1)),
        }
      : undefined
    // Class colours fill the nodes, so emphasis is drawn in ink rather than a data colour.
    return { nodes, edges, frame, ...(summaries.some((v) => v?.shares) && { accent: 'ink' as const }) }
  }, [
    tree,
    orientation,
    heightAxis,
    siblingGap,
    levelGap,
    nodeLabels,
    nodeNotes,
    edgeLabels,
    nodeState,
    edgeState,
    nodeTone,
    edgeTone,
    highlight,
    hidden,
    shape,
    live,
    node,
    edge,
    summaryOf,
    emphasis,
    pruned,
    selected,
    onSelect,
  ])

  const toggle = (v: number) =>
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(v)) next.delete(v)
      else next.add(v)
      return next
    })
  const click =
    onNodeClick || collapsible || onSelect
      ? (s: string) => {
          const v = Number(s.slice(1))
          if (collapsible && tree.nodes[v].children.length) toggle(v)
          onNodeClick?.(v)
          onSelect?.(v)
        }
      : undefined
  const shown = (v: number | undefined | null) =>
    v !== undefined && v !== null && v < tree.nodes.length && !at(hidden, v) && !pruned.below.has(v)
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!onSelect) return
    const keys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']
    if (!keys.includes(event.key)) return
    event.preventDefault()
    if (selected === null || !shown(selected)) return onSelect(tree.root)
    const node = tree.nodes[selected]
    let next: number | null | undefined = null
    if (event.key === 'ArrowUp') next = node.parent
    else if (event.key === 'ArrowDown')
      next = live.has(selected) || pruned.cutSet.has(selected) ? null : node.children.find((c) => shown(c))
    else if (node.parent !== null) {
      const siblings = tree.nodes[node.parent].children.filter((c) => shown(c))
      next = siblings[siblings.indexOf(selected) + (event.key === 'ArrowLeft' ? -1 : 1)]
    }
    if (shown(next)) onSelect(next!)
  }
  const diagram = (
    <Diagram
      spec={spec}
      ariaLabel={ariaLabel}
      height={height}
      onNodeClick={click}
      onNodeHover={onNodeHover && ((s) => onNodeHover(s === null ? null : Number(s.slice(1))))}
    />
  )
  if (!onSelect) return diagram
  return (
    <div
      role="group"
      aria-label={`${ariaLabel}: click a node, or use the arrow keys, to choose it`}
      onKeyDown={onKeyDown}
      style={{ display: 'contents' }}
    >
      {diagram}
    </div>
  )
}
