import {
  useContext,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react'
import { chrome, seriesColor, useTheme, type Mode } from '../design'
import { FrameContext, useElementSize } from '../viz'
import { parseEnd } from './ends'
import { layeredLayout } from './layout'
import { nodeSize } from './size'
import { MathText } from './MathText'
import type {
  DiagramEdge,
  DiagramGroup,
  DiagramNode,
  DiagramSpec,
  Direction,
  ElementState,
  PlacedNode,
  Side,
  Tone,
} from './types'

type Pt = { x: number; y: number }
type Box = { x0: number; y0: number; x1: number; y1: number }

const ROUND = new Set(['circle', 'latent', 'noise', 'op', 'dot'])
const ACCENT_SLOT = 0
/** Ratio of the narrow to the wide side of an encoder or decoder trapezoid. */
const TAPER = 0.45
const LABEL_H = 0.42
/** Height of a note or an outside label, in grid units. */
const NOTE_H = 0.5
/** Font sizes in diagram pixels at natural size; they scale with the diagram. */
const FONT = { label: 13, small: 11, note: 12, group: 11 }
/** How long a change of state takes to show. */
const TRANSITION =
  'fill 180ms ease, stroke 180ms ease, stroke-width 180ms ease, opacity 180ms ease, fill-opacity 180ms ease'
const TEXT_TRANSITION = 'color 180ms ease, opacity 180ms ease'

const size = (n: PlacedNode): [number, number] => nodeSize(n)

function extent(n: PlacedNode): Box {
  const [w, h] = size(n)
  return { x0: n.x - w / 2, y0: n.y - h / 2, x1: n.x + w / 2, y1: n.y + h / 2 }
}

function toneColour(mode: Mode, tone: Tone | undefined): string {
  const c = chrome(mode)
  if (tone === undefined || tone === 'ink') return c.ink
  if (tone === 'neutral') return c.muted
  return seriesColor(mode, tone)
}

/** The point on a node's outline on the given side. */
function port(n: PlacedNode, side: Side): Pt {
  const [w, h] = size(n)
  switch (side) {
    case 'n':
      return { x: n.x, y: n.y - h / 2 }
    case 's':
      return { x: n.x, y: n.y + h / 2 }
    case 'e':
      return { x: n.x + w / 2, y: n.y }
    case 'w':
      return { x: n.x - w / 2, y: n.y }
  }
}

/** Where the ray from a node's centre towards `toward` leaves its outline (circles and rectangles). */
function boundary(n: PlacedNode, toward: Pt): Pt {
  const [w, h] = size(n)
  const dx = toward.x - n.x
  const dy = toward.y - n.y
  const len = Math.hypot(dx, dy) || 1
  if (ROUND.has(n.shape ?? 'box')) {
    const r = Math.min(w, h) / 2
    return { x: n.x + (dx / len) * r, y: n.y + (dy / len) * r }
  }
  const t = Math.min(dx === 0 ? Infinity : w / 2 / Math.abs(dx), dy === 0 ? Infinity : h / 2 / Math.abs(dy))
  return { x: n.x + dx * t, y: n.y + dy * t }
}

/** The side of `from` that faces `to`. */
function facing(from: Pt, to: Pt): Side {
  const dx = to.x - from.x
  const dy = to.y - from.y
  if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? 'e' : 'w'
  return dy >= 0 ? 's' : 'n'
}

const horizontal = (s: Side) => s === 'e' || s === 'w'

type Loop = { s0: Pt; c0: Pt; c1: Pt; s1: Pt; apex: Pt; dir: Pt }

const SIDE_ANGLE: Record<Side, number> = { e: 0, s: Math.PI / 2, w: Math.PI, n: -Math.PI / 2 }

/** A self-loop outside one side of a node: a cubic curve leaving and re-entering the outline either side of it. */
function loopEdge(n: PlacedNode, side: Side): Loop {
  const [w, h] = size(n)
  const r = Math.min(w, h) / 2
  const theta = SIDE_ANGLE[side]
  const ray = (a: number, d: number): Pt => ({ x: n.x + Math.cos(a) * d, y: n.y + Math.sin(a) * d })
  const out = (a: number) => boundary(n, ray(a, 1))
  const reach = r + 0.9
  const s0 = out(theta - 0.45)
  const s1 = out(theta + 0.45)
  const c0 = ray(theta - 0.6, reach)
  const c1 = ray(theta + 0.6, reach)
  const apex = { x: (s0.x + s1.x) / 8 + (3 * (c0.x + c1.x)) / 8, y: (s0.y + s1.y) / 8 + (3 * (c0.y + c1.y)) / 8 }
  return { s0, c0, c1, s1, apex, dir: { x: Math.cos(theta), y: Math.sin(theta) } }
}

/** Corner points of an edge in grid units, from port to port. */
function routeEdge(e: DiagramEdge, byId: Map<string, PlacedNode>): Pt[] {
  const a = parseEnd(e.from)
  const b = parseEnd(e.to)
  const na = byId.get(a.id)
  const nb = byId.get(b.id)
  if (!na || !nb) throw new Error(`diagram edge ${e.from} → ${e.to}: unknown node`)
  const via = (e.via ?? []).map(([x, y]) => ({ x, y }))
  if ((e.route === 'straight' || e.route === 'curve') && !a.side && !b.side) {
    // Direct lines leave each node along the line itself, so diagonal edges meet circles cleanly.
    const firstStop = via[0] ?? nb
    const lastStop = via.at(-1) ?? na
    return [boundary(na, firstStop), ...via, boundary(nb, lastStop)]
  }
  const firstTarget = via[0] ?? nb
  const lastSource = via.at(-1) ?? na
  const sa = a.side ?? facing(na, firstTarget)
  const sb = b.side ?? facing(nb, lastSource)
  const p0 = port(na, sa)
  const p1 = port(nb, sb)
  if (e.route === 'straight' || e.route === 'curve') return [p0, ...via, p1]
  const pts: Pt[] = [p0]
  const stops = [...via, p1]
  let hFirst = horizontal(sa)
  for (let i = 0; i < stops.length; i++) {
    const prev = pts[pts.length - 1]
    const next = stops[i]
    const last = i === stops.length - 1
    if (prev.x !== next.x && prev.y !== next.y) {
      if (last && via.length === 0 && horizontal(sa) === horizontal(sb)) {
        // Same orientation at both ends: two elbows at the midpoint.
        if (horizontal(sa)) {
          const mx = (prev.x + next.x) / 2
          pts.push({ x: mx, y: prev.y }, { x: mx, y: next.y })
        } else {
          const my = (prev.y + next.y) / 2
          pts.push({ x: prev.x, y: my }, { x: next.x, y: my })
        }
      } else if (last) {
        // Arrive along the axis of the target side.
        pts.push(horizontal(sb) ? { x: prev.x, y: next.y } : { x: next.x, y: prev.y })
      } else {
        pts.push(hFirst ? { x: next.x, y: prev.y } : { x: prev.x, y: next.y })
      }
    }
    pts.push(next)
    hFirst = true
  }
  return pts.filter((p, i) => i === 0 || p.x !== pts[i - 1].x || p.y !== pts[i - 1].y)
}

/** An SVG path through the points with rounded corners of radius r (pixels). */
function roundedPath(pts: Pt[], r: number): string {
  let d = `M ${pts[0].x} ${pts[0].y}`
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i - 1]
    const c = pts[i]
    const n = pts[i + 1]
    const l1 = Math.hypot(c.x - p.x, c.y - p.y)
    const l2 = Math.hypot(n.x - c.x, n.y - c.y)
    const k = Math.min(r, l1 / 2, l2 / 2)
    const a = { x: c.x - ((c.x - p.x) / l1) * k, y: c.y - ((c.y - p.y) / l1) * k }
    const b = { x: c.x + ((n.x - c.x) / l2) * k, y: c.y + ((n.y - c.y) / l2) * k }
    d += ` L ${a.x} ${a.y} Q ${c.x} ${c.y} ${b.x} ${b.y}`
  }
  const last = pts[pts.length - 1]
  return `${d} L ${last.x} ${last.y}`
}

function arrowHead(tip: Pt, from: Pt, size: number): string {
  const len = Math.hypot(tip.x - from.x, tip.y - from.y) || 1
  const ux = (tip.x - from.x) / len
  const uy = (tip.y - from.y) / len
  const bx = tip.x - ux * size
  const by = tip.y - uy * size
  const px = -uy * size * 0.45
  const py = ux * size * 0.45
  return `${tip.x},${tip.y} ${bx + px},${by + py} ${bx - px},${by - py}`
}

function trapezoid(n: PlacedNode, u: number): string {
  const [w, h] = size(n)
  const cx = n.x * u
  const cy = n.y * u
  const hw = (w * u) / 2
  const hh = (h * u) / 2
  const dir: Direction = n.dir ?? 'right'
  const [start, end] = n.shape === 'encoder' ? [1, TAPER] : [TAPER, 1]
  // Along a horizontal flow the height tapers; along a vertical flow the width does.
  if (dir === 'right' || dir === 'left') {
    const [l, r] = dir === 'right' ? [start, end] : [end, start]
    return `${cx - hw},${cy - hh * l} ${cx + hw},${cy - hh * r} ${cx + hw},${cy + hh * r} ${cx - hw},${cy + hh * l}`
  }
  const [t, b] = dir === 'down' ? [start, end] : [end, start]
  return `${cx - hw * t},${cy - hh} ${cx + hw * t},${cy - hh} ${cx + hw * b},${cy + hh} ${cx - hw * b},${cy + hh}`
}

/** A label drawn as HTML over the SVG, positioned in the diagram's pixel coordinates (natural size). */
type Overlay = {
  key: string
  /** A box the label is centred or aligned in; or, for edge labels, a point the label is centred on. */
  box?: Box
  at?: Pt
  rotate?: number
  /** Which point of the label sits at `at`: its centre, or the middle of its left (start) or right (end) edge. */
  anchor?: 'start' | 'center' | 'end'
  /** Set on node labels so they can be measured and the node grown to fit. */
  fitNode?: string
  /** Font size in diagram pixels; it scales with the rendered diagram. */
  size: number
  align: 'start' | 'center' | 'end'
  opacity?: number
  content: ReactNode
}

function labelLines(text: string, colour: string): ReactNode {
  return (
    <div className="leading-tight" style={{ color: colour, transition: TEXT_TRANSITION }}>
      {text.split('\n').map((l, i) => (
        <div key={i} className="whitespace-nowrap">
          <MathText text={l} />
        </div>
      ))}
    </div>
  )
}

function groupBox(g: DiagramGroup, byId: Map<string, PlacedNode>): Box {
  if (g.rect) return { x0: g.rect.x, y0: g.rect.y, x1: g.rect.x + g.rect.w, y1: g.rect.y + g.rect.h }
  const pad = g.pad ?? 0.35
  const boxes = (g.around ?? []).map((id) => {
    const n = byId.get(id)
    if (!n) throw new Error(`diagram group ${g.id}: unknown node ${id}`)
    return extent(n)
  })
  const top = g.label && (g.labelAt ?? 'top-left').startsWith('top') ? LABEL_H : 0
  const bottom = g.label && (g.labelAt ?? 'top-left').startsWith('bottom') ? LABEL_H : 0
  return {
    x0: Math.min(...boxes.map((b) => b.x0)) - pad,
    y0: Math.min(...boxes.map((b) => b.y0)) - pad - top,
    x1: Math.max(...boxes.map((b) => b.x1)) + pad,
    y1: Math.max(...boxes.map((b) => b.y1)) + pad + bottom,
  }
}

const INSIDE_LABEL = new Set(['box', 'pill', 'stack', 'circle', 'latent', 'noise', 'encoder', 'decoder', 'op'])

/** What a node's measured size depends on: a cached size is reused only while these are unchanged. */
const fitKey = (n: DiagramNode) =>
  JSON.stringify([n.label ?? '', n.shape ?? 'box', n.small ?? false, n.w ?? null, n.h ?? null, n.bar ? 1 : 0])

/** A measured label size, with the key of the node it was measured on. */
type Fit = { key: string; size: [number, number] }

/** Lay out if asked, apply `spread` to every position and grow nodes to the measured label sizes. */
function prepare(source: DiagramSpec, fit: Record<string, Fit>): Omit<DiagramSpec, 'nodes'> & { nodes: PlacedNode[] } {
  const fitted = (n: DiagramNode): [number, number] => {
    const [w, h] = nodeSize(n)
    // A size measured for a different label or shape under the same id (a new tree reusing node ids) is stale.
    const cached = fit[n.id]
    const f = cached && cached.key === fitKey(n) ? cached.size : undefined
    const round = ROUND.has(n.shape ?? 'box')
    // Circles grow evenly; trapezoids need extra width because their narrow end is only TAPER of the height.
    const [fw, fh] = f ? (round ? [Math.max(f[0], f[1]), Math.max(f[0], f[1])] : f) : [0, 0]
    return [Math.max(w, fw), Math.max(h, fh)]
  }
  // The layered layout spaces nodes by their fitted sizes, so labels that grow a node push its neighbours apart.
  const spec = source.layout === 'layered' ? layeredLayout(source, fitted) : source
  const [sx, sy] = Array.isArray(spec.spread) ? spec.spread : [spec.spread ?? 1, spec.spread ?? 1]
  return {
    ...spec,
    nodes: spec.nodes.map((n) => {
      if (n.x === undefined || n.y === undefined)
        throw new Error(`diagram node ${n.id}: no position (give x and y, or use layout: 'layered')`)
      const [w, h] = fitted(n)
      return { ...n, x: n.x * sx, y: n.y * sy, w, h }
    }),
    edges: spec.edges?.map((e) => ({ ...e, via: e.via?.map(([x, y]) => [x * sx, y * sy] as [number, number]) })),
    groups: spec.groups?.map((g) =>
      g.rect ? { ...g, rect: { x: g.rect.x * sx, y: g.rect.y * sy, w: g.rect.w * sx, h: g.rect.h * sy } } : g,
    ),
  }
}

/** Rough rendered length in diagram pixels of a label at a font size: maths counts by its visible characters. */
function labelLength(text: string, px: number): number {
  const plain = text
    .replace(/\$([^$]*)\$/g, (_, m: string) => m.replace(/\\[a-zA-Z]+|[{}^_\\ ]/g, '').replace(/./g, 'x'))
    .replace(/\n.*/s, '')
  return plain.length * px * 0.55 + 8
}

/** The point and direction (radians) a fraction `t` of the way along a polyline. */
function alongPolyline(pts: Pt[], t: number): { p: Pt; angle: number } {
  const lens = pts.slice(1).map((q, i) => Math.hypot(q.x - pts[i].x, q.y - pts[i].y))
  let target = lens.reduce((a, b) => a + b, 0) * t
  for (let i = 0; i < lens.length; i++) {
    if (target <= lens[i] || i === lens.length - 1) {
      const f = lens[i] ? Math.min(target / lens[i], 1) : 0
      const a = pts[i]
      const b = pts[i + 1]
      return { p: { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f }, angle: Math.atan2(b.y - a.y, b.x - a.x) }
    }
    target -= lens[i]
  }
  return { p: pts[0], angle: 0 }
}

/** The box beside a node for an outside label or note on side `s`, `row` rows out (pixels at natural size). */
function besideBox(n: PlacedNode, b: Box, s: Side, u: number, row = 0): Box {
  const lw = Math.max(1.8, (n.w ?? 0) + 0.6) * u
  const lh = NOTE_H * u
  const gap = 4 + row * lh
  if (s === 'n') return { x0: n.x * u - lw / 2, y0: b.y0 - gap - lh, x1: n.x * u + lw / 2, y1: b.y0 - gap }
  if (s === 's') return { x0: n.x * u - lw / 2, y0: b.y1 + gap, x1: n.x * u + lw / 2, y1: b.y1 + gap + lh }
  if (s === 'e') return { x0: b.x1 + gap, y0: n.y * u - lh / 2, x1: b.x1 + gap + lw, y1: n.y * u + lh / 2 }
  return { x0: b.x0 - gap - lw, y0: n.y * u - lh / 2, x1: b.x0 - gap, y1: n.y * u + lh / 2 }
}

/** Room, in diagram pixels, that a node's share bar takes under its label. */
const barStrip = (n: PlacedNode) => (n.barHeight ?? 5) + 4

/** A node's stacked share bar: along the bottom under a label, or filling the node when it has none. */
function ShareBar({
  n,
  b,
  mode,
  strip,
  opacity,
}: {
  n: PlacedNode
  b: Box
  mode: Mode
  strip: number
  opacity: number
}) {
  const shares = n.bar ?? []
  const total = shares.reduce((a, v) => a + Math.max(v, 0), 0)
  if (!(total > 0)) return null
  const inset = ROUND.has(n.shape ?? 'box') ? (b.x1 - b.x0) * 0.2 : 4
  const x0 = b.x0 + inset
  const width = b.x1 - b.x0 - 2 * inset
  const height = strip ? strip - 4 : Math.max(b.y1 - b.y0 - 2 * inset, 3)
  const y = strip ? b.y1 - strip : b.y0 + inset
  const widths = shares.map((v) => (Math.max(v, 0) / total) * width)
  const starts = widths.map((_, k) => x0 + widths.slice(0, k).reduce((acc, w) => acc + w, 0))
  return (
    <g style={{ opacity, transition: TRANSITION }}>
      {widths.map((wk, k) => (
        <rect key={k} x={starts[k]} y={y} width={wk} height={height} fill={seriesColor(mode, k)} />
      ))}
    </g>
  )
}

/** Colour, weight and opacity of an element in a step state. */
function stateLook(state: ElementState | undefined, colour: string, muted: string, active: string) {
  if (state === 'idle') return { colour: muted, opacity: 0.45, weight: 1, fill: 0.6 }
  if (state === 'active') return { colour: active, opacity: 1, weight: 1.8, fill: 2.2 }
  return { colour, opacity: 1, weight: 1, fill: 1 }
}

export type DiagramProps = {
  spec: DiagramSpec
  ariaLabel: string
  className?: string
  /**
   * The height of the drawing area in pixels, or `fill` to take the parent's height (the parent needs a definite
   * height). By default a diagram inside a `Figure` takes the frame's height, and elsewhere fits the width.
   */
  height?: number | 'fill'
  /** Makes nodes clickable, e.g. to choose which node a figure is about. */
  onNodeClick?: (id: string) => void
  /** Called with a node's id when the pointer enters it, and with null when it leaves. */
  onNodeHover?: (id: string | null) => void
  /** With `onNodeClick`: each node takes the focus by Tab and clicks on Enter or Space (default true). */
  focusable?: boolean
  /**
   * Makes nodes draggable: called with the node's id and the pointer's position in grid units on every move (`move`)
   * and on release (`end`). The figure writes the position back into the spec. A press that moves the pointer less than
   * 4 px is a click, not a drag. Give the spec a `frame` so the view does not refit under the pointer.
   */
  onNodeDrag?: (id: string, x: number, y: number, phase: 'move' | 'end') => void
}

/**
 * Renders a diagram spec as SVG, scaled to fit its box (the `Figure` frame's size by default) and centred. Colours come
 * from the data palette and the chrome tokens, so the diagram follows the light and dark themes. Labels, notes and edge
 * chips are HTML laid over the SVG and set by KaTeX; they scale with the drawing.
 */
export function Diagram({
  spec: source,
  ariaLabel,
  className,
  height: heightProp,
  onNodeClick,
  onNodeHover,
  focusable = true,
  onNodeDrag,
}: DiagramProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  // The node being dragged and where the press began, and whether the pointer has moved far enough to be a drag.
  const drag = useRef<{ id: string; x: number; y: number; moved: boolean } | null>(null)
  const justDragged = useRef(false)
  const { resolved: mode } = useTheme()
  const frame = useContext(FrameContext)
  const u = source.unit ?? 40
  const [fit, setFit] = useState<Record<string, Fit>>({})
  const spec = useMemo(() => prepare(source, fit), [source, fit])
  const [wrapper, box] = useElementSize<HTMLDivElement>()
  const c = chrome(mode)

  const layout = useMemo(() => {
    const byId = new Map(spec.nodes.map((n) => [n.id, n]))
    const groups = (spec.groups ?? []).map((g) => ({ g, box: groupBox(g, byId) }))
    // An edge whose ends coincide (e.g. swallowed by a wide text node) has nothing to draw.
    const edges = (spec.edges ?? [])
      .map((e) => {
        const a = parseEnd(e.from)
        if (a.id !== parseEnd(e.to).id) return { e, pts: routeEdge(e, byId) }
        const n = byId.get(a.id)
        if (!n) throw new Error(`diagram edge ${e.from} → ${e.to}: unknown node`)
        const loop = loopEdge(n, a.side ?? 'n')
        return { e, pts: [loop.s0, loop.s1], loop }
      })
      .filter(({ pts }) => pts.length >= 2)
    // A curved edge bows out by half its bend at the middle; that apex must be inside the view too.
    const apex = ({ e, pts, loop }: { e: DiagramEdge; pts: Pt[]; loop?: Loop }): Pt[] => {
      if (loop) {
        // A loop's label sits beyond its apex, so leave room for it.
        const room = e.label ? (Math.abs(loop.dir.x) > 0.5 ? 1.2 : 0.55) : 0
        return [{ x: loop.apex.x + loop.dir.x * room, y: loop.apex.y + loop.dir.y * room }]
      }
      if (e.route !== 'curve' || pts.length !== 2) return []
      const [p, q] = pts
      const len = Math.hypot(q.x - p.x, q.y - p.y) || 1
      const half = (e.bend ?? 0.6) / 2
      return [{ x: (p.x + q.x) / 2 - ((q.y - p.y) / len) * half, y: (p.y + q.y) / 2 + ((q.x - p.x) / len) * half }]
    }
    // Outside labels and notes take room beside their node.
    const beside = spec.nodes.flatMap((n) => {
      const b = extent(n)
      const sides = Object.keys(n.notes ?? {}) as Side[]
      const labelSide = n.labelSide ?? (n.shape === 'factor' || n.shape === 'dot' ? 'n' : undefined)
      if (n.label && labelSide) sides.push(labelSide)
      return sides.map((s): Box => {
        const half = Math.max(0.9, labelLength(n.notes?.[s] ?? n.label ?? '', FONT.note) / u / 2)
        const rows = s === labelSide && n.notes?.[s] ? 2 : 1
        if (s === 'n') return { x0: n.x - half, y0: b.y0 - NOTE_H * rows - 0.1, x1: n.x + half, y1: b.y0 }
        if (s === 's') return { x0: n.x - half, y0: b.y1, x1: n.x + half, y1: b.y1 + NOTE_H * rows + 0.1 }
        if (s === 'e') return { x0: b.x1, y0: n.y - 0.25, x1: b.x1 + 2 * half + 0.1, y1: n.y + 0.25 }
        return { x0: b.x0 - 2 * half - 0.1, y0: n.y - 0.25, x1: b.x0, y1: n.y + 0.25 }
      })
    })
    const all: Box[] = [
      ...(spec.frame ? [spec.frame] : []),
      ...spec.nodes.map(extent),
      ...beside,
      ...groups.map((g) => g.box),
      ...edges.flatMap((edge) => [...edge.pts, ...apex(edge)].map((p) => ({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }))),
    ]
    const margin = 0.3
    // An empty diagram (e.g. a tree before its first node) is a small blank box, not an Infinity-sized one.
    if (all.length === 0) all.push({ x0: 0, y0: 0, x1: 1, y1: 1 })
    const view = {
      x0: Math.min(...all.map((b) => b.x0)) - margin,
      y0: Math.min(...all.map((b) => b.y0)) - margin,
      x1: Math.max(...all.map((b) => b.x1)) + margin,
      y1: Math.max(...all.map((b) => b.y1)) + margin,
    }
    return { byId, groups, edges, view }
  }, [spec, u])

  const { view } = layout
  const width = (view.x1 - view.x0) * u
  const height = (view.y1 - view.y0) * u
  // The box to fit: a fixed height (prop or frame) fits both ways; otherwise the width alone.
  const fixed = heightProp ?? frame.height
  const maxScale = spec.maxScale ?? 1.4
  // Fitting the width alone (no frame height: a `fit="width"` dashboard row, or outside a Figure), a deep diagram is
  // still kept within about three quarters of the window's height, so it never needs scrolling past.
  const tallest = typeof window === 'undefined' ? Infinity : 0.75 * window.innerHeight
  const scale = box.width
    ? Math.min(
        box.width / width,
        fixed !== undefined && box.height ? box.height / height : fixed === undefined ? tallest / height : Infinity,
        maxScale,
      )
    : 1
  const scaled = (b: Box): Box => ({ x0: b.x0 * u, y0: b.y0 * u, x1: b.x1 * u, y1: b.y1 * u })
  const overlays: Overlay[] = []
  const vx = view.x0 * u
  const vy = view.y0 * u
  const accentColour = spec.accent === 'ink' ? c.ink : seriesColor(mode, ACCENT_SLOT)
  const accent = (tone: Tone | undefined) =>
    tone === undefined || tone === 'ink' || tone === 'neutral' ? accentColour : toneColour(mode, tone)
  const chip = (text: string, colour: string, border?: string) => (
    <span
      className="rounded px-1 whitespace-nowrap"
      style={{
        background: c.surface,
        color: colour,
        border: border ? `1px solid ${border}` : undefined,
        transition: TEXT_TRANSITION,
      }}
    >
      <MathText text={text} />
    </span>
  )

  /** The pointer's position in grid units (the SVG's user space divided by the unit). */
  const gridPoint = (event: ReactPointerEvent): { x: number; y: number } | null => {
    const el = svgRef.current
    const m = el?.getScreenCTM()
    if (!el || !m) return null
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(m.inverse())
    return { x: p.x / u, y: p.y / u }
  }
  const dragHandlers = (id: string) =>
    onNodeDrag
      ? {
          onPointerDown: (event: ReactPointerEvent<SVGGElement>) => {
            if (event.button !== 0) return
            event.currentTarget.setPointerCapture(event.pointerId)
            drag.current = { id, x: event.clientX, y: event.clientY, moved: false }
          },
          onPointerMove: (event: ReactPointerEvent<SVGGElement>) => {
            const d = drag.current
            if (!d || d.id !== id) return
            if (!d.moved && Math.hypot(event.clientX - d.x, event.clientY - d.y) < 4) return
            d.moved = true
            const p = gridPoint(event)
            if (p) onNodeDrag(id, p.x, p.y, 'move')
          },
          onPointerUp: (event: ReactPointerEvent<SVGGElement>) => {
            const d = drag.current
            drag.current = null
            if (!d || d.id !== id || !d.moved) return
            justDragged.current = true
            const p = gridPoint(event)
            if (p) onNodeDrag(id, p.x, p.y, 'end')
          },
          onPointerCancel: () => {
            drag.current = null
          },
        }
      : {}

  const svg = (
    <svg
      ref={svgRef}
      viewBox={`${vx} ${vy} ${width} ${height}`}
      role="img"
      aria-label={ariaLabel}
      width={width * scale}
      height={height * scale}
      style={{ display: 'block' }}
    >
      {layout.groups.map(({ g, box: gb }) => {
        const colour = toneColour(mode, g.tone ?? 'neutral')
        const b = scaled(gb)
        const at = g.labelAt ?? 'top-left'
        const right = at.endsWith('right')
        // At least 6 units wide, so a label longer than a narrow group still shows in full.
        const span = Math.max(b.x1 - b.x0 - 16, 6 * u)
        const lx = right ? b.x1 - 8 - span : b.x0 + 8
        const labelBox: Box = at.startsWith('top')
          ? { x0: lx, y0: b.y0 + 2, x1: lx + span, y1: b.y0 + LABEL_H * u }
          : { x0: lx, y0: b.y1 - LABEL_H * u, x1: lx + span, y1: b.y1 - 2 }
        if (g.label)
          overlays.push({
            key: `g-${g.id}`,
            box: labelBox,
            size: FONT.group,
            align: right ? 'end' : 'start',
            content: (
              <div
                className="font-sans font-medium tracking-wide whitespace-nowrap uppercase [&_.katex]:normal-case"
                style={{ color: colour }}
              >
                <MathText text={g.label} />
              </div>
            ),
          })
        return (
          <rect
            key={g.id}
            x={b.x0}
            y={b.y0}
            width={b.x1 - b.x0}
            height={b.y1 - b.y0}
            rx={10}
            fill={colour}
            fillOpacity={0.06}
            stroke={colour}
            strokeOpacity={0.55}
            strokeDasharray={g.dashed ? '5 4' : undefined}
          />
        )
      })}

      {layout.edges.map(({ e, pts, loop }, i) => {
        const base = e.highlight ? accentColour : e.tone === undefined ? c.inkSecondary : toneColour(mode, e.tone)
        const look = stateLook(e.state, base, c.muted, accent(e.tone))
        const colour = look.colour
        const strokeWidth = (e.highlight ? 2.5 : 1.5) * look.weight
        const paint: CSSProperties = { stroke: colour, strokeWidth, opacity: look.opacity, transition: TRANSITION }
        const head = 7
        if (loop) {
          const [s0, c0, c1, s1, apex] = [loop.s0, loop.c0, loop.c1, loop.s1, loop.apex].map((p) => ({
            x: p.x * u,
            y: p.y * u,
          }))
          // Beside a node the label grows away from the loop; above or below it is centred on the apex.
          const sideways = Math.abs(loop.dir.x) > 0.5
          const len = Math.hypot(s1.x - c1.x, s1.y - c1.y) || 1
          const end = { x: s1.x - ((s1.x - c1.x) / len) * head * 0.8, y: s1.y - ((s1.y - c1.y) / len) * head * 0.8 }
          if (e.label)
            overlays.push({
              key: `e-${i}`,
              at: { x: apex.x + loop.dir.x * (sideways ? 5 : 0), y: apex.y + loop.dir.y * 12 },
              anchor: sideways ? (loop.dir.x > 0 ? 'start' : 'end') : 'center',
              size: FONT.small,
              align: 'center',
              opacity: look.opacity,
              content: chip(e.label, c.inkSecondary),
            })
          return (
            <g key={i}>
              <path
                d={`M ${s0.x} ${s0.y} C ${c0.x} ${c0.y} ${c1.x} ${c1.y} ${end.x} ${end.y}`}
                fill="none"
                style={paint}
                strokeDasharray={e.dashed ? '5 4' : undefined}
              />
              {(e.arrow ?? 'end') !== 'none' && (
                <polygon
                  points={arrowHead(s1, c1, head)}
                  style={{ fill: colour, opacity: look.opacity, transition: TRANSITION }}
                />
              )}
            </g>
          )
        }
        let px = pts.map((p) => ({ x: p.x * u, y: p.y * u }))
        let control: Pt | undefined
        if (e.route === 'curve' && px.length === 2) {
          // A quadratic curve bowed to the left of the direction of travel; the ends re-aim at the control point.
          const [p, q] = px
          const bend = (e.bend ?? 0.6) * u
          const len = Math.hypot(q.x - p.x, q.y - p.y) || 1
          control = {
            x: (p.x + q.x) / 2 + (-(q.y - p.y) / len) * bend,
            y: (p.y + q.y) / 2 + ((q.x - p.x) / len) * bend,
          }
          const a = layout.byId.get(parseEnd(e.from).id)!
          const b = layout.byId.get(parseEnd(e.to).id)!
          const ctrl = { x: control.x / u, y: control.y / u }
          const s0 = boundary(a, ctrl)
          const s1 = boundary(b, ctrl)
          px = [
            { x: s0.x * u, y: s0.y * u },
            { x: s1.x * u, y: s1.y * u },
          ]
        }
        // A point and direction a fraction t of the way along the drawn edge.
        const along = (t: number) => {
          if (!control) return alongPolyline(px, t)
          const [a, b] = px
          const p = {
            x: (1 - t) ** 2 * a.x + 2 * (1 - t) * t * control.x + t * t * b.x,
            y: (1 - t) ** 2 * a.y + 2 * (1 - t) * t * control.y + t * t * b.y,
          }
          const d = {
            x: 2 * (1 - t) * (control.x - a.x) + 2 * t * (b.x - control.x),
            y: 2 * (1 - t) * (control.y - a.y) + 2 * t * (b.y - control.y),
          }
          return { p, angle: Math.atan2(d.y, d.x) }
        }
        const before = (k: 'start' | 'end') => control ?? (k === 'end' ? px[px.length - 2] : px[1])
        let arrow = e.arrow ?? 'end'
        if (e.reverse) arrow = arrow === 'end' ? 'start' : arrow === 'start' ? 'end' : arrow
        // Pull the line back from the tip so the stroke does not poke through the arrowhead.
        const trimmed = px.map((p) => ({ ...p }))
        const shorten = (tip: Pt, from: Pt) => {
          const len = Math.hypot(tip.x - from.x, tip.y - from.y) || 1
          tip.x -= ((tip.x - from.x) / len) * head * 0.8
          tip.y -= ((tip.y - from.y) / len) * head * 0.8
        }
        if (arrow === 'end' || arrow === 'both') shorten(trimmed[trimmed.length - 1], before('end'))
        if (arrow === 'start' || arrow === 'both') shorten(trimmed[0], before('start'))
        let mid: string | null = null
        if (arrow === 'mid') {
          const { p, angle } = along(0.5)
          const sign = e.reverse ? -1 : 1
          const dir = { x: Math.cos(angle) * sign, y: Math.sin(angle) * sign }
          const tip = { x: p.x + (dir.x * head) / 2, y: p.y + (dir.y * head) / 2 }
          mid = arrowHead(tip, { x: tip.x - dir.x, y: tip.y - dir.y }, head)
        }
        if (e.label) {
          // Position along the edge: the middle of the longest straight run, or `labelPos` of the way along.
          let at: { p: Pt; angle: number }
          let run: number
          if (control) {
            at = along(e.labelPos ?? 0.5)
            run = Math.hypot(px[1].x - px[0].x, px[1].y - px[0].y)
          } else if (e.labelPos !== undefined) {
            at = alongPolyline(px, e.labelPos)
            run = Math.min(...px.slice(1).map((q, m) => Math.hypot(q.x - px[m].x, q.y - px[m].y)))
          } else {
            let k = 1
            for (let m = 2; m < px.length; m++)
              if (
                Math.hypot(px[m].x - px[m - 1].x, px[m].y - px[m - 1].y) >
                Math.hypot(px[k].x - px[k - 1].x, px[k].y - px[k - 1].y)
              )
                k = m
            at = alongPolyline([px[k - 1], px[k]], 0.5)
            run = Math.hypot(px[k].x - px[k - 1].x, px[k].y - px[k - 1].y)
          }
          // Offset along the normal on the chosen side of the direction of travel, then keep the text upright.
          const side = e.labelSide === 'right' ? -1 : 1
          const gap = (e.labelOffset ?? 0.12) * u + FONT.small * 0.75
          const nx = Math.sin(at.angle) * side
          const ny = -Math.cos(at.angle) * side
          // Rotate along the edge only when the label fits along its run; otherwise keep it level and set it beside the
          // edge, anchored at its near end so it grows away from the line.
          const fits = labelLength(e.label, FONT.small) < run - 12
          const rotate = e.labelRotate ?? fits
          let deg = rotate ? (at.angle * 180) / Math.PI : 0
          if (deg > 90) deg -= 180
          if (deg <= -90) deg += 180
          const steep = Math.abs(Math.sin(at.angle)) > 0.7
          const anchor = !rotate && steep ? (nx < 0 ? 'end' : 'start') : 'center'
          overlays.push({
            key: `e-${i}`,
            at: { x: at.p.x + nx * (anchor === 'center' ? gap : 6), y: at.p.y + ny * gap },
            rotate: deg,
            anchor,
            size: FONT.small,
            align: 'center',
            opacity: look.opacity,
            content: chip(e.label, c.inkSecondary),
          })
        }
        if (e.note) {
          const noteColour = e.note.tone === undefined ? colour : toneColour(mode, e.note.tone)
          overlays.push({
            key: `en-${i}`,
            at: along(e.note.at ?? 0.5).p,
            anchor: 'center',
            size: FONT.small,
            align: 'center',
            opacity: look.opacity,
            content: chip(e.note.text, noteColour, noteColour),
          })
        }
        const fillStyle: CSSProperties = { fill: colour, opacity: look.opacity, transition: TRANSITION }
        return (
          <g key={i}>
            <path
              d={
                control
                  ? `M ${trimmed[0].x} ${trimmed[0].y} Q ${control.x} ${control.y} ${trimmed[1].x} ${trimmed[1].y}`
                  : roundedPath(trimmed, 8)
              }
              fill="none"
              style={paint}
              strokeDasharray={e.dashed ? '5 4' : undefined}
            />
            {(arrow === 'end' || arrow === 'both') && (
              <polygon points={arrowHead(px[px.length - 1], before('end'), head)} style={fillStyle} />
            )}
            {(arrow === 'start' || arrow === 'both') && (
              <polygon points={arrowHead(px[0], before('start'), head)} style={fillStyle} />
            )}
            {mid && <polygon points={mid} style={fillStyle} />}
          </g>
        )
      })}

      {spec.nodes.map((n) => {
        const shape = n.shape ?? 'box'
        const tone = n.tone ?? (shape === 'op' || shape === 'text' || shape === 'factor' ? 'ink' : 'neutral')
        const base = n.highlight ? accentColour : toneColour(mode, tone)
        const look = stateLook(n.state, base, c.muted, accent(n.tone))
        const colour = look.colour
        const b = scaled(extent(n))
        const w = b.x1 - b.x0
        const h = b.y1 - b.y0
        const dash = n.dashed || shape === 'noise' ? '4 3' : undefined
        const fillOpacity =
          n.shade !== undefined ? 0.04 + 0.5 * Math.min(Math.max(n.shade, 0), 1) : n.filled ? 0.35 : 0.14
        const strokeWidth = (n.highlight ? 2.5 : 1.5) * look.weight
        const tint: CSSProperties = {
          fill: colour,
          fillOpacity: Math.min(fillOpacity * look.fill, 0.6),
          stroke: colour,
          strokeWidth,
          opacity: look.opacity,
          transition: TRANSITION,
        }
        const labelColour =
          n.highlight || n.state === 'active' || (shape === 'text' && n.tone !== undefined)
            ? colour
            : n.state === 'idle'
              ? c.muted
              : c.ink
        const side = n.labelSide ?? (shape === 'factor' || shape === 'dot' ? 'n' : undefined)
        const strip = n.bar && n.label && !side ? barStrip(n) : 0
        if (n.label)
          overlays.push({
            key: `n-${n.id}`,
            box: side ? besideBox(n, b, side, u) : { ...b, y1: b.y1 - strip },
            fitNode: side || !INSIDE_LABEL.has(shape) ? undefined : n.id,
            size: n.small ? FONT.small : FONT.label,
            align: side === 'e' ? 'start' : side === 'w' ? 'end' : 'center',
            content: labelLines(n.label, labelColour),
          })
        for (const [s, text] of Object.entries(n.notes ?? {}) as [Side, string][])
          overlays.push({
            key: `nn-${n.id}-${s}`,
            box: besideBox(n, b, s, u, side === s && n.label ? 1 : 0),
            size: FONT.note,
            align: s === 'e' ? 'start' : s === 'w' ? 'end' : 'center',
            opacity: n.state === 'idle' ? 0.6 : 1,
            content: labelLines(text, n.state === 'active' ? colour : n.state === 'idle' ? c.muted : c.inkSecondary),
          })
        let body: ReactNode = null
        if (shape === 'factor') {
          const grow = n.state === 'active' ? 3 : 0
          body = (
            <rect
              x={b.x0 - grow}
              y={b.y0 - grow}
              width={w + 2 * grow}
              height={h + 2 * grow}
              style={{ fill: colour, opacity: look.opacity, transition: TRANSITION }}
            />
          )
        } else if (shape === 'box' || shape === 'pill') {
          body = <rect x={b.x0} y={b.y0} width={w} height={h} rx={shape === 'pill' ? h / 2 : 6} style={tint} />
        } else if (shape === 'stack') {
          body = (
            <>
              <rect x={b.x0 + 8} y={b.y0 - 8} width={w} height={h} rx={6} style={{ ...tint, fillOpacity: 0.06 }} />
              <rect x={b.x0 + 4} y={b.y0 - 4} width={w} height={h} rx={6} style={{ ...tint, fillOpacity: 0.09 }} />
              <rect
                x={b.x0}
                y={b.y0}
                width={w}
                height={h}
                rx={6}
                style={{ ...tint, fill: c.surface, fillOpacity: 1 }}
              />
              <rect x={b.x0} y={b.y0} width={w} height={h} rx={6} style={tint} />
            </>
          )
        } else if (shape === 'encoder' || shape === 'decoder') {
          body = <polygon points={trapezoid(n, u)} style={tint} strokeLinejoin="round" />
        } else if (ROUND.has(shape)) {
          const r = Math.min(w, h) / 2
          if (shape === 'dot')
            body = <circle cx={n.x * u} cy={n.y * u} r={r} style={{ fill: colour, transition: TRANSITION }} />
          else if (shape === 'op')
            body = <circle cx={n.x * u} cy={n.y * u} r={r} style={{ ...tint, fill: c.surface, fillOpacity: 1 }} />
          else
            body = (
              <>
                <circle cx={n.x * u} cy={n.y * u} r={r} style={tint} strokeDasharray={dash} />
                {shape === 'latent' && (
                  <circle cx={n.x * u} cy={n.y * u} r={r - 4} fill="none" style={{ ...tint, fillOpacity: 0 }} />
                )}
              </>
            )
        }
        const ring = n.selected ? 4 : 0
        return (
          <g
            key={n.id}
            data-node-id={n.id}
            data-node-label={n.ariaLabel ?? n.label}
            {...dragHandlers(n.id)}
            onClick={
              onNodeClick
                ? () => {
                    // A drag ends with a click on the same node; it is not a choice.
                    if (justDragged.current) {
                      justDragged.current = false
                      return
                    }
                    onNodeClick(n.id)
                  }
                : undefined
            }
            onKeyDown={
              onNodeClick
                ? (event) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return
                    event.preventDefault()
                    onNodeClick(n.id)
                  }
                : undefined
            }
            onPointerEnter={onNodeHover ? () => onNodeHover(n.id) : undefined}
            onPointerLeave={onNodeHover ? () => onNodeHover(null) : undefined}
            tabIndex={onNodeClick && focusable ? 0 : undefined}
            role={onNodeClick ? 'button' : undefined}
            aria-label={onNodeClick ? (n.ariaLabel ?? n.label ?? n.id) : undefined}
            aria-pressed={onNodeClick && n.selected !== undefined ? n.selected : undefined}
            style={
              onNodeDrag
                ? { cursor: 'grab', outline: 'none', touchAction: 'none' }
                : onNodeClick
                  ? { cursor: 'pointer', outline: 'none' }
                  : undefined
            }
          >
            {body}
            {n.bar && (shape === 'box' || shape === 'pill' || !n.label) && (
              <ShareBar n={n} b={b} mode={mode} strip={n.label ? barStrip(n) : 0} opacity={look.opacity} />
            )}
            {ring > 0 &&
              (ROUND.has(shape) ? (
                <circle
                  cx={n.x * u}
                  cy={n.y * u}
                  r={Math.min(w, h) / 2 + ring}
                  fill="none"
                  stroke={c.ink}
                  strokeWidth={2}
                />
              ) : (
                <rect
                  x={b.x0 - ring}
                  y={b.y0 - ring}
                  width={w + 2 * ring}
                  height={h + 2 * ring}
                  rx={shape === 'pill' ? h / 2 + ring : 8}
                  fill="none"
                  stroke={c.ink}
                  strokeWidth={2}
                />
              ))}
          </g>
        )
      })}
    </svg>
  )

  // Grow any node whose label is wider or taller than it (measured at the rendered scale, converted to grid units).
  useLayoutEffect(() => {
    const root = wrapper.current
    if (!root || source.fitLabels === false || !box.width) return
    const grow: Record<string, Fit> = {}
    const source_ = new Map(source.nodes.map((n) => [n.id, n]))
    root.querySelectorAll<HTMLElement>('[data-fit-node]').forEach((el) => {
      const id = el.dataset.fitNode!
      const node = layout.byId.get(id)
      const given = source_.get(id)
      const inner = el.firstElementChild as HTMLElement | null
      if (!node || !inner || !given) return
      const key = fitKey(given)
      // Measure against the node's own size when the cached one is stale, so a smaller label can shrink it again.
      const [w, h] =
        fit[id] && fit[id].key !== key ? size({ ...node, w: given.w, h: given.h } as PlacedNode) : size(node)
      // A trapezoid's height at its centre is (1 + TAPER)/2 of its full height, so it needs proportionally more.
      const middle = node.shape === 'encoder' || node.shape === 'decoder' ? (1 + TAPER) / 2 : 1
      const wide = (inner.scrollWidth / scale + 14) / u
      const tall = (inner.scrollHeight / scale + 8 + (node.bar ? barStrip(node) : 0)) / u / middle
      const stale = fit[id] !== undefined && fit[id].key !== key
      if (stale || wide > w + 0.02 || tall > h + 0.02) grow[id] = { key, size: [Math.max(w, wide), Math.max(h, tall)] }
    })
    // Sizes come from the rendered DOM, so they can only be read after layout; growth is monotone, so this settles.
    // oxlint-disable-next-line react/set-state-in-effect
    if (Object.keys(grow).length) setFit((prev) => ({ ...prev, ...grow }))
  }, [layout, scale, u, source.fitLabels, source.nodes, fit, box.width, wrapper])

  const px = (v: number) => `${v * scale}px`
  return (
    <div
      ref={wrapper}
      className={className}
      style={{
        position: 'relative',
        width: '100%',
        height: fixed === 'fill' ? '100%' : fixed,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}
    >
      <div style={{ position: 'relative', width: width * scale, height: height * scale, flex: 'none' }}>
        {svg}
        {overlays.map((o) => {
          const common: CSSProperties = {
            position: 'absolute',
            fontSize: px(o.size),
            lineHeight: 1.2,
            pointerEvents: 'none',
            opacity: o.opacity,
            transition: TEXT_TRANSITION,
          }
          if (o.at)
            return (
              <div
                key={o.key}
                className="font-prose"
                style={{
                  ...common,
                  left: px(o.at.x - vx),
                  top: px(o.at.y - vy),
                  transform: `translate(${o.anchor === 'start' ? '0' : o.anchor === 'end' ? '-100%' : '-50%'}, -50%) rotate(${o.rotate ?? 0}deg)`,
                }}
              >
                {o.content}
              </div>
            )
          if (!o.box) return null
          return (
            <div
              key={o.key}
              data-fit-node={o.fitNode}
              className="font-prose"
              style={{
                ...common,
                left: px(o.box.x0 - vx),
                top: px(o.box.y0 - vy),
                width: px(o.box.x1 - o.box.x0),
                height: px(o.box.y1 - o.box.y0),
                display: 'flex',
                alignItems: 'center',
                justifyContent: o.align === 'start' ? 'flex-start' : o.align === 'end' ? 'flex-end' : 'center',
                textAlign: o.align === 'start' ? 'left' : o.align === 'end' ? 'right' : 'center',
              }}
            >
              {o.content}
            </div>
          )
        })}
      </div>
    </div>
  )
}
