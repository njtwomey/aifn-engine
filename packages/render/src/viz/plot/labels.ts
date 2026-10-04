/**
 * Label placement for point maps (a word map, a labelled scatter): greedy collision avoidance in pixel space. Labels
 * are placed in priority order; each takes the first of its candidate positions (right, left, above, below its point)
 * that stays inside the plot and overlaps no label placed before it and no other marker. A label with no free position is hidden; `Points`
 * shows it while its point is hovered.
 */

export type LabelPosition = 'right' | 'left' | 'top' | 'bottom'

export type Placement = { position: LabelPosition; shown: boolean }

type Rect = { x0: number; y0: number; x1: number; y1: number }

const POSITIONS: readonly LabelPosition[] = ['right', 'left', 'top', 'bottom']

/** Pixel width of a label: measured on a canvas in the browser, estimated from its length elsewhere (server render). */
let context: CanvasRenderingContext2D | null | undefined
export function textWidth(text: string, fontSize: number, font = 'system-ui, sans-serif'): number {
  if (context === undefined)
    context = typeof document === 'undefined' ? null : document.createElement('canvas').getContext('2d')
  if (!context) return text.length * fontSize * 0.6
  context.font = `${fontSize}px ${font}`
  return context.measureText(text).width
}

/** The box a label takes at a position beside a point at (px, py), `gap` pixels from it. */
function boxAt(position: LabelPosition, px: number, py: number, w: number, h: number, gap: number): Rect {
  switch (position) {
    case 'right':
      return { x0: px + gap, y0: py - h / 2, x1: px + gap + w, y1: py + h / 2 }
    case 'left':
      return { x0: px - gap - w, y0: py - h / 2, x1: px - gap, y1: py + h / 2 }
    case 'top':
      return { x0: px - w / 2, y0: py - gap - h, x1: px + w / 2, y1: py - gap }
    case 'bottom':
      return { x0: px - w / 2, y0: py + gap, x1: px + w / 2, y1: py + gap + h }
  }
}

const overlaps = (a: Rect, b: Rect) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1

/**
 * Places labels greedily. `px`, `py`: the points in pixels (y down), `widths` the labels' widths, `height` their line
 * height; `priority` (higher first; ties keep the given order); `bounds` the plot area's width and height. Points that
 * are not finite, or have an empty label, are hidden. `reserved` boxes (already drawn labels, e.g. a pinned word's) are
 * avoided too.
 */
export function placeLabels(
  px: ArrayLike<number>,
  py: ArrayLike<number>,
  widths: ArrayLike<number>,
  height: number,
  options: {
    priority?: ArrayLike<number>
    bounds: { width: number; height: number }
    gap?: number
    reserved?: readonly Rect[]
    /** Markers (centres in pixels, half-size `radius`) no label may cover, except its own point's. */
    markers?: { x: ArrayLike<number>; y: ArrayLike<number>; radius: number }
  },
): Placement[] {
  const n = px.length
  const gap = Math.max(options.gap ?? 5, (options.markers?.radius ?? 0) + 1)
  const { width: W, height: H } = options.bounds
  const out: Placement[] = Array.from({ length: n }, () => ({ position: 'right', shown: false }))
  const order = Array.from({ length: n }, (_, i) => i)
  const pr = options.priority
  if (pr) order.sort((a, b) => pr[b] - pr[a] || a - b)
  const placed: Rect[] = [...(options.reserved ?? [])]
  const m = options.markers
  const marks: (Rect & { k: number })[] = []
  if (m)
    for (let k = 0; k < m.x.length; k++)
      if (Number.isFinite(m.x[k]) && Number.isFinite(m.y[k]))
        marks.push({ k, x0: m.x[k] - m.radius, y0: m.y[k] - m.radius, x1: m.x[k] + m.radius, y1: m.y[k] + m.radius })
  for (const i of order) {
    const x = px[i]
    const y = py[i]
    if (!(Number.isFinite(x) && Number.isFinite(y)) || !(widths[i] > 0)) continue
    if (x < 0 || x > W || y < 0 || y > H) continue
    for (const position of POSITIONS) {
      const r = boxAt(position, x, y, widths[i], height, gap)
      if (r.x0 < 0 || r.y0 < 0 || r.x1 > W || r.y1 > H) continue
      if (placed.some((q) => overlaps(q, r))) continue
      if (marks.some((q) => q.k !== i && overlaps(q, r))) continue
      placed.push(r)
      out[i] = { position, shown: true }
      break
    }
  }
  return out
}

/** A data value as a pixel offset along an axis of `size` pixels drawn over `range` (log10 when `log`). */
export function toPixel(v: number, range: readonly [number, number], size: number, log: boolean): number {
  const f = log ? Math.log10 : (t: number) => t
  const [a, b] = [f(range[0]), f(range[1])]
  return ((f(v) - a) / (b - a)) * size
}
