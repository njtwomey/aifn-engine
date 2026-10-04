import { chrome, seriesColor, type Mode } from '@render/design/palette'
import type { Range } from './viewport'

/**
 * An arrow from `from` to `to`. Ink by default; `slot` colours it as that palette slot. `label` names it at its tip, or
 * with `labelAt: 'middle'` on a small pill at its midpoint (a count on a path's move, say). `width` thickens the shaft.
 */
export type Vector = {
  from: [number, number]
  to: [number, number]
  slot?: number
  label?: string
  labelAt?: 'end' | 'middle'
  width?: number
  /** Arrowhead size in pixels (default 10); smaller for a dense field. */
  head?: number
  /** Draw in the muted chrome colour (a background field under trails and points). */
  muted?: boolean
}

/** The plot's visible box in data coordinates, with each axis's scale. */
export type Box = { x: Range; y: Range; xLog?: boolean; yLog?: boolean }

type Point = [number, number]

/**
 * The part of the segment from `a` to `b` inside the box (Liang and Barsky, 1984), or null if none of it is.
 * `tipInside` says whether `b` itself survived. Log axes are clipped in log space, where the drawn line is straight.
 */
export function clipSegment(a: Point, b: Point, box: Box): { from: Point; to: Point; tipInside: boolean } | null {
  const fx = (v: number) => (box.xLog ? Math.log10(v) : v)
  const fy = (v: number) => (box.yLog ? Math.log10(v) : v)
  const gx = (v: number) => (box.xLog ? 10 ** v : v)
  const gy = (v: number) => (box.yLog ? 10 ** v : v)
  const [x0, y0, x1, y1] = [fx(a[0]), fy(a[1]), fx(b[0]), fy(b[1])]
  const [xmin, xmax, ymin, ymax] = [fx(box.x[0]), fx(box.x[1]), fy(box.y[0]), fy(box.y[1])]
  if (![x0, y0, x1, y1, xmin, xmax, ymin, ymax].every(Number.isFinite)) return null
  // Pull the edges in by a hair, so a clipped end never lands a rounding error outside the axis extent.
  const [ex, ey] = [(xmax - xmin) * 1e-9, (ymax - ymin) * 1e-9]
  const [dx, dy] = [x1 - x0, y1 - y0]
  const p = [-dx, dx, -dy, dy]
  const q = [x0 - (xmin + ex), xmax - ex - x0, y0 - (ymin + ey), ymax - ey - y0]
  let [t0, t1] = [0, 1]
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (q[i] < 0) return null
      continue
    }
    const t = q[i] / p[i]
    if (p[i] < 0) t0 = Math.max(t0, t)
    else t1 = Math.min(t1, t)
    if (t0 > t1) return null
  }
  const at = (t: number): Point => [gx(x0 + t * dx), gy(y0 + t * dy)]
  return { from: at(t0), to: at(t1), tipInside: t1 === 1 }
}

/** A notched chevron, pointing up like ECharts' own arrow so markLine rotates it along the line. */
const CHEVRON = 'path://M5 0 L10 10 L5 6 L0 10 Z'

/**
 * ECharts markLine data for arrows, clipped to `box` when given. ECharts drops a markLine whose end lies outside the
 * axes, so a vector running off the plot is cut at the edge: its shaft ends there with a small chevron (not an
 * arrowhead) pointing the way it goes, and its label sits at that point. A vector wholly outside draws nothing.
 */
export function vectorLines(vectors: readonly Vector[], mode: Mode, box?: Box) {
  return vectors.flatMap((v) => {
    const clipped = box ? clipSegment(v.from, v.to, box) : { from: v.from, to: v.to, tipInside: true }
    if (!clipped) return []
    const color = v.muted ? chrome(mode).muted : v.slot === undefined ? chrome(mode).ink : seriesColor(mode, v.slot)
    // A midpoint label is drawn upright on its own (`vectorMidLabels`), not along the line.
    const label =
      v.label && v.labelAt !== 'middle'
        ? { show: true, formatter: v.label, position: 'end', color, fontSize: 11 }
        : { show: false }
    const lineStyle = v.width === undefined ? { color } : { color, width: v.width }
    return [
      [
        { coord: clipped.from, lineStyle, label, symbol: 'none' },
        clipped.tipInside
          ? { coord: clipped.to, symbol: 'arrow', symbolSize: v.head ?? 10 }
          : { coord: clipped.to, symbol: CHEVRON, symbolSize: [8, 8] },
      ],
    ]
  })
}

/**
 * Scatter data for the labels of vectors with `labelAt: 'middle'`: upright text on a small pill in the plot's surface
 * colour, centred on the shaft's midpoint so the shaft runs under it. A label whose midpoint is outside `box` is dropped.
 */
export function vectorMidLabels(vectors: readonly Vector[], mode: Mode, box?: Box) {
  const inside = (p: Point) => !box || (p[0] >= box.x[0] && p[0] <= box.x[1] && p[1] >= box.y[0] && p[1] <= box.y[1])
  return vectors.flatMap((v) => {
    if (!v.label || v.labelAt !== 'middle') return []
    const mid: Point = [(v.from[0] + v.to[0]) / 2, (v.from[1] + v.to[1]) / 2]
    if (!inside(mid)) return []
    const color = v.muted ? chrome(mode).muted : v.slot === undefined ? chrome(mode).ink : seriesColor(mode, v.slot)
    return [
      {
        value: mid,
        label: {
          show: true,
          position: 'inside',
          formatter: v.label,
          color,
          fontSize: 10,
          fontWeight: 600,
          backgroundColor: chrome(mode).surface,
          borderColor: color,
          borderWidth: 1,
          borderRadius: 7,
          padding: [1, 4],
        },
      },
    ]
  })
}

/** Every vector's two ends, for including vectors in an axis's automatic fit. */
export function vectorEnds(vectors: readonly Vector[]): Point[] {
  return vectors.flatMap((v) => [v.from, v.to])
}
