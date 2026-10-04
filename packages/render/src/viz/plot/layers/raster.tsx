/** A value grid drawn as one cached canvas image (the heatmap), and contour lines of a field. */
import { categorical, chrome, interpolateColors, mute, scaleStops, type Mode } from '@render/design/palette'
import { argmaxMargins, contourSeries } from '../../contours'
import { formatNumber } from '../../format'
import { BAR, colorBarTicks } from '../../scale-bar'
import { ScaleBar } from '../../ScaleBar'
import type { Range } from '../../viewport'
import type { AxisModel } from '../axis'
import { defineLayer, escapeHtml, type CommonProps, type LayerContext } from '../layer'
import type { CustomApi } from './marks'

type Grid = readonly (readonly number[])[]

export type RasterProps = CommonProps & {
  /** Cell centres along x (columns) and y (rows), evenly spaced. */
  x: ArrayLike<number>
  y: ArrayLike<number>
  /** Row-major: z[i][j] is the value at (x[j], y[i]). Non-finite cells are left empty. */
  z: Grid
  /**
   * `sequential` for magnitude; `diverging` for signed values, symmetric about zero by default so zero is the pale
   * midpoint and the extremes are saturated; `categorical` for classes (k ≥ 0 in slot k; −1 unassigned, −2 contested).
   */
  scale?: 'sequential' | 'diverging' | 'categorical'
  /** The colour scale's ends (default the data's range; symmetric about zero for `diverging`). */
  range?: Range
  /** A colour axis (`useAxis({ hold: 'initial' })`) to hold the colour scale while z changes; its key refits. */
  colorAxis?: AxisModel
  /** Cell colour strength below 1 mutes the field under full-strength overlays. */
  fillOpacity?: number
  /** Categorical only: a name per class index, for the tooltip and readout. */
  categoryNames?: readonly string[]
  /** The colour bar beside the plot (default true; never for categorical). */
  colorBar?: boolean
  /** Label the colour bar at these values rather than at 3–5 nice ticks. */
  scaleTicks?: readonly number[]
  /** The value's name in tooltips, readouts and over the colour bar. */
  valueLabel?: string
  /**
   * The decision boundary in ink: for a categorical raster the argmax boundaries between classes; for a numeric field
   * its contour at this level (`true`: 0.5 for a probability field within [0, 1], 0 for a diverging one, else the
   * middle of the colour range).
   */
  boundary?: boolean | number
}

/** The boundary's contour inputs, per grid (a zoom rebuilds the layer but not the field). */
const boundaries = new WeakMap<object, { key: string; fields: Grid[]; level: number }>()

function boundaryFields(p: RasterProps, lo: number, hi: number): { fields: Grid[]; level: number } | null {
  if (p.boundary === undefined || p.boundary === false) return null
  const categorical = p.scale === 'categorical'
  const level = categorical
    ? 0
    : typeof p.boundary === 'number'
      ? p.boundary
      : lo >= 0 && hi <= 1
        ? 0.5
        : p.scale === 'diverging'
          ? 0
          : (lo + hi) / 2
  const key = `${categorical}|${level}`
  const hit = boundaries.get(p.z)
  if (hit?.key === key) return hit
  let fields: Grid[] = [p.z]
  if (categorical) {
    // One indicator field per class present; the argmax margins' zero contours are the boundaries between regions.
    const classes = [...new Set(p.z.flatMap((row) => row.filter((v) => Number.isFinite(v) && v >= 0)))].sort(
      (a, b) => a - b,
    )
    const indicators = classes.map((c) => p.z.map((row) => row.map((v) => (v === c ? 1 : 0))))
    const margins = indicators.length > 1 ? argmaxMargins(indicators) : []
    fields = margins.length === 2 ? margins.slice(0, 1) : margins
  }
  const out = { key, fields, level }
  boundaries.set(p.z, out)
  return out
}

const step = (v: ArrayLike<number>) => (v.length > 1 ? v[1] - v[0] : 1)

function bounds(p: RasterProps): { x: Range; y: Range } {
  const [dx, dy] = [Math.abs(step(p.x)), Math.abs(step(p.y))]
  const span = (v: ArrayLike<number>, d: number): Range => [
    Math.min(v[0], v[v.length - 1]) - d / 2,
    Math.max(v[0], v[v.length - 1]) + d / 2,
  ]
  return { x: span(p.x, dx), y: span(p.y, dy) }
}

/** The value range per grid and scale, kept while the grid is the same object (the Plot asks on every render). */
const extents = new WeakMap<object, { scale: RasterProps['scale']; range: Range }>()

/** The data's value range: min and max, or ±max|z| for a diverging scale. */
function valueExtent(p: RasterProps): Range {
  const hit = extents.get(p.z)
  if (hit && hit.scale === p.scale) return hit.range
  const range = computeExtent(p)
  extents.set(p.z, { scale: p.scale, range })
  return range
}

function computeExtent(p: RasterProps): Range {
  let [lo, hi] = [Infinity, -Infinity]
  for (const row of p.z) for (const v of row) if (Number.isFinite(v)) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)]
  if (!Number.isFinite(lo)) return [0, 1]
  if (p.scale === 'diverging') {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1
    return [-m, m]
  }
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi]
}

function categoryColor(mode: Mode, value: number): string {
  if (value === -2) return chrome(mode).muted
  if (value < 0) return chrome(mode).grid
  const colours = categorical(mode)
  return colours[Math.round(value) % colours.length]
}

const categoryName = (v: number, names?: readonly string[]) =>
  v === -1 ? 'unassigned' : v === -2 ? 'contested' : (names?.[v] ?? formatNumber(v))

/**
 * The cells drawn into an offscreen canvas, `k` pixels per cell, so the grid is one image to ECharts: a drag that
 * patches an overlay re-renders one element instead of every cell. The last image per z is kept, keyed by everything
 * that colours it, so rebuilding the layer for another reason (a zoom, a theme's other layers) reuses it.
 */
const images = new WeakMap<object, { key: string; canvas: HTMLCanvasElement | null }>()

function rasterImage(p: RasterProps, k: number, lo: number, hi: number, mode: Mode): HTMLCanvasElement | null {
  const scale = p.scale ?? 'sequential'
  const key = [k, lo, hi, scale, mode, p.fillOpacity ?? 1, p.x.length, p.y.length, p.x[0], p.y[0]].join('|')
  const hit = images.get(p.z)
  if (hit?.key === key) return hit.canvas
  let canvas: HTMLCanvasElement | null = null
  if (typeof document !== 'undefined' && p.x.length && p.y.length) {
    canvas = document.createElement('canvas')
    canvas.width = p.x.length * k
    canvas.height = p.y.length * k
    const ctx = canvas.getContext('2d')
    if (ctx) {
      const { surface } = chrome(mode)
      const stops = scaleStops(scale === 'diverging' ? 'diverging' : 'sequential', mode)
      const opacity = p.fillOpacity ?? 1
      const colorOf = (v: number) => {
        const c =
          scale === 'categorical'
            ? categoryColor(mode, v)
            : interpolateColors(stops, hi > lo ? Math.min(Math.max((v - lo) / (hi - lo), 0), 1) : 0.5)
        return opacity < 1 ? mute(c, surface, opacity) : c
      }
      // Colours repeat across cells (classes, equal values), so each is computed once per value.
      const cache = new Map<number, string>()
      const flipX = p.x.length > 1 && p.x[1] < p.x[0]
      const flipY = !(p.y.length > 1 && p.y[1] < p.y[0])
      for (let i = 0; i < p.y.length; i++) {
        const row = flipY ? p.y.length - 1 - i : i
        for (let j = 0; j < p.x.length; j++) {
          const v = p.z[i]?.[j]
          if (!Number.isFinite(v)) continue
          let color = cache.get(v)
          if (color === undefined) cache.set(v, (color = colorOf(v)))
          ctx.fillStyle = color
          ctx.fillRect((flipX ? p.x.length - 1 - j : j) * k, row * k, k, k)
        }
      }
    } else canvas = null
  }
  images.set(p.z, { key, canvas })
  return canvas
}

/**
 * The image element that stretches `image` over the box from (x0, y0) to (x1, y1) in pixels. The image is drawn at its
 * own size and scaled by the element's transform, not by `style.width` and `style.height`: the SVG renderer writes those
 * to an `<image>` whose default `preserveAspectRatio` (xMidYMid meet) fits the image inside them at its own aspect and
 * centres it, so a 5 × 40 grid became a narrow centred strip. A transform stretches it under both renderers.
 */
export function stretchedImage(
  image: { width: number; height: number },
  x0: number,
  y0: number,
  x1: number,
  y1: number,
) {
  const [w, h] = [Math.max(image.width, 1), Math.max(image.height, 1)]
  return {
    type: 'image' as const,
    x: x0,
    y: y0,
    scaleX: (x1 - x0) / w,
    scaleY: (y1 - y0) / h,
    style: { image, x: 0, y: 0, width: w, height: h },
  }
}

/** The colour scale's ends: explicit, held on the colour axis, or the data's. */
function colorRange(p: RasterProps, ctx: LayerContext): Range {
  return p.range ?? (p.colorAxis ? ctx.range(p.colorAxis, valueExtent(p)) : undefined) ?? valueExtent(p)
}

/**
 * The colour bar's ticks and the room it takes right of the plot (0 without a bar): wide enough for its tick labels
 * and for the value's name above it, so the name never reaches over the plot area (where the legend is).
 */
function colorBar(p: RasterProps, lo: number, hi: number): { ticks: readonly number[]; room: number } {
  const ticks = p.scaleTicks ?? colorBarTicks(lo, hi)
  const shown = p.colorBar !== false && p.scale !== 'categorical'
  const labels = BAR.width + BAR.tick + BAR.labelGap + Math.max(...ticks.map((v) => formatNumber(v).length * 6.5), 12)
  const name = (p.valueLabel ?? 'value').length * 6.5
  const room = shown ? BAR.gap + Math.max(labels, name) + 8 : 0
  return { ticks, room }
}

/**
 * A value grid on the Plot's numeric axes: one image for the cells (redrawn only when the cells, the colour scale or
 * the resolution change), a transparent hit layer for the tooltip, and a colour bar beside the plot whose room the
 * Plot's column keeps, so every panel above and below lines up with it.
 */
export const Raster = defineLayer<RasterProps>({
  kind: 'Raster',
  slotted: () => false,
  needsBox: true,
  needsPlot: true,
  canvas: () => true,
  covers: true,
  axes: (p) => [p.colorAxis],
  legend: () => [],
  margins: (p, ctx) => {
    const [lo, hi] = colorRange(p, ctx)
    const { room } = colorBar(p, lo, hi)
    // The bar's value label sits 20 px above the plot area.
    return { right: room, top: room ? 24 : 0 }
  },
  extent: (p) => ({ ...bounds(p), tight: true }),
  build: (p, ctx) => {
    const scale = p.scale ?? 'sequential'
    const isCategorical = scale === 'categorical'
    const [lo, hi] = colorRange(p, ctx)
    const b = bounds(p)
    const box = ctx.box
    const plot = ctx.plot ?? { width: 400, height: 300 }
    const [dx, dy] = [Math.abs(step(p.x)), Math.abs(step(p.y))]
    // One cell's size on screen in CSS pixels (the hit area) and, in device pixels rounded up to a power of two so a
    // zoom seldom redraws it, the image's pixels per cell, at most 4096 pixels a side.
    const cellW = box ? (plot.width * dx) / (box.x[1] - box.x[0]) : 8
    const cellH = box ? (plot.height * dy) / (box.y[1] - box.y[0]) : 8
    const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1
    const k = Math.min(
      2 ** Math.ceil(Math.log2(Math.max(cellW, cellH, 1) * dpr)),
      2 ** Math.max(0, Math.floor(Math.log2(4096 / Math.max(p.x.length, p.y.length, 1)))),
    )
    const image = rasterImage(p, k, lo, hi, ctx.mode)
    const label = p.valueLabel ?? 'value'
    const valueText = (v: number) => (isCategorical ? categoryName(v, p.categoryNames) : formatNumber(v))
    const cells: number[][] = []
    for (let i = 0; i < p.y.length; i++) for (let j = 0; j < p.x.length; j++) cells.push([p.x[j], p.y[i], p.z[i][j]])
    const xName = ctx.x.options.label ?? 'x'
    const yName = ctx.y.options.label ?? 'y'
    const { ticks, room } = colorBar(p, lo, hi)
    const showBar = room > 0
    const stops = scaleStops(scale === 'diverging' ? 'diverging' : 'sequential', ctx.mode)
    const boundary = boundaryFields(p, lo, hi)
    const xs = boundary ? Array.from(p.x) : []
    const ys = boundary ? Array.from(p.y) : []
    return {
      series: [
        ...(boundary
          ? boundary.fields.flatMap((f, k) =>
              contourSeries(xs, ys, f, [boundary.level], ctx.mode, {
                id: `${ctx.id}:boundary${k}`,
                labels: false,
                width: 1.5,
              }),
            )
          : []),
        {
          id: `${ctx.id}:image`,
          name: '__raster',
          type: 'custom',
          silent: true,
          clip: true,
          data: [[b.x[0], b.y[0]]],
          encode: { x: 0, y: 1 },
          tooltip: { show: false },
          renderItem: (_: unknown, api: CustomApi) => {
            if (!image) return null
            const [px0, py0] = api.coord([b.x[0], b.y[1]])
            const [px1, py1] = api.coord([b.x[1], b.y[0]])
            return stretchedImage(image, px0, py0, px1, py1)
          },
          z: 0,
        },
        // Hover targets: a transparent rectangle per cell in a large-mode scatter, one path rather than an element per
        // cell, so the tooltip costs a drag nothing.
        {
          id: `${ctx.id}:cells`,
          name: '__cells',
          type: 'scatter',
          large: true,
          largeThreshold: 0,
          clip: true,
          data: cells,
          symbol: 'rect',
          symbolSize: [Math.ceil(cellW) + 1, Math.ceil(cellH) + 1],
          itemStyle: { opacity: 0 },
          z: 0,
        },
      ],
      tooltip: {
        [`${ctx.id}:cells`]: (q) => {
          const v = q.value as number[]
          return `${escapeHtml(xName)} ${formatNumber(v[0])}, ${escapeHtml(yName)} ${formatNumber(v[1])}<br/>${escapeHtml(label)}: <b>${escapeHtml(valueText(v[2]))}</b>`
        },
      },
      pointer: ([px, py]) => {
        const j = Math.round((px - p.x[0]) / step(p.x))
        const i = Math.round((py - p.y[0]) / step(p.y))
        if (!(i >= 0 && i < p.y.length && j >= 0 && j < p.x.length)) return null
        return {
          at: `${xName} = ${formatNumber(p.x[j])}, ${yName} = ${formatNumber(p.y[i])}`,
          rows: [{ label, value: valueText(p.z[i][j]) }],
        }
      },
      overlay: showBar
        ? (g) => (
            <ScaleBar
              key={`${ctx.id}:bar`}
              stops={stops}
              lo={lo}
              hi={hi}
              ticks={ticks}
              label={label}
              left={g.left + g.width + BAR.gap}
              top={g.top}
              height={Math.max(g.height, 40)}
              room={g.boxWidth - g.left - g.width - BAR.gap}
            />
          )
        : undefined,
      extents: p.colorAxis ? [{ axis: p.colorAxis, range: valueExtent(p) }] : undefined,
      data: { kind: 'raster', label, x: Array.from(p.x), y: Array.from(p.y), z: p.z },
    }
  },
})

// ── Contours ─────────────────────────────────────────────────────────────────────────────────────────────────────────

export type ContoursProps = CommonProps & {
  x: readonly number[]
  y: readonly number[]
  /** Row-major field: z[i][j] at (x[j], y[i]). */
  z: Grid
  levels: readonly number[]
  /** A label per level (default true). */
  labels?: boolean
}

/** Contour lines of a field at the given levels (marching squares), ink unless given a slot or colour. */
export const Contours = defineLayer<ContoursProps>({
  kind: 'Contours',
  slotted: () => false,
  legend: () => [],
  build: (p, ctx) => ({
    series: contourSeries(p.x, p.y, p.z, p.levels, ctx.mode, {
      id: ctx.id,
      color: p.slot !== undefined || p.color ? ctx.color : undefined,
      labels: p.labels,
    }),
  }),
})
