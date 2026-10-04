/**
 * The basic layers: curves, points, bars, areas, a signed area, segments, vectors, a vector field, a rug, annotations
 * and handles.
 */
import { chrome, coolWarm, interpolateColors, MARKER_SHAPES, scaleStops, seriesColor } from '@render/design/palette'
import { formatNumber } from '../../format'
import type { Handle as HandleSpec } from '../../handles'
import { LINE_WIDTH, MARKER_SIZE } from '../../theme'
import {
  fieldArrows,
  vectorEnds,
  vectorLines,
  vectorMidLabels,
  type ArrowStyle,
  type FieldArrowOptions,
  type Vector,
} from '../../vectors'
import {
  defineLayer,
  extentOf,
  oriented,
  orientedExtent,
  withZero,
  type CommonProps,
  type HoverSeries,
  type LayerContext,
  type Orient,
} from '../layer'
import { signedArea, signedParts } from '../probability'
import { placeLabels, textWidth, toPixel, type Placement } from '../labels'

type Values = ArrayLike<number>

/** Room above the plot area for a label drawn there (an x handle's, a vertical annotation's). */
export const LABEL_ROW = 16

/** ECharts' custom-series API, as far as the layers use it. */
export type CustomApi = { value: (dim: number) => number; coord: (point: number[]) => number[] }
export type CustomParams = { dataIndex: number; coordSys: { x: number; y: number; width: number; height: number } }

const pairs = (x: Values, y: Values) => Array.from(x, (v, i) => [v, y[i]])

// ── Curve ────────────────────────────────────────────────────────────────────────────────────────────────────────────

export type CurveProps = CommonProps & {
  x: Values
  y: Values
  dashed?: boolean
  /** Thin and translucent, for many draws of one thing (chains, sample paths). */
  thin?: boolean
  /** Stroke width in pixels (default 2, or 1 when thin or muted). */
  width?: number
  /** Mark every vertex, e.g. each step of an optimiser. */
  showPoints?: boolean
  /** Left out of hover (no tooltip row or readout), e.g. hundreds of draws under the curves the tooltip reports. */
  silent?: boolean
}

/** A line through (x[i], y[i]); NaN breaks it. Hovered by x: the tooltip and readout report its value there. */
export const Curve = defineLayer<CurveProps>({
  kind: 'Curve',
  extent: (p) => ({ x: extentOf(p.x), y: extentOf(p.y) }),
  build: (p, ctx) => {
    const faint = p.thin || p.muted
    const name = p.name ?? 'curve'
    const hover: HoverSeries[] = p.silent ? [] : [{ label: name, color: ctx.color, x: p.x, y: p.y }]
    return {
      hover,
      series: [
        {
          id: ctx.id,
          // A silent curve keeps its name: it is left out of hover, not out of the legend.
          name,
          type: 'line',
          data: pairs(p.x, p.y),
          showSymbol: !!p.showPoints,
          symbolSize: p.showPoints ? 5 : 7,
          silent: !!p.silent,
          clip: true,
          lineStyle: {
            width: p.width ?? (faint ? 1 : LINE_WIDTH),
            color: ctx.color,
            opacity: p.thin ? 0.45 : p.muted ? 0.7 : 1,
            type: p.dashed ? 'dashed' : 'solid',
          },
          itemStyle: { color: ctx.color },
          emphasis: { focus: 'none', scale: 1.4 },
          z: faint ? 2 : 3,
        },
      ],
      data: { kind: 'curve', name, x: Array.from(p.x), y: Array.from(p.y) },
    }
  },
})

// ── Points ───────────────────────────────────────────────────────────────────────────────────────────────────────────

export type PointsProps = CommonProps & {
  x: Values
  y: Values
  /** A class per point: class k takes slot k and marker shape k, so class colours match everywhere. */
  group?: Values | null
  /** Names of the classes, for the legend. */
  groupNames?: readonly string[]
  /** A marker shape per point (an index into the shape list), or one for all, independent of colour. */
  shape?: number | Values
  /** Names of the shapes, for the legend when shapes carry a second variable. */
  shapeNames?: readonly string[]
  /** A colour per point (e.g. from `useScaleColor`), drawn with an ink outline. */
  colors?: readonly string[]
  /** Marker size in pixels. */
  size?: number
  /**
   * Thousands of small marks (an attractor, a bifurcation diagram): tiny unoutlined dots drawn in ECharts' large mode,
   * one path, no tooltip. The Plot switches to the canvas renderer.
   */
  dense?: boolean
  /** Smaller, lighter marks, for many points in a small panel. */
  thin?: boolean
  /**
   * Text beside each point (a word map). Labels are placed greedily without overlap, in `labelPriority` order: each
   * takes the first free spot right, left, above or below its point; one with no free spot is hidden and shows while
   * its point is hovered. An `emphasis` layer's labels are bold, in ink, and always shown.
   */
  labels?: readonly (string | null | undefined)[]
  /** Placement priority per point, higher first (e.g. frequency); default the order given. */
  labelPriority?: Values
}

const LABEL_FONT = 10
const STRONG_FONT = 12

/** Scatter marks. Grouped points split by class (colour and shape); shapes may also vary on their own. */
export const Points = defineLayer<PointsProps>({
  kind: 'Points',
  // Grouped points without group names (a faded copy under a highlighted subset) add no legend entries: their series are
  // named per group, so a single entry under the layer's name would match no series.
  legend: (p) =>
    p.dense || (p.group && !p.groupNames)
      ? []
      : p.groupNames
        ? [...p.groupNames]
        : p.shapeNames
          ? [...p.shapeNames]
          : [p.name ?? 'points'],
  slotted: (p) => !p.group && !p.muted && !p.emphasis && !p.tone,
  canvas: (p) => !!p.dense || p.x.length > 4000,
  // Labels are placed in pixels: rebuilt when the ranges or the plot's size change.
  needsBox: (p) => !!p.labels,
  needsPlot: (p) => !!p.labels,
  extent: (p) => ({ x: extentOf(p.x), y: extentOf(p.y) }),
  build: (p, ctx) => {
    const c = chrome(ctx.mode)
    const name = p.name ?? 'points'
    if (p.dense) {
      return {
        series: [
          {
            id: ctx.id,
            name,
            type: 'scatter',
            large: true,
            largeThreshold: 0,
            data: pairs(p.x, p.y),
            symbol: 'circle',
            symbolSize: p.size ?? 2,
            silent: true,
            clip: true,
            itemStyle: { color: ctx.color, opacity: 0.55, borderWidth: 0 },
            tooltip: { show: false },
            z: 2,
          },
        ],
      }
    }
    const placement = p.labels ? placePointLabels(p, ctx) : null
    const labelOf = (i: number) => {
      const text = p.labels?.[i]
      if (!text || !placement) return null
      const at = placement[i]
      return { name: text, label: { show: at.shown, position: at.position } }
    }
    const shapeAt = (i: number) =>
      typeof p.shape === 'number' ? p.shape : p.shape ? p.shape[i] : p.group ? p.group[i] : p.emphasis ? 3 : 0
    // One series per (class, shape): ECharts gives a series one symbol.
    const buckets = new Map<string, { g: number | null; s: number; data: unknown[] }>()
    for (let i = 0; i < p.x.length; i++) {
      const g = p.group ? p.group[i] : null
      const s = shapeAt(i)
      const key = `${g}:${s}`
      let b = buckets.get(key)
      if (!b) buckets.set(key, (b = { g, s, data: [] }))
      const text = labelOf(i)
      b.data.push(
        p.colors || text
          ? {
              value: [p.x[i], p.y[i]],
              ...(p.colors ? { itemStyle: { color: p.colors[i] } } : {}),
              ...(text ?? {}),
            }
          : [p.x[i], p.y[i]],
      )
    }
    // Every named class, and an ungrouped layer with no points, keeps its (empty) series: the legend entry stays, and a
    // live layer whose points come and go keeps one structure, so it is patched rather than redrawn.
    const keep = (g: number | null, sh: number) => {
      if (!buckets.has(`${g}:${sh}`)) buckets.set(`${g}:${sh}`, { g, s: sh, data: [] })
    }
    if (p.group && p.groupNames && p.shape === undefined) p.groupNames.forEach((_, g) => keep(g, g))
    // With shapes of their own (e.g. right and wrong), a named class absent from the data still needs a series of its
    // name, or the legend entry matches nothing.
    else if (p.group && p.groupNames)
      p.groupNames.forEach((_, g) => {
        if (![...buckets.values()].some((b) => b.g === g)) keep(g, typeof p.shape === 'number' ? p.shape : 0)
      })
    else if (!p.group && !p.shapeNames && buckets.size === 0)
      keep(null, typeof p.shape === 'number' ? p.shape : p.emphasis ? 3 : 0)
    const size = markerSize(p)
    const series = [...buckets.entries()]
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([key, b]) => {
        const label =
          b.g !== null
            ? (p.groupNames?.[b.g] ?? `${name} ${b.g}`)
            : p.shapeNames
              ? (p.shapeNames[b.s] ?? `${name} ${b.s}`)
              : name
        const color = b.g !== null ? seriesColor(ctx.mode, b.g) : ctx.color
        return {
          id: `${ctx.id}:${key}`,
          name: label,
          type: 'scatter',
          data: b.data,
          symbol: MARKER_SHAPES[b.s % MARKER_SHAPES.length],
          symbolSize: size,
          clip: true,
          itemStyle: p.colors
            ? { color: c.ink, borderColor: c.ink, borderWidth: 1 }
            : {
                color,
                opacity: p.emphasis ? 1 : p.thin ? 0.75 : 0.85,
                // A surface-coloured ring separates overlapping marks.
                borderColor: c.surface,
                borderWidth: p.emphasis ? 2 : p.thin ? 0.5 : 1,
              },
          ...(p.labels
            ? {
                // Shown per point as placed; a hidden label appears while its point is hovered.
                label: {
                  show: false,
                  formatter: '{b}',
                  distance: labelGap(p),
                  color: p.emphasis ? c.ink : c.inkSecondary,
                  fontWeight: p.emphasis ? 'bold' : 'normal',
                  fontSize: p.emphasis ? STRONG_FONT : LABEL_FONT,
                },
                emphasis: { scale: false, label: { show: true } },
                labelLayout: placement ? undefined : { hideOverlap: !p.emphasis },
              }
            : {}),
          z: p.emphasis ? 5 : 3,
        }
      })
    return {
      series,
      data: {
        kind: 'points',
        name,
        x: Array.from(p.x),
        y: Array.from(p.y),
        ...(p.group ? { group: Array.from(p.group) } : {}),
      },
    }
  },
})

const markerSize = (p: PointsProps) => p.size ?? (p.emphasis ? (p.thin ? 11 : 16) : p.thin ? 5 : MARKER_SIZE)
/** Pixels between a point and its label: clear of the marker. */
const labelGap = (p: PointsProps) => Math.max(5, markerSize(p) / 2 + 2)

/** Where each of a Points layer's labels goes, from the drawn box; null before the plot is measured. */
function placePointLabels(p: PointsProps, ctx: LayerContext): Placement[] | null {
  const labels = p.labels!
  const n = p.x.length
  if (p.emphasis) return Array.from({ length: n }, () => ({ position: 'right' as const, shown: true }))
  if (!ctx.box || !ctx.plot || ctx.plot.width < 2 || ctx.plot.height < 2) return null
  const { box, plot } = ctx
  const px = Array.from({ length: n }, (_, i) => toPixel(p.x[i], box.x, plot.width, box.xLog))
  const py = Array.from({ length: n }, (_, i) => plot.height - toPixel(p.y[i], box.y, plot.height, box.yLog))
  const widths = Array.from({ length: n }, (_, i) => (labels[i] ? textWidth(labels[i]!, LABEL_FONT) : 0))
  // Markers are obstacles too, so a label never covers another point.
  const r = markerSize(p) / 2 + 1
  return placeLabels(px, py, widths, LABEL_FONT + 2, {
    priority: p.labelPriority,
    bounds: plot,
    gap: labelGap(p),
    markers: { x: px, y: py, radius: r },
  })
}

// ── Bars ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

export type BarsProps = CommonProps & {
  /** Bar centres. */
  x: Values
  /** Bar heights. */
  y: Values
  /** Bin edges (one more than the bars): bars span them exactly and touch, as a histogram's do. */
  edges?: Values
  /** Bar width in data units (default 0.8 of the smallest gap between centres). */
  width?: number
  /** Where bars start (default 0). */
  base?: number
  orient?: Orient
  /** Fill opacity (default 1, or 0.5 for touching bars so overlapping histograms show through). */
  opacity?: number
  /** A colour per bar, e.g. each bar coloured by its class. */
  colors?: readonly string[]
}

type Bar = [number, number, number, number]

/** Bars as [left, right, base, height] in data units. */
function barsOf(p: BarsProps): Bar[] {
  const base = p.base ?? 0
  const out: Bar[] = []
  if (p.edges) {
    for (let i = 0; i < p.y.length; i++) out.push([p.edges[i], p.edges[i + 1], base, p.y[i]])
    return out
  }
  let gap = Infinity
  const sorted = Array.from(p.x).sort((a, b) => a - b)
  for (let i = 1; i < sorted.length; i++) if (sorted[i] > sorted[i - 1]) gap = Math.min(gap, sorted[i] - sorted[i - 1])
  const w = p.width ?? (Number.isFinite(gap) ? 0.8 * gap : 0.8)
  for (let i = 0; i < p.x.length; i++) out.push([p.x[i] - w / 2, p.x[i] + w / 2, base, p.y[i]])
  return out
}

/** Bars as one custom series of rectangles in data units, so they line up exactly with the axes in any orientation. */
export function barSeries(
  id: string,
  name: string,
  bars: Bar[],
  orient: Orient | undefined,
  style: { color: string; opacity: number; outline?: string; colors?: readonly string[] },
) {
  return {
    id,
    name,
    type: 'custom',
    data: bars,
    encode: orient === 'y' ? { x: [2, 3], y: [0, 1] } : { x: [0, 1], y: [2, 3] },
    clip: true,
    itemStyle: { color: style.color },
    renderItem: (params: CustomParams, api: CustomApi) => {
      const [a, b] = [
        api.coord(oriented(orient, api.value(0), api.value(2))),
        api.coord(oriented(orient, api.value(1), api.value(3))),
      ]
      return {
        type: 'rect',
        shape: {
          x: Math.min(a[0], b[0]),
          y: Math.min(a[1], b[1]),
          width: Math.abs(b[0] - a[0]),
          height: Math.abs(b[1] - a[1]),
        },
        style: {
          fill: style.colors?.[params.dataIndex] ?? style.color,
          opacity: style.opacity,
          ...(style.outline ? { stroke: style.outline, lineWidth: 1 } : {}),
        },
      }
    },
    z: 1,
  }
}

const barsHover = (name: string, color: string, bars: Bar[], orient: Orient | undefined): HoverSeries[] =>
  orient === 'y' ? [] : [{ label: name, color, x: bars.map((b) => (b[0] + b[1]) / 2), y: bars.map((b) => b[3]) }]

/** Bars from a base (default 0), vertical or, with `orient="y"`, horizontal. */
export const Bars = defineLayer<BarsProps>({
  kind: 'Bars',
  extent: (p) => {
    const bars = barsOf(p)
    const along = extentOf(
      bars.map((b) => b[0]),
      bars.map((b) => b[1]),
    )
    const heights = extentOf(
      bars.map((b) => b[2]),
      bars.map((b) => b[3]),
    )
    return orientedExtent(p.orient, along, heights)
  },
  build: (p, ctx) => {
    const name = p.name ?? 'bars'
    const bars = barsOf(p)
    return {
      series: [
        barSeries(ctx.id, name, bars, p.orient, {
          color: ctx.color,
          opacity: p.opacity ?? (p.edges ? 0.5 : 1),
          colors: p.colors,
          outline: p.edges ? ctx.color : undefined,
        }),
      ],
      hover: barsHover(name, ctx.color, bars, p.orient),
      data: { kind: 'bars', name, x: Array.from(p.x), y: Array.from(p.y) },
    }
  },
})

// ── Area ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

export type AreaProps = CommonProps & {
  x: Values
  y: Values
  /** The lower edge: a constant (default 0) or a value per x (a band between two curves). */
  base?: number | Values
  orient?: Orient
  /** Fill opacity (default 0.25). */
  opacity?: number
  /** Draw the upper edge as a line (default true). */
  line?: boolean
}

/** A filled polygon through the points `pts` (data units), as one custom-series element. */
export function polygonSeries(id: string, name: string, pts: number[][], color: string, opacity: number) {
  return {
    id,
    name,
    type: 'custom',
    data: pts.length ? [pts[0]] : [],
    encode: { x: 0, y: 1 },
    clip: true,
    silent: true,
    itemStyle: { color },
    tooltip: { show: false },
    renderItem: (_: CustomParams, api: CustomApi) => ({
      type: 'polygon',
      shape: { points: pts.map((q) => api.coord(q)) },
      style: { fill: color, opacity },
    }),
    z: 1,
  }
}

const baseAt = (base: AreaProps['base'], i: number) =>
  base === undefined ? 0 : typeof base === 'number' ? base : base[i]

/** The region between a curve and a base (zero, a constant or a second curve), filled, with its edge drawn. */
export const Area = defineLayer<AreaProps>({
  kind: 'Area',
  extent: (p) => {
    const lows = typeof p.base === 'object' ? p.base : [p.base ?? 0]
    return orientedExtent(p.orient, extentOf(p.x), extentOf(p.y, lows))
  },
  build: (p, ctx) => {
    const name = p.name ?? 'area'
    const top: number[][] = []
    const bottom: number[][] = []
    for (let i = 0; i < p.x.length; i++) {
      if (!Number.isFinite(p.y[i]) || !Number.isFinite(p.x[i])) continue
      top.push(oriented(p.orient, p.x[i], p.y[i]))
      bottom.push(oriented(p.orient, p.x[i], baseAt(p.base, i)))
    }
    const series: Record<string, unknown>[] = [
      polygonSeries(`${ctx.id}:fill`, name, [...top, ...bottom.reverse()], ctx.color, p.opacity ?? 0.25),
    ]
    if (p.line !== false)
      series.push({
        id: `${ctx.id}:edge`,
        name,
        type: 'line',
        data: top,
        showSymbol: false,
        clip: true,
        lineStyle: { width: LINE_WIDTH, color: ctx.color },
        itemStyle: { color: ctx.color },
        z: 3,
      })
    return {
      series,
      hover: p.orient === 'y' ? [] : [{ label: name, color: ctx.color, x: p.x, y: p.y }],
      data: { kind: 'area', name, x: Array.from(p.x), y: Array.from(p.y) },
    }
  },
})

// ── SignedArea ───────────────────────────────────────────────────────────────────────────────────────────────────────

export type SignedAreaProps = CommonProps & {
  x: Values
  y: Values
  /** The label before the net area, drawn in the plot's top-left corner (default "net area"); false hides it. */
  label?: string | false
}

/**
 * A curve whose area above zero and below zero take the two ends of the diverging scale, with its net area (by the
 * trapezoid rule, split at zero crossings) written in the plot, e.g. the KL integrand p log(p/q).
 */
export const SignedArea = defineLayer<SignedAreaProps>({
  kind: 'SignedArea',
  slotted: () => false,
  extent: (p) => ({ x: extentOf(p.x), y: withZero(extentOf(p.y)) }),
  build: (p, ctx) => {
    const name = p.name ?? 'signed area'
    const stops = scaleStops('diverging', ctx.mode)
    // The saturated middle of each half of the diverging scale (its darkest ends turn muddy when translucent).
    const [neg, pos] = [stops[2], stops[stops.length - 3]]
    const { positive, negative } = signedParts(p.x, p.y)
    const close = (pts: number[][]) => [...pts, [pts[pts.length - 1]?.[0] ?? 0, 0], [pts[0]?.[0] ?? 0, 0]]
    const area = signedArea(p.x, p.y)
    const ink = p.color ?? chrome(ctx.mode).ink
    const label = p.label === false ? null : `${p.label ?? 'net area'} = ${formatNumber(area.net)}`
    return {
      series: [
        polygonSeries(`${ctx.id}:pos`, '__positive', close(positive), pos, 0.5),
        polygonSeries(`${ctx.id}:neg`, '__negative', close(negative), neg, 0.5),
        {
          id: `${ctx.id}:line`,
          name,
          type: 'line',
          data: pairs(p.x, p.y),
          showSymbol: false,
          clip: true,
          lineStyle: { width: LINE_WIDTH, color: ink },
          itemStyle: { color: ink },
          z: 3,
        },
      ],
      hover: [{ label: name, color: ink, x: p.x, y: p.y }],
      overlay: label
        ? (g) => (
            // At the plot area's top-left corner: the net area, then its two parts.
            <div
              key={`${ctx.id}:net`}
              className="pointer-events-none absolute rounded-sm bg-background/80 px-1 text-[11px] text-foreground tabular-nums"
              style={{ left: g.left + 6, top: g.top + 4 }}
            >
              {label}
              <span className="ml-1.5 text-muted-foreground">
                (+{formatNumber(area.positive)} − {formatNumber(area.negative)})
              </span>
            </div>
          )
        : undefined,
      data: { kind: 'signed-area', name, x: Array.from(p.x), y: Array.from(p.y), ...area },
    }
  },
})

// ── Segments ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One segment from `from` to `to` in data coordinates (a residual, a matched pair, an edge). */
export type Segment = { from: [number, number]; to: [number, number] }

export type SegmentsProps = CommonProps & {
  segments: readonly { from: readonly [number, number]; to: readonly [number, number] }[]
  width?: number
  dashed?: boolean
}

/** Line segments, muted and thin by default (residuals, a mesh, the steps of a walk); one series split by nulls. */
export const Segments = defineLayer<SegmentsProps>({
  kind: 'Segments',
  legend: () => [],
  slotted: () => false,
  extent: (p) => ({
    x: extentOf(p.segments.flatMap((s) => [s.from[0], s.to[0]])),
    y: extentOf(p.segments.flatMap((s) => [s.from[1], s.to[1]])),
  }),
  build: (p, ctx) => {
    const own = p.slot !== undefined || p.color || p.emphasis
    const color = own ? ctx.color : chrome(ctx.mode).muted
    return {
      series: [
        {
          id: ctx.id,
          name: p.name ?? '__segments',
          type: 'line',
          data: p.segments.flatMap((s) => [s.from, s.to, [null, null]]),
          connectNulls: false,
          showSymbol: false,
          silent: true,
          clip: true,
          lineStyle: { width: p.width ?? 1, color, opacity: own ? 1 : 0.6, type: p.dashed ? 'dashed' : 'solid' },
          tooltip: { show: false },
          z: 1,
        },
      ],
    }
  },
})

// ── Vectors ──────────────────────────────────────────────────────────────────────────────────────────────────────────

export type VectorsProps = CommonProps & { vectors: readonly Vector[] }

/**
 * Arrows, ink unless a vector sets `slot`; a vector's `label` sits at its tip, or upright on a pill at its midpoint with
 * `labelAt: 'middle'`. Clipped to the drawn box: an arrow leaving the plot ends at the edge with a chevron pointing the
 * way it goes (ECharts would drop it otherwise).
 */
export const Vectors = defineLayer<VectorsProps>({
  kind: 'Vectors',
  legend: () => [],
  slotted: () => false,
  needsBox: true,
  extent: (p) => {
    const ends = vectorEnds(p.vectors)
    return { x: extentOf(ends.map((e) => e[0])), y: extentOf(ends.map((e) => e[1])) }
  },
  build: (p, ctx) => ({
    series: [
      {
        id: ctx.id,
        name: '__vectors',
        type: 'line',
        data: [],
        silent: true,
        tooltip: { show: false },
        markLine: {
          silent: true,
          symbol: ['none', 'arrow'],
          symbolSize: 10,
          label: { show: false },
          lineStyle: { color: chrome(ctx.mode).ink, width: LINE_WIDTH, type: 'solid' },
          animation: false,
          data: vectorLines(p.vectors, ctx.mode, ctx.box),
        },
        z: 6,
      },
      // Midpoint labels (a count on a move), upright over the shaft.
      {
        id: `${ctx.id}:labels`,
        name: '__vector-labels',
        type: 'scatter',
        data: vectorMidLabels(p.vectors, ctx.mode, ctx.box),
        // ECharts drops the label of a symbol 'none' point; an invisible dot carries it.
        symbol: 'circle',
        symbolSize: 1,
        itemStyle: { color: 'transparent' },
        silent: true,
        tooltip: { show: false },
        animation: false,
        z: 7,
      },
    ],
  }),
})

// ── Vector field ─────────────────────────────────────────────────────────────────────────────────────────────────────

export type VectorFieldProps = CommonProps &
  FieldArrowOptions & {
    /** The field: the vector at the point (x, y). */
    field: (x: number, y: number) => readonly [number, number]
    /**
     * The magnitudes at the two ends of the colour scale (default: the smallest and largest on the grid). Fix it when
     * the field changes (a slider, an animation), so a colour keeps its meaning.
     */
    range?: readonly [number, number]
    /** Arrowhead size in pixels (default 5). */
    head?: number
    /** How each arrow is drawn (default `'arrow'`): `'arrow'`, `'triangle'`, `'line'` (no head) or `'dot'` (a dot at the base). */
    arrow?: ArrowStyle
  }

/**
 * A vector field on a grid. By default every arrow has the same length, thin and with a small head, and its colour
 * shows the field's magnitude there, from blue at the smallest to red at the largest, so direction stays readable
 * where the field is weak and the picture is not dominated by its largest values. `length="magnitude"` draws lengths in proportion
 * instead, in ink (or in `color`, or muted).
 */
export const VectorField = defineLayer<VectorFieldProps>({
  kind: 'VectorField',
  legend: () => [],
  slotted: () => false,
  needsBox: true,
  extent: (p) => {
    // Arrows are centred on the grid points, so those on the boundary reach half a length beyond it.
    const [nx, ny] = typeof p.n === 'number' ? [p.n, p.n] : (p.n ?? [15, 15])
    const reach = (p.scale ?? 0.8) / 2
    const px = (reach * (p.x[1] - p.x[0])) / (nx - 1)
    const py = (reach * (p.y[1] - p.y[0])) / (ny - 1)
    return { x: [p.x[0] - px, p.x[1] + px], y: [p.y[0] - py, p.y[1] + py] }
  },
  build: (p, ctx) => {
    const arrows = fieldArrows(p.field, p)
    const magnitudes = arrows.map((a) => a.magnitude)
    const [lo, hi] = p.range ?? [Math.min(...magnitudes), Math.max(...magnitudes)]
    const stops = coolWarm(ctx.mode)
    const coloured = (p.length ?? 'unit') === 'unit' && !p.color && !p.muted
    const vectors = arrows.map(({ from, to, magnitude }): Vector => {
      const t = hi > lo ? (magnitude - lo) / (hi - lo) : 1
      return {
        from,
        to,
        width: 1,
        head: p.head ?? 5,
        style: p.arrow,
        muted: p.muted,
        color: coloured ? interpolateColors(stops, t) : p.color,
      }
    })
    return {
      series: [
        {
          id: ctx.id,
          name: '__vector-field',
          type: 'line',
          data: [],
          silent: true,
          tooltip: { show: false },
          markLine: {
            silent: true,
            symbol: ['none', 'arrow'],
            symbolSize: p.head ?? 5,
            label: { show: false },
            lineStyle: { color: chrome(ctx.mode).ink, width: 1, type: 'solid' },
            animation: false,
            data: vectorLines(vectors, ctx.mode, ctx.box),
          },
          z: 5,
        },
      ],
    }
  },
})

// ── Rug ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

export type RugProps = CommonProps & {
  values: Values
  orient?: Orient
  /** Tick length in pixels (default 8). */
  length?: number
}

/** A tick per value along the bottom edge (or the left edge with `orient="y"`), e.g. the draws under a density. */
export const Rug = defineLayer<RugProps>({
  kind: 'Rug',
  legend: () => [],
  extent: (p) => orientedExtent(p.orient, extentOf(p.values), undefined),
  build: (p, ctx) => {
    const values = Array.from(p.values).filter(Number.isFinite)
    const length = p.length ?? 8
    const other = (axis: 'x' | 'y') => ((axis === 'x' ? ctx.x : ctx.y).log ? 1 : 0)
    return {
      series: [
        {
          id: ctx.id,
          name: p.name ?? '__rug',
          type: 'custom',
          data: values.length ? [[values[0], 0]] : [],
          encode: { x: 0, y: 1 },
          silent: true,
          clip: true,
          tooltip: { show: false },
          itemStyle: { color: ctx.color },
          renderItem: (params: CustomParams, api: CustomApi) => {
            const box = params.coordSys
            return {
              type: 'group',
              children: values.map((v) => {
                if (p.orient === 'y') {
                  const py = api.coord([other('x'), v])[1]
                  return {
                    type: 'line',
                    shape: { x1: box.x, y1: py, x2: box.x + length, y2: py },
                    style: { stroke: ctx.color, opacity: 0.6, lineWidth: 1 },
                  }
                }
                const px = api.coord([v, other('y')])[0]
                const bottom = box.y + box.height
                return {
                  type: 'line',
                  shape: { x1: px, y1: bottom, x2: px, y2: bottom - length },
                  style: { stroke: ctx.color, opacity: 0.6, lineWidth: 1 },
                }
              }),
            }
          },
          z: 4,
        },
      ],
    }
  },
})

// ── Annotation ───────────────────────────────────────────────────────────────────────────────────────────────────────

export type AnnotationProps = CommonProps & {
  /** A labelled point. */
  at?: readonly [number, number]
  /** A labelled vertical line at this x. */
  x?: number
  /** A labelled horizontal line at this y. */
  y?: number
  text?: string
  dashed?: boolean
}

/** A labelled point, vertical line or horizontal line; ink unless given a slot or colour. */
export const Annotation = defineLayer<AnnotationProps>({
  kind: 'Annotation',
  legend: () => [],
  slotted: () => false,
  // A vertical line's text sits above the plot area.
  margins: (p) => (p.x !== undefined && p.text ? { labelRow: LABEL_ROW } : {}),
  extent: (p) => ({
    x: p.at ? [p.at[0], p.at[0]] : p.x !== undefined ? [p.x, p.x] : undefined,
    y: p.at ? [p.at[1], p.at[1]] : p.y !== undefined ? [p.y, p.y] : undefined,
  }),
  build: (p, ctx) => {
    const color = p.slot !== undefined || p.color ? ctx.color : chrome(ctx.mode).ink
    const label = { show: !!p.text, formatter: p.text ?? '', color, fontSize: 11 }
    // A vertical line's text above its top end (in the row the Plot keeps for it); a horizontal line's inside the plot,
    // above its right end, where the right margin cannot clip it.
    const lines = [
      ...(p.x !== undefined ? [{ xAxis: p.x, label: { ...label, position: 'end' } }] : []),
      ...(p.y !== undefined ? [{ yAxis: p.y, label: { ...label, position: 'insideEndTop' } }] : []),
    ]
    return {
      series: [
        {
          id: ctx.id,
          name: '__annotation',
          type: 'scatter',
          data: p.at ? [[p.at[0], p.at[1]]] : [],
          symbol: 'circle',
          symbolSize: 7,
          silent: true,
          clip: true,
          itemStyle: { color },
          label: { ...label, position: 'right' },
          tooltip: { show: false },
          markLine: lines.length
            ? {
                silent: true,
                symbol: ['none', 'none'],
                animation: false,
                lineStyle: { color, width: 1, type: p.dashed === false ? 'solid' : 'dashed' },
                data: lines,
              }
            : undefined,
          z: 6,
        },
      ],
    }
  },
})

// ── Handle ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A draggable handle bound to a parameter with a natural place on the chart (a start point, a threshold, a centroid):
 * `kind="point"` with `at=[x, y]`, or `kind="x"`/`"y"` for a guide line. It moves by patch while dragged, and the axes
 * hold still. Distribution parameters stay on sliders.
 */
export const HandleLayer = defineLayer<HandleSpec & CommonProps>({
  kind: 'Handle',
  legend: () => [],
  slotted: () => false,
  // An x guide's label sits above the plot area.
  margins: (p) => (p.kind === 'x' && p.label ? { labelRow: LABEL_ROW } : {}),
  build: (p, ctx) => {
    const { name: _n, slot: _s, emphasis: _e, muted: _m, color: _c, live: _l, id: _i, stale: _st, ...handle } = p
    // A palette slot (or an explicit colour) colours the handle like the series it moves; otherwise it stays ink.
    const color = p.color ?? (p.slot !== undefined ? seriesColor(ctx.mode, p.slot) : undefined)
    return { series: [], handles: [{ ...handle, ...(color ? { color } : {}) } as HandleSpec] }
  },
})
