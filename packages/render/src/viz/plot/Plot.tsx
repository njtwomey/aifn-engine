import { useId, useLayoutEffect, useState, type ReactNode } from 'react'
import { useTheme } from '@render/design/theme'
import { EChart, type PlotPointer } from '../EChart'
import { formatNumber, formatPower } from '../format'
import { useChartHeight, useElementSize, useFrameData, useFrameHover, type HoverRow } from '../frame'
import type { Handle as HandleSpec } from '../handles'
import { LABEL_GAP, tickLabelWidth, Y_NAME_ROOM, type Margins } from './ticks'
import { GRID } from '../theme'
import type { Range } from '../viewport'
import { useAxisVersion, type AxisModel, type Extent } from './axis'
import { AxisToolbar } from './AxisToolbar'
import {
  collectLayers,
  escapeHtml,
  layerColor,
  type CommonProps,
  type HoverSeries,
  type LayerContext,
  type LayerDef,
  type LayerOutput,
  type PlotBox,
} from './layer'
import { usePlotsCell } from './plots-context'

export type PlotProps = {
  /** The axis models (from `useAxis`). Plots given the same model share that axis: one range, one zoom, one toolbar. */
  x: AxisModel
  y: AxisModel
  /** Layers (`Curve`, `Points`, `Raster`, …), in drawing order. */
  children?: ReactNode
  /** A small title at the top left of the plot, e.g. a panel's name in a grid. */
  title?: string
  /** Force the legend on or off; by default it shows when the layers name two or more things. */
  legend?: boolean
  /** Plots sharing a hover group share the hovered x. Inside `Plots` the grid's group is used. */
  hoverGroup?: string
  /** The axis toolbar of a Plot on its own (default false). In a `Plots` grid the grid draws one for all axes. */
  toolbar?: boolean
  /** No axes, ticks or grid lines (a grid of people, where coordinates mean nothing). */
  bare?: boolean
  /** Pixels; inside a Figure or a Plots grid the frame sets the height. */
  height?: number
  /** A share of that height, e.g. 0.5 for a strip under a larger chart in the same Figure. */
  scale?: number
  /**
   * With equal units and both ranges fixed (glyphs, a strip of shapes): the height follows the width, so the plot area
   * takes exactly the ranges' aspect at any width. Overrides `height`.
   */
  fitHeight?: boolean
  /** Override the renderer (default: canvas when a layer has thousands of marks, else SVG). Fixed at mount. */
  renderer?: 'svg' | 'canvas'
  /** Hover, click and leave positions in data coordinates. */
  onPointer?: (event: PlotPointer) => void
  /** Clicks in the plot area, in data coordinates. Prefer a `Handle` where the value has a place on the chart. */
  onPlotClick?: (point: [number, number]) => void
  /** Rectangle brushing: the selected region on release, or null on a plain click. */
  onBrush?: (rect: { x: Range; y: Range } | null) => void
  ariaLabel?: string
}

type Entry = { def: LayerDef<CommonProps>; props: CommonProps; key: string; slot: number }
type Built = { def: LayerDef<CommonProps>; props: CommonProps; ctxKey: string; out: LayerOutput }
type Extented = {
  def: LayerDef<CommonProps>
  props: CommonProps
  extent: { x?: Range; y?: Range; tight?: boolean } | undefined
}

/** Per-Plot memo of each layer's extent and output, keyed by its props (shallow) and the context it was built in. */
type Memo = {
  extents: Map<string, Extented>
  builds: Map<string, Built>
  option?: { deps: readonly unknown[]; value: unknown }
  patch?: { deps: readonly unknown[]; value: unknown }
  data?: { deps: readonly unknown[]; value: () => unknown }
  handles?: { key: string; list: HandleSpec[] }
  latestHandles: HandleSpec[]
  /** Every layer's hover series this render, live ones included: the tooltip reads it when it opens. */
  hover: HoverSeries[]
  /** The static layers' outputs the current option was built from; later outputs of the same structure are patched. */
  optionDrawn: LayerOutput[]
}

/** Props equal by identity, except short arrays (an inline list of names or a point), which compare by value. */
const sameProp = (a: unknown, b: unknown) =>
  Object.is(a, b) ||
  (Array.isArray(a) &&
    Array.isArray(b) &&
    a.length === b.length &&
    a.length <= 64 &&
    a.every((v, i) => Object.is(v, b[i])))
const shallowEqual = (a: object, b: object) => {
  const ka = Object.keys(a)
  if (ka.length !== Object.keys(b).length) return false
  return ka.every((k) => sameProp((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))
}
const sameDeps = (a: readonly unknown[], b: readonly unknown[]) =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]))

const unionOf = (ranges: (Range | undefined)[]): Range | undefined => {
  let lo = Infinity
  let hi = -Infinity
  for (const r of ranges) {
    if (!r) continue
    lo = Math.min(lo, r[0])
    hi = Math.max(hi, r[1])
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined
}

/**
 * The union of the layers' extents on one axis, tight when both its ends are set by tight extents (a raster's grid
 * alone, or with marks inside it), so the axis ends at the grid's edges.
 */
function tightUnion(
  extents: ({ x?: Range; y?: Range; tight?: boolean } | undefined)[],
  axis: 'x' | 'y',
): Extent | undefined {
  const r = unionOf(extents.map((e) => e?.[axis]))
  if (!r) return undefined
  const tight = (end: 0 | 1) => extents.some((e) => e?.tight && e[axis]?.[end] === r[end])
  return tight(0) && tight(1) ? [r[0], r[1], true] : r
}

const FONT = "'Geist Variable', system-ui, sans-serif"
let measure: CanvasRenderingContext2D | null | undefined
const textWidth = (text: string, size = 11) => {
  if (measure === undefined)
    measure = typeof document === 'undefined' ? null : document.createElement('canvas').getContext('2d')
  if (!measure) return text.length * size * 0.6
  measure.font = `${size}px ${FONT}`
  return measure.measureText(text).width
}

/** The legend's geometry (theme.ts): icon, gap to the text, gap between items and between rows, row pitch. */
const LEGEND = { icon: 8, iconGap: 5, itemGap: 10, row: 22, first: 28 }

/** How many rows the legend wraps onto in `width` pixels: items laid out left to right, a new row when one is full. */
function legendRows(names: readonly string[], width: number): number {
  if (!names.length || width <= 0) return 1
  let rows = 1
  let used = 0
  for (const name of names) {
    const w = LEGEND.icon + LEGEND.iconGap + textWidth(name, 12)
    if (used > 0 && used + LEGEND.itemGap + w > width) {
      rows++
      used = w
    } else used += (used > 0 ? LEGEND.itemGap : 0) + w
  }
  return rows
}

/** The room a panel title needs above the plot area. */
const TITLE_ROW = 15

/** The widest tick label of a y axis over `r`, rounded up to 8 px so small changes do not move the margins. */
function labelWidthOf(axis: AxisModel, r: Range | undefined): number {
  const raw = axis.options.categories
    ? Math.max(24, ...axis.options.categories.map(textWidth))
    : tickLabelWidth(r, axis.log)
  return Math.ceil(raw / 8) * 8
}

const BARE: Margins = { left: 8, right: 8, top: 8, bottom: 8 }

/**
 * A chart: two axis models and the layers drawn on them, as one ECharts option through `EChart`. Layers declare their
 * extents, which feed the axes' fit (except `live` layers); hover, legend and palette slots belong to the Plot, so
 * every chart behaves the same. Only layers whose props changed are rebuilt, and `live` layers and handles reach
 * ECharts as a patch, so a drag redraws nothing else.
 */
export function Plot({
  x,
  y,
  children,
  title,
  legend: legendProp,
  hoverGroup,
  toolbar = false,
  bare = false,
  height: ownHeight,
  scale = 1,
  fitHeight = false,
  renderer,
  onPointer,
  onPlotClick,
  onBrush,
  ariaLabel,
}: PlotProps) {
  const { resolved: mode } = useTheme()
  const cell = usePlotsCell()
  const height = Math.round(useChartHeight(ownHeight) * scale)
  const source = useId()
  const [memo] = useState<Memo>(() => ({
    extents: new Map(),
    builds: new Map(),
    latestHandles: [],
    hover: [],
    optionDrawn: [],
  }))
  const [box, size] = useElementSize<HTMLDivElement>()

  // ── Layers, slots and extents ──────────────────────────────────────────────────────────────────────────────────────
  const layers = collectLayers(children)
  const extra = layers.flatMap((l) => l.def.axes?.(l.props) ?? []).filter((a): a is AxisModel => !!a)
  useAxisVersion([x, y, ...extra])
  let category = 0
  const entries: Entry[] = layers.map((l, i) => {
    const slotted = l.def.slotted ? l.def.slotted(l.props) : !l.props.muted && !l.props.emphasis && !l.props.tone
    const slot = l.props.slot ?? (slotted ? category++ : 0)
    // A layer that is live by definition (a probe) is live unless its props say otherwise.
    const props = l.def.live && l.props.live === undefined ? { ...l.props, live: true } : l.props
    return { def: l.def, props, key: l.props.id ?? `${l.def.kind}${i}`, slot }
  })
  const extents = entries.map((e) => {
    if (e.props.live || !e.def.extent) return undefined
    const hit = memo.extents.get(e.key)
    if (hit && hit.def === e.def && shallowEqual(hit.props, e.props)) return hit.extent
    const extent = e.def.extent(e.props)
    memo.extents.set(e.key, { def: e.def, props: e.props, extent })
    return extent
  })
  const xExtent = tightUnion(extents, 'x')
  const yExtent = tightUnion(extents, 'y')
  const xr = x.resolve(source, xExtent)
  const yr = y.resolve(source, yExtent)

  // ── Margins and the plot area ──────────────────────────────────────────────────────────────────────────────────────
  // Live layers keep their legend entry: their series sit in the option (without data), so the name is drawn and
  // toggles the patched data like any other.
  const legendNames = [
    ...new Set(entries.flatMap((e) => (e.def.legend ? e.def.legend(e.props) : e.props.name ? [e.props.name] : []))),
  ]
  const legendShown = legendProp ?? legendNames.length > 1
  const ctxBase = (e: Entry): LayerContext => ({
    mode,
    color: layerColor(e.props, e.slot, mode),
    slot: e.slot,
    id: e.key,
    range: (axis, extent) => axis.resolve(`${source}:${e.key}`, extent),
    x,
    y,
  })
  const needs = entries.map((e) => e.def.margins?.(e.props, ctxBase(e)) ?? {})
  const right = Math.max(GRID.right, ...needs.map((n) => n.right ?? 0))
  const labelWidth = labelWidthOf(y, yr)
  const xName = !!x.options.label
  const yName = !!y.options.label
  const ownLeft = labelWidth + LABEL_GAP + (yName ? Y_NAME_ROOM : 0) + 4
  // Above the plot area, from the top: the legend (wrapped, never paged), the title, then a row for labels drawn above
  // the plot (an x handle's, a vertical annotation's). Room a layer needs beside the plot (a colour bar's name, right
  // of the plot area) does not stack with them: the legend ends at the plot area's right edge, clear of it, and may
  // reach over the y labels.
  const legendWidth = size.width - (cell?.margins.right ?? right) - 8
  const rows = legendShown ? legendRows(legendNames, legendWidth) : 0
  const labelRow = Math.max(0, ...needs.map((n) => n.labelRow ?? 0))
  const stacked = (rows ? LEGEND.first + (rows - 1) * LEGEND.row : 12) + (title ? TITLE_ROW : 0) + labelRow
  const top = Math.max(stacked, ...needs.map((n) => n.top ?? 0))
  const own: Margins = bare ? BARE : { left: ownLeft, right, top, bottom: xName ? 44 : 28 }
  const margins = cell?.margins ?? own
  const showX = !bare && (cell?.labels.x ?? true)
  const showY = !bare && (cell?.labels.y ?? true)

  // Equal units: the plot area takes the ranges' aspect and is centred in the room the margins leave (no padding of
  // the axes). In a single-column Plots grid the grid sizes the panel so this leaves nothing over.
  const equal = x.partner === y || y.partner === x
  const fitted =
    fitHeight && equal && xr && yr && size.width > 0 && !x.log && !y.log
      ? Math.round(
          ((size.width - margins.left - margins.right) * (yr[1] - yr[0])) / (xr[1] - xr[0]) +
            margins.top +
            margins.bottom,
        )
      : undefined
  let grid = margins
  const roomW = size.width - margins.left - margins.right
  const roomH = size.height - margins.top - margins.bottom
  if (equal && roomW > 0 && roomH > 0 && xr && yr && !x.log && !y.log) {
    const aspect = (yr[1] - yr[0]) / (xr[1] - xr[0])
    if (roomW * aspect <= roomH) {
      const extraH = Math.floor((roomH - roomW * aspect) / 2)
      grid = { ...margins, top: margins.top + extraH, bottom: margins.bottom + extraH }
    } else {
      const extraW = Math.floor((roomW - roomH / aspect) / 2)
      grid = { ...margins, left: margins.left + extraW, right: margins.right + extraW }
    }
  }
  const plot = {
    width: Math.max(size.width - grid.left - grid.right, 1),
    height: Math.max(size.height - grid.top - grid.bottom, 1),
  }
  const plotBox: PlotBox | undefined = xr && yr ? { x: xr, y: yr, xLog: x.log, yLog: y.log } : undefined

  // ── Build the layers (memoised) ────────────────────────────────────────────────────────────────────────────────────
  const boxKey = plotBox ? `${plotBox.x.join(',')}|${plotBox.y.join(',')}` : ''
  const plotKey = `${plot.width}x${plot.height}`
  const axesKey = `${x.id}:${x.options.label}:${x.log}|${y.id}:${y.options.label}:${y.log}`
  const outputs = entries.map((e) => {
    const extraKey = (e.def.axes?.(e.props) ?? []).map((a) => a?.getVersion() ?? '').join(',')
    const ctxKey = [
      mode,
      e.slot,
      axesKey,
      extraKey,
      flagFor(e.def.needsBox, e.props) ? boxKey : '',
      flagFor(e.def.needsPlot, e.props) ? plotKey : '',
    ].join('|')
    const hit = memo.builds.get(e.key)
    if (hit && hit.def === e.def && hit.ctxKey === ctxKey && shallowEqual(hit.props, e.props)) return hit.out
    const built = e.def.build(e.props, { ...ctxBase(e), box: plotBox, plot })
    const out = e.props.stale ? { ...built, series: built.series.map(dimmed) } : built
    if (out !== built) undimmed.set(out, built)
    memo.builds.set(e.key, { def: e.def, props: e.props, ctxKey, out })
    return out
  })
  for (const key of memo.builds.keys()) if (!entries.some((e) => e.key === key)) memo.builds.delete(key)
  const staticOuts = outputs.filter((_, i) => !entries[i].props.live)
  const liveOuts = outputs.filter((_, i) => entries[i].props.live)
  // Live layers are hovered too: the tooltip and the readout read the latest hover series when they open, so a patch
  // that moves a live curve needs no new option.
  memo.hover = outputs.flatMap((o) => o.hover ?? [])
  const hovered = memo.hover.length > 0

  // Live layers sit in the option without data (their structure, keyed by value); their data goes in the patch.
  const liveKey = liveOuts.map(structureOf).join('\u0001')

  // ── The option ─────────────────────────────────────────────────────────────────────────────────────────────────────
  const [x0, x1] = xr ?? []
  const [y0, y1] = yr ?? []
  const yNameGap = cell?.yNameGap ?? labelWidth + LABEL_GAP
  const covered = entries.some((e) => e.def.covers && !e.props.live)
  // Layers that draw nothing in the option (a Handle: its marks go through the handles patch) do not redraw it.
  const drawn = staticOuts.filter((o) => o.series.length || o.hover?.length || o.tooltip)
  // The option depends on the static layers' structure (series ids, types, styles), not their data: a layer whose data
  // alone changed (a result from the scheduler, a worker) reaches ECharts in the patch, as a live layer's does.
  const staticKey = drawn.map(structureOf).join('\u0001')
  const optionDeps = [
    covered,
    mode,
    hovered,
    staticKey,
    liveKey,
    x0,
    x1,
    y0,
    y1,
    grid.left,
    grid.right,
    grid.top,
    grid.bottom,
    axesKey,
    showX,
    showY,
    legendShown,
    legendNames.join('\u0000'),
    legendWidth,
    size.width,
    bare,
    yNameGap,
  ]
  if (!memo.option || !sameDeps(memo.option.deps, optionDeps)) {
    const series: Record<string, unknown>[] = []
    outputs.forEach((o, i) => {
      if (entries[i].props.live) series.push(...o.series.map(strip))
      else series.push(...o.series)
    })
    const hover = staticOuts.flatMap((o) => o.hover ?? [])
    const tooltips = Object.assign({}, ...staticOuts.map((o) => o.tooltip ?? {})) as NonNullable<LayerOutput['tooltip']>
    // Hovered by x when any layer reports values along x. Bars and areas are custom series, which do not trigger the
    // axis tooltip themselves: an invisible line through the hovered xs carries it.
    if (hover.length && !series.some((s) => s.type === 'line' && !s.silent && !String(s.name).startsWith('__')))
      series.push({
        id: '__hover',
        name: '__hover',
        type: 'line',
        data: [...new Set(hover.flatMap((h) => Array.from(h.x)))].sort((a, b) => a - b).map((v) => [v, y0 ?? 0]),
        showSymbol: false,
        lineStyle: { opacity: 0 },
        silent: true,
        z: 0,
      })
    const fx = (v: number) =>
      x.options.categories ? x.label(v) || formatNumber(v) : (x.options.format ?? formatNumber)(v)
    const fy = y.options.format ?? formatNumber
    const xLabel = x.options.label ?? 'x'
    const icon = (name: string) => {
      const s = series.find((v) => v.name === name && v.type === 'scatter')
      return s && typeof s.symbol === 'string' ? { icon: s.symbol } : {}
    }
    memo.optionDrawn = drawn
    memo.option = {
      deps: optionDeps,
      value: {
        grid: { ...grid, outerBoundsMode: 'none' },
        // Right-aligned with the plot area and wrapped (never paged), so it clears a colour bar's name.
        legend: {
          data: legendNames.map((name) => ({ name, ...icon(name) })),
          show: legendShown,
          right: grid.right,
          width: Math.max(size.width - grid.right - 8, 1),
        },
        tooltip: hovered
          ? {
              trigger: 'axis',
              axisPointer: { type: 'line', snap: false, label: { show: false } },
              formatter: (params: { axisValue: number }[]) => {
                if (!params.length) return ''
                const at = params[0].axisValue
                return (
                  `<div style="opacity:0.7;margin-bottom:2px">${escapeHtml(xLabel)} = ${escapeHtml(fx(at))}</div>` +
                  rowsAt(memo.hover, at, fy)
                    .map(
                      (r) =>
                        `<div style="display:flex;gap:12px;justify-content:space-between"><span><span style="display:inline-block;width:8px;height:8px;border-radius:4px;margin-right:6px;background:${r.color}"></span>${escapeHtml(r.label)}</span>` +
                        `<span style="font-variant-numeric:tabular-nums;font-weight:500">${escapeHtml(r.value)}</span></div>`,
                    )
                    .join('')
                )
              },
            }
          : {
              trigger: 'item',
              formatter: (p: {
                seriesId?: string
                seriesName: string
                name?: string
                value: unknown
                marker: string
                dataIndex: number
              }) => {
                const own = p.seriesId ? tooltips[p.seriesId] : undefined
                if (own) return own(p)
                const v = Array.isArray(p.value) ? (p.value as number[]) : []
                // A labelled point (Points' `labels`) names itself first.
                const head = p.name
                  ? `<b>${escapeHtml(p.name)}</b> · ${escapeHtml(p.seriesName)}`
                  : escapeHtml(p.seriesName)
                return `${p.marker}${head}<br/>(${escapeHtml(fx(v[0]))}, ${escapeHtml(fy(v[1]))})`
              },
            },
        xAxis: {
          ...axisOption(x, xr, showX, bare, 'x', plot.width),
          ...(covered ? { splitLine: { show: false } } : {}),
        },
        yAxis: {
          ...axisOption(y, yr, showY, bare, 'y', plot.height),
          nameGap: yNameGap,
          ...(covered ? { splitLine: { show: false } } : {}),
        },
        series,
      },
    }
  }
  const option = memo.option.value as Record<string, unknown>

  // Static layers whose data changed since the option was built ride in the patch with the live layers.
  const changed = drawn.filter((o, i) => o !== memo.optionDrawn[i])
  const patchDeps = [...liveOuts, ...changed]
  if (!memo.patch || !sameDeps(memo.patch.deps, patchDeps))
    memo.patch = {
      deps: patchDeps,
      value: patchDeps.length ? { series: patchDeps.flatMap((o) => o.series) } : undefined,
    }
  const patch = memo.patch.value as Record<string, unknown> | undefined

  // Handles: a stable list while their positions and labels are unchanged; each delegates to the latest setter, since
  // widgets rebuild their callbacks every render.
  const latest = outputs.flatMap((o) => o.handles ?? [])
  memo.latestHandles = latest
  const handlesKey = JSON.stringify(latest.map((h) => [h.kind, h.at, h.label]))
  if (!memo.handles || memo.handles.key !== handlesKey)
    memo.handles = {
      key: handlesKey,
      list: latest.map((h, i) => ({
        ...h,
        onDrag: (v: never) => (memo.latestHandles[i]?.onDrag as (v: never) => void)?.(v),
        onRelease: () => memo.latestHandles[i]?.onRelease?.(),
      })) as HandleSpec[],
    }
  const handles = latest.length ? memo.handles.list : undefined

  // ── Reports: extents to the axes, needs to the grid ────────────────────────────────────────────────────────────────
  const [ex0, ex1, xTight] = xExtent ?? []
  const [ey0, ey1, yTight] = yExtent ?? []
  const xKey = x.options.key
  const yKey = y.options.key
  useLayoutEffect(() => {
    x.report(source, ex0 !== undefined && ex1 !== undefined ? (xTight ? [ex0, ex1, true] : [ex0, ex1]) : undefined)
  }, [x, source, ex0, ex1, xTight, xKey])
  useLayoutEffect(() => {
    y.report(source, ey0 !== undefined && ey1 !== undefined ? (yTight ? [ey0, ey1, true] : [ey0, ey1]) : undefined)
  }, [y, source, ey0, ey1, yTight, yKey])
  useLayoutEffect(() => {
    return () => {
      x.report(source, undefined)
      y.report(source, undefined)
    }
  }, [x, y, source])
  // Other axes a layer reads (a raster's colour axis).
  const extraReports = staticOuts.flatMap((o) => o.extents ?? [])
  const extraKey = extraReports.map((r) => `${r.axis.id}:${r.range.join(',')}`).join('|')
  useLayoutEffect(() => {
    const sent = extraReports.map((r) => {
      r.axis.report(`${source}:${r.axis.id}`, r.range)
      return r.axis
    })
    return () => sent.forEach((a) => a.report(`${source}:${a.id}`, undefined))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the reports' values
  }, [extraKey, source])

  const baseX = x.base()
  const baseY = y.base()
  const equalSpans = equal && baseX && baseY ? { xSpan: baseX[1] - baseX[0], ySpan: baseY[1] - baseY[0] } : undefined
  const reportTo = cell?.report
  const eqX = equalSpans?.xSpan
  const eqY = equalSpans?.ySpan
  useLayoutEffect(() => {
    reportTo?.(source, {
      x,
      y,
      labelWidth,
      yName,
      xName,
      right,
      top,
      ...(eqX !== undefined && eqY !== undefined ? { equal: { xSpan: eqX, ySpan: eqY } } : {}),
    })
  }, [reportTo, source, x, y, labelWidth, yName, xName, right, top, eqX, eqY])
  useLayoutEffect(() => () => reportTo?.(source, null), [reportTo, source])

  // ── Hover readout, data export ─────────────────────────────────────────────────────────────────────────────────────
  const setHover = useFrameHover()
  const pointers = staticOuts.flatMap((o) => (o.pointer ? [o.pointer] : []))
  const fy = y.options.format ?? formatNumber
  const fxLabel = (v: number) => (x.options.categories ? x.label(v) : (x.options.format ?? formatNumber)(v))
  const onAxisHover = (at: number | null) =>
    at === null
      ? setHover(null)
      : setHover({ at: `${x.options.label ?? 'x'} = ${fxLabel(at)}`, rows: rowsAt(memo.hover, at, fy) })
  const pointer = (event: PlotPointer) => {
    onPointer?.(event)
    if (!pointers.length) return
    if (event.type === 'leave') return setHover(null)
    if (event.type !== 'move') return
    for (const f of pointers) {
      const info = f(event.point)
      if (info) return setHover(info)
    }
    setHover(null)
  }
  const dataOf = staticOuts.map((o) => o.data)
  if (!memo.data || !sameDeps(memo.data.deps, dataOf)) {
    const [xl, yl] = [x.options.label, y.options.label]
    memo.data = { deps: dataOf, value: () => ({ kind: 'plot', x: xl, y: yl, layers: dataOf.filter(Boolean) }) }
  }
  useFrameData(memo.data.value)

  const onDragChange = (dragging: boolean) => {
    x.setFrozen(dragging ? xr : undefined)
    y.setFrozen(dragging ? yr : undefined)
  }
  const zoomable = x.zoomable || y.zoomable
  const onWheelZoom = zoomable
    ? (factor: number, [px, py]: [number, number]) => {
        if (x.zoomable) x.zoom(factor, px, false)
        if (y.zoomable) y.zoom(factor, py, false)
      }
    : undefined
  // A layer with thousands of marks that appears later (a stage switch from points to a raster) moves the chart to the
  // canvas renderer, once: the EChart is keyed by it, so it remounts with the new renderer and never switches back.
  const wantsCanvas = entries.some((e) => e.def.canvas?.(e.props))
  const [canvasLater, setCanvasLater] = useState(false)
  const canvas = renderer ?? (wantsCanvas || canvasLater ? 'canvas' : 'svg')
  if (!renderer && wantsCanvas && !canvasLater) setCanvasLater(true)
  const standaloneToolbar = !cell && toolbar && zoomable && !bare
  const geometry = { left: grid.left, top: grid.top, width: plot.width, height: plot.height, boxWidth: size.width }
  const overlays = size.width > 0 ? outputs.flatMap((o) => (o.overlay ? [o.overlay(geometry)] : [])) : []

  return (
    <div className="flex w-full shrink-0 flex-col gap-1 overflow-hidden" style={{ height: fitted ?? height }}>
      {standaloneToolbar && (
        <AxisToolbar
          className="px-1"
          axes={[
            { axis: x, direction: 'x' },
            { axis: y, direction: 'y' },
          ]}
        />
      )}
      <div ref={box} className="relative min-h-0 flex-1">
        <EChart
          key={canvas}
          option={option}
          patch={patch}
          height="fill"
          className="absolute inset-0"
          ariaLabel={ariaLabel}
          handles={handles}
          renderer={canvas}
          cartesian
          onPlotClick={onPlotClick}
          onPointer={onPointer || pointers.length ? pointer : undefined}
          onBrush={onBrush}
          onWheelZoom={onWheelZoom}
          onAxisHover={hovered ? onAxisHover : undefined}
          hoverGroup={hoverGroup ?? cell?.hoverGroup}
          onDragChange={onDragChange}
        />
        {title && (
          <div
            className="pointer-events-none absolute truncate text-xs font-medium text-muted-foreground"
            // Just above the plot area (and its label row), so it stays with a plot centred for equal units.
            style={{ left: grid.left, top: Math.max(0, grid.top - labelRow - TITLE_ROW - 1), maxWidth: plot.width }}
          >
            {title}
          </div>
        )}
        {overlays}
      </div>
    </div>
  )
}

/** How much of its opacity a stale layer keeps while a newer answer is computed (DESIGN.md §8a). */
const STALE_OPACITY = 0.35

/** A series drawn faded: its line, marks and fill at `STALE_OPACITY` of their opacity. */
function dimmed(series: Record<string, unknown>): Record<string, unknown> {
  const fade = (style: unknown) => {
    const o = (style ?? {}) as { opacity?: number }
    return { ...o, opacity: (o.opacity ?? 1) * STALE_OPACITY }
  }
  return {
    ...series,
    lineStyle: fade(series.lineStyle),
    itemStyle: fade(series.itemStyle),
    ...(series.areaStyle ? { areaStyle: fade(series.areaStyle) } : {}),
  }
}

const structures = new WeakMap<LayerOutput, string>()
/** A stale (faded) output's unfaded original: fading changes styles only, which the patch carries. */
const undimmed = new WeakMap<LayerOutput, LayerOutput>()

/**
 * A layer output's series without their data, as a string: equal strings differ only in data (or in the fading of a
 * stale layer), which the patch carries without a new option. Cached per output.
 */
function structureOf(o: LayerOutput): string {
  let key = structures.get(o)
  if (key === undefined) {
    const raw = undimmed.get(o) ?? o
    key = JSON.stringify(
      [raw.series.map(strip), Object.keys(raw.tooltip ?? {}), raw.hover?.map((h) => h.label)],
      (_, v) => (typeof v === 'function' ? 'ƒ' : v),
    )
    structures.set(o, key)
  }
  return key
}

/** A series without its data, for the option of a live layer (its data arrives in the patch). */
function strip(s: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...s, data: [] }
  if (s.markLine && typeof s.markLine === 'object') out.markLine = { ...(s.markLine as object), data: [] }
  return out
}

/** The nearest point of each hovered series at x, one row per name. */
function rowsAt(hover: readonly HoverSeries[], at: number, fy: (v: number) => string): HoverRow[] {
  const rows: HoverRow[] = []
  const seen = new Set<string>()
  for (const h of hover) {
    if (seen.has(h.label)) continue
    seen.add(h.label)
    let best = -1
    let distance = Infinity
    for (let i = 0; i < h.x.length; i++) {
      const d = Math.abs(h.x[i] - at)
      if (d < distance) [best, distance] = [i, d]
    }
    if (best >= 0) rows.push({ label: h.label, value: fy(h.y[best]), color: h.color })
  }
  return rows
}

/** One axis of the option, from its model and its resolved range. */
/** A number rounded to 12 significant digits: 0.7000000000000001 becomes 0.7. */
const clean = (v: number) => (v === 0 ? 0 : Number(v.toPrecision(12)))

/**
 * A range ECharts can tick: finite ends, cleaned of floating-point noise, and a positive span (a degenerate one, every
 * value equal, is widened about its value). ECharts asserts on anything else and takes the whole figure down: a min of
 * 0.7000000000000001 (a nice-rounding artefact) makes its own nice extent start at 0.7, below the min, and its tick
 * code asserts.
 */
function drawableRange(given: Range | undefined, log: boolean): Range | undefined {
  if (!given || !Number.isFinite(given[0]) || !Number.isFinite(given[1])) return undefined
  const r: Range = [clean(given[0]), clean(given[1])]
  if (log && !(r[0] > 0 && r[1] > 0)) return undefined
  if (r[1] > r[0]) return r
  const v = r[0]
  if (log) return [v / 10, v * 10]
  const half = Math.max(Math.abs(v), 1) / 2
  return [v - half, v + half]
}

function axisOption(
  axis: AxisModel,
  given: Range | undefined,
  labels: boolean,
  bare: boolean,
  which: 'x' | 'y',
  pixels: number,
) {
  const o = axis.options
  const r = drawableRange(given, axis.log)
  const formatter = axis.categorical ? (v: number) => axis.label(v) : axis.log ? formatPower : o.format
  return {
    // A log axis ticks every decade, or every few when the decades would crowd (ECharts alone steps by 10 decades
    // once the range spans 10 or more). The interval is in decades.
    ...(axis.log && r && r[0] > 0 ? { interval: logInterval(r, pixels, which) } : {}),
    ...(o.integer && !axis.log && !axis.categorical ? { minInterval: 1 } : {}),
    type: axis.log ? 'log' : 'value',
    inverse: o.inverse,
    name: labels ? o.label : undefined,
    min: r?.[0],
    max: r?.[1],
    scale: true,
    show: !bare,
    // An explicit range stays exact (ECharts 6 would widen it to fit bars).
    containShape: false,
    // Axis lines on the plot's edges, never through zero across the data.
    axisLine: { onZero: false },
    axisLabel: {
      show: labels,
      ...(formatter ? { formatter } : {}),
      ...(axis.categorical && which === 'x' ? { hideOverlap: false } : {}),
    },
    // Categories at integers: ticks every half unit from the axis minimum (−0.5), labelled at integers only.
    ...(axis.categorical ? { interval: 0.5, splitLine: { show: false } } : {}),
  }
}

/** Decades per tick on a log axis over `r`: 1, or more when a label per decade would not fit in `pixels`. */
function logInterval(r: Range, pixels: number, which: 'x' | 'y'): number {
  const decades = Math.log10(r[1] / r[0])
  const fit = Math.max(2, Math.floor(pixels / (which === 'y' ? 20 : 44)))
  return Math.max(1, Math.ceil(decades / fit))
}

/** A layer definition's `needsBox`/`needsPlot`, which may depend on the layer's props. */
function flagFor(flag: boolean | ((props: CommonProps) => boolean) | undefined, props: CommonProps): boolean {
  return typeof flag === 'function' ? flag(props) : !!flag
}
