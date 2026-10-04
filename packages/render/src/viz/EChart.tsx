import { useEffect, useRef } from 'react'
import { chrome, type Mode } from '@render/design/palette'
import { useTheme } from '@render/design/theme'
import { cn } from '@render/lib/utils'
import { echarts, type EChartsOption } from './echarts'
import { formatNumber } from './format'
import { GRAB_RADIUS, type Handle } from './handles'
import { publishHover, subscribeHover } from './hover'
import { baseOption } from './theme'
import type { Range } from './viewport'

export type EChartProps = {
  /** Chart-specific option, merged over the lab theme from `baseOption`. Keep it referentially stable (useMemo). */
  option: EChartsOption
  /** Pixels, 'fill' to take the full height of the parent, or 'auto' to size from CSS. */
  height?: number | 'fill' | 'auto'
  className?: string
  /**
   * Small, frequently changing update merged onto the chart without redrawing the rest, e.g. a moving marker. Series
   * in a patch are matched by `id`, so the base option must declare them with the same `id`.
   */
  patch?: EChartsOption
  /** ECharts click events (`seriesId`, `seriesName`, `value`, ...). */
  onClick?: (event: EChartClick) => void
  /** Clicks anywhere inside the plot area, in data coordinates. Prefer handles when a parameter has a place. */
  onPlotClick?: (point: [number, number]) => void
  /** Accessible description of what the chart shows. */
  ariaLabel?: string
  /** False for charts without x/y axes, so the theme's default axes and grid are left out. */
  cartesian?: boolean
  /**
   * Draggable handles bound to parameters (see handles.ts). With one handle, pressing anywhere on the plot moves it;
   * with several, the nearest within GRAB_RADIUS pixels is grabbed. Axes freeze during a drag and refit on release.
   */
  handles?: Handle[]
  /** SVG (default) keeps lines and text crisp; canvas is much faster for thousands of marks. Fixed at mount. */
  renderer?: 'svg' | 'canvas'
  /** Pointer position over any grid in that grid's data coordinates: move (per frame), click, leave. */
  onPointer?: (event: PlotPointer) => void
  /** The first grid's axis extents after every full redraw, e.g. to seed a zoom toolbar with the fitted ranges. */
  onExtents?: (extents: { x: Range; y: Range }) => void
  /**
   * Pinch or Ctrl/⌘-scroll over the plot: `factor` above 1 zooms out, below 1 in, about `point` (data coordinates of
   * the pointer). Plain scrolling still scrolls the page.
   */
  onWheelZoom?: (factor: number, point: [number, number]) => void
  /** The x value under an axis-triggered pointer (tooltip `trigger: 'axis'`), or null when it leaves. */
  onAxisHover?: (x: number | null) => void
  /** Charts sharing a hover group share the hovered x value (see hover.ts). Needs an axis-triggered tooltip. */
  hoverGroup?: string
  /** Called with true when a handle drag starts and false when it ends, e.g. to freeze axes shared with other charts. */
  onDragChange?: (dragging: boolean) => void
  /**
   * Rectangle brushing on the first grid: pressing and dragging draws a rectangle, and on release its x and y ranges
   * (data coordinates) are reported; a press without a drag reports null (clear). Not for charts with handles. The
   * chart draws the rectangle only while dragging; draw the committed one yourself (e.g. as a live series).
   */
  onBrush?: (rect: { x: Range; y: Range } | null) => void
  /**
   * Parallel coordinates: the brushed intervals on each parallel axis, by axis `id`, re-applied after every full redraw
   * (ECharts forgets them when the option is replaced). Keep in step with `onAxisAreaSelect`.
   */
  axisAreas?: Readonly<Record<string, readonly (readonly [number, number])[]>>
  /** Parallel coordinates: the reader brushed (or cleared) intervals on the axis with this `id`. */
  onAxisAreaSelect?: (axisId: string, intervals: [number, number][]) => void
}

export type PlotPointer =
  { type: 'move' | 'click'; grid: number; point: [number, number] } | { type: 'leave'; grid?: never; point?: never }

export type EChartClick = {
  seriesId?: string
  seriesName?: string
  value?: unknown
  dataIndex: number
  dataType?: string
  data?: unknown
}

type Plain = Record<string, unknown>
const isPlain = (v: unknown): v is Plain => typeof v === 'object' && v !== null && !Array.isArray(v)
const asArray = (v: unknown): unknown[] => (Array.isArray(v) ? v : v ? [v] : [])

/** Deep-merge `over` onto `base`. An array over a plain object merges the object into every element (axis lists). */
function merge(base: unknown, over: unknown): unknown {
  if (isPlain(base) && Array.isArray(over)) return over.map((item) => merge(base, item))
  if (!isPlain(base) || !isPlain(over)) return over === undefined ? base : over
  const out: Plain = { ...base }
  for (const [k, v] of Object.entries(over)) out[k] = merge(base[k], v)
  return out
}

/** The tick step ECharts picks for a value axis of this span (1, 2, 3 or 5 times a power of ten). */
function tickStep(span: number): number {
  const raw = span / 5
  const unit = 10 ** Math.floor(Math.log10(raw))
  const f = raw / unit
  return unit * (f <= 1 ? 1 : f <= 2 ? 2 : f <= 3 ? 3 : f <= 5 ? 5 : 10)
}

const onTick = (v: number, step: number) => Math.abs(v / step - Math.round(v / step)) < 1e-6

/**
 * Tidy every numeric axis: labels are rounded, and a fixed min or max off the tick grid (a zoomed or typed range) keeps
 * its gridline but loses its label, which would otherwise crowd the regular ticks beside it.
 */
function tidyAxes(axes: unknown): unknown {
  if (Array.isArray(axes)) return axes.map(tidyAxes)
  if (!isPlain(axes) || (axes.type !== undefined && axes.type !== 'value')) return axes
  const label: Plain = isPlain(axes.axisLabel) ? { ...axes.axisLabel } : {}
  if (label.formatter === undefined) label.formatter = (v: number) => formatNumber(v)
  const { min, max } = axes
  if (typeof min === 'number' && typeof max === 'number' && max > min) {
    const step = tickStep(max - min)
    if (label.showMinLabel === undefined && !onTick(min, step)) label.showMinLabel = false
    if (label.showMaxLabel === undefined && !onTick(max, step)) label.showMaxLabel = false
  }
  return { ...axes, axisLabel: label }
}

const HANDLES_ID = '__handles'

/** An option ECharts rejected, named by what it held: the bare ECharts errors ("Error" from an assert) say nothing. */
export class ChartError extends Error {
  override name = 'ChartError'
}

const fmt = (v: unknown) => (typeof v === 'number' ? String(v) : v === undefined ? 'auto' : JSON.stringify(v))

/** The axes (name, type, min, max) and series (id, type, points, non-finite values) of an option, in one line each. */
function describeOption(option: unknown): string {
  const o = (isPlain(option) ? option : {}) as Plain
  const axes = (key: 'xAxis' | 'yAxis') =>
    asArray(o[key])
      .filter(isPlain)
      .map((a) => `${key[0]} ${fmt(a.name ?? '')} ${fmt(a.type ?? 'value')} [${fmt(a.min)}, ${fmt(a.max)}]`)
  const series = asArray(o.series)
    .filter(isPlain)
    .map((q) => {
      const data = Array.isArray(q.data) ? (q.data as unknown[]) : []
      const bad = data.filter((d) =>
        (Array.isArray(d) ? d : [d]).some((v) => typeof v === 'number' && !Number.isFinite(v)),
      ).length
      return `${fmt(q.id ?? q.name ?? '?')} ${fmt(q.type ?? '')} ${data.length} points${bad ? `, ${bad} non-finite` : ''}`
    })
  return [...axes('xAxis'), ...axes('yAxis'), ...series].join('; ')
}

/** setOption, rethrowing any ECharts failure as a `ChartError` that names the axes and series it was given. */
function setOptionNamed(instance: echarts.ECharts, option: unknown, opts?: { notMerge?: boolean }) {
  try {
    instance.setOption(option as EChartsOption, opts)
  } catch (e) {
    const cause = e instanceof Error ? e.message || e.name : String(e)
    const err = new ChartError(
      `ECharts rejected the chart (${cause || 'internal assertion'}${e instanceof Error && e.stack?.includes('assert') ? ', in an ECharts assert' : ''}): ${describeOption(option)}`,
      { cause: e },
    )
    console.error(err.message, e)
    throw err
  }
}

/**
 * Dev-only counters on `window.__labStats`, read by `make lab-shots ARGS='--profile'`: `setOption` counts every call
 * into ECharts, `full` the full redraws among them (`notMerge`, a new base option). Production builds drop it.
 */
function countSetOption(full: boolean) {
  if (!import.meta.env.DEV) return
  const w = window as unknown as { __labStats?: { setOption: number; full: number } }
  const stats = (w.__labStats ??= { setOption: 0, full: 0 })
  stats.setOption++
  if (full) stats.full++
}

type Extents = { x: Range; y: Range }

/** Everything the chart is drawn from, in a ref so pointer handlers and effects share one source. */
type Inputs = {
  option: EChartsOption
  patch?: EChartsOption
  handles?: Handle[]
  mode: Mode
  cartesian: boolean
  axisAreas?: EChartProps['axisAreas']
}

type Callbacks = Pick<
  EChartProps,
  | 'onClick'
  | 'onPlotClick'
  | 'onPointer'
  | 'onExtents'
  | 'onWheelZoom'
  | 'onAxisHover'
  | 'hoverGroup'
  | 'onDragChange'
  | 'onBrush'
  | 'onAxisAreaSelect'
>

/** The only component that touches ECharts. Chart components build an option and render this. */
export function EChart({
  option,
  patch,
  height = 320,
  className,
  ariaLabel,
  cartesian = true,
  handles,
  renderer = 'svg',
  axisAreas,
  ...callbacks
}: EChartProps) {
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  const { resolved } = useTheme()
  const inputs = useRef<Inputs>({ option, patch, handles, mode: resolved, cartesian, axisAreas })
  // Set while the chart re-applies `axisAreas` itself, so those selections are not reported back as the reader's.
  const restoring = useRef(false)
  const cb = useRef<Callbacks>(callbacks)
  const frozen = useRef<Extents | null>(null)
  const rendererRef = useRef(renderer)
  const link = useRef<Link>({ source: Symbol('chart'), linking: false })
  const { hoverGroup } = callbacks

  useEffect(() => {
    cb.current = callbacks
    inputs.current = { option, patch, handles, mode: resolved, cartesian, axisAreas }
  })

  /**
   * `data-chart-ready` is set when ECharts reports a frame with no animation left ('finished') and cleared before
   * every setOption; `make lab-shots` waits for it on every chart before capturing. ECharts emits 'finished' only after
   * a frame it actually paints, so a setOption that changes nothing visible (a patch that removes a series, one whose
   * data is unchanged) would leave the chart unready for good. A watcher therefore also marks the chart ready once two
   * frames have passed (ECharts flushes a pending update in the next frame) and zrender has no animation running.
   */
  const watch = useRef(0)
  const unready = () => {
    const el = ref.current
    const instance = chart.current
    if (!el || !instance) return
    el.removeAttribute('data-chart-ready')
    cancelAnimationFrame(watch.current)
    let frames = 0
    const tick = () => {
      if (chart.current !== instance || instance.isDisposed() || el.hasAttribute('data-chart-ready')) return
      if (++frames >= 2 && instance.getZr().animation.isFinished()) el.setAttribute('data-chart-ready', '')
      else watch.current = requestAnimationFrame(tick)
    }
    watch.current = requestAnimationFrame(tick)
  }

  /** Draw from the latest inputs: `full` replaces the base option; otherwise only the patch and handles merge in. */
  const render = useRef((full: boolean) => {
    const instance = chart.current
    if (!instance) return
    const { option, patch, handles, mode, cartesian, axisAreas } = inputs.current
    if (full) {
      const base: Plain = { ...baseOption(mode) }
      if (!cartesian) {
        delete base.xAxis
        delete base.yAxis
        delete base.grid
      }
      const merged = merge(base, option) as Plain
      if (frozen.current && cartesian) {
        merged.xAxis = freeze(merged.xAxis, frozen.current.x)
        merged.yAxis = freeze(merged.yAxis, frozen.current.y)
        merged.tooltip = { ...(isPlain(merged.tooltip) ? merged.tooltip : {}), show: false }
      }
      if (cartesian) {
        merged.xAxis = tidyAxes(merged.xAxis)
        merged.yAxis = tidyAxes(merged.yAxis)
      }
      if (handles && cartesian) merged.series = [...asArray(merged.series), handlesSeries(mode)]
      unready()
      setOptionNamed(instance, merged, { notMerge: true })
      countSetOption(true)
      if (cartesian && !frozen.current) {
        const extents = axisExtents(instance)
        if (extents) cb.current.onExtents?.(extents)
      }
      if (axisAreas) {
        restoring.current = true
        try {
          for (const [id, intervals] of Object.entries(axisAreas))
            if (intervals.length)
              instance.dispatchAction({
                type: 'axisAreaSelect',
                parallelAxisId: id,
                intervals: intervals.map((v) => [...v]),
              })
        } finally {
          restoring.current = false
        }
      }
    }
    // The patch and the handles go in one call: each setOption runs ECharts' whole update, every series included.
    const withHandles = handles && cartesian
    if (patch || withHandles) {
      const p = { ...(patch ?? {}) } as Plain
      if (withHandles) p.series = [...asArray(p.series), handlesPatch(handles)]
      unready()
      setOptionNamed(instance, p)
      countSetOption(false)
    }
  })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const instance = echarts.init(el, undefined, { renderer: rendererRef.current })
    chart.current = instance
    instance.on('finished', () => el.setAttribute('data-chart-ready', ''))
    instance.on('click', (e) => cb.current.onClick?.(e as unknown as EChartClick))
    instance.on('axisareaselected', (e) => {
      if (restoring.current) return
      const { parallelAxisId, intervals } = e as { parallelAxisId?: string; intervals?: [number, number][] }
      if (parallelAxisId) cb.current.onAxisAreaSelect?.(parallelAxisId, intervals ?? [])
    })
    instance.getZr().on('click', (e) => {
      const pixel = [e.offsetX, e.offsetY]
      if (!cb.current.onPlotClick || !instance.containPixel({ gridIndex: 0 }, pixel)) return
      const [x, y] = instance.convertFromPixel({ gridIndex: 0 }, pixel) as number[]
      cb.current.onPlotClick([x, y])
    })
    // Registered before the handle listeners, so a press starts ungrabbed and `start` below marks it grabbed.
    const pointer = attachPointer(instance, () => inputs.current.option, cb)
    const detachHandles = attachHandles(instance, () => inputs.current.handles, {
      start: () => {
        cb.current.onDragChange?.(true)
        pointer.grab()
        frozen.current = axisExtents(instance)
        // Handles win over hover: no tooltip or axis pointer while dragging (the full redraw on release restores them).
        instance.dispatchAction({ type: 'hideTip' })
        instance.setOption({ tooltip: { show: false } })
        countSetOption(false)
      },
      end: () => {
        cb.current.onDragChange?.(false)
        pointer.release()
        if (!frozen.current) return
        frozen.current = null
        render.current(true)
      },
    })
    const detachWheel = attachWheelZoom(instance, cb)
    const detachBrush = attachBrush(instance, cb, () => chrome(inputs.current.mode))
    const detachHover = attachAxisHover(instance, cb, link.current)
    render.current(true)
    let size = ''
    const observer = new ResizeObserver(([entry]) => {
      const next = `${Math.round(entry.contentRect.width)}x${Math.round(entry.contentRect.height)}`
      if (next === size) return
      size = next
      instance.resize({ animation: { duration: 0 } })
    })
    observer.observe(el)
    return () => {
      cancelAnimationFrame(watch.current)
      observer.disconnect()
      pointer.detach()
      detachHandles()
      detachWheel()
      detachBrush()
      detachHover()
      instance.dispose()
      chart.current = null
    }
  }, [])

  useEffect(() => {
    if (!hoverGroup) return
    const state = link.current
    return subscribeHover(hoverGroup, (x, from) => {
      if (from === state.source || !chart.current) return
      state.linking = true
      try {
        showAt(chart.current, x)
      } finally {
        state.linking = false
      }
    })
  }, [hoverGroup])

  useEffect(() => render.current(true), [option, resolved, cartesian])
  useEffect(() => render.current(false), [patch, handles])

  return (
    <div
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      className={cn(
        'w-full',
        height === 'fill' && 'h-full',
        height === 'auto' && 'h-auto',
        callbacks.onPlotClick && 'cursor-pointer',
        callbacks.onBrush && 'cursor-crosshair touch-none select-none',
        handles && 'touch-none select-none',
        className,
      )}
      style={height === 'fill' || height === 'auto' ? undefined : { height }}
    />
  )
}

/** Fix an axis to `[min, max]`. With several axes (grids) only the first, which handles use, is frozen. */
function freeze(axis: unknown, [min, max]: Range): unknown {
  if (Array.isArray(axis)) return axis.map((a, i) => (i === 0 ? merge(a, { min, max }) : a))
  return merge(axis, { min, max })
}

const localPixel = (el: HTMLElement, e: MouseEvent) => {
  const rect = el.getBoundingClientRect()
  return [e.clientX - rect.left, e.clientY - rect.top]
}

/** Ctrl/⌘-wheel and trackpad pinch (which arrives as a wheel event with ctrlKey) over the plot zoom about the pointer. */
function attachWheelZoom(instance: echarts.ECharts, cb: { current: Callbacks }) {
  const el = instance.getDom()
  const wheel = (e: WheelEvent) => {
    if (!cb.current.onWheelZoom || !(e.ctrlKey || e.metaKey)) return
    const pixel = localPixel(el, e)
    if (!instance.containPixel({ gridIndex: 0 }, pixel)) return
    // A native, non-passive listener: React's onWheel is passive and cannot stop the browser's page zoom.
    e.preventDefault()
    const [x, y] = instance.convertFromPixel({ gridIndex: 0 }, pixel) as number[]
    const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY
    cb.current.onWheelZoom(Math.exp(Math.max(-1, Math.min(1, delta * 0.01))), [x, y])
  }
  el.addEventListener('wheel', wheel, { passive: false })
  return () => el.removeEventListener('wheel', wheel)
}

/**
 * Rectangle brushing (`onBrush`): press in the first grid and drag. A dashed rectangle follows the pointer (a DOM
 * overlay, so ECharts redraws nothing while dragging); on release its corners are converted to data coordinates, clipped
 * to the axes. A press that barely moves reports null, which clears the selection.
 */
function attachBrush(instance: echarts.ECharts, cb: { current: Callbacks }, colours: () => { ink: string }) {
  const el = instance.getDom()
  let start: number[] | null = null
  let box: HTMLDivElement | null = null
  const place = (a: number[], b: number[]) => {
    if (!box) return
    Object.assign(box.style, {
      left: `${Math.min(a[0], b[0])}px`,
      top: `${Math.min(a[1], b[1])}px`,
      width: `${Math.abs(a[0] - b[0])}px`,
      height: `${Math.abs(a[1] - b[1])}px`,
    })
  }
  const down = (e: PointerEvent) => {
    if (!cb.current.onBrush || e.button !== 0) return
    const p = localPixel(el, e)
    if (!instance.containPixel({ gridIndex: 0 }, p)) return
    e.preventDefault()
    start = p
    try {
      el.setPointerCapture(e.pointerId)
    } catch {
      // Synthetic events have no pointer to capture.
    }
    box = document.createElement('div')
    const ink = colours().ink
    Object.assign(box.style, {
      position: 'absolute',
      pointerEvents: 'none',
      border: `1px dashed ${ink}`,
      background: `color-mix(in srgb, ${ink} 8%, transparent)`,
      zIndex: '5',
    })
    el.appendChild(box)
    place(p, p)
  }
  const move = (e: PointerEvent) => {
    if (start) place(start, localPixel(el, e))
  }
  const up = (e: PointerEvent) => {
    if (!start) return
    const a = start
    const b = localPixel(el, e)
    start = null
    box?.remove()
    box = null
    if (el.hasPointerCapture?.(e.pointerId)) el.releasePointerCapture(e.pointerId)
    if (Math.abs(a[0] - b[0]) < 4 && Math.abs(a[1] - b[1]) < 4) return cb.current.onBrush?.(null)
    const extents = axisExtents(instance)
    const [x0, y0] = instance.convertFromPixel({ gridIndex: 0 }, a) as number[]
    const [x1, y1] = instance.convertFromPixel({ gridIndex: 0 }, b) as number[]
    const clip = (lo: number, hi: number, r?: Range): Range =>
      r ? [Math.max(Math.min(lo, hi), r[0]), Math.min(Math.max(lo, hi), r[1])] : [Math.min(lo, hi), Math.max(lo, hi)]
    cb.current.onBrush?.({ x: clip(x0, x1, extents?.x), y: clip(y0, y1, extents?.y) })
  }
  const listeners = { pointerdown: down, pointermove: move, pointerup: up, pointercancel: up }
  for (const [type, fn] of Object.entries(listeners)) el.addEventListener(type, fn as EventListener)
  return () => {
    box?.remove()
    for (const [type, fn] of Object.entries(listeners)) el.removeEventListener(type, fn as EventListener)
  }
}

type Link = { source: symbol; linking: boolean }

/**
 * Axis hover: report the x under the axis pointer and publish it to the chart's hover group. Values that arrive from the
 * group (see the effect in EChart) are shown with `link.linking` set, so they are not published back.
 */
function attachAxisHover(instance: echarts.ECharts, cb: { current: Callbacks }, link: Link) {
  let last: number | null = null
  instance.on('updateAxisPointer', (event) => {
    const info = (event as { axesInfo?: { axisDim: string; value: number }[] }).axesInfo ?? []
    const x = info.find((a) => a.axisDim === 'x')?.value ?? null
    if (x === last) return
    last = x
    cb.current.onAxisHover?.(x)
    const group = cb.current.hoverGroup
    if (group && !link.linking) publishHover(group, x, link.source)
  })
  return () => instance.off('updateAxisPointer')
}

/** Show the axis tooltip at data x (or hide it), with the pointer at the middle of the plot's height. */
function showAt(instance: echarts.ECharts, x: number | null) {
  const extents = axisExtents(instance)
  if (x === null || !extents || x < extents.x[0] || x > extents.x[1]) {
    instance.dispatchAction({ type: 'hideTip' })
    instance.dispatchAction({ type: 'updateAxisPointer', currTrigger: 'leave' })
    return
  }
  const px = instance.convertToPixel({ xAxisIndex: 0 }, x) as number
  const [top, bottom] = [
    instance.convertToPixel({ yAxisIndex: 0 }, extents.y[1]) as number,
    instance.convertToPixel({ yAxisIndex: 0 }, extents.y[0]) as number,
  ]
  instance.dispatchAction({ type: 'showTip', x: px, y: (top + bottom) / 2 })
}

/** Hover and click positions over every grid, for `onPointer`. Moves are batched per frame and paused during drags. */
function attachPointer(instance: echarts.ECharts, option: () => EChartsOption, cb: { current: Callbacks }) {
  const el = instance.getDom()
  let grabbed = false
  let dragging = false
  let frame = 0
  let pending: number[] | null = null

  const locate = (pixel: number[]): { type: 'move'; grid: number; point: [number, number] } | null => {
    const grids = Math.max(asArray((option() as Plain).grid).length, 1)
    for (let grid = 0; grid < grids; grid++) {
      if (!instance.containPixel({ gridIndex: grid }, pixel)) continue
      const [x, y] = instance.convertFromPixel({ gridIndex: grid }, pixel) as number[]
      return { type: 'move', grid, point: [x, y] }
    }
    return null
  }
  const down = () => {
    grabbed = false
  }
  const move = (e: PointerEvent) => {
    if (!cb.current.onPointer || dragging) return
    pending = localPixel(el, e)
    if (!frame)
      frame = requestAnimationFrame(() => {
        frame = 0
        const hit = pending && locate(pending)
        pending = null
        cb.current.onPointer?.(hit ?? { type: 'leave' })
      })
  }
  const leave = (e: PointerEvent) => {
    // A touch lifts off after every tap; keep what the tap showed rather than clearing it at once.
    if (!cb.current.onPointer || dragging || e.pointerType === 'touch') return
    cancelAnimationFrame(frame)
    frame = 0
    pending = null
    cb.current.onPointer({ type: 'leave' })
  }
  const click = (e: MouseEvent) => {
    if (!cb.current.onPointer || grabbed) return
    const hit = locate(localPixel(el, e))
    if (hit) cb.current.onPointer({ ...hit, type: 'click' })
  }
  el.addEventListener('pointerdown', down)
  el.addEventListener('pointermove', move)
  el.addEventListener('pointerleave', leave)
  el.addEventListener('click', click)
  return {
    grab: () => {
      grabbed = true
      dragging = true
    },
    release: () => {
      dragging = false
    },
    detach: () => {
      cancelAnimationFrame(frame)
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerleave', leave)
      el.removeEventListener('click', click)
    },
  }
}

/** The series that draws every handle: points as ink markers, x and y handles as dashed ink guide lines. */
function handlesSeries(mode: Mode) {
  const c = chrome(mode)
  return {
    id: HANDLES_ID,
    type: 'scatter',
    data: [],
    silent: true,
    symbol: 'circle',
    symbolSize: 14,
    itemStyle: { color: c.ink, borderColor: c.surface, borderWidth: 2 },
    tooltip: { show: false },
    z: 20,
    markLine: {
      silent: true,
      symbol: ['none', 'none'],
      animation: false,
      lineStyle: { color: c.ink, width: 1.5, type: 'dashed' },
      label: { color: c.ink, fontSize: 11, position: 'end' },
      data: [],
    },
  }
}

function handlesPatch(handles: Handle[]) {
  // An x guide's label above its top end (the Plot keeps a row for it); a y guide's inside the plot above its right
  // end, where the right margin cannot clip it.
  return {
    id: HANDLES_ID,
    data: handles.flatMap((h) =>
      h.kind === 'point'
        ? [
            h.symbol || h.color
              ? {
                  value: h.at,
                  ...(h.symbol ? { symbol: h.symbol } : {}),
                  ...(h.color ? { itemStyle: { color: h.color } } : {}),
                }
              : h.at,
          ]
        : [],
    ),
    markLine: {
      data: handles.flatMap((h) =>
        h.kind === 'point'
          ? []
          : [
              {
                [h.kind === 'x' ? 'xAxis' : 'yAxis']: h.at,
                label: { formatter: h.label ?? '', position: h.kind === 'x' ? 'end' : 'insideEndTop' },
                ...(h.color ? { lineStyle: { color: h.color } } : {}),
              },
            ],
      ),
    },
  }
}

type AxisModel = { axis?: { scale: { getExtent: () => [number, number] } } }

/** Current extents of the first grid's axes, as the axis scales computed them (so nice ticks survive a freeze). */
function axisExtents(instance: echarts.ECharts): Extents | null {
  const model = (
    instance as unknown as { getModel: () => { getComponent: (type: string, i: number) => unknown } | undefined }
  ).getModel()
  if (!model) return null
  const x = (model.getComponent('xAxis', 0) as AxisModel | undefined)?.axis?.scale.getExtent()
  const y = (model.getComponent('yAxis', 0) as AxisModel | undefined)?.axis?.scale.getExtent()
  return x && y ? { x: [x[0], x[1]], y: [y[0], y[1]] } : null
}

/**
 * Pointer logic for handles: grab on press (the nearest handle, or the only one), follow the pointer while pressed,
 * release on pointer-up anywhere. Moves are batched per frame; the final position is always applied. The active
 * handle is tracked by index and its setter read from the latest handles, since widgets rebuild them every render.
 */
function attachHandles(
  instance: echarts.ECharts,
  current: () => Handle[] | undefined,
  hooks: { start: () => void; end: () => void },
): () => void {
  const zr = instance.getZr()
  let active: number | null = null
  let frame = 0
  let pending: number[] | null = null

  const toData = (pixel: number[]) => instance.convertFromPixel({ gridIndex: 0 }, pixel) as number[]
  const toPixel = (point: number[]) => instance.convertToPixel({ gridIndex: 0 }, point) as number[]
  const inside = (pixel: number[]) => instance.containPixel({ gridIndex: 0 }, pixel)

  const distance = (h: Handle, pixel: number[]) => {
    const here = toData(pixel)
    if (h.kind === 'point') {
      const [x, y] = toPixel(h.at)
      return Math.hypot(x - pixel[0], y - pixel[1])
    }
    if (h.kind === 'x') return Math.abs(toPixel([h.at, here[1]])[0] - pixel[0])
    return Math.abs(toPixel([here[0], h.at])[1] - pixel[1])
  }
  const nearest = (pixel: number[]): number | null => {
    const list = current()
    if (!list?.length || !inside(pixel)) return null
    if (list.length === 1) return 0
    let best: number | null = null
    let bestDistance = Infinity
    list.forEach((h, i) => {
      const d = distance(h, pixel)
      if (d < bestDistance) [best, bestDistance] = [i, d]
    })
    return bestDistance <= GRAB_RADIUS ? best : null
  }
  const apply = (pixel: number[]) => {
    const h = active === null ? undefined : current()?.[active]
    if (!h) return
    const [x, y] = toData(pixel)
    if (h.kind === 'point') h.onDrag([x, y])
    else if (h.kind === 'x') h.onDrag(x)
    else h.onDrag(y)
  }
  const cursorFor = (i: number | null) => {
    const h = i === null ? undefined : current()?.[i]
    return !h ? 'default' : h.kind === 'x' ? 'col-resize' : h.kind === 'y' ? 'row-resize' : 'grab'
  }

  // DOM pointer events with pointer capture: the drag keeps tracking outside the chart, and works for touch.
  const el = instance.getDom()
  const down = (e: PointerEvent) => {
    if (e.button !== 0) return
    active = nearest(localPixel(el, e))
    if (active === null) return
    e.preventDefault()
    try {
      el.setPointerCapture(e.pointerId)
    } catch {
      // Synthetic events have no active pointer to capture; the drag still works while the pointer stays inside.
    }
    hooks.start()
    zr.setCursorStyle('grabbing')
    apply(localPixel(el, e))
  }
  const move = (e: PointerEvent) => {
    if (active === null) {
      if (current()?.length) zr.setCursorStyle(cursorFor(nearest(localPixel(el, e))))
      return
    }
    pending = localPixel(el, e)
    if (!frame)
      frame = requestAnimationFrame(() => {
        frame = 0
        if (pending) apply(pending)
        pending = null
      })
  }
  const release = (e: PointerEvent) => {
    if (active === null) return
    cancelAnimationFrame(frame)
    frame = 0
    pending = null
    apply(localPixel(el, e))
    const released = current()?.[active]
    active = null
    released?.onRelease?.()
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId)
    hooks.end()
  }
  const listeners = { pointerdown: down, pointermove: move, pointerup: release, pointercancel: release }
  for (const [type, fn] of Object.entries(listeners)) el.addEventListener(type, fn as EventListener)
  return () => {
    cancelAnimationFrame(frame)
    for (const [type, fn] of Object.entries(listeners)) el.removeEventListener(type, fn as EventListener)
  }
}
