/**
 * The layer protocol. A layer is a small component with one job (a curve, a histogram, a raster) that a `Plot` reads
 * rather than renders: the component itself returns null, and the Plot calls its definition with the element's props.
 * Reading layers synchronously (no registration effects) keeps a drag to one React render and lets the Plot memoise
 * each layer by its props, so ECharts is sent only the layers that changed.
 *
 * A definition declares the layer's data extent (which feeds the axes' fit, unless the layer is `live`) and builds its
 * ECharts series. A `live` layer is sent as a patch (merged by series id), so moving it redraws nothing else and never
 * moves the axes.
 */
import { Children, Fragment, isValidElement, type ReactElement, type ReactNode } from 'react'
import { chrome, seriesColor, type Mode } from '@render/design/palette'
import type { HoverInfo } from '../frame'
import type { Handle } from '../handles'
import type { Range } from '../viewport'
import type { AxisModel } from './axis'

/** Props every layer takes. */
export type CommonProps = {
  /** The legend and tooltip name. Layers sharing a name share one legend entry. */
  name?: string
  /** Fixed categorical slot. Defaults to the layer's position among the Plot's slotted layers. */
  slot?: number
  /** Ink, not a category (e.g. the true curve, a centroid). Takes no slot. */
  emphasis?: boolean
  /** The muted chrome colour, for background marks. Takes no slot. */
  muted?: boolean
  /** A data colour from a scale helper in place of the slot. */
  color?: string
  /** An outcome colour in place of the slot: the theme's destructive red (a failure) or its success green. */
  tone?: 'destructive' | 'success'
  /** Sent as a patch on every change without redrawing the rest; never widens the axes. */
  live?: boolean
  /** A fixed series id prefix, so the layer keeps its identity when layers before it come and go. */
  id?: string
  /**
   * The layer shows an answer older than the current inputs while a newer one is computed (`useComputed`'s `stale`):
   * it is drawn faded until the new answer lands.
   */
  stale?: boolean
}

/** Along which axis a one-dimensional layer's values run: `x` (default) or `y` (rotated, e.g. an output density). */
export type Orient = 'x' | 'y'

/** The plot's visible box in data coordinates. */
export type PlotBox = { x: Range; y: Range; xLog: boolean; yLog: boolean }

export type LayerContext = {
  mode: Mode
  /** The layer's resolved colour: its `color`, ink for `emphasis`, muted for `muted`, else its slot's colour. */
  color: string
  slot: number
  /** A prefix for the layer's series ids, stable across renders. */
  id: string
  /** The drawn ranges (only for layers that declare `needsBox`). */
  box?: PlotBox
  /** The plot area in CSS pixels (only for layers that declare `needsPlot`). */
  plot?: { width: number; height: number }
  /** The resolved range of an axis the layer declared in `axes` (e.g. a raster's colour axis). */
  range: (axis: AxisModel, extent?: Range) => Range | undefined
  /** The Plot's axes. */
  x: AxisModel
  y: AxisModel
}

/** One line-like series the axis tooltip and the hover readout report at the hovered x. */
export type HoverSeries = { label: string; color: string; x: ArrayLike<number>; y: ArrayLike<number> }

/** Where the plot area sits in the chart box, for overlays drawn as DOM (a colour bar, a net-area label). */
export type Geometry = { left: number; top: number; width: number; height: number; boxWidth: number }

type Series = Record<string, unknown>

export type LayerOutput = {
  series: Series[]
  handles?: Handle[]
  /** Line-like series for the axis-triggered tooltip and the hover readout. */
  hover?: HoverSeries[]
  /** Item tooltips, by series id. */
  tooltip?: Record<string, (p: { seriesName: string; value: unknown; marker: string; dataIndex: number }) => string>
  /** The hover readout for a pointer at a data position (a raster's cell). */
  pointer?: (point: [number, number]) => HoverInfo | null
  /** DOM drawn over the chart box, placed from the plot area's geometry. */
  overlay?: (g: Geometry) => ReactNode
  /** Extents on axes other than the Plot's x and y (a raster's values on its colour axis). */
  extents?: { axis: AxisModel; range: Range }[]
  /** What the Figure's copy button exports for this layer. */
  data?: unknown
}

export type LayerDef<P> = {
  kind: string
  /** Whether the layer takes a default palette slot (default: unless `muted` or `emphasis`). */
  slotted?: (props: P) => boolean
  /**
   * The data extent on each axis; absent leaves that axis to other layers. `tight` extents (a raster's grid) end the
   * axis exactly where they end, rather than at the next whole tick.
   */
  extent?: (props: P) => { x?: Range; y?: Range; tight?: boolean } | undefined
  build: (props: P, ctx: LayerContext) => LayerOutput
  /** Axes other than the Plot's x and y that the layer reads (a raster's colour axis); the Plot subscribes to them. */
  axes?: (props: P) => (AxisModel | undefined)[]
  /**
   * Room the layer needs beside the plot area, known before the build (`box` and `plot` are unset): `right` and `top`
   * in the margin right of the plot (a colour bar and its name), `labelRow` for a label drawn just above the plot area
   * (an x handle's), which the Plot stacks under the legend and title.
   */
  margins?: (props: P, ctx: LayerContext) => { right?: number; top?: number; labelRow?: number }
  /** The names the layer puts in the legend (default its `name`, if any). */
  legend?: (props: P) => string[]
  /** The layer is rebuilt when the drawn ranges change (clipping, pixel-anchored marks); may depend on its props. */
  needsBox?: boolean | ((props: P) => boolean)
  /** The layer is rebuilt when the plot area's pixel size changes (a raster's resolution); may depend on its props. */
  needsPlot?: boolean | ((props: P) => boolean)
  /** The layer covers the plot area (a raster): the Plot draws no grid lines across it. */
  covers?: boolean
  /** Live by default (sent as a patch, never widening the axes), e.g. a probe the reader moves. */
  live?: boolean
  /** Thousands of marks: the Plot picks the canvas renderer. */
  canvas?: (props: P) => boolean
}

export type LayerComponent<P> = ((props: P) => null) & { layer: LayerDef<P> }

/** A layer component from its definition. The component renders nothing; the enclosing Plot reads its props. */
export function defineLayer<P extends CommonProps>(def: LayerDef<P>): LayerComponent<P> {
  const component = (() => null) as unknown as LayerComponent<P>
  component.layer = def
  Object.defineProperty(component, 'name', { value: def.kind })
  return component
}

export type LayerElement = { def: LayerDef<CommonProps>; props: CommonProps; key: string }

/** The layer elements among a Plot's children, through fragments and arrays, in order. Other children are ignored. */
export function collectLayers(children: ReactNode): LayerElement[] {
  const out: LayerElement[] = []
  const walk = (node: ReactNode) => {
    Children.forEach(node, (child) => {
      if (!isValidElement(child)) return
      const element = child as ReactElement<{ children?: ReactNode }>
      if (element.type === Fragment) return walk(element.props.children)
      const def = (element.type as Partial<LayerComponent<CommonProps>>).layer
      if (def) out.push({ def, props: element.props as CommonProps, key: String(out.length) })
    })
  }
  walk(children)
  return out
}

/** A layer's colour from its props and slot. */
export function layerColor(props: CommonProps, slot: number, mode: Mode): string {
  const c = chrome(mode)
  if (props.color) return props.color
  if (props.tone) return c[props.tone]
  if (props.emphasis) return c.ink
  if (props.muted) return c.muted
  return seriesColor(mode, slot)
}

/** Values and heights as [x, y] points, swapped for `orient="y"`. */
export function oriented(orient: Orient | undefined, v: number, h: number): [number, number] {
  return orient === 'y' ? [h, v] : [v, h]
}

/** The finite extent of some arrays, or undefined. */
export function extentOf(...arrays: (ArrayLike<number> | undefined)[]): Range | undefined {
  let lo = Infinity
  let hi = -Infinity
  for (const a of arrays) {
    if (!a) continue
    for (let i = 0; i < a.length; i++) {
      const v = a[i]
      if (!Number.isFinite(v)) continue
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined
}

/** `r` widened to include zero, for bars and areas measured from zero. */
export const withZero = (r: Range | undefined): Range | undefined => r && [Math.min(0, r[0]), Math.max(0, r[1])]

/** An extent along the value axis and the height axis, placed by orientation. */
export function orientedExtent(
  orient: Orient | undefined,
  values: Range | undefined,
  heights: Range | undefined,
): { x?: Range; y?: Range } {
  return orient === 'y' ? { x: heights, y: values } : { x: values, y: heights }
}

/** Text safe inside tooltip HTML. */
export const escapeHtml = (s: string) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!)
