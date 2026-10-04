/**
 * A data-driven list of series for charts whose series are built by code (one per k, conditional extras, a helper
 * shared by several charts). Each spec picks its layer by `type`; `seriesLayers(list)` turns the list into layer
 * elements for a `Plot`. A fixed set of series reads better written out as layer elements.
 */
import { createElement, type ReactElement } from 'react'
import { Area, Bars, Curve, Points } from './layers/marks'

type Values = ArrayLike<number>

export type SeriesSpec = {
  name: string
  /** 'line' → `Curve` (`Area` with `area`), 'scatter' → `Points`, 'bar' → `Bars`. */
  type: 'line' | 'scatter' | 'bar'
  x: Values
  y: Values
  /** Fill to zero under a line. */
  area?: boolean
  /** Points: a class per point (class k takes slot and marker shape k), and the classes' legend names. */
  group?: Values | null
  groupNames?: readonly string[]
  /** Points and bars: an explicit colour per point. */
  colors?: readonly string[]
  slot?: number
  color?: string
  emphasis?: boolean
  muted?: boolean
  /** Lines: dashed, thin (one of many draws), a marker at every vertex. */
  dashed?: boolean
  thin?: boolean
  showPoints?: boolean
  /** Patched without redrawing the rest, and never moves the axes (a path that follows a drag). */
  live?: boolean
}

/** Layer elements for a series list, in order; `live` applies to every series that does not set its own. */
export function seriesLayers(series: readonly SeriesSpec[], options: { live?: boolean } = {}): ReactElement[] {
  return series.map((s, i) => {
    const common = {
      key: `${i}:${s.name}`,
      name: s.name,
      x: s.x,
      y: s.y,
      slot: s.slot,
      color: s.color,
      emphasis: s.emphasis,
      muted: s.muted,
      live: s.live ?? options.live,
    }
    if (s.type === 'scatter')
      return createElement(Points, { ...common, group: s.group, groupNames: s.groupNames, colors: s.colors })
    if (s.type === 'bar') return createElement(Bars, { ...common, colors: s.colors })
    if (s.area) return createElement(Area, common)
    return createElement(Curve, { ...common, dashed: s.dashed, thin: s.thin, showPoints: s.showPoints })
  })
}
