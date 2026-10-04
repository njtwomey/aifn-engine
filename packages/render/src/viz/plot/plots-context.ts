/** What a Plot learns from the `Plots` grid around it, and what it reports back. Null outside a grid. */
import { createContext, useContext } from 'react'
import type { Margins } from './ticks'
import type { AxisModel } from './axis'

export type CellReport = {
  x: AxisModel
  y: AxisModel
  /** The widest y tick label in pixels, and whether the y axis has a name beside it. */
  labelWidth: number
  yName: boolean
  /** Whether the x axis has a name under it. */
  xName: boolean
  /** Room the layers need right of the plot area (a colour bar). */
  right: number
  /** Room above the plot area (a legend, a title). */
  top: number
  /** Equal units: the base ranges' spans, so a single-column grid can size this panel's height. */
  equal?: { xSpan: number; ySpan: number }
}

export type PlotsCell = {
  row: number
  col: number
  /** Aligned plot margins: shared by the column (left, right) and the row (top, bottom). */
  margins: Margins
  /** Tick labels and names on this panel's axes (false for inner panels of an axis shared down a column or a row). */
  labels: { x: boolean; y: boolean }
  /** The y-axis name gap that clears the column's widest tick labels. */
  yNameGap: number
  hoverGroup?: string
  report: (key: string, report: CellReport | null) => void
}

export const PlotsContext = createContext<PlotsCell | null>(null)

export const usePlotsCell = () => useContext(PlotsContext)
