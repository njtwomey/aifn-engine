/** Tick arithmetic and margins shared by `Plot` and `Plots`: nice ranges, the width of y tick labels, label gaps. */
import { formatNumber, formatPower } from '../format'
import type { Range } from '../viewport'

/** Pixels around a plot area: left and right shared by a grid column, top and bottom by a grid row. */
export type Margins = { left: number; right: number; top: number; bottom: number }

/** The tick step ECharts picks for a span (1, 2, 3 or 5 times a power of ten). */
function tickStep(span: number): number {
  const raw = span / 5
  const unit = 10 ** Math.floor(Math.log10(raw))
  const f = raw / unit
  return unit * (f <= 1 ? 1 : f <= 2 ? 2 : f <= 3 ? 3 : f <= 5 ? 5 : 10)
}

/** `r` widened outward to whole ticks (whole decades on a log axis), as ECharts' own fit would. */
export function niceRange([lo, hi]: Range, log = false): Range {
  if (log && lo > 0) return [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi))]
  if (!(hi > lo)) return [lo - 1, hi + 1]
  const step = tickStep(hi - lo)
  return [Math.floor(lo / step + 1e-9) * step, Math.ceil(hi / step - 1e-9) * step]
}

const FONT = "11px 'Geist Variable', system-ui, sans-serif"
let context: CanvasRenderingContext2D | null | undefined

/** The pixel width of the widest tick label a y axis over `range` shows, measured in the axis font. */
export function tickLabelWidth(range: Range | undefined, log = false): number {
  if (!range || !(range[1] > range[0])) return 24
  const labels: string[] = []
  if (log && range[0] > 0) {
    for (let e = Math.ceil(Math.log10(range[0])); e <= Math.floor(Math.log10(range[1])); e++)
      labels.push(formatPower(10 ** e))
  } else {
    const step = tickStep(range[1] - range[0])
    for (let v = Math.ceil(range[0] / step) * step; v <= range[1] + step * 1e-9; v += step) labels.push(formatNumber(v))
  }
  if (context === undefined)
    context = typeof document === 'undefined' ? null : document.createElement('canvas').getContext('2d')
  if (!context) return Math.max(24, ...labels.map((l) => l.length * 6.5))
  context.font = FONT
  return Math.ceil(Math.max(24, ...labels.map((l) => context!.measureText(l).width)))
}

/** Room for the rotated y-axis name, beside the tick labels. */
export const Y_NAME_ROOM = 22
/** Gap between the tick labels and the axis line. */
export const LABEL_GAP = 8
