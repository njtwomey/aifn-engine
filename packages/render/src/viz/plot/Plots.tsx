import { Children, useCallback, useContext, useId, useMemo, useState, type ReactNode } from 'react'
import { cn } from '../../lib/utils'
import { FrameContext, useChartHeight, useElementSize } from '../frame'
import { LABEL_GAP, Y_NAME_ROOM, type Margins } from './ticks'
import { equalRowWidths, layoutColumn, layoutEqualColumn } from './grid-layout'
import { GRID } from '../theme'
import type { AxisModel } from './axis'
import { AxisToolbar, type ToolbarAxis } from './AxisToolbar'
import { PlotsContext, type CellReport, type PlotsCell } from './plots-context'

export type PlotsProps = {
  rows?: number
  cols?: number
  /** Relative row heights, e.g. [2, 1]; the rows split the frame's height. */
  heights?: readonly number[]
  /** Relative column widths. */
  widths?: readonly number[]
  /** Link hover across every panel: true, or the name of a hover group shared with plots elsewhere. */
  hoverGroup?: boolean | string
  /** Total height in pixels outside a Figure; inside one the frame sets it. */
  height?: number
  /** A share of that height, e.g. 0.6 for a grid that sits beside other content in the same Figure. */
  scale?: number
  /** The grid's axis toolbar: one button per axis (each shared axis once) and one auto-scale (default false). */
  toolbar?: boolean
  /**
   * With an equal-units panel in a single column: `frame` (default) splits the rest of the frame among the other rows
   * by `heights`; `equal` makes each other row's plot `heights[i] / heights[equal row]` times the equal panel's plot.
   */
  ratiosOf?: 'frame' | 'equal'
  /** Panels sharing x sit nearly edge to edge, for a strip that belongs to the panel above (bin counts). */
  tight?: boolean
  /** One `Plot` per cell, in row-major order. */
  children: ReactNode
  className?: string
}

const GAP = 8
const TOOLBAR = 28
/** The least plot-area height of a panel beside an equal-units one. */
const MIN_PLOT = 80
/** The bottom margin of a panel whose x tick labels are hidden, and the tight grid's gap and margins. */
const INNER_BOTTOM = 14
const TIGHT_GAP = 2
const TIGHT_BOTTOM = 6
const TIGHT_TOP = 8

type Reports = ReadonlyMap<string, { row: number; col: number; report: CellReport }>

const sameReport = (a: CellReport, b: CellReport) =>
  a.x === b.x &&
  a.y === b.y &&
  a.labelWidth === b.labelWidth &&
  a.yName === b.yName &&
  a.xName === b.xName &&
  a.right === b.right &&
  a.top === b.top &&
  a.equal?.xSpan === b.equal?.xSpan &&
  a.equal?.ySpan === b.equal?.ySpan

/**
 * A grid of Plots, like matplotlib's `subplots`. Axes are shared by passing the same axis model to several Plots: one
 * range, one zoom, and tick labels only on the outer panel (x on the lowest panel of a column that uses it, y on the
 * leftmost of a row). Every panel in a column has the same left and right plot edges and every panel in a row the same
 * top and bottom, whatever it draws (a raster's colour bar widens its whole column). A panel with equal units in a
 * single column is sized so its plot area has the ranges' aspect, the others share the rest of the frame, and the
 * column narrows (centred) rather than run past the window. The grid draws one toolbar for all its axes.
 */
export function Plots({
  rows = 1,
  cols = 1,
  heights: heightRatios,
  widths,
  hoverGroup,
  height: ownHeight,
  scale = 1,
  toolbar = false,
  ratiosOf = 'frame',
  tight = false,
  children,
  className,
}: PlotsProps) {
  const gap = tight ? TIGHT_GAP : GAP
  const toolbarHeight = toolbar ? TOOLBAR : 0
  const height = Math.round(useChartHeight(ownHeight) * scale)
  const generated = `plots${useId()}`
  const group = hoverGroup === true ? generated : hoverGroup || undefined
  const [reports, setReports] = useState<Reports>(new Map())
  const [outer, outerSize] = useElementSize<HTMLDivElement>()

  const report = useCallback((key: string, r: CellReport | null, row: number, col: number) => {
    setReports((m) => {
      const old = m.get(key)
      if (!r) {
        if (!old) return m
        const next = new Map(m)
        next.delete(key)
        return next
      }
      if (old && old.row === row && old.col === col && sameReport(old.report, r)) return m
      return new Map(m).set(key, { row, col, report: r })
    })
  }, [])
  // One reporter per cell, stable across renders (a new one would withdraw and re-add the panel's report every render).
  const [reporters] = useState(() => new Map<string, PlotsCell['report']>())
  const reporterAt = (row: number, col: number) => {
    const at = `${row},${col}`
    let f = reporters.get(at)
    if (!f) reporters.set(at, (f = (key, r) => report(key, r, row, col)))
    return f
  }

  const all = [...reports.values()]
  const at = (row: number, col: number) => all.find((v) => v.row === row && v.col === col)?.report
  const labelsAt = (row: number, col: number) => {
    const r = at(row, col)
    if (!r) return { x: true, y: true }
    return {
      x: !all.some((o) => o.col === col && o.row > row && o.report.x === r.x),
      y: !all.some((o) => o.row === row && o.col < col && o.report.y === r.y),
    }
  }
  const columnLabels = Array.from({ length: cols }, (_, col) =>
    Math.max(0, ...all.filter((v) => v.col === col && labelsAt(v.row, v.col).y).map((v) => v.report.labelWidth)),
  )
  const columnNamed = Array.from({ length: cols }, (_, col) =>
    all.some((v) => v.col === col && v.report.yName && labelsAt(v.row, v.col).y),
  )
  const columnRight = Array.from({ length: cols }, (_, col) =>
    Math.max(GRID.right, ...all.filter((v) => v.col === col).map((v) => v.report.right)),
  )
  const marginsAt = (row: number, col: number): Margins => {
    const labels = labelsAt(row, col)
    const left = columnLabels[col] ? columnLabels[col] + LABEL_GAP + (columnNamed[col] ? Y_NAME_ROOM : 0) + 4 : 16
    const inRow = all.filter((v) => v.row === row)
    const above = row > 0 && all.some((v) => v.row === row - 1 && !labelsAt(v.row, v.col).x)
    const top = tight && above ? TIGHT_TOP : Math.max(12, ...inRow.map((v) => v.report.top))
    const bottom = Math.max(
      ...(inRow.length ? inRow : [{ row, col, report: undefined }]).map((v) => {
        const l = labelsAt(v.row, v.col)
        if (!l.x) return tight ? TIGHT_BOTTOM : INNER_BOTTOM
        return v.report?.xName === false ? 28 : 44
      }),
    )
    return { left: labels.y || columnLabels[col] ? left : 16, right: columnRight[col], top, bottom }
  }

  const ratios = useMemo(
    () => Array.from({ length: rows }, (_, i) => heightRatios?.[i] ?? 1),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the ratios' values
    [rows, heightRatios?.join(',')],
  )
  const total = ratios.reduce((a, b) => a + b, 0)
  const free = Math.max(height - toolbarHeight - gap * (rows - 1), rows * 60)
  const ratioHeights = ratios.map((r) => Math.round((free * r) / total))

  // Equal units in a single column: size the panel from its ranges' aspect (grid-layout.ts), the rest by ratio; when
  // every panel has equal units, all share one plot width.
  const equal = all.find((v) => v.report.equal && v.col === 0)
  const allEqual =
    cols === 1 &&
    rows > 1 &&
    Array.from({ length: rows }, (_, row) => at(row, 0)).every((r) => r?.equal && r.equal.xSpan > 0)
  const layout = (() => {
    if (cols !== 1 || !equal?.report.equal || outerSize.width <= 0) return null
    const m = Array.from({ length: rows }, (_, row) => marginsAt(row, 0))
    const cap = typeof window === 'undefined' ? height * 1.5 : 0.85 * window.innerHeight - toolbarHeight
    if (allEqual)
      return layoutEqualColumn({
        width: outerSize.width,
        frameHeight: height - toolbarHeight,
        cap,
        aspects: Array.from({ length: rows }, (_, row) => {
          const e = at(row, 0)!.equal!
          return e.ySpan / e.xSpan
        }),
        margins: { left: m[0].left, right: m[0].right, top: m.map((v) => v.top), bottom: m.map((v) => v.bottom) },
        gap,
      })
    return layoutColumn({
      width: outerSize.width,
      frameHeight: height - toolbarHeight,
      cap,
      ratios,
      equal: { row: equal.row, ...equal.report.equal },
      margins: { left: m[0].left, right: m[0].right, top: m.map((v) => v.top), bottom: m.map((v) => v.bottom) },
      chrome: ratios.map(() => 0),
      gap,
      minHeight: ratiosOf === 'equal' ? 24 : MIN_PLOT,
      ratiosOf,
    })
  })()
  const heights = layout?.heights ?? ratioHeights
  // A row of equal-units panels (and no explicit widths): columns as wide as one shared plot width plus their own
  // margins, so the first column's tick labels do not make its panel smaller than the others.
  const equalRow =
    cols > 1 && !widths && outerSize.width > 0
      ? Array.from({ length: rows }, (_, row) => row).find((row) =>
          Array.from({ length: cols }, (_, col) => at(row, col)).every((r) => r?.equal),
        )
      : undefined
  const columns =
    equalRow !== undefined
      ? equalRowWidths(
          outerSize.width,
          gap,
          Array.from({ length: cols }, (_, col) => marginsAt(equalRow, col)),
        )
          .map((w) => `${w}px`)
          .join(' ')
      : Array.from({ length: cols }, (_, i) => `minmax(0, ${widths?.[i] ?? 1}fr)`).join(' ')

  // The toolbar: every axis once, x axes first, in grid order.
  const toolbarAxes: ToolbarAxis[] = []
  const seen = new Set<AxisModel>()
  const ordered = [...all].sort((a, b) => a.row - b.row || a.col - b.col)
  for (const direction of ['x', 'y'] as const)
    for (const v of ordered) {
      const axis = v.report[direction]
      if (seen.has(axis)) continue
      seen.add(axis)
      toolbarAxes.push({ axis, direction })
    }

  const cells = Children.toArray(children)
  return (
    <div ref={outer} className={cn('flex w-full flex-col', className)} style={{ gap: toolbar ? 4 : 0 }}>
      {toolbar && <AxisToolbar axes={toolbarAxes} className="px-1" />}
      <div
        className="mx-auto grid w-full"
        style={{
          gridTemplateColumns: columns,
          gridTemplateRows: heights.map((h) => `${h}px`).join(' '),
          gap,
          ...(layout && layout.width < outerSize.width ? { width: layout.width } : {}),
        }}
      >
        {cells.map((child, i) => {
          const row = Math.floor(i / cols)
          const col = i % cols
          return (
            <Cell
              key={i}
              height={heights[row] ?? heights[0]}
              row={row}
              col={col}
              margins={marginsAt(row, col)}
              labels={labelsAt(row, col)}
              yNameGap={(columnLabels[col] || 24) + LABEL_GAP}
              hoverGroup={group}
              report={reporterAt(row, col)}
            >
              {child}
            </Cell>
          )
        })}
      </div>
    </div>
  )
}

/** One cell: its place, aligned margins and row height, as context for the Plot inside. */
function Cell({
  height,
  row,
  col,
  margins,
  labels,
  yNameGap,
  hoverGroup,
  report,
  children,
}: Omit<PlotsCell, 'margins' | 'labels'> & {
  margins: Margins
  labels: { x: boolean; y: boolean }
  height: number
  children: ReactNode
}) {
  const outer = useContext(FrameContext)
  const frame = useMemo(() => ({ ...outer, height }), [outer, height])
  const { left, right, top, bottom } = margins
  const { x, y } = labels
  const value = useMemo(
    (): PlotsCell => ({
      row,
      col,
      margins: { left, right, top, bottom },
      labels: { x, y },
      yNameGap,
      hoverGroup,
      report,
    }),
    [row, col, left, right, top, bottom, x, y, yNameGap, hoverGroup, report],
  )
  return (
    <div className="min-w-0">
      <PlotsContext.Provider value={value}>
        <FrameContext.Provider value={frame}>{children}</FrameContext.Provider>
      </PlotsContext.Provider>
    </div>
  )
}
