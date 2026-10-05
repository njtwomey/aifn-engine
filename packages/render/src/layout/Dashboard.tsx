import {
  Children,
  createContext,
  isValidElement,
  useContext,
  useMemo,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
} from 'react'
import { cn } from '../lib/utils'
import { FrameContext, useChartHeight, useElementSize } from '../viz'

/**
 * A dashboard: a Figure's chart area split into rows of cells holding charts, tables or views. Rows share the frame's
 * height by `ratio` (each at least `minHeight`); cells share their row's width by `ratio`, except cells with an
 * `aspect`, which take the row's height and a width that follows from it (`aspect="square"` for a ROC or PR curve), so
 * a square is never stretched. Every cell tells the charts inside how tall to be, so they fill it. Below `stackBelow`
 * pixels of width (a phone, a small figure) every cell stacks at the full width: plain cells keep their row's height
 * and aspect cells (and `stackAspect` cells) take their height from the width.
 *
 *   <Dashboard>
 *     <DashboardRow ratio={1.2}>
 *       <DashboardCell><Plot x={x} y={y}><Raster … /></Plot></DashboardCell>
 *       <DashboardCell><Plot x={t} y={v}><Curve … /></Plot></DashboardCell>
 *     </DashboardRow>
 *     <DashboardRow>
 *       <DashboardCell ratio={1.4}><ContingencyTableView … /></DashboardCell>
 *       <DashboardCell aspect="square"><Plot x={fpr} y={tpr}><Curve … /></Plot></DashboardCell>
 *     </DashboardRow>
 *   </Dashboard>
 *
 * Charts in different cells share nothing unless given the same axis model; aligned plot areas are a `Plots` concern
 * (a Plots grid may sit in a cell).
 */
export function Dashboard({
  children,
  gap = 12,
  stackBelow = 640,
  className,
}: {
  children: ReactNode
  /** Pixels between rows and between cells. */
  gap?: number
  /** Stack every cell in one column when the dashboard is narrower than this, in pixels. */
  stackBelow?: number
  className?: string
}) {
  const height = useChartHeight()
  const [ref, size] = useElementSize<HTMLDivElement>()
  const rows = Children.toArray(children).filter(isRow)
  // Rows that fit their width (a diagram) take the height their content needs; the others share the frame's height.
  const total = rows.reduce((s, r) => s + (r.props.fit === 'width' ? 0 : (r.props.ratio ?? 1)), 0) || 1
  const room = Math.max(height - gap * (rows.length - 1), 0)
  const stacked = size.width > 0 && size.width < stackBelow
  return (
    <div ref={ref} className={cn('flex w-full flex-col', className)} style={{ gap }}>
      {size.width > 0 &&
        rows.map((row, i) => (
          <RowContext.Provider
            key={row.key ?? i}
            value={{
              height: Math.max(row.props.minHeight ?? 200, Math.round((room * (row.props.ratio ?? 1)) / total)),
              width: size.width,
              gap,
              stacked,
            }}
          >
            {row}
          </RowContext.Provider>
        ))}
    </div>
  )
}

type RowProps = {
  children: ReactNode
  /** This row's share of the frame's height (default 1). */
  ratio?: number
  /** The least height of the row, in pixels (default 200); the frame grows rather than squash it. */
  minHeight?: number
  /**
   * `width`: the row's cells fit the full width and take the height that needs (a diagram such as a tree grows taller
   * as the figure widens), instead of a share of the frame's height. The frame grows to hold it.
   */
  fit?: 'width'
  className?: string
}

type CellProps = {
  children: ReactNode
  /** This cell's share of the row's width left after the aspect cells (default 1). */
  ratio?: number
  /**
   * Width over height: the cell takes the row's height and this times it as its width (`square` is 1). Use it for a
   * chart whose plot must keep a shape, e.g. a ROC curve with `aspect="equal"` on [0, 1]².
   */
  aspect?: 'square' | number
  /**
   * Width over height when stacked (narrow screens) for a cell without `aspect`, e.g. `square` for a feature-space
   * plot that would otherwise be letterboxed into a short strip. Default: the row's height.
   */
  stackAspect?: 'square' | number
  /** The least width of a cell without `aspect`, in pixels (default 160): aspect cells shrink, and the row with them. */
  minWidth?: number
  className?: string
}

type RowLayout = { height: number; width: number; gap: number; stacked: boolean }
const RowContext = createContext<RowLayout>({ height: 240, width: 0, gap: 12, stacked: false })

const isRow = (c: unknown): c is ReactElement<RowProps> => isValidElement(c) && c.type === DashboardRow
const isCell = (c: unknown): c is ReactElement<CellProps> => isValidElement(c) && c.type === DashboardCell
const aspectOf = (a: CellProps['aspect']) => (a === 'square' ? 1 : a)

/** One row of a `Dashboard`; its children are `DashboardCell`s. */
export function DashboardRow({ children, className, fit, minHeight }: RowProps) {
  const { height, width, gap, stacked } = useContext(RowContext)
  const cells = Children.toArray(children).filter(isCell)
  if (fit === 'width')
    return (
      <div
        className={cn('flex w-full', stacked ? 'flex-col items-center' : 'flex-row', className)}
        style={{ gap, minHeight }}
      >
        {cells.map((cell, i) => (
          <CellFrame
            key={cell.key ?? i}
            box={{ height: undefined, flex: `${cell.props.ratio ?? 1} 1 0` }}
            stacked={stacked}
            className={cell.props.className}
          >
            {cell.props.children}
          </CellFrame>
        ))}
      </div>
    )
  // Aspect cells first take their width from the row's height; if they would not fit, they shrink together.
  const fixed = cells.map((c) => aspectOf(c.props.aspect))
  const wanted = fixed.reduce<number>((s, a) => s + (a ? a * height : 0), 0)
  const flexibleWidth = cells.reduce((sum, c, i) => sum + (fixed[i] ? 0 : (c.props.minWidth ?? 160)), 0)
  const space = width - gap * (cells.length - 1) - flexibleWidth
  const shrink = wanted > 0 && wanted > space ? Math.max(space, 0) / wanted : 1
  // Aspect cells that would not fit shrink with the whole row, so they keep their shape and leave no empty band.
  const rowHeight = Math.round(height * shrink)
  return (
    <div
      className={cn('flex w-full', stacked ? 'flex-col items-center' : 'flex-row', className)}
      style={{ gap, ...(stacked ? {} : { height: rowHeight }) }}
    >
      {cells.map((cell, i) => {
        const a = fixed[i]
        const box: { width?: number; height: number; flex?: string } = stacked
          ? a || aspectOf(cell.props.stackAspect)
            ? { width, height: Math.round(width / (a || aspectOf(cell.props.stackAspect)!)) }
            : { height }
          : a
            ? { width: Math.round(a * rowHeight), height: rowHeight, flex: 'none' }
            : { height: rowHeight, flex: `${cell.props.ratio ?? 1} 1 0` }
        return (
          <CellFrame key={cell.key ?? i} box={box} stacked={stacked} className={cell.props.className}>
            {cell.props.children}
          </CellFrame>
        )
      })}
    </div>
  )
}

/** One cell of a `DashboardRow`: a chart, a table or a view. See `Dashboard`. */
export function DashboardCell(props: CellProps) {
  // Laid out by its row, which reads these props; rendered on its own it is a plain box.
  return <div className={props.className}>{props.children}</div>
}

function CellFrame({
  box,
  stacked,
  className,
  children,
}: {
  /** No height: the content sets it (a `fit="width"` row). */
  box: { width?: number; height: number | undefined; flex?: string }
  stacked: boolean
  className?: string
  children: ReactNode
}) {
  const outer = useContext(FrameContext)
  const value = useMemo(() => ({ ...outer, height: box.height }), [outer, box.height])
  const style: CSSProperties = {
    height: box.height,
    ...(box.width !== undefined ? { width: box.width } : stacked ? { width: '100%' } : {}),
    ...(box.flex ? { flex: box.flex } : {}),
  }
  return (
    <div className={cn('relative min-w-0 overflow-hidden', className)} style={style}>
      <FrameContext.Provider value={value}>{children}</FrameContext.Provider>
    </div>
  )
}
