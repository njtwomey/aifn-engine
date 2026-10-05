/**
 * One drawing of a grid environment (gridworld, maze, FrozenLake, the cliff) or of a search on a grid, from a `render`
 * spec (compute `GridRender`). Layers, bottom to top: the cells by kind, or in their place a value field (V(s), max Q,
 * visit counts) or tones (colour only: a search's visited cells, frontier and path); policy arrows; a path drawn as
 * move arrows (`pathMoves`: one arrow per direction of a cell edge, counted when walked more than once, the two
 * directions side by side, the latest move in ink); the agent, in the success or destructive tone at an episode's
 * ending; then the page's own layers (`children`: handles, markers). Arrows are optional: without `policy` and `path`
 * the grid is colour only. Cells are square and the axes hold their first fit.
 */
import { useMemo, type ReactNode } from 'react'
import type { EpisodeEnd, GridRender } from 'aifn-compute/foundation/contracts'
import { Plot, Points, Raster, useAxis, Vectors, type AxisModel, type Range } from '../viz'
import { cellXY, GRID_KINDS, kindRows, pathMoves, policyArrows, toneRows, valueRows } from './grid'

export type GridValueField = {
  /** A value per cell index (V(s), or max_a Q(s, a)); walls and cliffs are left blank. */
  values: ArrayLike<number>
  /** The value's name in the tooltip and over the colour bar. */
  label: string
  /** The colour scale's ends (default symmetric about zero from the values). */
  range?: Range
  /** The colour bar beside the grid (default true). */
  colorBar?: boolean
  /** `diverging` (default) for signed values, `sequential` for values of one sign. */
  scale?: 'diverging' | 'sequential'
  /** Cell colour strength below 1, so a path in a colour near the scale's still reads (default 1). */
  fillOpacity?: number
}

/** A colour per cell for colour-only drawing (a search): `values[c]` indexes `names` (−1 for none); walls stay empty. */
export type GridTones = {
  values: ArrayLike<number>
  names: readonly string[]
}

export type GridViewProps<S> = {
  render: GridRender<S>
  title?: string
  /** A value field drawn as a raster in place of the cell kinds. */
  value?: GridValueField | null
  /** Tones drawn in place of the cell kinds (ignored under a value field). */
  tones?: GridTones | null
  /** An action per cell index, drawn as arrows (a negative action draws nothing). */
  policy?: ArrayLike<number> | null
  /** A path of states, drawn as move arrows up to `step`. */
  path?: readonly S[] | null
  /** The path's last drawn index (default its end); the agent stands at this state. */
  step?: number
  /** The palette slot of the path's moves (default 1), for pages with more than one path. */
  pathSlot?: number
  /** Draw the path's latest move in ink (default true; false for a whole route such as a greedy path). */
  inkLatest?: boolean
  /** Mark the agent at the path's state at `step` (default true when there is a path). */
  agent?: boolean
  /** At an episode's last step, how it ended: the agent takes the success or destructive tone. */
  end?: EpisodeEnd | null
  /** Shared axes, when several grids zoom together; by default the view makes its own with equal units. */
  x?: AxisModel
  y?: AxisModel
  /** The plot's size in a `Plots` group. */
  scale?: number
  /** Pixels, outside a Figure's frame. */
  height?: number
  /** A click on the grid, in data units (round to get the cell): toggling walls, say. */
  onPlotClick?: (point: [number, number]) => void
  ariaLabel?: string
  /** Extra layers drawn on top (handles for a goal or a trap, markers). */
  children?: ReactNode
}

const axisOf = (n: number) => Array.from({ length: n }, (_, i) => i)

export function GridView<S>({
  render: r,
  title,
  value,
  tones,
  policy,
  path,
  step,
  pathSlot = 1,
  inkLatest = true,
  agent = true,
  end,
  x,
  y,
  scale,
  height: plotHeight,
  onPlotClick,
  ariaLabel,
  children,
}: GridViewProps<S>) {
  const { width, height } = r
  const xs = useMemo(() => axisOf(width), [width])
  const ys = useMemo(() => axisOf(height), [height])
  const kinds = useMemo(() => kindRows(r as GridRender<unknown>), [r])
  const values = value?.values
  const field = useMemo(() => (values ? valueRows(r as GridRender<unknown>, values) : null), [r, values])
  const toneValues = tones?.values
  const toneNames = tones?.names
  const toned = useMemo(
    () => (toneValues && toneNames ? toneRows(r as GridRender<unknown>, toneValues, toneNames) : null),
    [r, toneValues, toneNames],
  )
  const arrows = useMemo(() => (policy ? policyArrows(r as GridRender<unknown>, policy) : null), [r, policy])
  const cells = useMemo(() => (path ? path.map((s) => cellXY(width, r.cell(s))) : null), [path, r, width])
  const at = cells ? Math.min(step ?? cells.length - 1, cells.length - 1) : -1
  const moves = useMemo(
    () => (cells ? pathMoves(cells, at, { slot: pathSlot, inkLatest }) : null),
    [cells, at, pathSlot, inkLatest],
  )
  const ownX = useAxis({ label: 'x', hold: 'initial', key: r })
  const ownY = useAxis({ label: 'y', equal: ownX, hold: 'initial', key: r })
  const tone = end ? (end.success ? 'success' : 'destructive') : undefined
  return (
    <Plot
      x={x ?? ownX}
      y={y ?? ownY}
      title={title}
      scale={scale}
      height={plotHeight}
      onPlotClick={onPlotClick}
      ariaLabel={ariaLabel}
    >
      {field ? (
        <Raster
          x={xs}
          y={ys}
          z={field}
          scale={value?.scale ?? 'diverging'}
          range={value?.range}
          valueLabel={value?.label}
          colorBar={value?.colorBar}
          fillOpacity={value?.fillOpacity}
        />
      ) : toned && toneNames ? (
        <Raster x={xs} y={ys} z={toned} scale="categorical" categoryNames={toneNames} />
      ) : (
        <Raster x={xs} y={ys} z={kinds} scale="categorical" categoryNames={GRID_KINDS} />
      )}
      {arrows && <Vectors vectors={arrows} />}
      {moves && <Vectors vectors={moves} />}
      {cells && agent && at >= 0 && (
        <Points
          name="agent"
          x={[cells[at][0]]}
          y={[cells[at][1]]}
          emphasis={!end}
          tone={tone}
          size={end ? 16 : undefined}
        />
      )}
      {children}
    </Plot>
  )
}
