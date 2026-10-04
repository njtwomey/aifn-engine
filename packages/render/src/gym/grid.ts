/**
 * Pure helpers behind `GridView`: a grid environment's cells as raster rows, a value table or a colour per cell (a
 * search's visited cells and frontier) as raster rows, a policy as arrows, and a path of cells as move arrows that keep
 * backtracking visible.
 */
import type { GridRender } from 'aifn-compute/foundation/contracts'
import type { Vector } from '@render/viz'

/** Cell kinds drawn in colour, in slot order; other cells are blank. */
export const GRID_KINDS = ['wall', 'goal', 'trap', 'start', 'hole', 'cliff', 'terminal'] as const

/** Cell kinds left blank under a value field: no agent stays in them. */
const NO_VALUE = new Set(['wall', 'cliff'])

/** A cell index as (x, y), y = 0 the bottom row. */
export const cellXY = (width: number, c: number): [number, number] => [c % width, Math.floor(c / width)]

/** The cells by kind as categorical rows (bottom row first): kind k of `GRID_KINDS` in slot k, others −1 (blank). */
export function kindRows(r: GridRender<unknown>): number[][] {
  return Array.from({ length: r.height }, (_, y) =>
    Array.from({ length: r.width }, (_, x) =>
      GRID_KINDS.indexOf(r.cells[y * r.width + x] as (typeof GRID_KINDS)[number]),
    ),
  )
}

/** A value per cell index as raster rows; walls and cliffs are blank. */
export function valueRows(r: GridRender<unknown>, values: ArrayLike<number>): number[][] {
  return Array.from({ length: r.height }, (_, y) =>
    Array.from({ length: r.width }, (_, x) => {
      const c = y * r.width + x
      return NO_VALUE.has(r.cells[c]) ? NaN : values[c]
    }),
  )
}

/**
 * A tone per cell index as categorical raster rows for colour-only drawing (a search's visited cells, frontier and
 * path): tone k of `names` in slot k, other cells unassigned (−1, a neutral fill), and walls left empty (NaN, the
 * background) unless toned.
 */
export function toneRows(r: GridRender<unknown>, tones: ArrayLike<number>, names: readonly string[]): number[][] {
  return Array.from({ length: r.height }, (_, y) =>
    Array.from({ length: r.width }, (_, x) => {
      const c = y * r.width + x
      const t = tones[c]
      return t >= 0 && t < names.length ? t : r.cells[c] === 'wall' ? NaN : -1
    }),
  )
}

/** An action per cell index as arrows from each cell's centre (a negative action draws nothing). Ink. */
export function policyArrows(r: GridRender<unknown>, policy: ArrayLike<number>): Vector[] {
  const out: Vector[] = []
  for (let c = 0; c < policy.length; c++) {
    const a = policy[c]
    if (a < 0 || NO_VALUE.has(r.cells[c])) continue
    const [x, y] = cellXY(r.width, c)
    const [dx, dy] = r.actionVectors[a]
    out.push({ from: [x - 0.2 * dx, y - 0.2 * dy], to: [x + 0.3 * dx, y + 0.3 * dy] })
  }
  return out
}

/** The sideways shift, in cells, of each direction's arrow on an edge walked both ways. */
export const LANE_OFFSET = 0.12
/** Stroke width of a move walked once, and the extra width per further walk, capped at `MAX_WIDTH`. */
const BASE_WIDTH = 1.5
const WIDTH_PER_WALK = 0.4
const MAX_WIDTH = 3.5

export type PathMovesOptions = {
  /** The palette slot of the moves (default 1). */
  slot?: number
  /** Draw the arrow holding the latest move in ink rather than in `slot` (default true). */
  inkLatest?: boolean
}

/**
 * The moves of a path of cells (x, y) up to `step`, aggregated by directed edge: at most one arrow per direction of a
 * cell edge, labelled at its midpoint with its count when walked more than once. An edge walked one way only gets one
 * centred arrow; an edge walked both ways gets two, shifted `LANE_OFFSET` to either side, so backtracking shows as a
 * pair of opposite arrows. The stroke thickens a little with the count. A move that stays in its cell (a bump into a
 * wall) draws nothing, nor does a jump of more than one cell (a trap's return to the start). The arrow holding the
 * latest move is in ink (unless `inkLatest` is false) and drawn last.
 */
export function pathMoves(
  cells: readonly (readonly [number, number])[],
  step: number,
  { slot = 1, inkLatest = true }: PathMovesOptions = {},
): Vector[] {
  const last = Math.min(step, cells.length - 1)
  const edges = new Map<string, { a: readonly [number, number]; b: readonly [number, number]; n: number }>()
  let latest: string | null = null
  for (let i = 0; i < last; i++) {
    const [a, b] = [cells[i], cells[i + 1]]
    // A stay (a bump into a wall) draws nothing, nor does a jump (a trap or a cliff sending the agent back).
    if (a[0] === b[0] && a[1] === b[1]) continue
    if (Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])) > 1) continue
    const key = `${a[0]},${a[1]}>${b[0]},${b[1]}`
    const e = edges.get(key)
    if (e) e.n += 1
    else edges.set(key, { a, b, n: 1 })
    if (i === last - 1) latest = key
  }
  const out: Vector[] = []
  let ink: Vector | null = null
  for (const [key, { a, b, n }] of edges) {
    const [ax, ay] = a
    const [bx, by] = b
    const len = Math.hypot(bx - ax, by - ay)
    const [dx, dy] = [(bx - ax) / len, (by - ay) / len]
    // The normal turns with the move, so the two directions of an edge land on opposite sides.
    const shift = edges.has(`${bx},${by}>${ax},${ay}`) ? LANE_OFFSET : 0
    const [ox, oy] = [-dy * shift, dx * shift]
    const v: Vector = {
      from: [ax + 0.18 * dx + ox, ay + 0.18 * dy + oy],
      to: [bx - 0.18 * dx + ox, by - 0.18 * dy + oy],
      width: Math.min(MAX_WIDTH, BASE_WIDTH + WIDTH_PER_WALK * (n - 1)),
      ...(n > 1 ? { label: `${n}`, labelAt: 'middle' as const } : {}),
    }
    if (inkLatest && key === latest) ink = v
    else out.push({ ...v, slot })
  }
  if (ink) out.push(ink)
  return out
}
