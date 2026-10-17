/**
 * Scan orders: the orders in which a sequence layer reads the cells of a 2-D grid, and what each order does to
 * neighbouring cells. A state-space or other recurrent layer reads an image as a sequence of patches. The scan order
 * decides how far apart in the sequence two neighbouring patches land, and which patches each output can see. This
 * file builds the common orders on a grid of $R \times C$ cells (`rows` by `cols`), numbered `r * cols + c`:
 *
 * - `row`: rows top to bottom, each left to right (raster order, as a flattened image in a vision transformer);
 * - `column`: columns left to right, each top to bottom;
 * - `snake`: rows top to bottom, alternating left to right and right to left (boustrophedon), so consecutive cells are
 *   always neighbours;
 * - `hilbert`: the Hilbert curve (square grids whose side is a power of two), which keeps cells of every
 *   $2^k \times 2^k$ block together.
 *
 * A scan scheme is a list of routes, each an order and a direction. Vision Mamba (Zhu et al., 2024) runs the row order
 * forwards and backwards; VMamba's 2-D selective scan (Liu et al., 2024, SS2D) runs four routes, the row and column
 * orders in both directions, and merges their outputs. The sequence distance of two cells under a scheme is the
 * smallest distance over its routes, and a cell's output sees the cells that come no later than it on at least one
 * route. Orders and positions are index lists (`number[]`), as `positionRange` of `aifn-compute/nn/attention` returns
 * positions.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A cell order: `row` (raster), `column`, `snake` (boustrophedon) or `hilbert` (the Hilbert curve, square grids of a
 * power-of-two side).
 */
export type ScanOrderKind = 'row' | 'column' | 'snake' | 'hilbert'

/** One route of a scheme: an order, read forwards or backwards. */
export type ScanRoute = {
  /** The cell order the route follows. */
  order: ScanOrderKind
  /** Read the order backwards (default false). */
  reverse?: boolean
}

/**
 * The cell indices of a grid of `rows` by `cols` cells in the given order. Throws `DomainError` for a Hilbert order
 * on a grid that is not square with a power-of-two side.
 *
 * @param kind The order.
 * @param rows The number of rows $R$.
 * @param cols The number of columns $C$.
 * @returns The $RC$ cell indices (`r * cols + c`) in the order the sequence visits them.
 *
 * @example The four orders on a 4 by 4 grid
 * for (const kind of ['row', 'column', 'snake', 'hilbert']) print(kind, scanOrder(kind, 4, 4))
 */
export function scanOrder(kind: ScanOrderKind, rows: number, cols: number): number[] {
  const out: number[] = []
  if (kind === 'row') for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out.push(r * cols + c)
  else if (kind === 'column') for (let c = 0; c < cols; c++) for (let r = 0; r < rows; r++) out.push(r * cols + c)
  else if (kind === 'snake')
    for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) out.push(r * cols + (r % 2 === 0 ? k : cols - 1 - k))
  else {
    if (rows !== cols || (rows & (rows - 1)) !== 0)
      throw new DomainError('scanOrder', 'scanOrder: hilbert needs a square grid whose side is a power of two')
    for (let d = 0; d < rows * cols; d++) {
      const [x, y] = hilbertPoint(rows, d)
      out.push(y * cols + x)
    }
  }
  return out
}

/**
 * The cell $(x, y)$ at distance $d$ along the Hilbert curve of an $n \times n$ grid ($n$ a power of two), by the
 * classic rotate-and-flip loop over scales.
 *
 * @param n The side of the grid, a power of two.
 * @param d The distance along the curve, $0 \le d < n^2$.
 * @returns The column $x$ and row $y$ of the cell.
 */
function hilbertPoint(n: number, d: number): [number, number] {
  let x = 0
  let y = 0
  let t = d
  for (let s = 1; s < n; s *= 2) {
    const rx = 1 & (t >> 1)
    const ry = 1 & (t ^ rx)
    if (ry === 0) {
      if (rx === 1) {
        x = s - 1 - x
        y = s - 1 - y
      }
      ;[x, y] = [y, x]
    }
    x += s * rx
    y += s * ry
    t >>= 2
  }
  return [x, y]
}

/**
 * For each route of a scheme, the position of every cell in its sequence.
 *
 * @param routes The routes of the scheme.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @returns One array per route; `positions[route][cell]` is the cell's position in that route's sequence.
 *
 * @example A raster scan read both ways on a 2 by 3 grid
 * print('positions =', routePositions([{ order: 'row' }, { order: 'row', reverse: true }], 2, 3))
 */
export function routePositions(routes: readonly ScanRoute[], rows: number, cols: number): number[][] {
  return routes.map(({ order, reverse }) => {
    const seq = scanOrder(order, rows, cols)
    if (reverse) seq.reverse()
    const pos = new Array<number>(rows * cols)
    seq.forEach((cell, i) => (pos[cell] = i))
    return pos
  })
}

/**
 * Every pair of horizontally or vertically adjacent cells of a grid.
 *
 * @param rows The number of rows $R$.
 * @param cols The number of columns $C$.
 * @returns The $R(C - 1) + C(R - 1)$ pairs `[a, b]` of cell indices with `a < b`.
 *
 * @example The 7 neighbour pairs of a 2 by 3 grid
 * print('pairs =', neighbourPairs(2, 3))
 */
export function neighbourPairs(rows: number, cols: number): [number, number][] {
  const pairs: [number, number][] = []
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      if (c + 1 < cols) pairs.push([r * cols + c, r * cols + c + 1])
      if (r + 1 < rows) pairs.push([r * cols + c, (r + 1) * cols + c])
    }
  return pairs
}

/**
 * The sequence distance of two cells under a scheme: the smallest absolute difference of their positions over its
 * routes.
 *
 * @param positions The positions of every cell on each route, as `routePositions` returns them.
 * @param a The first cell index.
 * @param b The second cell index.
 * @returns The smallest $\lvert p_r(a) - p_r(b) \rvert$ over the routes $r$.
 *
 * @example Vertical neighbours: 8 apart in raster order on an 8 by 8 grid, 1 apart with a column route added
 * print('row:', sequenceDistance(routePositions([{ order: 'row' }], 8, 8), 0, 8))
 * print('row and column:', sequenceDistance(routePositions([{ order: 'row' }, { order: 'column' }], 8, 8), 0, 8))
 */
export function sequenceDistance(positions: readonly (readonly number[])[], a: number, b: number): number {
  return Math.min(...positions.map((p) => Math.abs(p[a] - p[b])))
}

/**
 * Whether cell `source` is visible to the output at cell `target`: it comes no later on at least one route.
 *
 * @param positions The positions of every cell on each route, as `routePositions` returns them.
 * @param target The cell whose output is asked about.
 * @param source The cell that may or may not be seen.
 * @returns True when `source` is at or before `target` on some route.
 *
 * @example A forward raster scan sees only the past; adding the reverse route sees everything
 * const fwd = routePositions([{ order: 'row' }], 4, 4)
 * const both = routePositions([{ order: 'row' }, { order: 'row', reverse: true }], 4, 4)
 * print('cell 15 seen from cell 5, forward:', visibleTo(fwd, 5, 15))
 * print('both ways:', visibleTo(both, 5, 15))
 */
export function visibleTo(positions: readonly (readonly number[])[], target: number, source: number): boolean {
  return positions.some((p) => p[source] <= p[target])
}
