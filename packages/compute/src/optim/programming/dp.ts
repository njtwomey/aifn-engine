/**
 * Dynamic programming over tables (Bellman, 1957, "Dynamic Programming"): a generic table-filling algorithm that a
 * figure can step row by row. The sequence programmes built on it (longest common subsequence, edit distance, global
 * and local alignment) are in `sequences.ts`; the knapsacks in `aifn-methods/algorithms/dynamic-programming`.
 *
 * A program gives the table's shape and a rule for one cell from the cells before it in row-major order. The table is
 * filled in that order; a cell may also record an integer choice (which option won), from which a caller traces the
 * optimal decisions back.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { Index, Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { intTensor, matrix, vector } from './input'

/**
 * A dynamic program over a table of rank 1 or 2. `cell(i, j, get)` returns the value of cell $(i, j)$ ($j$ is 0 for a
 * rank-1 table), optionally with an integer `choice` recording which option won (for the traceback). `get(i, j)` reads
 * a cell computed earlier in row-major order; reading one not yet computed, or outside the table, throws.
 */
export interface DynamicProgram {
  /** `[rows]` or `[rows, columns]`. */
  shape: readonly [Size] | readonly [Size, Size]
  /**
   * The value of cell $(i, j)$, from the cells `get` reads: a number, or `{ value, choice }` to record which option
   * won. Called once per cell, in row-major order.
   */
  cell(i: Index, j: Index, get: (i: Index, j: Index) => Scalar): Scalar | { value: Scalar; choice: Index }
}

/**
 * One state of `dynamicProgram`: the table filled up to (not including) row `row` (`t` = `row`). `converged` once
 * every row is filled.
 */
export interface DynamicProgramState extends Status {
  /** The values, with NaN in cells not yet computed; the shape of the program. */
  table: Tensor
  /** The choice recorded for each cell ($-1$ when none), int32, same shape. */
  choice: Tensor
  /** Rows filled so far. */
  row: Size
  /** True once every row is filled (at the start already, for a table with no rows). */
  converged: boolean
}

/**
 * Fill one row of a dynamic program's table in place, cell by cell, giving `cell` a reader that throws on a cell not
 * yet computed.
 *
 * @param p The dynamic program.
 * @param values The table's values, row-major; row `i` is overwritten, and the rows before it are read.
 * @param choices The table's choices, row-major; the entries of row `i` whose cell records a choice are overwritten.
 * @param i The row to fill (for a rank-1 table, the one cell $i$).
 */
function fillRow(p: DynamicProgram, values: Float64Array, choices: Int32Array, i: number): void {
  const cols = p.shape.length === 2 ? p.shape[1] : 1
  let limit = i * cols
  const get = (a: number, b: number) => {
    const k = a * cols + b
    if (a < 0 || b < 0 || b >= cols || k >= limit)
      throw new Error(`dp: cell (${a}, ${b}) read before it was computed (computing row ${i})`)
    return values[k]
  }
  for (let j = 0; j < cols; j++) {
    const out = p.cell(i, j, get)
    const k = i * cols + j
    if (typeof out === 'number') values[k] = out
    else {
      values[k] = out.value
      choices[k] = out.choice
    }
    limit = k + 1
  }
}

/**
 * The dynamic program `p` filled row by row (Bellman, 1957), as a traceable algorithm with no start: each step fills
 * one row (one cell for a rank-1 table), so a figure can show the table growing. `dp` runs it to the end. A `cell`
 * that reads a cell not yet computed throws when its row is filled.
 *
 * @param p The dynamic program: the table's shape and the rule for one cell.
 * @returns The algorithm. Its start is ignored; it starts from an all-NaN table, and once the table is full a step
 *   returns the state unchanged.
 *
 * @example Count the monotone lattice paths of a grid, one row per step
 * // Paths from the top-left corner that move only right or down: each cell adds the one above and the one to its left.
 * const p = { shape: [3, 4], cell: (i, j, get) => (i === 0 || j === 0 ? 1 : get(i - 1, j) + get(i, j - 1)) }
 * const tr = trace(dynamicProgram(p), {}, 10)
 * print('after one row:', tr.steps[1].table)
 * print('filled:', tr.steps.at(-1).table)
 * print('steps =', tr.steps.at(-1).t)
 */
export function dynamicProgram(p: DynamicProgram): Algorithm<object, DynamicProgramState> {
  const size = p.shape.length === 2 ? p.shape[0] * p.shape[1] : p.shape[0]
  return {
    name: 'dynamic-program',
    init: () => ({
      t: 0,
      table: tableOf(new Float64Array(size).fill(NaN), p.shape),
      choice: intTensor(new Int32Array(size).fill(-1), p.shape),
      row: 0,
      converged: p.shape[0] === 0,
    }),
    step: (s) => {
      if (s.converged) return s
      const values = Float64Array.from(s.table.data)
      const choices = Int32Array.from(s.choice.data)
      fillRow(p, values, choices, s.row)
      const row = s.row + 1
      return {
        t: s.t + 1,
        table: tableOf(values, p.shape),
        choice: intTensor(choices, p.shape),
        row,
        converged: row >= p.shape[0],
      }
    },
  }
}

/**
 * Wrap a table's values as a tensor of the program's shape.
 *
 * @param values The values, row-major (copied).
 * @param shape The program's shape, `[rows]` or `[rows, columns]`.
 * @returns A vector or a matrix tensor.
 */
const tableOf = (values: Float64Array, shape: readonly number[]) =>
  shape.length === 2 ? matrix(values, shape[0], shape[1]) : vector(values)

/**
 * Fill a dynamic program's whole table: the final state of `dynamicProgram`, computed in place without the per-row
 * copies. Returns the table of values and of choices.
 *
 * @param program The dynamic program: the table's shape and the rule for one cell.
 * @returns The final state, `converged`, with every cell filled.
 *
 * @example Fibonacci numbers in a rank-1 table
 * const s = dp({ shape: [11], cell: (i, _j, get) => (i < 2 ? i : get(i - 1, 0) + get(i - 2, 0)) })
 * print('F(0..10) =', s.table)
 *
 * @example A minimum-cost path, with the choices for the traceback
 * // Move right or down through the grid; each cell adds its cost to the cheaper of the cell above (choice 0) and the
 * // cell to its left (choice 1).
 * const cost = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
 * const s = dp({
 *   shape: [3, 3],
 *   cell: (i, j, get) => {
 *     if (i === 0 && j === 0) return cost[0][0]
 *     const up = i > 0 ? get(i - 1, j) : Infinity
 *     const left = j > 0 ? get(i, j - 1) : Infinity
 *     return up <= left ? { value: cost[i][j] + up, choice: 0 } : { value: cost[i][j] + left, choice: 1 }
 *   },
 * })
 * print('cheapest cost to each cell =', s.table)
 * print('choices =', s.choice)
 */
export function dp(program: DynamicProgram): DynamicProgramState {
  const values = new Float64Array(program.shape.length === 2 ? program.shape[0] * program.shape[1] : program.shape[0])
  const choices = new Int32Array(values.length).fill(-1)
  for (let i = 0; i < program.shape[0]; i++) fillRow(program, values, choices, i)
  return {
    t: program.shape[0],
    table: tableOf(values, program.shape),
    choice: intTensor(choices, program.shape),
    row: program.shape[0],
    converged: true,
  }
}
