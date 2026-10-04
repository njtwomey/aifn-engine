/**
 * Dynamic programming over tables (Bellman, 1957, "Dynamic Programming"): a generic table-filling algorithm that a
 * figure can step row by row. The sequence programmes built on it (longest common subsequence, edit distance, global
 * and local alignment) are in `sequences.ts`; the knapsacks in `aifn-methods/algorithms/dynamic-programming`.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { Index, Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { intTensor, matrix, vector } from './input'

/**
 * A dynamic program over a table of rank 1 or 2. `cell(i, j, get)` returns the value of cell (i, j) (j is 0 for a rank-1
 * table), optionally with an integer `choice` recording which option won (for the traceback). `get(i, j)` reads a cell
 * computed earlier in row-major order; reading one not yet computed throws.
 */
export interface DynamicProgram {
  /** `[rows]` or `[rows, columns]`. */
  shape: readonly [Size] | readonly [Size, Size]
  cell(i: Index, j: Index, get: (i: Index, j: Index) => Scalar): Scalar | { value: Scalar; choice: Index }
}

/**
 * One state of `dynamicProgram`: the table filled up to (not including) row `row` (`t` = `row`). `converged` once
 * every row is filled.
 */
export interface DynamicProgramState extends Status {
  /** The values, with NaN in cells not yet computed; the shape of the program. */
  table: Tensor
  /** The choice recorded for each cell (−1 when none), int32, same shape. */
  choice: Tensor
  /** Rows filled so far. */
  row: Size
  converged: boolean
}

/** Fill one row of a dynamic program's table in place. */
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
 * one row (one cell for a rank-1 table), so a figure can show the table growing. `dp` runs it to the end.
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

const tableOf = (values: Float64Array, shape: readonly number[]) =>
  shape.length === 2 ? matrix(values, shape[0], shape[1]) : vector(values)

/**
 * Fill a dynamic program's whole table: the final state of `dynamicProgram`, computed in place without the per-row
 * copies. Returns the table of values and of choices.
 * @example dp({ shape: [11], cell: (i, _j, get) => (i < 2 ? i : get(i - 1, 0) + get(i - 2, 0)) }) // Fibonacci
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
