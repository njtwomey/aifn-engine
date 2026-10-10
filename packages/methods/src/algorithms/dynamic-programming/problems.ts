/**
 * The 0/1 and unbounded knapsacks as dynamic programs on the `dp` engine of `aifn-compute/optim/programming` (whose
 * `sequences` hold the longest common subsequence, edit distance and alignments), with solvers that trace the best
 * choice back through the table.
 *
 * Weights and the capacity $C$ are non-negative integers, so the table has one column per capacity $0, \dots, C$ and
 * the solve takes $O(nC)$ time for $n$ items (Bellman, 1957, "Dynamic Programming"); values may be any numbers.
 * Invalid input throws `DomainError` (or `ShapeError` when the lengths differ). Each `…Program` function gives the
 * problem as a `DynamicProgram`, for `dp` or for stepping through with `dynamicProgram`; the plain function solves it.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { type DynamicProgram, dp } from 'aifn-compute/optim/programming'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * Entry $(i, j)$ of a rank-1 or rank-2 tensor, by its strides.
 *
 * @param t The tensor.
 * @param i The row (for a rank-1 tensor, the index).
 * @param j The column (ignored for a rank-1 tensor).
 * @returns The entry.
 */
const at = (t: Tensor, i: number, j: number) => t.data[t.offset + i * t.strides[0] + j * (t.strides[1] ?? 0)]

// ---------------------------------------------------------------------------------------------------------------------
// Knapsack.

/**
 * A vector input as float64, of length `n` when given (else a `ShapeError`).
 *
 * @param v The input vector.
 * @param where The caller's name for the input, used in error messages.
 * @param n The required length, if any.
 * @returns The values, as a new array.
 */
function readVector(v: VectorLike, where: string, n?: number): Float64Array {
  const out = dense.toF64(v, where)
  if (n !== undefined && out.length !== n)
    throw new ShapeError(where, `${where}: expected ${n} entries, got ${out.length}`)
  return out
}

/**
 * An int32 tensor of the given shape (a vector by default) from integers.
 *
 * @param values The entries, row-major.
 * @param shape The tensor's shape (default a vector of all the values).
 * @returns The tensor.
 */
const intTensor = (values: ArrayLike<number>, shape: readonly number[] = [values.length]): Tensor =>
  fromData(Int32Array.from(values), shape)
/**
 * An int32 vector from integers.
 *
 * @param values The entries.
 * @returns The vector.
 */
const intVector = (values: ArrayLike<number>): Tensor => intTensor(values)

/**
 * The items of a knapsack problem, checked: as many weights as values, the weights and the capacity non-negative
 * integers. Throws `ShapeError` or `DomainError` otherwise.
 *
 * @param values The item values $v_i$ ($n$ numbers).
 * @param weights The item weights $w_i$ ($n$ non-negative integers).
 * @param capacity The capacity $C$.
 * @param where The caller's name, used in error messages.
 * @returns The values `v` and weights `w` as float64 arrays.
 */
function readItems(values: VectorLike, weights: VectorLike, capacity: number, where: string) {
  const v = readVector(values, `${where}: values`)
  const w = readVector(weights, `${where}: weights`, v.length)
  if (!Number.isInteger(capacity) || capacity < 0)
    throw new DomainError(where, `${where}: capacity must be a non-negative integer`)
  for (const x of w)
    if (!Number.isInteger(x) || x < 0) throw new DomainError(where, `${where}: weights must be non-negative integers`)
  return { v, w }
}

/**
 * The 0/1 knapsack table as a dynamic program: cell $(i, c)$ is the best value from the first $i$ items within
 * capacity $c$, $T_{i,c} = \max(T_{i-1,c}, T_{i-1,c-w_i} + v_i)$ (the second option only when $w_i \le c$), with
 * $T_{0,c} = 0$; the choice is 1 when item $i$ is taken, which happens only when that is strictly better. Shape
 * $(n + 1) \times (C + 1)$.
 *
 * @param values The item values $v_i$ ($n$ numbers).
 * @param weights The item weights $w_i$ ($n$ non-negative integers).
 * @param capacity The capacity $C$, a non-negative integer.
 * @returns The program: its shape and its cell rule, for `dp`.
 *
 * @example The table, filled row by row as `dp` does
 * const program = knapsackProgram([60, 100, 120], [1, 2, 3], 5)
 * const T = []
 * for (let i = 0; i < program.shape[0]; i++) {
 *   T.push([])
 *   for (let c = 0; c < program.shape[1]; c++) T[i].push(program.cell(i, c, (k, l) => T[k][l]).value)
 * }
 * print('shape =', program.shape)
 * print('T =', T)
 */
export function knapsackProgram(values: VectorLike, weights: VectorLike, capacity: number): DynamicProgram {
  const { v, w } = readItems(values, weights, capacity, 'knapsack')
  return {
    shape: [v.length + 1, capacity + 1],
    cell: (i, c, get) => {
      if (i === 0) return { value: 0, choice: 0 }
      const skip = get(i - 1, c)
      const wi = w[i - 1]
      if (wi <= c) {
        const take = get(i - 1, c - wi) + v[i - 1]
        if (take > skip) return { value: take, choice: 1 }
      }
      return { value: skip, choice: 0 }
    },
  }
}

/** The result of `knapsack` and `unboundedKnapsack`. */
export interface KnapsackResult {
  /** The best total value. */
  value: number
  /** Total weight of the chosen items. */
  weight: number
  /** 1 for each chosen item, else 0 (0/1 knapsack); how many of each item (unbounded knapsack); int32, length n. */
  take: Tensor
  /** The DP table: $(n + 1) \times (C + 1)$ for 0/1, $C + 1$ for unbounded. */
  table: Tensor
  /**
   * The traceback through the table: the cells visited, from the last, shape $[n + 1, 2]$ (rows $(i, c)$) for 0/1, or
   * $[k]$ (the capacities, down to 0) for unbounded.
   */
  path: Tensor
}

/**
 * The 0/1 knapsack problem: choose items (each at most once) of integer weights $w_i$ and values $v_i$ to maximise
 * the total value within the integer capacity $C$, by dynamic programming in $O(nC)$, with the table and the
 * traceback. Of equally good choices, the one that leaves out later items is returned.
 *
 * @param values The item values $v_i$ ($n$ numbers).
 * @param weights The item weights $w_i$ ($n$ non-negative integers).
 * @param capacity The capacity $C$, a non-negative integer.
 * @returns The best value, its weight, the items taken, the table and the traceback.
 *
 * @example Three items, capacity 5
 * const r = knapsack([60, 100, 120], [1, 2, 3], 5)
 * print('best value =', r.value, ' weight =', r.weight)
 * print('take =', r.take)
 * print('path (i, c) =', r.path)
 *
 * @example Nothing fits
 * const r = knapsack([5, 7], [4, 6], 3)
 * print('value =', r.value, ' take =', r.take)
 */
export function knapsack(values: VectorLike, weights: VectorLike, capacity: number): KnapsackResult {
  const program = knapsackProgram(values, weights, capacity)
  const { w } = readItems(values, weights, capacity, 'knapsack')
  const { table, choice } = dp(program)
  const n = w.length
  const take = new Int32Array(n)
  const path: number[] = []
  let c = capacity
  for (let i = n; i >= 1; i--) {
    path.push(i, c)
    if (at(choice, i, c) === 1) {
      take[i - 1] = 1
      c -= w[i - 1]
    }
  }
  path.push(0, c)
  return {
    value: at(table, n, capacity),
    weight: capacity - c,
    take: intVector(take),
    table,
    path: intTensor(path, [path.length / 2, 2]),
  }
}

/**
 * The unbounded knapsack table as a dynamic program: cell $c$ is the best value within capacity $c$ when items may
 * repeat, $T_c = \max(T_{c-1}, \max_i T_{c-w_i} + v_i)$ over the items with $0 < w_i \le c$, with $T_0 = 0$; the
 * choice is the index of the item added, or $-1$ when $T_c = T_{c-1}$ (a unit of capacity left unused). An item of
 * weight 0 and positive value makes the problem unbounded and throws `DomainError`.
 *
 * @param values The item values $v_i$ ($n$ numbers).
 * @param weights The item weights $w_i$ ($n$ non-negative integers).
 * @param capacity The capacity $C$, a non-negative integer.
 * @returns The program: its shape $[C + 1]$ and its cell rule, for `dp`.
 *
 * @example The table and the choices, cell by cell
 * const program = unboundedKnapsackProgram([10, 40, 50, 70], [1, 3, 4, 5], 8)
 * const T = []
 * const choice = []
 * for (let c = 0; c < program.shape[0]; c++) {
 *   const cell = program.cell(c, 0, (k) => T[k])
 *   T.push(cell.value)
 *   choice.push(cell.choice)
 * }
 * print('T =', T)
 * print('choice =', choice)
 */
export function unboundedKnapsackProgram(values: VectorLike, weights: VectorLike, capacity: number): DynamicProgram {
  const { v, w } = readItems(values, weights, capacity, 'unboundedKnapsack')
  for (let i = 0; i < w.length; i++)
    if (w[i] === 0 && v[i] > 0)
      throw new DomainError(
        'unboundedKnapsack',
        'unboundedKnapsack: an item of weight 0 and positive value is unbounded',
      )
  return {
    shape: [capacity + 1],
    cell: (c, _j, get) => {
      if (c === 0) return { value: 0, choice: -1 }
      let best = get(c - 1, 0)
      let choice = -1
      for (let i = 0; i < v.length; i++) {
        if (w[i] === 0 || w[i] > c) continue
        const option = get(c - w[i], 0) + v[i]
        if (option > best) {
          best = option
          choice = i
        }
      }
      return { value: best, choice }
    },
  }
}

/**
 * The unbounded knapsack problem (each item may be taken any number of times), by dynamic programming in $O(nC)$,
 * with the table and the traceback. The reported weight is that of the items taken, which may be below $C$.
 *
 * @param values The item values $v_i$ ($n$ numbers).
 * @param weights The item weights $w_i$ ($n$ non-negative integers); one of weight 0 must not have a positive value.
 * @param capacity The capacity $C$, a non-negative integer.
 * @returns The best value, its weight, how many of each item are taken, the table and the capacities visited by the
 *   traceback.
 *
 * @example Repeats allowed
 * const r = unboundedKnapsack([5, 12], [2, 3], 9)
 * print('best value =', r.value, ' weight =', r.weight)
 * print('how many of each =', r.take)
 * print('capacities visited =', r.path)
 * print('0/1 best, for comparison =', knapsack([5, 12], [2, 3], 9).value)
 */
export function unboundedKnapsack(values: VectorLike, weights: VectorLike, capacity: number): KnapsackResult {
  const program = unboundedKnapsackProgram(values, weights, capacity)
  const { w } = readItems(values, weights, capacity, 'unboundedKnapsack')
  const { table, choice } = dp(program)
  const take = new Int32Array(w.length)
  const path: number[] = []
  let c = capacity
  while (c > 0) {
    path.push(c)
    const i = choice.data[c]
    if (i < 0) c -= 1
    else {
      take[i]++
      c -= w[i]
    }
  }
  path.push(c)
  let weight = 0
  take.forEach((k, i) => (weight += k * w[i]))
  return { value: table.data[capacity], weight, take: intVector(take), table, path: intVector(path) }
}
