/**
 * Knapsacks on the `dp` engine of `aifn-compute/optim/programming` (whose `sequences` hold the longest common subsequence,
 * edit distance and alignments).
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { type DynamicProgram, dp } from 'aifn-compute/optim/programming'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const at = (t: Tensor, i: number, j: number) => t.data[t.offset + i * t.strides[0] + j * (t.strides[1] ?? 0)]

// ---------------------------------------------------------------------------------------------------------------------
// Knapsack.

/** A vector input as float64, of length `n` when given. */
function readVector(v: VectorLike, where: string, n?: number): Float64Array {
  const out = dense.toF64(v, where)
  if (n !== undefined && out.length !== n)
    throw new ShapeError(where, `${where}: expected ${n} entries, got ${out.length}`)
  return out
}

/** An int32 tensor of the given shape (a vector by default) from integers. */
const intTensor = (values: ArrayLike<number>, shape: readonly number[] = [values.length]): Tensor =>
  fromData(Int32Array.from(values), shape)
const intVector = (values: ArrayLike<number>): Tensor => intTensor(values)

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
 * The 0/1 knapsack table as a dynamic program: cell (i, c) is the best value from the first i items within capacity c,
 * T[i][c] = max(T[i−1][c], T[i−1][c − wᵢ] + vᵢ); the choice is 1 when item i is taken. Shape (n + 1) × (C + 1).
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

/** The result of `knapsack`. */
export interface KnapsackResult {
  /** The best total value. */
  value: number
  /** Total weight of the chosen items. */
  weight: number
  /** 1 for each chosen item, else 0 (0/1 knapsack); how many of each item (unbounded knapsack); int32, length n. */
  take: Tensor
  /** The DP table: (n + 1) × (C + 1) for 0/1, C + 1 for unbounded. */
  table: Tensor
  /** The traceback through the table: the cells visited, shape [k, 2] (rows (i, c)) or [k] (capacities). */
  path: Tensor
}

/**
 * The 0/1 knapsack problem: choose items (each at most once) of integer weights wᵢ and values vᵢ to maximise total
 * value within integer capacity C, by dynamic programming in O(nC), with the table and the traceback.
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
 * The unbounded knapsack table as a dynamic program: cell c is the best value within capacity c when items may repeat,
 * T[c] = max(T[c − 1], maxᵢ T[c − wᵢ] + vᵢ); the choice is the item added, or −1 when T[c] = T[c − 1].
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

/** The unbounded knapsack problem (each item may be taken any number of times), with the table and the traceback. */
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
