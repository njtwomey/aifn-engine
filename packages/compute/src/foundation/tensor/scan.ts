/**
 * The associative scan (parallel prefix): every prefix combination y_k = x_0 ⊕ x_1 ⊕ … ⊕ x_k of a sequence under an
 * associative operation ⊕ (Blelloch, 1990, "Prefix sums and their applications"). Running sums, running products,
 * running maxima and the linear recurrence h_t = a_t·h_{t−1} + b_t (the core of state-space models and linear
 * attention) are all scans.
 *
 * - `associativeScan(op, elems, options)`: the scan as a composition of primitives (strided slices, `op` on whole
 *   tensors, `stack` and `reshape`), after JAX's `lax.associative_scan`: the odd–even recursion of Ladner and Fischer
 *   (1980), O(n) work in O(log n) rounds, each round one call of `op` on a whole tensor. Being a composition, it
 *   differentiates, batches and nests like any other function of primitives. Elements may be tuples of tensors (a
 *   pytree level), which `op` combines pairwise.
 * - `hillisSteeleScanSteps` and `blellochScanSteps`: the two textbook parallel scans as step-through algorithms, one
 *   round per step, for exposition (Hillis and Steele, 1986; Blelloch, 1990). Their results equal `associativeScan`.
 */

import type { Algorithm, Size, Status, Value } from 'aifn-compute/foundation/contracts'
import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import type { Tensor } from './core'
import { gather } from './gather'
import { concat, permute, reshape, shapeOfValue, slice, stack } from './structure'
import { unwrap } from './trace'

/** A scan element: one tensor, or a tuple of tensors sharing the scanned axis. */
export type ScanElement = Value | readonly Value[]

/** Options of `associativeScan`. */
export type ScanOptions = {
  /** The scanned axis (default 0). */
  axis?: number
  /** Scan from the end: y_k = x_k ⊕ x_{k+1} ⊕ … ⊕ x_{n−1} (the operand order is kept). Default false. */
  reverse?: boolean
}

type Leaves = Value[]

const leavesOf = (e: ScanElement): Leaves => (Array.isArray(e) ? [...(e as readonly Value[])] : [e as Value])

/** The length of the leading axis of every leaf (they must agree). */
function lengthOf(xs: Leaves, where: string): Size {
  let n = -1
  for (const x of xs) {
    const s = shapeOfValue(x)
    if (s.length === 0) throw new ShapeError(where, `${where}: elements need rank ≥ 1 (the scanned axis)`)
    if (n >= 0 && s[0] !== n) throw new ShapeError(where, `${where}: leaves disagree on the scanned length`)
    n = s[0]
  }
  return n
}

/** Even and odd positions interleaved: [e0, o0, e1, o1, …]; `even` has as many entries as `odd` or one more. */
function interleave(even: Value, odd: Value): Value {
  const ne = shapeOfValue(even)[0]
  const no = shapeOfValue(odd)[0]
  const rest = shapeOfValue(odd).slice(1)
  if (no === 0) return even
  const pairs = reshape(stack([slice(even, [0, no]), odd], 1), [2 * no, ...rest])
  return ne > no ? concat([pairs, slice(even, [no, ne])], 0) : pairs
}

/** The inclusive scan along axis 0 by the odd–even recursion. */
function scanFront(op: (a: Leaves, b: Leaves) => Leaves, xs: Leaves): Leaves {
  const n = lengthOf(xs, 'associativeScan')
  if (n < 2) return xs
  // Combine neighbours (x0 ⊕ x1, x2 ⊕ x3, …) and scan those: the results are the scans at the odd positions.
  const pairsLeft = xs.map((x) => slice(x, [0, n - 1, 2]))
  const pairsRight = xs.map((x) => slice(x, [1, n, 2]))
  const odd = scanFront(op, op(pairsLeft, pairsRight))
  const m = lengthOf(odd, 'associativeScan')
  // The even positions after the first: y_{2i} = y_{2i−1} ⊕ x_{2i}.
  const evensAfter = n > 2 ? Math.ceil(n / 2) - 1 : 0
  const even =
    evensAfter === 0
      ? xs.map((x) => slice(x, [0, 1]))
      : op(
          odd.map((o) => slice(o, [0, Math.min(m, evensAfter)])),
          xs.map((x) => slice(x, [2, n, 2])),
        ).map((r, j) => concat([slice(xs[j], [0, 1]), r], 0))
  return even.map((e, j) => interleave(e, odd[j]))
}

/**
 * The inclusive scan of `elems` along `axis` under the associative operation `op`: y_k = x_0 ⊕ x_1 ⊕ … ⊕ x_k, with
 * `op(a, b)` = a ⊕ b applied to whole tensors of elements (it must broadcast over the leading axis, as elementwise
 * operations do). Elements are a tensor or a tuple of tensors (then `op` takes and returns tuples). With `reverse`,
 * y_k = x_k ⊕ … ⊕ x_{n−1}.
 *
 * A composition of primitives (strided slices, `op`, `stack`, `reshape`), so its gradient is exact through `grad`, and
 * `op` may itself be any differentiable function. O(n) applications of ⊕ in ⌈log₂ n⌉ rounds; the result equals the
 * sequential scan up to rounding, and exactly when ⊕ is exact.
 *
 * @example associativeScan(add, x)                          // cumulative sum
 * @example associativeScan(([a1, b1], [a2, b2]) => [mul(a1, a2), add(mul(a2, b1), b2)], [a, b]) // h_t = a_t h_{t−1} + b_t
 */
export function associativeScan<E extends ScanElement>(op: (a: E, b: E) => E, elems: E, options: ScanOptions = {}): E {
  const tuple = Array.isArray(elems)
  let xs = leavesOf(elems)
  if (xs.length === 0) throw new AifnError('associativeScan', 'associativeScan: no elements')
  const rank = shapeOfValue(xs[0]).length
  const axis = options.axis ?? 0
  const a = axis < 0 ? axis + rank : axis
  if (a < 0 || a >= rank) throw new ShapeError('associativeScan', `associativeScan: axis ${axis} out of range`)
  const order = a === 0 ? null : [a, ...Array.from({ length: rank }, (_, k) => k).filter((k) => k !== a)]
  const back = order === null ? null : order.map((_, k) => order.indexOf(k))
  if (order) xs = xs.map((x) => permute(x, order))
  if (options.reverse) xs = xs.map((x) => slice(x, [null, null, -1]))
  const wrap = (v: Leaves): E => (tuple ? (v as unknown as E) : (v[0] as E))
  const leafOp = (p: Leaves, q: Leaves): Leaves =>
    leavesOf(options.reverse ? op(wrap(q), wrap(p)) : op(wrap(p), wrap(q)))
  let ys = scanFront(leafOp, xs)
  if (options.reverse) ys = ys.map((y) => slice(y, [null, null, -1]))
  if (back) ys = ys.map((y) => permute(y, back))
  return wrap(ys)
}

// ── Step-through scans ───────────────────────────────────────────────────────────────────────────────────────────────

/** An update of one round: position `to` becomes op(value at `from`, value at `to`) (or a copy for Blelloch's swap). */
export type ScanMove = { readonly from: number; readonly to: number }

/** The state of `hillisSteeleScanSteps` after t rounds. */
export interface HillisSteeleState extends Status {
  /** The partial scans: position i holds x_{max(0, i − 2^t + 1)} ⊕ … ⊕ x_i. */
  readonly values: Tensor
  /** The distance combined in the next round, 2^t. */
  readonly offset: Size
  /** The combinations made by the last round (none at t = 0). */
  readonly moves: readonly ScanMove[]
  /** Applications of ⊕ so far (the work: n log n in total). */
  readonly work: Size
  readonly terminated: boolean
}

/** Rows `ids` of x along axis 0 (a gather, so differentiable). */
function rows(x: Value, ids: readonly number[]): Value {
  const s = shapeOfValue(x)
  const w = s.slice(1).reduce((p, q) => p * q, 1)
  const flat = new Int32Array(ids.length * w)
  ids.forEach((r, i) => {
    for (let j = 0; j < w; j++) flat[i * w + j] = r * w + j
  })
  return gather(x, flat, [ids.length, ...s.slice(1)])
}

/** x with rows `to` replaced by the rows of `values` (in order), by one gather from their concatenation. */
function replaceRows(x: Value, to: readonly number[], values: Value): Tensor {
  const n = shapeOfValue(x)[0]
  const source = Array.from({ length: n }, (_, i) => i)
  to.forEach((r, j) => (source[r] = n + j))
  return unwrap(rows(concat([x, values], 0), source)) as Tensor
}

const asTensor = (x: Value): Tensor => unwrap(x) as Tensor

/**
 * The Hillis–Steele scan (Hillis and Steele, 1986) as a step-through algorithm: in round t every position i ≥ 2^t
 * combines with position i − 2^t at once, so after ⌈log₂ n⌉ rounds every position holds its inclusive prefix. It does
 * n⌈log₂ n⌉ − (2^⌈log₂ n⌉ − 1) applications of ⊕ (more than the n − 1 of a sequential scan) but has the fewest rounds.
 * `op` combines whole tensors of elements along axis 0, as in `associativeScan`; the final `values` equal
 * `associativeScan(op, x)`.
 */
export function hillisSteeleScanSteps(
  op: (a: Value, b: Value) => Value,
  x: Tensor,
): Algorithm<void, HillisSteeleState> {
  const n = lengthOf([x], 'hillisSteeleScanSteps')
  return {
    name: 'hillis-steele-scan',
    init: () => ({ t: 0, values: x, offset: 1, moves: [], work: 0, terminated: n <= 1 }),
    step: (state) => {
      const { values, offset } = state
      const to = Array.from({ length: n - offset }, (_, i) => i + offset)
      const combined = op(slice(values, [0, n - offset]), slice(values, [offset, n]))
      const next = asTensor(concat([slice(values, [0, offset]), combined], 0))
      return {
        t: state.t + 1,
        values: next,
        offset: 2 * offset,
        moves: to.map((i) => ({ from: i - offset, to: i })),
        work: state.work + to.length,
        terminated: 2 * offset >= n,
      }
    },
  }
}

/** The phase of `blellochScanSteps`. */
export type BlellochPhase = 'up-sweep' | 'clear' | 'down-sweep' | 'done'

/** The state of `blellochScanSteps`. */
export interface BlellochState extends Status {
  /** The working array, padded with the identity to a power of two P. */
  readonly values: Tensor
  /** The phase the next round belongs to. */
  readonly phase: BlellochPhase
  /** The tree level of the next round (0 = the leaves' parents). */
  readonly level: Size
  /** The updates of the last round: `to` combined with `from` (up-sweep) or swapped with it (down-sweep). */
  readonly moves: readonly ScanMove[]
  /** Applications of ⊕ so far (the work: about 2P). */
  readonly work: Size
  /** The exclusive scan x_0 ⊕ … ⊕ x_{k−1} (identity at k = 0), once done. */
  readonly exclusive: Tensor | null
  /** The inclusive scan, exclusive ⊕ x, once done: equal to `associativeScan(op, x)`. */
  readonly inclusive: Tensor | null
  readonly terminated: boolean
}

/**
 * Blelloch's work-efficient scan (Blelloch, 1990) as a step-through algorithm. The input is padded to a power of two
 * P with `identity` (an element e with e ⊕ x = x ⊕ e = x, one row: shape [1, ...] or broadcastable to it). The
 * up-sweep builds a balanced tree of partial reductions in log₂ P rounds (the root holds the total); the root is set
 * to the identity; the down-sweep pushes prefixes back down in log₂ P rounds, each left child receiving its parent's
 * prefix and each right child the parent's prefix ⊕ its left sibling's sum. About 2P applications of ⊕ in 2 log₂ P
 * rounds. The result is the exclusive scan; the final step also forms the inclusive one.
 */
export function blellochScanSteps(
  op: (a: Value, b: Value) => Value,
  x: Tensor,
  identity: Value,
): Algorithm<void, BlellochState> {
  const n = lengthOf([x], 'blellochScanSteps')
  const levels = Math.max(0, Math.ceil(Math.log2(Math.max(1, n))))
  const P = 2 ** levels
  const rest = x.shape.slice(1)
  const width = rest.reduce((p, q) => p * q, 1)
  const idRow = asTensor(reshape(concat([reshape(identity, [-1])], 0), [1, ...rest]))
  if (idRow.shape.reduce((p, q) => p * q, 1) !== width)
    throw new ShapeError('blellochScanSteps', 'blellochScanSteps: the identity must have one element’s shape')
  const padding = P - n
  const padded = padding === 0 ? x : asTensor(concat([x, rows(idRow, Array(padding).fill(0))], 0))
  /** Positions of the left and right child roots at tree level d: (k + 2^d − 1, k + 2^{d+1} − 1). */
  const pairsAt = (d: number) => {
    const left: number[] = []
    const right: number[] = []
    for (let k = 0; k < P; k += 2 ** (d + 1)) {
      left.push(k + 2 ** d - 1)
      right.push(k + 2 ** (d + 1) - 1)
    }
    return { left, right }
  }
  const finish = (state: BlellochState, values: Tensor, moves: ScanMove[], work: Size): BlellochState => {
    const exclusive = asTensor(slice(values, [0, n]))
    return {
      ...state,
      t: state.t + 1,
      values,
      phase: 'done',
      moves,
      work: work + n,
      exclusive,
      inclusive: asTensor(op(exclusive, x)),
      terminated: true,
    }
  }
  return {
    name: 'blelloch-scan',
    init: () => ({
      t: 0,
      values: padded,
      phase: levels === 0 ? 'clear' : 'up-sweep',
      level: 0,
      moves: [],
      work: 0,
      exclusive: null,
      inclusive: null,
      terminated: false,
    }),
    step: (state) => {
      const { values, level } = state
      if (state.phase === 'up-sweep') {
        const { left, right } = pairsAt(level)
        const next = replaceRows(values, right, op(rows(values, left), rows(values, right)))
        const up = level + 1 < levels
        return {
          ...state,
          t: state.t + 1,
          values: next,
          phase: up ? 'up-sweep' : 'clear',
          level: up ? level + 1 : levels - 1,
          moves: left.map((l, i) => ({ from: l, to: right[i] })),
          work: state.work + left.length,
        }
      }
      if (state.phase === 'clear') {
        const next = replaceRows(values, [P - 1], idRow)
        if (levels === 0) return finish(state, next, [], state.work)
        return { ...state, t: state.t + 1, values: next, phase: 'down-sweep', level: levels - 1, moves: [] }
      }
      // Down-sweep: left ← parent prefix (held at right); right ← parent prefix ⊕ left sum.
      const { left, right } = pairsAt(level)
      const leftSum = rows(values, left)
      const prefix = rows(values, right)
      const next = replaceRows(values, [...left, ...right], concat([prefix, op(prefix, leftSum)], 0))
      const moves = left.map((l, i) => ({ from: right[i], to: l }))
      const work = state.work + left.length
      if (level === 0) return finish(state, next, moves, work)
      return { ...state, t: state.t + 1, values: next, level: level - 1, moves, work }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const algorithm = definer<AlgorithmInfo>('algorithm', 'foundation/tensor')

algorithm(
  {
    key: 'hillisSteeleScanSteps',
    name: 'Hillis–Steele scan',
    summary: 'The inclusive parallel prefix in ⌈log₂ n⌉ rounds, each position combining with the one 2^t before it.',
    problem: 'sequence',
    state: { iterate: 'values', flags: ['terminated'] },
    notes: ['parallel-scan'],
    cite: ['blelloch1990'],
  },
  hillisSteeleScanSteps,
)
algorithm(
  {
    key: 'blellochScanSteps',
    name: 'Blelloch scan',
    summary: 'The work-efficient parallel prefix: an up-sweep of partial reductions, then a down-sweep of prefixes.',
    problem: 'sequence',
    state: { iterate: 'values', flags: ['terminated'] },
    notes: ['parallel-scan'],
    cite: ['blelloch1990'],
  },
  blellochScanSteps,
)

/** The step-through scans, keyed by factory name. */
export const scanAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', { hillisSteeleScanSteps, blellochScanSteps }) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
