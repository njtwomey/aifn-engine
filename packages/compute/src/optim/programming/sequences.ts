/**
 * Sequence dynamic programmes on the `dp` engine: longest common subsequence, edit distance (Wagner and Fischer,
 * 1974, J. ACM 21(1)) with its operations and their counts, and global and local alignment with a linear gap penalty
 * (Needleman and Wunsch, 1970, J. Mol. Biol. 48(3); Smith and Waterman, 1981, J. Mol. Biol. 147(1)), each with its
 * table and traceback. Each `…Program` is the problem for `dynamicProgram` (so a figure can step the table); the plain
 * function solves it. Word and character error rates (`aifn-methods/evaluation/text`) count their edits here.
 *
 * Every table is $(\lvert a \rvert + 1) \times (\lvert b \rvert + 1)$: cell $(i, j)$ is about the prefixes of
 * $i$ items of `a` and $j$ items of `b`. Each cell records the move that produced it (`DIAGONAL`, `UP`, `LEFT` or
 * `STOP`), and the traceback follows those moves back. All run in $O(\lvert a \rvert \lvert b \rvert)$ time.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import { dp, type DynamicProgram } from './dp'
import { intTensor } from './input'

/**
 * The entry in row $i$ and column $j$ of a table tensor, through its offset and strides.
 *
 * @param t The table, of rank 2 (a rank-1 tensor reads with $j$ ignored).
 * @param i The row.
 * @param j The column.
 * @returns The entry.
 */
const at = (t: Tensor, i: number, j: number) => t.data[t.offset + i * t.strides[0] + j * (t.strides[1] ?? 0)]

/** A sequence: a string (compared by character) or an array of comparable items (compared with `===`). */
export type Sequence<T> = string | readonly T[]

/**
 * The items of a sequence: a string's characters (split by code point), or the array itself.
 *
 * @param s The sequence.
 * @returns Its items, in order.
 */
const itemsOf = <T>(s: Sequence<T>): readonly (T | string)[] => (typeof s === 'string' ? Array.from(s) : s)

/** Traceback moves, used as choices: diagonal (match or substitution), up (item of a only), left (item of b only). */
export const DIAGONAL = 0

/** Traceback move up: an item of `a` against nothing (a deletion, or a gap in `b`). */
export const UP = 1

/** Traceback move left: an item of `b` against nothing (an insertion, or a gap in `a`). */
export const LEFT = 2

/**
 * The traceback ends here: the origin cell of every table, and in local alignment every cell whose score is 0 (the
 * alignment starts there).
 */
export const STOP = 3

/**
 * Longest common subsequence as a dynamic program: cell $(i, j)$ is the length of the longest common subsequence of
 * the first $i$ items of `a` and the first $j$ of `b`, $L_{ij} = L_{i-1,j-1} + 1$ when the items match and
 * $\max(L_{i-1,j}, L_{i,j-1})$ otherwise (`UP` on ties).
 *
 * @param a The first sequence.
 * @param b The second sequence.
 * @returns The program, with a table of $(\lvert a \rvert + 1) \times (\lvert b \rvert + 1)$ cells.
 *
 * @example Fill the table one row at a time
 * const tr = trace(dynamicProgram(lcsProgram('ABCB', 'BCB')), {}, 10)
 * print('after three rows:', tr.steps[3].table)
 * print('filled:', tr.steps.at(-1).table)
 */
export function lcsProgram<T>(a: Sequence<T>, b: Sequence<T>): DynamicProgram {
  const x = itemsOf(a)
  const y = itemsOf(b)
  return {
    shape: [x.length + 1, y.length + 1],
    cell: (i, j, get) => {
      if (i === 0 || j === 0) return { value: 0, choice: i === 0 ? (j === 0 ? STOP : LEFT) : UP }
      if (x[i - 1] === y[j - 1]) return { value: get(i - 1, j - 1) + 1, choice: DIAGONAL }
      const up = get(i - 1, j)
      const left = get(i, j - 1)
      return up >= left ? { value: up, choice: UP } : { value: left, choice: LEFT }
    },
  }
}

/** The result of `lcs`. */
export interface LCSResult<T> {
  /** The length of the longest common subsequence. */
  length: number
  /** The subsequence: a string when both inputs are strings, else an array. */
  subsequence: T[] | string
  /** Matched positions, shape `[length, 2]`: rows (index in `a`, index in `b`); int32. */
  pairs: Tensor
  /** The table of `lcsProgram`, $(\lvert a \rvert + 1) \times (\lvert b \rvert + 1)$. */
  table: Tensor
  /**
   * The traceback path through the table from $(\lvert a \rvert, \lvert b \rvert)$ to $(0, 0)$, shape `[k, 2]`;
   * int32.
   */
  path: Tensor
}

/**
 * Walk the choice table back from $(i_0, j_0)$ until the origin or a `STOP` move, and in local mode also a cell with
 * no choice; returns the cells visited, the last one included.
 *
 * @param choice The choice table of a sequence program, as `dp` returns it.
 * @param i0 The row the walk starts at.
 * @param j0 The column the walk starts at.
 * @param local Whether to stop as well at a cell with no recorded choice (negative).
 * @returns `cells`, the visited cells as flat (row, column) pairs from the start, and `moves`, the move taken out of
 *   each cell but the last.
 */
function traceback(choice: Tensor, i0: number, j0: number, local: boolean): { cells: number[]; moves: number[] } {
  const cells: number[] = []
  const moves: number[] = []
  let i = i0
  let j = j0
  for (;;) {
    cells.push(i, j)
    const move = at(choice, i, j)
    if ((i === 0 && j === 0) || move === STOP || (local && move < 0)) break
    moves.push(move)
    if (move === DIAGONAL) {
      i--
      j--
    } else if (move === UP) i--
    else j--
  }
  return { cells, moves }
}

/**
 * Flat (row, column) pairs as an int32 tensor of shape `[k, 2]`.
 *
 * @param cells The pairs, flattened: $2k$ values.
 * @returns The tensor, one pair per row.
 */
const pairsTensor = (cells: number[]) => intTensor(cells, [cells.length / 2, 2])

/**
 * The longest common subsequence of two sequences, by dynamic programming in $O(\lvert a \rvert \lvert b \rvert)$,
 * with the traceback. Of several longest subsequences, the one the traceback finds is returned (it prefers a move up
 * over a move left on ties).
 *
 * @param a The first sequence.
 * @param b The second sequence.
 * @returns The length, the subsequence, the matched positions, the table and the traceback path.
 *
 * @example The longest common subsequence of two strings
 * const r = lcs('ABCBDAB', 'BDCABA')
 * print('length =', r.length, ' subsequence =', r.subsequence)
 * print('positions (in a, in b) =', r.pairs)
 *
 * @example Arrays of items work too
 * print(lcs([1, 2, 3, 4], [2, 4, 5]).subsequence)
 */
export function lcs<T>(a: Sequence<T>, b: Sequence<T>): LCSResult<T> {
  const x = itemsOf(a)
  const { table, choice } = dp(lcsProgram(a, b))
  const { cells, moves } = traceback(choice, x.length, itemsOf(b).length, false)
  const pairs: number[] = []
  let i = x.length
  let j = itemsOf(b).length
  for (const move of moves) {
    if (move === DIAGONAL) {
      pairs.unshift(i - 1, j - 1)
      i--
      j--
    } else if (move === UP) i--
    else j--
  }
  const items = pairs.filter((_, k) => k % 2 === 0).map((p) => x[p])
  return {
    length: at(table, x.length, itemsOf(b).length),
    subsequence: typeof a === 'string' && typeof b === 'string' ? items.join('') : (items as T[]),
    pairs: pairsTensor(pairs),
    table,
    path: pairsTensor(cells),
  }
}

/** Costs of the edit operations. */
export interface EditCosts {
  /** Cost of inserting an item of `b` (default 1). */
  insert?: number
  /** Cost of deleting an item of `a` (default 1). */
  delete?: number
  /** Cost of replacing an item of `a` by a different item of `b` (default 1); a match costs 0. */
  substitute?: number
}

/**
 * Edit distance as a dynamic program (Wagner and Fischer, 1974): cell $(i, j)$ is the cost $D_{ij}$ of turning the
 * first $i$ items of `a` into the first $j$ of `b`, $D_{ij} = \min(D_{i-1,j-1} + s_{ij}, D_{i-1,j} + d, D_{i,j-1} + e)$
 * with $s_{ij}$ the substitution cost (0 for equal items), $d$ the deletion and $e$ the insertion cost. The choice
 * prefers `DIAGONAL`, then `UP`, then `LEFT` on ties.
 *
 * @param a The sequence edited from.
 * @param b The sequence edited to.
 * @param costs The costs of the operations; each defaults to 1.
 * @returns The program, with a table of $(\lvert a \rvert + 1) \times (\lvert b \rvert + 1)$ cells.
 *
 * @example The table of kitten to sitting
 * const s = dp(editDistanceProgram('kitten', 'sitting'))
 * print('table =', s.table)
 * print('distance =', s.table.data[s.table.data.length - 1])
 */
export function editDistanceProgram<T>(a: Sequence<T>, b: Sequence<T>, costs: EditCosts = {}): DynamicProgram {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const ins = costs.insert ?? 1
  const del = costs.delete ?? 1
  const sub = costs.substitute ?? 1
  return {
    shape: [x.length + 1, y.length + 1],
    cell: (i, j, get) => {
      if (i === 0 && j === 0) return { value: 0, choice: STOP }
      if (i === 0) return { value: get(0, j - 1) + ins, choice: LEFT }
      if (j === 0) return { value: get(i - 1, 0) + del, choice: UP }
      const diag = get(i - 1, j - 1) + (x[i - 1] === y[j - 1] ? 0 : sub)
      const up = get(i - 1, j) + del
      const left = get(i, j - 1) + ins
      if (diag <= up && diag <= left) return { value: diag, choice: DIAGONAL }
      return up <= left ? { value: up, choice: UP } : { value: left, choice: LEFT }
    },
  }
}

/**
 * One edit operation, with positions in `a` (`i`) and `b` (`j`): a `match` or `substitute` pairs `a[i]` with `b[j]`,
 * a `delete` removes `a[i]`, and an `insert` adds `b[j]`.
 */
export type EditOperation =
  { op: 'match' | 'substitute'; i: number; j: number } | { op: 'delete'; i: number } | { op: 'insert'; j: number }

/** The result of `editDistance`. */
export interface EditDistanceResult {
  /** The least total cost of turning `a` into `b`. */
  distance: number
  /** The operations turning `a` into `b`, in order. */
  operations: EditOperation[]
  /** How many of each operation the alignment uses (hits, substitutions, deletions, insertions in WER terms). */
  counts: { match: number; substitute: number; delete: number; insert: number }
  /** The table of `editDistanceProgram`, $(\lvert a \rvert + 1) \times (\lvert b \rvert + 1)$. */
  table: Tensor
  /** The traceback path from $(\lvert a \rvert, \lvert b \rvert)$ to $(0, 0)$, shape `[k, 2]`; int32. */
  path: Tensor
}

/**
 * The edit (Levenshtein) distance between two sequences with its operations and their counts; costs default to 1. On
 * ties the traceback prefers a match or substitution, then a deletion, then an insertion.
 *
 * @param a The sequence edited from.
 * @param b The sequence edited to.
 * @param costs The costs of insertion, deletion and substitution; each defaults to 1.
 * @returns The distance, the operations of one cheapest edit, their counts, the table and the traceback path.
 *
 * @example Kitten to sitting takes three edits
 * const r = editDistance('kitten', 'sitting')
 * print('distance =', r.distance)
 * print('counts =', r.counts)
 * print('operations =', r.operations.map((o) => o.op))
 *
 * @example Words as items, as a word error rate counts them
 * const r = editDistance('the cat sat'.split(' '), 'the cat sat down'.split(' '))
 * print('distance =', r.distance, ' counts =', r.counts)
 */
export function editDistance<T>(a: Sequence<T>, b: Sequence<T>, costs: EditCosts = {}): EditDistanceResult {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const { table, choice } = dp(editDistanceProgram(a, b, costs))
  const { cells, moves } = traceback(choice, x.length, y.length, false)
  const operations: EditOperation[] = []
  let i = x.length
  let j = y.length
  for (const move of moves) {
    if (move === DIAGONAL) {
      operations.unshift({ op: x[i - 1] === y[j - 1] ? 'match' : 'substitute', i: i - 1, j: j - 1 })
      i--
      j--
    } else if (move === UP) operations.unshift({ op: 'delete', i: --i })
    else operations.unshift({ op: 'insert', j: --j })
  }
  const counts = { match: 0, substitute: 0, delete: 0, insert: 0 }
  for (const o of operations) counts[o.op]++
  return { distance: at(table, x.length, y.length), operations, counts, table, path: pairsTensor(cells) }
}

/** Scores for sequence alignment with a linear gap penalty (a gap of length $k$ scores $k$ times `gap`). */
export interface AlignmentScoring<T> {
  /** Score of aligning two equal items (default 1). */
  match?: number
  /** Score of aligning two different items (default $-1$). */
  mismatch?: number
  /** Score of each gap position (default $-1$). */
  gap?: number
  /** A substitution score $s(x, y)$ used instead of `match` and `mismatch` (e.g. from BLOSUM62). */
  score?: (x: T | string, y: T | string) => number
}

/**
 * Sequence alignment as a dynamic program. `global` is Needleman–Wunsch:
 * $F_{ij} = \max(F_{i-1,j-1} + s(a_i, b_j), F_{i-1,j} + g, F_{i,j-1} + g)$ ($g$ the gap score) with gap-filled
 * borders. `local` is Smith–Waterman: the same with 0 as a fourth option (the alignment may start anywhere) and zero
 * borders; a cell of score 0 or less is set to 0 with choice `STOP`. On ties the choice prefers `DIAGONAL`, then
 * `UP`, then `LEFT`.
 *
 * @param a The first sequence (the rows of the table).
 * @param b The second sequence (the columns).
 * @param scoring The match, mismatch and gap scores, or a substitution function.
 * @param mode `global` (Needleman–Wunsch) or `local` (Smith–Waterman).
 * @returns The program, with a table of $(\lvert a \rvert + 1) \times (\lvert b \rvert + 1)$ cells.
 *
 * @example Global and local tables of the same pair
 * print('global:', dp(alignmentProgram('ACG', 'AG')).table)
 * print('local:', dp(alignmentProgram('ACG', 'AG', {}, 'local')).table)
 */
export function alignmentProgram<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
  mode: 'global' | 'local' = 'global',
): DynamicProgram {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const gap = scoring.gap ?? -1
  const s = scoring.score ?? ((p, q) => (p === q ? (scoring.match ?? 1) : (scoring.mismatch ?? -1)))
  const local = mode === 'local'
  return {
    shape: [x.length + 1, y.length + 1],
    cell: (i, j, get) => {
      if (i === 0 && j === 0) return { value: 0, choice: STOP }
      if (i === 0) return local ? { value: 0, choice: STOP } : { value: get(0, j - 1) + gap, choice: LEFT }
      if (j === 0) return local ? { value: 0, choice: STOP } : { value: get(i - 1, 0) + gap, choice: UP }
      const diag = get(i - 1, j - 1) + s(x[i - 1], y[j - 1])
      const up = get(i - 1, j) + gap
      const left = get(i, j - 1) + gap
      let value = diag
      let choice = DIAGONAL
      if (up > value) [value, choice] = [up, UP]
      if (left > value) [value, choice] = [left, LEFT]
      if (local && value <= 0) return { value: 0, choice: STOP }
      return { value, choice }
    },
  }
}

/** The result of `needlemanWunsch` and `smithWaterman`. */
export interface AlignmentResult<T> {
  /** The score of the alignment (the table's entry at the end cell). */
  score: number
  /**
   * `a` aligned, with gaps: a string with `-` for a gap when both inputs are strings, else an array with null.
   */
  alignedA: string | (T | null)[]
  /** `b` aligned, with gaps, in the same form as `alignedA` and of the same length. */
  alignedB: string | (T | null)[]
  /**
   * Where the aligned region starts: it is `a.slice(start[0], end[0])` against `b.slice(start[1], end[1])` (the whole
   * sequences for global alignment).
   */
  start: readonly [number, number]
  /** Where the aligned region ends (exclusive); see `start`. */
  end: readonly [number, number]
  /** Score table, $(\lvert a \rvert + 1) \times (\lvert b \rvert + 1)$. */
  table: Tensor
  /** Traceback choices (`DIAGONAL`, `UP`, `LEFT`, `STOP`), int32, same shape. */
  choice: Tensor
  /** Traceback path from the end cell back to the start cell, shape `[k, 2]`; int32. */
  path: Tensor
}

/**
 * Align two sequences and trace the alignment back. Global alignment ends at the last cell; local alignment at the
 * highest-scoring cell (the first in row-major order on ties) and starts where the traceback reaches a `STOP`.
 *
 * @param a The first sequence.
 * @param b The second sequence.
 * @param scoring The match, mismatch and gap scores, or a substitution function.
 * @param mode `global` or `local`.
 * @returns The alignment, as `AlignmentResult`.
 */
function align<T>(a: Sequence<T>, b: Sequence<T>, scoring: AlignmentScoring<T>, mode: 'global' | 'local') {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const { table, choice } = dp(alignmentProgram(a, b, scoring, mode))
  let ei = x.length
  let ej = y.length
  if (mode === 'local') {
    // The local alignment ends at the highest-scoring cell (the first one in row-major order on ties).
    let best = -Infinity
    for (let i = 0; i <= x.length; i++)
      for (let j = 0; j <= y.length; j++)
        if (at(table, i, j) > best) {
          best = at(table, i, j)
          ei = i
          ej = j
        }
  }
  const { cells, moves } = traceback(choice, ei, ej, mode === 'local')
  const ga: (T | string | null)[] = []
  const gb: (T | string | null)[] = []
  let i = ei
  let j = ej
  for (const move of moves) {
    if (move === DIAGONAL) {
      ga.unshift(x[--i])
      gb.unshift(y[--j])
    } else if (move === UP) {
      ga.unshift(x[--i])
      gb.unshift(null)
    } else {
      ga.unshift(null)
      gb.unshift(y[--j])
    }
  }
  const strings = typeof a === 'string' && typeof b === 'string'
  const show = (g: (T | string | null)[]) => (strings ? g.map((v) => v ?? '-').join('') : (g as (T | null)[]))
  return {
    score: at(table, ei, ej),
    alignedA: show(ga),
    alignedB: show(gb),
    start: [i, j] as const,
    end: [ei, ej] as const,
    table,
    choice,
    path: pairsTensor(cells),
  }
}

/**
 * Global alignment of two sequences (Needleman and Wunsch, 1970) with a linear gap penalty, with the traceback.
 *
 * @param a The first sequence.
 * @param b The second sequence.
 * @param scoring The scores: match 1, mismatch $-1$ and gap $-1$ by default.
 * @returns The score, both sequences aligned with gaps, the table and the traceback.
 *
 * @example Align two DNA strings end to end
 * const r = needlemanWunsch('GATTACA', 'GCATGCU')
 * print('score =', r.score)
 * print(r.alignedA)
 * print(r.alignedB)
 */
export function needlemanWunsch<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
): AlignmentResult<T> {
  return align(a, b, scoring, 'global')
}

/**
 * Local alignment of two sequences (Smith and Waterman, 1981) with a linear gap penalty, with the traceback: the
 * best-scoring pair of segments, one from each sequence.
 *
 * @param a The first sequence.
 * @param b The second sequence.
 * @param scoring The scores: match 1, mismatch $-1$ and gap $-1$ by default.
 * @returns The score, the two segments aligned with gaps, where they lie, the table and the traceback.
 *
 * @example The best-matching segments of two DNA strings
 * const r = smithWaterman('TGTTACGG', 'GGTTGACTA', { match: 3, mismatch: -3, gap: -2 })
 * print('score =', r.score)
 * print(r.alignedA)
 * print(r.alignedB)
 * print('a from', r.start[0], 'to', r.end[0], ' b from', r.start[1], 'to', r.end[1])
 */
export function smithWaterman<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
): AlignmentResult<T> {
  return align(a, b, scoring, 'local')
}
