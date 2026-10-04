/**
 * Sequence dynamic programmes on the `dp` engine: longest common subsequence, edit distance (Wagner and Fischer, 1974,
 * J. ACM 21(1)) with its operations and their counts, and global and local alignment with a linear gap penalty
 * (Needleman and Wunsch, 1970, J. Mol. Biol. 48(3); Smith and Waterman, 1981, J. Mol. Biol. 147(1)), each with its
 * table and traceback. Each `…Program` is the problem for `dynamicProgram` (so a figure can step the table); the plain
 * function solves it. Word and character error rates (`aifn-methods/evaluation/text`) count their edits here.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import { dp, type DynamicProgram } from './dp'
import { intTensor } from './input'

const at = (t: Tensor, i: number, j: number) => t.data[t.offset + i * t.strides[0] + j * (t.strides[1] ?? 0)]

/** A sequence: a string (compared by character) or an array of comparable items (compared with ===). */
export type Sequence<T> = string | readonly T[]

const itemsOf = <T>(s: Sequence<T>): readonly (T | string)[] => (typeof s === 'string' ? Array.from(s) : s)

/** Traceback moves, used as choices: diagonal (match or substitution), up (item of a only), left (item of b only). */
export const DIAGONAL = 0

export const UP = 1

export const LEFT = 2

/** Local alignment only: the alignment starts here (the cell is 0). */
export const STOP = 3

/** Longest common subsequence as a dynamic program; table (|a| + 1) × (|b| + 1). */
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
  length: number
  /** The subsequence: a string when both inputs are strings, else an array. */
  subsequence: T[] | string
  /** Matched positions, shape [length, 2]: rows (index in a, index in b); int32. */
  pairs: Tensor
  table: Tensor
  /** The traceback path through the table from (|a|, |b|) to (0, 0), shape [k, 2]; int32. */
  path: Tensor
}

/** Walk the choice table back from (i, j) until `stop(i, j, move)` or the origin; returns the cells visited. */
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

const pairsTensor = (cells: number[]) => intTensor(cells, [cells.length / 2, 2])

/** The longest common subsequence of two sequences, by dynamic programming in O(|a||b|), with the traceback. */
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
  insert?: number
  delete?: number
  substitute?: number
}

/**
 * Edit distance as a dynamic program (Wagner and Fischer, 1974): cell (i, j) is the cost of turning a[0:i] into
 * b[0:j], D[i][j] = min(D[i−1][j−1] + [aᵢ ≠ bⱼ]·substitute, D[i−1][j] + delete, D[i][j−1] + insert).
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

/** One edit operation, with positions in a (i) and b (j). */
export type EditOperation =
  { op: 'match' | 'substitute'; i: number; j: number } | { op: 'delete'; i: number } | { op: 'insert'; j: number }

/** The result of `editDistance`. */
export interface EditDistanceResult {
  distance: number
  /** The operations turning a into b, in order. */
  operations: EditOperation[]
  /** How many of each operation the alignment uses (hits, substitutions, deletions, insertions in WER terms). */
  counts: { match: number; substitute: number; delete: number; insert: number }
  table: Tensor
  /** The traceback path from (|a|, |b|) to (0, 0), shape [k, 2]; int32. */
  path: Tensor
}

/**
 * The edit (Levenshtein) distance between two sequences with its operations and their counts; costs default to 1. On
 * ties the traceback prefers a match or substitution, then a deletion, then an insertion.
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

/** Scores for sequence alignment with a linear gap penalty (a gap of length k scores k · gap). */
export interface AlignmentScoring<T> {
  /** Score of aligning two equal items (default 1). */
  match?: number
  /** Score of aligning two different items (default −1). */
  mismatch?: number
  /** Score of each gap position (default −1). */
  gap?: number
  /** A substitution score s(x, y) used instead of match and mismatch (e.g. from BLOSUM62). */
  score?: (x: T | string, y: T | string) => number
}

/**
 * Sequence alignment as a dynamic program. `global` is Needleman–Wunsch: F[i][j] = max(F[i−1][j−1] + s(aᵢ, bⱼ),
 * F[i−1][j] + gap, F[i][j−1] + gap) with gap-filled borders. `local` is Smith–Waterman: the same with 0 as a fourth
 * option (the alignment may start anywhere) and zero borders.
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
  score: number
  /** The aligned sequences with gaps: strings with `-` when the inputs are strings, else arrays with null. */
  alignedA: string | (T | null)[]
  alignedB: string | (T | null)[]
  /** The aligned region: a[start[0]:end[0]] against b[start[1]:end[1]] (the whole sequences for global alignment). */
  start: readonly [number, number]
  end: readonly [number, number]
  /** Score table, (|a| + 1) × (|b| + 1). */
  table: Tensor
  /** Traceback choices (DIAGONAL, UP, LEFT, STOP), int32, same shape. */
  choice: Tensor
  /** Traceback path from the end cell back to the start cell, shape [k, 2]; int32. */
  path: Tensor
}

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

/** Global alignment of two sequences (Needleman and Wunsch, 1970) with a linear gap penalty, with the traceback. */
export function needlemanWunsch<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
): AlignmentResult<T> {
  return align(a, b, scoring, 'global')
}

/** Local alignment of two sequences (Smith and Waterman, 1981) with a linear gap penalty, with the traceback. */
export function smithWaterman<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
): AlignmentResult<T> {
  return align(a, b, scoring, 'local')
}
