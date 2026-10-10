/**
 * The matrix profile (Yeh et al. 2016, "Matrix Profile I: all pairs similarity joins for time series", ICDM): for every
 * subsequence of length $m$, the z-normalised distance to its nearest neighbour elsewhere in the series and that
 * neighbour's index. Matches within the exclusion zone $\lvert i - j \rvert \le \lceil m/4 \rceil$ (stumpy's
 * default) are trivial and skipped.
 *
 * - **STOMP** (Zhu et al. 2016, "Matrix Profile II", ICDM) computes the distance matrix row by row in $O(n^2)$ from one
 *   FFT: the dot products update along each diagonal,
 *   $QT_{i,j} = QT_{i-1,j-1} - t_{i-1}t_{j-1} + t_{i+m-1}t_{j+m-1}$.
 * - **SCRIMP++** (Zhu et al. 2018, "Matrix Profile XI: SCRIMP++", ICDM) is anytime: PreSCRIMP computes the full
 *   distance profiles of every $s$-th subsequence ($s = \lfloor m/4 \rfloor$, at least 1) and walks each best match's
 *   diagonal $s - 1$ steps either way, which already finds most motifs; then SCRIMP visits the diagonals in random
 *   order, each one exactly, so the profile only falls and is exact once every diagonal is done.
 *
 * Motifs are the pairs with the smallest profile values; discords the subsequences with the largest (the most unusual
 * shapes), each pick excluding its neighbourhood from later ones. Distances between constant subsequences follow
 * `zDistance`.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { readSamples, type SignalInput } from '../signal'
import { checkWindow, meanStd, slidingDotProduct, zDistance } from './profile'

/** A matrix profile. */
export interface MatrixProfile {
  /** The brand of a matrix profile. */
  readonly kind: 'matrix-profile'
  /** The subsequence length $m$. */
  readonly m: Size
  /** The exclusion zone: trivial matches with $\lvert i - j \rvert \le$ `exclusion` are skipped. */
  readonly exclusion: Size
  /** Nearest-neighbour distance of every subsequence ($n - m + 1$ of them); Infinity where none exists. */
  readonly profile: Tensor
  /** The nearest neighbour's index (int32, $-1$ where none). */
  readonly index: Tensor
}

/** Options of the matrix-profile functions. */
export interface MatrixProfileOptions {
  /** The exclusion zone, in samples (default $\lceil m/4 \rceil$). */
  exclusion?: Size
}

/**
 * The exclusion zone to use: the option if given, else $\lceil m/4 \rceil$.
 *
 * @param m The subsequence length.
 * @param options The options, whose `exclusion` overrides the default.
 * @returns The exclusion zone, in samples.
 */
const exclusionOf = (m: number, options: MatrixProfileOptions) => options.exclusion ?? Math.ceil(m / 4)

/**
 * Reads a series and its sliding statistics for a matrix profile. Throws `DomainError` unless $m$ is an integer in
 * $2, \dots, n$ and the series has at least two subsequences.
 *
 * @param series The series: a single-channel signal or its samples.
 * @param m The subsequence length.
 * @param where The caller's name for error messages.
 * @returns The samples `t`, the number of subsequences `k` ($n - m + 1$), and each subsequence's `mean` and `std`.
 */
function prepare(series: SignalInput, m: Size, where: string) {
  const t = readSamples(series, where).values
  checkWindow(m, t.length, where)
  const k = t.length - m + 1
  if (k < 2) throw new DomainError(where, `${where}: the series needs at least two subsequences`)
  const { mean, std } = meanStd(t, m)
  return { t, k, mean, std }
}

/**
 * A `MatrixProfile` from its parts, copying the arrays.
 *
 * @param m The subsequence length.
 * @param exclusion The exclusion zone used.
 * @param P The nearest-neighbour distances.
 * @param I The nearest-neighbour indices ($-1$ where none).
 * @returns The matrix profile.
 */
const result = (m: number, exclusion: number, P: Float64Array, I: Int32Array): MatrixProfile => ({
  kind: 'matrix-profile',
  m,
  exclusion,
  profile: fromData(Float64Array.from(P)),
  index: fromData(Int32Array.from(I)),
})

/**
 * The matrix profile of a series by STOMP (exact, $O(n^2)$; see the file notes). Throws `DomainError` unless $m$ is an
 * integer in $2, \dots, n$ and there are at least two subsequences.
 *
 * @param series The series, of length $n$: a single-channel signal or its samples.
 * @param m The subsequence length.
 * @param options `exclusion`, the exclusion zone (default $\lceil m/4 \rceil$).
 * @returns The profile and index of all $n - m + 1$ subsequences.
 *
 * @example A shape that recurs: its two occurrences point at each other
 * const x = [0, 1, 3, 1, 0, 2, 0, 1, 3, 1, 0, 1, 2, 2]
 * const mp = matrixProfile(x, 4)
 * print('profile =', mp.profile)
 * print('index =', mp.index)
 */
export function matrixProfile(series: SignalInput, m: Size, options: MatrixProfileOptions = {}): MatrixProfile {
  const { t, k, mean, std } = prepare(series, m, 'matrixProfile')
  const ex = exclusionOf(m, options)
  const P = new Float64Array(k).fill(Infinity)
  const I = new Int32Array(k).fill(-1)
  const first = toFlat(slidingDotProduct(t.subarray(0, m), t))
  let qt = Float64Array.from(first)
  for (let i = 0; i < k; i++) {
    if (i > 0) {
      const next = new Float64Array(k)
      next[0] = first[i]
      for (let j = 1; j < k; j++) next[j] = qt[j - 1] - t[i - 1] * t[j - 1] + t[i + m - 1] * t[j + m - 1]
      qt = next
    }
    for (let j = 0; j < k; j++) {
      if (Math.abs(i - j) <= ex) continue
      const d = zDistance(qt[j], m, mean[i], std[i], mean[j], std[j])
      if (d < P[i]) {
        P[i] = d
        I[i] = j
      }
    }
  }
  return result(m, ex, P, I)
}

/** One state of SCRIMP++: the profile so far and the diagonals done. */
export interface ScrimpState extends Status {
  /** The nearest-neighbour distances found so far; each only falls, to the exact value. */
  readonly profile: Tensor
  /** The nearest-neighbour indices found so far ($-1$ where none yet). */
  readonly index: Tensor
  /** Diagonals evaluated so far, out of `diagonals`. */
  readonly done: number
  /** The number of diagonals outside the exclusion zone, all of which are evaluated by convergence. */
  readonly diagonals: number
  /** The diagonal offsets evaluated in the last step. */
  readonly last: readonly number[]
  /** The order in which the diagonals (offsets $j - i$) are visited. */
  readonly order: readonly number[]
}

/** Options of {@link scrimpSteps}. */
export interface ScrimpOptions extends MatrixProfileOptions {
  /** Diagonals per step (default: about 2% of them, at least 1). */
  diagonalsPerStep?: Size
  /** Run PreSCRIMP in the initial state (default true); without it step 0 is the empty profile. */
  prescrimp?: boolean
}

/**
 * SCRIMP++ as steps (see the file notes): the initial state is PreSCRIMP's approximate profile; each step evaluates
 * the next batch of diagonals in a random order drawn from the `init` stream; converged once every diagonal is done,
 * when the profile equals {@link matrixProfile}'s (up to rounding, and to which neighbour a tie picks). Throws
 * `DomainError` as `matrixProfile` does.
 *
 * @param series The series, of length $n$: a single-channel signal or its samples.
 * @param m The subsequence length.
 * @param options `exclusion` (default $\lceil m/4 \rceil$), `diagonalsPerStep` (default about 2% of the diagonals)
 *   and `prescrimp` (default true).
 * @returns The algorithm, to run with `run`; its start is unused.
 *
 * @example The anytime profile on noise: PreSCRIMP is nearly exact, the steps finish the job
 * const x = normals(stream(3), 80)
 * const exact = toArray(matrixProfile(x, 8).profile)
 * const alg = scrimpSteps(x, 8, { diagonalsPerStep: 20 })
 * for (const steps of [0, 1, 2, 3, 4]) {
 *   const s = run(alg, undefined, steps, { stream: stream(1) })
 *   const wrong = toArray(s.profile).filter((v, i) => Math.abs(v - exact[i]) > 1e-9).length
 *   print(`${steps} steps:`, s.done, 'of', s.diagonals, 'diagonals; entries not yet exact:', wrong)
 * }
 */
export function scrimpSteps(series: SignalInput, m: Size, options: ScrimpOptions = {}): Algorithm<void, ScrimpState> {
  const { t, k, mean, std } = prepare(series, m, 'scrimpSteps')
  const ex = exclusionOf(m, options)
  const offsets = Array.from({ length: Math.max(0, k - ex - 1) }, (_, i) => ex + 1 + i)
  const perStep = options.diagonalsPerStep ?? Math.max(1, Math.round(0.02 * offsets.length))
  const dot = (i: number, j: number) => {
    let s = 0
    for (let c = 0; c < m; c++) s += t[i + c] * t[j + c]
    return s
  }
  const dist = (qt: number, i: number, j: number) => zDistance(qt, m, mean[i], std[i], mean[j], std[j])
  const offer = (P: Float64Array, I: Int32Array, i: number, j: number, d: number) => {
    if (d < P[i]) {
      P[i] = d
      I[i] = j
    }
    if (d < P[j]) {
      P[j] = d
      I[j] = i
    }
  }
  /** The exact pass along diagonal j − i = off. */
  const diagonal = (P: Float64Array, I: Int32Array, off: number) => {
    let qt = 0
    for (let i = 0; i + off < k; i++) {
      const j = i + off
      qt = i === 0 ? dot(0, off) : qt - t[i - 1] * t[j - 1] + t[i + m - 1] * t[j + m - 1]
      offer(P, I, i, j, dist(qt, i, j))
    }
  }
  /** PreSCRIMP (Zhu et al. 2018, Algorithm 2): sampled full profiles and short walks along their best diagonals. */
  const prescrimp = (P: Float64Array, I: Int32Array) => {
    const s = Math.max(1, Math.floor(m / 4))
    for (let i = 0; i < k; i += s) {
      const qt = toFlat(slidingDotProduct(t.subarray(i, i + m), t))
      let best = -1
      let bestD = Infinity
      for (let j = 0; j < k; j++) {
        if (Math.abs(i - j) <= ex) continue
        const d = dist(qt[j], i, j)
        offer(P, I, i, j, d)
        if (d < bestD) {
          bestD = d
          best = j
        }
      }
      if (best < 0) continue
      let q = qt[best]
      for (let r = 1; r < s && i + r < k && best + r < k; r++) {
        const a = i + r
        const b = best + r
        q = q - t[a - 1] * t[b - 1] + t[a + m - 1] * t[b + m - 1]
        if (Math.abs(a - b) > ex) offer(P, I, a, b, dist(q, a, b))
      }
      q = qt[best]
      for (let r = 1; r < s && i - r >= 0 && best - r >= 0; r++) {
        const a = i - r
        const b = best - r
        q = q - t[a + m] * t[b + m] + t[a] * t[b]
        if (Math.abs(a - b) > ex) offer(P, I, a, b, dist(q, a, b))
      }
    }
  }
  return {
    name: 'scrimp',
    init: (_start, s: Stream) => {
      const P = new Float64Array(k).fill(Infinity)
      const I = new Int32Array(k).fill(-1)
      if (options.prescrimp !== false) prescrimp(P, I)
      const order = Array.from(toFlat(permutation(child(s, 'order'), offsets.length)), (r) => offsets[r])
      return {
        t: 0,
        profile: fromData(P),
        index: fromData(I),
        done: 0,
        diagonals: offsets.length,
        last: [],
        converged: offsets.length === 0,
        order,
      }
    },
    step: (state) => {
      const P = Float64Array.from(dense.data(state.profile))
      const I = Int32Array.from(dense.data(state.index))
      const batch = state.order.slice(state.done, state.done + perStep)
      for (const off of batch) diagonal(P, I, off)
      const done = state.done + batch.length
      return {
        ...state,
        t: state.t + 1,
        profile: fromData(P),
        index: fromData(I),
        done,
        last: batch,
        converged: done >= offsets.length,
      }
    },
  }
}

/** Options of {@link motifs} and {@link discords}. */
export interface PickOptions {
  /** How many to return at most (default 3); fewer when the exclusions leave no candidate. */
  count?: Size
  /**
   * Picks within this many samples of an earlier pick are skipped (default: the profile's $m$, a whole
   * subsequence).
   */
  exclusion?: Size
}

/** A motif: a pair of subsequences, the closest pair not yet excluded, and their distance. */
export interface Motif {
  /** The start of the earlier subsequence of the pair. */
  readonly a: number
  /** The start of the later subsequence of the pair. */
  readonly b: number
  /** Their z-normalised distance. */
  readonly distance: number
}

/**
 * The top motifs of a matrix profile: smallest profile values first, each paired with its nearest neighbour. A pick
 * excludes the neighbourhoods of both its subsequences from later picks, and a candidate whose neighbour is excluded
 * is passed over.
 *
 * @param mp The matrix profile, from `matrixProfile` or `scrimpProfile`.
 * @param options `count`, the most motifs to return (default 3), and `exclusion`, the neighbourhood radius in samples
 *   (default the profile's $m$).
 * @returns The motifs, closest first.
 *
 * @example A shape repeated in a sine wave
 * const x = Array.from({ length: 40 }, (_, i) => (i === 25 ? 3 : Math.sin((2 * Math.PI * i) / 10)))
 * print('motifs =', motifs(matrixProfile(x, 10), { count: 2 }))
 */
export function motifs(mp: MatrixProfile, options: PickOptions = {}): Motif[] {
  const { count = 3, exclusion = mp.m } = options
  const P = toFlat(mp.profile)
  const I = toFlat(mp.index)
  const banned = new Uint8Array(P.length)
  const out: Motif[] = []
  const ban = (c: number) => {
    for (let j = Math.max(0, c - exclusion); j <= Math.min(P.length - 1, c + exclusion); j++) banned[j] = 1
  }
  while (out.length < count) {
    let best = -1
    for (let i = 0; i < P.length; i++)
      if (!banned[i] && I[i] >= 0 && !banned[I[i]] && Number.isFinite(P[i]) && (best < 0 || P[i] < P[best])) best = i
    if (best < 0) break
    const a = Math.min(best, I[best])
    const b = Math.max(best, I[best])
    out.push({ a, b, distance: P[best] })
    ban(a)
    ban(b)
  }
  return out
}

/** A discord: the subsequence farthest from its nearest neighbour. */
export interface Discord {
  /** The start of the subsequence. */
  readonly at: number
  /** Its distance to its nearest neighbour: the profile value. */
  readonly distance: number
  /** The start of that nearest neighbour. */
  readonly neighbour: number
}

/**
 * The top discords of a matrix profile: largest finite profile values first, each pick excluding its neighbourhood
 * from later ones.
 *
 * @param mp The matrix profile, from `matrixProfile` or `scrimpProfile`.
 * @param options `count`, the most discords to return (default 3), and `exclusion`, the neighbourhood radius in
 *   samples (default the profile's $m$).
 * @returns The discords, most unusual first.
 *
 * @example A spike in a sine wave is the most unusual shape
 * const x = Array.from({ length: 40 }, (_, i) => (i === 25 ? 3 : Math.sin((2 * Math.PI * i) / 10)))
 * print('discord =', discords(matrixProfile(x, 10), { count: 1 }))
 */
export function discords(mp: MatrixProfile, options: PickOptions = {}): Discord[] {
  const { count = 3, exclusion = mp.m } = options
  const P = toFlat(mp.profile)
  const I = toFlat(mp.index)
  const banned = new Uint8Array(P.length)
  const out: Discord[] = []
  while (out.length < count) {
    let best = -1
    for (let i = 0; i < P.length; i++) if (!banned[i] && Number.isFinite(P[i]) && (best < 0 || P[i] > P[best])) best = i
    if (best < 0) break
    out.push({ at: best, distance: P[best], neighbour: I[best] })
    for (let j = Math.max(0, best - exclusion); j <= Math.min(P.length - 1, best + exclusion); j++) banned[j] = 1
  }
  return out
}

/**
 * The matrix profile held by a SCRIMP++ state, so that `motifs` and `discords` can read an anytime result. The arrays
 * are shared with the state, not copied.
 *
 * @param state A state of `scrimpSteps`, converged or not.
 * @param m The subsequence length the state was computed with.
 * @param options `exclusion`, the zone the state was computed with (default $\lceil m/4 \rceil$); only recorded.
 * @returns The matrix profile.
 *
 * @example The discord of a finished SCRIMP++ run
 * const x = Array.from({ length: 40 }, (_, i) => (i === 25 ? 3 : Math.sin((2 * Math.PI * i) / 10)))
 * const s = run(scrimpSteps(x, 10), undefined, 100, { stream: stream(1) })
 * print('converged after', s.t, 'steps:', s.converged)
 * print('discord =', discords(scrimpProfile(s, 10), { count: 1 }))
 */
export function scrimpProfile(state: ScrimpState, m: Size, options: MatrixProfileOptions = {}): MatrixProfile {
  return {
    kind: 'matrix-profile',
    m,
    exclusion: exclusionOf(m, options),
    profile: state.profile,
    index: state.index,
  }
}
