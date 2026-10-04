/**
 * The matrix profile (Yeh et al. 2016, "Matrix Profile I: all pairs similarity joins for time series", ICDM): for every
 * subsequence of length m, the z-normalised distance to its nearest neighbour elsewhere in the series and that
 * neighbour's index. Matches within the exclusion zone |i − j| ≤ ⌈m/4⌉ (stumpy's default) are trivial and skipped.
 *
 * - **STOMP** (Zhu et al. 2016, "Matrix Profile II", ICDM) computes the distance matrix row by row in O(n²) from one
 *   FFT: the dot products update along each diagonal, QTᵢ,ⱼ = QTᵢ₋₁,ⱼ₋₁ − tᵢ₋₁tⱼ₋₁ + tᵢ₊ₘ₋₁tⱼ₊ₘ₋₁.
 * - **SCRIMP++** (Zhu et al. 2018, "Matrix Profile XI: SCRIMP++", ICDM) is anytime: PreSCRIMP computes the full
 *   distance profiles of every s-th subsequence (s = ⌊m/4⌋) and walks each best match's diagonal s steps either way,
 *   which already finds most motifs; then SCRIMP visits the diagonals in random order, each one exactly, so the
 *   profile only falls and is exact once every diagonal is done.
 *
 * Motifs are the pairs with the smallest profile values; discords the subsequences with the largest (the most unusual
 * shapes), each pick excluding its neighbourhood from later ones.
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
  readonly kind: 'matrix-profile'
  /** The subsequence length. */
  readonly m: Size
  /** Trivial matches |i − j| ≤ exclusion are skipped. */
  readonly exclusion: Size
  /** Nearest-neighbour distance of every subsequence (n − m + 1); Infinity where none exists. */
  readonly profile: Tensor
  /** The nearest neighbour's index (int32, −1 where none). */
  readonly index: Tensor
}

/** Options of the matrix-profile functions. */
export interface MatrixProfileOptions {
  /** The exclusion zone (default ⌈m/4⌉). */
  exclusion?: Size
}

const exclusionOf = (m: number, options: MatrixProfileOptions) => options.exclusion ?? Math.ceil(m / 4)

function prepare(series: SignalInput, m: Size, where: string) {
  const t = readSamples(series, where).values
  checkWindow(m, t.length, where)
  const k = t.length - m + 1
  if (k < 2) throw new DomainError(where, `${where}: the series needs at least two subsequences`)
  const { mean, std } = meanStd(t, m)
  return { t, k, mean, std }
}

const result = (m: number, exclusion: number, P: Float64Array, I: Int32Array): MatrixProfile => ({
  kind: 'matrix-profile',
  m,
  exclusion,
  profile: fromData(Float64Array.from(P)),
  index: fromData(Int32Array.from(I)),
})

/** The matrix profile of a series by STOMP (exact, O(n²)); see the module notes. */
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
  readonly profile: Tensor
  readonly index: Tensor
  /** Diagonals evaluated so far, out of `diagonals`. */
  readonly done: number
  readonly diagonals: number
  /** The diagonal offsets evaluated in the last step. */
  readonly last: readonly number[]
  /** The order in which the diagonals (offsets j − i) are visited. */
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
 * SCRIMP++ as steps (module notes): the initial state is PreSCRIMP's approximate profile; each step evaluates the next
 * batch of diagonals in a random order drawn from the `init` stream; converged once every diagonal is done, when the
 * profile equals {@link matrixProfile}'s.
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
  /** How many to return (default 3). */
  count?: Size
  /** Picks closer than this to an earlier pick are skipped (default: the profile's m, a whole subsequence). */
  exclusion?: Size
}

/** A motif: a pair of subsequences, the closest pair not yet excluded, and their distance. */
export interface Motif {
  readonly a: number
  readonly b: number
  readonly distance: number
}

/** The top motifs of a matrix profile: smallest profile values first, excluding the neighbourhoods of earlier picks. */
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
  readonly at: number
  readonly distance: number
  readonly neighbour: number
}

/** The top discords of a matrix profile: largest finite profile values first, excluding earlier picks' neighbourhoods. */
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

/** The matrix profile held by a SCRIMP++ state. */
export function scrimpProfile(state: ScrimpState, m: Size, options: MatrixProfileOptions = {}): MatrixProfile {
  return {
    kind: 'matrix-profile',
    m,
    exclusion: exclusionOf(m, options),
    profile: state.profile,
    index: state.index,
  }
}
