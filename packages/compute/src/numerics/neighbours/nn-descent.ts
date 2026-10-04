/**
 * Approximate k-nearest neighbours by nearest-neighbour descent (Dong, Charikar and Li, 2011, "Efficient k-nearest
 * neighbor graph construction for generic similarity measures", WWW '11), the method umap-learn uses through
 * pynndescent. The principle: a neighbour of a neighbour is likely a neighbour. Each point starts with k random
 * candidates; every round compares, for each point, the pairs among its new and old neighbours and reverse neighbours
 * (the "local join"), and each pair may enter the other's list. Only pairs with at least one entry new since the last
 * round are compared (the incremental search), and each list is sampled to ⌈ρk⌉ entries. The search stops when a
 * round changes fewer than δ·n·k list entries. Cost is about O(n k² ρ²) distances per round instead of the O(n²) of
 * exact search.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { integers, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { rowsOf } from './search'

/** Options of {@link nearestNeighbourDescent}. */
export interface NearestNeighbourDescentOptions {
  /** The randomness of the initial lists and the sampling. */
  stream: Stream
  /** Sample rate ρ of each round's candidate lists (default 1, umap-learn's choice; Dong et al. use 0.5–1). */
  sampleRate?: number
  /** Stop when a round updates fewer than δ·n·k entries (default 0.001). */
  delta?: number
  /** Most rounds (default 30). */
  maxRounds?: number
}

/** The k nearest other points of each row, nearest first, and the number of rounds run. */
export interface NeighbourLists {
  readonly kind: 'neighbours'
  /** Indices [n, k] (int32). */
  readonly indices: Tensor
  /** Euclidean distances [n, k], matching `indices`. */
  readonly distances: Tensor
  /** Distances computed (the initial lists and every local join). */
  readonly distanceEvaluations: number
  readonly rounds: number
  /** Entries changed in each round. */
  readonly updates: readonly number[]
}

/**
 * The approximate k nearest neighbours (the point itself excluded) of the n rows of x (n × d) under the Euclidean
 * distance, by nearest-neighbour descent (module notes). k must lie in 1 … n − 1.
 */
export function nearestNeighbourDescent(
  x: MatrixLike,
  k: Size,
  options: NearestNeighbourDescentOptions,
): NeighbourLists {
  const { n, d, data: v } = rowsOf(x, 'nearestNeighbourDescent')
  if (!(Number.isInteger(k) && k >= 1 && k < n))
    throw new DomainError('nearestNeighbourDescent', `nearestNeighbourDescent: k must lie in 1 … ${n - 1}`)
  let evaluations = 0
  const { stream: s, sampleRate = 1, delta = 0.001, maxRounds = 30 } = options
  const dist2 = (i: number, j: number) => {
    evaluations++
    let acc = 0
    const a = i * d
    const b = j * d
    for (let c = 0; c < d; c++) {
      const t = v[a + c] - v[b + c]
      acc += t * t
    }
    return acc
  }
  // Each list is kept sorted by distance (k is small); `isNew` marks entries not yet used in a local join.
  const idx = new Int32Array(n * k).fill(-1)
  const dst = new Float64Array(n * k).fill(Infinity)
  const isNew = new Uint8Array(n * k)
  /** Insert j into i's list at squared distance q; 1 when the list changed. */
  const push = (i: number, j: number, q: number): number => {
    const base = i * k
    if (q >= dst[base + k - 1]) return 0
    for (let r = 0; r < k; r++) if (idx[base + r] === j) return 0
    let r = k - 1
    while (r > 0 && dst[base + r - 1] > q) {
      idx[base + r] = idx[base + r - 1]
      dst[base + r] = dst[base + r - 1]
      isNew[base + r] = isNew[base + r - 1]
      r--
    }
    idx[base + r] = j
    dst[base + r] = q
    isNew[base + r] = 1
    return 1
  }
  for (let i = 0; i < n; i++) {
    let filled = 0
    while (filled < k) {
      const j = integers(s, n)
      if (j !== i) filled += push(i, j, dist2(i, j))
    }
  }
  const sample = Math.max(1, Math.ceil(sampleRate * k))
  /** Up to `cap` items of a list, a uniform sample when it is longer (partial Fisher–Yates). */
  const take = (list: number[], cap: number): number[] => {
    if (list.length <= cap) return list
    for (let r = 0; r < cap; r++) {
      const t = r + integers(s, list.length - r)
      ;[list[r], list[t]] = [list[t], list[r]]
    }
    return list.slice(0, cap)
  }
  let rounds = 0
  const history: number[] = []
  while (rounds < maxRounds) {
    rounds++
    const fresh: number[][] = Array.from({ length: n }, () => [])
    const old: number[][] = Array.from({ length: n }, () => [])
    for (let i = 0; i < n; i++) {
      const newSlots: number[] = []
      for (let r = 0; r < k; r++) {
        const slot = i * k + r
        if (idx[slot] < 0) continue
        if (isNew[slot]) newSlots.push(slot)
        else old[i].push(idx[slot])
      }
      // Sampled new entries become old: they take part in this round's join and not in later ones.
      for (const slot of take(newSlots, sample)) {
        isNew[slot] = 0
        fresh[i].push(idx[slot])
      }
    }
    const freshReverse: number[][] = Array.from({ length: n }, () => [])
    const oldReverse: number[][] = Array.from({ length: n }, () => [])
    for (let i = 0; i < n; i++) {
      for (const j of fresh[i]) freshReverse[j].push(i)
      for (const j of old[i]) oldReverse[j].push(i)
    }
    let updates = 0
    for (let i = 0; i < n; i++) {
      const nw = [...new Set([...fresh[i], ...take(freshReverse[i], sample)])]
      const ol = [...new Set([...old[i], ...take(oldReverse[i], sample)])]
      for (let a = 0; a < nw.length; a++) {
        const u = nw[a]
        for (let b = a + 1; b < nw.length; b++) {
          const w = nw[b]
          if (u === w) continue
          const q = dist2(u, w)
          updates += push(u, w, q) + push(w, u, q)
        }
        for (const w of ol) {
          if (u === w) continue
          const q = dist2(u, w)
          updates += push(u, w, q) + push(w, u, q)
        }
      }
    }
    history.push(updates)
    if (updates < delta * n * k) break
  }
  return {
    kind: 'neighbours',
    indices: fromData(idx, [n, k]),
    distances: fromData(Float64Array.from(dst, Math.sqrt), [n, k]),
    distanceEvaluations: evaluations,
    rounds,
    updates: history,
  }
}
