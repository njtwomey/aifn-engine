import type { Tensor } from 'aifn-compute/foundation/tensor'
import { toSequence, vectorOf, type Data } from './input'
import { correlation, requireSameLength } from './descriptive'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * How `ranks` treats ties, as in `scipy.stats.rankdata`:
 * - `average` (default, the aifn convention): tied values share the mean of the ranks they span.
 * - `min` / `max`: the lowest / highest rank they span (competition ranking).
 * - `dense`: consecutive ranks with no gaps after ties.
 * - `ordinal`: distinct ranks in order of appearance.
 */
export type TiePolicy = 'average' | 'min' | 'max' | 'dense' | 'ordinal'

function argsortOf(x: ArrayLike<number>): Int32Array {
  const order = Int32Array.from({ length: x.length }, (_, i) => i)
  // Array.prototype.sort is stable; a typed-array sort with a comparator is too, since ES2019.
  return order.sort((a, b) => x[a] - x[b] || a - b)
}

/** Indices that sort x ascending, stable (ties keep their order of appearance), as an int32 rank-1 tensor. */
export function argsort(xData: Data): Tensor {
  return vectorOf(argsortOf(toSequence(xData, 'argsort')))
}

/**
 * Ranks of x from 1 (smallest) to n, with ties resolved by `ties` (default `average`). Matches
 * `scipy.stats.rankdata(x, method=...)`. NaN values are not allowed.
 */
export function ranks(xData: Data, ties: TiePolicy = 'average'): Tensor {
  return vectorOf(ranksOf(toSequence(xData, 'ranks'), ties))
}

function ranksOf(x: ArrayLike<number>, ties: TiePolicy): Float64Array {
  const n = x.length
  for (let i = 0; i < n; i++)
    if (Number.isNaN(x[i])) throw new DomainError('stats', 'stats: ranks of data containing NaN')
  const order = argsortOf(x)
  const out = new Float64Array(n)
  let dense = 0
  for (let start = 0; start < n;) {
    let end = start + 1
    while (end < n && x[order[end]] === x[order[start]]) end++
    dense++
    // Positions start … end − 1 (0-based) hold one tied value; their 1-based ranks are start + 1 … end.
    for (let k = start; k < end; k++) {
      const i = order[k]
      out[i] =
        ties === 'average'
          ? (start + 1 + end) / 2
          : ties === 'min'
            ? start + 1
            : ties === 'max'
              ? end
              : ties === 'dense'
                ? dense
                : k + 1
    }
    start = end
  }
  return out
}

/** Spearman's rank correlation ρ: Pearson's correlation of the average ranks (`scipy.stats.spearmanr`). */
export function spearman(xData: Data, yData: Data): number {
  const x = toSequence(xData, 'spearman')
  const y = toSequence(yData, 'spearman')
  requireSameLength(x, y, 'spearman')
  return correlation(ranksOf(x, 'average'), ranksOf(y, 'average'))
}

/**
 * Kendall's τ_b, which corrects for ties in either variable (Kendall 1945):
 * τ_b = (C − D) / √((n₀ − n₁)(n₀ − n₂)), with C and D the concordant and discordant pairs, n₀ = n(n − 1)/2, and n₁,
 * n₂ the pairs tied in x and in y. Without ties it equals τ_a. Matches `scipy.stats.kendalltau` (variant 'b'). This
 * direct count is O(n²), which is fine for figure-sized data; Knight's O(n log n) algorithm gives the same value.
 * NaN when either variable is constant.
 */
export function kendallTau(xData: Data, yData: Data): number {
  const x = toSequence(xData, 'kendallTau')
  const y = toSequence(yData, 'kendallTau')
  requireSameLength(x, y, 'kendallTau')
  const n = x.length
  let concordant = 0
  let discordant = 0
  let tiedX = 0
  let tiedY = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = Math.sign(x[i] - x[j])
      const dy = Math.sign(y[i] - y[j])
      if (dx === 0) tiedX++
      if (dy === 0) tiedY++
      if (dx !== 0 && dy !== 0) {
        if (dx === dy) concordant++
        else discordant++
      }
    }
  }
  const pairs = (n * (n - 1)) / 2
  const denominator = Math.sqrt((pairs - tiedX) * (pairs - tiedY))
  return denominator === 0 ? NaN : (concordant - discordant) / denominator
}
