/**
 * Sorting orders, ranks with a choice of tie policy, and the rank correlations of Spearman and Kendall, matching
 * `numpy.argsort` (stable) and `scipy.stats.rankdata`, `spearmanr` and `kendalltau`. Every function takes arrays or
 * rank-1 tensors.
 */

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

/**
 * The stable sorting order of a sequence.
 *
 * @param x The values; not modified.
 * @returns The indices that sort `x` ascending, tied values in their order of appearance.
 */
function argsortOf(x: ArrayLike<number>): Int32Array {
  const order = Int32Array.from({ length: x.length }, (_, i) => i)
  // Array.prototype.sort is stable; a typed-array sort with a comparator is too, since ES2019.
  return order.sort((a, b) => x[a] - x[b] || a - b)
}

/**
 * Indices that sort $\xvec$ ascending, stable (ties keep their order of appearance), as an int32 rank-1 tensor. As
 * `np.argsort(x, kind='stable')`.
 *
 * @param xData The values: an array or a rank-1 tensor.
 * @returns The 0-based indices of the values in ascending order.
 *
 * @example The order of a sequence with a tie
 * const x = [3, 1, 4, 1, 5]
 * const order = argsort(x)
 * print('order =', order)
 * print('sorted =', take(tensor(x), order))
 */
export function argsort(xData: Data): Tensor {
  return vectorOf(argsortOf(toSequence(xData, 'argsort')))
}

/**
 * Ranks of $\xvec$ from 1 (smallest) to $n$, with ties resolved by `ties` (default `average`). Matches
 * `scipy.stats.rankdata(x, method=...)`. NaN values throw `DomainError`.
 *
 * @param xData The values: an array or a rank-1 tensor.
 * @param ties How tied values are ranked (see `TiePolicy`).
 * @returns The rank of each value, in the order of the data (float64, as average ranks can be halves).
 *
 * @example The five tie policies on one tie
 * const x = [3, 1, 4, 1, 5]
 * print('average =', ranks(x))
 * print('min =', ranks(x, 'min'))
 * print('max =', ranks(x, 'max'))
 * print('dense =', ranks(x, 'dense'))
 * print('ordinal =', ranks(x, 'ordinal'))
 */
export function ranks(xData: Data, ties: TiePolicy = 'average'): Tensor {
  return vectorOf(ranksOf(toSequence(xData, 'ranks'), ties))
}

/**
 * The ranks of a sequence, as `ranks` describes. Throws `DomainError` when a value is NaN.
 *
 * @param x The values.
 * @param ties How tied values are ranked.
 * @returns The rank of each value, from 1, in the order of `x`.
 */
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

/**
 * Spearman's rank correlation $\rho$: Pearson's correlation of the average ranks (`scipy.stats.spearmanr`). NaN when
 * either variable is constant; throws `ShapeError` when the lengths differ and `DomainError` for NaN or empty data.
 *
 * @param xData The first variable: an array or a rank-1 tensor.
 * @param yData The second variable, of the same length.
 * @returns $\rho$, in $[-1, 1]$.
 *
 * @example A monotone curve has rank correlation 1
 * const x = [1, 2, 3, 4]
 * const y = [1, 8, 27, 64]
 * print('Spearman =', spearman(x, y))
 * print('Pearson =', correlation(x, y))
 */
export function spearman(xData: Data, yData: Data): number {
  const x = toSequence(xData, 'spearman')
  const y = toSequence(yData, 'spearman')
  requireSameLength(x, y, 'spearman')
  return correlation(ranksOf(x, 'average'), ranksOf(y, 'average'))
}

/**
 * Kendall's $\tau_b$, which corrects for ties in either variable (Kendall 1945):
 * $\tau_b = (C - D) / \sqrt{(n_0 - n_1)(n_0 - n_2)}$, with $C$ and $D$ the concordant and discordant pairs,
 * $n_0 = n(n - 1)/2$, and $n_1$, $n_2$ the pairs tied in $\xvec$ and in $\yvec$. Without ties it equals $\tau_a$.
 * Matches `scipy.stats.kendalltau` (variant 'b'). This direct count is $O(n^2)$, which is fine for figure-sized data;
 * Knight's $O(n \log n)$ algorithm gives the same value. NaN when either variable is constant (or has fewer than two
 * values); throws `ShapeError` when the lengths differ.
 *
 * @param xData The first variable: an array or a rank-1 tensor.
 * @param yData The second variable, of the same length.
 * @returns $\tau_b$, in $[-1, 1]$.
 *
 * @example With a tie in y, beside Spearman's rho
 * const x = [1, 2, 3, 4, 5]
 * const y = [5, 6, 7, 8, 7]
 * print('Kendall tau_b =', kendallTau(x, y))
 * print('Spearman =', spearman(x, y))
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
