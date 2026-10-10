/**
 * Beyond-accuracy metrics of recommendation lists: intra-list diversity, catalogue coverage, Gini and Herfindahl
 * concentration of exposure, and novelty.
 *
 * They score what a list of recommendations looks like rather than whether it was right: how varied a list is, how
 * much of the catalogue the lists reach, how evenly exposure is spread over items, and how unpopular the recommended
 * items are. Each is a metric of the `aifn-compute/learning/metrics` registry (with `info` for its range and
 * direction), and takes precomputed quantities (dissimilarities, item ids, exposure counts, popularities) rather than
 * a model.
 */

import {
  defineMetric,
  denseMatrix as dense,
  divide,
  metricValues as values,
  type Data,
  type Rows,
} from 'aifn-compute/learning/metrics'

// ── Beyond accuracy ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Intra-list diversity (Ziegler et al. 2005; beyond-accuracy-metrics): the mean dissimilarity
 * $\frac{2}{k(k-1)} \sum_{i<j} d_{ij}$ over the $k(k-1)/2$ pairs of a list of $k$ items. NaN for a list of one item.
 *
 * @param dissimilarities The $k \times k$ matrix of pairwise dissimilarities $d_{ij}$ between the list's items (rows
 *   or a tensor). Only the strict upper triangle is read, so the diagonal and lower triangle may hold anything.
 * @returns The mean pairwise dissimilarity, in the units of $d_{ij}$.
 *
 * @example Three items, one of them far from the other two
 * const d = [
 *   [0, 0.2, 0.9],
 *   [0.2, 0, 0.8],
 *   [0.9, 0.8, 0],
 * ]
 * print('intra-list diversity =', intraListDiversity(d))
 */
export const intraListDiversity = defineMetric(
  {
    module: 'applied/evaluation/beyond-accuracy',
    key: 'intraListDiversity',
    name: 'Intra-list diversity',
    inputs: 'vectors',
    direction: 'higher',
    range: [0, Infinity],
    notes: ['beyond-accuracy-metrics'],
  },
  (dissimilarities: Rows): number => {
    const d = dense(dissimilarities, 'intraListDiversity')
    const k = d.rows
    let s = 0
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) s += d.data[i * k + j]
    return divide(2 * s, k * (k - 1))
  },
)

/**
 * Catalogue coverage: the fraction of a catalogue that appears in at least one recommendation list.
 *
 * @param lists The recommendation lists, each an array of item ids (numbers or strings). An item in several lists, or
 *   several times in one, counts once.
 * @param options `catalogueSize`, the number of items in the catalogue, which divides the count of distinct items.
 * @returns The number of distinct recommended items over `catalogueSize`, in $[0, 1]$ when every id is in the
 *   catalogue.
 *
 * @example Three lists over a catalogue of ten items
 * const lists = [
 *   ['a', 'b', 'c'],
 *   ['a', 'c', 'd'],
 *   ['b', 'e', 'a'],
 * ]
 * print('coverage =', catalogueCoverage(lists, { catalogueSize: 10 }))
 */
export const catalogueCoverage = defineMetric(
  {
    module: 'applied/evaluation/beyond-accuracy',
    key: 'catalogueCoverage',
    name: 'Catalogue coverage',
    inputs: 'exposure',
    direction: 'higher',
    range: [0, 1],
    notes: ['beyond-accuracy-metrics'],
  },
  (lists: ArrayLike<ArrayLike<number | string>>, options: { catalogueSize: number }): number => {
    const seen = new Set<number | string>()
    for (let q = 0; q < lists.length; q++) for (let i = 0; i < lists[q].length; i++) seen.add(lists[q][i])
    return seen.size / options.catalogueSize
  },
)

/**
 * The Gini coefficient of exposure counts over $M$ items, $\frac{2\sum_i i\, x_{(i)}}{M \sum_i x_i} - \frac{M+1}{M}$
 * with $x_{(1)} \le \dots \le x_{(M)}$ sorted increasing: 0 for equal exposure, and $(M-1)/M$, approaching 1, when
 * one item takes it all. NaN when every count is 0.
 *
 * @param exposure The exposure of each item (how often it was recommended), one non-negative value per item, in any
 *   order. It is copied before sorting.
 * @returns The Gini coefficient, in $[0, 1)$.
 *
 * @example Equal, skewed and concentrated exposure
 * print('equal:', giniCoefficient([5, 5, 5, 5]))
 * print('skewed:', giniCoefficient([1, 2, 3, 10]))
 * print('one item:', giniCoefficient([0, 0, 0, 20]))
 */
export const giniCoefficient = defineMetric(
  {
    module: 'applied/evaluation/beyond-accuracy',
    key: 'giniCoefficient',
    name: 'Gini coefficient of exposure',
    inputs: 'exposure',
    direction: 'lower',
    range: [0, 1],
    notes: ['beyond-accuracy-metrics'],
  },
  (exposure: Data): number => {
    const x = values(exposure).sort()
    const M = x.length
    let weighted = 0
    let total = 0
    x.forEach((v, i) => {
      weighted += (i + 1) * v
      total += v
    })
    return divide(2 * weighted, M * total) - (M + 1) / M
  },
)

/**
 * The Herfindahl index $\sum_i s_i^2$ of the exposure shares $s_i = x_i / \sum_j x_j$: $1/M$ for equal exposure over
 * $M$ items, 1 when one item takes it all. NaN when every count is 0.
 *
 * @param exposure The exposure of each item (how often it was recommended), one non-negative value per item.
 * @returns The sum of squared shares, in $[1/M, 1]$.
 *
 * @example Equal exposure over four items, and one dominant item
 * print('equal:', herfindahlIndex([5, 5, 5, 5]))
 * print('dominant:', herfindahlIndex([1, 1, 1, 17]))
 */
export const herfindahlIndex = defineMetric(
  {
    module: 'applied/evaluation/beyond-accuracy',
    key: 'herfindahlIndex',
    name: 'Herfindahl index',
    inputs: 'exposure',
    direction: 'lower',
    range: [0, 1],
    notes: ['beyond-accuracy-metrics'],
  },
  (exposure: Data): number => {
    const x = values(exposure)
    const total = x.reduce((a, b) => a + b, 0)
    return x.reduce((s, v) => s + (v / total) ** 2, 0)
  },
)

/**
 * Novelty (Vargas and Castells 2011): the mean self-information $-\log_2 p(d)$ of a list's items, with $p(d)$ the
 * fraction of users who interacted with item $d$. In bits; an item nobody interacted with ($p(d) = 0$) makes it
 * infinite.
 *
 * @param popularity The popularity $p(d) \in (0, 1]$ of each recommended item: the fraction of users who interacted
 *   with it, one value per item of the list.
 * @returns The mean of $-\log_2 p(d)$ over the list, in bits.
 *
 * @example A list of popular items, and a list of niche ones
 * print('popular:', novelty([0.5, 0.4, 0.25]))
 * print('niche:', novelty([0.01, 0.02, 0.05]))
 */
export const novelty = defineMetric(
  {
    module: 'applied/evaluation/beyond-accuracy',
    key: 'novelty',
    name: 'Novelty',
    inputs: 'exposure',
    direction: 'higher',
    range: [0, Infinity],
    notes: ['beyond-accuracy-metrics'],
  },
  (popularity: Data): number => {
    const p = values(popularity)
    return p.reduce((s, v) => s - Math.log2(v), 0) / p.length
  },
)
