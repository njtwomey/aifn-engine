/** Beyond-accuracy metrics of recommendation lists: intra-list diversity, catalogue coverage, Gini and Herfindahl concentration, novelty. */

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
 * Intra-list diversity (Ziegler et al. 2005; beyond-accuracy-metrics): the mean dissimilarity over the k(k − 1)/2
 * pairs of a list, from a k × k matrix of pairwise dissimilarities (only the upper triangle is read).
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

/** Catalogue coverage: the fraction of a catalogue of `catalogueSize` items that appears in at least one list. */
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
 * The Gini coefficient of exposure counts, 2Σᵢ i·x₍ᵢ₎/(MΣx) − (M + 1)/M with x sorted increasing: 0 for equal exposure,
 * approaching 1 when one item takes it all.
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

/** The Herfindahl index Σ sᵢ² of exposure shares: 1/M for equal exposure over M items, 1 for one item. */
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
 * Novelty: the mean self-information −log₂ p(d) of a list's items, with p(d) the fraction of users who interacted with
 * item d (Vargas and Castells 2011). In bits.
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
