/**
 * Clustering metrics. External scores compare a predicted partition with a reference through their contingency table:
 * pair counting (Rand, adjusted Rand, Fowlkes–Mallows) and information theory (mutual information, NMI with four
 * normalisations, AMI, homogeneity, completeness, V-measure, variation of information), in nats as scikit-learn.
 * Internal indices judge a partition by the data's geometry: silhouette, Calinski–Harabasz, Davies–Bouldin, Dunn.
 *
 * External scores take two labellings of the same $n$ items (numbers, strings or booleans, compared by value), so they
 * do not depend on the names of the clusters; throughout, $C$ is the reference partition with class sizes $a_i$, $K$
 * the predicted one with cluster sizes $b_j$, and $n_{ij}$ the items in class $i$ and cluster $j$. Internal indices
 * take the data as an $n \times d$ matrix with a cluster label per row, use Euclidean distances, and throw
 * `DomainError` for fewer than two clusters.
 */

import { logFactorial } from 'aifn-compute/numerics/special'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { pairwiseDistances } from 'aifn-compute/numerics/linalg'
import {
  classesOf,
  defineMetric,
  dense,
  divide,
  encodeLabels,
  labelList,
  matrix,
  sameLength,
  vector,
  type Label,
  type Labels,
  type Rows,
} from './core'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A contingency table of two labellings, as `contingencyTable` returns it. */
export type Contingency = {
  /**
   * The $R \times C$ counts: `table[i][j]` counts the items in class $i$ of the first labelling and cluster $j$ of the
   * second.
   */
  table: Tensor
  /** Row sums $a_i$ (sizes of the first labelling's classes). */
  rows: Tensor
  /** Column sums $b_j$ (sizes of the second labelling's clusters). */
  cols: Tensor
  /** The number of items $n$. */
  n: number
  /** The first labelling's classes in label order, one per row of `table`. */
  rowLabels: Label[]
  /** The second labelling's clusters in label order, one per column of `table`. */
  colLabels: Label[]
}

/**
 * A contingency table as working arrays: `c`, the $R \times C$ counts $n_{ij}$ row-major; `a`, the $R$ row sums
 * $a_i$; `b`, the $C$ column sums $b_j$; `R` and `C`, the numbers of classes and clusters; `n`, the number of items.
 */
type Table = { c: Float64Array; a: Float64Array; b: Float64Array; R: number; C: number; n: number }

/**
 * Count two labellings into a contingency table, rows and columns in label order. Throws `ShapeError` when the
 * labellings differ in length.
 *
 * @param labelsTrue The reference labelling, one label per item: the rows.
 * @param labelsPred The predicted labelling of the same items: the columns.
 * @returns The counts and their margins.
 */
function tableOf(labelsTrue: Labels, labelsPred: Labels): Table {
  const t = labelList(labelsTrue)
  const p = labelList(labelsPred)
  sameLength(t, p, 'clustering metric')
  const rl = classesOf(t)
  const cl = classesOf(p)
  const ti = encodeLabels(t, rl)
  const pi = encodeLabels(p, cl)
  const R = rl.length
  const C = cl.length
  const c = new Float64Array(R * C)
  const a = new Float64Array(R)
  const b = new Float64Array(C)
  for (let i = 0; i < t.length; i++) {
    c[ti[i] * C + pi[i]]++
    a[ti[i]]++
    b[pi[i]]++
  }
  return { c, a, b, R, C, n: t.length }
}

/**
 * The contingency table of two labellings of the same $n$ items (rows: `labelsTrue`, columns: `labelsPred`), as
 * sklearn's `contingency_matrix`. Throws `ShapeError` when the labellings differ in length.
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @returns The counts with their margins and the labels of their rows and columns.
 *
 * @example Class 1 split into two clusters
 * const t = contingencyTable([0, 0, 1, 1], ['a', 'a', 'b', 'c'])
 * print('table', t.table)
 * print('class sizes', t.rows, 'cluster sizes', t.cols)
 * print('columns', t.colLabels)
 */
export function contingencyTable(labelsTrue: Labels, labelsPred: Labels): Contingency {
  const t = tableOf(labelsTrue, labelsPred)
  return {
    table: matrix(t.c, t.R, t.C),
    rows: vector(t.a),
    cols: vector(t.b),
    n: t.n,
    rowLabels: classesOf(labelList(labelsTrue)),
    colLabels: classesOf(labelList(labelsPred)),
  }
}

/**
 * The number of unordered pairs among $m$ items, $\binom{m}{2} = m(m - 1)/2$.
 *
 * @param m The number of items (a count, or a cell of a contingency table).
 * @returns The number of pairs.
 */
const pairs = (m: number) => (m * (m - 1)) / 2

/**
 * Pair counts over the $\binom{n}{2}$ unordered pairs of items: `tp`, pairs together in both labellings; `fn`,
 * together only in the truth; `fp`, together only in the prediction; `tn`, apart in both. sklearn's
 * `pair_confusion_matrix` counts ordered pairs, so each of its counts is twice these.
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @returns The four pair counts, summing to $\binom{n}{2}$.
 *
 * @example Splitting a class loses one pair
 * print(pairConfusion([0, 0, 1, 1], [0, 0, 1, 2]))
 */
export function pairConfusion(
  labelsTrue: Labels,
  labelsPred: Labels,
): { tp: number; fp: number; fn: number; tn: number } {
  const t = tableOf(labelsTrue, labelsPred)
  let tp = 0
  for (const v of t.c) tp += pairs(v)
  const together = t.a.reduce((s, v) => s + pairs(v), 0)
  const predicted = t.b.reduce((s, v) => s + pairs(v), 0)
  const fn = together - tp
  const fp = predicted - tp
  return { tp, fp, fn, tn: pairs(t.n) - tp - fp - fn }
}

/**
 * The registry metadata of a partition-comparison metric: stable, read from two `partitions`, higher is better.
 *
 * @param key The metric's registry key (its export name).
 * @param name The metric's display name.
 * @param note The key of the note that explains it.
 * @param range The metric's range of values.
 * @returns The metadata, with its literal fields kept.
 */
const partitionInfo = (key: string, name: string, note: string, range: readonly [number, number] = [0, 1]) =>
  ({
    key,
    name,
    stability: 'stable',
    inputs: 'partitions',
    direction: 'higher',
    range,
    notes: [note],
    capability: 'decide',
  }) as const

/**
 * The Rand index $(\mathrm{TP} + \mathrm{TN})/\binom{n}{2}$, the accuracy of the pairwise "together or apart"
 * decisions (Rand 1971), as sklearn's `rand_score`; 1 when there are no pairs.
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @returns The Rand index, in $[0, 1]$.
 *
 * @example Five of the six pairs agree
 * print('Rand', randIndex([0, 0, 1, 1], [0, 0, 1, 2]))
 * print('relabelled', randIndex([0, 0, 1, 1], [1, 1, 0, 0]))
 */
export const randIndex = defineMetric(
  partitionInfo('randIndex', 'Rand index', 'rand-index-and-adjusted-rand-index'),
  (labelsTrue: Labels, labelsPred: Labels): number => {
    const p = pairConfusion(labelsTrue, labelsPred)
    const total = p.tp + p.fp + p.fn + p.tn
    return total === 0 ? 1 : (p.tp + p.tn) / total
  },
)

/**
 * The adjusted Rand index (Hubert and Arabie 1985):
 * $(\mathrm{Index} - \expect[\mathrm{Index}])/(\mathrm{Max} - \expect[\mathrm{Index}])$ with
 * $\mathrm{Index} = \sum_{ij} \binom{n_{ij}}{2}$,
 * $\expect[\mathrm{Index}] = \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} / \binom{n}{2}$ and $\mathrm{Max}$ the mean
 * of the two marginal sums. 1 for identical partitions (also when both are trivial), about 0 for random ones. As
 * sklearn's `adjusted_rand_score`.
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @returns The adjusted Rand index, in $[-\tfrac{1}{2}, 1]$.
 *
 * @example Chance-corrected, so lower than the Rand index
 * print('ARI', adjustedRandIndex([0, 0, 1, 1], [0, 0, 1, 2]))
 * print('Rand', randIndex([0, 0, 1, 1], [0, 0, 1, 2]))
 * print('identical up to names', adjustedRandIndex([0, 0, 1, 1], ['b', 'b', 'a', 'a']))
 */
export const adjustedRandIndex = defineMetric(
  partitionInfo('adjustedRandIndex', 'Adjusted Rand index', 'rand-index-and-adjusted-rand-index', [-0.5, 1]),
  (labelsTrue: Labels, labelsPred: Labels): number => {
    const t = tableOf(labelsTrue, labelsPred)
    let index = 0
    for (const v of t.c) index += pairs(v)
    const sa = t.a.reduce((s, v) => s + pairs(v), 0)
    const sb = t.b.reduce((s, v) => s + pairs(v), 0)
    const expected = divide(sa * sb, pairs(t.n), 0)
    const max = (sa + sb) / 2
    return max === expected ? 1 : (index - expected) / (max - expected)
  },
)

/**
 * The Fowlkes–Mallows index $\mathrm{TP}/\sqrt{(\mathrm{TP} + \mathrm{FP})(\mathrm{TP} + \mathrm{FN})}$, the geometric
 * mean of pairwise precision and recall. NaN when either labelling puts every item in a cluster of its own (sklearn
 * returns 0).
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @returns The Fowlkes–Mallows index, in $[0, 1]$.
 *
 * @example Pairwise precision 1, recall 1/2
 * print('FM', fowlkesMallows([0, 0, 1, 1], [0, 0, 1, 2]))
 */
export const fowlkesMallows = defineMetric(
  partitionInfo('fowlkesMallows', 'Fowlkes–Mallows index', 'fowlkes-mallows-index'),
  (labelsTrue: Labels, labelsPred: Labels): number => {
    const p = pairConfusion(labelsTrue, labelsPred)
    return divide(p.tp, Math.sqrt((p.tp + p.fp) * (p.tp + p.fn)))
  },
)

// ── Information theory ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The entropy $-\sum_k (c_k/n) \log(c_k/n)$ in nats of the partition with cluster sizes $c_k$; empty clusters add
 * nothing.
 *
 * @param counts The cluster sizes $c_k$ (a margin of a contingency table).
 * @param n The number of items, the sum of `counts`.
 * @returns The entropy, 0 for a single cluster.
 */
const entropy = (counts: Float64Array, n: number) => {
  let h = 0
  for (const c of counts) if (c > 0) h -= (c / n) * Math.log(c / n)
  return h
}

/**
 * The mutual information $I(C; K) = \sum_{ij} (n_{ij}/n) \log(n \, n_{ij}/(a_i b_j))$ in nats of a contingency table.
 *
 * @param t The contingency table, as `tableOf` makes it.
 * @returns The mutual information, at least 0.
 */
function mutualInformationOf(t: Table): number {
  let mi = 0
  for (let i = 0; i < t.R; i++)
    for (let j = 0; j < t.C; j++) {
      const c = t.c[i * t.C + j]
      if (c > 0) mi += (c / t.n) * Math.log((t.n * c) / (t.a[i] * t.b[j]))
    }
  return mi
}

/**
 * The expected mutual information of two random labellings with the same cluster sizes (Vinh, Epps and Bailey 2010,
 * eq. 24a): a sum over the possible cell counts weighted by hypergeometric probabilities, in log space by log
 * factorials. Its cost grows with $R \, C$ times the cluster sizes.
 *
 * @param t The contingency table, as `tableOf` makes it; only its margins and $n$ are read.
 * @returns $\expect[I(C; K)]$ in nats.
 */
function expectedMutualInformation(t: Table): number {
  const n = t.n
  const lfN = logFactorial(n)
  let emi = 0
  for (const a of t.a)
    for (const b of t.b) {
      const lead = logFactorial(a) + logFactorial(b) + logFactorial(n - a) + logFactorial(n - b) - lfN
      for (let nij = Math.max(1, a + b - n); nij <= Math.min(a, b); nij++) {
        const logP =
          lead - logFactorial(nij) - logFactorial(a - nij) - logFactorial(b - nij) - logFactorial(n - a - b + nij)
        emi += (nij / n) * Math.log((n * nij) / (a * b)) * Math.exp(logP)
      }
    }
  return emi
}

/**
 * The mean $m$ of the two entropies used to normalise mutual information: $(H(C) + H(K))/2$, $\sqrt{H(C) H(K)}$,
 * $\min$ or $\max$.
 */
export type EntropyMean = 'arithmetic' | 'geometric' | 'min' | 'max'

/**
 * A mean of two entropies.
 *
 * @param h1 The first entropy.
 * @param h2 The second entropy.
 * @param m Which mean: arithmetic (anything not named below), geometric, minimum or maximum.
 * @returns The mean.
 */
function meanOfEntropies(h1: number, h2: number, m: EntropyMean): number {
  if (m === 'min') return Math.min(h1, h2)
  if (m === 'max') return Math.max(h1, h2)
  if (m === 'geometric') return Math.sqrt(h1 * h2)
  return (h1 + h2) / 2
}

/**
 * Mutual information $I(C; K)$ between two labellings, in nats (information-theoretic-clustering-metrics), as
 * sklearn's `mutual_info_score`.
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @returns The mutual information, in $[0, \min(H(C), H(K))]$.
 *
 * @example One bit, in nats, between two copies of a balanced split
 * print('I', mutualInformationScore([0, 0, 1, 1], [1, 1, 0, 0]), 'log 2 =', Math.LN2)
 * print('independent', mutualInformationScore([0, 0, 1, 1], [0, 1, 0, 1]))
 */
export const mutualInformationScore = defineMetric(
  partitionInfo('mutualInformationScore', 'Mutual information', 'information-theoretic-clustering-metrics', [
    0,
    Infinity,
  ]),
  (labelsTrue: Labels, labelsPred: Labels): number => mutualInformationOf(tableOf(labelsTrue, labelsPred)),
)

/**
 * Normalised mutual information $I(C; K)/m(H(C), H(K))$ for a mean $m$ (default arithmetic, which is the V-measure).
 * 1 when both labellings are trivial (one cluster each), as scikit-learn. NaN when the mean is 0 otherwise, which the
 * geometric and minimum means give when one labelling is a single cluster (scikit-learn returns 0).
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @param options `average`, the mean of the two entropies to divide by (default `'arithmetic'`).
 * @returns The normalised mutual information, in $[0, 1]$.
 *
 * @example Four normalisations of the same table
 * const c = [0, 0, 1, 1]
 * const k = [0, 0, 1, 2]
 * for (const average of ['arithmetic', 'geometric', 'min', 'max'])
 *   print(average, normalisedMutualInformation(c, k, { average }))
 */
export const normalisedMutualInformation = defineMetric(
  partitionInfo(
    'normalisedMutualInformation',
    'Normalised mutual information',
    'information-theoretic-clustering-metrics',
  ),
  (labelsTrue: Labels, labelsPred: Labels, options: { average?: EntropyMean } = {}): number => {
    const t = tableOf(labelsTrue, labelsPred)
    if ((t.R === 1 && t.C === 1) || (t.R === 0 && t.C === 0)) return 1
    const mi = mutualInformationOf(t)
    return divide(mi, meanOfEntropies(entropy(t.a, t.n), entropy(t.b, t.n), options.average ?? 'arithmetic'))
  },
)

/**
 * Adjusted mutual information (Vinh, Epps and Bailey 2010): $(I - \expect[I])/(m(H(C), H(K)) - \expect[I])$, with
 * $\expect[I]$ under random labellings of the same cluster sizes. About 0 for random partitions and 1 for identical
 * ones; 1 also when both are trivial or both put every item alone. As sklearn's `adjusted_mutual_info_score`.
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @param options `average`, the mean of the two entropies in the denominator (default `'arithmetic'`).
 * @returns The adjusted mutual information, at most 1.
 *
 * @example Chance-corrected, so lower than the NMI
 * const c = [0, 0, 0, 1, 1, 1]
 * const k = [0, 0, 1, 1, 2, 2]
 * print('AMI', adjustedMutualInformation(c, k))
 * print('NMI', normalisedMutualInformation(c, k))
 */
export const adjustedMutualInformation = defineMetric(
  partitionInfo(
    'adjustedMutualInformation',
    'Adjusted mutual information',
    'information-theoretic-clustering-metrics',
    [-1, 1],
  ),
  (labelsTrue: Labels, labelsPred: Labels, options: { average?: EntropyMean } = {}): number => {
    const t = tableOf(labelsTrue, labelsPred)
    if ((t.R === 1 && t.C === 1) || (t.R === 0 && t.C === 0) || (t.R === t.n && t.C === t.n)) return 1
    const mi = mutualInformationOf(t)
    const emi = expectedMutualInformation(t)
    const norm = meanOfEntropies(entropy(t.a, t.n), entropy(t.b, t.n), options.average ?? 'arithmetic')
    let denominator = norm - emi
    // Guard as scikit-learn: a denominator within machine precision of 0 keeps its sign at ±ε.
    if (Math.abs(denominator) < Number.EPSILON) denominator = denominator < 0 ? -Number.EPSILON : Number.EPSILON
    return (mi - emi) / denominator
  },
)

/**
 * Homogeneity, completeness and V-measure together (Rosenberg and Hirschberg 2007), from one contingency table. As
 * sklearn's `homogeneity_completeness_v_measure`.
 *
 * @param labelsTrue The reference labelling (the classes $C$), one label per item.
 * @param labelsPred The predicted labelling (the clusters $K$) of the same items.
 * @param options `beta`, the weight $\beta$ of completeness against homogeneity in the V-measure (default 1).
 * @returns `homogeneity` $I(C; K)/H(C)$, `completeness` $I(C; K)/H(K)$ (each 1 when its entropy is 0) and `vMeasure`
 *   $(1 + \beta) h c/(\beta h + c)$ (0 when both are 0).
 *
 * @example Splitting a class keeps homogeneity but costs completeness
 * print(homogeneityCompletenessV([0, 0, 1, 1], [0, 0, 1, 2]))
 */
export function homogeneityCompletenessV(
  labelsTrue: Labels,
  labelsPred: Labels,
  options: { beta?: number } = {},
): { homogeneity: number; completeness: number; vMeasure: number } {
  const t = tableOf(labelsTrue, labelsPred)
  const mi = mutualInformationOf(t)
  const hC = entropy(t.a, t.n)
  const hK = entropy(t.b, t.n)
  const homogeneity = hC === 0 ? 1 : mi / hC
  const completeness = hK === 0 ? 1 : mi / hK
  const beta = options.beta ?? 1
  const vMeasure =
    homogeneity + completeness === 0
      ? 0
      : ((1 + beta) * homogeneity * completeness) / (beta * homogeneity + completeness)
  return { homogeneity, completeness, vMeasure }
}

/**
 * Homogeneity $I(C; K)/H(C)$: 1 when each cluster holds one class. 1 when $H(C) = 0$.
 *
 * @param labelsTrue The reference labelling (the classes), one label per item.
 * @param labelsPred The predicted labelling (the clusters) of the same items.
 * @returns The homogeneity, in $[0, 1]$.
 *
 * @example Finer clusters stay homogeneous; merged ones do not
 * print('split', homogeneity([0, 0, 1, 1], [0, 0, 1, 2]))
 * print('merged', homogeneity([0, 0, 1, 1], [0, 0, 0, 0]))
 */
export const homogeneity = defineMetric(
  partitionInfo('homogeneity', 'Homogeneity', 'information-theoretic-clustering-metrics'),
  (labelsTrue: Labels, labelsPred: Labels): number => homogeneityCompletenessV(labelsTrue, labelsPred).homogeneity,
)

/**
 * Completeness $I(C; K)/H(K)$: 1 when each class sits in one cluster. 1 when $H(K) = 0$.
 *
 * @param labelsTrue The reference labelling (the classes), one label per item.
 * @param labelsPred The predicted labelling (the clusters) of the same items.
 * @returns The completeness, in $[0, 1]$.
 *
 * @example Merged clusters stay complete; split ones do not
 * print('merged', completeness([0, 0, 1, 1], [0, 0, 0, 0]))
 * print('split', completeness([0, 0, 1, 1], [0, 0, 1, 2]))
 */
export const completeness = defineMetric(
  partitionInfo('completeness', 'Completeness', 'information-theoretic-clustering-metrics'),
  (labelsTrue: Labels, labelsPred: Labels): number => homogeneityCompletenessV(labelsTrue, labelsPred).completeness,
)

/**
 * V-measure, the ($\beta$-weighted) harmonic mean of homogeneity and completeness; at $\beta = 1$ it is the
 * arithmetic NMI.
 *
 * @param labelsTrue The reference labelling (the classes), one label per item.
 * @param labelsPred The predicted labelling (the clusters) of the same items.
 * @param options `beta`, the weight $\beta$ of completeness against homogeneity (default 1; above 1 favours
 *   completeness).
 * @returns The V-measure, in $[0, 1]$.
 *
 * @example Equal to the arithmetic NMI at beta = 1
 * const c = [0, 0, 1, 1]
 * const k = [0, 0, 1, 2]
 * print('V', vMeasure(c, k), 'NMI', normalisedMutualInformation(c, k))
 * print('beta = 2', vMeasure(c, k, { beta: 2 }))
 */
export const vMeasure = defineMetric(
  partitionInfo('vMeasure', 'V-measure', 'information-theoretic-clustering-metrics'),
  (labelsTrue: Labels, labelsPred: Labels, options: { beta?: number } = {}): number =>
    homogeneityCompletenessV(labelsTrue, labelsPred, options).vMeasure,
)

/**
 * Variation of information $H(C) + H(K) - 2I(C; K)$, a metric on partitions, in nats (Meilă 2007). Lower is better.
 *
 * @param labelsTrue The reference labelling, one label per item.
 * @param labelsPred The predicted labelling of the same items.
 * @returns The variation of information, 0 for identical partitions.
 *
 * @example Splitting one class in two costs half of log 2
 * print('VI', variationOfInformation([0, 0, 1, 1], [0, 0, 1, 2]), 'log(2)/2 =', Math.LN2 / 2)
 * print('identical', variationOfInformation([0, 0, 1, 1], [1, 1, 0, 0]))
 */
export const variationOfInformation = defineMetric(
  {
    ...partitionInfo('variationOfInformation', 'Variation of information', 'information-theoretic-clustering-metrics'),
    direction: 'lower',
    range: [0, Infinity],
  },
  (labelsTrue: Labels, labelsPred: Labels): number => {
    const t = tableOf(labelsTrue, labelsPred)
    return entropy(t.a, t.n) + entropy(t.b, t.n) - 2 * mutualInformationOf(t)
  },
)

// ── Internal indices ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The data matrix ($n \times d$) and cluster index of each row, with what the internal indices share: the cluster
 * sizes, the centroids and the Euclidean distances between rows. Throws `ShapeError` when the labels and rows differ in
 * number and `DomainError` for fewer than two clusters.
 *
 * @param x The data: an $n \times d$ matrix (rows of numbers or a tensor), one row per point; a flat array is $n$
 *   points in one dimension.
 * @param labels The cluster label of each row.
 * @returns `X`, the data row-major; `k`, each row's cluster index (clusters in label order); `K`, the number of
 *   clusters; `sizes`, their sizes; `centroids`, their means ($K \times d$, row-major); and `dist(i, j)`, the distance
 *   between rows $i$ and $j$.
 */
function clustered(x: Rows, labels: Labels) {
  const X = dense(x, 'clustering data')
  const l = labelList(labels)
  sameLength(l, { length: X.rows }, 'internal clustering index')
  const ids = classesOf(l)
  const k = encodeLabels(l, ids)
  const K = ids.length
  if (K < 2) throw new DomainError('metrics', 'metrics: internal clustering indices need at least two clusters')
  const D = toFlat(pairwiseDistances(matrix(X.data, X.rows, X.cols)))
  const dist = (i: number, j: number) => D[i * X.rows + j]
  const sizes = new Float64Array(K)
  const centroids = new Float64Array(K * X.cols)
  for (let i = 0; i < X.rows; i++) {
    sizes[k[i]]++
    for (let c = 0; c < X.cols; c++) centroids[k[i] * X.cols + c] += X.data[i * X.cols + c]
  }
  for (let j = 0; j < K; j++) for (let c = 0; c < X.cols; c++) centroids[j * X.cols + c] /= sizes[j]
  return { X, k, K, sizes, centroids, dist }
}

/**
 * Silhouette of each point (Rousseeuw 1987): $(b_i - a_i)/\max(a_i, b_i)$, with $a_i$ the mean Euclidean distance
 * to the other points of its cluster and $b_i$ the mean distance to the nearest other cluster; 0 for a point alone in
 * its cluster. As sklearn's `silhouette_samples`. Throws `DomainError` for fewer than two clusters.
 *
 * @param x The data: an $n \times d$ matrix, one row per point.
 * @param labels The cluster label of each row.
 * @returns The $n$ silhouettes, each in $[-1, 1]$.
 *
 * @example Two tight, distant clusters on a line
 * print(silhouetteSamples([[0], [1], [10], [11]], [0, 0, 1, 1]))
 */
export function silhouetteSamples(x: Rows, labels: Labels): Tensor {
  const { X, k, K, sizes, dist } = clustered(x, labels)
  const out = new Float64Array(X.rows)
  for (let i = 0; i < X.rows; i++) {
    const sums = new Float64Array(K)
    for (let j = 0; j < X.rows; j++) if (j !== i) sums[k[j]] += dist(i, j)
    if (sizes[k[i]] === 1) continue
    const a = sums[k[i]] / (sizes[k[i]] - 1)
    let b = Infinity
    for (let c = 0; c < K; c++) if (c !== k[i]) b = Math.min(b, sums[c] / sizes[c])
    out[i] = (b - a) / Math.max(a, b)
  }
  return vector(out)
}

/**
 * The silhouette score, the mean silhouette over all points (internal-clustering-indices), as sklearn's
 * `silhouette_score` without sampling. Computes all $O(n^2)$ distances.
 *
 * @param x The data: an $n \times d$ matrix, one row per point.
 * @param labels The cluster label of each row (at least two clusters).
 * @returns The mean silhouette, in $[-1, 1]$.
 *
 * @example A good and a bad clustering of the same points
 * const x = [[0], [1], [10], [11]]
 * print('by position', silhouetteScore(x, [0, 0, 1, 1]))
 * print('alternating', silhouetteScore(x, [0, 1, 0, 1]))
 */
export const silhouetteScore = defineMetric(
  {
    key: 'silhouetteScore',
    stability: 'stable',
    name: 'Silhouette score',
    inputs: 'features',
    direction: 'higher',
    range: [-1, 1],
    notes: ['internal-clustering-indices'],
  },
  (x: Rows, labels: Labels): number => {
    const s = silhouetteSamples(x, labels).data as Float64Array
    return s.reduce((a, b) => a + b, 0) / s.length
  },
)

/**
 * The Calinski–Harabasz index (variance ratio criterion; Caliński and Harabasz 1974): $(B/(k - 1))/(W/(n - k))$ with
 * $B = \sum_j \lvert C_j \rvert \lVert \muvec_j - \muvec \rVert^2$ and
 * $W = \sum_j \sum_{\xvec \in C_j} \lVert \xvec - \muvec_j \rVert^2$, for $k$ clusters $C_j$ with centroids
 * $\muvec_j$ and overall mean $\muvec$. 1 when $W = 0$, as sklearn's `calinski_harabasz_score`.
 *
 * @param x The data: an $n \times d$ matrix, one row per point.
 * @param labels The cluster label of each row (at least two clusters).
 * @returns The index, higher for tighter, better-separated clusters.
 *
 * @example Between-cluster spread 100, within 1
 * print('CH', calinskiHarabasz([[0], [1], [10], [11]], [0, 0, 1, 1]))
 */
export const calinskiHarabasz = defineMetric(
  {
    key: 'calinskiHarabasz',
    stability: 'stable',
    name: 'Calinski–Harabasz index',
    inputs: 'features',
    direction: 'higher',
    range: [0, Infinity],
    notes: ['internal-clustering-indices'],
  },
  (x: Rows, labels: Labels): number => {
    const { X, k, K, sizes, centroids } = clustered(x, labels)
    const d = X.cols
    const mean = new Float64Array(d)
    for (let i = 0; i < X.rows; i++) for (let c = 0; c < d; c++) mean[c] += X.data[i * d + c] / X.rows
    let B = 0
    let W = 0
    for (let j = 0; j < K; j++) for (let c = 0; c < d; c++) B += sizes[j] * (centroids[j * d + c] - mean[c]) ** 2
    for (let i = 0; i < X.rows; i++) for (let c = 0; c < d; c++) W += (X.data[i * d + c] - centroids[k[i] * d + c]) ** 2
    return W === 0 ? 1 : B / (K - 1) / (W / (X.rows - K))
  },
)

/**
 * The Davies–Bouldin index (Davies and Bouldin 1979):
 * $\frac{1}{k} \sum_j \max_{l \ne j} (S_j + S_l)/\lVert \muvec_j - \muvec_l \rVert$, with $S_j$ the mean distance of
 * cluster $j$'s points to its centroid $\muvec_j$. Lower is better. Two clusters with the same centroid make it
 * infinite (NaN when both have zero spread), where sklearn's `davies_bouldin_score` skips the pair.
 *
 * @param x The data: an $n \times d$ matrix, one row per point.
 * @param labels The cluster label of each row (at least two clusters).
 * @returns The index, at least 0.
 *
 * @example Spreads of 0.5 about centroids 10 apart
 * print('DB', daviesBouldin([[0], [1], [10], [11]], [0, 0, 1, 1]))
 */
export const daviesBouldin = defineMetric(
  {
    key: 'daviesBouldin',
    stability: 'stable',
    name: 'Davies–Bouldin index',
    inputs: 'features',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['internal-clustering-indices'],
  },
  (x: Rows, labels: Labels): number => {
    const { X, k, K, sizes, centroids } = clustered(x, labels)
    const d = X.cols
    const S = new Float64Array(K)
    for (let i = 0; i < X.rows; i++) {
      let s = 0
      for (let c = 0; c < d; c++) s += (X.data[i * d + c] - centroids[k[i] * d + c]) ** 2
      S[k[i]] += Math.sqrt(s) / sizes[k[i]]
    }
    let total = 0
    for (let j = 0; j < K; j++) {
      let worst = 0
      for (let l = 0; l < K; l++) {
        if (l === j) continue
        let m = 0
        for (let c = 0; c < d; c++) m += (centroids[j * d + c] - centroids[l * d + c]) ** 2
        worst = Math.max(worst, (S[j] + S[l]) / Math.sqrt(m))
      }
      total += worst
    }
    return total / K
  },
)

/**
 * The Dunn index (Dunn 1974): the smallest distance between points of different clusters over the largest cluster
 * diameter (the largest distance between two points of one cluster). NaN when every cluster is a single point.
 *
 * @param x The data: an $n \times d$ matrix, one row per point.
 * @param labels The cluster label of each row (at least two clusters).
 * @returns The index, higher for compact, well-separated clusters.
 *
 * @example Gap 9 between clusters of diameter 1
 * print('Dunn', dunnIndex([[0], [1], [10], [11]], [0, 0, 1, 1]))
 */
export const dunnIndex = defineMetric(
  {
    key: 'dunnIndex',
    stability: 'stable',
    name: 'Dunn index',
    inputs: 'features',
    direction: 'higher',
    range: [0, Infinity],
    notes: ['internal-clustering-indices'],
  },
  (x: Rows, labels: Labels): number => {
    const { X, k, dist } = clustered(x, labels)
    let separation = Infinity
    let diameter = 0
    for (let i = 0; i < X.rows; i++)
      for (let j = i + 1; j < X.rows; j++) {
        const dij = dist(i, j)
        if (k[i] === k[j]) diameter = Math.max(diameter, dij)
        else separation = Math.min(separation, dij)
      }
    return divide(separation, diameter)
  },
)
