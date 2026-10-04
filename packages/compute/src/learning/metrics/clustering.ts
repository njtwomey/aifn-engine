/**
 * Clustering metrics. External scores compare a predicted partition with a reference through their contingency table:
 * pair counting (Rand, adjusted Rand, Fowlkes–Mallows) and information theory (mutual information, NMI with four
 * normalisations, AMI, homogeneity, completeness, V-measure, variation of information), in nats as scikit-learn.
 * Internal indices judge a partition by the data's geometry: silhouette, Calinski–Harabasz, Davies–Bouldin, Dunn.
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

/** A contingency table of two labellings: `table[i][j]` counts items in class i of the first and cluster j of the second. */
export type Contingency = {
  table: Tensor
  /** Row sums aᵢ (sizes of the first labelling's classes). */
  rows: Tensor
  /** Column sums bⱼ (sizes of the second labelling's clusters). */
  cols: Tensor
  n: number
  rowLabels: Label[]
  colLabels: Label[]
}

type Table = { c: Float64Array; a: Float64Array; b: Float64Array; R: number; C: number; n: number }

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

/** The contingency table of two labellings of the same n items (rows: `labelsTrue`, columns: `labelsPred`). */
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

const pairs = (m: number) => (m * (m - 1)) / 2

/** Pair counts: TP pairs together in both, FN together only in the truth, FP only in the prediction, TN apart in both. */
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

/** The Rand index (TP + TN)/C(n, 2), the accuracy of the pairwise "together or apart" decisions (Rand 1971). */
export const randIndex = defineMetric(
  partitionInfo('randIndex', 'Rand index', 'rand-index-and-adjusted-rand-index'),
  (labelsTrue: Labels, labelsPred: Labels): number => {
    const p = pairConfusion(labelsTrue, labelsPred)
    const total = p.tp + p.fp + p.fn + p.tn
    return total === 0 ? 1 : (p.tp + p.tn) / total
  },
)

/**
 * The adjusted Rand index (Hubert and Arabie 1985): (Index − E[Index])/(Max − E[Index]) with Index = Σ C(nᵢⱼ, 2),
 * E[Index] = Σ C(aᵢ, 2) Σ C(bⱼ, 2)/C(n, 2) and Max the mean of the two marginal sums. 1 for identical partitions (also
 * when both are trivial), about 0 for random ones.
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

/** The Fowlkes–Mallows index TP/√((TP + FP)(TP + FN)), the geometric mean of pairwise precision and recall. */
export const fowlkesMallows = defineMetric(
  partitionInfo('fowlkesMallows', 'Fowlkes–Mallows index', 'fowlkes-mallows-index'),
  (labelsTrue: Labels, labelsPred: Labels): number => {
    const p = pairConfusion(labelsTrue, labelsPred)
    return divide(p.tp, Math.sqrt((p.tp + p.fp) * (p.tp + p.fn)))
  },
)

// ── Information theory ───────────────────────────────────────────────────────────────────────────────────────────────

const entropy = (counts: Float64Array, n: number) => {
  let h = 0
  for (const c of counts) if (c > 0) h -= (c / n) * Math.log(c / n)
  return h
}

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
 * eq. 24a): a sum over the possible cell counts weighted by hypergeometric probabilities.
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

/** The mean used to normalise mutual information. */
export type EntropyMean = 'arithmetic' | 'geometric' | 'min' | 'max'

function meanOfEntropies(h1: number, h2: number, m: EntropyMean): number {
  if (m === 'min') return Math.min(h1, h2)
  if (m === 'max') return Math.max(h1, h2)
  if (m === 'geometric') return Math.sqrt(h1 * h2)
  return (h1 + h2) / 2
}

/** Mutual information I(C; K) between two labellings, in nats (information-theoretic-clustering-metrics). */
export const mutualInformationScore = defineMetric(
  partitionInfo('mutualInformationScore', 'Mutual information', 'information-theoretic-clustering-metrics', [
    0,
    Infinity,
  ]),
  (labelsTrue: Labels, labelsPred: Labels): number => mutualInformationOf(tableOf(labelsTrue, labelsPred)),
)

/**
 * Normalised mutual information I(C; K)/m(H(C), H(K)) for a mean m (default arithmetic, which is the V-measure). 1 when
 * both labellings are trivial (one cluster each), as scikit-learn.
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
 * Adjusted mutual information (Vinh, Epps and Bailey 2010): (I − E[I])/(m(H(C), H(K)) − E[I]), with E[I] under random
 * labellings of the same cluster sizes. About 0 for random partitions and 1 for identical ones.
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

/** Homogeneity, completeness and V-measure together (Rosenberg and Hirschberg 2007). */
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

/** Homogeneity I(C; K)/H(C): each cluster holds one class. 1 when H(C) = 0. */
export const homogeneity = defineMetric(
  partitionInfo('homogeneity', 'Homogeneity', 'information-theoretic-clustering-metrics'),
  (labelsTrue: Labels, labelsPred: Labels): number => homogeneityCompletenessV(labelsTrue, labelsPred).homogeneity,
)

/** Completeness I(C; K)/H(K): each class sits in one cluster. 1 when H(K) = 0. */
export const completeness = defineMetric(
  partitionInfo('completeness', 'Completeness', 'information-theoretic-clustering-metrics'),
  (labelsTrue: Labels, labelsPred: Labels): number => homogeneityCompletenessV(labelsTrue, labelsPred).completeness,
)

/** V-measure, the (β-weighted) harmonic mean of homogeneity and completeness; at β = 1 it is the arithmetic NMI. */
export const vMeasure = defineMetric(
  partitionInfo('vMeasure', 'V-measure', 'information-theoretic-clustering-metrics'),
  (labelsTrue: Labels, labelsPred: Labels, options: { beta?: number } = {}): number =>
    homogeneityCompletenessV(labelsTrue, labelsPred, options).vMeasure,
)

/** Variation of information H(C) + H(K) − 2I(C; K), a metric on partitions, in nats (Meilă 2007). */
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

/** The data matrix (n × d) and cluster index of each row. */
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
 * Silhouette of each point (Rousseeuw 1987): (bᵢ − aᵢ)/max(aᵢ, bᵢ), with aᵢ the mean Euclidean distance to the other
 * points of its cluster and bᵢ the mean distance to the nearest other cluster; 0 for a point alone in its cluster.
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

/** The silhouette score, the mean silhouette over all points (internal-clustering-indices). O(n²) distances. */
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
 * The Calinski–Harabasz index (variance ratio criterion; Caliński and Harabasz 1974): (B/(k − 1))/(W/(n − k)) with
 * B = Σⱼ |Cⱼ| ‖μⱼ − μ‖² and W = Σⱼ Σ_{x∈Cⱼ} ‖x − μⱼ‖².
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
 * The Davies–Bouldin index (Davies and Bouldin 1979): (1/k) Σⱼ max_{l≠j} (Sⱼ + Sₗ)/‖μⱼ − μₗ‖, with Sⱼ the mean distance
 * of cluster j's points to its centroid. Lower is better.
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

/** The Dunn index: the smallest distance between points of different clusters over the largest cluster diameter. */
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
