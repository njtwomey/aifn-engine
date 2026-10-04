/**
 * Every detector of the module behind one call, `anomalyScores(kind, train, queries)`: fit on the training points and
 * score them (and any query points, such as a grid for drawing the score field), higher meaning more anomalous. The
 * ensemble normalises each member's scores by the empirical distribution of its training scores, so a query is ranked
 * against the training data, and averages them.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import { auroc, averagePrecision } from 'aifn-compute/learning/metrics'
import { knnScore, localOutlierFactor, localOutlierScore } from './distance'
import { isolationForest, isolationScore } from './isolation'
import { oneClassScore, oneClassSvm, supportVectorDataDescription } from './oneclass'
import { mahalanobisModel, mahalanobisScore, pcaModel, pcaReconstructionScore } from './statistical'

/** The detectors. */
export type DetectorKind =
  | 'isolation-forest'
  | 'local-outlier-factor'
  | 'knn'
  | 'one-class-svm'
  | 'svdd'
  | 'mahalanobis'
  | 'robust-mahalanobis'
  | 'pca-reconstruction'
  | 'ensemble'

/** The detectors in the order a page lists them, with a display name. */
export const DETECTORS: readonly { kind: DetectorKind; name: string }[] = [
  { kind: 'isolation-forest', name: 'Isolation forest' },
  { kind: 'local-outlier-factor', name: 'Local outlier factor' },
  { kind: 'knn', name: 'k-NN distance' },
  { kind: 'one-class-svm', name: 'One-class SVM' },
  { kind: 'svdd', name: 'SVDD' },
  { kind: 'mahalanobis', name: 'Mahalanobis (classical)' },
  { kind: 'robust-mahalanobis', name: 'Mahalanobis (MCD)' },
  { kind: 'pca-reconstruction', name: 'PCA reconstruction' },
  { kind: 'ensemble', name: 'Ensemble (mean rank)' },
]

/** Hyperparameters of the detectors (each reads its own). */
export type DetectorOptions = {
  /** Neighbours of k-NN and LOF (default 10 and 20). */
  k?: Size
  /** Isolation forest: trees and subsample size (default 100 and 256). */
  trees?: Size
  sampleSize?: Size
  /** One-class SVM and SVDD: ν and the Gaussian kernel's γ. */
  nu?: number
  gamma?: number
  /** PCA reconstruction: principal components kept (default 1). */
  components?: Size
  /** Ensemble members (default isolation forest, LOF, k-NN, robust Mahalanobis). */
  members?: readonly Exclude<DetectorKind, 'ensemble'>[]
  /** The seed of the randomised detectors (default 0). */
  seed?: number | string
}

/** Training and query scores of a detector. */
export type AnomalyScores = { train: Float64Array; queries: Float64Array | null }

/** The share of `reference` at or below each value (the empirical cdf), as a normaliser shared by train and queries. */
function ecdf(reference: ArrayLike<number>, values: ArrayLike<number>): Float64Array {
  const sorted = Float64Array.from(reference).sort()
  const n = sorted.length
  return Float64Array.from(values, (v) => {
    let lo = 0
    let hi = n
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (sorted[mid] <= v) lo = mid + 1
      else hi = mid
    }
    return lo / n
  })
}

/** Fit a detector on `train` and score the training points and, when given, the query points. */
export function anomalyScores(
  kind: DetectorKind,
  train: MatrixLike,
  queries?: MatrixLike,
  options: DetectorOptions = {},
): AnomalyScores {
  const both = (score: (x: MatrixLike) => Float64Array): AnomalyScores => ({
    train: score(train),
    queries: queries === undefined ? null : score(queries),
  })
  switch (kind) {
    case 'isolation-forest': {
      const f = isolationForest(train, {
        trees: options.trees,
        sampleSize: options.sampleSize,
        stream: stream(options.seed ?? 0),
      })
      return both((x) => isolationScore(f, x))
    }
    case 'local-outlier-factor': {
      const m = localOutlierFactor(train, { k: options.k ?? 20 })
      return { train: m.factor, queries: queries === undefined ? null : localOutlierScore(m, queries) }
    }
    case 'knn':
      return {
        train: knnScore(train, undefined, { k: options.k ?? 10 }),
        queries: queries === undefined ? null : knnScore(train, queries, { k: options.k ?? 10 }),
      }
    case 'one-class-svm':
    case 'svdd': {
      const m = (kind === 'svdd' ? supportVectorDataDescription : oneClassSvm)(train, {
        nu: options.nu,
        gamma: options.gamma,
      })
      return both((x) => oneClassScore(m, x))
    }
    case 'mahalanobis':
    case 'robust-mahalanobis': {
      const m = mahalanobisModel(train, { robust: kind === 'robust-mahalanobis' })
      return both((x) => mahalanobisScore(m, x))
    }
    case 'pca-reconstruction': {
      const m = pcaModel(train, { components: options.components ?? 1 })
      return both((x) => pcaReconstructionScore(m, x))
    }
    case 'ensemble': {
      const members = options.members ?? ['isolation-forest', 'local-outlier-factor', 'knn', 'robust-mahalanobis']
      const parts = members.map((m) => anomalyScores(m, train, queries, options))
      const mean = (pick: (p: AnomalyScores) => Float64Array) => {
        const n = pick(parts[0]).length
        const out = new Float64Array(n)
        for (const p of parts) {
          const r = ecdf(p.train, pick(p))
          for (let i = 0; i < n; i++) out[i] += r[i] / parts.length
        }
        return out
      }
      return { train: mean((p) => p.train), queries: queries === undefined ? null : mean((p) => p.queries!) }
    }
  }
}

/** One detector's ranking quality against known anomaly labels. */
export type DetectorComparison = { kind: DetectorKind; auroc: number; averagePrecision: number }

/**
 * Fit every detector (or the given ones) on the same points and measure how well each ranks the known anomalies
 * (`labels`: 1 anomaly, 0 inlier) above the inliers: the area under the ROC curve and the average precision of
 * `aifn-compute/learning/metrics`, threshold-free summaries of the score.
 */
export function compareDetectors(
  train: MatrixLike,
  labels: ArrayLike<number>,
  options: DetectorOptions & { kinds?: readonly DetectorKind[] } = {},
): DetectorComparison[] {
  const kinds = options.kinds ?? DETECTORS.map((d) => d.kind)
  const y = Array.from(labels)
  return kinds.map((kind) => {
    const s = anomalyScores(kind, train, undefined, options).train
    return { kind, auroc: auroc(y, s), averagePrecision: averagePrecision(y, s) }
  })
}
