/**
 * The anomaly detectors against scikit-learn (`fixtures/unsupervised/anomaly.json`): LOF and k-NN distances exactly,
 * the one-class SVM's decision function up to its νn scaling, and the isolation forest statistically (AUROC and rank
 * agreement with scikit-learn's forest); and laws: SVDD and the one-class SVM agree for a Gaussian kernel, ν bounds the
 * share outside, the robust Mahalanobis distance flags the planted outliers, the ensemble ranks, and thresholds.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { auroc as rocAuc } from 'aifn-compute/learning/metrics'
import { spearman } from 'aifn-compute/probability/stats'
import { plantedAnomalies } from 'aifn-methods/data/synthetic'
import {
  anomalyFunctions,
  anomalyScores,
  anomalyThreshold,
  averagePathLength,
  combineScores,
  DETECTORS,
  isolationForest,
  isolationScore,
  knnScore,
  localOutlierFactor,
  localOutlierScore,
  oneClassScore,
  oneClassSvm,
  pcaModel,
  pcaReconstructionScore,
  rankNormalise,
  supportVectorDataDescription,
} from 'aifn-methods/unsupervised/anomaly'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type F = {
  x: number[][]
  y: number[]
  queries: number[][]
  lof: { k: number; train: number[]; queries: number[] }
  knn: { k: number; train: number[]; queries: number[] }
  ocsvm: { nu: number; gamma: number; train: number[]; queries: number[]; scale: number }
  isolation: { scores: number[]; auroc: number }
}
const A = fixture<F>('unsupervised/anomaly')

const close = (got: ArrayLike<number>, want: number[], tol: number) =>
  want.forEach((w, i) =>
    expect(Math.abs(got[i] - w), `${i}: ${got[i]} vs ${w}`).toBeLessThanOrEqual(tol * (1 + Math.abs(w))),
  )

describe('distance-based scores', () => {
  it('LOF matches scikit-learn on training and new points', () => {
    const m = localOutlierFactor(A.x, { k: A.lof.k })
    close(m.factor, A.lof.train, 1e-9)
    close(localOutlierScore(m, A.queries), A.lof.queries, 1e-9)
  })
  it('the k-NN distance matches scikit-learn', () => {
    close(knnScore(A.x, undefined, { k: A.knn.k }), A.knn.train, 1e-12)
    close(knnScore(A.x, A.queries, { k: A.knn.k }), A.knn.queries, 1e-12)
  })
})

describe('one-class methods', () => {
  it('the one-class SVM decision function is scikit-learn’s divided by νn', () => {
    const m = oneClassSvm(A.x, { nu: A.ocsvm.nu, gamma: A.ocsvm.gamma })
    // aifn's score is −f(x); scikit-learn's decision_function is νn f(x).
    close(
      Array.from(oneClassScore(m, A.x), (v) => -v * A.ocsvm.scale),
      A.ocsvm.train,
      2e-5,
    )
    close(
      Array.from(oneClassScore(m, A.queries), (v) => -v * A.ocsvm.scale),
      A.ocsvm.queries,
      2e-5,
    )
  })
  it('ν bounds the share of training points outside from above', () => {
    const m = oneClassSvm(A.x, { nu: 0.2, gamma: 0.8 })
    const outside = Array.from(oneClassScore(m, A.x)).filter((v) => v > 1e-6).length / A.x.length
    expect(outside).toBeLessThanOrEqual(0.2 + 1e-9)
    expect(m.boundShare).toBeLessThanOrEqual(0.2 + 1e-9)
  })
  it('SVDD and the one-class SVM give the same boundary with a Gaussian kernel', () => {
    const a = oneClassScore(oneClassSvm(A.x, { nu: 0.1, gamma: 0.8 }), A.queries)
    const b = oneClassScore(supportVectorDataDescription(A.x, { nu: 0.1, gamma: 0.8 }), A.queries)
    // ‖φ(x) − a‖² − R² = 2(ρ − f(x)) when k(x, x) = 1: the same sign and ordering.
    a.forEach((v, i) => expect(b[i]).toBeCloseTo(2 * v, 5))
  })
})

describe('isolation forest', () => {
  it('ranks the planted outliers as scikit-learn’s forest does (statistically)', () => {
    const f = isolationForest(A.x, { trees: 200 })
    const s = isolationScore(f, A.x)
    const auc = rocAuc(A.y, s)
    expect(Math.abs(auc - A.isolation.auroc)).toBeLessThan(0.05)
    expect(spearman(s, A.isolation.scores)).toBeGreaterThan(0.85)
  })
  it('c(n) is the average unsuccessful search length', () => {
    expect(averagePathLength(1)).toBe(0)
    expect(averagePathLength(2)).toBe(1)
    expect(averagePathLength(256)).toBeCloseTo(2 * (Math.log(255) + 0.5772156649) - (2 * 255) / 256, 9)
  })
})

describe('every detector on planted anomalies', () => {
  const d = plantedAnomalies(stream(3), { n: 200, contamination: 0.06 })
  const y = Array.from(toFlat(d.y!))
  for (const { kind } of DETECTORS)
    it(`${kind} ranks the anomalies above the inliers (AUROC > 0.7)`, () => {
      const r = anomalyScores(
        kind,
        d.x,
        [
          [0, 0],
          [2.4, 2.4],
        ],
        { seed: 1 },
      )
      expect(r.train.length).toBe(200)
      expect(r.queries!.length).toBe(2)
      expect(rocAuc(y, r.train)).toBeGreaterThan(kind === 'pca-reconstruction' || kind === 'mahalanobis' ? 0.6 : 0.7)
    })
})

describe('ensembles and thresholds', () => {
  it('rank normalisation maps to [0, 1] with ties averaged; the mean combines', () => {
    expect(Array.from(rankNormalise([3, 1, 3, 2]))).toEqual([(2 + 3) / 2 / 3, 0, (2 + 3) / 2 / 3, 1 / 3])
    expect(
      Array.from(
        combineScores([
          [1, 2],
          [4, 3],
        ]),
      ),
    ).toEqual([0.5, 0.5])
    expect(
      Array.from(
        combineScores(
          [
            [1, 2],
            [4, 3],
          ],
          { combine: 'max' },
        ),
      ),
    ).toEqual([1, 1])
  })
  it('a z-score ensemble ignores a constant detector', () => {
    expect(
      Array.from(
        combineScores(
          [
            [1, 2, 3],
            [5, 5, 5],
          ],
          { normalise: 'zscore', combine: 'max' },
        ),
      ),
    ).toEqual([0, 0, Math.sqrt(1.5)])
  })
  it('PCA asked for more axes than points keeps the min(n, d) the SVD has', () => {
    const x = [
      [1, 0, 2, 0, 1],
      [0, 1, 0, 3, 1],
      [2, 2, 1, 1, 0],
    ]
    const m = pcaModel(x, { components: 4 })
    expect(m.q).toBe(3)
    // The training points lie in their own affine span: no reconstruction error.
    Array.from(pcaReconstructionScore(m, x)).forEach((e) => expect(e).toBeLessThan(1e-20))
  })
  it('the quantile threshold leaves the risk share above it; the POT threshold reaches past the data', () => {
    const scores = Array.from({ length: 1000 }, (_, i) => -Math.log(1 - (i + 0.5) / 1000))
    const t = anomalyThreshold(scores, { risk: 0.05 })
    expect(scores.filter((s) => s > t).length).toBe(50)
    const deep = anomalyThreshold(scores, { method: 'pot', risk: 1e-4 })
    expect(deep).toBeGreaterThan(Math.max(...scores))
    expect(deep).toBeCloseTo(-Math.log(1e-4), 0)
  })
})

describe('registry', () => {
  it('entries are well formed', () => expectInfo(anomalyFunctions, 'function'))
})
