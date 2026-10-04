/**
 * aifn-compute/learning/metrics: one or two hand-checked cases per family, mostly the worked examples of the site's notes;
 * the registry; distances against numerics/linalg; the ordinal metrics.
 */
import { describe, expect, it } from 'vitest'
import {
  accuracy,
  adjustedMutualInformation,
  adjustedRandIndex,
  auroc,
  averagePrecision,
  balancedAccuracy,
  binaryRates,
  binormalAuroc,
  binormalAveragePrecision,
  bootstrapMetric,
  brierDecomposition,
  brierScore,
  calinskiHarabasz,
  cohensKappa,
  concordanceCorrelation,
  confusionMargins,
  confusionMatrix,
  cramersV,
  crpsEnsemble,
  crpsGaussian,
  daviesBouldin,
  dcg,
  dunnIndex,
  expectedCalibrationError,
  expectedReciprocalRank,
  explainedVariance,
  f1,
  fBeta,
  fleissKappa,
  fowlkesMallows,
  getMetric,
  hammingLoss,
  hausdorffDistances,
  hitRate,
  intervalScore,
  intraclassCorrelation,
  jaccardScore,
  kappaFromTable,
  krippendorffAlpha,
  listMetrics,
  logLoss,
  macroMeanAbsoluteError,
  matthewsCorrelation,
  maximumCalibrationError,
  meanAbsoluteError,
  meanAbsolutePercentageError,
  meanAbsoluteScaledError,
  meanAveragePrecision,
  meanReciprocalRank,
  meanSquaredError,
  normalisedRootMeanSquaredError,
  metricRegistry,
  minkowskiDistance,
  cosineSimilarity,
  ndcg,
  normalisedMutualInformation,
  partialAuroc,
  pinballLoss,
  precision,
  precisionAtK,
  precisionRecallCurve,
  precisionRecallTrapezoid,
  procrustesDisparity,
  r2Score,
  rPrecision,
  randIndex,
  recall,
  recallAtK,
  reliabilityDiagram,
  rocCurve,
  rootMeanSquaredScaledError,
  silhouetteScore,
  symmetricMeanAbsolutePercentageError,
  theilsU,
  tweedieDeviance,
  vMeasure,
  wilsonInterval,
  quadraticWeightedKappa,
  logCoshError,
  meanSquaredLogError,
  angularDistance,
  chebyshevDistance,
  consistencyBars,
  cosineDistance,
  costCurve,
  detCurve,
  precisionRecallGainCurve,
  rocConvexHull,
  delongTest,
  equalErrorRate,
  euclideanDistance,
  gainCurve,
  hausdorffDistance,
  isBetter,
  mahalanobisDistance,
  manhattanDistance,
  operatingPoint,
  ordinalMeanAbsoluteError,
  orthogonalProcrustes,
  pitValues,
  withinToleranceAccuracy,
  youdenPoint,
} from 'aifn-compute/learning/metrics'
import { pairwiseDistances, squaredDistances } from 'aifn-compute/numerics/linalg'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** The screening data of the confusion-matrix note: TP 40, FN 10, FP 90, TN 860. */
function screening() {
  const yTrue: number[] = []
  const yPred: number[] = []
  const push = (t: number, p: number, n: number) => {
    for (let i = 0; i < n; i++) {
      yTrue.push(t)
      yPred.push(p)
    }
  }
  push(1, 1, 40)
  push(1, 0, 10)
  push(0, 1, 90)
  push(0, 0, 860)
  return { yTrue, yPred }
}

/** Labels reproducing a confusion matrix (rows true, columns predicted). */
function fromMatrix(m: number[][]) {
  const yTrue: number[] = []
  const yPred: number[] = []
  m.forEach((row, j) =>
    row.forEach((c, k) => {
      for (let i = 0; i < c; i++) {
        yTrue.push(j)
        yPred.push(k)
      }
    }),
  )
  return { yTrue, yPred }
}

describe('classification', () => {
  const { yTrue, yPred } = screening()
  it('reproduces the screening table', () => {
    const r = binaryRates({ tp: 40, fp: 90, fn: 10, tn: 860 })
    expect(r.accuracy).toBeCloseTo(0.9, 10)
    expect(r.balancedAccuracy).toBeCloseTo(0.853, 3)
    expect(r.matthewsCorrelation).toBeCloseTo(0.457, 3)
    expect(r.cohensKappa).toBeCloseTo(0.401, 3)
    expect(r.jaccard).toBeCloseTo(0.286, 3)
    expect(r.positiveLikelihoodRatio).toBeCloseTo(8.44, 2)
    expect(accuracy(yTrue, yPred)).toBeCloseTo(0.9, 12)
    expect(precision(yTrue, yPred)).toBeCloseTo(40 / 130, 12)
    expect(recall(yTrue, yPred)).toBeCloseTo(0.8, 12)
    expect(f1(yTrue, yPred)).toBeCloseTo(0.444, 3)
    expect(fBeta(yTrue, yPred, { beta: 2 })).toBeCloseTo(0.606, 3)
    expect(fBeta(yTrue, yPred, { beta: 0.5 })).toBeCloseTo(0.351, 3)
    expect(matthewsCorrelation(yTrue, yPred)).toBeCloseTo(0.457, 3)
    expect(cohensKappa(yTrue, yPred)).toBeCloseTo(0.401, 3)
    // Positive class first: rows [TP FN; FP TN], so the margins read TPR/FNR, TNR/FPR, PPV/FDR, NPV/FOR.
    const m = confusionMargins(confusionMatrix(yTrue, yPred, { labels: [1, 0] }))
    expect(toFlat(m.actual)).toEqual([50, 950])
    expect(toFlat(m.predicted)).toEqual([130, 870])
    expect(m.n).toBe(1000)
    expect(toFlat(m.rowRate)[0]).toBeCloseTo(0.8, 12)
    expect(toFlat(m.rowRate)[1]).toBeCloseTo(860 / 950, 12)
    expect(toFlat(m.rowMiss)[1]).toBeCloseTo(90 / 950, 12)
    expect(toFlat(m.columnRate)[0]).toBeCloseTo(40 / 130, 12)
    expect(toFlat(m.columnMiss)[1]).toBeCloseTo(10 / 870, 12)
    expect(toFlat(m.prevalence)[0]).toBeCloseTo(0.05, 12)
    expect(m.accuracy).toBeCloseTo(0.9, 12)
    expect(balancedAccuracy(yTrue, yPred)).toBeCloseTo(0.853, 3)
  })
  it('averages the three-class example', () => {
    const { yTrue: t, yPred: p } = fromMatrix([
      [70, 6, 4],
      [4, 10, 1],
      [2, 1, 2],
    ])
    expect(precision(t, p, { average: 'micro' })).toBeCloseTo(0.82, 12)
    expect(precision(t, p, { average: 'macro' })).toBeCloseTo(0.598, 3)
    expect(recall(t, p, { average: 'macro' })).toBeCloseTo(0.647, 3)
    expect(f1(t, p, { average: 'macro' })).toBeCloseTo(0.619, 3)
    expect(f1(t, p, { average: 'weighted' })).toBeCloseTo(0.828, 3)
    expect(matthewsCorrelation(t, p)).toBeCloseTo(0.507, 3)
    expect(toFlat(confusionMatrix(t, p).matrix)).toEqual([70, 6, 4, 4, 10, 1, 2, 1, 2])
  })
  it('handles multi-label rows', () => {
    const t = [
      [1, 0, 1],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
    ]
    const p = [
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [1, 0, 1],
    ]
    expect(accuracy(t, p)).toBeCloseTo(0.5, 12)
    expect(hammingLoss(t, p)).toBeCloseTo(2 / 12, 12)
    expect(jaccardScore(t, p, { average: 'samples' })).toBeCloseTo(0.75, 12)
    expect(jaccardScore(t, p, { average: 'micro' })).toBeCloseTo(5 / 7, 12)
    expect(jaccardScore(t, p, { average: 'macro' })).toBeCloseTo(0.722, 3)
  })
  it('weights kappa and scores ordinal predictions', () => {
    const table = [
      [20, 5, 1],
      [4, 15, 5],
      [1, 4, 15],
    ]
    expect(kappaFromTable(table)).toBeCloseTo(0.57, 3)
    expect(kappaFromTable(table, 'linear')).toBeCloseTo(0.642, 3)
    expect(kappaFromTable(table, 'quadratic')).toBeCloseTo(0.715, 3)
    const a = fromMatrix([
      [3, 1, 0],
      [1, 2, 1],
      [0, 0, 2],
    ])
    expect(quadraticWeightedKappa(a.yTrue, a.yPred)).toBeCloseTo(0.762, 3)
    expect(macroMeanAbsoluteError(a.yTrue, a.yPred)).toBeCloseTo(0.25, 12)
  })
})

describe('curves', () => {
  const y = [1, 1, 1, 1, 0, 0, 0, 0, 0]
  const s = [0.9, 0.8, 0.6, 0.55, 0.7, 0.5, 0.4, 0.3, 0.2]
  it('computes AUROC, AP and the trapezoid of the worked example', () => {
    expect(auroc(y, s)).toBeCloseTo(0.9, 12)
    expect(rocCurve(y, s).area).toBeCloseTo(0.9, 12)
    expect(averagePrecision(y, s)).toBeCloseTo(0.8875, 12)
    expect(precisionRecallTrapezoid(precisionRecallCurve(y, s))).toBeCloseTo(0.871, 3)
  })
  it('counts ties as one half', () => {
    expect(auroc([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1])).toBeCloseTo(0.875, 12)
    const c = rocCurve([1, 0], [0.5, 0.5])
    expect(toFlat(c.x)).toEqual([0, 1])
    expect(toFlat(c.y)).toEqual([0, 1])
  })
  it('returns every curve as the contract Curve', () => {
    const curves = [
      rocCurve(y, s),
      precisionRecallCurve(y, s),
      detCurve(y, s),
      gainCurve(y, s),
      costCurve(y, s),
      precisionRecallGainCurve(y, s),
      rocConvexHull(rocCurve(y, s)),
      reliabilityDiagram(y, s, { bins: 4 }),
    ]
    expect(curves.map((c) => c.curve)).toEqual(['roc', 'pr', 'det', 'gain', 'cost', 'prg', 'roc', 'reliability'])
    for (const c of curves) {
      expect(c.kind).toBe('curve')
      expect(c.x.shape).toEqual(c.y.shape)
      if (c.thresholds) expect(c.thresholds.shape).toEqual(c.x.shape)
    }
    expect(precisionRecallCurve(y, s).area).toBeCloseTo(0.8875, 12)
    expect(precisionRecallCurve(y, s).prevalence).toBeCloseTo(4 / 9, 12)
    // The hull dominates the curve, so its area is at least the AUROC.
    expect(rocConvexHull(rocCurve(y, s)).area).toBeGreaterThanOrEqual(0.9)
    expect(toFlat(detCurve(y, s).y)).toEqual(toFlat(rocCurve(y, s).y).map((v) => 1 - v))
  })
  it('standardises partial AUROC', () => {
    // A perfect ranking scores 1 whatever the cut-off.
    expect(partialAuroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], { maxFpr: 0.3 })).toBeCloseTo(1, 12)
  })
  it('matches the binormal closed forms', () => {
    expect(binormalAuroc(1.5)).toBeCloseTo(0.856, 3)
    expect(binormalAveragePrecision({ separation: 1.5, prevalence: 0.5 })).toBeCloseTo(0.854, 3)
    expect(binormalAveragePrecision({ separation: 1.5, prevalence: 0.1 })).toBeCloseTo(0.478, 3)
  })
  it('computes multiclass AUROC', () => {
    const t = [0, 1, 2, 0, 1, 2]
    const p = [
      [0.8, 0.1, 0.1],
      [0.2, 0.7, 0.1],
      [0.1, 0.2, 0.7],
      [0.6, 0.3, 0.1],
      [0.3, 0.4, 0.3],
      [0.2, 0.3, 0.5],
    ]
    expect(auroc(t, p)).toBeCloseTo(1, 12)
    expect(auroc(t, p, { multiClass: 'ovo' })).toBeCloseTo(1, 12)
  })
})

describe('probabilistic', () => {
  const p = [0.1, 0.15, 0.25, 0.3, 0.45, 0.55, 0.7, 0.75, 0.85, 0.95]
  const y = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  it('reproduces the log loss, Brier and ECE examples', () => {
    expect(logLoss(y, p)).toBeCloseTo(0.466, 3)
    expect(brierScore(y, p)).toBeCloseTo(0.157, 3)
    expect(expectedCalibrationError(y, p, { bins: 5 })).toBeCloseTo(0.135, 12)
    expect(maximumCalibrationError(y, p, { bins: 5 })).toBeCloseTo(0.225, 12)
    expect(reliabilityDiagram(y, p, { bins: 5 }).ece).toBeCloseTo(0.135, 12)
  })
  it('decomposes the Brier score exactly by distinct forecasts', () => {
    const d = brierDecomposition(y, p)
    expect(d.residual).toBeCloseTo(0, 12)
    expect(d.brier).toBeCloseTo(brierScore(y, p), 12)
  })
  it('scores Gaussian and ensemble forecasts', () => {
    expect(crpsGaussian([0.8], { mean: 0, sd: 1 })).toBeCloseTo(0.476, 3)
    expect(crpsGaussian([0], { mean: 0, sd: 1 })).toBeCloseTo(0.234, 3)
    expect(crpsEnsemble(0.8, [0.3])).toBeCloseTo(0.5, 12)
    expect(crpsEnsemble([1], [[0, 2]])).toBeCloseTo(1 - 0.5, 12)
    expect(intervalScore([2], { lower: -1.2816, upper: 1.2816 }, { alpha: 0.2 })).toBeCloseTo(9.75, 2)
  })
})

describe('regression and forecasting', () => {
  const y = [3, -0.5, 2, 7]
  const p = [2.5, 0, 2, 8]
  it('reproduces the notes', () => {
    expect(meanSquaredError(y, p)).toBeCloseTo(0.375, 12)
    expect(meanAbsoluteError(y, p)).toBeCloseTo(0.5, 12)
    expect(r2Score(y, p)).toBeCloseTo(0.949, 3)
    expect(explainedVariance(y, p)).toBeCloseTo(0.957, 3)
    const a = [1, 2, 4, 8]
    const b = [1.5, 2, 3, 10]
    expect(tweedieDeviance(a, b, { power: 1 })).toBeCloseTo(0.2301, 4)
    expect(tweedieDeviance(a, b, { power: 1.5 })).toBeCloseTo(0.1179, 4)
    expect(tweedieDeviance(a, b, { power: 2 })).toBeCloseTo(0.0705, 4)
    expect(meanSquaredLogError(a, b)).toBeCloseTo(0.035, 3)
    expect(logCoshError(a, b)).toBeCloseTo(0.4697, 4)
    expect(pinballLoss(a, b, { tau: 0.9 })).toBeCloseTo(0.2875, 12)
  })
  it('computes percentage and scaled errors', () => {
    expect(meanAbsolutePercentageError([100, 50, 20, 80], [110, 40, 30, 80])).toBeCloseTo(0.2, 12)
    expect(symmetricMeanAbsolutePercentageError([100, 50, 20, 80], [110, 40, 30, 80])).toBeCloseTo(0.179, 3)
    const train = [10, 12, 11, 13, 15, 14]
    expect(meanAbsoluteScaledError([16, 15], [15, 16], { train })).toBeCloseTo(0.625, 12)
    expect(rootMeanSquaredScaledError([16, 15], [15, 16], { train })).toBeCloseTo(0.6, 2)
  })
  it('normalises RMSE by the range of a long series (review: Math.max(...y) overflowed the stack past ~1e5)', () => {
    const n = 300_000
    const y = Float64Array.from({ length: n }, (_, i) => i / (n - 1))
    const p = Float64Array.from(y, (v) => v + 0.5)
    expect(normalisedRootMeanSquaredError(y, p, { by: 'range' })).toBeCloseTo(0.5, 12)
  })
})

describe('ranking', () => {
  const grades = [3, 2, 3, 0, 1, 2]
  it('reproduces the nDCG example with both gains', () => {
    expect(ndcg(grades, { gain: 'linear' })).toBeCloseTo(0.961, 3)
    expect(ndcg(grades)).toBeCloseTo(0.949, 3)
    expect(ndcg(grades, { gain: 'linear', k: 3 })).toBeCloseTo(0.978, 3)
    expect(dcg(grades, { gain: 'linear' })).toBeCloseTo(6.861, 3)
  })
  it('ranks by scores with tie averaging', () => {
    // Reversed scores rank the items in the listed order.
    expect(ndcg(grades, [6, 5, 4, 3, 2, 1], { gain: 'linear' })).toBeCloseTo(0.961, 3)
    // All tied: the expected DCG is the mean gain times the sum of discounts.
    const discounts = [1, 2, 3].reduce((s, i) => s + 1 / Math.log2(i + 1), 0)
    expect(dcg([1, 0, 2], [0, 0, 0], { gain: 'linear' })).toBeCloseTo(discounts, 12)
  })
  it('computes the retrieval metrics', () => {
    const list = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    expect(precisionAtK(list, { k: 5 })).toBeCloseTo(0.6, 12)
    expect(recallAtK(list, { k: 10, totalRelevant: 5 })).toBeCloseTo(0.8, 12)
    expect(rPrecision(list, { totalRelevant: 5 })).toBeCloseTo(0.6, 12)
    expect(meanAveragePrecision(list, { totalRelevant: 5 })).toBeCloseTo(0.598, 3)
    // AP@2 of a perfect top 2 with R = 3: 2/3 by R, 1 by min(R, k).
    expect(meanAveragePrecision([1, 1, 0, 1], { k: 2 })).toBeCloseTo(2 / 3, 14)
    expect(meanAveragePrecision([1, 1, 0, 1], { k: 2, normaliser: 'cutoff' })).toBe(1)
    const firsts = [
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]
    expect(meanReciprocalRank(firsts)).toBeCloseTo(0.611, 3)
    expect(
      hitRate(
        [
          [1, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
          [0, 0, 0],
        ],
        { k: 2 },
      ),
    ).toBeCloseTo(0.5, 12)
  })
  it('reproduces ERR and the ideal-including nDCG', () => {
    const a = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    const b = [1, 1, 2, 1, 1, 0, 0, 0, 0, 0]
    expect(expectedReciprocalRank(a, { maxGrade: 3 })).toBeCloseTo(0.875, 3)
    expect(expectedReciprocalRank(b, { maxGrade: 3 })).toBeCloseTo(0.301, 3)
    expect(ndcg(a, { gain: 'linear', ideal: [3, 1, 1, 1, 1, 1] })).toBeCloseTo(0.566, 3)
  })
})

describe('clustering', () => {
  const t = [0, 0, 0, 1, 1, 1, 2, 2, 2]
  const p = [0, 0, 1, 1, 1, 1, 2, 2, 0]
  it('reproduces the pair-counting and information-theoretic examples', () => {
    expect(randIndex(t, p)).toBeCloseTo(0.75, 12)
    expect(adjustedRandIndex(t, p)).toBeCloseTo(0.357, 3)
    expect(fowlkesMallows(t, p)).toBeCloseTo(0.527, 3)
    expect(vMeasure(t, p)).toBeCloseTo(0.59, 3)
    expect(normalisedMutualInformation(t, p, { average: 'geometric' })).toBeCloseTo(0.59, 3)
    expect(adjustedMutualInformation(t, p)).toBeCloseTo(0.409, 3)
  })
  it('reproduces the internal indices on five points', () => {
    const x = [[0], [1], [5], [6], [7]]
    const l = [0, 0, 1, 1, 1]
    expect(silhouetteScore(x, l)).toBeCloseTo(0.777, 3)
    expect(calinskiHarabasz(x, l)).toBeCloseTo(43.56, 2)
    expect(daviesBouldin(x, l)).toBeCloseTo(0.212, 3)
    expect(dunnIndex(x, l)).toBeCloseTo(2, 12)
  })
})

describe('agreement', () => {
  it('reproduces the notes', () => {
    expect(concordanceCorrelation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])).toBeCloseTo(0.421, 3)
    expect(
      fleissKappa([
        [4, 0, 0],
        [2, 2, 0],
        [0, 3, 1],
        [1, 1, 2],
        [0, 0, 4],
      ]),
    ).toBeCloseTo(0.398, 3)
    const table = [
      [20, 10, 5],
      [10, 20, 15],
    ]
    expect(cramersV(table)).toBeCloseTo(0.364, 3)
    expect(theilsU(table)).toBeCloseTo(0.062, 3)
    expect(theilsU(table, { of: 'rows' })).toBeCloseTo(0.098, 3)
  })
  it("reproduces Krippendorff's reliability-data example", () => {
    const n = NaN
    const data = [
      [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
      [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
      [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
      [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
    ]
    expect(krippendorffAlpha(data)).toBeCloseTo(0.743, 3)
    expect(krippendorffAlpha(data, { level: 'interval' })).toBeCloseTo(0.849, 3)
  })
  it('reproduces Shrout and Fleiss', () => {
    const r = [
      [9, 2, 5, 8],
      [6, 1, 3, 2],
      [8, 4, 6, 8],
      [7, 1, 2, 6],
      [10, 5, 6, 9],
      [6, 2, 4, 7],
    ]
    expect(intraclassCorrelation(r, { form: 'ICC1' })).toBeCloseTo(0.17, 2)
    expect(intraclassCorrelation(r, { form: 'ICC2' })).toBeCloseTo(0.29, 2)
    expect(intraclassCorrelation(r, { form: 'ICC3' })).toBeCloseTo(0.71, 2)
    expect(intraclassCorrelation(r, { form: 'ICC3k' })).toBeCloseTo(0.91, 2)
  })
})

describe('distances', () => {
  it('reproduces the distance examples', () => {
    expect(minkowskiDistance([1, 2, 3], [4, 0, 3], { p: 3 })).toBeCloseTo(3.271, 3)
    expect(cosineSimilarity([1, 2, 3], [4, 0, 3])).toBeCloseTo(0.695, 3)
    const h = hausdorffDistances(
      [
        [0, 0],
        [1, 0],
        [2, 0],
      ],
      [
        [0, 1],
        [1, 1],
        [2, 1],
        [5, 1],
      ],
    )
    expect(h.directedXY).toBeCloseTo(1, 12)
    expect(h.hausdorff).toBeCloseTo(Math.sqrt(10), 12)
    const square = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 2],
    ]
    const moved = square.map(([x, y]) => [
      2 * (x * Math.cos(0.5) - y * Math.sin(0.5)) + 3,
      2 * (x * Math.sin(0.5) + y * Math.cos(0.5)) - 1,
    ])
    expect(procrustesDisparity(square, moved)).toBeCloseTo(0, 10)
  })
})

describe('uncertainty', () => {
  it('reproduces the Wilson intervals and bootstraps deterministically', () => {
    const [lo, hi] = wilsonInterval(900, 1000)
    expect(lo).toBeCloseTo(0.88, 3)
    expect(hi).toBeCloseTo(0.917, 3)
    const { yTrue, yPred } = screening()
    const a = bootstrapMetric(stream(1), (t: number[], p: number[]) => f1(t, p), yTrue, yPred, { resamples: 200 })
    const b = bootstrapMetric(stream(1), (t: number[], p: number[]) => f1(t, p), yTrue, yPred, { resamples: 200 })
    expect(a.interval).toEqual(b.interval)
    expect(a.interval[0]).toBeLessThan(0.444)
    expect(a.interval[1]).toBeGreaterThan(0.444)
  })
})

describe('registry', () => {
  it('lists every metric with metadata', () => {
    const all = Object.values(metricRegistry)
    expect(all.length).toBeGreaterThan(100)
    for (const m of all) {
      expect(m.info.kind).toBe('metric')
      expect(m.info.notes?.length, m.info.key).toBeGreaterThan(0)
      for (const n of m.info.notes ?? []) expect(n).toMatch(/^[a-z0-9-]+$/)
      expect(m.info.range[0]).toBeLessThanOrEqual(m.info.range[1])
    }
    expect(getMetric('auroc')).toBe(auroc)
    expect(listMetrics({ capability: 'predictive' })).toContain(logLoss)
    expect(listMetrics({ note: 'ordinal-classification-metrics' })).toContain(ordinalMeanAbsoluteError)
    expect(() => getMetric('nope')).toThrow()
  })
  it('compares values by direction', () => {
    expect(isBetter(accuracy, 0.9, 0.8)).toBe(true)
    expect(isBetter(meanSquaredError, 0.9, 0.8)).toBe(false)
    expect(isBetter(accuracy, NaN, 0.1)).toBe(false)
    expect(isBetter(accuracy, 0.1, NaN)).toBe(true)
  })
})

describe('operating points, Procrustes and uncertainty', () => {
  it('finds operating points', () => {
    const y = [1, 1, 1, 1, 0, 0, 0, 0]
    const s = [0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.05]
    expect(equalErrorRate(y, s).rate).toBeCloseTo(0.25, 12)
    expect(youdenPoint(y, s).j).toBeCloseTo(0.75, 12)
    expect(operatingPoint(y, s, { maxFpr: 0 }).tpr).toBeCloseTo(0.75, 12)
    expect(costCurve(y, s).area).toBeGreaterThan(0)
    expect(toFlat(gainCurve(y, s).y).at(-1)).toBe(1)
  })
  it('recovers a Procrustes rotation', () => {
    const a = Math.PI / 6
    const x = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 2],
    ]
    const y2 = x.map(([u, v]) => [u * Math.cos(a) - v * Math.sin(a), u * Math.sin(a) + v * Math.cos(a)])
    const r = toFlat(orthogonalProcrustes(x, y2).rotation)
    expect(Math.atan2(r[2], r[0])).toBeCloseTo(a, 10)
  })
  it('computes PIT values, the DeLong test and consistency bars', () => {
    expect(toFlat(pitValues([0], { mean: 0, sd: 1 }))[0]).toBeCloseTo(0.5, 12)
    const y = [1, 0, 1, 0, 1, 0]
    const s = [0.9, 0.2, 0.6, 0.7, 0.8, 0.1]
    expect(delongTest(y, s, s).difference).toBe(0)
    const bars = consistencyBars(stream(3), [0.1, 0.5, 0.9, 0.95], { bins: 2, resamples: 50 })
    expect(bars.lower.shape).toEqual([2])
  })
})

describe('distances agree with numerics/linalg', () => {
  const x = [
    [1, 2, 3],
    [4, 0, 3],
    [-1, 0.5, 2],
  ]
  const at = (d: Tensor, i: number, j: number) => toFlat(d)[i * 3 + j]
  it('Minkowski family and cosine', () => {
    const cases = [
      [euclideanDistance, pairwiseDistances(x)],
      [manhattanDistance, pairwiseDistances(x, x, { metric: 'manhattan' })],
      [chebyshevDistance, pairwiseDistances(x, x, { metric: 'chebyshev' })],
      [cosineDistance, pairwiseDistances(x, x, { metric: 'cosine' })],
    ] as const
    for (const [metric, table] of cases)
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) expect(metric(x[i], x[j]), metric.info.key).toBeCloseTo(at(table, i, j), 12)
    const p3 = pairwiseDistances(x, x, { metric: 'minkowski', p: 3 })
    expect(minkowskiDistance(x[0], x[1], { p: 3 })).toBeCloseTo(at(p3, 0, 1), 12)
    const sq = squaredDistances(x)
    expect(euclideanDistance(x[0], x[2]) ** 2).toBeCloseTo(at(sq, 0, 2), 12)
  })
  it('angular and Mahalanobis', () => {
    expect(angularDistance([1, 0], [0, 1])).toBeCloseTo(0.5, 12)
    expect(angularDistance([1, 0], [-1, 0])).toBeCloseTo(1, 12)
    // Σ = diag(4, 1): the first coordinate is scaled by 1/2.
    const cov = [
      [4, 0],
      [0, 1],
    ]
    expect(mahalanobisDistance([0, 0], [2, 1], { covariance: cov })).toBeCloseTo(Math.SQRT2, 12)
    expect(hausdorffDistance([[0, 0]], [[3, 4]])).toBeCloseTo(5, 12)
  })
})

describe('ordinal', () => {
  // Grades 0 < 1 < 2: errors of 0, 1 and 2 steps.
  const t = [0, 0, 1, 1, 2, 2]
  const p = [0, 1, 1, 2, 0, 2]
  it('mean absolute error in class steps', () => {
    expect(ordinalMeanAbsoluteError(t, p)).toBeCloseTo((0 + 1 + 0 + 1 + 2 + 0) / 6, 12)
    // Class-wise MAE: class 0 → 0.5, class 1 → 0.5, class 2 → 1.
    expect(macroMeanAbsoluteError(t, p)).toBeCloseTo((0.5 + 0.5 + 1) / 3, 12)
  })
  it('accuracy within a tolerance', () => {
    expect(withinToleranceAccuracy(t, p)).toBeCloseTo(5 / 6, 12)
    expect(withinToleranceAccuracy(t, p, { tolerance: 0 })).toBeCloseTo(accuracy(t, p), 12)
    expect(withinToleranceAccuracy(t, p, { tolerance: 2 })).toBe(1)
  })
  it('uses the given label order and rejects unknown labels', () => {
    const tt = ['low', 'high', 'mid']
    const pp = ['mid', 'high', 'low']
    const labels = ['low', 'mid', 'high']
    expect(ordinalMeanAbsoluteError(tt, pp, { labels })).toBeCloseTo((1 + 0 + 1) / 3, 12)
    expect(() => ordinalMeanAbsoluteError(tt, pp, { labels: ['low', 'mid'] })).toThrow()
    expect(() => ordinalMeanAbsoluteError([0, 1], [0])).toThrow()
  })
  it('quadratic weighted kappa is 1 for perfect agreement', () => {
    expect(quadraticWeightedKappa(t, t)).toBeCloseTo(1, 12)
  })
})
