/**
 * aifn-compute/learning/validate, spatial splitters and cross-validation's class handling: `spatialBlocks` and
 * `buffered` against brute force in numpy; `nndmLeaveOneOut` and `knndm` against numpy ports of CAST's `nndm` and
 * `knndm` (scipy's Ward linkage, Kolmogorov-Smirnov test and Wasserstein distance); `crossValidate` with classes from
 * all of `y` against scikit-learn's recall (`labels=[0, 1]`, `pos_label=1`, `zero_division=nan`), and its pointwise
 * values against per-row log loss, Brier score and accuracy.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, full, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  bernoulliPredictive,
  dataset,
  type Decides,
  type Estimator,
  type Predicts,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import {
  accuracy,
  brierScore,
  logLoss,
  meanSquaredError,
  recall,
  rootMeanSquaredError,
} from 'aifn-compute/learning/metrics'
import {
  buffered,
  crossValidate,
  groupKFold,
  kFold,
  knndm,
  leaveOneOut,
  nndmLeaveOneOut,
  spatialBlocks,
} from 'aifn-compute/learning/validate'
import { fixture } from '../../fixtures'

interface NndmCase {
  coords: number[][]
  prediction: number[][]
  phi: number
  minTrain: number
  excluded: number[][]
  gij: number[]
  gj: number[]
  gstar: number[]
}
interface KnndmCase {
  coords: number[][]
  prediction: number[][]
  k: number
  maxp: number
  folds: number[]
  W: number
  q: number
  ksPValue: number
  gij: number[]
  gj: number[]
  gstar: number[]
}
interface Fixture {
  spatial: {
    blocks: { coords: number[][]; byBlocks: number[]; bySize: number[] }
    buffered: { coords: number[][]; radius: number; looTrain: number[][]; kFoldTrain: number[][] }
    nndm: NndmCase[]
    knndm: KnndmCase[]
  }
  classes: {
    x: number[][]
    y: number[]
    recall: number[]
    logLoss: number[]
    brierScore: number[]
    accuracy: number[]
    foldLogLoss: number[]
  }
}

const F = fixture<Fixture>('learning/validate')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(b[i])) expect(a[i]).toBeNaN()
    else expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
  }
}

describe('spatial splitters', () => {
  it('spatialBlocks labels grid cells by count and by size', () => {
    const { coords, byBlocks, bySize } = F.spatial.blocks
    expect(spatialBlocks({ coords, blocks: [3, 2] })).toEqual(byBlocks)
    expect(spatialBlocks({ coords, size: 1.5, origin: [0, 0] })).toEqual(bySize)
    const groups = spatialBlocks({ coords, blocks: 2 })
    for (const s of groupKFold({ k: 2 }).split({ n: coords.length, groups })) {
      const tested = new Set(toFlat(s.test).map((i) => groups[i]))
      for (const i of toFlat(s.train)) expect(tested.has(groups[i])).toBe(false)
    }
    expect(() => spatialBlocks({ coords })).toThrow(/one of size and blocks/)
    expect(() => spatialBlocks({ coords, size: 1, origin: [1, 1] })).toThrow(/below the origin/)
  })

  it('buffered drops training rows within the radius, around leave-one-out and k-fold', () => {
    const { coords, radius, looTrain, kFoldTrain } = F.spatial.buffered
    const n = coords.length
    const loo = buffered(leaveOneOut(), coords, { radius }).split({ n })
    expect(loo.map((s) => toFlat(s.train))).toEqual(looTrain)
    expect(loo.map((s) => toFlat(s.test))).toEqual(Array.from({ length: n }, (_, i) => [i]))
    expect(
      buffered(kFold({ k: 4 }), coords, { radius })
        .split({ n })
        .map((s) => toFlat(s.train)),
    ).toEqual(kFoldTrain)
    expect(() => buffered(leaveOneOut(), coords, { radius: 100 }).split({ n })).toThrow(/no training rows/)
    expect(() => buffered(leaveOneOut(), coords, { radius }).split({ n: n - 1 })).toThrow(/coordinates/)
  })

  it.each(F.spatial.nndm.map((c, i) => [i, c] as const))('nndmLeaveOneOut matches CAST::nndm (case %i)', (i, c) => {
    // Case 0 uses the default phi (the largest distance), case 1 a given one.
    const s = nndmLeaveOneOut(c.coords, c.prediction, { phi: i === 0 ? undefined : c.phi, minTrain: c.minTrain })
    expect(s.phi).toBeCloseTo(c.phi, 12)
    expect(s.excluded).toEqual(c.excluded)
    close(toFlat(s.nearest.prediction), c.gij, 1e-14)
    close(toFlat(s.nearest.sample), c.gj, 1e-14)
    close(toFlat(s.nearest.cv), c.gstar, 1e-14)
    const n = c.coords.length
    const splits = s.split({ n })
    splits.forEach((split, j) => {
      expect(toFlat(split.test)).toEqual([j])
      expect(toFlat(split.train)).toEqual(
        Array.from({ length: n }, (_, k) => k).filter((k) => k !== j && !c.excluded[j].includes(k)),
      )
    })
  })

  it.each(F.spatial.knndm.map((c) => [c.k, c] as const))('knndm matches CAST::knndm (k = %i)', (_, c) => {
    const s = knndm(c.coords, c.prediction, { k: c.k, maxp: c.maxp })
    expect(toFlat(s.folds)).toEqual(c.folds)
    expect(s.clusters).toBe(c.q)
    expect(s.W).toBeCloseTo(c.W, 12)
    // Both p-values are far below 0.05, the only use made of them; in this far tail they differ (scipy 2e-19).
    expect(s.ksPValue).toBeLessThan(1e-10)
    expect(c.ksPValue).toBeLessThan(1e-10)
    close(toFlat(s.nearest.prediction), c.gij, 1e-14)
    close(toFlat(s.nearest.sample), c.gj, 1e-14)
    close(toFlat(s.nearest.cv), c.gstar, 1e-14)
    const splits = s.split({ n: c.coords.length })
    expect(splits.length).toBe(c.k)
    splits.forEach((split, f) => expect(toFlat(split.test).every((i) => c.folds[i] === f)).toBe(true))
  })

  it('knndm falls back to random folds when samples are as spread as the prediction points', () => {
    const coords = toRows(
      fromData(
        Float64Array.from({ length: 60 }, (_, i) => ((i * 7919) % 101) / 10),
        [30, 2],
      ),
    )
    const s = knndm(coords, coords, { k: 3, stream: stream('knndm') })
    expect(s.clusters).toBe('random')
    expect(s.ksPValue).toBeGreaterThanOrEqual(0.05)
    expect(toFlat(s.folds)).toEqual(toFlat(knndm(coords, coords, { k: 3, stream: stream('knndm') }).folds))
    expect(() => knndm(coords, coords, { k: 3 })).toThrow(/stream/)
    expect(() => knndm(coords, coords, { k: 3, maxp: 0.3 })).toThrow(/maxp/)
  })
})

/** Nearest centroid on two classes; P(y = 1 | x) = σ((‖x − μ₀‖² − ‖x − μ₁‖²) / 2). */
type Centroid = Decides<Tensor, Tensor> & Predicts<Tensor, ReturnType<typeof bernoulliPredictive>>
const centroid = (): Estimator<Supervised<Tensor, Tensor>, Centroid> => ({
  name: 'centroid',
  fit({ x, y }) {
    const rows = toRows(x)
    const labels = toFlat(y)
    const centre = (c: number) => {
      const members = rows.filter((_, i) => labels[i] === c)
      return members[0].map((_, j) => members.reduce((s, r) => s + r[j], 0) / members.length)
    }
    const [m0, m1] = [centre(0), centre(1)]
    const d2 = (r: number[], m: number[]) => r.reduce((s, v, j) => s + (v - m[j]) ** 2, 0)
    const p = (z: Tensor) =>
      fromData(
        Float64Array.from(toRows(z), (r) => 1 / (1 + Math.exp(-(d2(r, m0) - d2(r, m1)) / 2))),
        [z.shape[0]],
      )
    return {
      decide: (z) =>
        fromData(
          Int32Array.from(toFlat(p(z)), (q) => (q >= 0.5 ? 1 : 0)),
          [z.shape[0]],
        ),
      predictive: (z) => bernoulliPredictive(p(z)),
    }
  },
})

describe('crossValidate: classes from all of y, and pointwise values', () => {
  const c = F.classes
  const data = { kind: 'dataset', x: tensor(c.x), y: tensor(c.y) } as const

  it('scores every fold for the positive class of the whole data set', () => {
    const cv = crossValidate(centroid(), data, kFold({ k: 3 }), [recall, logLoss])
    close(toFlat(cv.scores.recall), c.recall, 1e-14)
    close(toFlat(cv.scores.logLoss), c.foldLogLoss, 1e-12)
    expect(cv.classes).toEqual({ labels: [0, 1], positive: 1 })
  })

  it('classes per fold reproduce the old behaviour: a fold of negatives scores recall for class 0', () => {
    const zeros = { name: 'zeros', fit: () => ({ decide: (z: Tensor) => full([z.shape[0]], 0) }) }
    const d = dataset(tensor([[0], [1], [2], [3]]), tensor([1, 1, 0, 0]))
    expect(toFlat(crossValidate(zeros, d, kFold({ k: 2 }), [recall]).scores.recall)).toEqual([0, NaN])
    expect(toFlat(crossValidate(zeros, d, kFold({ k: 2 }), [recall], { classes: 'fold' }).scores.recall)).toEqual([
      0, 1,
    ])
    expect(toFlat(crossValidate(zeros, d, kFold({ k: 2 }), [recall], { positive: 0 }).scores.recall)).toEqual([NaN, 1])
  })

  it('keeps each row’s held-out value of the metrics that are means over rows', () => {
    const cv = crossValidate(centroid(), data, kFold({ k: 3 }), [recall, logLoss, brierScore, accuracy], {
      pointwise: true,
    })
    expect(Object.keys(cv.pointwise!)).toEqual(['logLoss', 'brierScore', 'accuracy'])
    close(toFlat(cv.pointwise!.logLoss), c.logLoss, 1e-12)
    close(toFlat(cv.pointwise!.brierScore), c.brierScore, 1e-14)
    close(toFlat(cv.pointwise!.accuracy), c.accuracy, 0)
    cv.folds.forEach((fold) => {
      const v = toFlat(fold.pointwise!.logLoss)
      expect(v.reduce((a, b) => a + b, 0) / v.length).toBeCloseTo(fold.metrics.logLoss, 12)
    })
    // Without the option, nothing changes.
    expect(crossValidate(centroid(), data, kFold({ k: 3 }), [logLoss]).pointwise).toBeUndefined()
  })

  it('regression: squared errors per row, and RMSE left out', () => {
    const meanModel = {
      name: 'mean',
      fit: (d: { y: Tensor }) => {
        const m = toFlat(d.y).reduce((a, b) => a + b, 0) / d.y.shape[0]
        return { decide: (z: Tensor) => full([z.shape[0]], m) }
      },
    }
    const d = dataset(tensor([[0], [1], [2], [3], [4], [5]]), tensor([1, 2, 3, 4, 5, 7]))
    const cv = crossValidate(meanModel, d, kFold({ k: 3 }), [meanSquaredError, rootMeanSquaredError], {
      pointwise: true,
    })
    expect(Object.keys(cv.pointwise!)).toEqual(['meanSquaredError'])
    expect(cv.classes).toBeUndefined()
    close(toFlat(cv.pointwise!.meanSquaredError), [14.0625, 7.5625, 0.5625, 0.0625, 6.25, 20.25], 1e-14)
  })
})
