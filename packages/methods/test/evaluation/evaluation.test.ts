/**
 * aifn-methods/evaluation: hand-checked cases of the detection, segmentation, text, quality, generative, fairness and
 * beyond-accuracy metrics, mostly the worked examples of the site's notes.
 */
import { describe, expect, it } from 'vitest'
import {
  averagePrecisionFromMatches,
  boundaryF1,
  boxOverlap,
  detectionMeanAveragePrecision,
  dice,
  matchDetections,
  meanIou,
  panopticFromMatches,
  panopticQuality,
} from 'aifn-methods/evaluation/detection'
import {
  bleuScore,
  chrFScore,
  rougeLScores,
  rougeNScores,
  squadExactMatch,
  squadF1,
  translationEditRate,
  wordErrorRate,
  editAlignment,
  bertScore,
  backretrieval,
} from 'aifn-methods/evaluation/text'
import { demographicParityDifference, equalisedOddsDifference } from 'aifn-methods/evaluation/fairness'
import {
  frechetDistance,
  generativePrecisionRecall,
  inceptionScore,
  inceptionScoreSplits,
  kid,
  kidSubsets,
} from 'aifn-methods/evaluation/generative'
import { DomainError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'
import { giniCoefficient } from 'aifn-methods/evaluation/beyond-accuracy'
import {
  psnr,
  siSdr,
  snr,
  ssimMap,
  ergas,
  spectralAngle,
  permutationInvariantScore,
} from 'aifn-methods/evaluation/quality'

describe('detection and segmentation', () => {
  it('reproduces the box and AP examples', () => {
    const o = boxOverlap([2, 2, 6, 6], [0, 0, 4, 4])
    expect(o.iou).toBeCloseTo(4 / 28, 12)
    expect(o.giou).toBeCloseTo(-0.079, 3)
    expect(o.diou).toBeCloseTo(0.032, 3)
    const hits = [true, false, true, true, false]
    expect(averagePrecisionFromMatches(hits, 4).averagePrecision).toBeCloseTo(0.625, 12)
    expect(averagePrecisionFromMatches(hits, 4, '11-point').averagePrecision).toBeCloseTo(0.614, 3)
    expect(averagePrecisionFromMatches(hits, 4, '101-point').averagePrecision).toBeCloseTo(0.629, 3)
    const pq = panopticFromMatches([0.9, 0.7, 0.6], 1, 2)
    expect(pq.pq).toBeCloseTo(0.489, 3)
    expect(dice([1, 1, 0, 0], [1, 0, 1, 0])).toBeCloseTo(0.5, 12)
  })
  it('evaluates detections and segments', () => {
    const gt = [
      { image: 0, box: [0, 0, 2, 2] as const },
      { image: 0, box: [5, 5, 7, 7] as const },
    ]
    const det = [
      { image: 0, box: [0, 0, 2, 2] as const, score: 0.9 },
      { image: 0, box: [0, 0, 2, 2] as const, score: 0.8 },
      { image: 0, box: [5, 5, 7, 7] as const, score: 0.7 },
    ]
    expect(matchDetections(gt, det).map((d) => d.outcome)).toEqual(['tp', 'fp', 'tp'])
    expect(detectionMeanAveragePrecision(gt, det, { iouThresholds: [0.5], interpolation: 'all-points' })).toBeCloseTo(
      0.5 + 0.5 * (2 / 3),
      12,
    )
    const truth = [1, 1, 2, 2, 0, 0]
    const pred = [5, 5, 6, 6, 6, 0]
    const pq = panopticQuality(truth, pred)
    // 1↔5 (IoU 1) and 2↔6 (IoU 2/3) match; truth 0 and prediction 0 overlap at IoU 0.5, not above it.
    expect(pq.rq).toBeCloseTo(2 / 3, 12)
    expect(pq.pq).toBeCloseTo((1 + 2 / 3) / 3, 12)
    const mask = [
      [0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0],
    ]
    expect(boundaryF1(mask, mask)).toBe(1)
    expect(meanIou([0, 0, 1, 1], [0, 1, 1, 1])).toBeCloseTo((0.5 + 2 / 3) / 2, 12)
  })
})

describe('text', () => {
  it('scores embeddings with BERTScore and back-retrieval', () => {
    const e = [
      [1, 0],
      [0, 1],
    ]
    expect(bertScore(e, e).f).toBeCloseTo(1, 12)
    expect(
      backretrieval(
        [
          [1, 0],
          [0, 1],
        ],
        [
          [1, 0.2],
          [0.1, 1],
        ],
        { k: 1 },
      ),
    ).toBe(1)
  })
  it('reproduces the BLEU and chrF examples', () => {
    const ref = 'the cat is on the mat'
    const cand = 'the cat sat on the mat'
    expect(bleuScore(ref, cand).score).toBe(0)
    expect(bleuScore(ref, cand, { smoothing: 'add-one' }).score).toBeCloseTo(0.485, 3)
    const corpus = bleuScore([ref, 'there is a dog in the yard'], [cand, 'there is a dog in the garden'])
    expect(corpus.score).toBeCloseTo(0.619, 3)
    expect(bleuScore(ref, 'the cat is on').score).toBeCloseTo(0.607, 3)
    const c = chrFScore(ref, cand)
    expect(c.precision).toBeCloseTo(0.611, 3)
    expect(c.recall).toBeCloseTo(0.655, 3)
    expect(c.score).toBeCloseTo(0.646, 3)
  })
  it('reproduces ROUGE, WER, TER and SQuAD', () => {
    expect(rougeNScores('the cat is on the mat', 'the cat sat on the mat', { n: 2 }).f).toBeCloseTo(0.6, 12)
    expect(rougeLScores('police killed the gunman', 'the gunman killed police').f).toBeCloseTo(0.5, 12)
    const ref = 'the quick brown fox jumps over the lazy dog'
    const hyp = 'the quick brown fox jumped over a lazy dog today'
    expect(wordErrorRate(ref, hyp)).toBeCloseTo(1 / 3, 12)
    expect(editAlignment(ref.split(' '), hyp.split(' ')).insertions).toBe(1)
    expect(translationEditRate('the cat sat on the mat', 'on the mat the cat sat')).toBeCloseTo(1 / 6, 12)
    expect(squadExactMatch([['the Eiffel Tower', 'Eiffel Tower in Paris']], ['Eiffel Tower, Paris'])).toBe(0)
    expect(squadF1([['the Eiffel Tower', 'Eiffel Tower in Paris']], ['Eiffel Tower, Paris'])).toBeCloseTo(0.857, 3)
  })
})

describe('image, audio and generative', () => {
  it('reproduces SNR, SI-SDR and PIT', () => {
    expect(snr([1, 2, 3, 4], [2.5, 3.5, 6.5, 7.5])).toBeCloseTo(0.147, 3)
    expect(siSdr([1, 2, 3, 4], [2.5, 3.5, 6.5, 7.5])).toBeCloseTo(20.8, 1)
    const r = permutationInvariantScore(
      [
        [1, 2, 3, 4],
        [4, -1, 2, 0],
      ],
      [
        [4, -1, 2, 0.1],
        [1, 2, 3, 4.1],
      ],
    )
    expect(r.permutation).toEqual([1, 0])
  })
  it('reproduces PSNR, SSIM and the remote-sensing example', () => {
    const a = Array.from({ length: 64 }, (_, i) => (i * 37) % 255)
    const b = a.map((v) => v + 10)
    expect(psnr(a, b, { dataRange: 255 })).toBeCloseTo(10 * Math.log10(255 ** 2 / 100), 10)
    expect(ssimMap(a, a, { dataRange: 255, width: 8 }).mean).toBeCloseTo(1, 12)
    const ref = [
      [100, 80, 60],
      [90, 85, 70],
    ]
    const est = [
      [98, 84, 58],
      [95, 80, 72],
    ]
    expect((spectralAngle(ref, est) * 180) / Math.PI).toBeCloseTo(2.44, 2)
    expect(ergas(ref, est, { ratio: 0.25 })).toBeCloseTo(1.08, 2)
  })
  it('reproduces FID, KID, IS and precision/recall', () => {
    expect(
      frechetDistance(
        [0, 0],
        [
          [1, 0.5],
          [0.5, 1],
        ],
        [1, 0.5],
        [
          [2, 0],
          [0, 0.5],
        ],
      ).distance,
    ).toBeCloseTo(1.636, 3)
    expect(kid([[0], [1]], [[1], [2]])).toBeCloseTo(9.5, 12)
    const p = [
      [0.9, 0.05, 0.05],
      [0.05, 0.9, 0.05],
      [0.05, 0.05, 0.9],
    ]
    expect(inceptionScore(p)).toBeCloseTo(2.02, 2)
    const pr = generativePrecisionRecall(
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      [
        [0.1, 0.1],
        [0.9, 0.1],
        [3, 3],
      ],
      { k: 1 },
    )
    expect(pr.precision).toBeCloseTo(2 / 3, 12)
    expect(pr.recall).toBeCloseTo(3 / 4, 12)
  })
  it('validates splits, subsetSize and allows negative PSNR', () => {
    expect(psnr([10], [20], { dataRange: 5 })).toBeLessThan(0)
    expect(() => inceptionScoreSplits([[0.5, 0.5]], { splits: 2 })).toThrow(DomainError)
    expect(() => inceptionScoreSplits([[0.5, 0.5]], { splits: 0 })).toThrow(DomainError)
    expect(() => kidSubsets(stream(1), [[1]], [[2]], { subsetSize: 5 })).toThrow(DomainError)
  })
})

describe('fairness and beyond accuracy', () => {
  it('reproduces the two-group example', () => {
    const yTrue: number[] = []
    const yPred: number[] = []
    const groups: string[] = []
    const add = (g: string, tp: number, fn: number, fp: number, tn: number) => {
      for (const [t, p, n] of [
        [1, 1, tp],
        [1, 0, fn],
        [0, 1, fp],
        [0, 0, tn],
      ])
        for (let i = 0; i < n; i++) {
          yTrue.push(t)
          yPred.push(p)
          groups.push(g)
        }
    }
    add('A', 30, 10, 10, 50)
    add('B', 12, 8, 8, 72)
    expect(demographicParityDifference(yTrue, yPred, { groups })).toBeCloseTo(0.2, 12)
    expect(equalisedOddsDifference(yTrue, yPred, { groups })).toBeCloseTo(0.15, 12)
  })
  it('computes the Gini coefficient of exposure', () => {
    expect(giniCoefficient([50, 30, 10, 5, 5])).toBeCloseTo(0.46, 12)
  })
})
