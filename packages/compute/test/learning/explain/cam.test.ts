/**
 * aifn-compute/learning/explain class activation maps: the 4-channel 3 x 3 worked example of the
 * class-activation-mapping note (CAM by hand, its mean the class score), Grad-CAM weights of a pooling-and-linear
 * head, Grad-CAM++ weights and `normaliseMap` (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { tensor } from 'aifn-compute/foundation/tensor'
import {
  classActivationMap,
  featureMaps,
  gradCamPlusPlusWeights,
  gradCamWeights,
  normaliseMap,
  weightedMapSum,
  type FeatureMaps,
} from 'aifn-compute/learning/explain'

const maps: FeatureMaps = {
  data: Float64Array.from([
    ...[2, 1, 0, 1, 0, 0, 0, 0, 0],
    ...[1, 1, 0, 0, 0, 0, 0, 0, 0],
    ...[0, 0, 0, 0, 0, 1, 0, 1, 2],
    ...[0, 0, 0, 0, 1, 0, 0, 0, 1],
  ]),
  channels: 4,
  height: 3,
  width: 3,
}
const mean = (a: Float64Array) => a.reduce((s, v) => s + v, 0) / a.length

describe('classActivationMap', () => {
  it('gives the worked example by hand, with mean the class score', () => {
    const disc = classActivationMap(maps, [1, 0.5, -1, 0])
    const cross = classActivationMap(maps, [-0.5, 0, 1, 1])
    expect(Array.from(disc)).toEqual([2.5, 1.5, 0, 1, 0, -1, 0, -1, -2])
    expect(Array.from(cross)).toEqual([-1, -0.5, 0, -0.5, 1, 1, 0, 1, 3])
    expect(mean(disc)).toBeCloseTo(1 / 9, 12)
    expect(mean(cross)).toBeCloseTo(4 / 9, 12)
  })

  it('is weightedMapSum without ReLU', () => {
    expect(Array.from(weightedMapSum(maps, [1, 0.5, -1, 0], true))).toEqual([2.5, 1.5, 0, 1, 0, 0, 0, 0, 0])
    expect(() => weightedMapSum(maps, [1, 2])).toThrow(AifnError)
  })
})

describe('Grad-CAM weights', () => {
  it('of a pooling-and-linear head are the classifier weights over hw', () => {
    const grads: FeatureMaps = {
      ...maps,
      data: Float64Array.from({ length: 36 }, (_, i) => [-0.5, 0, 1, 1][Math.floor(i / 9)] / 9),
    }
    const w = gradCamWeights(grads)
    ;[-0.5 / 9, 0, 1 / 9, 1 / 9].forEach((v, k) => expect(w[k]).toBeCloseTo(v, 15))
  })

  it('of Grad-CAM++ count positive gradients only', () => {
    const a: FeatureMaps = { data: Float64Array.of(1, 1, 0, 0, 1, 1, 0, 0), channels: 2, height: 2, width: 2 }
    const g: FeatureMaps = { data: Float64Array.of(1, 1, 1, 1, -1, -1, -1, -1), channels: 2, height: 2, width: 2 }
    const w = gradCamPlusPlusWeights(a, g)
    // Channel 0: a = 1 / (2 + 2) per position, four positions of gradient 1.
    expect(w[0]).toBeCloseTo(1, 15)
    expect(w[1]).toBe(0)
  })
})

describe('featureMaps and normaliseMap', () => {
  it('read one example and scale by the largest magnitude', () => {
    const m = featureMaps(
      tensor([
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [0, 0],
            [1, 1],
          ],
        ],
      ]),
    )
    expect([m.channels, m.height, m.width]).toEqual([2, 2, 2])
    expect(Array.from(m.data)).toEqual([1, 2, 3, 4, 0, 0, 1, 1])
    expect(() => featureMaps(tensor([[[[1]]], [[[2]]]]))).toThrow(AifnError)
    expect(Array.from(normaliseMap([2, -4, 1]))).toEqual([0.5, -1, 0.25])
    expect(Array.from(normaliseMap([0, 0]))).toEqual([0, 0])
  })
})
