/**
 * Evaluation of explanations by its laws: for a linear model and the attribution w ⊙ x, deletion and insertion curves
 * sum to f(x) + f(baseline) at every step and faithfulness correlation is 1; the true order deletes faster than its
 * reverse; cascading randomisation replaces the top k layers and keeps the rest; similarity is ±1 for equal and
 * reversed rankings; a linear explanation map 2x has local Lipschitz constant 2.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { dense, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  deletionCurve,
  explanationSimilarity,
  faithfulnessCorrelation,
  localLipschitz,
  randomiseNetwork,
  relevanceMass,
  type DenseNetwork,
} from 'aifn-compute/learning/explain'

const w = [2, -1, 0.5, 3, 0, -2]
const f = (X: Tensor) => {
  const x = dense.data(X)
  return Float64Array.from({ length: X.shape[0] }, (_, i) => w.reduce((s, v, j) => s + v * x[i * 6 + j], 0.7))
}
const x = [1, 1, -2, 0.5, 4, -1]
const attribution = w.map((v, j) => v * x[j])

describe('deletion and insertion', () => {
  it('are complementary for a linear model', () => {
    const del = deletionCurve(f, x, attribution)
    const ins = deletionCurve(f, x, attribution, { mode: 'insertion' })
    const total = 0.7 + w.reduce((s, v, j) => s + v * x[j], 0) + 0.7
    for (let s = 0; s < del.output.length; s++) expect(del.output[s] + ins.output[s]).toBeCloseTo(total, 12)
    expect(del.fraction.at(-1)).toBe(1)
    const reversed = deletionCurve(
      f,
      x,
      attribution.map((a) => -a),
    )
    expect(del.area).toBeLessThan(reversed.area)
  })
  it('steps several features at a time', () => {
    const r = deletionCurve(f, x, attribution, { step: 4 })
    expect(Array.from(r.fraction)).toEqual([0, 4 / 6, 1])
  })
})

describe('faithfulness correlation', () => {
  it('is 1 for the exact linear attribution', () => {
    const r = faithfulnessCorrelation(f, x, attribution, stream('faith'), { subsetSize: 2, samples: 40 })
    expect(r.correlation).toBeCloseTo(1, 12)
  })
})

describe('sanity checks', () => {
  const net: DenseNetwork = {
    weights: [
      [
        [1, 2],
        [3, 4],
      ],
      [[0.5], [-0.5]],
    ],
    biases: [[0.1, 0.2], [0.3]],
    activation: 'tanh',
  }
  it('randomise the top k layers', () => {
    const nets = randomiseNetwork(net, stream('rand'))
    expect(nets).toHaveLength(3)
    expect(Array.from(toFlat(nets[0].weights[1] as Tensor))).toEqual([0.5, -0.5])
    expect(Array.from(toFlat(nets[1].weights[0] as Tensor))).toEqual([1, 2, 3, 4])
    expect(Array.from(toFlat(nets[1].weights[1] as Tensor))).not.toEqual([0.5, -0.5])
    expect(Array.from(toFlat(nets[2].weights[0] as Tensor))).not.toEqual([1, 2, 3, 4])
  })
  it('measure similarity by rank', () => {
    const a = [3, 1, 2, -5]
    expect(explanationSimilarity(a, a).spearman).toBeCloseTo(1, 12)
    expect(explanationSimilarity(a, [-3, -1, -2, 5]).spearman).toBeCloseTo(-1, 12)
    expect(explanationSimilarity(a, [-3, -1, -2, 5]).spearmanAbsolute).toBeCloseTo(1, 12)
    expect(explanationSimilarity(a, [0, 0, 0, 9], { top: 1 }).topIntersection).toBe(1)
  })
})

describe('stability', () => {
  it('of a linear map 2x is 2', () => {
    const r = localLipschitz((z) => z.map((v) => 2 * v), [0.3, -1, 2], stream('lip'), { radius: 0.5 })
    expect(r.max).toBeCloseTo(2, 10)
    expect(r.mean).toBeCloseTo(2, 10)
  })
})

describe('relevance mass', () => {
  it('is the absolute share on the mask', () => {
    expect(relevanceMass([1, -3, 0, 4], [0, 1, 0, 1])).toBeCloseTo(7 / 8, 12)
    expect(relevanceMass([0, 0], [1, 0])).toBe(0)
  })
})
