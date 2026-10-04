/**
 * TCAV against its parts: the linear probe matches scikit-learn's L2 logistic regression
 * (`fixtures/learning/explain.json`, `probe`); with a linear head the conceptual sensitivity is the head's weights along
 * the CAV, so a concept aligned with the head scores 1 and is significant against random concepts, and one opposed to
 * it scores 0.
 */
import { describe, expect, it } from 'vitest'
import { normal, stream } from 'aifn-compute/foundation/random'
import { fromData, mul, sum, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  activationGradients,
  conceptActivationVector,
  conceptSensitivity,
  linearProbe,
  tcav,
  tcavScore,
} from 'aifn-compute/learning/explain'
import { fixture } from '../../fixtures'

type F = { positive: number[][]; negative: number[][]; l2: number; weights: number[]; bias: number }
const F = fixture<{ probe: F }>('learning/explain').probe

const H = 4
const U = [1, -0.5, 0.25, 0]
const head = (h: Tensor) => sum(mul(h, fromData(Float64Array.from(U), [H])))
const cloud = (seed: string, n: number, shift: number[]) => {
  const z = toFlat(normal(stream(seed), 0, 1, { shape: [n * H] }))
  return fromData(
    Float64Array.from(z, (v, i) => v + shift[i % H]),
    [n, H],
  )
}

describe('linear probe', () => {
  it('matches scikit-learn', () => {
    const p = linearProbe(F.positive, F.negative, { l2: F.l2 })
    F.weights.forEach((w, i) => expect(p.weights[i]).toBeCloseTo(w, 5))
    expect(p.bias).toBeCloseTo(F.bias, 5)
  })
})

describe('TCAV', () => {
  const examples = cloud('class', 60, [0, 0, 0, 0])
  const randoms = Array.from({ length: 6 }, (_, k) => cloud(`random-${k}`, 50, [0, 0, 0, 0]))
  it('has the head’s gradient as sensitivity', () => {
    const g = activationGradients(head, examples)
    const cav = conceptActivationVector(cloud('c', 50, [2, 0, 0, 0]), randoms[0]).vector
    const s = conceptSensitivity(g, cav)
    const want = U.reduce((a, u, i) => a + u * cav[i], 0)
    for (const v of s) expect(v).toBeCloseTo(want, 10)
    expect(tcavScore(g, cav)).toBe(1)
  })
  it('finds an aligned concept significant and an opposed one at zero', () => {
    const aligned = tcav(head, examples, cloud('aligned', 50, [1.5, -0.75, 0.4, 0]), randoms)
    expect(aligned.mean).toBe(1)
    expect(aligned.significant).toBe(true)
    expect(Math.min(...aligned.accuracies)).toBeGreaterThan(0.75)
    const opposed = tcav(head, examples, cloud('opposed', 50, [-1.5, 0.75, -0.4, 0]), randoms)
    expect(opposed.mean).toBe(0)
  })
})
