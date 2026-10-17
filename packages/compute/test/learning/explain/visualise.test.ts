/**
 * aifn-compute/learning/explain `activationMaximisation`: the worked example of the feature-visualisation note (a
 * linear unit with an L2 penalty, plain ascent from zero, $x_t = 0.5(1 - 0.5^t)w$) and jittered normalised ascent on
 * a shift-invariant unit (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { mul, sub, sum, tensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { activationMaximisation } from 'aifn-compute/learning/explain'

const wk = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
const kernel = tensor([
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1],
])

describe('activationMaximisation', () => {
  const alg = activationMaximisation((img: Tensor) => sum(mul(img, kernel)), [3, 3], {
    stepSize: 0.25,
    l2: 1,
    normalise: false,
  })
  const zero = { x: new Array(9).fill(0) }

  it.each([1, 2, 3])('climbs a linear unit as x_t = 0.5 (1 - 0.5^t) w (step %i)', (t) => {
    const x = run(alg, zero, t).x
    wk.forEach((v, i) => expect(x[i]).toBeCloseTo(0.5 * (1 - 0.5 ** t) * v, 12))
  })

  it('reaches w / (2 lambda) with value 3 and objective 1.5', () => {
    const limit = run(alg, zero, 60)
    wk.forEach((v, i) => expect(limit.x[i]).toBeCloseTo(0.5 * v, 12))
    expect(limit.value).toBeCloseTo(3, 12)
    expect(limit.objective).toBeCloseTo(1.5, 12)
  })

  it('with jitter reaches the maximum 4.5 of a shift-invariant unit', () => {
    const g = (img: Tensor) => sub(sum(img), mul(0.5, sum(mul(img, img))))
    const jittered = run(activationMaximisation(g, [3, 3], { jitter: 1, stepSize: 0.05, blur: 0 }), undefined, 40)
    expect(Math.abs(jittered.value - 4.5)).toBeLessThan(1e-2)
    expect(jittered.value).toBeCloseTo(4.4993, 3)
  })

  it('blurs and clips, and throws for a bad shape or start', () => {
    const unit = (img: Tensor) => sum(mul(img, kernel))
    const blurred = run(activationMaximisation(unit, [1, 3, 3], { blur: 1, blurEvery: 1, clip: [0, 1] }), undefined, 3)
    expect(blurred.x.every((v) => v >= 0 && v <= 1)).toBe(true)
    expect(() => activationMaximisation((img: Tensor) => sum(img), [9])).toThrow(AifnError)
    expect(() => run(alg, { x: [1, 2] }, 1)).toThrow(AifnError)
  })
})
