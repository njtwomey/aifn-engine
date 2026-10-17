/**
 * aifn-compute/learning/metrics `relativeL2Error`: the relative $L^2$ error of operator learning, over every value of
 * a batch at once, with the metric's argument order (true values first) and registry metadata.
 */
import { describe, expect, it } from 'vitest'
import { tensor } from 'aifn-compute/foundation/tensor'
import { metricRegistry, relativeL2Error } from 'aifn-compute/learning/metrics'

describe('relativeL2Error', () => {
  it('is ||pred - true|| / ||true|| over all rows together', () => {
    expect(relativeL2Error([3, 4], [3.3, 4.4])).toBeCloseTo(0.1, 14)
    expect(relativeL2Error([3, 4], [0, 0])).toBe(1)
    const truth = tensor([
      [1, 0],
      [0, 2],
    ])
    const pred = tensor([
      [1, 1],
      [0, 2],
    ])
    expect(relativeL2Error(truth, pred)).toBeCloseTo(1 / Math.sqrt(5), 14)
    expect(relativeL2Error([0, 0], [1, 1])).toBeNaN()
  })

  it('is registered as a lower-is-better metric of values', () => {
    expect(metricRegistry.relativeL2Error).toBe(relativeL2Error)
    expect(relativeL2Error.info.direction).toBe('lower')
    expect(relativeL2Error.info.inputs).toBe('values')
  })
})
