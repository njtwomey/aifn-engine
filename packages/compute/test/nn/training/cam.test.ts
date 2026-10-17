/**
 * aifn-compute/nn/training `gradCam` on a random Conv-ReLU-GAP-Linear network: the Grad-CAM weights times $hw$ are
 * the classifier weights, the Grad-CAM map is ReLU(CAM) / $hw$, and the mean of the CAM plus the bias is the logit
 * (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { classActivationMap } from 'aifn-compute/learning/explain'
import { ActivationLayer, AvgPool2d, Conv2d, Flatten, Linear, Sequential } from 'aifn-compute/nn/layers'
import { gradCam } from 'aifn-compute/nn/training'

const net = Sequential(Conv2d(1, 3, 3, { padding: 1 }), ActivationLayer('relu'), AvgPool2d(6), Flatten(1), Linear(3, 2))
const params = net.init(stream(4)) as Record<string, Tensor>[]
const x = fromData(
  Float64Array.from({ length: 36 }, (_, i) => Math.sin(i) + 0.5),
  [1, 1, 6, 6],
)

describe('gradCam', () => {
  it.each([0, 1])('reduces to the CAM of a pooling-and-linear head (class %i)', (c) => {
    const g = gradCam(net, params, x, c, '1')
    const W = toFlat(params[4].weight)
    const w = [W[c], W[2 + c], W[4 + c]]
    Array.from(g.weights, (v) => v * 36).forEach((v, k) => expect(Math.abs(v - w[k])).toBeLessThan(1e-9))
    const cam = classActivationMap(g.maps, w)
    Array.from(g.map).forEach((v, p) => expect(Math.abs(v - Math.max(0, cam[p]) / 36)).toBeLessThan(1e-12))
    const logit = cam.reduce((a, b) => a + b, 0) / 36 + toFlat(params[4].bias)[c]
    expect(Math.abs(logit - g.output[c])).toBeLessThan(1e-9)
    expect([g.maps.channels, g.maps.height, g.maps.width]).toEqual([3, 6, 6])
  })

  it('weights by Grad-CAM++ on request', () => {
    const g = gradCam(net, params, x, 0, '1', { weighting: 'grad-cam++' })
    expect(g.weights.length).toBe(3)
    expect(g.map.every((v) => v >= 0)).toBe(true)
  })

  it('throws for a path with no activation', () => {
    expect(() => gradCam(net, params, x, 0, '9')).toThrow(AifnError)
  })
})
