/**
 * Generators for the weak-supervision papers: WebTraffic (MILLET) plants each signature where its mask says (none has
 * no mask, a flip reverses the base inside its window, cutoffs sit near zero, windows span ¼ to 2 days, values are
 * non-negative); class-conditional label noise flips labels at the requested rates; the half-kernel's arcs have their
 * radii.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { dense, toFlat } from 'aifn-compute/foundation/tensor'
import { classConditionalNoise, halfKernel, WEB_TRAFFIC_CLASSES, webTraffic } from 'aifn-methods/data/synthetic'

describe('webTraffic', () => {
  const perDay = 24
  const d = webTraffic(stream(3), { perClass: 6, samplesPerDay: perDay })
  const t = 7 * perDay
  const X = dense.data(d.x)
  const base = dense.data(d.base)
  const M = dense.data(d.discriminatory)
  const y = toFlat(d.y)
  const start = toFlat(d.windowStart)
  const len = toFlat(d.windowLength)
  it('plants the signatures where the mask says', () => {
    expect(d.x.shape).toEqual([60, t])
    expect(Math.min(...X)).toBeGreaterThanOrEqual(0)
    for (let i = 0; i < 60; i++) {
      const name = WEB_TRAFFIC_CLASSES[y[i]]
      const mask = M.slice(i * t, (i + 1) * t)
      const ones = mask.reduce((a, v) => a + v, 0)
      if (name === 'none') expect(ones).toBe(0)
      else if (name === 'spikes') expect(start[i]).toBe(-1)
      else {
        expect(len[i]).toBeGreaterThanOrEqual(perDay / 4)
        expect(len[i]).toBeLessThanOrEqual(2 * perDay)
        expect(ones).toBe(len[i])
      }
      // Outside the mask the series is its (clipped) base.
      for (let j = 0; j < t; j++) if (!mask[j]) expect(X[i * t + j]).toBeCloseTo(Math.max(0, base[i * t + j]), 12)
      if (name === 'flip')
        for (let k = 0; k < len[i]; k++)
          expect(X[i * t + start[i] + k]).toBeCloseTo(Math.max(0, base[i * t + start[i] + len[i] - 1 - k]), 12)
      if (name === 'cutoff') for (let k = 0; k < len[i]; k++) expect(X[i * t + start[i] + k]).toBeLessThan(0.7)
    }
  })
})

describe('classConditionalNoise', () => {
  it('flips labels at α for class 1 and β for class 0', () => {
    const d = classConditionalNoise(stream(4), { n: 20000, alpha: 0.05, beta: 0.2, truth: false })
    const clean = toFlat(d.clean)
    const flipped = toFlat(d.flipped)
    let a = 0
    let b = 0
    for (let i = 0; i < 20000; i++)
      if (clean[i] === 1) a += flipped[i]
      else b += flipped[i]
    expect(a / 10000).toBeCloseTo(0.05, 1)
    expect(b / 10000).toBeCloseTo(0.2, 1)
  })
})

describe('halfKernel', () => {
  it('puts each class on its arc', () => {
    const d = halfKernel(stream(5), { n: 400, noise: 0, scale: 1 })
    const X = dense.data(d.x)
    const y = toFlat(d.y!)
    for (let i = 0; i < 400; i++) {
      const r = Math.hypot(X[2 * i] + 20, X[2 * i + 1] / 0.6)
      expect(r).toBeCloseTo(y[i] === 1 ? 20 : 35, 8)
    }
  })
})
