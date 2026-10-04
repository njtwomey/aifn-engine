import { describe, expect, it } from 'vitest'
import { placeLabels, toPixel } from './labels'

describe('placeLabels', () => {
  const bounds = { width: 200, height: 100 }

  it('places a lone label right of its point', () => {
    expect(placeLabels([50], [50], [30], 12, { bounds })).toEqual([{ position: 'right', shown: true }])
  })

  it('moves a colliding label to the next free side, in priority order', () => {
    // Two points 10 px apart: the higher-priority second point takes the right; the first goes left.
    const out = placeLabels([100, 110], [50, 50], [40, 40], 12, { bounds, priority: [1, 2] })
    expect(out[1]).toEqual({ position: 'right', shown: true })
    expect(out[0]).toEqual({ position: 'left', shown: true })
  })

  it('hides the lowest-priority label when every side is taken', () => {
    const px = [100, 100, 100, 100, 100]
    const out = placeLabels(px, [50, 50, 50, 50, 50], [6, 6, 6, 6, 6], 12, { bounds })
    expect(out.slice(0, 4).every((o) => o.shown)).toBe(true)
    expect(out[4].shown).toBe(false)
  })

  it('keeps labels inside the plot and skips empty or non-finite points', () => {
    const out = placeLabels([195, NaN, 20], [50, 50, 50], [40, 40, 0], 12, { bounds })
    expect(out[0]).toEqual({ position: 'left', shown: true })
    expect(out[1].shown).toBe(false)
    expect(out[2].shown).toBe(false)
  })
})

describe('toPixel', () => {
  it('maps linear and log ranges', () => {
    expect(toPixel(5, [0, 10], 100, false)).toBe(50)
    expect(toPixel(10, [1, 100], 100, true)).toBeCloseTo(50)
  })
})

describe('placeLabels with markers', () => {
  it('keeps a label off other markers', () => {
    // A marker 20 px right of the point blocks the right side; the label goes left.
    const out = placeLabels([100], [50], [30], 12, {
      bounds: { width: 200, height: 100 },
      markers: { x: [100, 120], y: [50, 50], radius: 6 },
    })
    expect(out[0]).toEqual({ position: 'left', shown: true })
  })
})
