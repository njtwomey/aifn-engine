/** `parallelWaves`: balanced alternating classes, and noise-free points exactly on the two waves. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { parallelWaves } from 'aifn-methods/data/synthetic'

describe('parallelWaves', () => {
  it('alternates the classes, so they are balanced', () => {
    const d = parallelWaves(stream(1), { n: 201 })
    const y = Array.from(toFlat(d.y))
    expect(y.slice(0, 4)).toEqual([0, 1, 0, 1])
    expect(y.filter((c) => c === 1).length).toBe(100)
    expect(d.y.dtype).toBe('int32')
    expect(d.x.shape).toEqual([201, 2])
  })

  it('puts noise-free points on (t, (2c - 1) o + a sin(f t)), with t in [-2, 2]', () => {
    const o = 0.4
    const a = 0.7
    const f = 2
    const d = parallelWaves(stream(2), { n: 50, offset: o, amplitude: a, frequency: f, noise: 0 })
    const x = toFlat(d.x)
    const y = toFlat(d.y)
    for (let i = 0; i < 50; i++) {
      const t = d.style[i]
      expect(t).toBeGreaterThanOrEqual(-2)
      expect(t).toBeLessThanOrEqual(2)
      expect(x[2 * i]).toBe(t)
      expect(Math.abs(x[2 * i + 1] - ((2 * y[i] - 1) * o + a * Math.sin(f * t)))).toBe(0)
    }
  })

  it('is deterministic in its stream and adds noise of the given size', () => {
    const p = parallelWaves(stream(3), { n: 400 })
    const q = parallelWaves(stream(3), { n: 400 })
    expect(Array.from(toFlat(p.x))).toEqual(Array.from(toFlat(q.x)))
    const x = toFlat(p.x)
    let ss = 0
    for (let i = 0; i < 400; i++) ss += (x[2 * i] - p.style[i]) ** 2
    expect(Math.sqrt(ss / 400)).toBeCloseTo(0.08, 1)
  })
})
