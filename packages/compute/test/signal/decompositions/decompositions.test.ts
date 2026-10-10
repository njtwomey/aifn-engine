import { describe, expect, it } from 'vitest'
import { eemd, emd, extrema, siftImf, siftSteps } from 'aifn-compute/signal/decompositions'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

const x = Array.from(
  { length: 512 },
  (_, i) => Math.sin((2 * Math.PI * i) / 16) + 0.8 * Math.sin((2 * Math.PI * i) / 128) + i / 512,
)

describe('empirical mode decomposition', () => {
  it('emd reconstructs the signal and separates the fast tone first', () => {
    const d = emd(x, { maxImfs: 3 })
    expect(d.kind).toBe('decomposition')
    expect(d.method).toBe('emd')
    const n = x.length
    for (let i = 0; i < n; i++) {
      let s = d.residual ? toFlat(d.residual)[i] : 0
      for (const c of d.components) s += toFlat(c.values)[i]
      expect(s).toBeCloseTo(x[i], 10)
    }
    const first = toFlat(d.components[0].values).slice(100, 400)
    const fast = first.map((_, i) => Math.sin((2 * Math.PI * (i + 100)) / 16))
    const err = first.reduce((a, v, i) => a + (v - fast[i]) ** 2, 0) / first.length
    expect(err).toBeLessThan(0.02)
  })

  it('siftSteps ends when the rule holds and satisfies the Algorithm protocol', () => {
    const alg = siftSteps(x, { rule: { kind: 'fixed', sifts: 6 } })
    const t = trace(alg, undefined, 20)
    expect(t.meta.stopped).toBe('done')
    expect(t.meta.steps).toBe(6)
    checkProtocol(alg, undefined, { steps: 5, record: { first: (s) => toFlat(s.h)[0] } })
    checkProtocol(siftSteps(x.slice(0, 128)), undefined, { steps: 4 })
    const one = siftImf(x, { rule: { kind: 'fixed', sifts: 6 } })
    expect(one.sifts).toBe(6)
    expect(toFlat(one.imf)).toEqual(toFlat(t.final.h))
    expect(siftImf(x, { maxSteps: 1 }).sifts).toBe(1)
  })

  it('extrema counts maxima, minima and zero crossings', () => {
    const e = extrema([0, 1, 0, -1, 0, 1, 0])
    expect(toFlat(e.maxima)).toEqual([1, 5])
    expect(toFlat(e.minima)).toEqual([3])
  })

  it('eemd is deterministic in its stream', () => {
    const a = eemd(stream(1), x.slice(0, 128), { trials: 4, maxImfs: 2 })
    const b = eemd(stream(1), x.slice(0, 128), { trials: 4, maxImfs: 2 })
    const c = eemd(stream(2), x.slice(0, 128), { trials: 4, maxImfs: 2 })
    expect(a.components.length).toBe(2)
    expect(toFlat(a.components[0].values)).toEqual(toFlat(b.components[0].values))
    expect(toFlat(a.components[0].values)).not.toEqual(toFlat(c.components[0].values))
  })
})
