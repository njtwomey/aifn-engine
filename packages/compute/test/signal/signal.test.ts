import { describe, expect, it } from 'vitest'
import { magnitude, phase, sampleTimes, signal, spectrumDecibels, unwrapPhase } from 'aifn-compute/signal'
import { freqz } from 'aifn-compute/signal/filters'
import { decibels } from 'aifn-compute/foundation/fourier'
import { transferFunction } from 'aifn-compute/systems'
import { complex, tensor, toFlat } from 'aifn-compute/foundation/tensor'

describe('the Signal, Spectrum and TimeFrequency helpers', () => {
  it('signal() records the sample rate and start time', () => {
    const s = signal([0, 1, 0, -1], { fs: 4, t0: 1, unit: 'V' })
    expect(s.kind).toBe('signal')
    expect(toFlat(sampleTimes(s))).toEqual([1, 1.25, 1.5, 1.75])
    expect(signal(s, { fs: 8 }).unit).toBe('V')
    expect(() => signal([1, 2], { fs: 0 })).toThrow()
  })

  it('magnitude, phase and decibels of a response', () => {
    const z = complex(tensor([3, 0]), tensor([4, -2]))
    expect(toFlat(magnitude(z))).toEqual([5, 2])
    expect(toFlat(phase(z, { degrees: true }))[1]).toBeCloseTo(-90, 12)
    expect(toFlat(decibels([1, 10, 100]))).toEqual([0, 10, 20])
    expect(toFlat(decibels([0]))).toEqual([-Infinity])
    // A response's dB is 20 log₁₀ |H|: the two-tap average is −6.02 dB at a quarter of the sample rate.
    const r = freqz(transferFunction([0.5, 0.5], [1], { dt: 1 }), { n: 4 })
    const db = toFlat(spectrumDecibels(r))
    expect(db[0]).toBeCloseTo(0, 12)
    expect(db[2]).toBeCloseTo(20 * Math.log10(Math.SQRT1_2), 12)
  })

  it('unwrapPhase removes 2π jumps', () => {
    const p = [0, 3, -3, -0.5]
    const u = toFlat(unwrapPhase(p))
    expect(u[2]).toBeCloseTo(-3 + 2 * Math.PI, 12)
    expect(u[3]).toBeCloseTo(-0.5 + 2 * Math.PI, 12)
  })
})
