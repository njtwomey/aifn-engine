import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { hzToMel, melFilterbank, melToHz, mfcc } from 'aifn-compute/signal/audio'
import { toFlat } from 'aifn-compute/foundation/tensor'

describe('audio features', () => {
  it('mel scales invert; Slaney is linear below 1 kHz', () => {
    for (const f of [0, 300, 1000, 4000]) {
      expect(melToHz(hzToMel(f))).toBeCloseTo(f, 8)
      expect(melToHz(hzToMel(f, 'slaney'), 'slaney')).toBeCloseTo(f, 8)
    }
    expect(hzToMel(1000, 'slaney')).toBeCloseTo(15, 12)
  })

  it('filter banks and MFCCs have the right shapes', () => {
    const bank = melFilterbank(20, 512, 16000)
    expect(bank.weights.shape).toEqual([20, 257])
    const t = Array.from({ length: 4000 }, (_, i) => i / 16000)
    // A linear chirp from 200 Hz to 3 kHz over 0.25 s.
    const x = t.map((s) => Math.cos(2 * Math.PI * (200 * s + ((3000 - 200) / (2 * 0.25)) * s * s)))
    const m = mfcc(x, 16000)
    expect(m.mfcc.shape[1]).toBe(13)
    expect(m.mfcc.shape[0]).toBe(m.logMel.shape[0])
    expect(toFlat(m.mfcc).every(Number.isFinite)).toBe(true)
  })

  it('throws DomainError when nMfcc > nMels', () => {
    expect(() => mfcc([0, 1, 0, -1], 1000, { nfft: 4, nMels: 5, nMfcc: 6 })).toThrow(DomainError)
  })
})
