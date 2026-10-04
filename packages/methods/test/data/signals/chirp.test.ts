import { describe, expect, it } from 'vitest'
import { chirp, tones } from 'aifn-methods/data/signals'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
// The scipy.signal.chirp references live in compute's `signal` fixture.
import { fixture } from '../../../../compute/test/fixtures'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const C = fixture<any>('signal').chirp

const close = (a: Tensor, e: number[], tol: number) => {
  const x = toFlat(a)
  expect(x.length).toBe(e.length)
  x.forEach((v, i) => expect(Math.abs(v - e[i])).toBeLessThan(tol))
}

describe('test signals', () => {
  it('chirps match scipy (linear, quadratic, logarithmic)', () => {
    close(chirp(C.t, 1, 2, 6), C.linear, 1e-12)
    close(chirp(C.t, 1, 2, 6, { method: 'quadratic' }), C.quadratic, 1e-12)
    close(chirp(C.t, 1, 2, 6, { method: 'logarithmic' }), C.logarithmic, 1e-12)
  })

  it('tones sum sinusoids', () => {
    const t = [0, 0.25, 0.5]
    close(tones(t, [{ frequency: 1 }, { frequency: 2, amplitude: 0.5, phase: Math.PI / 2 }]), [0.5, 0.5, 0.5], 1e-12)
  })
})
