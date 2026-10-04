import { describe, expect, it } from 'vitest'
import {
  butter,
  cheby1,
  cheby2,
  filtfilt,
  firwin,
  freqz,
  groupDelay,
  kaiserOrder,
  lfilter,
  lfilterZi,
  sosfilt,
  sosfiltZi,
} from 'aifn-compute/signal/filters'
import { signal, unwrapPhase } from 'aifn-compute/signal'
import { toSecondOrderSections, toTransferFunction, toZerosPolesGain, transferFunction } from 'aifn-compute/systems'
import { grad } from 'aifn-compute/foundation/autodiff'
import {
  imagPart,
  realPart,
  sum,
  square,
  tensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { close, closeC, F } from '../helpers'

describe('filters against scipy.signal', () => {
  it('firwin in every band type, and kaiserord', () => {
    const taps = (s: ReturnType<typeof firwin>) => s.repr.b
    close(taps(firwin(31, 0.3)), F.firwin.lowpass, 1e-12)
    close(taps(firwin(31, 0.3, { passZero: false })), F.firwin.highpass, 1e-12)
    close(
      taps(firwin(41, [0.2, 0.5], { passZero: false, window: { name: 'kaiser', beta: 6 } })),
      F.firwin.bandpass,
      1e-12,
    )
    close(taps(firwin(41, [0.2, 0.5])), F.firwin.bandstop, 1e-12)
    close(taps(firwin(21, 10, { fs: 100, window: 'hann' })), F.firwin.fs, 1e-12)
    expect(firwin(21, 10, { fs: 100 }).dt).toBeCloseTo(0.01, 15)
    const k = kaiserOrder(60, 0.05)
    expect(k.numtaps).toBe(F.firwin.kaiserord[0])
    expect(k.beta).toBeCloseTo(F.firwin.kaiserord[1], 12)
  })

  it('Butterworth and Chebyshev designs, in zpk, tf and sos forms', () => {
    const designs: Record<string, (output: 'zpk' | 'tf' | 'sos') => ReturnType<typeof butter>> = {
      'butter-low': (output) => butter(4, 0.2, { output }),
      'butter-high': (output) => butter(3, 0.4, { btype: 'highpass', output }),
      'butter-band': (output) => butter(3, [0.2, 0.5], { btype: 'bandpass', output }),
      'butter-stop': (output) => butter(2, [0.2, 0.5], { btype: 'bandstop', output }),
      'cheby1-low': (output) => cheby1(4, 1, 0.3, { output }),
      'cheby2-low': (output) => cheby2(4, 40, 0.3, { output }),
      'cheby2-odd': (output) => cheby2(5, 30, 0.3, { output }),
      'butter-fs': (output) => butter(4, 10, { fs: 100, output }),
    }
    for (const [key, design] of Object.entries(designs)) {
      const zpk = design('zpk')
      expect(zpk.repr.form).toBe('zpk')
      expect(toZerosPolesGain(zpk).repr.gain).toBeCloseTo(F.iir[key].k, 10)
      const tf = design('tf')
      expect(tf.repr.form).toBe('tf')
      for (const sys of [tf, zpk]) {
        const r = toTransferFunction(sys).repr
        close(r.b, F.iir[key].b, 1e-9)
        close(r.a, F.iir[key].a, 1e-9)
      }
      // Sections multiply out to 2·sections + 1 coefficients (trailing zeros for an odd order), as scipy's sos2tf:
      // compare the responses.
      const sos = design('sos')
      expect(sos.repr.form).toBe('sos')
      const want = freqz(tf, { n: 32 }).values
      const got = freqz(sos, { n: 32 }).values
      close(realPart(got), toFlat(realPart(want)), 1e-9)
      close(imagPart(got), toFlat(imagPart(want)), 1e-9)
    }
  })

  it('lfilter, lfilter_zi and filtfilt on coefficients and on systems', () => {
    const f = F.filtering
    const ba = { b: f.b, a: f.a }
    close(lfilter(ba, f.x as number[]).y, f.lfilter, 1e-10)
    close(lfilter(transferFunction(f.b, f.a, { dt: 1 }), f.x as number[]).y, f.lfilter, 1e-10)
    close(lfilterZi(ba), f.zi, 1e-10)
    const r = lfilter(ba, f.x as number[], { zi: (f.zi as number[]).map((z) => z * f.x[0]) })
    close(r.y, f.lfilterZi.y, 1e-10)
    close(r.zf, f.lfilterZi.zf, 1e-10)
    close(filtfilt(ba, f.x as number[]), f.filtfilt, 1e-9)
    close(filtfilt(ba, f.x as number[], { padtype: 'even', padlen: 20 }), f.filtfiltEven, 1e-9)
    close(lfilter(firwin(15, 0.3), (f.x as number[]).slice(0, 50)).y, f.fir, 1e-12)
  })

  it('a Signal in is a Signal out, on the same axis', () => {
    const f = F.filtering
    const x = signal(f.x, { fs: 50, t0: 2 })
    const out = lfilter({ b: f.b, a: f.a }, x)
    expect(out.y.kind).toBe('signal')
    expect(out.y.fs).toBe(50)
    expect(out.y.t0).toBe(2)
    close(out.y.data, f.lfilter, 1e-10)
    expect(filtfilt({ b: f.b, a: f.a }, x).kind).toBe('signal')
  })

  it('sosfilt equals lfilter; split filtering with zf continues exactly', () => {
    const f = F.filtering
    const sys = transferFunction(f.b, f.a, { dt: 1 })
    const sos = toSecondOrderSections(sys)
    close(sosfilt(sos, f.x as number[]).y, f.lfilter, 1e-9)
    const x = f.x as number[]
    const zi = sosfiltZi(sos)
    expect((zi as Tensor).shape).toEqual([2, 2])
    const first = lfilter(sys, x.slice(0, 80))
    const second = lfilter(sys, x.slice(80), { zi: first.zf })
    close([...toFlat(first.y as Tensor), ...toFlat(second.y as Tensor)], f.lfilter, 1e-10)
  })

  it('filtering is differentiable in the coefficients', () => {
    const f = F.filtering
    const x = tensor((f.x as number[]).slice(0, 40))
    const loss = (b: Value) => sum(square(lfilter({ b, a: f.a }, x).y))
    const g = toFlat(grad(loss)(tensor(f.b)) as Tensor)
    const h = 1e-6
    const value = (v: Value) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])
    for (const i of [0, 2, 4]) {
      const up = [...f.b]
      const down = [...f.b]
      up[i] += h
      down[i] -= h
      const fd = (value(loss(tensor(up))) - value(loss(tensor(down)))) / (2 * h)
      expect(g[i]).toBeCloseTo(fd, 4)
    }
  })

  it('freqz, group delay and unwrapPhase', () => {
    const f = F.filtering
    const sys = transferFunction(f.b, f.a, { dt: 1 })
    const r = freqz(sys, { n: 64, axis: 'rad/sample' })
    expect(r.quantity).toBe('response')
    close(r.f, f.freqz.w)
    closeC(r.values, f.freqz.h, 1e-10)
    const g = groupDelay(sys, { n: 64, axis: 'rad/sample' })
    close(g.delay, f.groupDelay.delay, 1e-8)
    close(unwrapPhase(F.unwrap.wrapped), F.unwrap.unwrapped, 1e-12)
    expect(() => freqz(transferFunction([1], [1, 1]))).toThrow(/discrete/)
  })
})
