import { describe, expect, test } from 'vitest'
import {
  awqQuantise,
  dequantise,
  fakeQuantise,
  gptqQuantise,
  integerRange,
  quantisationError,
  quantisationParams,
  quantise,
  quantisedMatmul,
  roundHalfEven,
  type QuantisationParams,
} from 'aifn-compute/nn/quantise'
import { grad } from 'aifn-compute/foundation/autodiff'
import { normal, stream } from 'aifn-compute/foundation/random'
import { fromData, sum, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type M = number[][]
const F = fixture<{
  x: M
  observer: {
    affine: { scale: number; zeroPoint: number }
    symmetric: { scale: number; zeroPoint: number }
    channel: { scale: number[]; zeroPoint: number[] }
  }
  quantise: { scale: number; zeroPoint: number; q: M; halves: number[]; qHalves: number[] }
  quantisePerChannel: { q: M }
  fakeQuantise: { scale: number; zeroPoint: number; qmin: number; qmax: number; y: M; grad: M }
}>('nn/quantise')

const X = fromData(Float64Array.from(F.x.flat()), [6, 10])
const fixed = (
  scale: number | number[],
  zeroPoint: number | number[],
  bits: number,
  signed: boolean,
  axis: number | null = null,
): QuantisationParams => ({
  scale: Float64Array.from([scale].flat()),
  zeroPoint: Float64Array.from([zeroPoint].flat()),
  ...integerRange(bits, signed),
  bits,
  axis,
  scheme: 'affine',
})

describe('quantisation against torch', () => {
  test('min–max observers: affine quint8, symmetric qint8, per-channel symmetric', () => {
    const a = quantisationParams(X, { bits: 8, scheme: 'affine' })
    expect(a.scale[0]).toBeCloseTo(F.observer.affine.scale, 6)
    expect(a.zeroPoint[0]).toBe(F.observer.affine.zeroPoint)
    const s = quantisationParams(X, { bits: 8, scheme: 'symmetric' })
    expect(s.scale[0]).toBeCloseTo(F.observer.symmetric.scale, 6)
    expect(s.zeroPoint[0]).toBe(0)
    const c = quantisationParams(X, { bits: 8, scheme: 'symmetric', axis: 1 })
    toFlat(fromData(c.scale)).forEach((v, i) =>
      expect(Math.abs(v - F.observer.channel.scale[i])).toBeLessThan(1e-6 * v),
    )
  })

  test('quantize_per_tensor and quantize_per_channel integers, halves rounded to even', () => {
    const p = fixed(F.quantise.scale, F.quantise.zeroPoint, 8, false)
    expect(toFlat(quantise(X, p))).toEqual(F.quantise.q.flat())
    const h = fixed(1, 10, 8, false)
    expect(toFlat(quantise(fromData(Float64Array.from(F.quantise.halves)), h))).toEqual(F.quantise.qHalves)
    const c = quantisationParams(X, { bits: 8, scheme: 'symmetric', axis: 1 })
    // torch computes x/s in float32: the integers may differ by one only where x/s is within float32 error of a tie.
    const ours = toFlat(quantise(X, c))
    const theirs = F.quantisePerChannel.q.flat()
    const xs = toFlat(X)
    ours.forEach((v, i) => {
      if (v === theirs[i]) return
      const r = xs[i] / c.scale[i % 10]
      expect(Math.abs(v - theirs[i])).toBe(1)
      expect(Math.abs(Math.abs(r - Math.trunc(r)) - 0.5)).toBeLessThan(1e-5)
    })
  })

  test('fake_quantize_per_tensor_affine and its straight-through gradient', () => {
    const c = F.fakeQuantise
    const p = { ...fixed(c.scale, c.zeroPoint, 4, false), qmin: c.qmin, qmax: c.qmax }
    const y = fakeQuantise(X, p) as Tensor
    // torch holds the scale in float32.
    toFlat(y).forEach((v, i) => expect(v).toBeCloseTo(c.y.flat()[i], 6))
    const g = grad((x: Value) => sum(fakeQuantise(x, p)))(X) as Tensor
    expect(toFlat(g)).toEqual(c.grad.flat())
  })
})

describe('laws', () => {
  test('round half to even', () => {
    expect([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2.4, 2.6].map(roundHalfEven)).toEqual([-2, -2, -0, 0, 2, 2, 2, 3])
  })

  test('dequantise(quantise(x)) is within half a step inside the range; error falls about 6 dB per bit', () => {
    const x = normal(stream('q'), 0, 1, { shape: [4000] }) as Tensor
    let last = -Infinity
    for (const bits of [2, 4, 6, 8]) {
      const p = quantisationParams(x, { bits, scheme: 'symmetric' })
      const back = toFlat(dequantise(quantise(x, p), p))
      toFlat(x).forEach((v, i) => expect(Math.abs(v - back[i])).toBeLessThanOrEqual(p.scale[0] / 2 + 1e-12))
      const e = quantisationError(x, p)
      expect(e.sqnr).toBeGreaterThan(last)
      if (bits >= 6) expect(e.sqnr - last).toBeGreaterThan(10)
      last = e.sqnr
    }
  })

  test('stochastic rounding is unbiased; nearest rounding is not', () => {
    const v = 0.3
    const x = fromData(new Float64Array(20000).fill(v))
    const p = fixed(1, 0, 8, true)
    const sr = toFlat(quantise(x, p, { rounding: 'stochastic', stream: stream('sr') }))
    const mean = sr.reduce((a, b) => a + b, 0) / sr.length
    expect(Math.abs(mean - v)).toBeLessThan(0.01)
    expect(toFlat(quantise(x, p))[0]).toBe(0)
  })

  test('the integer matmul matches the dequantised product exactly and approximates AB', () => {
    const A = toFlat(normal(stream('A'), 0, 1, { shape: [5, 16] }) as Tensor)
    const B = toFlat(normal(stream('B'), 0, 0.3, { shape: [16, 4] }) as Tensor)
    const a = fromData(Float64Array.from(A), [5, 16])
    const b = fromData(Float64Array.from(B), [16, 4])
    const r = quantisedMatmul(a, b)
    const da = toFlat(dequantise(r.qa, r.paramsA))
    const db = toFlat(dequantise(r.qb, r.paramsB))
    const y = toFlat(r.y)
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 4; j++) {
        let exact = 0
        let deq = 0
        for (let k = 0; k < 16; k++) {
          exact += A[i * 16 + k] * B[k * 4 + j]
          deq += da[i * 16 + k] * db[k * 4 + j]
        }
        expect(y[i * 4 + j]).toBeCloseTo(deq, 10)
        expect(Math.abs(y[i * 4 + j] - exact)).toBeLessThan(0.1)
      }
    expect(r.maxAccumulator).toBeLessThan(2 ** 31)
  })

  test('GPTQ lowers the output error below round-to-nearest at 3 bits', () => {
    const W = normal(stream('W'), 0, 1, { shape: [8, 24] }) as Tensor
    // Correlated inputs: GPTQ's error feedback helps most when inputs are not white.
    const Z = toFlat(normal(stream('Z'), 0, 1, { shape: [200, 24] }) as Tensor)
    const Xc = Float64Array.from(Z, (v, i) => v + 0.8 * Z[i - (i % 24)])
    const r = gptqQuantise(W, fromData(Xc, [200, 24]), { bits: 3 })
    expect(r.outputError).toBeLessThan(r.roundToNearestError)
  })

  test('AWQ: α = 0 is round-to-nearest; with salient input channels the searched scale beats it', () => {
    const W = normal(stream('W'), 0, 1, { shape: [8, 24] }) as Tensor
    const Z = toFlat(normal(stream('Z'), 0, 1, { shape: [200, 24] }) as Tensor)
    // Three salient input channels carry activations 20 times larger than the rest (Lin et al., 2024, §3.1).
    const X = fromData(
      Float64Array.from(Z, (v, i) => (i % 24 < 3 ? 20 * v : v)),
      [200, 24],
    )
    const r = awqQuantise(W, X, { bits: 3 })
    // Round-to-nearest with per-row symmetric quantisers, computed independently.
    const p = quantisationParams(W, { bits: 3, scheme: 'symmetric', axis: 0 })
    const rtn = toFlat(dequantise(quantise(W, p), p))
    const w = toFlat(W)
    const x = toFlat(X)
    let err = 0
    for (let i = 0; i < 200; i++)
      for (let o = 0; o < 8; o++) {
        let e = 0
        for (let j = 0; j < 24; j++) e += x[i * 24 + j] * (w[o * 24 + j] - rtn[o * 24 + j])
        err += e * e
      }
    expect(r.roundToNearestError).toBeCloseTo(err / 200, 8)
    expect(toFlat(r.errors)[0]).toBe(r.roundToNearestError)
    // The scales as the reference code forms them (review G2): s = max(mean|X|^α, 1e−4) / √(max s · min s), on the grid
    // k/20, k = 0 … 20. An input channel that never fires is floored after the power, not before.
    expect(toFlat(r.grid)).toEqual(Array.from({ length: 21 }, (_, k) => k / 20))
    const Xz = fromData(
      Float64Array.from(toFlat(X), (v, i) => (i % 24 === 5 ? 0 : v)),
      [200, 24],
    )
    const z = awqQuantise(W, Xz, { bits: 3 })
    const xz = toFlat(Xz)
    const mag = Array.from({ length: 24 }, (_, j) => {
      let m = 0
      for (let i = 0; i < 200; i++) m += Math.abs(xz[i * 24 + j]) / 200
      return Math.max(m ** z.alpha, 1e-4)
    })
    const norm = Math.sqrt(Math.max(...mag) * Math.min(...mag))
    toFlat(z.scales).forEach((v, j) => expect(v).toBeCloseTo(mag[j] / norm, 10))
    expect(r.outputError).toBe(Math.min(...toFlat(r.errors)))
    expect(r.alpha).toBeGreaterThan(0)
    expect(r.outputError).toBeLessThan(0.7 * r.roundToNearestError)
    // The scales are normalised (max · min = 1) and largest on the salient channels.
    const s = toFlat(r.scales)
    expect(Math.max(...s) * Math.min(...s)).toBeCloseTo(1, 10)
    expect(Math.min(s[0], s[1], s[2])).toBeGreaterThan(Math.max(...s.slice(3)))
    // The returned weights are what the error was measured on: dividing the scale back out is exact.
    const wq = toFlat(r.weights)
    for (let o = 0; o < 8; o++) {
      const row = Float64Array.from({ length: 24 }, (_, j) => wq[o * 24 + j] * s[j])
      const step = r.params.scale[o]
      row.forEach((v) => expect(Math.abs(v / step - Math.round(v / step))).toBeLessThan(1e-9))
    }
  })
})
