import { describe, expect, test } from 'vitest'
import {
  armax,
  arx,
  arxOrderSelection,
  n4sid,
  outputError,
  poles,
  polynomialModel,
  predictionErrorMethod,
  stateSpace,
  transferFunction,
} from 'aifn-compute/systems'
import { linearFilter } from 'aifn-compute/foundation/convolution'
import { normal, stream } from 'aifn-compute/foundation/random'
import { fromData, toComplexFlat, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../protocol'

const N = 2000
const draw = (key: string, n = N, sd = 1) =>
  Float64Array.from(toFlat(normal(stream(key), 0, sd, { shape: [n] }) as Tensor))
const filt = (b: number[], a: number[], x: Float64Array) =>
  Float64Array.from(toFlat(linearFilter(b, a, fromData(x, [x.length])) as Tensor))
const plus = (a: Float64Array, b: Float64Array) => a.map((v, i) => v + b[i])

// The true system: A = 1 − 1.5q⁻¹ + 0.7q⁻², B = q⁻¹ + 0.5q⁻², C = 1 − 0.2q⁻¹ + 0.3q⁻².
const A = [1, -1.5, 0.7]
const B = [0, 1, 0.5]
const C = [1, -0.2, 0.3]
const u = draw('u')

describe('ARX', () => {
  test('recovers a noiseless ARX system exactly', () => {
    const y = filt(B, A, u)
    const m = arx(y, u, { na: 2, nb: 2, nk: 1 })
    m.A.forEach((v, i) => expect(v).toBeCloseTo(A[i], 10))
    m.B.forEach((v, i) => expect(v).toBeCloseTo(B[i], 10))
    expect(m.loss).toBeLessThan(1e-20)
    expect(m.system.domain).toBe('discrete')
  })

  test('is consistent under white equation noise, and FPE/AIC select the true order', () => {
    const y = filt(B, A, u)
    const e = draw('e', N, 0.3)
    const yn = plus(y, filt([1], A, e))
    const m = arx(yn, u, { na: 2, nb: 2 })
    m.A.forEach((v, i) => expect(Math.abs(v - A[i])).toBeLessThan(0.03))
    expect(m.loss).toBeCloseTo(0.09, 1)
    const sel = arxOrderSelection(yn, u, { maxOrder: 5 })
    const best = sel.reduce((a, b) => (b.aic < a.aic ? b : a))
    expect(best.order).toBe(2)
    // The loss never rises with the order (nested models).
    for (let k = 1; k < sel.length; k++) expect(sel[k].loss).toBeLessThanOrEqual(sel[k - 1].loss + 1e-12)
  })
})

describe('prediction-error method', () => {
  const e = draw('e2', N, 0.5)
  const y = plus(filt(B, A, u), filt(C, A, e))

  test('ARMAX recovers A, B and C where ARX is biased', () => {
    const m = armax(y, u, { na: 2, nb: 2, nc: 2 }, { maxSteps: 60 })
    m.A.forEach((v, i) => expect(Math.abs(v - A[i])).toBeLessThan(0.03))
    m.B.forEach((v, i) => expect(Math.abs(v - B[i])).toBeLessThan(0.06))
    m.C.forEach((v, i) => expect(Math.abs(v - C[i])).toBeLessThan(0.06))
    // The prediction error approaches the innovation variance 0.25.
    expect(m.loss).toBeLessThan(0.27)
    const biased = arx(y, u, { na: 2, nb: 2 })
    expect(biased.loss).toBeGreaterThan(m.loss)
  })

  test('the loss never increases, and the algorithm meets the step protocol', () => {
    const alg = predictionErrorMethod(y, u, { na: 2, nb: 2, nc: 2 })
    let s = alg.init({}, stream('pem'))
    for (let k = 0; k < 10; k++) {
      const next = alg.step(s, { stream: stream('pem') } as never)
      expect(next.loss).toBeLessThanOrEqual(s.loss + 1e-15)
      s = next
    }
    checkProtocol(predictionErrorMethod(y.slice(0, 300), u.slice(0, 300), { na: 1, nb: 1, nc: 1 }), {}, { steps: 4 })
  })

  test('counts every estimated B coefficient as a parameter, even one that is exactly 0', () => {
    // Zero steps from θ₀ = (a₁, b₁) = (0.5, 0): B = [0, 0], whose zero b₁ is still a fitted parameter.
    const m = polynomialModel(y, u, { na: 1, nb: 1 }, { theta0: [0.5, 0], maxSteps: 0 })
    expect(m.B).toEqual([0, 0])
    expect(m.parameters).toBe(2)
  })

  test('output error recovers B/F from noisy output', () => {
    const F = [1, -0.8]
    const yo = plus(filt([0, 2], F, u), draw('oe', N, 0.2))
    const m = outputError(yo, u, { nb: 1, nf: 1 }, { maxSteps: 60 })
    expect(m.B[1]).toBeCloseTo(2, 1)
    expect(m.F[1]).toBeCloseTo(-0.8, 1)
    expect(run(predictionErrorMethod(yo, u, { nb: 1, nf: 1 }), {}, 40).loss).toBeCloseTo(0.04, 2)
  })
})

describe('N4SID', () => {
  test('recovers the poles and order of a noiseless two-state system', () => {
    const sys = transferFunction(B, A, { dt: 1 })
    const y = filt(B, A, u)
    const m = n4sid(y, u, { horizon: 8 })
    expect(m.order).toBe(2)
    const want = toComplexFlat(poles(sys)).sort((p, q) => p.im - q.im)
    const got = toComplexFlat(poles(m.system)).sort((p, q) => p.im - q.im)
    got.forEach((z, i) => {
      expect(z.re).toBeCloseTo(want[i].re, 6)
      expect(z.im).toBeCloseTo(want[i].im, 6)
    })
    expect(m.loss).toBeLessThan(1e-12)
  })

  test('identifies a two-output system with noise; the singular values drop after the order', () => {
    const plant = stateSpace({
      A: [
        [0.9, 0.2, 0],
        [-0.2, 0.9, 0],
        [0, 0, 0.5],
      ],
      B: [
        [1, 0],
        [0, 0.5],
        [1, 1],
      ],
      C: [
        [1, 0, 1],
        [0, 1, 0],
      ],
      dt: 1,
    })
    const u2 = Float64Array.from({ length: 2 * N }, (_, i) => draw('u2', 2 * N)[i])
    // Simulate x_{k+1} = Ax_k + Bu_k, y_k = Cx_k + noise.
    const a = toFlat(plant.repr.A)
    const b = toFlat(plant.repr.B)
    const c = toFlat(plant.repr.C)
    const noise = draw('n2', 2 * N, 0.01)
    const y = new Float64Array(2 * N)
    let x = [0, 0, 0]
    for (let t = 0; t < N; t++) {
      for (let r = 0; r < 2; r++)
        y[2 * t + r] = c[3 * r] * x[0] + c[3 * r + 1] * x[1] + c[3 * r + 2] * x[2] + noise[2 * t + r]
      x = [0, 1, 2].map(
        (r) =>
          a[3 * r] * x[0] +
          a[3 * r + 1] * x[1] +
          a[3 * r + 2] * x[2] +
          b[2 * r] * u2[2 * t] +
          b[2 * r + 1] * u2[2 * t + 1],
      )
    }
    const m = n4sid(fromData(y, [N, 2]), fromData(u2, [N, 2]), { horizon: 6 })
    expect(m.order).toBe(3)
    const sv = toFlat(m.singularValues)
    expect(sv[2] / sv[3]).toBeGreaterThan(20)
    const want = toComplexFlat(poles(plant))
      .map((z) => Math.hypot(z.re, z.im))
      .sort()
    const got = toComplexFlat(poles(m.system))
      .map((z) => Math.hypot(z.re, z.im))
      .sort()
    got.forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThan(0.01))
    expect(m.loss).toBeLessThan(2e-4)
  })
})
