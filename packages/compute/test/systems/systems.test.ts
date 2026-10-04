import { describe, expect, test } from 'vitest'
import {
  bode,
  controllability,
  controllabilityGramian,
  convert,
  dimensions,
  discretise,
  feedback,
  frequencyResponse,
  impulseResponse,
  margins,
  observability,
  observabilityGramian,
  parallel,
  poles,
  respond,
  responseAt,
  secondOrderSections,
  series,
  simulate,
  stability,
  stateFeedback,
  stateSpace,
  stepResponse,
  systemZeros,
  toSecondOrderSections,
  toStateSpace,
  toTransferFunction,
  toZerosPolesGain,
  transferFunction,
  zerosPolesGain,
  type LtiSystem,
} from 'aifn-compute/systems'
import { lyapunov } from 'aifn-compute/numerics/linalg'
import { imagPart, realPart, toComplexFlat, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { fixture } from '../fixtures'
import { checkProtocol } from '../protocol'

type Mat = number[][]
type SS = { A: Mat; B: Mat; C: Mat; D: Mat }
type Complexes = { re: number[]; im: number[] }
const F = fixture<{
  cartpole: { A: Mat; B: Mat }
  lyapunov: { A: Mat; Q: Mat; X: Mat; Ad: Mat; Xd: Mat }
  discretise: SS & { dt: number; zoh: SS; tustin: SS }
  tf2ss: SS & { num: number[]; den: number[] }
  ss2tf: { num: number[]; den: number[] }
  bode: { num: number[]; den: number[]; w: number[]; mag: number[]; phase: number[] }
  freqresp: { num: number[]; den: number[]; w: number[]; re: number[]; im: number[] }
  tf2zpk: { num: number[]; den: number[]; zeros: Complexes; poles: Complexes; gain: number }
  discrete: { b: number[]; a: number[]; w: number[]; re: number[]; im: number[]; sos: Mat }
  margins: {
    num: number[]
    den: number[]
    gainCrossover: number
    phaseMargin: number
    phaseCrossover: number
    gainMargin: number
  }
}>('systems')

const close = (a: Tensor | null, b: Mat | number[], tol: number) => {
  expect(a).not.toBeNull()
  const x = toFlat(a!)
  const y = (b as (number | number[])[]).flat()
  expect(x.length).toBe(y.length)
  const scale = Math.max(1, ...y.map(Math.abs))
  x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(tol * scale))
}

/** Complex values sorted as numpy's sort_complex (by real part, then imaginary). */
const sorted = (z: Tensor) => toComplexFlat(z).sort((u, v) => u.re - v.re || u.im - v.im)
const closeComplex = (z: Tensor, ref: Complexes, tol: number) => {
  const s = sorted(z)
  expect(s.length).toBe(ref.re.length)
  s.forEach((c, i) => {
    expect(Math.abs(c.re - ref.re[i])).toBeLessThan(tol)
    expect(Math.abs(c.im - ref.im[i])).toBeLessThan(tol)
  })
}

const doubleIntegrator = () =>
  stateSpace({
    A: [
      [0, 1],
      [0, 0],
    ],
    B: [0, 1],
    C: [1, 0],
  })

describe('construction and conversion', () => {
  test('tf → ss is the controllable canonical form of scipy tf2ss', () => {
    const t = F.tf2ss
    const s = toStateSpace(transferFunction(t.num, t.den)).repr
    close(s.A, t.A, 1e-14)
    close(s.B, t.B, 1e-14)
    close(s.C, t.C, 1e-14)
    close(s.D, t.D, 1e-14)
  })

  test('ss → tf matches scipy ss2tf', () => {
    const g = toTransferFunction(stateSpace({ A: F.cartpole.A, B: F.cartpole.B, C: [[1, 0, 0, 0]] })).repr
    close(g.a, F.ss2tf.den, 1e-10)
    const b = toFlat(g.b)
    const num = [...new Array(F.ss2tf.num.length - b.length).fill(0), ...b]
    num.forEach((v, i) => expect(v).toBeCloseTo(F.ss2tf.num[i], 10))
  })

  test('tf → zpk matches scipy tf2zpk, as complex128 roots', () => {
    const t = F.tf2zpk
    const z = toZerosPolesGain(transferFunction(t.num, t.den))
    expect(z.repr.zeros.dtype).toBe('complex128')
    expect(z.repr.poles.dtype).toBe('complex128')
    closeComplex(z.repr.zeros, t.zeros, 1e-10)
    closeComplex(z.repr.poles, t.poles, 1e-10)
    expect(z.repr.gain).toBeCloseTo(t.gain, 12)
    closeComplex(poles(transferFunction(t.num, t.den)), t.poles, 1e-10)
    closeComplex(systemZeros(transferFunction(t.num, t.den)), t.zeros, 1e-10)
  })

  test('every form round-trips to the same response', () => {
    const t = F.tf2zpk
    const g = transferFunction(t.num, t.den)
    const w = F.freqresp.w
    for (const form of ['zpk', 'ss', 'tf'] as const) {
      const h = frequencyResponse(convert(convert(g, form), 'tf'), w).values
      close(realPart(h), F.freqresp.re, 1e-10)
      close(imagPart(h), F.freqresp.im, 1e-10)
    }
    // zpk built directly from conjugate pairs.
    const z = zerosPolesGain(
      t.zeros.re.map((re, i) => ({ re, im: t.zeros.im[i] })),
      t.poles.re.map((re, i) => ({ re, im: t.poles.im[i] })),
      t.gain,
    )
    close(toTransferFunction(z).repr.a, t.den, 1e-10)
  })

  test('a discrete filter converts to scipy zpk2sos sections and back', () => {
    const d = F.discrete
    const g = transferFunction(d.b, d.a, { dt: 1 })
    const sos = toSecondOrderSections(g)
    close(sos.repr.sections, d.sos, 1e-9)
    const back = toTransferFunction(secondOrderSections(d.sos))
    close(back.repr.b, d.b, 1e-10)
    close(back.repr.a, d.a, 1e-10)
    expect(() => toSecondOrderSections(transferFunction([1], [1, 1]))).toThrow()
  })

  test('shapes, dimensions and validation', () => {
    const di = doubleIntegrator()
    expect(dimensions(di)).toEqual({ states: 2, inputs: 1, outputs: 1 })
    expect(dimensions(transferFunction([1], [1, 2, 1]))).toEqual({ states: 2, inputs: 1, outputs: 1 })
    expect(() => stateSpace({ A: [[1, 2]], B: [1] })).toThrow()
    expect(() => transferFunction([1], [0])).toThrow()
    expect(() => toStateSpace(transferFunction([1, 0, 0], [1, 1]))).toThrow() // improper
  })
})

describe('frequency responses on complex128', () => {
  test('continuous response matches scipy freqresp', () => {
    const r = F.freqresp
    const spec = frequencyResponse(transferFunction(r.num, r.den), r.w)
    expect(spec.kind).toBe('spectrum')
    expect(spec.axis).toBe('rad/s')
    expect(spec.values.dtype).toBe('complex128')
    close(realPart(spec.values), r.re, 1e-12)
    close(imagPart(spec.values), r.im, 1e-12)
    const at = responseAt(transferFunction(r.num, r.den))(r.w[3])
    expect(at.re).toBeCloseTo(r.re[3], 12)
    expect(at.im).toBeCloseTo(r.im[3], 12)
  })

  test('discrete response matches scipy freqz, in every form', () => {
    const d = F.discrete
    const g = transferFunction(d.b, d.a, { dt: 1 })
    for (const sys of [g, toZerosPolesGain(g), toStateSpace(g), toSecondOrderSections(g)] as LtiSystem[]) {
      const spec = frequencyResponse(sys, d.w)
      expect(spec.axis).toBe('rad/sample')
      close(realPart(spec.values), d.re, 1e-9)
      close(imagPart(spec.values), d.im, 1e-9)
    }
  })

  test('Bode data match scipy', () => {
    const b = bode(transferFunction(F.bode.num, F.bode.den), F.bode.w)
    close(b.magnitudeDb, F.bode.mag, 1e-10)
    close(b.phase, F.bode.phase, 1e-9)
  })

  test('margins of 2/(s(s+1)(s+2)) and a delay', () => {
    const L = transferFunction(F.margins.num, F.margins.den)
    const m = margins(L)
    expect(m.gainMargin).toBeCloseTo(F.margins.gainMargin, 8)
    expect(m.phaseCrossover).toBeCloseTo(F.margins.phaseCrossover, 8)
    expect(m.phaseMargin).toBeCloseTo(F.margins.phaseMargin, 7)
    expect(m.gainCrossover).toBeCloseTo(F.margins.gainCrossover, 8)
    // Adding exactly the delay margin brings the phase margin to zero.
    const delayed = transferFunction(F.margins.num, F.margins.den, { delay: m.delayMargin })
    expect(margins(delayed).phaseMargin).toBeCloseTo(0, 6)
    expect(stability(feedback(L)).stable).toBe(true)
  })

  test('series, parallel and feedback compose transfer functions', () => {
    const g1 = transferFunction([1], [1, 1])
    const g2 = transferFunction([2], [1, 3])
    const w = [0.1, 1, 10]
    const h = (s: LtiSystem) => toComplexFlat(frequencyResponse(s, w).values)
    const a = h(g1)
    const b = h(g2)
    h(series(g1, g2)).forEach((z, i) => {
      expect(z.re).toBeCloseTo(a[i].re * b[i].re - a[i].im * b[i].im, 12)
      expect(z.im).toBeCloseTo(a[i].re * b[i].im + a[i].im * b[i].re, 12)
    })
    h(parallel(g1, g2)).forEach((z, i) => expect(z.re).toBeCloseTo(a[i].re + b[i].re, 12))
    // 1/(s+1) under unity feedback is 1/(s+2).
    close(toTransferFunction(feedback(g1)).repr.a, [1, 2], 1e-12)
  })
})

describe('structure and stability', () => {
  test('controllability and observability ranks', () => {
    const di = doubleIntegrator()
    expect(controllability(di).full).toBe(true)
    expect(observability(di).full).toBe(true)
    // Two decoupled identical modes driven by one input: rank 1.
    const bad = stateSpace({
      A: [
        [-1, 0],
        [0, -1],
      ],
      B: [1, 1],
      C: [[0, 1]],
    })
    expect(controllability(bad).rank).toBe(1)
    const unobservable = stateSpace({
      A: [
        [-1, 0],
        [0, -2],
      ],
      B: [1, 1],
      C: [[1, 0]],
    })
    expect(observability(unobservable).rank).toBe(1)
  })

  test('Lyapunov equations and Gramians match scipy', () => {
    close(lyapunov(F.lyapunov.A, F.lyapunov.Q).X, F.lyapunov.X, 1e-10)
    close(lyapunov(F.lyapunov.Ad, F.lyapunov.Q, { discrete: true }).X, F.lyapunov.Xd, 1e-10)
    close(controllabilityGramian(stateSpace({ A: [[-1]], B: [1] })).W, [[0.5]], 1e-12)
    close(observabilityGramian(stateSpace({ A: [[-2]], B: [1], C: [1] })).W, [[0.25]], 1e-12)
    expect(controllabilityGramian(stateSpace({ A: [[1]], B: [1] })).W).toBeNull()
  })

  test('poles are complex128 and stability reads their location', () => {
    const osc = stateSpace({
      A: [
        [-0.1, 1],
        [-1, -0.1],
      ],
      B: [0, 1],
    })
    const p = poles(osc)
    expect(p.dtype).toBe('complex128')
    closeComplex(p, { re: [-0.1, -0.1], im: [-1, 1] }, 1e-12)
    expect(stability(osc).stable).toBe(true)
    expect(stability(osc).abscissa).toBeCloseTo(-0.1, 12)
    expect(stability(doubleIntegrator()).stable).toBe(false) // marginal
    expect(stability([[0.5]], { discrete: true }).stable).toBe(true)
    expect(stability(transferFunction([1], [1, -1.5], { dt: 1 })).stable).toBe(false)
  })
})

describe('discretisation and simulation', () => {
  const c = F.discretise
  const sys = stateSpace({ A: c.A, B: c.B, C: c.C, D: c.D })

  test('ZOH and Tustin match scipy cont2discrete', () => {
    for (const method of ['zoh', 'tustin'] as const) {
      const d = discretise(sys, c.dt, method)
      expect(d.domain).toBe('discrete')
      expect(d.dt).toBe(c.dt)
      const ref = c[method]
      close(d.repr.A, ref.A, 1e-12)
      close(d.repr.B, ref.B, 1e-12)
      close(d.repr.C, ref.C, 1e-12)
      close(d.repr.D, ref.D, 1e-12)
    }
  })

  test('step and impulse responses of a first-order lag are exact at the samples', () => {
    const lag = stateSpace({ A: [[-2]], B: [2], C: [1] })
    const r = stepResponse(lag, { tEnd: 2, dt: 0.1 })
    expect(r.t.shape).toEqual([21])
    toFlat(r.t).forEach((t, k) => expect(toFlat(r.y)[k]).toBeCloseTo(1 - Math.exp(-2 * t), 12))
    const imp = impulseResponse(lag, { tEnd: 1, dt: 0.1 })
    toFlat(imp.t).forEach((t, k) => expect(toFlat(imp.y)[k]).toBeCloseTo(2 * Math.exp(-2 * t), 12))
    // A discrete system steps by its own dt; the moving average's step response is ½, 1, 1, …
    const ma = respond(transferFunction([0.5, 0.5], [1], { dt: 1 }), 1, { tEnd: 3 })
    expect(toFlat(ma.y)).toEqual([0.5, 1, 1, 1])
  })

  test('simulate closes state feedback; t counts steps and time is the time', () => {
    const di = doubleIntegrator()
    const K = [1, Math.sqrt(3)] // the LQR gain for Q = I, R = 1
    const alg = simulate(di, (_time, x) => -(K[0] * toFlat(x)[0] + K[1] * toFlat(x)[1]), { dt: 0.05, tEnd: 10 })
    const end = run(alg, { x0: [1, 0] }, 1000)
    expect(end.terminated).toBe(true)
    expect(end.t).toBe(200)
    expect(end.time).toBeCloseTo(10, 9)
    expect(Math.abs(toFlat(end.x)[0])).toBeLessThan(1e-3)
    expect(stability(stateFeedback(di, [K])).stable).toBe(true)
  })

  test('simulate satisfies the Algorithm protocol', () => {
    const alg = simulate(discretise(sys, 0.1), (time) => [Math.sin(time), 1], {})
    checkProtocol(alg, { x0: [1, 0, -1] }, { steps: 10, record: { t: (s) => s.t, y0: (s) => toFlat(s.y)[0] } })
    checkProtocol(simulate(doubleIntegrator(), 1, { dt: 0.1, tEnd: 0.5 }), {}, { steps: 10 })
  })
})
