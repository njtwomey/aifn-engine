import { describe, expect, test } from 'vitest'
import { ChiSquare } from 'aifn-compute/probability/distributions'
import { stream } from 'aifn-compute/foundation/random'
import {
  add,
  get,
  mul,
  sin,
  stack,
  toFlat,
  toRows,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  extendedKalmanFilter,
  filteringAlgorithms,
  filterAll,
  kalmanFilter,
  kalmanFilterSteps,
  kalmanStep,
  parseModel,
  rtsSmootherSteps,
  normalisedEstimationErrorSquared,
  normalisedInnovationSquared,
  rtsSmoother,
  simulateStateSpace,
  steadyStateKalman,
  unscentedKalmanFilter,
} from 'aifn-compute/inference/filtering'
import { fixture } from '../../fixtures'
import { checkProtocol } from '../../protocol'

type Fx = {
  ar2: number[]
  acf: number[]
  acov: number[]
  pacf: number[]
  yuleWalker: { ar: number[]; sigma2: number }
  burg: { ar: number[]; sigma2: number }
  armaAcov: { spec: { ar: number[]; ma: number[]; sigma: number }; acov: number[] }
  armaExact: { x: number[]; spec: { ar: number[]; ma: number[]; sigma: number }; logLikelihood: number }
  kalman: {
    A: number[][]
    C: number[][]
    Q: number[][]
    R: number[][]
    m0: number[]
    P0: number[][]
    y: number[][]
    filteredMean: number[][]
    filteredCov: number[][][]
    smoothedMean: number[][]
    smoothedCov: number[][][]
    lagOneCov: number[][][]
    logLikelihood: number
  }
  holtWinters: {
    y: number[]
    alpha: number
    beta: number
    gamma: number
    phi: number
    fitted: number[]
    forecast: number[]
  }
  garch: { x: number[]; omega: number; alpha: number; beta: number; logLikelihood: number }
}
const fx = fixture<Fx>('inference/filtering')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-10) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
}
const flat3 = (t: Tensor) => toFlat(t)

describe('state space', () => {
  const k = fx.kalman
  const model = { A: k.A, C: k.C, Q: k.Q, R: k.R, m0: k.m0, P0: k.P0 }
  test('the Kalman filter equals exact Gaussian conditioning', () => {
    const f = kalmanFilter(model, k.y)
    close(toFlat(f.mean), k.filteredMean.flat(), 1e-9)
    close(flat3(f.cov), k.filteredCov.flat(2), 1e-9)
    expect(f.logLikelihood).toBeCloseTo(k.logLikelihood, 9)
  })
  test('the RTS smoother equals exact Gaussian conditioning, lag-one covariances included', () => {
    const s = rtsSmoother(model, k.y)
    close(toFlat(s.mean), k.smoothedMean.flat(), 1e-9)
    close(flat3(s.cov), k.smoothedCov.flat(2), 1e-9)
    close(toFlat(s.lagOneCov).slice(4), k.lagOneCov.flat(2), 1e-9)
  })
  test('local level model: the steady-state gain has its closed form', () => {
    // P = P + q − P²/(P + r) ⇒ P² − qP − qr = 0 for the predicted variance; gain K = P/(P + r).
    const q = 0.5
    const r = 2
    const P = (q + Math.sqrt(q * q + 4 * q * r)) / 2
    const ss = steadyStateKalman({ A: 1, C: 1, Q: q, R: r, m0: 0, P0: 1 })
    expect(ss.converged).toBe(true)
    expect(toFlat(ss.gain)[0]).toBeCloseTo(P / (P + r), 10)
  })
  test('singular noise (a zero diagonal) filters and simulates without NaN', () => {
    const m = {
      A: [
        [1, 1],
        [0, 1],
      ],
      C: [[1, 0]],
      Q: [
        [0, 0],
        [0, 0.1],
      ],
      R: [[0.2]],
      m0: [0, 0],
      P0: [
        [0, 0],
        [0, 1],
      ],
    }
    const sim = simulateStateSpace(stream(5), m, 30)
    expect(toFlat(sim.states).every(Number.isFinite)).toBe(true)
    const f = kalmanFilter(m, sim.observations)
    expect(toFlat(f.mean).every(Number.isFinite)).toBe(true)
    expect(Number.isFinite(f.logLikelihood)).toBe(true)
    expect(toFlat(rtsSmoother(m, sim.observations).mean).every(Number.isFinite)).toBe(true)
  })
  test('missing observations are predicted through', () => {
    const y = k.y.map((r, t) => (t === 4 ? [NaN] : r))
    const f = kalmanFilter(model, y)
    const rows = toRows(f.mean)
    close(rows[4], toRows(f.predictedMean)[4])
  })
  test('EKF and UKF reduce to the Kalman filter for a linear model', () => {
    const lin = {
      f: (z: Vector): Value => stack([add(get(z, 0), get(z, 1)), get(z, 1)]),
      h: (z: Vector): Value => stack([get(z, 0)]),
      Q: k.Q,
      R: k.R,
      m0: k.m0,
      P0: k.P0,
    }
    const ref = kalmanFilter(model, k.y)
    close(toFlat(extendedKalmanFilter(lin, k.y).mean), toFlat(ref.mean), 1e-9)
    close(toFlat(unscentedKalmanFilter(lin, k.y).mean), toFlat(ref.mean), 1e-9)
    // A nonlinear pendulum runs and stays finite.
    const pend = {
      f: (z: Vector): Value => stack([add(get(z, 0), mul(0.1, get(z, 1))), add(get(z, 1), mul(-0.1, sin(get(z, 0))))]),
      h: (z: Vector): Value => stack([sin(get(z, 0))]),
      Q: [
        [1e-4, 0],
        [0, 1e-4],
      ],
      R: [[0.05]],
      m0: [1.2, 0],
      P0: [
        [0.1, 0],
        [0, 0.1],
      ],
    }
    const ys = Array.from({ length: 40 }, (_, t) => [Math.sin(1.2 * Math.cos(0.1 * t))])
    expect(toFlat(extendedKalmanFilter(pend, ys).mean).every(Number.isFinite)).toBe(true)
    expect(toFlat(unscentedKalmanFilter(pend, ys).mean).every(Number.isFinite)).toBe(true)
  })

  test('NIS and NEES of a consistent filter lie in their χ² bands', () => {
    const m = {
      A: [
        [1, 1],
        [0, 1],
      ],
      C: [[1, 0]],
      Q: [
        [0.01, 0],
        [0, 0.01],
      ],
      R: [[0.5]],
      m0: [0, 0],
      P0: [
        [1, 0],
        [0, 1],
      ],
    }
    const T = 2000
    const sim = simulateStateSpace(stream('consistency'), m, T)
    const f = kalmanFilter(m, sim.observations)
    const nis = toFlat(normalisedInnovationSquared(f))
    expect(nis.length).toBe(T)
    const avgNis = nis.reduce((a, b) => a + b, 0) / T
    // The average of T independent χ²₁ values is χ²_T / T: a two-sided 99.9% band.
    const band = (dof: number) => [0.0005, 0.9995].map((q) => (ChiSquare(dof * T).quantile(q) as number) / T)
    const [lo1, hi1] = band(1)
    expect(avgNis).toBeGreaterThan(lo1)
    expect(avgNis).toBeLessThan(hi1)
    const nees = toFlat(normalisedEstimationErrorSquared(f.mean, f.cov, sim.states))
    const avgNees = nees.reduce((a, b) => a + b, 0) / T
    const [lo2, hi2] = band(2)
    expect(avgNees).toBeGreaterThan(lo2)
    expect(avgNees).toBeLessThan(hi2)
    // An overconfident filter (R too small) fails the NIS test.
    const bad = kalmanFilter({ ...m, R: [[0.05]] }, sim.observations)
    const avgBad = toFlat(normalisedInnovationSquared(bad)).reduce((a, b) => a + b, 0) / T
    expect(avgBad).toBeGreaterThan(hi1)
  })

  test('NIS by hand for one step, NaN at missing steps', () => {
    const k1 = fx.kalman
    const model1 = { A: k1.A, C: k1.C, Q: k1.Q, R: k1.R, m0: k1.m0, P0: k1.P0 }
    const y = k1.y.map((r, t) => (t === 2 ? [NaN] : r))
    const f = kalmanFilter(model1, y)
    const nis = toFlat(normalisedInnovationSquared(f))
    const v = toFlat(f.innovation)
    const S = toFlat(f.innovationCov)
    expect(nis[0]).toBeCloseTo((v[0] * v[0]) / S[0], 12)
    expect(nis[2]).toBeNaN()
  })
})

describe('the filter and smoother as algorithms', () => {
  const k = fx.kalman
  const model = { A: k.A, C: k.C, Q: k.Q, R: k.R, m0: k.m0, P0: k.P0 }
  const T = k.y.length
  test('running the filter algorithm T steps gives the batch filter, step by step', () => {
    const batch = kalmanFilter(model, k.y)
    const alg = kalmanFilterSteps(model, k.y)
    const tr = trace(alg, undefined, T + 5, { keep: 'all' })
    expect(tr.final.t).toBe(T)
    expect(tr.final.terminated).toBe(true)
    expect(tr.final.logLikelihood).toBeCloseTo(batch.logLikelihood, 12)
    const means = toRows(batch.mean)
    tr.steps.slice(1).forEach((s, t) => close(toFlat(s.mean), means[t], 1e-12))
    close(toFlat(tr.final.cov), toFlat(batch.cov).slice(-(k.A.length ** 2)), 1e-12)
  })
  test('the smoother algorithm walks back to z₀ and matches rtsSmoother at every index', () => {
    const batch = rtsSmoother(model, k.y)
    const alg = rtsSmootherSteps(model, k.y)
    let s = alg.init(undefined, stream(0))
    const means = toRows(batch.mean)
    close(toFlat(s.mean), means[T - 1], 1e-12)
    for (let t = T - 2; t >= 0; t--) {
      s = alg.step(s, { t: T - 2 - t, stream: stream(0) })
      expect(s.index).toBe(t)
      close(toFlat(s.mean), means[t], 1e-12)
    }
    s = alg.step(s, { t: T - 1, stream: stream(0) })
    expect(s.terminated).toBe(true)
    close(toFlat(s.mean), toFlat(batch.initialMean), 1e-12)
    close(toFlat(s.cov), toFlat(batch.initialCov), 1e-12)
  })
  test('both satisfy the Algorithm protocol and are registered', () => {
    checkProtocol(kalmanFilterSteps(model, k.y), undefined, { steps: 4 })
    checkProtocol(rtsSmootherSteps(model, k.y), undefined, { steps: 4 })
    expect(Object.keys(filteringAlgorithms)).toEqual(expect.arrayContaining(['kalmanFilterSteps', 'rtsSmootherSteps']))
  })
  test('an empty series terminates at init', () => {
    expect(kalmanFilterSteps(model, []).init(undefined, stream(0)).terminated).toBe(true)
    expect(rtsSmootherSteps(model, []).init(undefined, stream(0)).terminated).toBe(true)
  })
  test('missing rows predict through; the step is the tensor-level kalmanStep', () => {
    const md = parseModel(model, 'test')
    const y = k.y.map((r, t) => (t === 2 ? r.map(() => NaN) : r))
    const f = filterAll(md, y)
    expect(f.steps[2].term).toBe(0)
    close(toFlat(f.steps[2].mean), toFlat(f.steps[2].predictedMean), 0)
    const again = kalmanStep(md, f.steps[0].mean, f.steps[0].cov, y[1])
    close(toFlat(again.cov), toFlat(f.steps[1].cov), 0)
  })
  test('a partly observed row updates with its observed entries (the reduced model)', () => {
    // Two sensors of one random walk; the second is missing at the step.
    const full = parseModel(
      {
        A: [[1]],
        C: [[1], [2]],
        Q: [[0.3]],
        R: [
          [0.5, 0.1],
          [0.1, 0.8],
        ],
        m0: [0],
        P0: [[1]],
      },
      'test',
    )
    const reduced = parseModel({ A: [[1]], C: [[1]], Q: [[0.3]], R: [[0.5]], m0: [0], P0: [[1]] }, 'test')
    const a = kalmanStep(full, full.m0 as Vector, full.P0, [1.3, NaN])
    const b = kalmanStep(reduced, reduced.m0 as Vector, reduced.P0, [1.3])
    close(toFlat(a.mean), toFlat(b.mean), 1e-15)
    close(toFlat(a.cov), toFlat(b.cov), 1e-15)
    expect(a.term).toBeCloseTo(b.term, 15)
    expect(Number.isNaN(toFlat(a.innovation)[1])).toBe(true)
    expect(toFlat(a.gain)[1]).toBe(0)
  })
})
