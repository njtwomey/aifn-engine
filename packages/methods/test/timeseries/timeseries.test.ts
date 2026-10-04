import { describe, expect, test } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  armaAutocorrelation,
  armaAutocovariance,
  armaLogLikelihood,
  armaRoots,
  classicalDecomposition,
  difference,
  exponentialSmoothing,
  fitArma,
  forecastArma,
  garchFitSteps,
  garchLogLikelihood,
  garchProperties,
  isInvertible,
  isStationary,
  psiWeights,
  simulateArma,
  simulateGarch,
  exponentialSmoothingFitSteps,
  stateSpaceEm,
  stl,
  undifference,
} from 'aifn-methods/timeseries'
import { simulateStateSpace } from 'aifn-compute/inference/filtering'
// The statsmodels/scipy references live in compute's `inference/filtering` fixture (generated with the Kalman checks).
import { fixture } from '../../../compute/test/fixtures'

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

describe('ARMA', () => {
  test('roots, stationarity and invertibility', () => {
    expect(isStationary([0.5])).toBe(true)
    expect(isStationary([1.2])).toBe(false)
    expect(isStationary([0.6, 0.5])).toBe(false) // 1 − 0.6z − 0.5z² has a root inside the unit circle
    expect(isInvertible([0.4])).toBe(true)
    expect(isInvertible([-2])).toBe(false)
    expect(armaRoots({ ar: [0.5] }).ar.minModulus).toBeCloseTo(2, 12)
  })
  test('ψ weights and the theoretical autocovariance', () => {
    close(toFlat(psiWeights({ ar: [0.5], ma: [0.4] }, 4)), [1, 0.9, 0.45, 0.225, 0.1125])
    close(toFlat(armaAutocovariance(fx.armaAcov.spec, 10)), fx.armaAcov.acov, 1e-10)
    // AR(1): ρ(k) = φ^k.
    close(toFlat(armaAutocorrelation({ ar: [0.7] }, 5)), [1, 0.7, 0.49, 0.343, 0.2401, 0.16807], 1e-12)
    expect(() => armaAutocovariance({ ar: [1.1] }, 3)).toThrow(/stationary/)
  })
  test('exact log-likelihood equals the dense Gaussian density', () => {
    const L = armaLogLikelihood(fx.armaExact.x, fx.armaExact.spec)
    expect(L.logLikelihood).toBeCloseTo(fx.armaExact.logLikelihood, 8)
    expect(armaLogLikelihood(fx.armaExact.x, { ar: [1.5] }).logLikelihood).toBe(-Infinity)
  })
  test('the innovation variances start at the stationary variance and fall to 1', () => {
    // AR(1): F₁ = 1/(1 − φ²), then exactly 1. MA(1): F_t = (1 − θ^{2(t+1)})/(1 − θ^{2t}) (Brockwell & Davis, §5.2).
    const x = [0.3, -0.1, 0.8, 0.2, -0.5, 0.1]
    const ar = toFlat(armaLogLikelihood(x, { ar: [0.6] }).innovationVariance)
    expect(ar[0]).toBeCloseTo(1 / (1 - 0.36), 12)
    ar.slice(1).forEach((f) => expect(f).toBeCloseTo(1, 12))
    const th = 0.5
    const ma = toFlat(armaLogLikelihood(x, { ma: [th] }).innovationVariance)
    ma.forEach((f, t) => expect(f).toBeCloseTo((1 - th ** (2 * (t + 2))) / (1 - th ** (2 * (t + 1))), 12))
  })
  test('simulation is seeded, never clips and reports explosion', () => {
    const a = simulateArma(stream(3), { ar: [0.5] }, 50)
    expect(toFlat(a.x)).toEqual(toFlat(simulateArma(stream(3), { ar: [0.5] }, 50).x))
    expect(a.stationary).toBe(true)
    const big = simulateArma(stream(3), { ar: [1.9] }, 2000)
    expect(big.stationary).toBe(false)
    expect(big.diverged).toBe(true)
    expect(Math.max(...toFlat(big.x).slice(0, big.divergedAt).map(Math.abs))).toBeGreaterThan(1e6)
  })
  test('CSS and exact fits recover an ARMA(1,1)', () => {
    const sim = simulateArma(stream(8), { ar: [0.6], ma: [0.3], sigma: 1.5 }, 2000)
    for (const method of ['css', 'exact'] as const) {
      const fit = fitArma(sim.x, { p: 1, q: 1, method })
      expect(fit.converged).toBe(true)
      expect(toFlat(fit.ar)[0]).toBeCloseTo(0.6, 1)
      expect(toFlat(fit.ma)[0]).toBeCloseTo(0.3, 1)
      expect(Math.sqrt(fit.sigma2)).toBeCloseTo(1.5, 1)
    }
  })
  test('forecasts revert to the mean with widening intervals', () => {
    const f = forecastArma([0, 0, 0, 4], { ar: [0.5], mean: 0, sigma: 1 }, 3)
    close(toFlat(f.mean), [2, 1, 0.5])
    close(toFlat(f.se), [1, Math.sqrt(1.25), Math.sqrt(1.3125)])
  })
})

describe('transformations', () => {
  test('differencing and its inverse', () => {
    const x = [1, 4, 9, 16, 25, 36]
    close(toFlat(difference(x, { order: 2 })), [2, 2, 2, 2])
    close(toFlat(undifference(difference(x, { lag: 2 }), [1, 4], { lag: 2 })), x)
  })
  test('decompositions recover a clean seasonal pattern', () => {
    const pattern = [3, -1, -4, 2]
    const y = Array.from({ length: 48 }, (_, t) => 10 + 0.3 * t + pattern[t % 4])
    close(toFlat(classicalDecomposition(y, 4).pattern), pattern, 1e-9)
    const s = stl(y, 4)
    close(toFlat(s.pattern), pattern, 0.05)
    expect(Math.max(...toFlat(s.remainder).map(Math.abs))).toBeLessThan(0.1)
  })
  test('robust STL puts outliers in the remainder instead of the seasonal pattern', () => {
    const pattern = [3, -1, -4, 2]
    // A little deterministic noise: on noise-free data the robustness scale collapses towards rounding level.
    const clean = Array.from({ length: 64 }, (_, t) => 10 + 0.2 * t + pattern[t % 4] + 0.1 * Math.sin(1.7 * t * t))
    const spikes = [9, 26, 41]
    const y = clean.map((v, t) => (spikes.includes(t) ? v + 40 : v))
    const plain = stl(y, 4)
    const robust = stl(y, 4, { robust: 5 })
    // Many passes stay finite (all-zero weight windows keep the nearest value rather than NaN).
    expect(toFlat(stl(clean, 4, { robust: 20 }).remainder).every(Number.isFinite)).toBe(true)
    const exact = Array.from({ length: 64 }, (_, t) => 10 + 0.2 * t + pattern[t % 4] + (spikes.includes(t) ? 40 : 0))
    expect(toFlat(stl(exact, 4, { robust: 5 }).remainder).every(Number.isFinite)).toBe(true)
    const err = (d: typeof plain) => Math.max(...toFlat(d.pattern).map((p, k) => Math.abs(p - pattern[k])))
    expect(err(robust)).toBeLessThan(0.25)
    expect(err(robust)).toBeLessThan(err(plain) / 4)
    // The spikes land in the remainder at nearly their full size; elsewhere the remainder stays small.
    const rem = toFlat(robust.remainder)
    spikes.forEach((t) => expect(rem[t]).toBeGreaterThan(35))
    rem.forEach((v, t) => spikes.includes(t) || expect(Math.abs(v)).toBeLessThan(0.6))
  })
})

describe('state space EM', () => {
  test('EM never decreases the log-likelihood ', () => {
    const truth = { A: [[0.9]], C: [[1]], Q: [[0.3]], R: [[0.5]], m0: [0], P0: [[1]] }
    const y = simulateStateSpace(stream(4), truth, 300).observations
    const alg = stateSpaceEm(y, { A: 0.5, C: 1, Q: 1, R: 1, m0: 0, P0: 1 }, { estimate: { C: false } })
    const t = trace(alg, undefined, 60, { record: { ll: (s) => s.logLikelihood } })
    const ll = toFlat(t.series.ll)
    for (let i = 1; i < ll.length; i++) expect(ll[i]).toBeGreaterThanOrEqual(ll[i - 1] - 1e-8)
    expect(toFlat(t.steps.at(-1)!.model.A)[0]).toBeCloseTo(0.9, 1)
  })
})

describe('exponential smoothing', () => {
  const h = fx.holtWinters
  test('additive damped Holt–Winters matches the reference recursion', () => {
    const r = exponentialSmoothing(
      h.y,
      { trend: 'damped', seasonal: 'additive', period: 4, alpha: h.alpha, beta: h.beta, gamma: h.gamma, phi: h.phi },
      { horizon: 8 },
    )
    close(toFlat(r.fitted), h.fitted, 1e-10)
    close(toFlat(r.forecast), h.forecast, 1e-10)
    const lo = toFlat(r.lower)
    const hi = toFlat(r.upper)
    for (let i = 1; i < 8; i++) expect(hi[i] - lo[i]).toBeGreaterThanOrEqual(hi[i - 1] - lo[i - 1])
  })
  test('simple smoothing: forecasts equal the last level', () => {
    const r = exponentialSmoothing([1, 2, 3, 2, 1], { alpha: 0.5 }, { horizon: 2 })
    const lv = toFlat(r.level)
    close(toFlat(r.forecast), [lv[4], lv[4]])
  })
  test('the fitter lowers the SSE', () => {
    const alg = exponentialSmoothingFitSteps(h.y, { trend: 'additive', seasonal: 'additive', period: 4 })
    const s0 = run(alg, undefined, 0)
    const s = run(alg, undefined, 500)
    expect(s.objective).toBeLessThan(s0.objective)
  })
})

describe('GARCH', () => {
  test('log-likelihood matches the reference recursion', () => {
    const g = fx.garch
    expect(garchLogLikelihood(g.x, g).logLikelihood).toBeCloseTo(g.logLikelihood, 9)
  })
  test('simulated returns have the unconditional variance and heavy tails', () => {
    const spec = { omega: 0.05, alpha: 0.08, beta: 0.9 }
    const p = garchProperties(spec)
    const r = toFlat(simulateGarch(stream(1), spec, 40000).returns)
    const m2 = r.reduce((a, v) => a + v * v, 0) / r.length
    const m4 = r.reduce((a, v) => a + v ** 4, 0) / r.length
    expect(m2).toBeCloseTo(p.unconditionalVariance, 0)
    expect(Math.abs(m2 / p.unconditionalVariance - 1)).toBeLessThan(0.15)
    expect(m4 / (m2 * m2)).toBeGreaterThan(3.3)
  })
  test('the fitter recovers persistence', () => {
    const spec = { omega: 0.05, alpha: 0.1, beta: 0.85 }
    const r = simulateGarch(stream(9), spec, 4000).returns
    const s = run(garchFitSteps(r), undefined, 3000)
    expect(s.params.alpha + s.params.beta).toBeCloseTo(0.95, 1)
  })
})
