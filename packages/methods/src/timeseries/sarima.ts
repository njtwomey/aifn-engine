/**
 * Seasonal ARIMA, SARIMA(p, d, q)(P, D, Q)_s (Box, Jenkins & Reinsel, 2008, ch. 9):
 *
 *   φ(B) Φ(Bˢ) ∇ᵈ ∇ₛᴰ x_t = θ(B) Θ(Bˢ) ε_t,  ε_t ~ N(0, σ²),
 *
 * with B the backshift, ∇ = 1 − B, ∇ₛ = 1 − Bˢ, φ(z) = 1 − Σ φᵢ zⁱ, Φ(z) = 1 − Σ Φᵢ zⁱ, θ(z) = 1 + Σ θⱼ zʲ and
 * Θ(z) = 1 + Σ Θⱼ zʲ (statsmodels' signs). The differenced series w = ∇ᵈ ∇ₛᴰ x is an ARMA(p + sP, q + sQ) process
 * whose lag polynomials are the products φ(z)Φ(zˢ) and θ(z)Θ(zˢ), so the exact likelihood is the ARMA one: the
 * Kalman filter of `aifn-compute/inference/filtering` on Harvey's state-space form of the expanded model (as `armaLogLikelihood`).
 * This is statsmodels SARIMAX with `simple_differencing=True`: the likelihood of w, not of x with a diffuse start.
 * Forecasts run the recursion of the integrated polynomial φ(z)Φ(zˢ)(1 − z)ᵈ(1 − zˢ)ᴰ on x itself.
 */

import type { Stream } from 'aifn-compute/foundation/random'
import { mean as meanOf, tensor, toFlat, type Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import {
  exactLikelihood,
  forecastArma,
  fromPacf,
  residuals,
  simulateArma,
  toPacf,
  type ArmaLikelihood,
  type Forecast,
} from './arma'
import { simplexFit, type FitState } from './fit'
import { toVec, type VectorLike } from './inputs'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A seasonal ARIMA model. Omitted parts are empty, d = D = 0, σ = 1 and μ = 0. */
export type SarimaSpec = {
  /** φ₁ … φ_p. */
  ar?: VectorLike
  /** θ₁ … θ_q. */
  ma?: VectorLike
  /** Φ₁ … Φ_P, at lags s, 2s, …. */
  seasonalAr?: VectorLike
  /** Θ₁ … Θ_Q, at lags s, 2s, …. */
  seasonalMa?: VectorLike
  /** The season length s (required with any seasonal term or seasonal difference). */
  period?: number
  /** d, the number of ordinary differences. */
  diff?: number
  /** D, the number of seasonal differences. */
  seasonalDiff?: number
  sigma?: number
  /** The mean μ of the differenced series (only without differencing; a mean after differencing is a drift). */
  mean?: number
}

type Parsed = {
  ar: number[]
  ma: number[]
  sar: number[]
  sma: number[]
  s: number
  d: number
  D: number
  sigma: number
  mean: number
}

function parse(m: SarimaSpec, where: string): Parsed {
  const sar = m.seasonalAr ? toVec(m.seasonalAr, where) : []
  const sma = m.seasonalMa ? toVec(m.seasonalMa, where) : []
  const D = m.seasonalDiff ?? 0
  const d = m.diff ?? 0
  const s = m.period ?? 0
  if ((sar.length || sma.length || D) && !(Number.isInteger(s) && s >= 2))
    throw new DomainError(where, `${where}: a seasonal term needs an integer period of at least 2`)
  if (!(Number.isInteger(d) && d >= 0 && Number.isInteger(D) && D >= 0))
    throw new DomainError(where, `${where}: the differencing orders must be non-negative integers`)
  if ((m.mean ?? 0) !== 0 && d + D > 0) throw new DomainError(where, `${where}: a non-zero mean needs d = D = 0`)
  return {
    ar: m.ar ? toVec(m.ar, where) : [],
    ma: m.ma ? toVec(m.ma, where) : [],
    sar,
    sma,
    s,
    d,
    D,
    sigma: m.sigma ?? 1,
    mean: m.mean ?? 0,
  }
}

/** The product of two polynomials given by their coefficients c₀, c₁, … */
function multiply(a: number[], b: number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0)
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j]
  return out
}

/** 1 + Σ cᵢ z^{s·i}. */
const seasonal = (c: number[], s: number): number[] => {
  const out = new Array<number>(c.length * s + 1).fill(0)
  out[0] = 1
  c.forEach((v, i) => (out[(i + 1) * s] = v))
  return out
}

/** The expanded lag polynomials as ARMA coefficients (φ* with x_t = Σ φ*ᵢ x_{t−i} + …, θ* as θ). */
function expand(p: Pick<Parsed, 'ar' | 'ma' | 'sar' | 'sma' | 's' | 'd' | 'D'>) {
  const neg = (c: number[]) => c.map((v) => -v)
  const arPoly = multiply([1, ...neg(p.ar)], seasonal(neg(p.sar), p.s))
  const maPoly = multiply([1, ...p.ma], seasonal(p.sma, p.s))
  let integrated = arPoly
  for (let k = 0; k < p.d; k++) integrated = multiply(integrated, [1, -1])
  for (let k = 0; k < p.D; k++) integrated = multiply(integrated, seasonal([-1], p.s))
  const coefficients = (poly: number[], sign: number) => {
    const c = poly.slice(1).map((v) => sign * v + 0)
    while (c.length && c[c.length - 1] === 0) c.pop()
    return c
  }
  return { ar: coefficients(arPoly, -1), ma: coefficients(maPoly, 1), integrated: coefficients(integrated, -1) }
}

/**
 * The ARMA coefficients of a seasonal model: `ar` and `ma` of the differenced series (the products φ(z)Φ(zˢ) and
 * θ(z)Θ(zˢ), so x_t = Σ arᵢ x_{t−i} + ε_t + Σ maⱼ ε_{t−j} for w), and `integrated`, the AR coefficients of
 * φ(z)Φ(zˢ)(1 − z)ᵈ(1 − zˢ)ᴰ, the model of x itself as a (non-stationary) ARMA.
 */
export function expandSarima(model: SarimaSpec): { ar: Vector; ma: Vector; integrated: Vector } {
  const e = expand(parse(model, 'expandSarima'))
  return { ar: tensor(e.ar), ma: tensor(e.ma), integrated: tensor(e.integrated) }
}

/** ∇ᵈ ∇ₛᴰ x: the series shortens by d + sD. */
function differenced(x: number[], d: number, D: number, s: number): number[] {
  let w = x
  for (let k = 0; k < D; k++) w = w.slice(s).map((v, t) => v - w[t])
  for (let k = 0; k < d; k++) w = w.slice(1).map((v, t) => v - w[t])
  return w
}

/**
 * The exact Gaussian log-likelihood of the differenced series ∇ᵈ∇ₛᴰx under a seasonal model (module notes), with
 * σ² concentrated out when `model.sigma` is omitted. −∞ when φ(z)Φ(zˢ) is not stationary.
 */
export function sarimaLogLikelihood(x: VectorLike, model: SarimaSpec): ArmaLikelihood {
  const p = parse(model, 'sarimaLogLikelihood')
  const e = expand(p)
  return exactLikelihood(differenced(toVec(x, 'sarimaLogLikelihood'), p.d, p.D, p.s), e.ar, e.ma, p.mean, model.sigma)
}

/** A simulated seasonal ARIMA series. */
export type SarimaSimulation = {
  /** x₁ … x_n. */
  x: Vector
  /** The stationary differenced process w that was integrated, n + d + sD values (the first d + sD unused). */
  differenced: Vector
  stationary: boolean
  diverged: boolean
}

/**
 * Simulate n values of a seasonal ARIMA model: the ARMA process w of the differenced series (`simulateArma` on the
 * expanded polynomials, with its burn-in), integrated d and D times from zero starting values.
 */
export function simulateSarima(s: Stream, model: SarimaSpec, n: number): SarimaSimulation {
  const p = parse(model, 'simulateSarima')
  const e = expand(p)
  const lead = p.d + p.s * p.D
  const sim = simulateArma(s, { ar: e.ar, ma: e.ma, sigma: p.sigma, mean: p.mean }, n + lead)
  let x = toFlat(sim.x)
  // Each integration inverts one difference from zero starting values: x_t = w_t + x_{t−lag}.
  const integrate = (w: number[], lag: number) => {
    const out = w.slice()
    for (let t = lag; t < out.length; t++) out[t] += out[t - lag]
    return out
  }
  for (let k = 0; k < p.d; k++) x = integrate(x, 1)
  for (let k = 0; k < p.D; k++) x = integrate(x, p.s)
  return { x: tensor(x.slice(lead)), differenced: sim.x, stationary: sim.stationary, diverged: sim.diverged }
}

/** The parameters of a fitted seasonal ARIMA model. */
export type SarimaFit = {
  ar: Vector
  ma: Vector
  seasonalAr: Vector
  seasonalMa: Vector
  period: number
  diff: number
  seasonalDiff: number
  /** The mean of the differenced series (0 unless it was estimated, which needs d = D = 0). */
  mean: number
  sigma2: number
  logLikelihood: number
  /** −2 log L + 2k, with k = p + q + P + Q + 1 (σ²) + 1 if the mean was estimated. */
  aic: number
  method: 'css' | 'exact'
}

/** Options for `sarimaFitSteps` and `fitSarima`. */
export type SarimaFitOptions = {
  /** The orders (p, d, q). */
  p: number
  d?: number
  q: number
  /** The seasonal orders (P, D, Q) and the period s. */
  P?: number
  D?: number
  Q?: number
  period?: number
  /** `css` minimises the conditional sum of squares; `exact` maximises the exact likelihood. Default `exact`. */
  method?: 'css' | 'exact'
  /** Estimate the mean of the differenced series as its sample mean (default: true when d = D = 0). */
  demean?: boolean
  /** Starting coefficients; default the CSS fit for `exact` and zeros for `css`. */
  start?: { ar?: VectorLike; ma?: VectorLike; seasonalAr?: VectorLike; seasonalMa?: VectorLike }
}

/**
 * A SARIMA(p, d, q)(P, D, Q)_s fitter as a traceable algorithm: Nelder–Mead over the partial-autocorrelation
 * coordinates of φ, Φ, θ and Θ separately (each factor stationary or invertible, so their products are), minimising the
 * conditional sum of squares or the negative exact profile log-likelihood of the differenced series (`armaFitSteps`
 * applied to the expanded model). `init` takes nothing.
 */
export function sarimaFitSteps(x: VectorLike, options: SarimaFitOptions): Algorithm<void, FitState<SarimaFit>> {
  const { p, q, P = 0, Q = 0, d = 0, D = 0, method = 'exact' } = options
  const s = options.period ?? 0
  const zeros = (m: number) => new Array<number>(m).fill(0)
  parse({ seasonalAr: zeros(P), seasonalMa: zeros(Q), period: s, diff: d, seasonalDiff: D }, 'sarimaFitSteps')
  const demean = options.demean ?? d + D === 0
  if (demean && d + D > 0) throw new DomainError('sarimaFitSteps', 'sarimaFitSteps: a mean needs d = D = 0')
  const w = differenced(toVec(x, 'sarimaFitSteps'), d, D, s)
  const mean = demean ? meanOf(tensor(w)) : 0
  const n = w.length
  const k = p + q + P + Q + 1 + (demean ? 1 : 0)
  const split = (u: number[]) => {
    const parts = [p, P, q, Q]
    const out: number[][] = []
    let at = 0
    for (const m of parts) {
      out.push(fromPacf(u.slice(at, at + m)))
      at += m
    }
    return { ar: out[0], sar: out[1], ma: out[2].map((v) => -v), sma: out[3].map((v) => -v) }
  }
  /** The factors and the expanded ARMA coefficients of w. */
  const model = (u: number[]) => {
    const c = split(u)
    return { c, e: expand({ ...c, s, d: 0, D: 0 }) }
  }
  const css = (ar: number[], ma: number[]) => {
    const e = residuals(w, ar, ma, mean)
    let ss = 0
    for (let t = ar.length; t < n; t++) ss += e[t] * e[t]
    return ss
  }
  const decode = (u: number[]): SarimaFit => {
    const { c, e } = model(u)
    const base = {
      ar: tensor(c.ar),
      ma: tensor(c.ma),
      seasonalAr: tensor(c.sar),
      seasonalMa: tensor(c.sma),
      period: s,
      diff: d,
      seasonalDiff: D,
      mean,
      method,
    }
    if (method === 'css') {
      const used = n - e.ar.length
      const sigma2 = css(e.ar, e.ma) / used
      const logLikelihood = -0.5 * used * (Math.log(2 * Math.PI * sigma2) + 1)
      return { ...base, sigma2, logLikelihood, aic: -2 * logLikelihood + 2 * k }
    }
    const L = exactLikelihood(w, e.ar, e.ma, mean)
    return { ...base, sigma2: L.sigma2Hat, logLikelihood: L.logLikelihood, aic: -2 * L.logLikelihood + 2 * k }
  }
  const objective = (u: number[]) => {
    const { e } = model(u)
    return method === 'css' ? css(e.ar, e.ma) : -exactLikelihood(w, e.ar, e.ma, mean).logLikelihood
  }
  const given = options.start
  let start: { ar: number[]; sar: number[]; ma: number[]; sma: number[] }
  if (given)
    start = {
      ar: given.ar ? toVec(given.ar, 'sarimaFitSteps') : new Array<number>(p).fill(0),
      sar: given.seasonalAr ? toVec(given.seasonalAr, 'sarimaFitSteps') : new Array<number>(P).fill(0),
      ma: given.ma ? toVec(given.ma, 'sarimaFitSteps') : new Array<number>(q).fill(0),
      sma: given.seasonalMa ? toVec(given.seasonalMa, 'sarimaFitSteps') : new Array<number>(Q).fill(0),
    }
  else if (method === 'exact' && p + q + P + Q > 0) {
    const c = run(sarimaFitSteps(x, { ...options, method: 'css', demean }), undefined, 4000).params
    start = { ar: toFlat(c.ar), sar: toFlat(c.seasonalAr), ma: toFlat(c.ma), sma: toFlat(c.seasonalMa) }
  } else
    start = {
      ar: new Array<number>(p).fill(0),
      sar: new Array<number>(P).fill(0),
      ma: new Array<number>(q).fill(0),
      sma: new Array<number>(Q).fill(0),
    }
  const neg = (c: number[]) => c.map((v) => -v)
  const u0 = [...toPacf(start.ar), ...toPacf(start.sar), ...toPacf(neg(start.ma)), ...toPacf(neg(start.sma))]
  const name = `sarima-${method}`
  // With no coefficients a one-dimensional dummy keeps the protocol uniform.
  if (u0.length === 0)
    return simplexFit(
      name,
      () => 0,
      () => decode([]),
      [0],
    )
  return simplexFit(name, objective, decode, u0)
}

/** Fit a seasonal ARIMA model (see `sarimaFitSteps`), with at most `maxSteps` Nelder–Mead steps (default 6000). */
export function fitSarima(
  x: VectorLike,
  options: SarimaFitOptions & { maxSteps?: number },
): SarimaFit & { converged: boolean; steps: number } {
  const s = run(sarimaFitSteps(x, options), undefined, options.maxSteps ?? 6000)
  return { ...s.params, converged: s.converged, steps: s.t }
}

/** A fitted model as a `SarimaSpec`. */
export function sarimaSpec(fit: SarimaFit): SarimaSpec {
  return {
    ar: fit.ar,
    ma: fit.ma,
    seasonalAr: fit.seasonalAr,
    seasonalMa: fit.seasonalMa,
    period: fit.period,
    diff: fit.diff,
    seasonalDiff: fit.seasonalDiff,
    sigma: Math.sqrt(fit.sigma2),
    mean: fit.mean,
  }
}

/**
 * h-step forecasts of a seasonal ARIMA model from the observed x: `forecastArma` on the integrated AR polynomial
 * φ(z)Φ(zˢ)(1 − z)ᵈ(1 − zˢ)ᴰ and the MA polynomial θ(z)Θ(zˢ), whose ψ weights grow without decaying when d + D > 0,
 * so the intervals widen without bound (Box, Jenkins & Reinsel, 2008, §5.2).
 */
export function forecastSarima(
  x: VectorLike,
  model: SarimaSpec,
  horizon: number,
  options: { level?: number } = {},
): Forecast {
  const p = parse(model, 'forecastSarima')
  const e = expand(p)
  return forecastArma(x, { ar: e.integrated, ma: e.ma, sigma: p.sigma, mean: p.mean }, horizon, options)
}

/** Conditional residuals of x under a seasonal model, from the integrated recursion (zero before the AR order). */
export function sarimaResiduals(x: VectorLike, model: SarimaSpec): Vector {
  const p = parse(model, 'sarimaResiduals')
  const e = expand(p)
  return tensor(residuals(toVec(x, 'sarimaResiduals'), e.integrated, e.ma, p.mean))
}
