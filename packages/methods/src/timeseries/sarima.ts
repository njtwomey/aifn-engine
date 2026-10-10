/**
 * Seasonal ARIMA, SARIMA($p$, $d$, $q$)($P$, $D$, $Q$)$_s$ (Box, Jenkins & Reinsel, 2008, ch. 9):
 * $\phi(B) \Phi(B^s) \nabla^d \nabla_s^D x_t = \theta(B) \Theta(B^s) \varepsilon_t$,
 * $\varepsilon_t \sim \Gauss(0, \sigma^2)$.
 *
 * Here $B$ is the backshift, $\nabla = 1 - B$, $\nabla_s = 1 - B^s$, $\phi(z) = 1 - \sum_i \phi_i z^i$,
 * $\Phi(z) = 1 - \sum_i \Phi_i z^i$, $\theta(z) = 1 + \sum_j \theta_j z^j$ and $\Theta(z) = 1 + \sum_j \Theta_j z^j$
 * (statsmodels' signs). The differenced series $w = \nabla^d \nabla_s^D x$ is an ARMA($p + sP$, $q + sQ$) process whose
 * lag polynomials are the products $\phi(z)\Phi(z^s)$ and $\theta(z)\Theta(z^s)$, so the exact likelihood is the ARMA
 * one: the Kalman filter of `aifn-compute/inference/filtering` on Harvey's state-space form of the expanded model (as
 * `armaLogLikelihood`). This is statsmodels SARIMAX with `simple_differencing=True`: the likelihood of $w$, not of $x$
 * with a diffuse start. Forecasts run the recursion of the integrated polynomial
 * $\phi(z)\Phi(z^s)(1 - z)^d(1 - z^s)^D$ on $x$ itself.
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

/** A seasonal ARIMA model. Omitted parts are empty, $d = D = 0$, $\sigma = 1$ and $\mu = 0$. */
export type SarimaSpec = {
  /** The AR coefficients $\phi_1, \dots, \phi_p$. */
  ar?: VectorLike
  /** The MA coefficients $\theta_1, \dots, \theta_q$. */
  ma?: VectorLike
  /** The seasonal AR coefficients $\Phi_1, \dots, \Phi_P$, at lags $s, 2s, \dots$. */
  seasonalAr?: VectorLike
  /** The seasonal MA coefficients $\Theta_1, \dots, \Theta_Q$, at lags $s, 2s, \dots$. */
  seasonalMa?: VectorLike
  /** The season length $s$, an integer of at least 2 (required with any seasonal term or seasonal difference). */
  period?: number
  /** $d$, the number of ordinary differences. */
  diff?: number
  /** $D$, the number of seasonal differences. */
  seasonalDiff?: number
  /** The innovation standard deviation $\sigma$. */
  sigma?: number
  /** The mean $\mu$ of the differenced series (only without differencing; a mean after differencing is a drift). */
  mean?: number
}

/**
 * A `SarimaSpec` with its defaults filled in: the coefficients as arrays (`sar`, `sma` the seasonal ones), the period
 * `s` (0 when not given), the differencing orders `d` and `D`, `sigma` and `mean`.
 */
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

/**
 * Read a `SarimaSpec` into arrays, filling the defaults. Throws `DomainError` when a seasonal term or difference has no
 * integer period of at least 2, a differencing order is not a non-negative integer, or a non-zero mean is given with
 * differencing.
 *
 * @param m The model as given by the caller.
 * @param where The caller's name, for error messages.
 * @returns The model with its defaults.
 */
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

/**
 * The product of two polynomials given by their coefficients $c_0, c_1, \dots$, constant term first.
 *
 * @param a The coefficients of the first polynomial.
 * @param b The coefficients of the second polynomial.
 * @returns The coefficients of the product, $\mathrm{len}(a) + \mathrm{len}(b) - 1$ of them.
 */
function multiply(a: number[], b: number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0)
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j]
  return out
}

/**
 * The seasonal polynomial $1 + \sum_i c_i z^{si}$, as coefficients constant term first.
 *
 * @param c The coefficients $c_1, c_2, \dots$ at lags $s, 2s, \dots$.
 * @param s The season length.
 * @returns The $\mathrm{len}(c) \cdot s + 1$ coefficients.
 */
const seasonal = (c: number[], s: number): number[] => {
  const out = new Array<number>(c.length * s + 1).fill(0)
  out[0] = 1
  c.forEach((v, i) => (out[(i + 1) * s] = v))
  return out
}

/**
 * The expanded lag polynomials as ARMA coefficients, trailing zeros dropped: `ar` the $\phi^*_i$ of
 * $\phi(z)\Phi(z^s) = 1 - \sum_i \phi^*_i z^i$, `ma` the $\theta^*_j$ of
 * $\theta(z)\Theta(z^s) = 1 + \sum_j \theta^*_j z^j$, and `integrated` the AR coefficients of
 * $\phi(z)\Phi(z^s)(1 - z)^d(1 - z^s)^D$ in the same sign convention as `ar`.
 *
 * @param p The parsed model: its coefficients, period and differencing orders.
 * @returns The coefficients `ar`, `ma` and `integrated` as arrays.
 */
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
 * The ARMA coefficients of a seasonal model: `ar` and `ma` of the differenced series $w$ (the products
 * $\phi(z)\Phi(z^s)$ and $\theta(z)\Theta(z^s)$ multiplied out, so
 * $w_t = \sum_i \mathrm{ar}_i w_{t-i} + \varepsilon_t + \sum_j \mathrm{ma}_j \varepsilon_{t-j}$), and `integrated`,
 * the AR coefficients of $\phi(z)\Phi(z^s)(1 - z)^d(1 - z^s)^D$, the model of $x$ itself as a (non-stationary) ARMA.
 * Trailing zero coefficients are dropped. Throws `DomainError` for an invalid model (see `SarimaSpec`).
 *
 * @param model The seasonal model; `sigma` and `mean` are not used.
 * @returns The expanded `ar`, `ma` and `integrated` coefficients, lag 1 first.
 *
 * @example $(1 - 0.5z)(1 - 0.3z^4)$ multiplied out
 * const e = expandSarima({ ar: [0.5], seasonalAr: [0.3], period: 4 })
 * print('ar:', e.ar)
 * print('ma:', e.ma)
 *
 * @example A difference adds a unit root to the integrated polynomial
 * // (1 - 0.5z)(1 - z) = 1 - 1.5z + 0.5z²
 * print('integrated:', expandSarima({ ar: [0.5], diff: 1 }).integrated)
 */
export function expandSarima(model: SarimaSpec): { ar: Vector; ma: Vector; integrated: Vector } {
  const e = expand(parse(model, 'expandSarima'))
  return { ar: tensor(e.ar), ma: tensor(e.ma), integrated: tensor(e.integrated) }
}

/**
 * The differenced series $\nabla^d \nabla_s^D x$ (seasonal differences first): the series shortens by $d + sD$.
 *
 * @param x The series.
 * @param d The number of ordinary differences.
 * @param D The number of seasonal differences.
 * @param s The season length (unused when $D = 0$).
 * @returns The differenced series, $d + sD$ values shorter than `x`.
 */
function differenced(x: number[], d: number, D: number, s: number): number[] {
  let w = x
  for (let k = 0; k < D; k++) w = w.slice(s).map((v, t) => v - w[t])
  for (let k = 0; k < d; k++) w = w.slice(1).map((v, t) => v - w[t])
  return w
}

/**
 * The exact Gaussian log-likelihood of the differenced series $\nabla^d \nabla_s^D x$ under a seasonal model (see the
 * file notes): `armaLogLikelihood` of the expanded ARMA, with $\sigma^2$ concentrated out when `model.sigma` is
 * omitted. $-\infty$ when $\phi(z)\Phi(z^s)$ is not stationary. The first $d + sD$ values of `x` only enter through the
 * differences.
 *
 * @param x The observed series, undifferenced.
 * @param model The seasonal model; all of it is read, `sigma` only if given.
 * @returns The log-likelihood of the differenced series, its prediction errors and their variances, and
 *   $\hat\sigma^2$.
 *
 * @example The likelihood peaks near the true seasonal coefficient
 * const { x } = simulateSarima(stream(2), { seasonalAr: [0.6], period: 4 }, 200)
 * for (const Phi of [0.3, 0.6, 0.9]) {
 *   print('Φ =', Phi, 'log L =', sarimaLogLikelihood(x, { seasonalAr: [Phi], period: 4 }).logLikelihood)
 * }
 *
 * @example A random walk's prediction errors are its steps
 * const L = sarimaLogLikelihood([1, 3, 6, 10, 15], { diff: 1, sigma: 1 })
 * print('innovations:', L.innovations)
 * print('log L:', L.logLikelihood)
 */
export function sarimaLogLikelihood(x: VectorLike, model: SarimaSpec): ArmaLikelihood {
  const p = parse(model, 'sarimaLogLikelihood')
  const e = expand(p)
  return exactLikelihood(differenced(toVec(x, 'sarimaLogLikelihood'), p.d, p.D, p.s), e.ar, e.ma, p.mean, model.sigma)
}

/** A simulated seasonal ARIMA series. */
export type SarimaSimulation = {
  /** The series $x_1, \dots, x_n$. */
  x: Vector
  /**
   * The ARMA process $w$ that was integrated, $n + d + sD$ values. Its first $d + sD$ values start the integration, and
   * the integrated values at those positions are dropped from `x`.
   */
  differenced: Vector
  /** Whether $\phi(z)\Phi(z^s)$ is stationary (as `simulateArma` reports it for $w$). */
  stationary: boolean
  /** Whether $w$ overflowed to a non-finite value (as `simulateArma` reports it). */
  diverged: boolean
}

/**
 * Simulate $n$ values of a seasonal ARIMA model: the ARMA process $w$ of the differenced series (`simulateArma` on the
 * expanded polynomials, with its default burn-in), integrated $d$ and $D$ times from zero starting values, and the
 * first $d + sD$ integrated values dropped. Throws `DomainError` for an invalid model (see `SarimaSpec`).
 *
 * @param s The random stream the innovations are drawn from; advanced in place.
 * @param model The seasonal model; all of it is read.
 * @param n The number of values returned.
 * @returns The series, the differenced process it integrates, and `simulateArma`'s flags for that process.
 *
 * @example A seasonal AR series of period 4
 * const sim = simulateSarima(stream(1), { seasonalAr: [0.6], period: 4 }, 8)
 * print('x:', sim.x)
 *
 * @example Differencing the simulated series gives back the ARMA process
 * const sim = simulateSarima(stream(1), { diff: 1 }, 6)
 * print('difference(x):', difference(sim.x))
 * print('w:            ', sim.differenced)
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
  /** The fitted AR coefficients $\phi_1, \dots, \phi_p$ (stationary). */
  ar: Vector
  /** The fitted MA coefficients $\theta_1, \dots, \theta_q$ (invertible). */
  ma: Vector
  /** The fitted seasonal AR coefficients $\Phi_1, \dots, \Phi_P$ (stationary). */
  seasonalAr: Vector
  /** The fitted seasonal MA coefficients $\Theta_1, \dots, \Theta_Q$ (invertible). */
  seasonalMa: Vector
  /** The season length $s$ (0 when none was given). */
  period: number
  /** $d$, as given. */
  diff: number
  /** $D$, as given. */
  seasonalDiff: number
  /** The mean of the differenced series (0 unless it was estimated, which needs $d = D = 0$). */
  mean: number
  /**
   * The innovation variance $\hat\sigma^2$: the concentrated estimate (`exact`), or the CSS over the $n - p - sP$
   * residuals it sums ($n$ the length of the differenced series; `css`).
   */
  sigma2: number
  /** The exact profile log-likelihood of the differenced series (`exact`), or the conditional one (`css`). */
  logLikelihood: number
  /** $-2 \log L + 2k$, with $k = p + q + P + Q + 1$ ($\sigma^2$) $+ 1$ if the mean was estimated. */
  aic: number
  /** The objective that was optimised. */
  method: 'css' | 'exact'
}

/** Options for `sarimaFitSteps` and `fitSarima`. */
export type SarimaFitOptions = {
  /** The AR order $p$. */
  p: number
  /** The number of ordinary differences $d$ (default 0). */
  d?: number
  /** The MA order $q$. */
  q: number
  /** The seasonal AR order $P$ (default 0). */
  P?: number
  /** The number of seasonal differences $D$ (default 0). */
  D?: number
  /** The seasonal MA order $Q$ (default 0). */
  Q?: number
  /** The season length $s$ (required with any seasonal order). */
  period?: number
  /** `css` minimises the conditional sum of squares; `exact` maximises the exact likelihood. Default `exact`. */
  method?: 'css' | 'exact'
  /** Estimate the mean of the differenced series as its sample mean (default: true when $d = D = 0$, and only then). */
  demean?: boolean
  /** Starting coefficients; default the CSS fit for `exact` and zeros for `css`. */
  start?: { ar?: VectorLike; ma?: VectorLike; seasonalAr?: VectorLike; seasonalMa?: VectorLike }
}

/**
 * A SARIMA($p$, $d$, $q$)($P$, $D$, $Q$)$_s$ fitter as a traceable algorithm: Nelder–Mead over the
 * partial-autocorrelation coordinates of $\phi$, $\Phi$, $\theta$ and $\Theta$ separately (each factor stationary or
 * invertible, so their products are), minimising the conditional sum of squares or the negative exact profile
 * log-likelihood of the differenced series (`armaFitSteps` applied to the expanded model). Without `start`, the `exact`
 * fit starts from a CSS fit of up to 4000 steps, run when the algorithm is built. Throws `DomainError` for an invalid
 * period or differencing order, or `demean` with differencing. `init` takes no start (`run(alg, undefined, steps)`).
 *
 * @param x The observed series, undifferenced.
 * @param options The orders, period, objective, mean and starting coefficients (see `SarimaFitOptions`).
 * @returns The algorithm; its state is a `FitState` whose `params` is a `SarimaFit`.
 *
 * @example Watch a CSS fit find a seasonal AR coefficient
 * const { x } = simulateSarima(stream(2), { seasonalAr: [0.6], period: 4 }, 100)
 * const alg = sarimaFitSteps(x, { p: 0, q: 0, P: 1, period: 4, method: 'css' })
 * for (const steps of [1, 10, 40]) print(steps, 'steps: Φ =', run(alg, undefined, steps).params.seasonalAr)
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

/**
 * Fit a seasonal ARIMA model by running `sarimaFitSteps` until Nelder–Mead converges or `maxSteps` steps (default
 * 6000) have run. Check `converged` before trusting the result.
 *
 * @param x The observed series, undifferenced.
 * @param options The options of `sarimaFitSteps` (see `SarimaFitOptions`), and `maxSteps`, the most Nelder–Mead steps
 *   to run (default 6000).
 * @returns The fit at the best vertex, with whether the optimiser converged and the number of steps taken.
 *
 * @example Recover the AR coefficient of an ARIMA(1, 1, 0)
 * const { x } = simulateSarima(stream(2), { ar: [0.5], diff: 1 }, 100)
 * const fit = fitSarima(x, { p: 1, d: 1, q: 0 })
 * print('φ̂ =', fit.ar, '(true 0.5)')
 * print('σ̂² =', fit.sigma2, '(true 1)')
 * print('converged:', fit.converged)
 */
export function fitSarima(
  x: VectorLike,
  options: SarimaFitOptions & { maxSteps?: number },
): SarimaFit & { converged: boolean; steps: number } {
  const s = run(sarimaFitSteps(x, options), undefined, options.maxSteps ?? 6000)
  return { ...s.params, converged: s.converged, steps: s.t }
}

/**
 * A fitted model as a `SarimaSpec`, ready for `forecastSarima`, `sarimaResiduals` or `simulateSarima`: the
 * coefficients, period and differencing orders as fitted, $\sigma = \sqrt{\hat\sigma^2}$ and the mean.
 *
 * @param fit A fit from `fitSarima` or a `sarimaFitSteps` state's `params`.
 * @returns The model the fit describes.
 *
 * @example Fit, then forecast with the fitted model
 * const { x } = simulateSarima(stream(2), { ar: [0.5], diff: 1 }, 100)
 * const spec = sarimaSpec(fitSarima(x, { p: 1, d: 1, q: 0 }))
 * print('σ =', spec.sigma)
 * print('forecast:', forecastSarima(x, spec, 3).mean)
 */
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
 * Forecasts of a seasonal ARIMA model $h = 1, \dots, \mathrm{horizon}$ steps past the observed $x$: `forecastArma` on
 * the integrated AR polynomial $\phi(z)\Phi(z^s)(1 - z)^d(1 - z^s)^D$ and the MA polynomial $\theta(z)\Theta(z^s)$,
 * whose $\psi$ weights do not decay when $d + D > 0$, so the intervals widen without bound (Box, Jenkins & Reinsel,
 * 2008, §5.2). The standard errors use the model's `sigma` (default 1).
 *
 * @param x The observed series, undifferenced.
 * @param model The seasonal model; all of it is read.
 * @param horizon The number of steps ahead to forecast.
 * @param options `level`, the coverage of the prediction intervals (default 0.95).
 * @returns The forecasts with their standard errors and intervals.
 *
 * @example A random walk forecasts its last value, with a $\sqrt{h}$ standard error
 * const f = forecastSarima([1, 2, 3, 4, 5], { diff: 1 }, 3)
 * print('forecast:', f.mean)
 * print('se:', f.se)
 *
 * @example A seasonal difference repeats the last season
 * const f = forecastSarima([10, 20, 30, 40, 11, 21, 31, 41], { seasonalDiff: 1, period: 4 }, 6)
 * print('forecast:', f.mean)
 * print('se:', f.se)
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

/**
 * Conditional residuals of $x$ under a seasonal model, from the recursion of the integrated AR polynomial and the
 * expanded MA polynomial (`armaResiduals` on $x$ itself). The first entries, as many as the degree of
 * $\phi(z)\Phi(z^s)(1 - z)^d(1 - z^s)^D$, are zero.
 *
 * @param x The observed series, undifferenced.
 * @param model The seasonal model; `sigma` is not used.
 * @returns The residuals, one per observation.
 *
 * @example The residuals of a random walk are its steps
 * print(sarimaResiduals([1, 3, 6, 10], { diff: 1 }))
 * print(sarimaResiduals([1, 3, 6, 10, 15], { ar: [0.5], diff: 1 }))
 */
export function sarimaResiduals(x: VectorLike, model: SarimaSpec): Vector {
  const p = parse(model, 'sarimaResiduals')
  const e = expand(p)
  return tensor(residuals(toVec(x, 'sarimaResiduals'), e.integrated, e.ma, p.mean))
}
