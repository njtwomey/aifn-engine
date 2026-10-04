/**
 * Distributional-regression families (GAMLSS; Rigby and Stasinopoulos, 2005, "Generalized additive models for
 * location, scale and shape", JRSS C 54(3)): a parametric response distribution D(θ₁, …, θ_K) whose every parameter
 * θ_k has its own link g_k, with what fitting by the RS algorithm needs from it: the log-density, the score ∂ℓ/∂θ_k
 * and the expected second derivative E[∂²ℓ/∂θ_k²] of each parameter (the cross derivatives are not used), the cdf, the
 * quantile function, the mean and variance, and per-observation starting values.
 *
 * Families, in the parameterisations and with the default links of the R package gamlss.dist (Stasinopoulos and
 * Rigby, 2007):
 * - `normal` NO(μ, σ): y ~ N(μ, σ²); links identity, log.
 * - `student-t` TF(μ, σ, ν): y = μ + σT with T ~ t_ν; links identity, log, log.
 * - `box-cox-cole-green` BCCG(μ, σ, ν), the LMS method of Cole and Green (1992): z = ((y/μ)^ν − 1)/(νσ) (log(y/μ)/σ
 *   at ν = 0) is standard normal truncated to the image of y > 0; μ is close to the median, σ to the coefficient of
 *   variation and ν is the Box–Cox power that makes y symmetric (ν = 1 none, ν < 1 right skew). Links identity, log,
 *   identity.
 * - `gamma` GA(μ, σ): shape 1/σ², mean μ, so σ is the coefficient of variation; links log, log.
 * - `poisson` PO(μ); link log.
 *
 * Every function takes and returns plain numbers (one observation), so a fitter evaluates them in a loop.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import {
  digamma,
  logGamma,
  normalCdf,
  normalLogCdf,
  normalQuantile,
  regularisedGammaP,
  regularisedGammaPInverse,
  regularisedGammaQ,
  studentTCdf,
  studentTQuantile,
  trigamma,
} from 'aifn-compute/numerics/special'
import { link, type Link, type LinkName } from './families'

/** The name of a distribution parameter, in gamlss's order. */
export type DistributionalParameter = 'mu' | 'sigma' | 'nu'

/** The families of `distributionalFamily`. */
export type DistributionalFamilyName = 'normal' | 'student-t' | 'box-cox-cole-green' | 'gamma' | 'poisson'

/** One parameter of a family: its name, symbol, role, the links it takes (the first is the default) and its range. */
export interface DistributionalParameterInfo {
  readonly name: DistributionalParameter
  /** TeX symbol, e.g. `\\sigma`. */
  readonly tex: string
  readonly role: 'location' | 'scale' | 'shape'
  readonly links: readonly LinkName[]
  /** The open range of valid values. */
  readonly range: 'real' | 'positive'
}

/**
 * A distributional-regression family (see the module comment). θ is one value per parameter, in `parameters` order.
 */
export interface DistributionalFamily {
  readonly name: DistributionalFamilyName
  /** gamlss.dist's abbreviation: NO, TF, BCCG, GA, PO. */
  readonly abbreviation: string
  readonly label: string
  readonly parameters: readonly DistributionalParameterInfo[]
  readonly support: 'real' | 'positive' | 'non-negative-integers'
  /** log p(y | θ). */
  logPdf(y: number, theta: readonly number[]): number
  /** ∂ℓ/∂θ_k at y. */
  score(k: number, y: number, theta: readonly number[]): number
  /** E[∂²ℓ/∂θ_k²] (negative): minus the expected information of θ_k. */
  expectedSecond(k: number, theta: readonly number[]): number
  cdf(y: number, theta: readonly number[]): number
  quantile(p: number, theta: readonly number[]): number
  /** The mean and variance (NaN or ∞ where they do not exist). */
  mean(theta: readonly number[]): number
  variance(theta: readonly number[]): number
  /** Whether θ lies in the parameter space. */
  valid(theta: readonly number[]): boolean
  /** Starting values of each parameter at each observation (gamlss.dist's `*.initial`). */
  initial(y: ArrayLike<number>): Float64Array[]
}

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)

function moments(y: ArrayLike<number>) {
  let m = 0
  for (let i = 0; i < y.length; i++) m += y[i]
  m /= y.length
  let v = 0
  for (let i = 0; i < y.length; i++) v += (y[i] - m) ** 2
  return { mean: m, sd: Math.sqrt(v / Math.max(1, y.length - 1)) }
}
const fill = (n: number, v: number) => new Float64Array(n).fill(v)
const halfway = (y: ArrayLike<number>, m: number) => Float64Array.from(y, (v) => (v + m) / 2)

/** Mean and variance by the midpoint rule over m quantiles (for families without closed forms). */
function quantileMoments(q: (p: number) => number, m = 400) {
  let s = 0
  let s2 = 0
  for (let i = 0; i < m; i++) {
    const v = q((i + 0.5) / m)
    s += v
    s2 += v * v
  }
  const mean = s / m
  return { mean, variance: Math.max(0, s2 / m - mean * mean) }
}

const param = (
  name: DistributionalParameter,
  tex: string,
  role: DistributionalParameterInfo['role'],
  links: LinkName[],
  range: DistributionalParameterInfo['range'],
): DistributionalParameterInfo => ({ name, tex, role, links, range })

/** NO(μ, σ): the normal distribution with mean μ and standard deviation σ. */
export function normalDistributional(): DistributionalFamily {
  return {
    name: 'normal',
    abbreviation: 'NO',
    label: 'Normal',
    parameters: [
      param('mu', '\\mu', 'location', ['identity', 'log'], 'real'),
      param('sigma', '\\sigma', 'scale', ['log', 'identity'], 'positive'),
    ],
    support: 'real',
    logPdf: (y, [m, s]) => -Math.log(s) - LOG_SQRT_2PI - 0.5 * ((y - m) / s) ** 2,
    score: (k, y, [m, s]) => (k === 0 ? (y - m) / (s * s) : ((y - m) ** 2 - s * s) / (s * s * s)),
    expectedSecond: (k, [, s]) => (k === 0 ? -1 / (s * s) : -2 / (s * s)),
    cdf: (y, [m, s]) => normalCdf((y - m) / s),
    quantile: (p, [m, s]) => m + s * normalQuantile(p),
    mean: ([m]) => m,
    variance: ([, s]) => s * s,
    valid: ([m, s]) => Number.isFinite(m) && s > 0,
    initial: (y) => {
      const { mean, sd } = moments(y)
      return [halfway(y, mean), fill(y.length, sd)]
    },
  }
}

/** TF(μ, σ, ν): y = μ + σT with T Student-t on ν degrees of freedom. */
export function studentTDistributional(): DistributionalFamily {
  return {
    name: 'student-t',
    abbreviation: 'TF',
    label: 'Student t',
    parameters: [
      param('mu', '\\mu', 'location', ['identity', 'log'], 'real'),
      param('sigma', '\\sigma', 'scale', ['log', 'identity'], 'positive'),
      param('nu', '\\nu', 'shape', ['log', 'identity'], 'positive'),
    ],
    support: 'real',
    logPdf: (y, [m, s, v]) =>
      logGamma((v + 1) / 2) -
      logGamma(v / 2) -
      0.5 * Math.log(Math.PI * v) -
      Math.log(s) -
      ((v + 1) / 2) * Math.log1p(((y - m) / s) ** 2 / v),
    score: (k, y, [m, s, v]) => {
      const d2 = ((y - m) / s) ** 2
      const omega = (v + 1) / (v + d2)
      if (k === 0) return (omega * (y - m)) / (s * s)
      if (k === 1) return (omega * d2 - 1) / s
      return -0.5 * Math.log1p(d2 / v) + (omega * d2 - 1) / (2 * v) + 0.5 * (digamma((v + 1) / 2) - digamma(v / 2))
    },
    // The expected information of the t (Lange, Little and Taylor, 1989, JASA 84(408), §2).
    expectedSecond: (k, [, s, v]) => {
      if (k === 0) return -(v + 1) / ((v + 3) * s * s)
      if (k === 1) return (-2 * v) / ((v + 3) * s * s)
      return 0.25 * (trigamma((v + 1) / 2) - trigamma(v / 2)) + (v + 5) / (2 * v * (v + 1) * (v + 3))
    },
    cdf: (y, [m, s, v]) => studentTCdf((y - m) / s, v),
    quantile: (p, [m, s, v]) => m + s * studentTQuantile(p, v),
    mean: ([m, , v]) => (v > 1 ? m : NaN),
    variance: ([, s, v]) => (v > 2 ? (s * s * v) / (v - 2) : v > 1 ? Infinity : NaN),
    valid: ([m, s, v]) => Number.isFinite(m) && s > 0 && v > 0,
    initial: (y) => {
      const { mean, sd } = moments(y)
      return [halfway(y, mean), fill(y.length, sd), fill(y.length, 10)]
    },
  }
}

/** |ν| below this is treated as ν = 0 (the log transform) by BCCG. */
const NU_ZERO = 1e-7

/** BCCG(μ, σ, ν): the Box–Cox Cole–Green distribution (the LMS method), truncated to y > 0. */
export function boxCoxColeGreenDistributional(): DistributionalFamily {
  const zOf = (y: number, m: number, s: number, v: number) =>
    Math.abs(v) < NU_ZERO ? Math.log(y / m) / s : (Math.pow(y / m, v) - 1) / (v * s)
  // The truncation: z is bounded by ∓1/(σ|ν|), and Φ(1/(σ|ν|)) is the mass kept; h is the inverse Mills ratio there.
  const bound = (s: number, v: number) => (Math.abs(v) < NU_ZERO ? Infinity : 1 / (s * Math.abs(v)))
  const mills = (s: number, v: number) => {
    const a = bound(s, v)
    if (!Number.isFinite(a)) return 0
    return Math.exp(-0.5 * a * a - LOG_SQRT_2PI - normalLogCdf(a))
  }
  const quantile = (p: number, [m, s, v]: readonly number[]) => {
    const a = bound(s, v)
    const kept = Number.isFinite(a) ? normalCdf(a) : 1
    const z = v <= 0 ? normalQuantile(p * kept) : normalQuantile(1 - (1 - p) * kept)
    return Math.abs(v) < NU_ZERO ? m * Math.exp(s * z) : m * Math.pow(Math.max(0, v * s * z + 1), 1 / v)
  }
  return {
    name: 'box-cox-cole-green',
    abbreviation: 'BCCG',
    label: 'Box–Cox Cole–Green',
    parameters: [
      param('mu', '\\mu', 'location', ['identity', 'log'], 'positive'),
      param('sigma', '\\sigma', 'scale', ['log', 'identity'], 'positive'),
      param('nu', '\\nu', 'shape', ['identity'], 'real'),
    ],
    support: 'positive',
    logPdf: (y, [m, s, v]) => {
      if (!(y > 0)) return -Infinity
      const z = zOf(y, m, s, v)
      const a = bound(s, v)
      const trunc = Number.isFinite(a) ? normalLogCdf(a) : 0
      return (v - 1) * Math.log(y) - v * Math.log(m) - Math.log(s) - LOG_SQRT_2PI - 0.5 * z * z - trunc
    },
    score: (k, y, [m, s, v]) => {
      const z = zOf(y, m, s, v)
      if (k === 0) return (z / s + v * (z * z - 1)) / m
      const h = mills(s, v)
      if (k === 1) return (z * z - 1) / s + (Math.abs(v) < NU_ZERO ? 0 : h / (s * s * Math.abs(v)))
      const l = Math.log(y / m)
      // The limit ν → 0, where z = l/σ and ∂z/∂ν = l²/(2σ).
      if (Math.abs(v) < NU_ZERO) return l - (l * l * l) / (2 * s * s)
      return (z / v) * (z - l / s) - l * (z * z - 1) + (Math.sign(v) * h) / (s * v * v)
    },
    // gamlss.dist's approximations, which ignore the truncation (Cole and Green, 1992, appendix).
    expectedSecond: (k, [m, s, v]) => {
      if (k === 0) return -(1 + 2 * v * v * s * s) / (m * m * s * s)
      if (k === 1) return -2 / (s * s)
      return -1.75 * s * s
    },
    cdf: (y, [m, s, v]) => {
      if (!(y > 0)) return 0
      const z = zOf(y, m, s, v)
      const a = bound(s, v)
      if (!Number.isFinite(a)) return normalCdf(z)
      return v > 0 ? (normalCdf(z) - normalCdf(-a)) / normalCdf(a) : normalCdf(z) / normalCdf(a)
    },
    quantile,
    mean: (theta) => quantileMoments((p) => quantile(p, theta)).mean,
    variance: (theta) => quantileMoments((p) => quantile(p, theta)).variance,
    valid: ([m, s, v]) => m > 0 && s > 0 && Number.isFinite(v),
    initial: (y) => {
      const { mean } = moments(y)
      return [halfway(y, mean), fill(y.length, 0.1), fill(y.length, 0.5)]
    },
  }
}

/** GA(μ, σ): the gamma distribution with mean μ and coefficient of variation σ (shape 1/σ², scale μσ²). */
export function gammaDistributional(): DistributionalFamily {
  return {
    name: 'gamma',
    abbreviation: 'GA',
    label: 'Gamma',
    parameters: [
      param('mu', '\\mu', 'location', ['log', 'identity', 'inverse'], 'positive'),
      param('sigma', '\\sigma', 'scale', ['log', 'identity'], 'positive'),
    ],
    support: 'positive',
    logPdf: (y, [m, s]) => {
      if (!(y > 0)) return -Infinity
      const a = 1 / (s * s)
      return (a - 1) * Math.log(y) - (a * y) / m + a * Math.log(a / m) - logGamma(a)
    },
    score: (k, y, [m, s]) => {
      if (k === 0) return (y - m) / (s * s * m * m)
      const a = 1 / (s * s)
      return (2 / (s * s * s)) * (y / m - Math.log(y) + Math.log(m) + Math.log(s * s) - 1 + digamma(a))
    },
    expectedSecond: (k, [m, s]) => (k === 0 ? -1 / (s * s * m * m) : 4 / s ** 4 - (4 / s ** 6) * trigamma(1 / (s * s))),
    cdf: (y, [m, s]) => (y > 0 ? regularisedGammaP(1 / (s * s), y / (m * s * s)) : 0),
    quantile: (p, [m, s]) => m * s * s * regularisedGammaPInverse(1 / (s * s), p),
    mean: ([m]) => m,
    variance: ([m, s]) => (s * m) ** 2,
    valid: ([m, s]) => m > 0 && s > 0,
    initial: (y) => {
      const { mean } = moments(y)
      return [halfway(y, mean), fill(y.length, 1)]
    },
  }
}

/** PO(μ): the Poisson distribution with mean μ. */
export function poissonDistributional(): DistributionalFamily {
  const cdf = (y: number, m: number) => (y < 0 ? 0 : regularisedGammaQ(Math.floor(y) + 1, m))
  return {
    name: 'poisson',
    abbreviation: 'PO',
    label: 'Poisson',
    parameters: [param('mu', '\\mu', 'location', ['log', 'identity', 'sqrt'], 'positive')],
    support: 'non-negative-integers',
    logPdf: (y, [m]) => (y >= 0 && Number.isInteger(y) ? y * Math.log(m) - m - logGamma(y + 1) : -Infinity),
    score: (_k, y, [m]) => y / m - 1,
    expectedSecond: (_k, [m]) => -1 / m,
    cdf: (y, [m]) => cdf(y, m),
    quantile: (p, [m]) => {
      // Start from the normal approximation and walk to the smallest k with F(k) ≥ p.
      let k = Math.max(0, Math.floor(m + Math.sqrt(m) * normalQuantile(Math.min(Math.max(p, 1e-12), 1 - 1e-12))))
      while (k > 0 && cdf(k - 1, m) >= p) k--
      while (cdf(k, m) < p) k++
      return k
    },
    mean: ([m]) => m,
    variance: ([m]) => m,
    valid: ([m]) => m > 0,
    initial: (y) => {
      const { mean } = moments(y)
      return [halfway(y, mean)]
    },
  }
}

const FAMILIES: Record<DistributionalFamilyName, () => DistributionalFamily> = {
  normal: normalDistributional,
  'student-t': studentTDistributional,
  'box-cox-cole-green': boxCoxColeGreenDistributional,
  gamma: gammaDistributional,
  poisson: poissonDistributional,
}

/** The distributional family by name (see the module comment). */
export function distributionalFamily(name: DistributionalFamilyName): DistributionalFamily {
  const f = FAMILIES[name]
  if (!f) throw new DomainError('distributionalFamily', `distributionalFamily: unknown family "${name}"`)
  return f()
}

/** The links of a family's parameters: the given names, or each parameter's default; a link it does not take throws. */
export function distributionalLinks(
  family: DistributionalFamily,
  names: Partial<Record<DistributionalParameter, LinkName>> = {},
): Link[] {
  return family.parameters.map((p) => {
    const name = names[p.name] ?? p.links[0]
    if (!p.links.includes(name))
      throw new DomainError('distributionalLinks', `${family.abbreviation}: ${p.name} takes ${p.links.join(', ')}`)
    return link(name)
  })
}

/**
 * The normalised quantile residual r = Φ⁻¹(F(y | θ)) (Dunn and Smyth, 1996); for a discrete family the mid-point
 * Φ⁻¹((F(y − 1) + F(y))/2). Under the true model the residuals are standard normal (continuous families).
 */
export function quantileResidual(family: DistributionalFamily, y: number, theta: readonly number[]): number {
  const u =
    family.support === 'non-negative-integers'
      ? (family.cdf(y - 1, theta) + family.cdf(y, theta)) / 2
      : family.cdf(y, theta)
  return normalQuantile(Math.min(Math.max(u, 1e-15), 1 - 1e-15))
}

/** A worm plot's data (see `wormPlot`). */
export type WormPlot = {
  /** Standard normal quantiles Φ⁻¹((i − ½)/n) of the sorted residuals, [n]. */
  x: Float64Array
  /** Each sorted residual minus its normal quantile, [n]. */
  y: Float64Array
  /** The pointwise approximate 95% band ±1.96 √(p(1 − p)/n)/φ(x) at `x`. */
  lower: Float64Array
  upper: Float64Array
}

/**
 * The worm plot (van Buuren and Fredriks, 2001): a normal Q–Q plot of residuals detrended by subtracting the identity
 * line, so departures from normality show as shapes about zero (a shift: the mean; a slope: the variance; a U or
 * inverted U: skewness; an S: kurtosis), with a pointwise 95% band.
 */
export function wormPlot(residuals: ArrayLike<number>): WormPlot {
  const r = Float64Array.from(residuals).sort()
  const n = r.length
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  const lower = new Float64Array(n)
  const upper = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const p = (i + 0.5) / n
    x[i] = normalQuantile(p)
    y[i] = r[i] - x[i]
    const half = (1.959964 * Math.sqrt((p * (1 - p)) / n)) / Math.exp(-0.5 * x[i] * x[i] - LOG_SQRT_2PI)
    lower[i] = -half
    upper[i] = half
  }
  return { x, y, lower, upper }
}
