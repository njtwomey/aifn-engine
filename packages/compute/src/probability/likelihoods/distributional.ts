/**
 * Distributional-regression (GAMLSS) families: response distributions whose every parameter has its own link, with
 * the scores and expected second derivatives that fitting needs, and the quantile residual and worm plot that check a
 * fit.
 *
 * GAMLSS is Rigby and Stasinopoulos (2005), "Generalized additive models for location, scale and shape", JRSS C
 * 54(3): a parametric response distribution $D(\theta_1, \dots, \theta_K)$ whose every parameter $\theta_k$ has its
 * own link $g_k$. A family provides what fitting by the RS algorithm needs: the log-density, the score
 * $\partial\ell/\partial\theta_k$ and the expected second derivative
 * $\expect[\partial^2\ell/\partial\theta_k^2]$ of each parameter (the cross derivatives are not used), the cdf,
 * the quantile function, the mean and variance, and per-observation starting values.
 *
 * Families, in the parameterisations and with the default links of the R package gamlss.dist (Stasinopoulos and
 * Rigby, 2007):
 * - `normal` $\operatorname{NO}(\mu, \sigma)$: $y \sim \Gauss(\mu, \sigma^2)$; links identity, log.
 * - `student-t` $\operatorname{TF}(\mu, \sigma, \nu)$: $y = \mu + \sigma T$ with $T \sim t_\nu$; links
 *   identity, log, log.
 * - `box-cox-cole-green` $\operatorname{BCCG}(\mu, \sigma, \nu)$, the LMS method of Cole and Green (1992):
 *   $z = ((y/\mu)^\nu - 1)/(\nu\sigma)$ ($\log(y/\mu)/\sigma$ at $\nu = 0$) is standard normal truncated to the
 *   image of $y > 0$; $\mu$ is close to the median, $\sigma$ to the coefficient of variation and $\nu$ is the
 *   Box–Cox power that makes $y$ symmetric ($\nu = 1$ none, $\nu < 1$ right skew). Links identity, log, identity.
 * - `gamma` $\operatorname{GA}(\mu, \sigma)$: shape $1/\sigma^2$, mean $\mu$, so $\sigma$ is the coefficient of
 *   variation; links log, log.
 * - `poisson` $\operatorname{PO}(\mu)$; link log.
 *
 * Every function takes and returns plain numbers (one observation), so a fitter evaluates them in a loop; none is
 * differentiable by `grad`, since the derivatives a fitter needs are given in closed form.
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
  /** The parameter's name. */
  readonly name: DistributionalParameter
  /** TeX symbol, e.g. `\sigma`. */
  readonly tex: string
  /** What the parameter controls: the location, the scale or the shape. */
  readonly role: 'location' | 'scale' | 'shape'
  /** The links it takes; the first is the default. */
  readonly links: readonly LinkName[]
  /** The open range of valid values. */
  readonly range: 'real' | 'positive'
}

/**
 * A distributional-regression family (see the file comment). $\thetavec$ is one value per parameter, in `parameters`
 * order, and $k$ indexes that order.
 */
export interface DistributionalFamily {
  /** The family's name, as `distributionalFamily` takes it. */
  readonly name: DistributionalFamilyName
  /** gamlss.dist's abbreviation: NO, TF, BCCG, GA, PO. */
  readonly abbreviation: string
  /** A readable name, e.g. "Student t". */
  readonly label: string
  /** The parameters, in gamlss's order: $\mu$, then $\sigma$ and $\nu$ where the family has them. */
  readonly parameters: readonly DistributionalParameterInfo[]
  /** The values a response can take. */
  readonly support: 'real' | 'positive' | 'non-negative-integers'
  /** $\log p(y \mid \thetavec)$ ($-\infty$ outside the support). */
  logPdf(y: number, theta: readonly number[]): number
  /** $\partial\ell/\partial\theta_k$ at $y$. */
  score(k: number, y: number, theta: readonly number[]): number
  /** $\expect[\partial^2\ell/\partial\theta_k^2]$ (negative): minus the expected information of $\theta_k$. */
  expectedSecond(k: number, theta: readonly number[]): number
  /** The cdf $F(y \mid \thetavec)$. */
  cdf(y: number, theta: readonly number[]): number
  /** The quantile function: the $y$ with $F(y \mid \thetavec) = p$ (the smallest with $F \ge p$ for counts). */
  quantile(p: number, theta: readonly number[]): number
  /** The mean (NaN where it does not exist). */
  mean(theta: readonly number[]): number
  /** The variance (NaN or $\infty$ where it does not exist). */
  variance(theta: readonly number[]): number
  /** Whether $\thetavec$ lies in the parameter space. */
  valid(theta: readonly number[]): boolean
  /** Starting values of each parameter at each observation (gamlss.dist's `*.initial`). */
  initial(y: ArrayLike<number>): Float64Array[]
}

/** $\log\sqrt{2\pi}$. */
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)

/**
 * The sample mean and standard deviation (with the $n - 1$ divisor, or 1 for a single value).
 *
 * @param y The responses.
 * @returns `mean` and `sd`.
 */
function moments(y: ArrayLike<number>) {
  let m = 0
  for (let i = 0; i < y.length; i++) m += y[i]
  m /= y.length
  let v = 0
  for (let i = 0; i < y.length; i++) v += (y[i] - m) ** 2
  return { mean: m, sd: Math.sqrt(v / Math.max(1, y.length - 1)) }
}
/**
 * A constant array.
 *
 * @param n The length.
 * @param v The value of every entry.
 * @returns `n` copies of `v`.
 */
const fill = (n: number, v: number) => new Float64Array(n).fill(v)
/**
 * Each response moved halfway to the mean, gamlss.dist's starting value for $\mu$.
 *
 * @param y The responses.
 * @param m Their mean.
 * @returns $(y_i + m)/2$ for each response.
 */
const halfway = (y: ArrayLike<number>, m: number) => Float64Array.from(y, (v) => (v + m) / 2)

/**
 * Mean and variance by the midpoint rule over $m$ quantiles (for families without closed forms): the moments of the
 * quantiles at $(i + 1/2)/m$.
 *
 * @param q The quantile function.
 * @param m The number of quantiles.
 * @returns `mean` and `variance` (rounding below 0 clipped).
 */
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

/**
 * A parameter's description, as `DistributionalParameterInfo`.
 *
 * @param name The parameter's name.
 * @param tex Its TeX symbol.
 * @param role What it controls.
 * @param links The links it takes, the default first.
 * @param range Its range of valid values.
 * @returns The description.
 */
const param = (
  name: DistributionalParameter,
  tex: string,
  role: DistributionalParameterInfo['role'],
  links: LinkName[],
  range: DistributionalParameterInfo['range'],
): DistributionalParameterInfo => ({ name, tex, role, links, range })

/**
 * $\operatorname{NO}(\mu, \sigma)$: the normal distribution with mean $\mu$ and standard deviation $\sigma$. The
 * expected information is $1/\sigma^2$ for $\mu$ and $2/\sigma^2$ for $\sigma$. Starting values: each response
 * halfway to the mean for $\mu$, the sample standard deviation for $\sigma$.
 *
 * @returns The family.
 *
 * @example The log-density, scores and a quantile of the standard normal
 * const f = normalDistributional()
 * print('log pdf at 0 = -log(2 pi)/2:', f.logPdf(0, [0, 1]))
 * print('scores at y = 1:', f.score(0, 1, [0, 1]), f.score(1, 1, [0, 1]))
 * print('quantile(0.975):', f.quantile(0.975, [0, 1]))
 */
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

/**
 * $\operatorname{TF}(\mu, \sigma, \nu)$: $y = \mu + \sigma T$ with $T$ Student t on $\nu$ degrees of freedom.
 * The expected information is that of Lange, Little and Taylor (1989), JASA 84(408), §2. The mean exists for
 * $\nu > 1$ and the variance $\sigma^2\nu/(\nu - 2)$ for $\nu > 2$ ($\infty$ for $1 < \nu \le 2$). Starting
 * values: $\mu$ as for the normal, the sample standard deviation for $\sigma$ and $\nu = 10$.
 *
 * @returns The family.
 *
 * @example Heavier tails than the normal
 * const f = studentTDistributional()
 * print('quantile(0.975), nu = 3:', f.quantile(0.975, [0, 1, 3]))
 * print('variance, nu = 3:', f.variance([0, 1, 3]), 'nu = 2:', f.variance([0, 1, 2]))
 * print('mean, nu = 1:', f.mean([0, 1, 1]))
 */
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

/** $\lvert \nu \rvert$ below this is treated as $\nu = 0$ (the log transform) by BCCG. */
const NU_ZERO = 1e-7

/**
 * $\operatorname{BCCG}(\mu, \sigma, \nu)$: the Box–Cox Cole–Green distribution (the LMS method of Cole and Green,
 * 1992), truncated to $y > 0$. The scores account for the truncation; the expected second derivatives are
 * gamlss.dist's approximations, which ignore it. The mean and variance have no closed form and come from 400
 * quantiles (midpoint rule). Starting values: $\mu$ as for the normal, $\sigma = 0.1$ and $\nu = 0.5$.
 *
 * @returns The family.
 *
 * @example At $\nu = 0$ it is log-normal, with median $\mu$ and mean $\mu e^{\sigma^2/2}$
 * const f = boxCoxColeGreenDistributional()
 * print('median:', f.quantile(0.5, [10, 0.2, 0]))
 * print('mean (from quantiles):', f.mean([10, 0.2, 0]), 'exact:', 10 * Math.exp(0.02))
 *
 * @example A power below 1 skews to the right
 * const f = boxCoxColeGreenDistributional()
 * const q = (nu) => [0.05, 0.5, 0.95].map((p) => f.quantile(p, [10, 0.3, nu]))
 * print('5%, 50%, 95%, nu = 1:', q(1))
 * print('5%, 50%, 95%, nu = -1:', q(-1))
 */
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

/**
 * $\operatorname{GA}(\mu, \sigma)$: the gamma distribution with mean $\mu$ and coefficient of variation $\sigma$
 * (shape $1/\sigma^2$, scale $\mu\sigma^2$), so the variance is $(\sigma\mu)^2$. Starting values: $\mu$ as for
 * the normal and $\sigma = 1$.
 *
 * @returns The family.
 *
 * @example At $\sigma = 1$ it is exponential with mean $\mu$
 * const f = gammaDistributional()
 * print('cdf(2), mu = 2, sigma = 1: 1 - 1/e =', f.cdf(2, [2, 1]))
 * print('variance, mu = 2, sigma = 0.5:', f.variance([2, 0.5]))
 */
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

/**
 * $\operatorname{PO}(\mu)$: the Poisson distribution with mean $\mu$. The log-density is $-\infty$ off the
 * non-negative integers; the quantile is the smallest $k$ with $F(k) \ge p$, found by walking from the normal
 * approximation. Starting value: each response halfway to the mean.
 *
 * @returns The family.
 *
 * @example The log-density, cdf and median at $\mu = 2$
 * const f = poissonDistributional()
 * print('log p(2) = log 2 - 2:', f.logPdf(2, [2]), 'log p(1.5):', f.logPdf(1.5, [2]))
 * print('F(2):', f.cdf(2, [2]), 'median:', f.quantile(0.5, [2]))
 */
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

/** The family factories by name. */
const FAMILIES: Record<DistributionalFamilyName, () => DistributionalFamily> = {
  normal: normalDistributional,
  'student-t': studentTDistributional,
  'box-cox-cole-green': boxCoxColeGreenDistributional,
  gamma: gammaDistributional,
  poisson: poissonDistributional,
}

/**
 * The distributional family by name (see the file comment). Throws `DomainError` for an unknown name.
 *
 * @param name The family's name.
 * @returns The family.
 *
 * @example The gamma family's parameters and their default links
 * const f = distributionalFamily('gamma')
 * print(f.abbreviation, f.parameters.map((p) => `${p.name}: ${p.links[0]}`))
 */
export function distributionalFamily(name: DistributionalFamilyName): DistributionalFamily {
  const f = FAMILIES[name]
  if (!f) throw new DomainError('distributionalFamily', `distributionalFamily: unknown family "${name}"`)
  return f()
}

/**
 * The links of a family's parameters: the given names, or each parameter's default. A link a parameter does not take
 * throws `DomainError`.
 *
 * @param family The family.
 * @param names A link name for any of the parameters; the others take their default (the first of their `links`).
 * @returns One link per parameter, in `parameters` order.
 *
 * @example Defaults, an override, and a link the parameter does not take
 * const f = distributionalFamily('box-cox-cole-green')
 * print('defaults:', distributionalLinks(f).map((g) => g.name))
 * print('log mu:', distributionalLinks(f, { mu: 'log' }).map((g) => g.name))
 * try {
 *   distributionalLinks(f, { nu: 'log' })
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
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
 * The normalised quantile residual $r = \Phi^{-1}(F(y \mid \thetavec))$ (Dunn and Smyth, 1996); for a discrete
 * family the mid-point $\Phi^{-1}((F(y - 1) + F(y))/2)$, not the randomised residual. Under the true model the
 * residuals are standard normal (continuous families). The probability is clamped to $[10^{-15}, 1 - 10^{-15}]$, so
 * $r$ stays finite.
 *
 * @param family The fitted family.
 * @param y The observed response.
 * @param theta The fitted parameters at that observation, in `parameters` order.
 * @returns The residual $r$.
 *
 * @example A response at the 97.5% point of its law, and a count
 * print('normal:', quantileResidual(normalDistributional(), 1.96, [0, 1]))
 * print('gamma, y = mu:', quantileResidual(gammaDistributional(), 2, [2, 0.5]))
 * print('Poisson, y = 2, mu = 2:', quantileResidual(poissonDistributional(), 2, [2]))
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
  /** The standard normal quantiles $\Phi^{-1}((i - 1/2)/n)$, $i = 1, \dots, n$, of the sorted residuals. */
  x: Float64Array
  /** Each sorted residual minus its normal quantile ($n$ values). */
  y: Float64Array
  /**
   * The lower edge of the pointwise approximate 95% band, $-1.96 \sqrt{p(1 - p)/n}/\varphi(x)$ at `x`, with
   * $p = (i - 1/2)/n$ and $\varphi$ the standard normal density.
   */
  lower: Float64Array
  /** The upper edge of the band, $+1.96 \sqrt{p(1 - p)/n}/\varphi(x)$. */
  upper: Float64Array
}

/**
 * The worm plot (van Buuren and Fredriks, 2001): a normal Q–Q plot of residuals detrended by subtracting the identity
 * line, so departures from normality show as shapes about zero (a shift: the mean; a slope: the variance; a U or
 * inverted U: skewness; an S: kurtosis), with a pointwise 95% band.
 *
 * @param residuals The residuals, usually quantile residuals; not modified (a sorted copy is used).
 * @returns The plot's points and band, one entry per residual in sorted order.
 *
 * @example Five residuals: the band is widest at the ends
 * const w = wormPlot([1.5, -1.2, 0.1, -0.3, 0.4])
 * print('x:', w.x)
 * print('y:', w.y)
 * print('upper:', w.upper)
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
