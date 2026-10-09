/**
 * Link functions and exponential-dispersion families for generalised linear models, and the likelihood of a response
 * given a linear predictor $\eta$ through a link.
 *
 * The families and links are those of Nelder and Wedderburn (1972) and McCullagh and Nelder (1989), "Generalized
 * Linear Models", 2nd ed., ch. 2 and Table 2.1, with R's `family` objects as the reference for conventions: a family
 * has $\var(y) = \phi V(\mu)/w$ for dispersion $\phi$ and prior weight $w$, a link $g$ gives $\eta = g(\mu)$, and
 * binomial responses are proportions with the number of trials as the prior weight. Links, variance functions, unit
 * deviances and pointwise log-likelihoods are compositions of `aifn-compute/foundation/tensor` and
 * `aifn-compute/numerics/special` primitives, so they accept numbers, tensors and traced values and are
 * differentiable to any order. Starting means, validity checks and predictive distributions work on raw tensors.
 */

import { Bernoulli, Binomial, Gamma, NegativeBinomial, Normal, Poisson } from 'aifn-compute/probability/distributions'
import {
  logChoose,
  logGamma,
  logit,
  normalCdf,
  normalPdf,
  normalQuantile,
  sigmoid,
  softplus,
  xlog1py,
  xlogy,
} from 'aifn-compute/numerics/special'
import { child, normal, uniform } from 'aifn-compute/foundation/random'
import {
  add,
  div,
  exp,
  expm1,
  fromData,
  full,
  log,
  mul,
  neg,
  pow,
  sqrt,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Distribution, SampleOptions, Scalar, Stream } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── Links ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The name of a link function. */
export type LinkName = 'identity' | 'log' | 'logit' | 'probit' | 'cloglog' | 'inverse' | 'inverse-squared' | 'sqrt'

/** A link $g$: the linear predictor is $\eta = g(\mu)$. Every function is a composition of primitives. */
export interface Link {
  /** The link's name, as `link` takes it. */
  readonly name: LinkName
  /** $\eta = g(\mu)$. */
  link(mu: Value): Value
  /** $\mu = g^{-1}(\eta)$, the mean function. */
  inverse(eta: Value): Value
  /** $d\mu/d\eta$ at $\eta$. */
  derivative(eta: Value): Value
  /**
   * True when $g^{-1}$ maps every finite $\eta$ into the open mean space (log, logit, probit, cloglog): any finite
   * $\eta$ is a valid linear predictor, even where $\mu$ rounds to the edge of the space ($\sigma(40) = 1$ in
   * float64). False when $\eta$ itself must stay inside a range (identity, inverse, sqrt, inverse squared).
   */
  readonly total: boolean
}

/**
 * 1 in the shape of `v` (a number, tensor or traced value), as $0 \cdot v + 1$ so that it stays traced.
 *
 * @param v The value whose shape is taken; only its shape matters.
 * @returns Ones with the shape of `v`.
 */
function onesLike(v: Value): Value {
  return add(mul(0, v), 1)
}

/** The links by name: the functions of each, and whether it is total. */
const LINKS: Record<LinkName, Omit<Link, 'name'>> = {
  identity: { link: (m) => m, inverse: (e) => e, derivative: onesLike, total: false },
  log: { link: log, inverse: exp, derivative: exp, total: true },
  logit: { link: logit, inverse: sigmoid, derivative: (e) => mul(sigmoid(e), sigmoid(neg(e))), total: true },
  probit: { link: normalQuantile, inverse: normalCdf, derivative: normalPdf, total: true },
  // μ = 1 − exp(−e^η), the extreme-value (Gumbel minimum) cdf.
  cloglog: {
    link: (m) => log(neg(log(sub(1, m)))),
    inverse: (e) => sub(1, exp(neg(exp(e)))),
    derivative: (e) => exp(sub(e, exp(e))),
    total: true,
  },
  inverse: {
    link: (m) => div(1, m),
    inverse: (e) => div(1, e),
    derivative: (e) => neg(div(1, square(e))),
    total: false,
  },
  'inverse-squared': {
    link: (m) => div(1, square(m)),
    inverse: (e) => div(1, sqrt(e)),
    derivative: (e) => mul(-0.5, pow(e, -1.5)),
    total: false,
  },
  sqrt: { link: sqrt, inverse: square, derivative: (e) => mul(2, e), total: false },
}

/**
 * The link function by name (McCullagh and Nelder, 1989, §2.2.2): identity, log, logit, probit, cloglog
 * ($\mu = 1 - \exp(-e^\eta)$, the extreme-value cdf), inverse ($\eta = 1/\mu$), inverse squared
 * ($\eta = 1/\mu^2$) or sqrt. Throws `DomainError` for an unknown name.
 *
 * @param name The link's name.
 * @returns The link, with $g$, $g^{-1}$ and $d\mu/d\eta$.
 *
 * @example The logit at $\mu = 3/4$ and its inverse at 0
 * const g = link('logit')
 * print('g(0.75) = log 3:', g.link(0.75))
 * print('inverse(0):', g.inverse(0))
 * print('dmu/deta at 0:', g.derivative(0))
 *
 * @example Each function works elementwise on tensors
 * print('exp:', link('log').inverse(tensor([0, 1, 2])))
 */
export function link(name: LinkName): Link {
  const l = LINKS[name]
  if (!l) throw new DomainError('link', `link: unknown link "${name}"`)
  return { name, ...l }
}

// ── Families ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** The name of a family. */
export type FamilyName = 'gaussian' | 'binomial' | 'poisson' | 'gamma' | 'inverse-gaussian' | 'negative-binomial'

/**
 * An exponential-dispersion family: $\var(y) = \phi V(\mu)/w$ for prior weight $w$. The deviance is
 * $\sum_i w_i d(y_i, \mu_i)$ with the unit deviance $d$. Binomial responses are proportions with the number of trials
 * as the prior weight (as R's `glm`).
 */
export interface Family {
  /** The family's name, as `family` takes it. */
  readonly name: FamilyName
  /** Parameters fixed when the family was made (e.g. the negative binomial's $\theta$). */
  readonly params: Readonly<Record<string, Scalar>>
  /** The canonical link: the one that makes $\eta$ the natural parameter. */
  readonly canonicalLink: LinkName
  /** The default link (the canonical one except for the negative binomial, whose conventional link is log). */
  readonly defaultLink: LinkName
  /**
   * The links the family is used with (McCullagh and Nelder, 1989, Table 2.1; R's `family` objects): each maps the
   * family's mean space onto a range a linear predictor can reach. `checkLink` rejects any other.
   */
  readonly links: readonly LinkName[]
  /** $\phi$ when it is known (1 for binomial, Poisson and negative binomial); null when it is estimated. */
  readonly dispersion: Scalar | null
  /** The variance function $V(\mu)$. */
  variance(mu: Value): Value
  /**
   * $d(y, \mu)$, the unit deviance (twice the log-likelihood ratio of the saturated model for one observation).
   */
  unitDeviance(y: Value, mu: Value): Value
  /**
   * The pointwise log-likelihood $\log p(y_i \mid \mu_i, \phi, w_i)$, elementwise over broadcast $y$, $\mu$ and
   * $w$, differentiable in $\mu$ (and $\phi$). Prior weights default to 1; binomial weights are trial counts.
   */
  logProb(y: Value, mu: Value, dispersion: Value, weights?: Value): Value
  /**
   * $\sum_i$ `logProb` as a number ($y$, $\mu$ and the weights untraced), for reports and information criteria.
   */
  logLikelihood(y: Tensor, mu: Tensor, dispersion: Scalar, weights: Tensor): Scalar
  /** True when every $\mu$ lies in the family's mean space. */
  validMean(mu: Tensor): boolean
  /** A starting mean for IRLS from the responses and prior weights (as R's `mustart`). */
  initialMean(y: Tensor, weights: Tensor): Tensor
  /** The distribution of a new response given its mean (and dispersion; prior weights as trials or precision). */
  predictive(mu: Tensor, dispersion: Scalar, weights?: Tensor): Distribution
}

/**
 * Whether every entry of a raw tensor passes a test.
 *
 * @param mu The means to check.
 * @param ok The test of one entry.
 * @returns True when `ok` holds for every entry.
 */
const all = (mu: Tensor, ok: (m: number) => boolean) => toFlat(mu).every(ok)
/**
 * An elementwise map of one or two same-shape raw tensors (starting means, predictive parameters).
 *
 * @param a The first tensor; the result has its shape.
 * @param b The second tensor, read at the same flat index (pass `a` again for a map of one tensor).
 * @param f The function of an entry of `a` and the matching entry of `b`.
 * @returns A float64 tensor of the shape of `a`.
 */
const zip = (a: Tensor, b: Tensor, f: (x: number, y: number) => number) => {
  const x = toFlat(a)
  const y = toFlat(b)
  return fromData(
    Float64Array.from(x, (v, i) => f(v, y[i])),
    a.shape,
  )
}
/**
 * A per-observation parameter computed from the prior weights, such as $\sqrt{\phi/w}$ or $w/\phi$, as a tensor.
 *
 * @param mu The means; give the shape when there are no weights.
 * @param w The prior weights, or undefined for weights of 1.
 * @param f The parameter as a function of one weight.
 * @returns $f(w_i)$ for each weight (shaped like `w`), or $f(1)$ in the shape of `mu`.
 */
const perObservation = (mu: Tensor, w: Tensor | undefined, f: (w: number) => number) =>
  w ? zip(w, w, f) : full(mu.shape, f(1))

/**
 * $\sum_i$ `logProb` as a number: the shared definition of `logLikelihood`.
 *
 * @param logProb The family's pointwise log-likelihood.
 * @returns The function of $(y, \mu, \phi, w)$ that sums it and unwraps the total to a number.
 */
function summed(logProb: Family['logProb']): Family['logLikelihood'] {
  return (y, mu, phi, w) => {
    const v = unwrap(sum(logProb(y, mu, phi, w)))
    return typeof v === 'number' ? v : toFlat(v)[0]
  }
}

/**
 * The Gaussian family: $V(\mu) = 1$, $d = (y - \mu)^2$, canonical link identity; links identity, log and inverse.
 * The dispersion $\phi = \sigma^2$ is estimated. Its log-likelihood is
 * $-\tfrac{w}{2}(\log(2\pi\phi/w) + (y - \mu)^2/\phi)$, and its predictive law is
 * $\Gauss(\mu, \phi/w)$.
 *
 * @returns The family.
 *
 * @example Variance, deviance and log-likelihood at a point
 * const f = gaussianFamily()
 * print('V(3):', f.variance(3), 'd(1, 3):', f.unitDeviance(1, 3))
 * print('log p(y = 1 | mu = 1, phi = 1) = -log(2 pi)/2:', f.logProb(1, 1, 1))
 */
export function gaussianFamily(): Family {
  // w · (−½)(log(2πφ/w) + (y − μ)²/φ): the weights count replicated observations, as R's `gaussian()$aic`.
  const logProb: Family['logProb'] = (y, mu, phi, w = 1) =>
    mul(w, mul(-0.5, add(log(div(mul(2 * Math.PI, phi), w)), div(square(sub(y, mu)), phi))))
  return {
    name: 'gaussian',
    params: {},
    canonicalLink: 'identity',
    defaultLink: 'identity',
    links: ['identity', 'log', 'inverse'],
    dispersion: null,
    variance: (mu) => onesLike(mu),
    unitDeviance: (y, mu) => square(sub(y, mu)),
    logProb,
    logLikelihood: summed(logProb),
    validMean: (mu) => all(mu, Number.isFinite),
    initialMean: (y) => y,
    predictive: (mu, phi, w) =>
      Normal(
        mu,
        perObservation(mu, w, (v) => Math.sqrt(phi / v)),
      ),
  }
}

/**
 * The binomial family for proportions $y \in [0, 1]$ with $w_i$ trials (Bernoulli when every $w = 1$):
 * $V(\mu) = \mu(1 - \mu)$, $d = 2[y \log(y/\mu) + (1 - y) \log((1 - y)/(1 - \mu))]$, canonical link logit;
 * links logit, probit, cloglog and log. The dispersion is 1. The log-likelihood counts $k = wy$ successes,
 * $\log\binom{w}{k} + k \log \mu + (w - k) \log(1 - \mu)$, and the starting mean is $(wy + 1/2)/(w + 1)$.
 *
 * @returns The family.
 *
 * @example One success in two trials, and a deviance
 * const f = binomialFamily()
 * print('log p(y = 1/2 | mu = 1/2, w = 2) = log(1/2):', f.logProb(0.5, 0.5, 1, 2))
 * print('d(1, 0.8) = 2 log(1.25):', f.unitDeviance(1, 0.8))
 * print('V(0.5):', f.variance(0.5))
 */
export function binomialFamily(): Family {
  // log C(w, k) + k log μ + (w − k) log(1 − μ) with k = w·y successes.
  const logProb: Family['logProb'] = (y, mu, _phi, w = 1) => {
    const k = mul(w, y)
    return add(logChoose(w, k), add(xlogy(k, mu), xlog1py(sub(w, k), neg(mu))))
  }
  return {
    name: 'binomial',
    params: {},
    canonicalLink: 'logit',
    defaultLink: 'logit',
    links: ['logit', 'probit', 'cloglog', 'log'],
    dispersion: 1,
    variance: (mu) => mul(mu, sub(1, mu)),
    unitDeviance: (y, mu) =>
      mul(2, add(sub(xlogy(y, y), xlogy(y, mu)), sub(xlogy(sub(1, y), sub(1, y)), xlogy(sub(1, y), sub(1, mu))))),
    logProb,
    logLikelihood: summed(logProb),
    validMean: (mu) => all(mu, (m) => m > 0 && m < 1),
    initialMean: (y, w) => zip(y, w, (a, b) => (b * a + 0.5) / (b + 1)),
    predictive: (mu, _phi, w) => (w && !toFlat(w).every((v) => v === 1) ? Binomial(w, mu) : Bernoulli(mu)),
  }
}

/**
 * The Poisson family: $V(\mu) = \mu$, $d = 2[y \log(y/\mu) - (y - \mu)]$, canonical link log; links log, identity
 * and sqrt. The dispersion is 1, prior weights multiply the log-likelihood, and the starting mean is $y + 0.1$.
 *
 * @returns The family.
 *
 * @example The log-likelihood and the deviance of a zero count
 * const f = poissonFamily()
 * print('log p(y = 2 | mu = 2) = log 2 - 2:', f.logProb(2, 2, 1))
 * print('d(0, 2) = 2 mu:', f.unitDeviance(0, 2))
 */
export function poissonFamily(): Family {
  const logProb: Family['logProb'] = (y, mu, _phi, w = 1) => mul(w, sub(sub(xlogy(y, mu), mu), logGamma(add(y, 1))))
  return {
    name: 'poisson',
    params: {},
    canonicalLink: 'log',
    defaultLink: 'log',
    links: ['log', 'identity', 'sqrt'],
    dispersion: 1,
    variance: (mu) => mu,
    unitDeviance: (y, mu) => mul(2, sub(sub(xlogy(y, y), xlogy(y, mu)), sub(y, mu))),
    logProb,
    logLikelihood: summed(logProb),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => zip(y, y, (a) => a + 0.1),
    predictive: (mu) => Poisson(mu),
  }
}

/**
 * The gamma family: $V(\mu) = \mu^2$, $d = 2[-\log(y/\mu) + (y - \mu)/\mu]$, canonical link inverse; links
 * inverse, log and identity. The response is $\GammaD(\alpha, \alpha/\mu)$ (shape and rate) with shape
 * $\alpha = w/\phi$, so $\phi$ is the squared coefficient of variation and is estimated.
 *
 * @returns The family.
 *
 * @example With $\phi = 1$ the response is exponential
 * const f = gammaFamily()
 * print('log p(y = 1 | mu = 1, phi = 1) = -1:', f.logProb(1, 1, 1))
 * print('V(2):', f.variance(2))
 * print('predictive mean, variance:', f.predictive(tensor([2]), 0.5).mean(), f.predictive(tensor([2]), 0.5).variance())
 */
export function gammaFamily(): Family {
  // With shape α = w/φ and rate α/μ: α log(αy/μ) − αy/μ − log y − log Γ(α).
  const logProb: Family['logProb'] = (y, mu, phi, w = 1) => {
    const shape = div(w, phi)
    const ratio = div(mul(shape, y), mu)
    return sub(sub(sub(mul(shape, log(ratio)), ratio), log(y)), logGamma(shape))
  }
  return {
    name: 'gamma',
    params: {},
    canonicalLink: 'inverse',
    defaultLink: 'inverse',
    links: ['inverse', 'log', 'identity'],
    dispersion: null,
    variance: (mu) => square(mu),
    unitDeviance: (y, mu) => mul(2, add(neg(log(div(y, mu))), div(sub(y, mu), mu))),
    logProb,
    logLikelihood: summed(logProb),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => y,
    predictive: (mu, phi, w) => {
      const shape = perObservation(mu, w, (v) => v / phi)
      return Gamma(
        shape,
        zip(shape, mu, (s, m) => s / m),
      )
    },
  }
}

/**
 * The inverse Gaussian family: $V(\mu) = \mu^3$, $d = (y - \mu)^2/(\mu^2 y)$, canonical link $1/\mu^2$ (inverse
 * squared); links inverse squared, inverse, log and identity. The response is inverse Gaussian with mean $\mu$ and
 * shape $w/\phi$; the dispersion is estimated.
 *
 * @returns The family.
 *
 * @example Variance and deviance, and the predictive law's moments
 * const f = inverseGaussianFamily()
 * print('V(2):', f.variance(2), 'd(1, 2):', f.unitDeviance(1, 2))
 * const law = f.predictive(tensor([2]), 0.5)
 * print('predictive mean:', law.mean(), 'variance = phi mu^3:', law.variance())
 */
export function inverseGaussianFamily(): Family {
  // With p = φ/w: −½(log(2πp y³) + (y − μ)²/(p μ² y)).
  const logProb: Family['logProb'] = (y, mu, phi, w = 1) => {
    const p = div(phi, w)
    return mul(-0.5, add(log(mul(mul(2 * Math.PI, p), pow(y, 3))), div(square(sub(y, mu)), mul(mul(p, square(mu)), y))))
  }
  return {
    name: 'inverse-gaussian',
    params: {},
    canonicalLink: 'inverse-squared',
    defaultLink: 'inverse-squared',
    links: ['inverse-squared', 'inverse', 'log', 'identity'],
    dispersion: null,
    variance: (mu) => pow(mu, 3),
    unitDeviance: (y, mu) => div(square(sub(y, mu)), mul(square(mu), y)),
    logProb,
    logLikelihood: summed(logProb),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => y,
    predictive: (mu, phi, w) =>
      inverseGaussianPredictive(
        mu,
        perObservation(mu, w, (v) => v / phi),
      ),
  }
}

/**
 * The negative binomial family with fixed shape $\theta > 0$ (NB2): $V(\mu) = \mu + \mu^2/\theta$,
 * $d = 2[y \log(y/\mu) - (y + \theta) \log((y + \theta)/(\mu + \theta))]$, default link log (Hilbe, 2011,
 * "Negative Binomial Regression"); links log, identity and sqrt. Its canonical link is
 * $\log(\mu/(\mu + \theta))$, which is not among the named links, so `canonicalLink` reports log. The predictive
 * counts failures before $\theta$ successes with $p = \theta/(\theta + \mu)$, so its mean is $\mu$. Throws
 * `DomainError` unless $\theta > 0$.
 *
 * @param theta The shape $\theta$ (the size): smaller values mean more overdispersion; $\theta \to \infty$ is the
 *   Poisson.
 * @returns The family, with $\theta$ in `params`.
 *
 * @example Overdispersion, and the probability of a zero count
 * const f = negativeBinomialFamily(2)
 * print('V(4) = 4 + 16/2:', f.variance(4))
 * print('log p(y = 0 | mu = 4) = 2 log(1/3):', f.logProb(0, 4, 1))
 * print('the predictive agrees:', f.predictive(tensor([4]), 1).logProb(0))
 */
export function negativeBinomialFamily(theta: Scalar): Family {
  if (!(theta > 0)) throw new DomainError('negativeBinomialFamily', 'negativeBinomialFamily: θ must be positive')
  // w · [log Γ(y + θ) − log Γ(θ) − log Γ(y + 1) + θ log(θ/(θ + μ)) + y log(μ/(θ + μ))].
  const logProb: Family['logProb'] = (y, mu, _phi, w = 1) => {
    const total = add(mu, theta)
    const normaliser = sub(sub(logGamma(add(y, theta)), logGamma(theta)), logGamma(add(y, 1)))
    return mul(w, add(add(normaliser, mul(theta, log(div(theta, total)))), xlogy(y, div(mu, total))))
  }
  return {
    name: 'negative-binomial',
    params: { theta },
    // log(μ/(μ + θ)) is not among the named links; the conventional log link is the default.
    canonicalLink: 'log',
    defaultLink: 'log',
    links: ['log', 'identity', 'sqrt'],
    dispersion: 1,
    variance: (mu) => add(mu, div(square(mu), theta)),
    unitDeviance: (y, mu) =>
      mul(
        2,
        sub(
          sub(xlogy(y, y), xlogy(y, mu)),
          sub(xlogy(add(y, theta), add(y, theta)), xlogy(add(y, theta), add(mu, theta))),
        ),
      ),
    logProb,
    logLikelihood: summed(logProb),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => zip(y, y, (a) => a + 0.1),
    predictive: (mu) =>
      NegativeBinomial(
        theta,
        zip(mu, mu, (m) => theta / (theta + m)),
      ),
  }
}

/**
 * A family by name. Throws `DomainError` for an unknown name.
 *
 * @param name The family's name.
 * @param params `theta`, the negative binomial's shape $\theta$ (default 1); ignored by the other families.
 * @returns The family.
 *
 * @example Families by name
 * print('gamma canonical link:', family('gamma').canonicalLink)
 * print('negative binomial params:', family('negative-binomial', { theta: 5 }).params)
 */
export function family(name: FamilyName, params: { theta?: Scalar } = {}): Family {
  switch (name) {
    case 'gaussian':
      return gaussianFamily()
    case 'binomial':
      return binomialFamily()
    case 'poisson':
      return poissonFamily()
    case 'gamma':
      return gammaFamily()
    case 'inverse-gaussian':
      return inverseGaussianFamily()
    case 'negative-binomial':
      return negativeBinomialFamily(params.theta ?? 1)
  }
  throw new DomainError('family', `family: unknown family "${name as string}"`)
}

/**
 * The link by name or as given, checked against the family's `links`: a link outside them maps the mean space onto a
 * range the linear predictor cannot be held to (a logit for counts, an identity for probabilities), so a fit would
 * leave the mean space or diverge. Throws a `DomainError` naming the valid links.
 *
 * @param fam The family whose `links` are allowed.
 * @param chosen The link: a name, or a `Link` (checked by its name). Left out, the family's default link.
 * @param where The caller's name, for error messages.
 * @returns The link.
 *
 * @example The Poisson family takes a log link but not a logit
 * print('ok:', checkLink(poissonFamily(), 'log').name)
 * try {
 *   checkLink(poissonFamily(), 'logit')
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
export function checkLink(fam: Family, chosen: LinkName | Link = fam.defaultLink, where = 'likelihood'): Link {
  const g = typeof chosen === 'object' ? chosen : link(chosen)
  if (!fam.links.includes(g.name))
    throw new DomainError(
      where,
      `${where}: the ${fam.name} family does not take the ${g.name} link; use one of ${fam.links.join(', ')}`,
    )
  return g
}

// ── Likelihood: family and link ──────────────────────────────────────────────────────────────────────────────────

/**
 * Options of a likelihood's functions of $(y, \eta)$: `dispersion`, the dispersion $\phi$ (default the family's,
 * else 1), and `weights`, the prior weights $w$ (default 1; trial counts for the binomial).
 */
export type LikelihoodOptions = { dispersion?: Value; weights?: Value }

/**
 * The likelihood of a response $y$ given the linear predictor $\eta$: a family with a link, $\mu = g^{-1}(\eta)$.
 * Every function is a composition of primitives, elementwise over broadcast arguments, so it differentiates in
 * $\eta$ (and $\phi$).
 */
export interface Likelihood {
  /** The family. */
  readonly family: Family
  /** The link, checked against the family's links. */
  readonly link: Link
  /** $\mu = g^{-1}(\eta)$. */
  mean(eta: Value): Value
  /** $\var(y) = \phi V(\mu)/w$ at $\eta$. */
  variance(eta: Value, options?: LikelihoodOptions): Value
  /** The pointwise log-likelihood $\log p(y \mid \mu = g^{-1}(\eta), \phi, w)$. */
  logLik(y: Value, eta: Value, options?: LikelihoodOptions): Value
  /**
   * The score $\partial\ell/\partial\eta = w (y - \mu) (g^{-1})'(\eta) / (\phi V(\mu))$ in closed form
   * (McCullagh and Nelder, 1989, eq. 2.13); equal to differentiating `logLik` in $\eta$ for the families whose
   * log-likelihood depends on $\mu$ only through the exponential-family kernel (all six here).
   */
  score(y: Value, eta: Value, options?: LikelihoodOptions): Value
  /**
   * The unit deviance $d(y, g^{-1}(\eta))$, from $\eta$ directly where $\mu$ would lose precision: for the binomial
   * with the logit link,
   * $2[y \log y + (1 - y) \log(1 - y) + y \operatorname{softplus}(-\eta) + (1 - y) \operatorname{softplus}(\eta)]$;
   * with the cloglog link, $\log(1 - \mu) = -e^\eta$ and
   * $\log \mu = \log(-\operatorname{expm1}(-e^\eta))$. Finite for every finite $\eta$, where the $\mu$ form
   * overflows once $\mu$ rounds to 0 or 1.
   */
  unitDeviance(y: Value, eta: Value): Value
}

/**
 * A likelihood from a family and a link (default: the family's default link), e.g. logistic regression's
 * `likelihood(binomialFamily())` or a log-linear gamma model's `likelihood(gammaFamily(), 'log')`. Throws
 * `DomainError` when the family does not take the link (see `checkLink`).
 *
 * @param fam The family.
 * @param linkName The link's name (default: the family's default link).
 * @returns The likelihood, with the mean, variance, log-likelihood, score and unit deviance as functions of $\eta$.
 *
 * @example Logistic regression at $\eta = 0$: the score is the gradient of the log-likelihood
 * const lik = likelihood(binomialFamily())
 * print('mean:', lik.mean(0), 'logLik(y = 1):', lik.logLik(1, 0))
 * print('score:', lik.score(1, 0), 'gradient:', grad((eta) => lik.logLik(1, eta))(0))
 *
 * @example The deviance from $\eta$ stays finite where $\mu$ rounds to 1
 * const lik = likelihood(binomialFamily())
 * print('from eta:', lik.unitDeviance(0, 40))
 * print('from mu:', lik.family.unitDeviance(0, lik.mean(40)))
 */
export function likelihood(fam: Family, linkName: LinkName = fam.defaultLink): Likelihood {
  const g = checkLink(fam, linkName)
  const phiOf = (o: LikelihoodOptions) => o.dispersion ?? fam.dispersion ?? 1
  return {
    family: fam,
    link: g,
    mean: (eta) => g.inverse(eta),
    variance: (eta, o = {}) => div(mul(phiOf(o), fam.variance(g.inverse(eta))), o.weights ?? 1),
    logLik: (y, eta, o = {}) => fam.logProb(y, g.inverse(eta), phiOf(o), o.weights),
    score: (y, eta, o = {}) => {
      const mu = g.inverse(eta)
      const numerator = mul(mul(o.weights ?? 1, sub(y, mu)), g.derivative(eta))
      return div(numerator, mul(phiOf(o), fam.variance(mu)))
    },
    unitDeviance: (y, eta) => {
      if (fam.name !== 'binomial' || (g.name !== 'logit' && g.name !== 'cloglog'))
        return fam.unitDeviance(y, g.inverse(eta))
      const saturated = add(xlogy(y, y), xlogy(sub(1, y), sub(1, y)))
      // −log μ and −log(1 − μ).
      const [minusLogMu, minusLogOneMinus] =
        g.name === 'logit' ? [softplus(neg(eta)), softplus(eta)] : [neg(log(neg(expm1(neg(exp(eta)))))), exp(eta)]
      return mul(2, add(saturated, add(mul(y, minusLogMu), mul(sub(1, y), minusLogOneMinus))))
    },
  }
}

// ── Inverse Gaussian predictive ──────────────────────────────────────────────────────────────────────────────────

/**
 * A batch of inverse Gaussian laws $\operatorname{IG}(\mu, \lambda)$ with mean $\mu$ and shape $\lambda$
 * (variance $\mu^3/\lambda$), as a `Distribution`; `aifn-compute/probability/distributions` has no inverse Gaussian
 * yet. Log-density, moments and mode in closed form (Chhikara and Folks, 1989, "The Inverse Gaussian Distribution",
 * ch. 2); draws by Michael, Schucany and Haas (1976). The entropy has no elementary form and throws.
 *
 * @param mu The means $\mu$; their shape is the batch shape.
 * @param lambda The shapes $\lambda$, one per mean (same shape as `mu`).
 * @returns The distribution, whose draws have shape `[...shape, ...batch]`.
 */
function inverseGaussianPredictive(mu: Tensor, lambda: Tensor): Distribution {
  const logProb = (x: Value): Value =>
    sub(
      mul(0.5, log(div(lambda, mul(2 * Math.PI, pow(x, 3))))),
      div(mul(lambda, square(sub(x, mu))), mul(mul(2, square(mu)), x)),
    )
  const variance = () => div(pow(mu, 3), lambda)
  return {
    kind: 'distribution',
    name: 'InverseGaussian',
    params: { mean: mu, shape: lambda },
    batchShape: mu.shape,
    eventShape: [],
    support: { type: 'interval', lower: 0, upper: Infinity, lowerOpen: true, upperOpen: true },
    discrete: false,
    logProb,
    prob: (x) => exp(logProb(x)),
    mean: () => mu,
    variance,
    stddev: () => sqrt(variance()),
    // μ (√(1 + 9μ²/(4λ²)) − 3μ/(2λ)).
    mode: () => {
      const r = div(mu, lambda)
      return mul(mu, sub(sqrt(add(1, mul(2.25, square(r)))), mul(1.5, r)))
    },
    entropy: () => {
      throw new RangeError('InverseGaussian: the entropy has no closed form here')
    },
    sample: (s: Stream, { shape = [] }: SampleOptions = {}) => {
      const m = toFlat(mu)
      const l = toFlat(lambda)
      const count = shape.reduce((a, b) => a * b, 1) * m.length
      const z = toFlat(normal(child(s, 'normal'), 0, 1, { shape: [count] }))
      const u = toFlat(uniform(child(s, 'uniform'), 0, 1, { shape: [count] }))
      const out = new Float64Array(count)
      for (let k = 0; k < count; k++) {
        const i = k % m.length
        const v = z[k] * z[k]
        const mi = m[i]
        const x = mi + (mi * mi * v) / (2 * l[i]) - (mi / (2 * l[i])) * Math.sqrt(4 * mi * l[i] * v + mi * mi * v * v)
        out[k] = u[k] <= mi / (mi + x) ? x : (mi * mi) / x
      }
      return fromData(out, [...shape, ...mu.shape])
    },
  }
}
