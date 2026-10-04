/**
 * Link functions and exponential-dispersion families for generalised linear models (Nelder and Wedderburn, 1972;
 * McCullagh and Nelder, 1989, "Generalized Linear Models", 2nd ed., ch. 2 and Table 2.1), and the likelihood of a
 * response given a linear predictor η through a link (`likelihood`). Links, variance functions, unit deviances and
 * pointwise log-likelihoods are compositions of `aifn-compute/foundation/tensor` and `aifn-compute/numerics/special` primitives, so
 * they accept numbers, tensors and traced values and are differentiable to any order.
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

/** A link g: the linear predictor is η = g(μ). Every function is a composition of primitives. */
export interface Link {
  readonly name: LinkName
  /** η = g(μ). */
  link(mu: Value): Value
  /** μ = g⁻¹(η), the mean function. */
  inverse(eta: Value): Value
  /** dμ/dη at η. */
  derivative(eta: Value): Value
  /**
   * True when g⁻¹ maps every finite η into the open mean space (log, logit, probit, cloglog): any finite η is a valid
   * linear predictor, even where μ rounds to the edge of the space (σ(40) = 1 in float64). False when η itself must
   * stay inside a range (identity, inverse, sqrt, inverse squared).
   */
  readonly total: boolean
}

/** 1 in the shape of v (a number, tensor or traced value). */
function onesLike(v: Value): Value {
  return add(mul(0, v), 1)
}

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

/** The link function by name (McCullagh and Nelder, 1989, §2.2.2). */
export function link(name: LinkName): Link {
  const l = LINKS[name]
  if (!l) throw new DomainError('link', `link: unknown link "${name}"`)
  return { name, ...l }
}

// ── Families ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** The name of a family. */
export type FamilyName = 'gaussian' | 'binomial' | 'poisson' | 'gamma' | 'inverse-gaussian' | 'negative-binomial'

/**
 * An exponential-dispersion family: Var(y) = φ V(μ)/w for prior weight w. The deviance is Σ wᵢ d(yᵢ, μᵢ) with the unit
 * deviance d. Binomial responses are proportions with the number of trials as the prior weight (as R's `glm`).
 */
export interface Family {
  readonly name: FamilyName
  /** Parameters fixed when the family was made (e.g. the negative binomial's θ). */
  readonly params: Readonly<Record<string, Scalar>>
  /** The canonical link: the one that makes η the natural parameter. */
  readonly canonicalLink: LinkName
  /** The default link (the canonical one except for the negative binomial, whose conventional link is log). */
  readonly defaultLink: LinkName
  /**
   * The links the family is used with (McCullagh and Nelder, 1989, Table 2.1; R's `family` objects): each maps the
   * family's mean space onto a range a linear predictor can reach. `checkLink` rejects any other.
   */
  readonly links: readonly LinkName[]
  /** φ when it is known (1 for binomial, Poisson and negative binomial); null when it is estimated. */
  readonly dispersion: Scalar | null
  /** V(μ). */
  variance(mu: Value): Value
  /** d(y, μ), the unit deviance (twice the log-likelihood ratio of the saturated model for one observation). */
  unitDeviance(y: Value, mu: Value): Value
  /**
   * The pointwise log-likelihood log p(yᵢ | μᵢ, φ, wᵢ), elementwise over broadcast (y, μ, w), differentiable in μ (and
   * φ). Prior weights default to 1; binomial weights are trial counts.
   */
  logProb(y: Value, mu: Value, dispersion: Value, weights?: Value): Value
  /** Σᵢ `logProb` as a number (y, μ and weights untraced), for reports and information criteria. */
  logLikelihood(y: Tensor, mu: Tensor, dispersion: Scalar, weights: Tensor): Scalar
  /** True when every μ lies in the family's mean space. */
  validMean(mu: Tensor): boolean
  /** A starting mean for IRLS from the responses and prior weights (as R's `mustart`). */
  initialMean(y: Tensor, weights: Tensor): Tensor
  /** The distribution of a new response given its mean (and dispersion; prior weights as trials or precision). */
  predictive(mu: Tensor, dispersion: Scalar, weights?: Tensor): Distribution
}

const all = (mu: Tensor, ok: (m: number) => boolean) => toFlat(mu).every(ok)
/** An elementwise map of one or two same-shape raw tensors (starting means, predictive parameters). */
const zip = (a: Tensor, b: Tensor, f: (x: number, y: number) => number) => {
  const x = toFlat(a)
  const y = toFlat(b)
  return fromData(
    Float64Array.from(x, (v, i) => f(v, y[i])),
    a.shape,
  )
}
/** The per-observation parameter φ/w (or its reciprocal) as a tensor shaped like μ. */
const perObservation = (mu: Tensor, w: Tensor | undefined, f: (w: number) => number) =>
  w ? zip(w, w, f) : full(mu.shape, f(1))

/** Σᵢ logProb as a number: the shared definition of `logLikelihood`. */
function summed(logProb: Family['logProb']): Family['logLikelihood'] {
  return (y, mu, phi, w) => {
    const v = unwrap(sum(logProb(y, mu, phi, w)))
    return typeof v === 'number' ? v : toFlat(v)[0]
  }
}

/** The Gaussian family: V(μ) = 1, d = (y − μ)², canonical link identity. */
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
 * The binomial family for proportions y ∈ [0, 1] with wᵢ trials (Bernoulli when every w = 1): V(μ) = μ(1 − μ),
 * d = 2[y log(y/μ) + (1 − y) log((1 − y)/(1 − μ))], canonical link logit.
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

/** The Poisson family: V(μ) = μ, d = 2[y log(y/μ) − (y − μ)], canonical link log. */
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

/** The gamma family: V(μ) = μ², d = 2[−log(y/μ) + (y − μ)/μ], canonical link inverse; shape w/φ. */
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

/** The inverse Gaussian family: V(μ) = μ³, d = (y − μ)²/(μ²y), canonical link 1/μ². */
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
 * The negative binomial family with fixed shape θ > 0 (NB2): V(μ) = μ + μ²/θ,
 * d = 2[y log(y/μ) − (y + θ) log((y + θ)/(μ + θ))], default link log (Hilbe, 2011, "Negative Binomial Regression").
 * Its canonical link is log(μ/(μ + θ)). The predictive counts failures before θ successes with p = θ/(θ + μ), so its
 * mean is μ.
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

/** A family by name (the negative binomial needs θ, default 1). */
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

/** Options of a likelihood's functions of (y, η): the dispersion φ (default the family's, else 1) and prior weights. */
export type LikelihoodOptions = { dispersion?: Value; weights?: Value }

/**
 * The likelihood of a response y given the linear predictor η: a family with a link, μ = g⁻¹(η). Every function is a
 * composition of primitives, elementwise over broadcast arguments, so it differentiates in η (and φ).
 */
export interface Likelihood {
  readonly family: Family
  readonly link: Link
  /** μ = g⁻¹(η). */
  mean(eta: Value): Value
  /** Var(y) = φ V(μ)/w at η. */
  variance(eta: Value, options?: LikelihoodOptions): Value
  /** The pointwise log-likelihood log p(y | μ = g⁻¹(η), φ, w). */
  logLik(y: Value, eta: Value, options?: LikelihoodOptions): Value
  /**
   * The score ∂ℓ/∂η = w (y − μ) g⁻¹′(η) / (φ V(μ)) in closed form (McCullagh and Nelder, 1989, eq. 2.13); equal to
   * differentiating `logLik` in η for the families whose log-likelihood depends on μ only through the exponential-family
   * kernel (all six here).
   */
  score(y: Value, eta: Value, options?: LikelihoodOptions): Value
  /**
   * The unit deviance d(y, g⁻¹(η)), from η directly where μ would lose precision: for the binomial with the logit
   * link, 2[y log y + (1 − y) log(1 − y) + y softplus(−η) + (1 − y) softplus(η)]; with the cloglog link,
   * log(1 − μ) = −e^η and log μ = log(−expm1(−e^η)). Finite for every finite η, where the μ form overflows once μ
   * rounds to 0 or 1.
   */
  unitDeviance(y: Value, eta: Value): Value
}

/**
 * A likelihood from a family and a link (default: the family's default link), e.g. logistic regression's
 * `likelihood(binomialFamily())` or a log-linear gamma model's `likelihood(gammaFamily(), 'log')`.
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
 * A batch of inverse Gaussian laws IG(μ, λ) with mean μ and shape λ (variance μ³/λ), as a `Distribution`;
 * `aifn-compute/probability/distributions` has no inverse Gaussian yet. Log-density, moments and mode in closed form
 * (Chhikara and Folks, 1989, "The Inverse Gaussian Distribution", ch. 2); draws by Michael, Schucany and Haas (1976).
 * The entropy has no elementary form and throws.
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
