/**
 * The registries of GLM link functions and exponential-dispersion families (McCullagh and Nelder, 1989, Table 2.1):
 * each link with the mean space it maps from, each family with its response support, canonical link, the links it is
 * used with and whether its dispersion is estimated.
 */

import {
  definer,
  entries,
  type Entry,
  type FunctionInfo,
  type LikelihoodInfo,
  type LinkInfo,
} from 'aifn-compute/foundation/registry'
import { real, space } from 'aifn-compute/foundation/space'
import * as distributional from './distributional'
import * as families from './families'
import { link, type Link, type LinkName } from './families'

const defineLink = definer<LinkInfo>('link', 'probability/likelihoods')
const defineFamily = definer<LikelihoodInfo>('likelihood', 'probability/likelihoods')
const cite = ['mccullagh1989', 'nelder1972']
const notes = ['generalised-linear-model']

/**
 * Register a link with its metadata, citing McCullagh and Nelder and the GLM note unless `spec` overrides them.
 *
 * @param name The link's name, its registry key.
 * @param spec The link's metadata: display name, TeX, mean space, and optionally its own notes.
 * @returns The registered entry.
 */
const linkEntry = (name: LinkName, spec: Omit<LinkInfo, 'key' | 'kind' | 'module' | 'stability'>) =>
  defineLink({ key: name, cite, notes, ...spec }, link(name))

/** Every link function, keyed by its name (the argument of `link`). */
export const linkRegistry: Readonly<Record<LinkName, Entry<Link, LinkInfo>>> = entries<LinkInfo>('link', {
  identity: linkEntry('identity', { name: 'identity', tex: '\\eta = \\mu', meanSpace: 'real' }),
  log: linkEntry('log', { name: 'log', tex: '\\eta = \\log \\mu', meanSpace: 'positive' }),
  logit: linkEntry('logit', {
    name: 'logit',
    tex: '\\eta = \\log \\frac{\\mu}{1 - \\mu}',
    meanSpace: 'unit-interval',
    notes: ['generalised-linear-model', 'logistic-regression'],
  }),
  probit: linkEntry('probit', {
    name: 'probit',
    tex: '\\eta = \\Phi^{-1}(\\mu)',
    meanSpace: 'unit-interval',
    notes: ['generalised-linear-model', 'expectation-propagation-probit-regression'],
  }),
  cloglog: linkEntry('cloglog', {
    name: 'complementary log-log',
    tex: '\\eta = \\log(-\\log(1 - \\mu))',
    meanSpace: 'unit-interval',
  }),
  inverse: linkEntry('inverse', { name: 'inverse', tex: '\\eta = 1/\\mu', meanSpace: 'positive' }),
  'inverse-squared': linkEntry('inverse-squared', {
    name: 'inverse squared',
    tex: '\\eta = 1/\\mu^2',
    meanSpace: 'positive',
  }),
  sqrt: linkEntry('sqrt', { name: 'square root', tex: '\\eta = \\sqrt{\\mu}', meanSpace: 'non-negative' }),
}) as Readonly<Record<LinkName, Entry<Link, LinkInfo>>>

/** The parameter space of a family with no parameters. */
const none = space({})

defineFamily(
  {
    key: 'gaussianFamily',
    name: 'Gaussian',
    support: 'real',
    canonicalLink: 'identity',
    links: families.gaussianFamily().links,
    dispersion: true,
    params: none,
    cite,
    notes,
  },
  families.gaussianFamily,
)
defineFamily(
  {
    key: 'binomialFamily',
    name: 'Binomial',
    support: 'unit-interval',
    canonicalLink: 'logit',
    links: families.binomialFamily().links,
    dispersion: false,
    params: none,
    cite,
    notes: ['generalised-linear-model', 'logistic-regression'],
  },
  families.binomialFamily,
)
defineFamily(
  {
    key: 'poissonFamily',
    name: 'Poisson',
    support: 'non-negative-integers',
    canonicalLink: 'log',
    links: families.poissonFamily().links,
    dispersion: false,
    params: none,
    cite,
    notes: ['poisson-regression', 'generalised-linear-model'],
  },
  families.poissonFamily,
)
defineFamily(
  {
    key: 'gammaFamily',
    name: 'Gamma',
    support: 'positive',
    canonicalLink: 'inverse',
    links: families.gammaFamily().links,
    dispersion: true,
    params: none,
    cite,
    notes: ['gamma-and-tweedie-regression', 'generalised-linear-model'],
  },
  families.gammaFamily,
)
defineFamily(
  {
    key: 'inverseGaussianFamily',
    name: 'Inverse Gaussian',
    support: 'positive',
    canonicalLink: 'inverse-squared',
    links: families.inverseGaussianFamily().links,
    dispersion: true,
    params: none,
    cite,
    notes,
  },
  families.inverseGaussianFamily,
)
defineFamily(
  {
    key: 'negativeBinomialFamily',
    name: 'Negative binomial',
    support: 'non-negative-integers',
    canonicalLink: 'log',
    links: families.negativeBinomialFamily(1).links,
    dispersion: false,
    params: space({
      theta: real(0.1, 100, { default: 1, scale: 'log', label: '\\theta', doc: 'size (shape) parameter' }),
    }),
    cite,
    notes: ['negative-binomial-and-overdispersion', 'generalised-linear-model'],
  },
  families.negativeBinomialFamily,
)

/** Every exponential-dispersion family factory, keyed by export name. */
export const likelihoodRegistry: Readonly<Record<string, Entry<() => families.Family, LikelihoodInfo>>> =
  entries<LikelihoodInfo>('likelihood', families) as Readonly<
    Record<string, Entry<() => families.Family, LikelihoodInfo>>
  >

const defineFunction = definer<FunctionInfo>('function', 'probability/likelihoods')
const GAMLSS = ['generalised-additive-models-for-location-scale-and-shape']
/**
 * Register a distributional family factory of `./distributional` as a construction function with the GAMLSS note,
 * citing Rigby and Stasinopoulos (2005), Stasinopoulos and Rigby (2007) and `cites`.
 *
 * @param key The factory's export name, its registry key.
 * @param name The display name.
 * @param summary One sentence on the family.
 * @param cites Further citation keys.
 * @returns The registered entry.
 */
const gamlssFamily = (key: string, name: string, summary: string, cites: string[] = []) =>
  defineFunction(
    { key, name, summary, role: 'construction', notes: GAMLSS, cite: ['rigby2005', 'stasinopoulos2007', ...cites] },
    distributional[key as keyof typeof distributional] as () => distributional.DistributionalFamily,
  )
gamlssFamily(
  'normalDistributional',
  'Normal NO(μ, σ)',
  'The normal with mean μ and sd σ, with scores and expected information.',
)
gamlssFamily(
  'studentTDistributional',
  'Student t TF(μ, σ, ν)',
  'Location-scale Student t: heavy tails through ν, with the expected information of Lange, Little and Taylor.',
  ['lange1989'],
)
gamlssFamily(
  'boxCoxColeGreenDistributional',
  'Box–Cox Cole–Green BCCG(μ, σ, ν)',
  'The LMS distribution: median μ, coefficient of variation σ and Box–Cox skewness power ν.',
  ['cole1992'],
)
gamlssFamily('gammaDistributional', 'Gamma GA(μ, σ)', 'The gamma with mean μ and coefficient of variation σ.')
gamlssFamily(
  'poissonDistributional',
  'Poisson PO(μ)',
  'The Poisson with mean μ as a one-parameter distributional family.',
)
defineFunction(
  {
    key: 'distributionalFamily',
    name: 'Distributional-regression family',
    summary: 'A GAMLSS family by name: links, scores, expected second derivatives, cdf and quantile per parameter.',
    role: 'construction',
    notes: GAMLSS,
    cite: ['rigby2005'],
  },
  distributional.distributionalFamily,
)
defineFunction(
  {
    key: 'quantileResidual',
    name: 'Normalised quantile residual',
    summary: 'Φ⁻¹ of the fitted cdf at the response: standard normal under the true model.',
    role: 'estimator',
    notes: [...GAMLSS, 'residual-diagnostics'],
    cite: ['dunn1996'],
  },
  distributional.quantileResidual,
)
defineFunction(
  {
    key: 'wormPlot',
    name: 'Worm plot',
    summary: 'A detrended normal Q–Q plot of residuals with a pointwise 95% band.',
    role: 'estimator',
    notes: [...GAMLSS, 'residual-diagnostics'],
    cite: ['vanbuuren2001'],
  },
  distributional.wormPlot,
)
