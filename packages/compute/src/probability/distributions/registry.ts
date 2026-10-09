/**
 * The registry of distribution families (design S §2.5): every constructor of the module with its parameters as a
 * `Space` (in argument order, with ranges and defaults), its support, whether it is discrete, its event rank and
 * whether its instances expose exponential-family structure. The lab's family explorer, pickers, the catalog and the
 * generated sampling tests enumerate this one table.
 *
 * Vector and matrix parameters (Categorical's probabilities, a multivariate normal's mean and covariance) cannot be a
 * scalar `Space` dimension: those families describe their size (`categories`, `dim`) instead, and a caller builds the
 * vectors.
 */

import {
  definer,
  entries,
  type DistributionInfo,
  type Entry,
  type FunctionInfo,
} from 'aifn-compute/foundation/registry'
import * as divergence from './divergence'
import * as klRules from './kl'
import { int, real, space } from 'aifn-compute/foundation/space'
import * as compose from './compose'
import * as continuous from './continuous'
import * as discrete from './discrete'
import * as multivariate from './multivariate'

const family = definer<DistributionInfo>('distribution', 'probability/distributions')

const loc = real(-5, 5, { default: 0, label: '\\mu', doc: 'location' })
const scale = real(0.1, 5, { default: 1, scale: 'log', label: '\\sigma', doc: 'scale' })
const ls = space({ loc, scale })
/**
 * A success probability $p \in [0, 1]$ as a `Space` dimension.
 *
 * @param d Its default value.
 */
const prob = (d: number) => real(0, 1, { default: d, label: 'p', doc: 'success probability' })
const none = space({})

// ── Continuous univariate ────────────────────────────────────────────────────────────────────────────────────────────

family(
  {
    key: 'Normal',
    stability: 'stable',
    name: 'Normal',
    params: ls,
    support: 'real',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['gaussian-distribution'],
    glossary: 'gaussian-distribution',
  },
  continuous.Normal,
)
family(
  {
    key: 'LogNormal',
    stability: 'stable',
    name: 'Log-normal',
    params: space({
      mu: real(-2, 2, { default: 0, label: '\\mu', doc: 'mean of log x' }),
      sigma: real(0.1, 2, { default: 0.5, scale: 'log', label: '\\sigma', doc: 'standard deviation of log x' }),
    }),
    support: 'positive',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['log-normal-distribution'],
  },
  continuous.LogNormal,
)
family(
  {
    key: 'StudentT',
    stability: 'stable',
    name: 'Student t',
    params: space({
      df: real(0.5, 30, { default: 4, scale: 'log', label: '\\nu', doc: 'degrees of freedom' }),
      loc,
      scale,
    }),
    support: 'real',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    notes: ['student-t-distribution'],
  },
  continuous.StudentT,
)
family(
  {
    key: 'Cauchy',
    stability: 'stable',
    name: 'Cauchy',
    params: ls,
    support: 'real',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    notes: ['cauchy-distribution'],
  },
  continuous.Cauchy,
)
family(
  {
    key: 'Laplace',
    stability: 'stable',
    name: 'Laplace',
    params: ls,
    support: 'real',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    notes: ['laplace-distribution'],
  },
  continuous.Laplace,
)
family(
  {
    key: 'Logistic',
    stability: 'stable',
    name: 'Logistic',
    params: ls,
    support: 'real',
    discrete: false,
    eventRank: 0,
    expFamily: false,
  },
  continuous.Logistic,
)
family(
  {
    key: 'Uniform',
    stability: 'stable',
    name: 'Uniform',
    params: space({
      low: real(-5, 0, { default: 0, label: 'a', doc: 'lower end' }),
      high: real(0.5, 5, { default: 1, label: 'b', doc: 'upper end' }),
    }),
    support: 'interval',
    discrete: false,
    eventRank: 0,
    expFamily: false,
  },
  continuous.Uniform,
)
family(
  {
    key: 'Exponential',
    stability: 'stable',
    name: 'Exponential',
    params: space({ rate: real(0.1, 5, { default: 1, scale: 'log', label: '\\lambda', doc: 'rate' }) }),
    support: 'non-negative',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['exponential-distribution'],
  },
  continuous.Exponential,
)
const gammaShape = real(0.1, 10, { default: 2, scale: 'log', label: '\\alpha', doc: 'shape' })
family(
  {
    key: 'Gamma',
    stability: 'stable',
    name: 'Gamma',
    params: space({
      shape: gammaShape,
      rate: real(0.1, 10, { default: 1, scale: 'log', label: '\\beta', doc: 'rate' }),
    }),
    support: 'non-negative',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['gamma-distribution'],
  },
  continuous.Gamma,
)
family(
  {
    key: 'GammaWithScale',
    stability: 'stable',
    name: 'Gamma (scale)',
    summary: 'The gamma family parameterised by shape and scale θ = 1/β; its instances are named Gamma.',
    params: space({
      shape: gammaShape,
      scale: real(0.1, 10, { default: 1, scale: 'log', label: '\\theta', doc: 'scale' }),
    }),
    support: 'non-negative',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['gamma-distribution'],
  },
  continuous.GammaWithScale,
)
family(
  {
    key: 'ChiSquare',
    stability: 'stable',
    name: 'Chi-square',
    params: space({ df: real(0.5, 30, { default: 3, label: 'k', doc: 'degrees of freedom' }) }),
    support: 'non-negative',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['chi-squared-distribution'],
  },
  continuous.ChiSquare,
)
family(
  {
    key: 'FisherSnedecor',
    name: 'F (Fisher–Snedecor)',
    params: space({
      df1: real(1, 30, { default: 5, label: 'd_1', doc: 'numerator degrees of freedom' }),
      df2: real(5, 60, { default: 20, label: 'd_2', doc: 'denominator degrees of freedom' }),
    }),
    support: 'non-negative',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    notes: ['f-distribution'],
  },
  continuous.FisherSnedecor,
)
family(
  {
    key: 'InverseGamma',
    stability: 'stable',
    name: 'Inverse gamma',
    params: space({
      shape: real(0.5, 10, { default: 3, scale: 'log', label: '\\alpha', doc: 'shape' }),
      scale: real(0.1, 10, { default: 1, scale: 'log', label: '\\beta', doc: 'scale' }),
    }),
    support: 'positive',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['inverse-gamma-distribution'],
  },
  continuous.InverseGamma,
)
family(
  {
    key: 'Beta',
    stability: 'stable',
    name: 'Beta',
    params: space({
      a: real(0.1, 10, { default: 2, scale: 'log', label: '\\alpha', doc: 'first shape' }),
      b: real(0.1, 10, { default: 3, scale: 'log', label: '\\beta', doc: 'second shape' }),
    }),
    support: 'unit-interval',
    discrete: false,
    eventRank: 0,
    expFamily: true,
    notes: ['beta-distribution'],
  },
  continuous.Beta,
)
family(
  {
    key: 'Weibull',
    stability: 'stable',
    name: 'Weibull',
    params: space({
      shape: real(0.2, 10, { default: 1.5, scale: 'log', label: 'k', doc: 'shape' }),
      scale: real(0.1, 10, { default: 1, scale: 'log', label: '\\lambda', doc: 'scale' }),
    }),
    support: 'non-negative',
    discrete: false,
    eventRank: 0,
    expFamily: false,
  },
  continuous.Weibull,
)
family(
  {
    key: 'Gumbel',
    stability: 'stable',
    name: 'Gumbel',
    params: ls,
    support: 'real',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    notes: ['gumbel-max-trick'],
  },
  continuous.Gumbel,
)
family(
  {
    key: 'GeneralisedPareto',
    name: 'Generalised Pareto',
    params: space({
      shape: real(-0.5, 1, { default: 0.2, label: '\\xi', doc: 'shape (tail index)' }),
      loc: real(-5, 5, { default: 0, label: '\\mu', doc: 'location (the threshold)' }),
      scale: real(0.1, 10, { default: 1, scale: 'log', label: '\\sigma', doc: 'scale' }),
    }),
    support: 'interval',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    notes: ['extreme-value-theory-for-anomalies'],
    cite: ['pickands1975'],
  },
  continuous.GeneralisedPareto,
)
family(
  {
    key: 'VonMises',
    name: 'Von Mises',
    params: space({
      loc: real(-Math.PI, Math.PI, { default: 0, label: '\\mu', doc: 'mean direction' }),
      concentration: real(0.01, 50, { default: 2, scale: 'log', label: '\\kappa', doc: 'concentration' }),
    }),
    support: 'circle',
    discrete: false,
    eventRank: 0,
    expFamily: false,
  },
  continuous.VonMises,
)
family(
  {
    key: 'TruncatedNormal',
    stability: 'stable',
    name: 'Truncated normal',
    params: space({
      loc,
      scale,
      low: real(-5, 0, { default: -1, label: 'a', doc: 'lower truncation point' }),
      high: real(0, 5, { default: 2, label: 'b', doc: 'upper truncation point' }),
    }),
    support: 'interval',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    notes: ['expectation-propagation-truncated-gaussian'],
  },
  continuous.TruncatedNormal,
)

// ── Discrete univariate ──────────────────────────────────────────────────────────────────────────────────────────────

family(
  {
    key: 'Bernoulli',
    stability: 'stable',
    name: 'Bernoulli',
    params: space({ p: prob(0.3) }),
    support: 'binary',
    discrete: true,
    eventRank: 0,
    expFamily: true,
    notes: ['bernoulli-distribution'],
  },
  discrete.Bernoulli,
)
family(
  {
    key: 'Binomial',
    stability: 'stable',
    name: 'Binomial',
    params: space({ n: int(1, 100, { default: 10, label: 'n', doc: 'trials' }), p: prob(0.3) }),
    support: 'integers',
    discrete: true,
    eventRank: 0,
    expFamily: true,
    notes: ['binomial-distribution'],
  },
  discrete.Binomial,
)
family(
  {
    key: 'Poisson',
    stability: 'stable',
    name: 'Poisson',
    params: space({ rate: real(0.1, 50, { default: 3, scale: 'log', label: '\\lambda', doc: 'rate' }) }),
    support: 'non-negative-integers',
    discrete: true,
    eventRank: 0,
    expFamily: true,
    notes: ['poisson-distribution', 'poisson-process'],
  },
  discrete.Poisson,
)
family(
  {
    key: 'Geometric',
    stability: 'stable',
    name: 'Geometric',
    params: space({ p: real(0.01, 1, { default: 0.3, label: 'p', doc: 'success probability' }) }),
    support: 'positive-integers',
    discrete: true,
    eventRank: 0,
    expFamily: true,
    notes: ['geometric-distribution'],
  },
  discrete.Geometric,
)
family(
  {
    key: 'NegativeBinomial',
    stability: 'stable',
    name: 'Negative binomial',
    params: space({
      r: real(0.1, 50, { default: 3, scale: 'log', label: 'r', doc: 'number of successes' }),
      p: real(0.01, 0.99, { default: 0.5, label: 'p', doc: 'success probability' }),
    }),
    support: 'non-negative-integers',
    discrete: true,
    eventRank: 0,
    expFamily: true,
    notes: ['negative-binomial-distribution', 'negative-binomial-and-overdispersion'],
  },
  discrete.NegativeBinomial,
)
family(
  {
    key: 'Hypergeometric',
    stability: 'stable',
    name: 'Hypergeometric',
    params: space({
      population: int(1, 100, { default: 20, label: 'N', doc: 'population size' }),
      successes: int(0, 100, { default: 7, label: 'K', doc: 'successes in the population' }),
      draws: int(0, 100, { default: 12, label: 'n', doc: 'draws without replacement' }),
    }),
    support: 'integers',
    discrete: true,
    eventRank: 0,
    expFamily: false,
    notes: ['hypergeometric-distribution'],
  },
  discrete.Hypergeometric,
)
family(
  {
    key: 'DiscreteUniform',
    stability: 'stable',
    name: 'Discrete uniform',
    params: space({
      low: int(-10, 10, { default: 0, label: 'a', doc: 'lowest value' }),
      high: int(-10, 20, { default: 5, label: 'b', doc: 'highest value' }),
    }),
    support: 'integers',
    discrete: true,
    eventRank: 0,
    expFamily: false,
  },
  discrete.DiscreteUniform,
)
family(
  {
    key: 'Categorical',
    stability: 'stable',
    name: 'Categorical',
    summary: 'One of K categories with probabilities p (a vector, or logits).',
    params: space({
      categories: int(2, 20, { default: 3, label: 'K', doc: 'number of categories (the length of p)' }),
    }),
    support: 'categories',
    discrete: true,
    eventRank: 0,
    expFamily: true,
    notes: ['categorical-distribution'],
  },
  discrete.Categorical,
)

// ── Multivariate ─────────────────────────────────────────────────────────────────────────────────────────────────────

const dim = int(1, 10, { default: 2, label: 'd', doc: 'dimension' })
family(
  {
    key: 'MultivariateNormal',
    stability: 'stable',
    name: 'Multivariate normal',
    summary: 'A normal distribution on ℝᵈ with mean vector μ and a covariance, precision or Cholesky factor.',
    params: space({ dim }),
    support: 'real-vector',
    discrete: false,
    eventRank: 1,
    expFamily: true,
    notes: ['multivariate-normal-distribution'],
    glossary: 'multivariate-gaussian',
  },
  multivariate.MultivariateNormal,
)
family(
  {
    key: 'Dirichlet',
    stability: 'stable',
    name: 'Dirichlet',
    params: space({ dim: int(2, 10, { default: 3, label: 'K', doc: 'number of categories' }) }),
    support: 'simplex',
    discrete: false,
    eventRank: 1,
    expFamily: true,
    notes: ['dirichlet-distribution'],
  },
  multivariate.Dirichlet,
)
family(
  {
    key: 'Multinomial',
    stability: 'stable',
    name: 'Multinomial',
    params: space({
      n: int(1, 100, { default: 10, label: 'n', doc: 'trials' }),
      categories: int(2, 10, { default: 3, label: 'K', doc: 'number of categories' }),
    }),
    support: 'count-vector',
    discrete: true,
    eventRank: 1,
    expFamily: false,
    notes: ['multinomial-distribution'],
  },
  multivariate.Multinomial,
)
family(
  {
    key: 'Wishart',
    stability: 'stable',
    name: 'Wishart',
    params: space({
      df: real(1, 30, { default: 4, label: '\\nu', doc: 'degrees of freedom (> d − 1)' }),
      dim: int(1, 6, { default: 2, label: 'd', doc: 'matrix size' }),
    }),
    support: 'positive-definite',
    discrete: false,
    eventRank: 2,
    expFamily: false,
    notes: ['wishart-distribution'],
  },
  multivariate.Wishart,
)

// ── Composition ──────────────────────────────────────────────────────────────────────────────────────────────────────

family(
  {
    key: 'Mixture',
    stability: 'stable',
    name: 'Mixture',
    summary: 'A finite mixture of univariate components with given weights.',
    params: space({ components: int(1, 10, { default: 2, label: 'K', doc: 'number of components' }) }),
    support: 'varies',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    composite: true,
    notes: ['gaussian-mixture-model'],
  },
  compose.Mixture,
)
family(
  {
    key: 'ZeroInflated',
    stability: 'stable',
    name: 'Zero-inflated',
    summary: 'A discrete distribution with extra structural zeros: P(0) = π + (1 − π)p(0), P(k) = (1 − π)p(k).',
    params: space({ pi: real(0, 1, { default: 0.3, label: '\\pi', doc: 'probability of a structural zero' }) }),
    support: 'varies',
    discrete: true,
    eventRank: 0,
    expFamily: false,
    composite: true,
    notes: ['poisson-regression', 'poisson-distribution'],
  },
  compose.ZeroInflated,
)
family(
  {
    key: 'Independent',
    stability: 'stable',
    name: 'Independent',
    summary: 'Reinterprets batch axes of a distribution as event axes.',
    params: space({ reinterpreted: int(1, 3, { default: 1, doc: 'batch axes moved into the event' }) }),
    support: 'varies',
    discrete: false,
    eventRank: 1,
    expFamily: false,
    composite: true,
  },
  compose.Independent,
)
family(
  {
    key: 'Transformed',
    stability: 'stable',
    name: 'Transformed',
    summary: 'A univariate distribution pushed through a monotone bijector, with the change-of-variables density.',
    params: none,
    support: 'varies',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    composite: true,
    notes: ['change-of-variables'],
  },
  compose.Transformed,
)
family(
  {
    key: 'Pushforward',
    stability: 'stable',
    name: 'Pushforward',
    summary: 'A univariate distribution pushed through a many-to-one map, summing over preimages.',
    params: none,
    support: 'varies',
    discrete: false,
    eventRank: 0,
    expFamily: false,
    composite: true,
    notes: ['change-of-variables'],
  },
  compose.Pushforward,
)

/** Every distribution family, keyed by constructor name. */
export const distributionRegistry: Readonly<Record<string, Entry<(...args: never[]) => unknown, DistributionInfo>>> =
  entries<DistributionInfo>('distribution', continuous, discrete, multivariate, compose) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, DistributionInfo>>
  >

// ── Functions ────────────────────────────────────────────────────────────────────────────────────────────────────────

const fn = definer<FunctionInfo>('function', 'probability/distributions')
const KL = ['kullback-leibler-divergence']

fn(
  {
    key: 'kl',
    name: 'KL divergence (closed form)',
    stability: 'stable',
    tex: 'D_{KL}(p \\| q)',
    summary: 'KL(p ‖ q) by the registered closed-form rule for the pair of families.',
    role: 'property',
    notes: KL,
    cite: ['kullback1951'],
  },
  klRules.kl,
)
fn(
  {
    key: 'klMonteCarlo',
    name: 'KL divergence (Monte Carlo)',
    role: 'estimator',
    random: true,
    notes: [...KL, 'monte-carlo-integration'],
  },
  klRules.klMonteCarlo,
)
fn(
  {
    key: 'klNumerical',
    name: 'KL divergence (quadrature)',
    role: 'estimator',
    notes: [...KL, 'numerical-integration'],
  },
  divergence.klNumerical,
)
fn(
  {
    key: 'klMonteCarloWithError',
    name: 'KL divergence (Monte Carlo, with error)',
    role: 'estimator',
    random: true,
    notes: KL,
  },
  divergence.klMonteCarloWithError,
)
fn(
  {
    key: 'klAuto',
    name: 'KL divergence (best available)',
    summary: 'The closed form when a rule exists, else quadrature in one dimension, else Monte Carlo.',
    role: 'estimator',
    notes: KL,
  },
  divergence.klAuto,
)
fn(
  {
    key: 'entropyNumerical',
    name: 'Entropy (quadrature)',
    role: 'estimator',
    notes: ['differential-entropy', 'entropy'],
  },
  divergence.entropyNumerical,
)
fn(
  {
    key: 'entropyAuto',
    name: 'Entropy (best available)',
    role: 'estimator',
    notes: ['differential-entropy', 'entropy'],
  },
  divergence.entropyAuto,
)
fn(
  {
    key: 'crossEntropyAuto',
    name: 'Cross-entropy (best available)',
    role: 'estimator',
    notes: ['cross-entropy-and-perplexity'],
  },
  divergence.crossEntropyAuto,
)
fn(
  {
    key: 'jensenShannonNumerical',
    name: 'Jensen–Shannon divergence (quadrature)',
    role: 'estimator',
    notes: ['f-divergences-and-jensen-shannon'],
  },
  divergence.jensenShannonNumerical,
)
fn(
  {
    key: 'normalFromNatural',
    name: 'Normal from natural parameters',
    role: 'construction',
    notes: ['exponential-family', 'gaussian-distribution'],
  },
  continuous.normalFromNatural,
)

/** The functions of the module (divergences and constructions), keyed by name. */
export const distributionFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', klRules, divergence, continuous) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
