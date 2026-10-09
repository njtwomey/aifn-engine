/**
 * `aifn-compute/probability/distributions`: probability distributions as plain objects (plan §5.2).
 *
 * - Continuous families: Normal (and `normalFromNatural`), LogNormal, StudentT, Cauchy, Laplace, Logistic, Uniform,
 *   Exponential, Gamma (and `GammaWithScale`), InverseGamma, Beta, ChiSquare, FisherSnedecor (F), Weibull, Gumbel,
 *   GeneralisedPareto, VonMises, TruncatedNormal.
 * - Discrete families: Bernoulli, Binomial, Categorical, Poisson, Geometric, NegativeBinomial, Hypergeometric,
 *   DiscreteUniform.
 * - Multivariate families: MultivariateNormal (with `condition` and `marginal`), Dirichlet, Multinomial, Wishart.
 * - Compositions: Mixture, ZeroInflated, Independent, Transformed (through a monotone bijector of
 *   `aifn-compute/probability/bijectors`, such as `affineBijector`, `expBijector`, `sigmoidBijector` or
 *   `chainBijectors`) and Pushforward (through a many-to-one map of the same module, such as `squareMap`, summing over
 *   preimages).
 * - A new univariate family: `univariate` builds one from a `UnivariateSpec` (its log-density and cdf at least) and
 *   derives the rest, numerical quantiles included.
 * - Divergences: `kl`, `klMonteCarlo`, `hasKl`, and the static table of closed forms `klRegistry`; for continuous
 *   univariate pairs without a closed form, `klNumerical`, `entropyNumerical`, `jensenShannonNumerical` (quadrature),
 *   `klAuto`, `entropyAuto`, `crossEntropyAuto` (closed form when registered, quadrature otherwise, with the method
 *   reported), `klMonteCarloWithError` and `klIntegrand`.
 * - Registries: `distributionRegistry` (every constructor with its parameter `Space`, support and structure) and
 *   `distributionFunctions` (the module's divergences and constructions).
 *
 * A distribution is built by calling its family with its parameters, `Normal(0, 2)`, and read through its methods:
 * `logProb(1)` gives a number, `cdf` of a tensor of points a tensor, `sample(stream(7), { shape: [1000] })` a tensor of
 * draws, and `Normal(tensor([0, 1]), 1)` is a batch of two normals with `batchShape` `[2]`.
 *
 * - Every distribution has `logProb`, `prob`, `sample(s, { shape })`, `mean`, `variance` (and `covariance` for
 *   vectors), `stddev`, `entropy`, `mode`, `support`, `batchShape` and `eventShape`. Univariate ones add `cdf`,
 *   `logcdf`, `survival`, `logSurvival`, `quantile` and `isf` (the inverse survival function). Each keeps its relative
 *   accuracy in both tails: log tails never take the log of a value near 1, and upper quantiles invert the survival
 *   function. Draws have shape `[...shape, ...batchShape, ...eventShape]`.
 * - Parameters and values may be numbers, tensors (batches broadcast, NumPy rules) or traced values: log-densities
 *   are compositions of `aifn-compute/numerics/special` primitives, so they are differentiable in values and
 *   parameters. Families with a pathwise draw add `rsample`, differentiable in the parameters. Parameters out of range
 *   throw a `DomainError` when the distribution is built.
 * - Exponential-family members expose `expFamily` (`naturalParams`, `sufficientStats`, `logPartition`,
 *   `logBaseMeasure`), used by `pgm` and `ep`.
 * - Parameterisations follow scipy.stats: Gaussians by mean and standard deviation, `Gamma(shape, rate)` with
 *   `GammaWithScale`, Geometric on $k \ge 1$, NegativeBinomial counting failures.
 */

export type {
  AnyMultivariate,
  AnyUnivariate,
  Distribution,
  EventKind,
  ExponentialFamily,
  Kind,
  LogDensity,
  Multivariate,
  SampleKind,
  SampleOptions,
  Support,
  TypedMultivariate,
  TypedUnivariate,
  Univariate,
} from './types'
export {
  Beta,
  Cauchy,
  ChiSquare,
  Exponential,
  FisherSnedecor,
  Gamma,
  GammaWithScale,
  Gumbel,
  GeneralisedPareto,
  InverseGamma,
  Laplace,
  Logistic,
  LogNormal,
  Normal,
  normalFromNatural,
  StudentT,
  TruncatedNormal,
  Uniform,
  VonMises,
  Weibull,
} from './continuous'
export {
  Bernoulli,
  Binomial,
  Categorical,
  DiscreteUniform,
  Geometric,
  Hypergeometric,
  NegativeBinomial,
  Poisson,
} from './discrete'
export { Dirichlet, Multinomial, MultivariateNormal, Wishart, type MultivariateNormalSpread } from './multivariate'
export { Independent, Mixture, Pushforward, Transformed, ZeroInflated } from './compose'
export { hasKl, kl, klMonteCarlo, klRegistry, type KlRule } from './kl'
export { distributionFunctions, distributionRegistry } from './registry'
export { univariate, type UnivariateSpec } from './util'
export {
  crossEntropyAuto,
  entropyAuto,
  entropyNumerical,
  jensenShannonNumerical,
  klAuto,
  klIntegrand,
  klMonteCarloWithError,
  klNumerical,
  type DivergenceMethod,
  type MethodResult,
  type MonteCarloEstimate,
  type NumericalResult,
} from './divergence'
