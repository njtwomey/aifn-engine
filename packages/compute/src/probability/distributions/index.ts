/**
 * `aifn-compute/probability/distributions`: probability distributions as plain objects (plan §5.2).
 *
 * ```ts
 * const d = Normal(0, 2)
 * d.logProb(1) // a number
 * d.cdf(linspace(-5, 5, 101)) // a tensor
 * d.sample(stream(7), { shape: [1000] }) // a tensor of draws
 * Normal(tensor([0, 1]), 1).batchShape // [2]: a batch of two normals
 * ```
 *
 * - Every distribution has `logProb`, `prob`, `sample(s, { shape })`, `mean`, `variance` (and `covariance` for
 *   vectors), `stddev`, `entropy`, `mode`, `support`, `batchShape` and `eventShape`. Univariate ones add `cdf`,
 *   `logcdf`, `survival`, `logSurvival`, `quantile` and `isf` (the inverse survival function). Each keeps its relative
 *   accuracy in both tails: log tails never take the log of a value near 1, and upper quantiles invert the survival
 *   function. Draws have shape `[...shape, ...batchShape, ...eventShape]`.
 * - Parameters and values may be numbers, tensors (batches broadcast, NumPy rules) or traced values: log-densities
 *   are compositions of `aifn-compute/numerics/special` primitives, so they are differentiable in values and parameters.
 * - Exponential-family members expose `expFamily` (`naturalParams`, `sufficientStats`, `logPartition`,
 *   `logBaseMeasure`), used by `pgm` and `ep`.
 * - Parameterisations follow scipy.stats: Gaussians by mean and standard deviation, `Gamma(shape, rate)` with
 *   `GammaWithScale`, Geometric on k ≥ 1, NegativeBinomial counting failures.
 *
 * Families (`distributionRegistry`: every constructor with its parameter `Space`, support and structure). Continuous: Normal, LogNormal, StudentT, Cauchy, Laplace, Logistic, Uniform, Exponential, Gamma,
 * InverseGamma, Beta, ChiSquare, FisherSnedecor (F), Weibull, Gumbel, GeneralisedPareto, VonMises, TruncatedNormal. Discrete: Bernoulli, Binomial,
 * Categorical, Poisson, Geometric, NegativeBinomial, Hypergeometric, DiscreteUniform. Multivariate:
 * MultivariateNormal (with `condition` and `marginal`), Dirichlet, Multinomial, Wishart. Composition: Mixture,
 * ZeroInflated, Independent, Transformed (through a monotone bijector: `affineBijector`, `expBijector`, `logBijector`,
 * `sigmoidBijector`, `tanhBijector`, `softplusBijector`, `powerBijector`, `normalCdfBijector`, `chainBijectors`),
 * Pushforward (through a many-to-one map such as `squareMap`, summing over preimages). Maps declare their `domain` and
 * `codomain` as `Interval`s; `imageOf`, `supportInterval` and `formatInterval` work with them. Divergences: `kl`,
 * `klMonteCarlo`, `hasKl`, and the static table of closed forms `klRegistry`; for continuous univariate pairs without a closed form, `klNumerical`,
 * `entropyNumerical`, `jensenShannonNumerical` (quadrature), `klAuto`, `entropyAuto`, `crossEntropyAuto` (closed
 * form when registered, quadrature otherwise, with the method reported), `klMonteCarloWithError` and `klIntegrand`.
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
