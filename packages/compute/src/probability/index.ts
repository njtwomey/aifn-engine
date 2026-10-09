/**
 * `aifn-compute/probability`: probability and statistics, as torch.distributions and scipy.stats.
 *
 * Its child modules:
 *
 * - `stats`: descriptive statistics, quantiles, ranks and correlations, histograms, kernel density estimates,
 *   resampling and power transforms, on arrays and tensors.
 * - `tests`: hypothesis tests on one result protocol, with intervals, effect sizes, power, multiple testing,
 *   sequential tests and survival estimators.
 * - `distributions`: distributions as objects with log-densities, cdfs, quantiles, moments and samplers, transformed
 *   and composed ones, and Kullback–Leibler divergences.
 * - `samplers`: draws from the named families that need special functions or factorisations (gamma, beta, Poisson,
 *   the multivariate normal, ...).
 * - `bijectors`: invertible maps of the real line with their log-Jacobians, for transformed distributions,
 *   constrained parameters and normalising flows, and log-densities reparameterised through them.
 * - `likelihoods`: links and exponential-dispersion families for generalised linear models, distributional (GAMLSS)
 *   and ordinal likelihoods.
 * - `information`: entropies, divergences and mutual information, from probability tables, distributions and samples.
 * - `extremes`: peaks over threshold, with the generalised Pareto fit and tail probabilities and quantiles.
 * - `markov`: finite Markov chains: classification of states, stationary distributions, hitting and mixing times.
 * - `privacy`: differential privacy: mechanisms, their calibration, privacy accounting and DP-SGD's gradient
 *   aggregation.
 *
 * The family's index re-exports the most used names (`Normal`, `kl`, `quantile`, `histogram`, `entropy`,
 * `mutualInformation`, ...); everything else is imported from its module.
 */

export { Normal, Bernoulli, Categorical, Beta, Gamma, Poisson, MultivariateNormal, kl } from './distributions'
export { quantile, histogram, kde } from './stats'
export { gammaVariate, beta, poisson, multivariateNormal } from './samplers'
export { entropy, mutualInformation } from './information'
