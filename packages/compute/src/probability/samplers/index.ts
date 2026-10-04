/**
 * `aifn-compute/probability/samplers`: draws from named families that need special functions or factorisations (decision D2):
 * gamma, log-gamma, beta, chi-square, Student t, Dirichlet, Poisson, binomial, multinomial and the multivariate
 * normal. Each takes the stream first, like the draws of `aifn-compute/foundation/random`.
 */
export {
  logGammaVariate,
  gammaVariate,
  beta,
  chiSquare,
  studentT,
  dirichlet,
  poisson,
  binomial,
  multinomial,
  multivariateNormal,
} from './samplers'
export { samplerFunctions } from './registry'
