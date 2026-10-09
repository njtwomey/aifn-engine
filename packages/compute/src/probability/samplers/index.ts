/**
 * `aifn-compute/probability/samplers`: draws from named families that need special functions or factorisations
 * (decision D2).
 *
 * - Gamma and its relatives, by Marsaglia and Tsang's rejection method: `gammaVariate`, `logGammaVariate` (in log
 *   space, for tiny shapes that would underflow to 0), `beta`, `chiSquare` and `studentT` (location and scale;
 *   $\nu = \infty$ gives a normal).
 * - Counts: `poisson` (inversion below $\lambda = 10$, Hörmann's PTRS above) and `binomial` (inversion, or an exact
 *   order-statistic recursion for large $n$).
 * - Vector draws, whose last axis is the event: `dirichlet` (on the simplex), `multinomial` (counts summing to $n$)
 *   and `multivariateNormal` ($\muvec + \Lmat\zvec$ from a covariance or a Cholesky factor).
 * - `samplerFunctions`: the samplers registered as functions that draw, keyed by name.
 *
 * Each takes the stream first, like the draws of `aifn-compute/foundation/random`, and keys its draws so that one
 * element's rejection trials never move another element's draws. Invalid parameters give NaN rather than an error
 * (a covariance that does not factor is the exception). None is differentiable: pathwise draws are the `rsample`
 * methods of `aifn-compute/probability/distributions`.
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
