/**
 * `aifn-compute/inference/variational`: variational inference by stochastic optimisation of the ELBO.
 *
 * - Families: `meanFieldGaussian` (diagonal, $2d$ parameters) and `fullRankGaussian` (a lower-triangular scale
 *   $\Lmat$, $d + d(d + 1)/2$ parameters), each a flat parameter vector $\lambdavec$ with its moments, density, entropy
 *   and reparameterisation $\xvec = \muvec + \Lmat\epsilonvec$.
 * - Estimators: `elbo` (the bound with its standard error), `elboGradient` (reparameterisation, or the score function
 *   with a leave-one-out or control-variate baseline; all unbiased) and `gradientVariance` (an estimator's spread
 *   over repeats, to compare them).
 * - Optimisation: `bbvi`, a step-through algorithm running Adam on the gradient estimates.
 *
 * The target is any `LogDensity` known up to a constant; what is minimised is the reverse $\KL(q \,\|\, p)$, which
 * under-covers $p$. Every estimate draws from the stream it is given, so a seeded stream reproduces it.
 */

export { fullRankGaussian, meanFieldGaussian, type FamilyKernels, type GaussianFamily, type VectorLike } from './family'
export {
  elbo,
  elboGradient,
  gradientVariance,
  type Baseline,
  type ElboEstimate,
  type ElboGradient,
  type ElboGradientOptions,
  type GradientEstimator,
  type GradientVariance,
} from './elbo'
export { bbvi, type BbviOptions, type BbviStart, type BbviState } from './bbvi'
export { variationalAlgorithms, variationalFunctions } from './registry'
