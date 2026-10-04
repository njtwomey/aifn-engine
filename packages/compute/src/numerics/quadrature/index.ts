/**
 * `aifn-compute/numerics/quadrature`: numerical integration with error estimates.
 *
 * - Fixed rules: `trapezoid`, `simpson` (composite, equal panels), `trapezoidSamples` (numpy's `trapezoid`),
 *   `integrateGauss`.
 * - Gaussian rules (nodes and weights, ascending): `gaussLegendre`, `gaussHermite` (physicists' or probabilists'),
 *   `gaussLaguerre` (generalised); `normalExpectation` for $\mathbb{E}[f(X)]$, $X \sim \mathcal{N}(\mu, \sigma^2)$.
 * - Traceable adaptive methods: `romberg`, `adaptiveSimpson`, `gaussKronrod` (7–15, globally adaptive, as QUADPACK's
 *   QAG); `kronrod15` for one interval; `integrate` (like scipy's `quad`, infinite limits allowed; Gauss–Kronrod or
 *   Romberg).
 * - Several dimensions: `productRule`, `integrate2d`; `monteCarlo` (traceable, with standard errors) and
 *   `integrateMonteCarlo(s, …)`; `halton`, `sobol` sequences and randomised `quasiMonteCarlo(s, …)` (stream first).
 */

export { romberg, simpson, trapezoid, trapezoidSamples, type Integrand, type RombergState } from './rules'
export {
  gaussHermite,
  gaussLaguerre,
  gaussLegendre,
  integrateGauss,
  normalExpectation,
  type QuadratureRule,
} from './gauss'
export {
  adaptiveSimpson,
  gaussKronrod,
  integrate,
  kronrod15,
  type AdaptiveSimpsonState,
  type GaussKronrodState,
  type IntegrateOptions,
  type IntegrationResult,
  type Interval,
} from './adaptive'
export {
  halton,
  integrate2d,
  integrateMonteCarlo,
  monteCarlo,
  productRule,
  quasiMonteCarlo,
  sobol,
  type MonteCarloOptions,
  type MonteCarloResult,
  type MonteCarloState,
  type MultivariateIntegrand,
} from './multivariate'
export { quadratureAlgorithms, quadratureFunctions } from './registry'
