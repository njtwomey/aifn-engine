/**
 * `aifn-compute/learning/kernels`: covariance kernels $k(\xvec, \xvec')$, their Gram matrices, and their
 * hyperparameters in log space for fitting.
 *
 * - Stationary kernels, functions of $r = \norm{\xvec - \xvec'} / \ell$: `rbf`, `matern` with
 *   $\nu \in \{\tfrac{1}{2}, \tfrac{3}{2}, \tfrac{5}{2}\}$ (also `matern12`, `matern32`, `matern52`),
 *   `rationalQuadratic` and `periodic`; `white` noise and the `constant` offset. A vector lengthscale gives automatic
 *   relevance determination (ARD) in every kernel with a lengthscale except `periodic`.
 * - Dot-product kernels: `linearKernel` (Bayesian linear regression) and `polynomial`.
 * - Combinations: `sumKernel` and `productKernel`.
 * - Evaluation: `gram(k, x, y?)` for the Gram matrix (with `y` left out, `white` noise lands on its diagonal),
 *   `kernelDiagonal(k, x)` for its diagonal alone, and `kernelProfile(k, lags)` for $k(\tau, 0)$ in one dimension.
 * - Fitting: `logParams(k)` and `kernelFromLog(k, logParams)` move the hyperparameter tree `k.params` to and from log
 *   space, where it is unconstrained.
 * - Registries: `kernelRegistry`, every kernel factory with its hyperparameter `Space` and stationarity, and
 *   `kernelsFunctions`, the module's other functions.
 * - Helpers: `asRows`, `scaledSquaredDistances`, `scaledDistances`.
 *
 * Inputs are `[n, d]` matrices of $n$ points, or `[n]` vectors of one-dimensional points. Kernels are immutable plain
 * objects (`Kernel` from `aifn-compute/foundation/contracts`) whose hyperparameters are a pytree of values; every
 * kernel is built from `aifn-compute/foundation/tensor` primitives, so Gram matrices are differentiable in the inputs
 * and in the hyperparameters (`k.withParams(traced)`), as `aifn-methods/learning/gp` uses for its marginal
 * likelihood's gradient. The forms and defaults follow Rasmussen and Williams (2006) and scikit-learn.
 */

export {
  asRows,
  constant,
  gram,
  kernelDiagonal,
  kernelFromLog,
  kernelProfile,
  linearKernel,
  logParams,
  matern,
  matern12,
  matern32,
  matern52,
  periodic,
  polynomial,
  productKernel,
  rationalQuadratic,
  rbf,
  scaledDistances,
  scaledSquaredDistances,
  sumKernel,
  white,
  type CombinedParams,
  type DotProductParams,
  type Kernel,
  type KernelParams,
  type LinearParams,
  type MaternNu,
  type PeriodicParams,
  type RationalQuadraticParams,
  type StationaryParams,
  type VarianceParams,
} from './kernels'
export { kernelRegistry, kernelsFunctions } from './registry'
