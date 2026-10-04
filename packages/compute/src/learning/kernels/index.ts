/**
 * `aifn-compute/learning/kernels`: covariance kernels by lengthscale ℓ, their Gram matrices, and hyperparameter vectors for fitting.
 *
 * ```ts
 * const k = sumKernel(rbf({ lengthscale: 0.3, variance: 2 }), white({ variance: 0.01 }))
 * gram(k, x) // [n, n], with the white noise on the diagonal
 * gram(k, x, xs) // [n, m]
 * grad((p) => sum(gram(rbf(p), x)))({ lengthscale: 0.3, variance: 1 }) // gradients in the hyperparameters
 * ```
 *
 * - Kernels: `rbf`, `matern` (ν = ½, 3⁄2, 5⁄2; `matern12`, `matern32`, `matern52`), `rationalQuadratic`, `periodic`,
 *   `linearKernel`, `polynomial`, `white`, `constant`; `sumKernel`, `productKernel`. A vector lengthscale [d] gives automatic
 *   relevance determination (ARD) in every stationary kernel except `periodic`.
 * - Evaluation: `gram(k, X, Y?)`, `kernelDiagonal(k, X)`, `kernelProfile(k, lags)` (k(τ, 0) in one dimension). Inputs
 *   are [n, d] or [n] (one-dimensional). Everything is built from `aifn-compute/foundation/tensor` primitives, so Gram matrices are
 *   differentiable in the inputs and in hyperparameters (`k.withParams(traced)`), e.g. for `aifn-methods/learning/gp`'s
 *   marginal likelihood gradient.
 * - Hyperparameters are a pytree (`k.params`, walked by `aifn-compute/foundation/pytree`); `logParams(k)` and
 *   `kernelFromLog(k, θ)` move them to and from log space for fitting.
 * - `kernelRegistry`: every kernel factory with its hyperparameter `Space` and stationarity.
 * - Helpers: `asRows`, `scaledSquaredDistances`, `scaledDistances`.
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
