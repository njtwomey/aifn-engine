/**
 * `aifn-methods/theory/optimism`: the optimism of the training error by simulation, the gap between in-sample and
 * training error, against the covariance penalty $\frac{2}{n}\sum_i \cov(\hat y_i, y_i) = 2\sigma^2\mathrm{df}/n$.
 *
 * - Studies: `optimismStudy` refits one model to many noisy samples on a fixed or random design and returns the mean
 *   training, in-sample and extra-sample errors, the optimism measured directly and through the covariances, the
 *   effective degrees of freedom and the degrees of freedom theory gives (`OptimismOptions`, `OptimismResult`);
 *   `optimismSweep` runs one study per polynomial degree, $k$, ridge penalty or number of candidates.
 * - Models (`OptimismModel`): polynomial least squares or ridge and $k$-nearest neighbours, linear smoothers with
 *   $\mathrm{df} = \trace\Smat$; and the best of $M$ pure-noise features, whose search makes its $\mathrm{df}$ larger
 *   than its one coefficient (Ye's generalised degrees of freedom).
 *
 * Studies draw their designs and noise from child streams, so the same stream gives every model the same samples.
 * Options and results are plain data, so the studies run in a compute worker. `optimismFunctions` registers them.
 */

export {
  optimismStudy,
  optimismSweep,
  type OptimismModel,
  type OptimismOptions,
  type OptimismResult,
  type OptimismSweep,
  type OptimismTarget,
} from './optimism'
export { optimismFunctions } from './registry'
