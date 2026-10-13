/**
 * `aifn-compute/numerics/regularisation`: regularisation of ill-posed linear least-squares problems
 * $\Amat\xvec \approx \bvec$, after Hansen's "Rank-Deficient and Discrete Ill-Posed Problems" (1998) and his
 * Regularization Tools.
 *
 * - Tikhonov regularisation, minimising $\lVert \Amat\xvec - \bvec \rVert^2 + \lambda \lVert \Lmat\xvec \rVert^2$:
 *   `tikhonov` (standard form by the SVD, or a general $\Lmat$ by a QR factorisation of the stacked system),
 *   `tikhonovPath` (many weights from one SVD; a general $\Lmat$ brought to standard form once, by Eldén's
 *   transformation, so the filter factors are those of the generalised singular values), `filterFactors`
 *   ($s_i^2/(s_i^2 + \lambda)$) and `differenceOperator` (an $\Lmat$ that penalises roughness).
 * - Test problems: `convolutionMatrix`, the matrix of a one-dimensional blur with zero, periodic or reflected ends.
 * - Choosing the weight: `lCurve` (the L-curve, its curvature in closed form and its corner) and
 *   `discrepancyPrinciple` (Morozov: the residual matched to the noise level, by bisection).
 * - Regularisation by early stopping: `landweberSteps`, Landweber iteration as a step-through algorithm, with the
 *   filter factors $1 - (1 - \omega s_i^2)^k$ of its iterates to set against Tikhonov's.
 * - `regularisationAlgorithms` and `regularisationFunctions` register the methods for generic views and workers.
 *
 * The weight $\lambda$ multiplies the squared norm, so standard-form Tikhonov is ridge regression with `l2` $=\lambda$
 * and no intercept (`linearRegression` of `aifn-methods/learning/linear`). Matrices and vectors are read, never
 * modified; bad weights and shapes throw `DomainError` and `ShapeError`, and divergence is reported, not thrown.
 */

export {
  convolutionMatrix,
  differenceOperator,
  filterFactors,
  tikhonov,
  tikhonovPath,
  type TikhonovOptions,
  type TikhonovPath,
  type TikhonovPathOptions,
  type TikhonovResult,
} from './tikhonov'
export { discrepancyPrinciple, lCurve, type Discrepancy, type DiscrepancyOptions, type LCurve } from './selection'
export { landweberSteps, type LandweberProblem, type LandweberState } from './landweber'
export { regularisationAlgorithms, regularisationFunctions } from './registry'
