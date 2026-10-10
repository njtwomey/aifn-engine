/**
 * `aifn-methods/vision/two-view`: a synthetic two-view scene with known ground truth, and the RANSAC problems that
 * recover its geometry.
 *
 * - The scene: `twoViewScene`, two pinhole cameras looking at points on a plane or spread in depth, noisy
 *   correspondences with a fraction of outliers, and the true homography $\Hmat$ (planar scenes only) and fundamental
 *   matrix $\Fmat$ to compare an estimate with.
 * - The problems, for `ransac` and `ransacFit` of `aifn-compute/numerics/robust`: `homographyProblem` (samples of 4,
 *   scored by transfer error) and `fundamentalProblem` (samples of 8, scored by the root Sampson distance).
 * - The registry: `twoViewFunctions`.
 *
 * Conventions: image points are $n \times 2$ matrices in pixels, row $i$ of view 1 matching row $i$ of view 2, with
 * $\xvec_2^\top\Fmat\xvec_1 = 0$ and $\xvec_2 \propto \Hmat\xvec_1$ for the homogeneous points; residuals are in
 * pixels.
 */

export { fundamentalProblem, homographyProblem, twoViewScene, type TwoViewOptions, type TwoViewScene } from './scene'
export { twoViewFunctions } from './registry'
