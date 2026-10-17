/**
 * `aifn-methods/vision`: worked computer-vision setups with known ground truth, on the geometry and robust-fitting
 * modules of `aifn-compute` (`aifn-compute/numerics/geometry`, `aifn-compute/numerics/robust`).
 *
 * - `vision/two-view`: two-view geometry. A synthetic scene of two pinhole cameras with noisy correspondences and
 *   outliers, its true homography and fundamental matrix, and the RANSAC problems that estimate them from the
 *   correspondences (as in Hartley and Zisserman, "Multiple View Geometry", chapters 4, 9 and 11).
 * - `vision/detection`: the box geometry of dense detectors: anchor grids, box offsets, anchor matching by IoU, FCOS
 *   targets and feature-pyramid levels, on the boxes of `aifn-methods/evaluation/detection`.
 */

export * from './two-view'
export * from './detection'
