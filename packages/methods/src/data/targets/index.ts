/**
 * `aifn-methods/data/targets`: target log-densities for samplers and variational inference, with known moments.
 *
 * - Gaussian references: `gaussianTarget` (any mean and covariance, with its precision) and `gaussianMixtureTarget`
 *   (isotropic components with weights: several modes).
 * - Hard geometry: `banana` (a Gaussian bent into a crescent, with exact moments), `funnel` (Neal's funnel, whose scale
 *   varies over orders of magnitude) and `nonCentredFunnel` (the same in its non-centred parameterisation, an
 *   axis-aligned Gaussian, with the maps to and from the funnel).
 *
 * Each returns a normalised `LogDensity` whose `logDensity` is differentiable and whose `grad` is the closed form, with
 * the exact mean in `truth`. All are registered as log densities (kind `log-density`) and collected in
 * `logDensityRegistry`.
 */

export { banana, funnel, gaussianMixtureTarget, gaussianTarget, nonCentredFunnel } from './targets'
