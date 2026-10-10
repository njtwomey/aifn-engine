/**
 * `aifn-methods/evaluation/quality`: signal and image quality metrics (PSNR, SSIM, SI-SDR, spectral angle), each
 * comparing a reference with an estimate of it.
 *
 * - Images: `psnr` from the mean squared error, and `ssim` of local means, variances and covariance, with `ssimMap`
 *   for the local values (`SsimOptions`: a uniform or Gaussian window).
 * - Multispectral images (pixels as rows, bands as columns): `spectralAngle` between spectra, which ignores gain, and
 *   the relative errors `ergas` and `rase`.
 * - Audio: `snr`, and `siSdr`, which ignores gain on the estimate; `permutationInvariantScore` scores separated
 *   sources under the best assignment to the references.
 *
 * An `Image` is a matrix or flat row-major data; ratios are in decibels. The metrics are collected in
 * `evaluationMetricRegistry` of `aifn-methods/evaluation`, and `qualityFunctions` registers the other functions.
 */

export {
  ergas,
  permutationInvariantScore,
  psnr,
  rase,
  siSdr,
  snr,
  spectralAngle,
  ssim,
  ssimMap,
  type Image,
  type SsimOptions,
} from './signal'
export { qualityFunctions } from './registry'
