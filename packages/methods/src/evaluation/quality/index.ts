/**
 * `aifn-methods/evaluation/quality`: signal and image quality metrics (PSNR, SSIM, SI-SDR, spectral angle).
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
