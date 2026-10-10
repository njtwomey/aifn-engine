/**
 * `aifn-compute/signal/multirate`: sample-rate change and filter banks, as scipy.signal.
 *
 * - Changing the rate: `resamplePoly` (by a rational factor $u/d$, with a polyphase FIR, as `resample_poly`),
 *   `decimateSignal` (by an integer $q$ after an anti-aliasing lowpass, as `decimate`) and `upfirdn` (upsample, FIR
 *   filter, downsample; `aifn-compute/foundation/convolution`'s).
 * - Filter banks: `polyphase` (the $M$ polyphase components of a filter) and `dftFilterBank` (the uniform DFT analysis
 *   filter bank, by its polyphase implementation).
 * - Reconstruction: `sincInterpolate` (Whittaker–Shannon, at arbitrary times).
 *
 * The functions take a single-channel `Signal` or bare samples, and the resamplers return a `Signal` at the new rate
 * with the input's start time. `multirateFunctions` lists them in the registry.
 */

export { upfirdn } from 'aifn-compute/foundation/convolution'
export {
  decimateSignal,
  dftFilterBank,
  polyphase,
  resamplePoly,
  sincInterpolate,
  type DecimateOptions,
  type ResamplePolyOptions,
} from './multirate'
export { multirateFunctions } from './registry'
