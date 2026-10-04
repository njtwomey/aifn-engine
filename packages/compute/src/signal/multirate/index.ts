/**
 * `aifn-compute/signal/multirate`: sample-rate change and filter banks, as scipy.signal: `resamplePoly` (rational resampling
 * by a polyphase FIR), `decimateSignal` (scipy's `decimate`: anti-aliasing filter, then every q-th sample), `upfirdn`
 * (upsample, FIR filter, downsample; `aifn-compute/foundation/convolution`'s), `polyphase` (the polyphase components of a
 * filter), `dftFilterBank` (the uniform DFT analysis filter bank, by its polyphase implementation) and
 * `sincInterpolate` (Whittaker–Shannon reconstruction).
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
