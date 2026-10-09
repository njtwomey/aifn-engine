/**
 * `aifn-compute/nn/quantise`: $b$-bit quantisation of tensors and weights, after torch.ao.quantization.
 *
 * - Quantisers: `quantisationParams` fits the scale $s$ and zero point $z$ by min–max calibration (affine or
 *   symmetric, per tensor or per channel); `quantise` maps to integers (round to nearest with halves to even, or
 *   stochastic rounding) and `dequantise` maps back, $\hat{x} = s(q - z)$. `roundHalfEven` and `integerRange` are the
 *   pieces they share.
 * - Measuring a quantiser: `quantisationError` gives the mean squared error, SQNR, clipped share and largest error.
 * - Quantisation-aware training: `fakeQuantise`, quantise and dequantise in the forward pass with the straight-through
 *   estimator as its gradient (a custom VJP).
 * - Integer inference: `quantisedMatmul`, an int8 matmul simulated exactly, with its accumulator.
 * - Post-training weight quantisation from calibration inputs: `gptqQuantise` (GPTQ's second-order error feedback) and
 *   `awqQuantise` (activation-aware channel scaling), each reported against round-to-nearest.
 *
 * Integers are stored as float64 values. Only `fakeQuantise` is differentiable; the rest take concrete tensors.
 */

export {
  awqQuantise,
  dequantise,
  fakeQuantise,
  gptqQuantise,
  integerRange,
  quantisationError,
  quantisationParams,
  quantise,
  quantisedMatmul,
  roundHalfEven,
  type AwqResult,
  type GptqResult,
  type QuantisationOptions,
  type QuantisationParams,
  type QuantisedMatmul,
  type RoundingOptions,
} from './quantise'
export { quantiseFunctions } from './registry'
