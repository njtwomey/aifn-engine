/**
 * `aifn-compute/nn/quantise`: b-bit quantisation of tensors after torch.ao.quantization: `quantisationParams` (affine or
 * symmetric, per tensor or per channel, min–max calibration), `quantise` (round to nearest, halves to even, or
 * stochastic rounding), `dequantise`, `fakeQuantise` (the straight-through estimator, a custom VJP), `quantisationError`,
 * `quantisedMatmul` (an exact int8 matmul simulation), `gptqQuantise` (GPTQ's second-order error feedback), `awqQuantise` (activation-aware channel scaling),
 * `roundHalfEven` and `integerRange`.
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
