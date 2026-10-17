/**
 * `aifn-methods/neural/convolutional`: worked forms of convolutional-network ideas, small enough to show step by step.
 *
 * - Fast convolution: `winogradTransforms` (Toom-Cook transforms of $F(m, r)$ from `winogradPoints`), `winogradTile`
 *   against `directTile`, `winogradRelativeError` for the float32 error as the tile grows, and
 *   `convolutionAlgorithmCosts`, a cost model of the direct sum, im2col, FFT and Winograd for one layer.
 * - Deformable convolution at one output position, tap by tap: `deformableConvolutionAt`, with each tap's bilinear
 *   sample and the gradient of the output in its offset.
 * - Capsules: `squash`, `capsulePredictions` and `dynamicRoutingTrace`, routing by agreement with every iteration
 *   recorded.
 *
 * Inputs and results are plain arrays and records. The differentiable operations these illustrate are in
 * `aifn-compute/nn/functional` (`im2col`, `deformableConv2d`, `conv2d`). `convolutionalFunctions` is the module's
 * registry.
 */

export {
  convolutionAlgorithmCosts,
  directTile,
  winogradPoints,
  winogradRelativeError,
  winogradTile,
  winogradTransforms,
  type AlgorithmCostSpec,
  type AlgorithmCosts,
  type Rows,
  type WinogradTransforms,
} from './fast'
export { deformableConvolutionAt, type DeformableOptions, type DeformableTap } from './deformable'
export { capsulePredictions, dynamicRoutingTrace, squash, type RoutingIteration } from './capsules'
export { convolutionalFunctions } from './registry'
