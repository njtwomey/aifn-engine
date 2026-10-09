/**
 * `aifn-compute/nn/init`: weight initialisers for layers' parameters, after torch.nn.init.
 *
 * - Scaled by the fans, for the activation that follows: `xavierUniform` and `xavierNormal` (tanh-like units),
 *   `heUniform` and `heNormal` (ReLU and leaky ReLU, by `fanIn` or `fanOut`), and `lecunUniform` (PyTorch's default
 *   for linear and convolution weights).
 * - Fixed: `normalInit` (a given standard deviation) and `zerosInit` (biases).
 *
 * Each returns an `Initialiser`, a function of a stream, a shape and the parameter's `Fans`, which the layer supplies;
 * the same stream gives the same weights.
 */

export {
  heNormal,
  heUniform,
  lecunUniform,
  normalInit,
  xavierNormal,
  xavierUniform,
  zerosInit,
  type Fans,
  type HeOptions,
  type Initialiser,
} from './init'
