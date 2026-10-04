/**
 * `aifn-compute/foundation`: the true core: everything else in aifn is built on it, and it imports nothing else in aifn. Its common surface is
 * also the package root (`aifn-compute`). Children: contracts, errors, registry, tensor, pytree, fourier, convolution,
 * autodiff, random, trace.
 */

export {
  tensor,
  zeros,
  ones,
  eye,
  arange,
  linspace,
  fromRows,
  toArray,
  toFlat,
  toRows,
  type Tensor,
  type Value,
} from './tensor'
export { grad, valueAndGrad, vjp, jvp, jacobian, hessian } from './autodiff'
export { stream, type Stream } from './random'
export { run, trace, type Algorithm, type Trace } from './trace'
export { fft, ifft, rfft } from './fourier'
export { convolve, correlate } from './convolution'
