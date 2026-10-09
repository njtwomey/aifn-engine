/**
 * `aifn-compute/foundation`: the true core: everything else in aifn is built on it, and it imports nothing else in
 * aifn. Its common surface (re-exported below) is also the package root (`aifn-compute`).
 *
 * - `tensor`: the n-dimensional array, its dtypes and constructors, and the primitives, each defined once with its
 *   derivative, batching and shape rules (as numpy's ndarray with JAX's primitives).
 * - `autodiff`: reverse- and forward-mode differentiation and batching (`grad`, `vjp`, `jvp`, `jacobian`, `hessian`,
 *   `vmap`) as interpreters over the primitives.
 * - `pytree`: nested arrays and plain objects of numbers, tensors and traced values, as the transforms take them.
 * - `random`: keyed, counter-based random streams (Philox4x32-10) as plain data, and samplers that take the stream
 *   first.
 * - `trace`: the runners (`run`, `trace`) that drive every iterative algorithm in aifn.
 * - `fourier` and `convolution`: discrete Fourier and cosine transforms (as numpy.fft), and differentiable
 *   convolution, correlation and filtering.
 * - `contracts`, `errors`, `registry` and `space`: the shared types, the error hierarchy, the pattern for named
 *   entries, and parameter spaces as data.
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
