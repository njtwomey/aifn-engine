/**
 * `aifn-compute`: the package root, foundation's common surface (decision D1), so that a learner writes
 * `import { tensor, grad, stream, trace } from 'aifn-compute'` as they write `import torch`.
 *
 * - Tensors: the whole of `aifn-compute/foundation/tensor` (constructors, elementwise and structural primitives, reductions,
 *   products, converters, dtypes) and the errors of `aifn-compute/foundation/errors`.
 * - Differentiation: `grad`, `valueAndGrad`, `vjp`, `jvp`, `jacobian`, `hessian`, `hvp`, `stopGradient`, `gradCheck`.
 * - Randomness: `stream` and `Stream`, the draws `uniform`, `normal`, `normals`, `bernoulli`, `categorical`,
 *   `shuffle`, `permutation`, `choice`.
 * - Iteration: the `Algorithm` and `Trace` types and the runners `run`, `trace`, `live`, `seek`, `profile`.
 *
 * Code inside aifn never imports this root: it imports the node that defines each name (the lint checks it).
 */

export * from 'aifn-compute/foundation/tensor'
export * from 'aifn-compute/foundation/errors'
export {
  grad,
  gradCheck,
  hessian,
  hvp,
  jacobian,
  jvp,
  stopGradient,
  valueAndGrad,
  vjp,
} from 'aifn-compute/foundation/autodiff'
export {
  bernoulli,
  categorical,
  choice,
  normal,
  normals,
  permutation,
  shuffle,
  stream,
  uniform,
  type Stream,
} from 'aifn-compute/foundation/random'
export { live, profile, run, seek, trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
