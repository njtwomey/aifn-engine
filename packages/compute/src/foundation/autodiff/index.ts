/**
 * `aifn-compute/foundation/autodiff`: automatic differentiation and batching as interpreters over aifn's primitives (design K
 * §4.1).
 *
 * Every operation in aifn is a primitive with its rules (`aifn-compute/foundation/tensor`'s `definePrimitive`, `elementwise`,
 * `defineOp`). This module supplies three interpreters of those primitives and the transforms built on them; it
 * defines no operations of its own (apart from `stopGradient`). A function transformed here is ordinary code over
 * numbers and tensors: the same `softplus`, `matmul` or `cholesky` that computes values is traced when given a tracer.
 *
 * - Interpreters: `ReverseInterpreter` (records, then a backward sweep), `ForwardInterpreter` (dual numbers),
 *   `BatchInterpreter` (a batch axis). They nest by level, as in JAX.
 * - Transforms: `grad`, `valueAndGrad` (with `argnums`), `vjp`, `jvp`, `linearize`, `hvp`, `jacobian` (forward or
 *   reverse), `hessian`, `vmap`, `stopGradient`. Arguments and results may be pytrees (nested arrays and plain objects).
 *   Transforms nest in any order: `grad(grad(f))`, `jvp(grad(f))`, `vmap(grad(f))`.
 * - Custom rules: `customVjp`, `customJvp` (composite functions with a derivative of the author's choosing),
 *   `defineCustomVjp` (customVjp with a name, a forward rule and residuals from the output), and `checkpoint`
 *   (recompute instead of store). Implicit differentiation (`implicitFixedPoint`, `implicitRoot`, `atConvergence`)
 *   needs a linear solve, so it lives in `aifn-compute/numerics/implicit`; `unrolled` is in `aifn-compute/foundation/trace`.
 * - Checking: `gradCheck(f, x)` compares gradients with central finite differences and reports every element.
 * - Inspection: `traceGraph(f, x)` returns the recorded nodes with values and adjoints in topological order.
 */

export { gradCheck, type GradCheckEntry, type GradCheckOptions, type GradCheckReport } from './check'
export { traceGraph, type Graph, type GraphInput, type GraphNode } from './graph'
export { NotDifferentiableError } from 'aifn-compute/foundation/errors'
export { BatchInterpreter, BatchTracer, batchExamples } from './batch'
export {
  checkpoint,
  customJvp,
  customVjp,
  defineCustomVjp,
  refuseTraced,
  type Cotangents,
  type CustomVjpSpec,
  type VjpForward,
} from './custom'
export { ForwardInterpreter, ForwardTracer } from './forward'
export { ReverseInterpreter, ReverseTracer, type Backward, type TapeRecord } from './reverse'
export {
  grad,
  hessian,
  hvp,
  jacobian,
  jvp,
  linearize,
  stopGradient,
  valueAndGrad,
  vjp,
  vmap,
  type GradOptions,
  type JacobianOptions,
  type JvpResult,
  type Lifted,
  type Linearized,
  type TreeOf,
  type ValueAndGrad,
  type VjpResult,
  type VmapOptions,
} from './transforms'
export { autodiffFunctions } from './registry'
