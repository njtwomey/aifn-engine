/**
 * `aifn-compute/foundation/tensor`: the n-dimensional array everything numeric builds on, and the primitive machinery that defines each
 * operation once with its derivative rule.
 *
 * - Type and storage: `Tensor` (shape, strides, offset, dtype, typed-array data), `Vector`, `Matrix`, `DType`.
 * - Constructors: `tensor`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `fromRows`, `scalar`, `fromData`.
 * - Converters (the chart boundary): `toArray`, `toRows`, `toFlat`, `item`; complex: `toComplexFlat`,
 *   `toComplexArray`, `complexItem`.
 * - Dtypes (design K §3.2): `bool`, `int32`, `float32`, `float64`, `complex128` (interleaved), one promotion table
 *   (`promoteTypes`, `weakType`) and the result rules (`resultType`). Complex numbers: `complex`, `conj`, `realPart`,
 *   `imagPart` (zero-copy views), `angle`, `complexAbs`, `expj`; autodiff treats them as ℝ² pairs (complex.ts).
 * - Primitives (numbers, tensors and traced values alike): elementwise, structural, reductions and products.
 * - Defining primitives: `definePrimitive` and `elementwise`, registered under node-path ids (`numerics/special/erf`) in `registry`
 *   (`registry.list()`); `defineOp` is a thin wrapper over
 *   `definePrimitive`; `sumLike`; the transform hook: `apply`, `Tracer`, `Interpreter`, `avalOf`, `nextLevel`.
 * - Constants and tolerances: `EPS`, `SQRT_EPS`, `TINY`, `DEFAULT_TOLERANCE`. The errors are
 *   `aifn-compute/foundation/errors`.
 * - The brand: `isTensor` checks it; `revive` re-brands tensors that crossed `structuredClone` or a worker.
 * - Indexed reads and writes (primitives): `gather`, `scatterAdd`, `take` (rows along the first axis).
 * - The associative scan (`associativeScan`, a composition) and the Hillis–Steele and Blelloch scans as step-through
 *   algorithms (`hillisSteeleScanSteps`, `blellochScanSteps`, registered in `scanAlgorithms`).
 * - Inputs: `VectorLike` and `MatrixLike`, the one sanctioned relaxation of `Tensor` for data arguments.
 * - Dense kernels for inner loops, on row-major `Float64Array`s: the `dense` namespace (`dense.toF64`,
 *   `dense.toMatrixF64`, `dense.dot`, `dense.matVec`, `dense.matMul`, `dense.axpy`, …). Not primitives.
 */

export type { Axes, DType, NestedArray, Tensor, TensorData, TensorLike, Vector, Matrix } from './core'
export type { ComplexNumber } from 'aifn-compute/foundation/contracts'
export {
  elementWidth,
  isComplexDType,
  isFloatDType,
  promoteTypes,
  resultType,
  weakType,
  type ResultRule,
} from './dtype'
export { angle, complex, complexAbs, conj, expj, imagPart, realPart, refuseComplex } from './complex'
export { complexKernel, joinComplex, splitComplex, type ComplexRule } from './kernels'
export { complexPartView } from './core'
export {
  float64Data,
  fromData,
  isContiguous,
  readonlyData,
  isTensor,
  revive,
  rowMajorStrides,
  showShape,
  size,
  type TensorBrand,
} from './core'
export { DEFAULT_TOLERANCE, EPS, EPS32, SQRT_EPS, TINY, tolerance, type Tolerance } from './numerics'
export {
  registry,
  type Domain,
  type Draw,
  type DTypeRule,
  type Primitive,
  type PrimitiveCase,
  type PrimitiveDoc,
  type PrimitiveTest,
  type RuleSource,
} from './registry'
export {
  arange,
  astype,
  complexItem,
  copy,
  eye,
  fromRows,
  full,
  item,
  linspace,
  ones,
  scalar,
  shapeOf,
  tensor,
  toArray,
  toComplexArray,
  toComplexFlat,
  toFlat,
  toRows,
  zeros,
  type NestedComplex,
} from './create'
export { broadcastShapes, type SliceSpec } from './views'
export {
  apply,
  avalOf,
  batchedValueError,
  isTraced,
  nextLevel,
  Tracer,
  unwrap,
  type Aval,
  type Interpreter,
  type Traced,
  type Value,
} from './trace'
export {
  defineOp,
  definePrimitive,
  elementwise,
  type ElementwiseDerivative,
  type ElementwiseSpec,
  type PrimitiveMeta,
  type PrimitiveSpec,
  batchToFront,
  broadcastBatch,
  fitTo,
  projectReal,
  sumLike,
  zerosOf,
  type Binary,
  type NumberResult,
  type Op,
  type OpBatch,
  type OpJvp,
  type OpTranspose,
  type OpVjp,
  type Raw,
  type ShapeRule,
  type Result2,
  type TensorResult,
  type Ternary,
  type Unary,
} from './primitive'
export {
  abs,
  add,
  clip,
  cos,
  div,
  equalTo,
  exp,
  expm1,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  log1p,
  map,
  map2,
  maximum,
  minimum,
  mul,
  neg,
  notEqualTo,
  pow,
  sign,
  sin,
  sqrt,
  square,
  sub,
  tanh,
  where,
  type Comparison,
} from './elementwise'
export {
  batchByLoop,
  broadcastTo,
  concat,
  diag,
  diagonal,
  expandDims,
  flatten,
  get,
  permute,
  reshape,
  set,
  shapeOfValue,
  slice,
  squeeze,
  stack,
  sumTo,
  transpose,
} from './structure'
export {
  argmax,
  argmin,
  cumsum,
  logsumexp,
  max,
  mean,
  min,
  norm,
  prod,
  std,
  sum,
  variance,
  type ArgReduction,
  type Reduction,
} from './reduce'
export { dot, einsum, linearCombination, matmul, outer } from './products'
export { allclose, equal, type CloseOptions } from './compare'
export { gather, scatterAdd, take } from './gather'
export type { MatrixLike, VectorLike } from './dense'
export * as dense from './dense'
export { meshgrid, logspace } from './grids'
export {
  associativeScan,
  blellochScanSteps,
  hillisSteeleScanSteps,
  scanAlgorithms,
  type BlellochPhase,
  type BlellochState,
  type HillisSteeleState,
  type ScanElement,
  type ScanMove,
  type ScanOptions,
} from './scan'
