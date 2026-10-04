/**
 * Numbers: the value types every signature in aifn is written in, and the input aliases that relax them.
 *
 * Mathematical values are `Scalar` (or `Scalar | Tensor`); integer metadata (counts, positions, axes, shapes) has its
 * own names, so a signature never confuses a value with a count: `erf(x: Scalar | Tensor)`, `zeros(shape: Shape)`,
 * `sum(x, axis?: Axes)`. Every name here is a type; nothing is emitted at run time.
 */

// ── Scalars and integer metadata ─────────────────────────────────────────────────────────────────────────────────────

/** A real number used as a mathematical value (a parameter, an observation, a result). An alias of `number`. */
export type Scalar = number

/** A non-negative integer count or length: a number of elements, rows, steps or draws. */
export type Size = number

/** A zero-based integer position along an axis or in a list. Negative indices count from the end where documented. */
export type Index = number

/** One axis of a tensor, zero-based; negative axes count from the end (−1 is the last). */
export type Axis = number

/** One axis or several (duplicates are an error); functions that reduce read `null` or omission as every axis. */
export type Axes = Axis | readonly Axis[]

/** The length of each axis of a tensor; `[]` for a scalar tensor. */
export type Shape = readonly Size[]

// ── Tensor ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Element type of a tensor's storage (design K §3.1): `bool` (Uint8Array, 0 or 1) for masks and comparison results,
 * `int32`, `float32` (storage only), `float64`, and `complex128`, stored interleaved (re, im, re, im, …) in a
 * Float64Array. Mixing dtypes promotes by one table (`aifn-compute/foundation/tensor`'s `promoteTypes`).
 */
export type DType = 'bool' | 'int32' | 'float32' | 'float64' | 'complex128'

/** The typed array that backs a tensor: Float64Array for float64 and complex128 (two per element), Uint8Array for bool. */
export type TensorData = Float64Array | Float32Array | Int32Array | Uint8Array

/** A complex number as plain data: its real and imaginary parts (what the complex converters return). */
export interface ComplexNumber {
  readonly re: number
  readonly im: number
}

declare const tensorBrand: unique symbol

/**
 * The type of the symbol that brands a tensor. The runtime symbol is `aifn-compute/foundation/tensor`'s `TENSOR`
 * (`Symbol.for('aifn.tensor')`), set only by its constructors and typed as this, so this `Tensor` and `aifn-compute/foundation/tensor`'s
 * are one type.
 */
export type TensorBrand = typeof tensorBrand

/**
 * An n-dimensional array: `shape[k]` elements along axis k, stored in `data` at `offset + Σ_k index[k] · strides[k]`
 * (NumPy's strided layout). Immutable by convention: operations return new tensors, and views share `data`. Only
 * `aifn-compute/foundation/tensor` constructs tensors.
 */
export interface Tensor {
  /** The brand set by `aifn-compute/foundation/tensor`'s constructors; `isTensor` checks it (no duck typing). */
  readonly [tensorBrand]: true
  /** Length of each axis; `[]` for a scalar tensor. */
  readonly shape: Shape
  /**
   * Step in `data`, in elements, for a unit step along each axis. Row-major (C order) by default. For complex128 an
   * element is a complex number (two doubles): element k starts at `data[2k]`.
   */
  readonly strides: readonly number[]
  /** Position in `data` of the element at index (0, …, 0), in elements (complex elements for complex128). */
  readonly offset: Index
  readonly dtype: DType
  readonly data: TensorData
}

/** A rank-1 tensor (documentation only: the rank is checked at run time). */
export type Vector = Tensor

/** A rank-2 tensor (documentation only: the rank is checked at run time). */
export type Matrix = Tensor

/**
 * A tensor on the wire (JSON, fixtures, aifn-py): dtype, shape and row-major data, with non-finite values written as
 * the strings `"nan"`, `"inf"` and `"-inf"`.
 */
export interface TensorWire {
  readonly dtype: DType
  readonly shape: Shape
  /** Row-major values; complex128 is interleaved (re, im, re, im, …), twice as many entries as elements. */
  readonly data: readonly (number | 'nan' | 'inf' | '-inf')[]
}

// ── Traced values ────────────────────────────────────────────────────────────────────────────────────────────────────

declare const tracedBrand: unique symbol

/**
 * The type of the symbol that marks a traced value. The runtime symbol is `Symbol.for('aifn.traced')`, created in
 * `aifn-compute/foundation/tensor` and typed as this.
 */
export type TracedBrand = typeof tracedBrand

/**
 * An abstract value: what is known about a value without its data (JAX's `ShapedArray`). `number` is true when the
 * value is a JS number rather than a tensor (a number has shape `[]`, but so does a rank-0 tensor).
 */
export interface Aval {
  readonly shape: Shape
  readonly dtype: DType
  readonly number: boolean
}

/**
 * A value seen by a function transform (design K §4.1): a tracer of one interpreter (reverse, forward or batch), at
 * that interpreter's `level`. Transforms nest by level: a primitive applied to tracers of several levels is handled by
 * the highest, which treats the others as constants. `aval` is the shape and dtype of the value the traced function
 * sees (for a batch tracer, one example's).
 */
export interface Traced {
  readonly [tracedBrand]: true
  readonly level: Index
  readonly aval: Aval
}

/** An untraced value: a number or a tensor. What a primitive computes on. */
export type Raw = Scalar | Tensor

/** Anything a primitive accepts: a number, a tensor, or a traced value. */
export type Value = Scalar | Tensor | Traced

// ── Inputs ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A number or a tensor, as elementwise operations accept: a number broadcasts as a scalar. The same type as `Raw`,
 * named for inputs.
 */
export type TensorLike = Raw

/** Nested arrays of numbers, as `tensor` accepts and `toArray` returns. */
export type NestedArray = Scalar | NestedArray[]

/**
 * A vector argument: a rank-1 tensor (any strides) or a plain or typed array of numbers. Always copied on the way in.
 * Accepted wherever a function reads one vector of data (a point, coefficients, weights, a signal).
 */
export type VectorLike = Tensor | ArrayLike<Scalar>

/**
 * A matrix argument: a rank-2 tensor or rows of numbers (all of one length). Always copied on the way in. Accepted
 * wherever a function reads one matrix (a system matrix, a design matrix, rows of points).
 */
export type MatrixLike = Tensor | ArrayLike<ArrayLike<Scalar>>

/**
 * Numeric data read flat: a tensor of any rank (read row-major) or an array of numbers. Accepted by statistics and
 * metrics, which treat their input as a sample rather than as a vector.
 */
export type DataLike = ArrayLike<Scalar> | Tensor
