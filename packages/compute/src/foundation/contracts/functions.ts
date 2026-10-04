/**
 * Signature types of the function families that several modules implement: elementwise primitives, reductions,
 * samplers, kernel evaluations, metrics and losses. An implementation is checked against its family with `satisfies`.
 */

import type { Axes, Raw, Scalar, Tensor, Traced, Value } from './numbers'
import type { SampleOptions, Stream } from './random'

/** An elementwise function of one argument: numbers map to numbers, tensors to tensors, traced to traced. */
export interface Unary {
  (x: number): number
  (x: Tensor): Tensor
  (x: Traced): Traced
  (x: Value): Value
}

/** An elementwise function of two broadcast arguments. */
export interface Binary {
  (a: number, b: number): number
  (a: Tensor, b: Tensor | number): Tensor
  (a: number, b: Tensor): Tensor
  (a: Traced, b: Value): Traced
  (a: Value, b: Traced): Traced
  (a: Value, b: Value): Value
}

/**
 * A reduction along axes: every axis without `keepDims` gives a number, otherwise a tensor; traced inputs give traced
 * results. `E` lists extra trailing parameters (e.g. `ddof` for `variance`).
 */
export interface Reduction<E extends unknown[] = []> {
  (x: number | Tensor, axis?: null, keepDims?: false, ...extra: E): number
  (x: Tensor, axis: Axes, keepDims?: boolean, ...extra: E): Tensor
  (x: Tensor, axis: Axes | null | undefined, keepDims: true, ...extra: E): Tensor
  (x: Traced, axis?: Axes | null, keepDims?: boolean, ...extra: E): Traced
  (x: Value, axis?: Axes | null, keepDims?: boolean, ...extra: E): Value
}

/**
 * A sampler with parameters `P` (numbers or tensors, broadcast together): draws from the stream only, a number when
 * every parameter is a number and no shape is given, otherwise a tensor.
 */
export type Sampler<P extends readonly Raw[] = readonly Raw[]> = (
  s: Stream,
  ...args: [...P, options?: SampleOptions]
) => Raw

/** A kernel evaluation: the cross-covariance [n, m] of the rows of x [n, d] and y [m, d] (`null`: y is x). */
export type KernelFn = (x: Value, y: Value | null) => Value

/**
 * A metric: a function of its inputs (targets and predictions, in the order its `info.inputs` names) and options,
 * returning a number. Never differentiated.
 */
export type MetricFn<Inputs extends readonly unknown[] = never[]> = (...inputs: Inputs) => Scalar

/**
 * A loss: a differentiable function of predictions (and constant targets) returning a number or per-example values,
 * written with primitives so `grad` works.
 */
export type LossFn<Inputs extends readonly unknown[] = never[]> = (...inputs: Inputs) => Value

/**
 * How per-example losses are combined: `mean` (the default, as in PyTorch), `sum`, or `none` (the per-example values).
 */
export type ReductionMode = 'mean' | 'sum' | 'none'
