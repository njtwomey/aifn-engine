/**
 * The shared layer of `aifn-compute/learning/losses`: the `Loss` type and its metadata, the reduction over examples,
 * and the conversions that read targets (labels, relevance grades) into constant tensors. `index.ts` re-exports the
 * types, `defineLoss`, `isLoss` and `oneHot`, and, for the ranking and retrieval losses of `aifn-methods/retrieval`,
 * `reduce`, `constant` (as `constantTarget`), `flatValues` and `expectRank`.
 *
 * The losses of the module share these conventions: predictions are traced or plain values, targets are constants
 * (never differentiated), and most end with an options object whose `reduction` (`mean` by default) combines the
 * per-example values through `reduce`.
 */

import {
  fromData,
  isTensor,
  isTraced,
  mean,
  shapeOfValue,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { define, isEntry } from 'aifn-compute/foundation/registry'
import { stopGradient } from 'aifn-compute/foundation/autodiff'
import { AifnError, DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { LossInfo, ReductionMode as Reduction, Stability } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { LossFamily, LossInput, LossInfo, ReductionMode as Reduction } from 'aifn-compute/foundation/contracts'

// ── Loss metadata ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Any function returning a value (a `never[]` parameter list accepts every signature). */
export type LossFunction = (...args: never[]) => Value

/** A loss: a differentiable function of predictions returning a number (or per-example values), with `info`. */
export type Loss<F extends LossFunction = LossFunction> = F & { readonly info: LossInfo }

/**
 * What a loss definition states: its `LossInfo` without the fields `defineLoss` fills (`kind`; `module`, default
 * `learning/losses`; `stability`, default `experimental`).
 */
export type LossSpec = Omit<LossInfo, 'kind' | 'module' | 'stability'> & {
  /** The module the loss is listed under (default `learning/losses`). */
  readonly module?: string
  /** How settled the loss's interface is (default `experimental`). */
  readonly stability?: Stability
}

/**
 * Attach metadata to a loss function: `aifn-compute/foundation/registry`'s `define` with kind `loss`, so the function
 * itself is returned with a frozen `info` added (`module` defaults to `learning/losses`, `stability` to
 * `experimental`). A function that already has `info` throws `DomainError`.
 *
 * @param spec The loss's metadata: `key`, `name`, `family`, `inputs`, and the optional fields of `LossInfo` (`notes`,
 *   `cite`, `target`, `pairedMetric`).
 * @param f The loss function; it gains the `info` property and is otherwise unchanged.
 * @returns `f` itself, typed as a `Loss`.
 *
 * @example Define a loss and read its metadata
 * const halfSquared = defineLoss(
 *   { key: 'halfSquared', name: 'Half squared error', family: 'regression', inputs: 'values' },
 *   (p, t) => reduce(mul(0.5, square(sub(p, t)))),
 * )
 * print('loss =', halfSquared(tensor([1, 3]), tensor([0, 1])))
 * print('family:', halfSquared.info.family, ' module:', halfSquared.info.module)
 * print('isLoss:', isLoss(halfSquared))
 */
export function defineLoss<F extends LossFunction>(spec: LossSpec, f: F): Loss<F> {
  const info: LossInfo = { module: 'learning/losses', stability: 'experimental', ...spec, kind: 'loss' }
  return define(info, f) as Loss<F>
}

/**
 * True when `x` is a loss defined with `defineLoss`: a function carrying `info` of kind `loss` with a `family`.
 *
 * @param x Any value.
 * @returns Whether `x` is a loss.
 *
 * @example Losses and other functions
 * print('huber:', isLoss(huber))
 * print('a plain function:', isLoss((x) => x))
 * print('learnedTemperature:', isLoss(learnedTemperature))
 */
export function isLoss(x: unknown): x is Loss {
  return (
    typeof x === 'function' &&
    isEntry(x, 'loss') &&
    typeof (x as { info: { family?: unknown } }).info.family === 'string'
  )
}

// ── Reduction ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Options every loss takes: `reduction` combines the per-example values, `mean` (default), `sum` or `none` (every
 * value, unreduced).
 */
export type ReductionOptions = { reduction?: Reduction }

/**
 * Combine per-example losses: the mean or sum over every entry, or the values unchanged. A number (no examples axis)
 * is returned as it is.
 *
 * @param v The per-example losses, of any shape.
 * @param reduction `mean` averages every entry, `sum` adds them, `none` returns `v` unchanged.
 * @returns A number for `mean` and `sum`; `v` itself for `none`.
 *
 * @example The three reductions
 * const v = tensor([1, 2, 6])
 * print('mean =', reduce(v))
 * print('sum =', reduce(v, 'sum'))
 * print('none =', reduce(v, 'none'))
 */
export function reduce(v: Value, reduction: Reduction = 'mean'): Value {
  if (reduction === 'none' || shapeOfValue(v).length === 0) return v
  return reduction === 'sum' ? sum(v) : mean(v)
}

// ── Constant inputs ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Targets, labels or grades: a number, an array of numbers, or a tensor (any dtype). Never differentiated. */
export type Target = number | ArrayLike<number> | Tensor

/**
 * A target as a float64 constant (a number stays a number). Traced targets are read as constants; inside `vmap`, where
 * a batched target (one per example, as for per-example gradients) has no single value, it stays a batched value with
 * its gradient stopped, which the elementwise losses combine as they would a tensor. Exported by `index.ts` as
 * `constantTarget`.
 *
 * @param t The target: a number, an array of numbers (read as a vector), a tensor of any dtype, or a traced value.
 * @returns A number for a number; otherwise a float64 tensor of the same shape (a float64 tensor is returned as it
 *   is, another dtype is converted).
 *
 * @example Arrays and integer tensors become float64 tensors
 * print('array:', constantTarget([1, 0, 2]))
 * print('dtype:', constantTarget(tensor([1, 2], undefined, 'int32')).dtype)
 * print('number:', constantTarget(3))
 */
export function constant(t: Target | Value): number | Tensor {
  if (typeof t === 'number') return t
  if (isTraced(t)) {
    try {
      return unwrap(t)
    } catch (e) {
      if (e instanceof AifnError) return stopGradient(t) as unknown as Tensor
      throw e
    }
  }
  if (isTensor(t)) return t.dtype === 'float64' ? t : fromData(Float64Array.from(t.data), t.shape)
  return fromData(Float64Array.from(t as ArrayLike<number>), [(t as ArrayLike<number>).length])
}

/**
 * A target's values as a flat `Float64Array`, row-major (a number gives one value).
 *
 * @param t The target: a number, an array of numbers, a tensor or a traced value (read as a constant).
 * @returns A new array of its values; changing it does not change `t`.
 *
 * @example A matrix flattened row by row
 * print(flatValues(tensor([[1, 2], [3, 4]])))
 */
export function flatValues(t: Target | Value): Float64Array {
  const c = constant(t)
  return typeof c === 'number' ? new Float64Array([c]) : Float64Array.from(toFlat(c))
}

/**
 * One-hot rows for integer class labels: a float64 tensor of the labels' shape with an axis of length $K$ appended,
 * holding 1 at each label's class and 0 elsewhere. A label that is not an integer in $[0, K)$ throws `DomainError`.
 *
 * @param labels The class labels: a number, an array or a tensor of integers in $[0, K)$.
 * @param K The number of classes.
 * @returns The one-hot rows, shape `[...labels.shape, K]` (`[K]` for a single number).
 *
 * @example Two labels among three classes
 * print(oneHot([2, 0], 3))
 */
export function oneHot(labels: Target, K: number): Tensor {
  const c = constant(labels)
  const shape = typeof c === 'number' ? [] : c.shape
  const ys = flatValues(labels)
  const out = new Float64Array(ys.length * K)
  ys.forEach((y, i) => {
    if (!Number.isInteger(y) || y < 0 || y >= K)
      throw new DomainError('oneHot', `oneHot: label ${y} is not a class in [0, ${K})`)
    out[i * K + y] = 1
  })
  return fromData(out, [...shape, K])
}

/**
 * Throw `ShapeError` unless a value has one of the expected ranks.
 *
 * @param x The value whose shape is checked.
 * @param ranks The ranks allowed (e.g. `[1, 2]` for a vector or a matrix).
 * @param what What `x` is, for the error message (e.g. `'infoNce anchors'`).
 * @returns The shape of `x`.
 *
 * @example A matrix passes, a vector does not
 * print('shape:', expectRank(tensor([[1, 2, 3]]), [2], 'scores'))
 * try {
 *   expectRank(tensor([1, 2, 3]), [2], 'scores')
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function expectRank(x: Value, ranks: readonly number[], what: string): number[] {
  const shape = shapeOfValue(x)
  if (!ranks.includes(shape.length)) {
    throw new ShapeError('losses', `losses: ${what} needs rank ${ranks.join(' or ')}, got shape [${shape.join(', ')}]`)
  }
  return shape
}
