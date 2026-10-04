/**
 * The shared layer of `aifn-compute/learning/losses`: the `Loss` type and its metadata, the reduction over examples, and the
 * conversions that read targets (labels, relevance grades) into constant tensors. Private to the module except for the
 * types, `defineLoss`, `isLoss` and `oneHot`, which `index.ts` re-exports.
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
  readonly module?: string
  readonly stability?: Stability
}

/**
 * Attach metadata to a loss function (the function itself is returned, with `info` added): `aifn-compute/foundation/registry`'s
 * `define` with kind `loss`.
 */
export function defineLoss<F extends LossFunction>(spec: LossSpec, f: F): Loss<F> {
  const info: LossInfo = { module: 'learning/losses', stability: 'experimental', ...spec, kind: 'loss' }
  return define(info, f) as Loss<F>
}

/** True when `x` is a loss defined with `defineLoss`. */
export function isLoss(x: unknown): x is Loss {
  return (
    typeof x === 'function' &&
    isEntry(x, 'loss') &&
    typeof (x as { info: { family?: unknown } }).info.family === 'string'
  )
}

// ── Reduction ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options every loss takes. */
export type ReductionOptions = { reduction?: Reduction }

/** Combine per-example losses. A number (no examples axis) is returned as it is. */
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
 * its gradient stopped, which the elementwise losses combine as they would a tensor.
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

/** A target's values as a flat Float64Array (row-major). */
export function flatValues(t: Target | Value): Float64Array {
  const c = constant(t)
  return typeof c === 'number' ? new Float64Array([c]) : Float64Array.from(toFlat(c))
}

/**
 * One-hot rows for integer class labels: a float64 tensor of shape [...labels shape, K] with 1 at each label's class.
 * Labels must be integers in [0, K).
 *
 * @example oneHot([2, 0], 3) // [[0, 0, 1], [1, 0, 0]]
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

/** Throw unless a value has the expected rank(s). */
export function expectRank(x: Value, ranks: readonly number[], what: string): number[] {
  const shape = shapeOfValue(x)
  if (!ranks.includes(shape.length)) {
    throw new ShapeError('losses', `losses: ${what} needs rank ${ranks.join(' or ')}, got shape [${shape.join(', ')}]`)
  }
  return shape
}
