/**
 * Datasets and estimators: the shapes every `fit` takes and returns. `Dataset` is the contract's (`kind: 'dataset'`);
 * `dataset(x, y?)` builds one, and `takeRows`/`takeData` select rows of tensors, label lists, tables and datasets.
 */

import type { Column, Dataset, Features, Size, Table } from 'aifn-compute/foundation/contracts'
import type { Stream } from 'aifn-compute/foundation/random'
import { copy, isTensor, take, type Tensor } from 'aifn-compute/foundation/tensor'
import type { TraceOptions } from 'aifn-compute/foundation/trace'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Column, Dataset, DatasetMeta, Features, Table } from 'aifn-compute/foundation/contracts'

/** A dataset with targets. */
export type Supervised<X extends Features = Tensor, Y = Tensor> = Dataset<X, Y> & { readonly y: Y }

/** The optional fields of a dataset besides its features and targets. */
export type DatasetExtras = Partial<Pick<Dataset, 'groups' | 't' | 'f' | 'meta'>>

/**
 * A dataset from features `x` (n rows), optional targets `y` and optional extras (group labels, a colouring
 * coordinate `t`, the noise-free target `f`, metadata).
 *
 * @example linearRegression().fit(dataset(x, y))
 */
export function dataset<X extends Features, Y = Tensor>(x: X, y: Y, extras?: DatasetExtras): Supervised<X, Y>
export function dataset<X extends Features>(x: X, y?: undefined, extras?: DatasetExtras): Dataset<X, never>
export function dataset<X extends Features, Y>(x: X, y?: Y, extras: DatasetExtras = {}): Dataset<X, Y> {
  return { kind: 'dataset', x, ...(y === undefined ? {} : { y }), ...extras }
}

/** Options every `fit` accepts. */
export interface FitOptions {
  /** The randomness of the fit (initialisation, random features, minibatches). */
  stream?: Stream
  /** How iterative fits record their training trace (default: every step, no recorders). */
  trace?: Pick<TraceOptions<unknown>, 'every' | 'record' | 'checkpointEvery'>
}

/**
 * An estimator: an unfitted model description with hyperparameters, whose `fit` returns a fitted model, a plain object
 * with its fitted state as public fields and its capabilities as methods. Fitting never mutates the estimator.
 */
export interface Estimator<D, M> {
  /** A readable name, e.g. "logistic-regression". */
  readonly name: string
  /** The hyperparameters it was made with. */
  readonly params?: unknown
  fit(data: D, options?: FitOptions): M
}

/** The fitted model type of an estimator. */
export type ModelOf<E> =
  E extends Estimator<never, infer M> ? M : E extends { fit(...args: never[]): infer M } ? M : never

/** The data type an estimator fits on. */
export type DataOf<E> = E extends { fit(data: infer D, ...rest: never[]): unknown } ? D : never

// ── Rows ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Number of rows of a tensor (its first axis), label list or table (all columns must agree). */
export function rowCount(x: Features | Column): Size {
  if (isTensor(x)) {
    if (x.shape.length === 0) throw new ShapeError('rowCount', 'rowCount: a scalar has no rows')
    return x.shape[0]
  }
  if (Array.isArray(x)) return x.length
  let n = -1
  for (const [name, column] of Object.entries(x as Table)) {
    const m = rowCount(column)
    if (n >= 0 && m !== n) throw new ShapeError('rowCount', `rowCount: column "${name}" has ${m} rows, others ${n}`)
    n = m
  }
  if (n < 0) throw new DomainError('rowCount', 'rowCount: a table with no columns')
  return n
}

/**
 * The rows `index` (in that order, repeats allowed) of a tensor (by `aifn-compute/foundation/tensor`'s `take`, keeping its
 * dtype), label list or table.
 */
export function takeRows<X extends Features | Column>(x: X, index: ArrayLike<number>): X {
  if (isTensor(x)) {
    const rows = take(x, index) as Tensor
    return (rows.dtype === x.dtype ? rows : copy(rows, x.dtype)) as X
  }
  if (Array.isArray(x)) return Array.from(index, (i) => (x as readonly unknown[])[i]) as unknown as X
  const out: Record<string, Column> = {}
  for (const [name, column] of Object.entries(x as Table)) out[name] = takeRows(column, index)
  return out as X
}

/** The rows `index` of every row-aligned field of a dataset (`x`, `y`, `groups`, `t`, `f`); `meta` is kept. */
export function takeData<D extends Dataset<Features, unknown>>(data: D, index: ArrayLike<number>): D {
  const out: Record<string, unknown> = { ...(data as object) }
  for (const key of ['x', 'y', 'groups', 't', 'f'] as const) {
    const v = (data as Record<string, unknown>)[key]
    if (v !== undefined) out[key] = takeRows(v as Features | Column, index)
  }
  return out as D
}
