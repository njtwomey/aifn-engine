/**
 * Datasets and estimators: the shapes every `fit` takes and returns. `Dataset` is the contract's (`kind: 'dataset'`);
 * `dataset(x, y?)` builds one, and `takeRows`/`takeData` select rows of tensors, label lists, tables and datasets.
 *
 * Features are a tensor whose first axis is the rows, a list of labels, or a table (a record of named columns with the
 * same number of rows). Selecting rows never changes its input: it returns new tensors, lists and records.
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
 * A dataset from features `x` ($n$ rows), optional targets `y` and optional extras (group labels, a colouring
 * coordinate `t`, the noise-free target `f`, metadata). Nothing is copied or checked: the fields are the values given.
 *
 * @param x The features: a tensor with one row per example, a label list or a table.
 * @param y The targets, one per row. Left out, the dataset has no `y` field (unsupervised).
 * @param extras Further fields: `groups`, `t`, `f` and `meta`.
 * @returns The dataset `{ kind: 'dataset', x, y, ...extras }`.
 *
 * @example A labelled dataset with groups
 * const data = dataset(tensor([[0, 1], [1, 0], [1, 1]]), tensor([0, 1, 1]), { groups: ['a', 'a', 'b'] })
 * print('kind:', data.kind)
 * print('rows:', rowCount(data.x))
 * print('groups:', data.groups)
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
  /** Fit on `data`, with randomness and tracing from `options`, returning a new fitted model. */
  fit(data: D, options?: FitOptions): M
}

/** The fitted model type of an estimator. */
export type ModelOf<E> =
  E extends Estimator<never, infer M> ? M : E extends { fit(...args: never[]): infer M } ? M : never

/** The data type an estimator fits on. */
export type DataOf<E> = E extends { fit(data: infer D, ...rest: never[]): unknown } ? D : never

// ── Rows ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Number of rows of a tensor (its first axis), label list or table (all columns must agree). Throws `ShapeError` for
 * a scalar tensor or a table whose columns disagree, and `DomainError` for a table with no columns.
 *
 * @param x The features or a column.
 * @returns The number of rows $n$.
 *
 * @example A tensor, a label list and a table
 * print('matrix:', rowCount(tensor([[1, 2], [3, 4], [5, 6]])))
 * print('labels:', rowCount(['a', 'b']))
 * print('table:', rowCount({ age: tensor([20, 30]), city: ['Cork', 'Bath'] }))
 */
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
 * The rows `index` (in that order, repeats allowed) of a tensor (by `aifn-compute/foundation/tensor`'s `take`, keeping
 * its dtype), label list or table (each column taken alike).
 *
 * @param x The features or a column; not modified.
 * @param index Row indices, each in $0, \dots, n - 1$.
 * @returns A new value of the same kind as `x` with one row per index.
 *
 * @example Reorder and repeat rows
 * print('tensor:', takeRows(tensor([[1, 2], [3, 4], [5, 6]]), [2, 0, 2]))
 * print('labels:', takeRows(['a', 'b', 'c'], [2, 0, 2]))
 * print('table:', takeRows({ age: tensor([20, 30, 40]), city: ['Cork', 'Bath', 'Oslo'] }, [1]))
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

/**
 * The rows `index` of every row-aligned field of a dataset (`x`, `y`, `groups`, `t`, `f`); every other field (`meta`,
 * `kind`) is kept as it is.
 *
 * @param data The dataset; not modified.
 * @param index Row indices, each in $0, \dots, n - 1$.
 * @returns A new dataset of the selected rows.
 *
 * @example A training fold of a grouped dataset
 * const data = dataset(tensor([[0], [1], [2], [3]]), tensor([0, 0, 1, 1]), { groups: ['a', 'a', 'b', 'b'] })
 * const fold = takeData(data, [0, 2])
 * print('x:', fold.x)
 * print('y:', fold.y)
 * print('groups:', fold.groups)
 */
export function takeData<D extends Dataset<Features, unknown>>(data: D, index: ArrayLike<number>): D {
  const out: Record<string, unknown> = { ...(data as object) }
  for (const key of ['x', 'y', 'groups', 't', 'f'] as const) {
    const v = (data as Record<string, unknown>)[key]
    if (v !== undefined) out[key] = takeRows(v as Features | Column, index)
  }
  return out as D
}
