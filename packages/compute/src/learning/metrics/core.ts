/**
 * The shared layer of `aifn-compute/learning/metrics`: the `Metric` type and its metadata, and the conversions every
 * metric uses to read its inputs (numbers, labels, rows) and to return tensors.
 *
 * The input conventions are those of sklearn.metrics. Numeric data is read flat (a tensor row-major), a matrix is a
 * rank-2 tensor or an array of rows, and labels may be numbers, strings or booleans, ordered as `compareLabels` orders
 * them. A ratio whose denominator is 0 is NaN unless the caller asks for another value (`divide`), so an undefined
 * metric is reported rather than hidden. `index.ts` re-exports the types, `defineMetric` and `isMetric`, and the input
 * helpers that the application metrics of `aifn-methods/evaluation` share; the rest is private to the module.
 */

import { copy, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { define, isEntry } from 'aifn-compute/foundation/registry'
import type { DataLike as Data, MetricInfo, MatrixLike as Rows, Stability } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type {
  InputKind,
  MetricCapability as Capability,
  MetricInfo,
  DataLike as Data,
  MatrixLike as Rows,
} from 'aifn-compute/foundation/contracts'

// ── Metric metadata ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Any function returning a number (a `never[]` parameter list accepts every signature). */
export type MetricFunction = (...args: never[]) => number

/**
 * A metric: a function returning a number, carrying its registry metadata as `info`. `I` keeps the literal fields a
 * definition states (its `capability` and `inputs`), so `evaluate` in `aifn-compute/learning/estimators` can check a
 * model against them at compile time.
 */
export type Metric<F extends MetricFunction = MetricFunction, I extends MetricInfo = MetricInfo> = F & {
  /** The metric's registry metadata: its key, name, inputs, direction, range, notes and capability. */
  readonly info: I
}

/**
 * What a metric definition states: its `MetricInfo` without the fields `defineMetric` fills (`kind`; `module`, default
 * `learning/metrics`; `stability`, default `experimental`).
 */
export type MetricSpec = Omit<MetricInfo, 'kind' | 'module' | 'stability'> & {
  /** The module that defines the metric (default `learning/metrics`). */
  readonly module?: string
  /** How settled the metric is (default `experimental`). */
  readonly stability?: Stability
}

/** The literal fields of a spec that `Metric` keeps. */
type Kept<S> = Pick<S, Extract<keyof S, 'key' | 'inputs' | 'direction' | 'capability'>>

/**
 * Attach metadata to a metric function (the function itself is returned, with `info` added): the `define` of
 * `aifn-compute/foundation/registry` with kind `metric`. A function already defined throws `DomainError`.
 *
 * @param spec The metric's metadata: `key`, `name`, `inputs`, `direction`, `range` and the optional fields of
 *   `MetricInfo`. `module` defaults to `learning/metrics` and `stability` to `experimental`; `kind` is set to
 *   `metric`.
 * @param f The metric itself: any function returning a number. It gains a frozen `info` property and is otherwise
 *   unchanged.
 * @returns `f`, typed as a `Metric` that keeps the literal `key`, `inputs`, `direction` and `capability` of `spec`.
 *
 * @example Define a metric of your own
 * const meanAbsError = defineMetric(
 *   { key: 'meanAbsError', name: 'Mean absolute error', inputs: 'values', direction: 'lower', range: [0, Infinity] },
 *   (yTrue, yPred) => yTrue.reduce((s, y, i) => s + Math.abs(y - yPred[i]), 0) / yTrue.length,
 * )
 * print('MAE =', meanAbsError([1, 2, 3], [2, 2, 5]))
 * print('info =', meanAbsError.info)
 * print('isMetric:', isMetric(meanAbsError))
 */
export function defineMetric<F extends MetricFunction, const S extends MetricSpec>(
  spec: S,
  f: F,
): Metric<F, MetricInfo & Kept<S>> {
  const info: MetricInfo = { module: 'learning/metrics', stability: 'experimental', ...spec, kind: 'metric' }
  return define(info, f) as Metric<F, MetricInfo & Kept<S>>
}

/**
 * True when `x` is a metric defined with `defineMetric`: a function whose registry entry has kind `metric`.
 *
 * @param x Any value.
 * @returns Whether `x` is a metric.
 *
 * @example A metric, and a plain function
 * print('accuracy:', isMetric(accuracy))
 * print('arrow function:', isMetric((a, b) => 0))
 */
export function isMetric(x: unknown): x is Metric {
  return typeof x === 'function' && isEntry(x, 'metric')
}

// ── Numeric inputs ───────────────────────────────────────────────────────────────────────────────────────────────────

/** A class label: a number, a string or a boolean. */
export type Label = number | string | boolean

/** Labels: an array of numbers, strings or booleans, or a tensor of numbers. */
export type Labels = ArrayLike<Label> | Tensor

/**
 * Every element of numeric data as a new Float64Array (row-major for tensors). Index exports it as `metricValues`.
 *
 * @param x Numbers: an array, a typed array or a tensor of any rank. It is copied, never shared or modified.
 * @returns A new Float64Array of every element, in row-major order.
 *
 * @example An array and a matrix, read flat
 * print('array:', metricValues([1, 2, 3]))
 * print('2 x 2 tensor:', metricValues(tensor([[1, 2], [3, 4]])))
 */
export function values(x: Data): Float64Array {
  return isTensor(x) ? (copy(x, 'float64').data as Float64Array) : Float64Array.from(x)
}

/**
 * A rank-1 float64 tensor holding `data`.
 *
 * @param data The values. A Float64Array is used as the tensor's storage without a copy; anything else is copied.
 * @returns A tensor of shape $[n]$, $n$ the length of `data`.
 */
export function vector(data: ArrayLike<number>): Tensor {
  return fromData(data instanceof Float64Array ? data : Float64Array.from(data), [data.length])
}

/**
 * A rank-2 float64 tensor from row-major data.
 *
 * @param data The $rc$ entries, row-major: row `i` occupies entries `i * cols` to `i * cols + cols - 1`. Used as the
 *   tensor's storage without a copy.
 * @param rows The number of rows $r$.
 * @param cols The number of columns $c$.
 * @returns A tensor of shape $[r, c]$.
 *
 * @example Six values as two rows of three
 * print(matrix(new Float64Array([1, 2, 3, 4, 5, 6]), 2, 3))
 */
export function matrix(data: Float64Array, rows: number, cols: number): Tensor {
  return fromData(data, [rows, cols])
}

/**
 * A matrix as row-major data with its dimensions: `rows` and `cols` count its rows and columns, and `data` holds its
 * entries row-major. A rank-1 input is one column ($n \times 1$).
 */
export type Dense = { rows: number; cols: number; data: Float64Array }

/**
 * Read a matrix (rank-2 tensor, rank-1 tensor as a column, or rows of numbers) into row-major data. A plain array of
 * numbers is one column, and an empty array is $0 \times 0$. A tensor of another rank, or rows of unequal length,
 * throws `ShapeError`. Index exports it as `denseMatrix`.
 *
 * @param x The matrix: a rank-2 tensor, a rank-1 tensor or array of numbers (read as a column), or an array of rows of
 *   equal length. It is copied, not modified.
 * @param what The caller's name for the input, for error messages.
 * @returns The dimensions and a new row-major copy of the entries.
 *
 * @example Rows, a column, and a ragged input
 * print('rows:', denseMatrix([[1, 2], [3, 4], [5, 6]], 'X'))
 * print('column:', denseMatrix([7, 8, 9], 'y'))
 * try {
 *   denseMatrix([[1, 2], [3]], 'X')
 * } catch (e) {
 *   print(e.name, ':', e.message)
 * }
 */
export function dense(x: Rows | Data, what: string): Dense {
  if (isTensor(x)) {
    if (x.shape.length === 2) return { rows: x.shape[0], cols: x.shape[1], data: values(x) }
    if (x.shape.length === 1) return { rows: x.shape[0], cols: 1, data: values(x) }
    throw new ShapeError('metrics', `metrics: ${what} needs a matrix, got shape [${x.shape.join(', ')}]`)
  }
  const n = x.length
  if (n === 0) return { rows: 0, cols: 0, data: new Float64Array(0) }
  const first = x[0] as unknown
  if (typeof first === 'number') return { rows: n, cols: 1, data: Float64Array.from(x as ArrayLike<number>) }
  const rows = x as ArrayLike<ArrayLike<number>>
  const cols = rows[0].length
  const data = new Float64Array(n * cols)
  for (let i = 0; i < n; i++) {
    const r = rows[i]
    if (r.length !== cols)
      throw new ShapeError('metrics', `metrics: ${what}: row ${i} has ${r.length} values, expected ${cols}`)
    for (let j = 0; j < cols; j++) data[i * cols + j] = r[j]
  }
  return { rows: n, cols, data }
}

/**
 * True for a rank-2 tensor or an array whose first element is itself an array: a matrix rather than a vector. Metrics
 * use it to tell multi-label rows from a list of labels. An empty array is not a matrix.
 *
 * @param x Any value.
 * @returns Whether `x` is read as a matrix.
 *
 * @example Rows, a vector and a rank-2 tensor
 * print('[[1, 0], [0, 1]]:', isMatrixLike([[1, 0], [0, 1]]))
 * print('[1, 0, 1]:', isMatrixLike([1, 0, 1]))
 * print('tensor([[1, 2]]):', isMatrixLike(tensor([[1, 2]])))
 */
export function isMatrixLike(x: unknown): boolean {
  if (isTensor(x)) return x.shape.length === 2
  if (typeof x !== 'object' || x === null || !('length' in x)) return false
  const a = x as ArrayLike<unknown>
  return a.length > 0 && typeof a[0] === 'object' && a[0] !== null && 'length' in (a[0] as object)
}

/**
 * True for numeric data or labels (an array, a typed array or a tensor), false for a plain options object. Lets a
 * metric accept an optional array argument or an options object in the same position.
 *
 * @param x Any value.
 * @returns Whether `x` is a tensor or has a `length`.
 */
export function isArrayInput(x: unknown): x is Data {
  return isTensor(x) || (typeof x === 'object' && x !== null && 'length' in x)
}

/**
 * Throw `ShapeError` unless two inputs have the same number of cases (the same `length`).
 *
 * @param a The first input, such as the true labels.
 * @param b The second input, such as the predictions.
 * @param what The caller's name, for the error message.
 *
 * @example Mismatched inputs are refused
 * sameLength([0, 1, 1], [0, 1, 1], 'demo')
 * print('equal lengths: no error')
 * try {
 *   sameLength([0, 1, 1], [0, 1], 'demo')
 * } catch (e) {
 *   print(e.name, ':', e.message)
 * }
 */
export function sameLength(a: { length: number }, b: { length: number }, what: string): void {
  if (a.length !== b.length)
    throw new ShapeError('metrics', `metrics: ${what}: inputs have ${a.length} and ${b.length} cases`)
}

/**
 * Throw `DomainError` on an empty input, where a mean would be $0/0$.
 *
 * @param n The number of cases.
 * @param what The caller's name, for the error message.
 */
export function nonEmpty(n: number, what: string): void {
  if (n === 0) throw new DomainError('metrics', `metrics: ${what} needs at least one case`)
}

// ── Labels ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Labels as a plain array (tensor elements become numbers).
 *
 * @param x The labels: an array of numbers, strings or booleans, or a tensor of numbers (read row-major).
 * @returns A new array of the labels, in order.
 *
 * @example From a tensor and from strings
 * print(labelList(tensor([0, 1, 1])))
 * print(labelList(['cat', 'dog']))
 */
export function labelList(x: Labels): Label[] {
  return isTensor(x) ? Array.from(values(x)) : Array.from(x)
}

/**
 * Order of labels: booleans (false before true), then numbers ascending, then strings in code-point order. The order
 * in which every metric lists classes.
 *
 * @param a The first label.
 * @param b The second label.
 * @returns A negative number when `a` comes first, a positive one when `b` does, 0 when they are equal.
 *
 * @example Sort mixed labels
 * print(['b', 2, true, 'a', 1, false].sort(compareLabels))
 */
export function compareLabels(a: Label, b: Label): number {
  const rank = (v: Label) => (typeof v === 'boolean' ? 0 : typeof v === 'number' ? 1 : 2)
  const d = rank(a) - rank(b)
  if (d !== 0) return d
  if (typeof a === 'string') return a < (b as string) ? -1 : a > (b as string) ? 1 : 0
  return Number(a) - Number(b)
}

/**
 * The sorted distinct labels of one or more label lists (as scikit-learn's `unique_labels`), in `compareLabels` order.
 *
 * @param lists The label lists, such as the true and the predicted labels.
 * @returns Every label that occurs in any list, once each, sorted.
 *
 * @example The classes of truth and prediction together
 * print(classesOf([2, 0, 1], [1, 3]))
 * print(classesOf(['dog', 'cat'], ['ant']))
 */
export function classesOf(...lists: Label[][]): Label[] {
  const seen = new Set<Label>()
  for (const l of lists) for (const v of l) seen.add(v)
  return [...seen].sort(compareLabels)
}

/**
 * Each label's index in `classes`; $-1$ for a label not in `classes`.
 *
 * @param list The labels to encode.
 * @param classes The class list that gives each label its index, as `classesOf` returns it.
 * @returns The index of each label of `list`, in order.
 *
 * @example Labels as class indices
 * print(encodeLabels(['b', 'a', 'c'], ['a', 'b']))
 */
export function encodeLabels(list: Label[], classes: readonly Label[]): Int32Array {
  const index = new Map<Label, number>()
  classes.forEach((c, k) => index.set(c, k))
  return Int32Array.from(list, (v) => index.get(v) ?? -1)
}

/**
 * The positive class of a binary problem: the given one, else `1` or `true` when present, else the last class in
 * label order (as scikit-learn's `pos_label=1` default, generalised to string labels). With no classes at all it is
 * `1`.
 *
 * @param classes The classes present, sorted as `classesOf` returns them.
 * @param positive The positive class the caller asked for; when given it is returned as it is, present or not.
 * @returns The positive class.
 *
 * @example The default and a chosen positive class
 * print(positiveOf([0, 1]))
 * print(positiveOf(['ham', 'spam']))
 * print(positiveOf(['ham', 'spam'], 'ham'))
 */
export function positiveOf(classes: readonly Label[], positive?: Label): Label {
  if (positive !== undefined) return positive
  if (classes.includes(1)) return 1
  if (classes.includes(true)) return true
  if (classes.length === 0) return 1
  return classes[classes.length - 1]
}

/**
 * Binary truth as 0/1 from labels and a positive class (default as `positiveOf`).
 *
 * @param yTrue The true labels.
 * @param positive The positive class; left out, it is chosen by `positiveOf` from the classes of `yTrue`.
 * @returns `y`, 1 where the label is the positive class and 0 elsewhere, and the `positive` class used.
 */
export function binaryTruth(yTrue: Labels, positive?: Label): { y: Uint8Array; positive: Label } {
  const list = labelList(yTrue)
  const pos = positiveOf(classesOf(list), positive)
  return { y: Uint8Array.from(list, (v) => (v === pos ? 1 : 0)), positive: pos }
}

// ── Small numeric helpers ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * $a / b$, or `zero` (default NaN) when $b = 0$: an undefined ratio is reported, not hidden (as the `zero_division`
 * argument of sklearn.metrics).
 *
 * @param a The numerator.
 * @param b The denominator.
 * @param zero The value returned when `b` is 0.
 * @returns The ratio, or `zero`.
 *
 * @example A ratio, and one that is undefined
 * print('1 / 4 =', divide(1, 4))
 * print('1 / 0 =', divide(1, 0))
 * print('1 / 0 with zero = 0:', divide(1, 0, 0))
 */
export function divide(a: number, b: number, zero = NaN): number {
  return b !== 0 ? a / b : zero
}

/**
 * The arithmetic mean of an array (NaN for an empty array).
 *
 * @param x The values.
 * @returns Their mean.
 */
export function meanOf(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return x.length ? s / x.length : NaN
}

/**
 * Weighted mean $\sum_i w_i x_i / \sum_i w_i$, or the plain mean without weights. Weights summing to 0 give NaN or an
 * infinity.
 *
 * @param x The values $x_i$.
 * @param w The weights $w_i$, one per value (not checked); left out, every value has weight 1.
 * @returns The weighted mean.
 */
export function weightedMeanOf(x: ArrayLike<number>, w?: ArrayLike<number>): number {
  if (!w) return meanOf(x)
  let s = 0
  let t = 0
  for (let i = 0; i < x.length; i++) {
    s += w[i] * x[i]
    t += w[i]
  }
  return s / t
}

/**
 * Case weights: the given ones as a Float64Array, checked against `n`, or undefined. A count other than `n` throws
 * `ShapeError`.
 *
 * @param w The weights, one per case, or undefined for none.
 * @param n The number of cases.
 * @param what The caller's name, for the error message.
 * @returns A new Float64Array of the weights, or undefined when `w` is.
 */
export function caseWeights(w: Data | undefined, n: number, what: string): Float64Array | undefined {
  if (w === undefined) return undefined
  const out = values(w)
  if (out.length !== n) throw new ShapeError('metrics', `metrics: ${what}: ${out.length} weights for ${n} cases`)
  return out
}

/**
 * Indices that sort `scores` in decreasing order, ties kept in their original order (a stable sort), so that ranking
 * metrics break ties by input position.
 *
 * @param scores The scores to rank.
 * @returns The case indices, highest score first.
 */
export function orderDescending(scores: ArrayLike<number>): Int32Array {
  const idx = Array.from({ length: scores.length }, (_, i) => i)
  idx.sort((a, b) => scores[b] - scores[a] || a - b)
  return Int32Array.from(idx)
}
