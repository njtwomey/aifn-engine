/**
 * Categorical encoders: one-hot, ordinal and target encoding, as scikit-learn's `OneHotEncoder`, `OrdinalEncoder` and
 * `TargetEncoder`. Categories are sorted (numbers ascending, strings by code unit), as `np.unique` sorts them.
 *
 * The input is a list of labels (one column), or a numeric tensor of $n$ codes (one column) or $n \times d$ ($d$
 * columns of codes); each column has its own categories, learnt by `fit`. A column may not mix numbers and strings.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { values, type FittedTransform, type Invertible, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A category label. */
export type Category = string | number

/**
 * Categorical input: a label list (one column), or a numeric tensor of $n$ values (one column) or $n \times d$ ($d$
 * columns of codes).
 */
export type CategoricalInput = Tensor | readonly Category[]

/**
 * Categorical input split into columns: `n` rows, `d` columns, `columns[j][i]` the label of row $i$ in column $j$, and
 * `list`, whether the input was a label list rather than a tensor.
 */
type Columns = { n: number; d: number; columns: Category[][]; list: boolean }

/**
 * Whether categorical input is a tensor rather than a label list.
 *
 * @param x The input.
 * @returns True for a tensor, false for an array of labels.
 */
function isTensor(x: CategoricalInput): x is Tensor {
  return !Array.isArray(x)
}

/**
 * Split categorical input into columns of labels. Throws `ShapeError` for a tensor of rank above 2.
 *
 * @param x A label list (one column), or a tensor of $n$ or $n \times d$ codes.
 * @param where The caller's name, for error messages.
 * @returns The input as columns of labels, with its size and whether it was a list.
 */
function columnsOf(x: CategoricalInput, where: string): Columns {
  if (!isTensor(x)) return { n: x.length, d: 1, columns: [Array.from(x)], list: true }
  if (x.shape.length > 2) throw new ShapeError(where, `${where}: expected [n] or [n, d], got [${x.shape.join(', ')}]`)
  const n = x.shape[0]
  const d = x.shape.length === 1 ? 1 : x.shape[1]
  const v = values(x)
  const columns = Array.from({ length: d }, (_, j) => Array.from({ length: n }, (_, i) => v[i * d + j]))
  return { n, d, columns, list: false }
}

/**
 * The distinct labels of a column, sorted: numbers ascending, strings by code unit. Throws `DomainError` when the
 * column mixes numbers and strings.
 *
 * @param c The labels of one column.
 * @returns Its distinct labels in sorted order, in a new array.
 */
function sortedUnique(c: readonly Category[]): Category[] {
  const unique = Array.from(new Set(c))
  const numeric = unique.every((v) => typeof v === 'number')
  if (!numeric && unique.some((v) => typeof v === 'number'))
    throw new DomainError('encoder', 'encoder: a column mixes numbers and strings')
  return numeric
    ? (unique as number[]).sort((a, b) => a - b)
    : (unique as string[]).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
}

/**
 * Rebuild categorical output of the input's kind from columns of labels: the first column as a list, or all of them as
 * a tensor.
 *
 * @param columns The labels, `columns[j][i]` for row $i$ of column $j$; numbers when a tensor is rebuilt.
 * @param n The number of rows.
 * @param list Whether to return a label list (the first column only) rather than a tensor.
 * @param tensorShape The shape of the tensor to return ($n$ or $n \times d$); ignored for a list.
 * @returns The labels as a list, or as a row-major tensor of shape `tensorShape`.
 */
function fromColumns(
  columns: Category[][],
  n: number,
  list: boolean,
  tensorShape: readonly number[],
): CategoricalInput {
  if (list) return columns[0]
  const d = columns.length
  const out = new Float64Array(n * d)
  for (let j = 0; j < d; j++) for (let i = 0; i < n; i++) out[i * d + j] = columns[j][i] as number
  return fromData(out, tensorShape)
}

// ── One-hot ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted one-hot encoder. */
export interface OneHotEncoder extends FittedTransform<CategoricalInput, Tensor>, Invertible<CategoricalInput, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'one-hot-encoder'
  /** The sorted categories of each input column. */
  readonly categories: readonly (readonly Category[])[]
  /** The index of the category dropped from each column (which has no indicator), or $-1$. */
  readonly dropped: readonly number[]
  /** A name per output column, "j=category" (or "category" for a single input column). */
  readonly featureNames: readonly string[]
}

/**
 * One indicator column per category of each input column, in the order of the sorted categories, as scikit-learn's
 * `OneHotEncoder`: the output is $n \times w$, with $w$ the total number of indicators over all columns.
 * `drop: 'first'` omits each column's first category (avoiding collinearity with an intercept); `'if-binary'` does so
 * only for two-category columns. An unseen category throws `DomainError`, or with `handleUnknown: 'ignore'` encodes as
 * all zeros. `inverseTransform` maps each block back to the category of its largest positive entry; an all-zero block
 * decodes to the dropped category, or to NaN when there is none (an ignored unknown).
 *
 * @param options What to drop and how to treat unseen categories.
 * @param options.drop `'first'` drops every column's first category, `'if-binary'` only that of two-category columns,
 *   `null` none.
 * @param options.handleUnknown `'error'` throws on a category not seen in fitting; `'ignore'` encodes it as all zeros.
 * @returns An estimator whose `fit({ x })` learns each column's categories and returns the fitted `OneHotEncoder`.
 *
 * @example Three categories become three indicator columns
 * const model = oneHotEncoder().fit({ x: ['red', 'green', 'blue'] })
 * print('features:', model.featureNames)
 * print(model.transform(['green', 'red', 'blue', 'green']))
 *
 * @example Dropping the first category, and decoding back
 * const model = oneHotEncoder({ drop: 'first' }).fit({ x: ['red', 'green', 'blue'] })
 * const z = model.transform(['blue', 'green', 'red'])
 * print('features:', model.featureNames)
 * print('z =', z)
 * print('decoded:', model.inverseTransform(z))
 */
export function oneHotEncoder({
  drop = null,
  handleUnknown = 'error',
}: { drop?: 'first' | 'if-binary' | null; handleUnknown?: 'error' | 'ignore' } = {}): Transformer<
  CategoricalInput,
  OneHotEncoder
> {
  return {
    name: 'one-hot-encoder',
    params: { drop, handleUnknown },
    fit({ x }) {
      const fitted = columnsOf(x, 'oneHotEncoder')
      const categories = fitted.columns.map(sortedUnique)
      const dropped = categories.map((c) => (drop === 'first' || (drop === 'if-binary' && c.length === 2) ? 0 : -1))
      const widths = categories.map((c, j) => c.length - (dropped[j] >= 0 ? 1 : 0))
      const offsets = widths.map((_, j) => widths.slice(0, j).reduce((a, b) => a + b, 0))
      const width = widths.reduce((a, b) => a + b, 0)
      const lookup = categories.map((c) => new Map(c.map((v, k) => [v, k])))
      const featureNames = categories.flatMap((c, j) =>
        c.filter((_, k) => k !== dropped[j]).map((v) => (fitted.d === 1 ? String(v) : `${j}=${v}`)),
      )
      const tensorShape = isTensor(x) ? x.shape.slice(1) : []
      return {
        kind: 'model',
        name: 'one-hot-encoder',
        categories,
        dropped,
        featureNames,
        transform(input) {
          const { n, d, columns } = columnsOf(input, 'oneHotEncoder.transform')
          if (d !== categories.length)
            throw new ShapeError('oneHotEncoder', `oneHotEncoder: fitted on ${categories.length} columns, given ${d}`)
          const out = new Float64Array(n * width)
          for (let j = 0; j < d; j++) {
            for (let i = 0; i < n; i++) {
              const k = lookup[j].get(columns[j][i])
              if (k === undefined) {
                if (handleUnknown === 'error')
                  throw new DomainError(
                    'oneHotEncoder',
                    `oneHotEncoder: unknown category ${columns[j][i]} in column ${j}`,
                  )
                continue
              }
              if (k === dropped[j]) continue
              out[i * width + offsets[j] + k - (dropped[j] >= 0 && k > dropped[j] ? 1 : 0)] = 1
            }
          }
          return fromData(out, [n, width])
        },
        inverseTransform(z) {
          const n = z.shape[0]
          const v = values(z)
          const columns = categories.map((c, j) =>
            Array.from({ length: n }, (_, i): Category => {
              let best = -1
              let bestValue = 0
              for (let k = 0; k < widths[j]; k++) {
                const e = v[i * width + offsets[j] + k]
                if (e > bestValue) [best, bestValue] = [k, e]
              }
              if (best < 0) return dropped[j] >= 0 ? c[dropped[j]] : NaN
              return c[best + (dropped[j] >= 0 && best >= dropped[j] ? 1 : 0)]
            }),
          )
          return fromColumns(columns, n, fitted.list, [n, ...tensorShape])
        },
      }
    },
  }
}

// ── Ordinal ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted ordinal encoder. */
export interface OrdinalEncoder
  extends FittedTransform<CategoricalInput, Tensor>, Invertible<CategoricalInput, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'ordinal-encoder'
  /** The sorted categories of each input column; a category is encoded as its index here. */
  readonly categories: readonly (readonly Category[])[]
}

/**
 * Each category becomes its index $0, \dots, K - 1$ among its column's $K$ sorted categories, as scikit-learn's
 * `OrdinalEncoder`: the output is $n \times d$. An unseen category throws `DomainError`, or with
 * `handleUnknown: 'use-encoded-value'` becomes `unknownValue`. `inverseTransform` maps a code that is not an index of
 * the column (such as `unknownValue`) to NaN.
 *
 * @param options How to treat unseen categories.
 * @param options.handleUnknown `'error'` throws on a category not seen in fitting; `'use-encoded-value'` encodes it as
 *   `unknownValue`.
 * @param options.unknownValue The code given to an unseen category under `'use-encoded-value'`.
 * @returns An estimator whose `fit({ x })` learns each column's categories and returns the fitted `OrdinalEncoder`.
 *
 * @example Categories are numbered in sorted order
 * const model = ordinalEncoder().fit({ x: ['low', 'mid', 'high', 'mid'] })
 * print('categories:', model.categories)
 * print(model.transform(['low', 'mid', 'high']))
 *
 * @example An unseen category as $-1$
 * const model = ordinalEncoder({ handleUnknown: 'use-encoded-value', unknownValue: -1 }).fit({ x: ['a', 'b'] })
 * print(model.transform(['b', 'c', 'a']))
 */
export function ordinalEncoder({
  handleUnknown = 'error',
  unknownValue = NaN,
}: { handleUnknown?: 'error' | 'use-encoded-value'; unknownValue?: number } = {}): Transformer<
  CategoricalInput,
  OrdinalEncoder
> {
  return {
    name: 'ordinal-encoder',
    params: { handleUnknown, unknownValue },
    fit({ x }) {
      const fitted = columnsOf(x, 'ordinalEncoder')
      const categories = fitted.columns.map(sortedUnique)
      const lookup = categories.map((c) => new Map(c.map((v, k) => [v, k])))
      const tensorShape = isTensor(x) ? x.shape.slice(1) : []
      return {
        kind: 'model',
        name: 'ordinal-encoder',
        categories,
        transform(input) {
          const { n, d, columns } = columnsOf(input, 'ordinalEncoder.transform')
          if (d !== categories.length)
            throw new ShapeError('ordinalEncoder', `ordinalEncoder: fitted on ${categories.length} columns, given ${d}`)
          const out = new Float64Array(n * d)
          for (let j = 0; j < d; j++) {
            for (let i = 0; i < n; i++) {
              const k = lookup[j].get(columns[j][i])
              if (k === undefined && handleUnknown === 'error') {
                throw new DomainError(
                  'ordinalEncoder',
                  `ordinalEncoder: unknown category ${columns[j][i]} in column ${j}`,
                )
              }
              out[i * d + j] = k ?? unknownValue
            }
          }
          return fromData(out, [n, d])
        },
        inverseTransform(z) {
          const n = z.shape[0]
          const d = categories.length
          const v = values(z)
          const columns = categories.map((c, j) => Array.from({ length: n }, (_, i) => c[v[i * d + j]] ?? NaN))
          return fromColumns(columns, n, fitted.list, [n, ...tensorShape])
        },
      }
    },
  }
}

// ── Target encoding ──────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted target encoder. */
export interface TargetEncoder extends FittedTransform<CategoricalInput, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'target-encoder'
  /** The sorted categories of each input column. */
  readonly categories: readonly (readonly Category[])[]
  /** The encoding of each category of each column, aligned with `categories`. */
  readonly encodings: readonly Tensor[]
  /** The mean target $\bar{y}$, used for unseen categories. */
  readonly targetMean: number
  /** The smoothing used: a number $m$, or "auto". */
  readonly smooth: number | 'auto'
}

/**
 * Replace each category $c$ by a shrunk estimate of $\expect[y \mid c]$ (Micci-Barreca, 2001, "A preprocessing scheme
 * for high-cardinality categorical attributes in classification and prediction problems", SIGKDD Explorations 3), as
 * scikit-learn's `TargetEncoder`. With a number $m$: $(\sum_{i \in c} y_i + m \bar{y}) / (n_c + m)$, with $n_c$ the
 * number of rows of category $c$ and $\bar{y}$ the mean target. With `'auto'` (eqs 5–6, scikit-learn's default):
 * $\lambda_c \bar{y}_c + (1 - \lambda_c) \bar{y}$ with $\lambda_c = \sigma^2 n_c / (\sigma^2 n_c + s_c^2)$,
 * $\bar{y}_c$ the category's mean target, $\sigma^2$ the population variance of $y$ and $s_c^2$ the within-category
 * mean squared deviation (when both are 0, the category gets $\bar{y}$). Unseen categories get $\bar{y}$. Encoding the
 * training rows with this fitted model leaks their own targets; use `targetEncodeCrossFit` for training features.
 * `fit` throws `DomainError` without targets, and `ShapeError` when `x` and `y` have different numbers of rows.
 *
 * @param options The shrinkage towards $\bar{y}$.
 * @param options.smooth A number $m \ge 0$, the weight of $\bar{y}$ counted as $m$ extra rows; or `'auto'`, the
 *   variance-based weight $\lambda_c$.
 * @returns An estimator whose `fit({ x, y })` returns the fitted `TargetEncoder`; `transform` gives an $n \times d$
 *   matrix of encodings.
 *
 * @example A rare category is pulled towards the overall mean
 * const x = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c']
 * const y = tensor([1, 1, 1, 0, 0, 0, 0, 1, 1])
 * const model = targetEncoder({ smooth: 2 }).fit({ x, y })
 * print('mean target =', model.targetMean)
 * print('categories:', model.categories[0])
 * print('encodings =', model.encodings[0])
 * print('unseen "d" ->', model.transform(['d']))
 */
export function targetEncoder({ smooth = 'auto' }: { smooth?: number | 'auto' } = {}): Transformer<
  CategoricalInput,
  TargetEncoder
> {
  return {
    name: 'target-encoder',
    params: { smooth },
    fit({ x, y }) {
      if (!y) throw new DomainError('targetEncoder', 'targetEncoder: needs targets y')
      const target = values(y)
      const fitted = columnsOf(x, 'targetEncoder')
      if (target.length !== fitted.n)
        throw new ShapeError('targetEncoder', 'targetEncoder: x and y have different numbers of rows')
      const n = fitted.n
      let yMean = 0
      for (const v of target) yMean += v / n
      let yVar = 0
      for (const v of target) yVar += (v - yMean) ** 2 / n
      const categories = fitted.columns.map(sortedUnique)
      const encodings = categories.map((cats, j) => {
        const index = new Map(cats.map((v, k) => [v, k]))
        const K = cats.length
        const sums = new Float64Array(K)
        const counts = new Float64Array(K)
        const col = fitted.columns[j]
        for (let i = 0; i < n; i++) {
          const k = index.get(col[i])!
          sums[k] += target[i]
          counts[k]++
        }
        if (smooth !== 'auto') return Float64Array.from(sums, (s, k) => (s + smooth * yMean) / (counts[k] + smooth))
        const means = Float64Array.from(sums, (s, k) => s / counts[k])
        const ss = new Float64Array(K)
        for (let i = 0; i < n; i++) {
          const k = index.get(col[i])!
          ss[k] += (target[i] - means[k]) ** 2
        }
        return Float64Array.from(means, (m, k) => {
          const lambda = (yVar * counts[k]) / (yVar * counts[k] + ss[k] / counts[k])
          return Number.isNaN(lambda) ? yMean : lambda * m + (1 - lambda) * yMean
        })
      })
      const lookup = categories.map((c) => new Map(c.map((v, k) => [v, k])))
      return {
        kind: 'model',
        name: 'target-encoder',
        categories,
        encodings: encodings.map((e) => fromData(e, [e.length])),
        targetMean: yMean,
        smooth,
        transform(input) {
          const { n: m, d, columns } = columnsOf(input, 'targetEncoder.transform')
          if (d !== categories.length)
            throw new ShapeError('targetEncoder', `targetEncoder: fitted on ${categories.length} columns, given ${d}`)
          const out = new Float64Array(m * d)
          for (let j = 0; j < d; j++) {
            for (let i = 0; i < m; i++) {
              const k = lookup[j].get(columns[j][i])
              out[i * d + j] = k === undefined ? yMean : encodings[j][k]
            }
          }
          return fromData(out, [m, d])
        },
      }
    },
  }
}

/**
 * Target-encode training rows without leaking their own targets: split the rows into `folds` contiguous blocks (as
 * scikit-learn's `KFold` without shuffling; the first $n \bmod k$ blocks have one row more), encode each block with an
 * encoder fitted on the other blocks, and also return the encoder fitted on all rows (for new data). Shuffle the rows
 * first if their order carries structure. Throws `DomainError` unless $2 \le k \le n$ for $k$ folds and $n$ rows.
 *
 * @param data The training rows: categorical inputs `x` and their targets `y` ($n$ values).
 * @param options The encoder's smoothing and the number of folds.
 * @param options.smooth The smoothing of each `targetEncoder`: a number $m$, or `'auto'`.
 * @param options.folds The number $k$ of contiguous blocks, from 2 to $n$.
 * @returns `encoded`, the out-of-fold encodings of the training rows ($n \times d$), and `model`, the encoder fitted on
 *   every row.
 *
 * @example Out-of-fold encodings differ from the in-sample ones
 * const x = ['a', 'b', 'a', 'b', 'a', 'b']
 * const y = tensor([1, 0, 1, 0, 0, 1])
 * const { encoded, model } = targetEncodeCrossFit({ x, y }, { smooth: 1, folds: 3 })
 * print('out of fold =', encoded)
 * print('in sample =', model.transform(x))
 */
export function targetEncodeCrossFit(
  data: { x: CategoricalInput; y: Tensor },
  { smooth = 'auto', folds = 5 }: { smooth?: number | 'auto'; folds?: number } = {},
): { encoded: Tensor; model: TargetEncoder } {
  const { n, d, columns, list } = columnsOf(data.x, 'targetEncodeCrossFit')
  if (folds < 2 || folds > n)
    throw new DomainError('targetEncodeCrossFit', 'targetEncodeCrossFit: folds must be in [2, n]')
  const y = values(data.y)
  const out = new Float64Array(n * d)
  let start = 0
  for (let f = 0; f < folds; f++) {
    const size = Math.floor(n / folds) + (f < n % folds ? 1 : 0)
    const test = Array.from({ length: size }, (_, k) => start + k)
    const train = Array.from({ length: n }, (_, i) => i).filter((i) => i < start || i >= start + size)
    const pick = (rows: number[]): CategoricalInput =>
      fromColumns(
        columns.map((c) => rows.map((i) => c[i])),
        rows.length,
        list,
        [rows.length, ...(isTensor(data.x) ? data.x.shape.slice(1) : [])],
      )
    const model = targetEncoder({ smooth }).fit({ x: pick(train), y: fromData(Float64Array.from(train, (i) => y[i])) })
    const z = values(model.transform(pick(test)))
    for (let r = 0; r < size; r++) for (let j = 0; j < d; j++) out[(start + r) * d + j] = z[r * d + j]
    start += size
  }
  return { encoded: fromData(out, [n, d]), model: targetEncoder({ smooth }).fit(data) }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'oneHotEncoder',
    module: 'learning/preprocessing',
    name: 'One-hot encoder',
    summary: 'One indicator column per category.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({ handleUnknown: oneOf(['error', 'ignore']) }),
    notes: ['categorical-encoding'],
    cite: ['pedregosa2011'],
  },
  oneHotEncoder,
)

defineModel(
  {
    key: 'ordinalEncoder',
    module: 'learning/preprocessing',
    name: 'Ordinal encoder',
    summary: 'Each category to its integer index.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({ handleUnknown: oneOf(['error', 'use-encoded-value']) }),
    notes: ['categorical-encoding'],
    cite: ['pedregosa2011'],
  },
  ordinalEncoder,
)

defineModel(
  {
    key: 'targetEncoder',
    module: 'learning/preprocessing',
    name: 'Target encoder',
    summary: 'Each category to a shrunk mean of the target.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({}),
    notes: ['categorical-encoding'],
    cite: ['micci2001'],
  },
  targetEncoder,
)
