/**
 * Categorical encoders: one-hot, ordinal and target encoding, as scikit-learn's `OneHotEncoder`, `OrdinalEncoder` and
 * `TargetEncoder`. Categories are sorted (numbers ascending, strings by code unit), as `np.unique` sorts them.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { values, type FittedTransform, type Invertible, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A category label. */
export type Category = string | number

/**
 * Categorical input: a label list (one column), or a numeric tensor [n] (one column) or [n, d] (d columns of codes).
 */
export type CategoricalInput = Tensor | readonly Category[]

type Columns = { n: number; d: number; columns: Category[][]; list: boolean }

function isTensor(x: CategoricalInput): x is Tensor {
  return !Array.isArray(x)
}

/** Split categorical input into columns of labels. */
function columnsOf(x: CategoricalInput, where: string): Columns {
  if (!isTensor(x)) return { n: x.length, d: 1, columns: [Array.from(x)], list: true }
  if (x.shape.length > 2) throw new ShapeError(where, `${where}: expected [n] or [n, d], got [${x.shape.join(', ')}]`)
  const n = x.shape[0]
  const d = x.shape.length === 1 ? 1 : x.shape[1]
  const v = values(x)
  const columns = Array.from({ length: d }, (_, j) => Array.from({ length: n }, (_, i) => v[i * d + j]))
  return { n, d, columns, list: false }
}

function sortedUnique(c: readonly Category[]): Category[] {
  const unique = Array.from(new Set(c))
  const numeric = unique.every((v) => typeof v === 'number')
  if (!numeric && unique.some((v) => typeof v === 'number'))
    throw new DomainError('encoder', 'encoder: a column mixes numbers and strings')
  return numeric
    ? (unique as number[]).sort((a, b) => a - b)
    : (unique as string[]).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
}

/** Rebuild categorical output of the input's kind from columns of labels. */
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
  /** Index of the category dropped from each column (no indicator), or −1. */
  readonly dropped: readonly number[]
  /** A name per output column, "j=category" (or "category" for a single column). */
  readonly featureNames: readonly string[]
}

/**
 * One indicator column per category of each input column, in the order of the sorted categories. `drop: 'first'`
 * omits each column's first category (avoiding collinearity with an intercept); `'if-binary'` does so only for
 * two-category columns. An unseen category throws, or with `handleUnknown: 'ignore'` encodes as all zeros. `inverse`
 * maps each block back to the category of its largest entry; an all-zero block decodes to the dropped category, or to
 * NaN when there is none (an ignored unknown).
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
  readonly categories: readonly (readonly Category[])[]
}

/**
 * Each category becomes its index 0 … K−1 among its column's sorted categories: output [n, d]. An unseen category
 * throws, or with `handleUnknown: 'use-encoded-value'` becomes `unknownValue` (default NaN).
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
  readonly categories: readonly (readonly Category[])[]
  /** The encoding of each category of each column, aligned with `categories`. */
  readonly encodings: readonly Tensor[]
  /** The mean target ȳ, used for unseen categories. */
  readonly targetMean: number
  /** The smoothing used: a number m, or "auto". */
  readonly smooth: number | 'auto'
}

/**
 * Replace each category c by a shrunk estimate of E[y | c] (Micci-Barreca, 2001, "A preprocessing scheme for
 * high-cardinality categorical attributes in classification and prediction problems", SIGKDD Explorations 3). With a
 * number m: (Σ_{i∈c} yᵢ + m ȳ) / (n_c + m). With `'auto'` (eqs 5–6, scikit-learn's default): λ_c ȳ_c + (1 − λ_c) ȳ with
 * λ_c = σ² n_c / (σ² n_c + s²_c), σ² the population variance of y and s²_c the within-category mean squared deviation.
 * Unseen categories get ȳ. Encoding the training rows with this fitted model leaks their own targets; use
 * `targetEncodeCrossFit` for training features.
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
 * scikit-learn's `KFold` without shuffling), encode each block with an encoder fitted on the other blocks, and also
 * return the encoder fitted on all rows (for new data). Shuffle the rows first if their order carries structure.
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
