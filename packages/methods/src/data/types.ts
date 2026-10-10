/**
 * The dataset shape every generator returns, and small private helpers for building it. A dataset is the contract's
 * `Dataset` (`kind: 'dataset'`, tensors plus metadata) with float64 features `x` ($n \times d$); its metadata adds the
 * per-row records that modifiers keep (clean labels, outlier and missingness masks) and the recipe knobs a generator
 * could not use. The table that pattern-mining generators return instead, `TableData`, is here too.
 */

import type {
  Dataset as DatasetContract,
  DatasetMeta as DatasetMetaContract,
  Key,
  Recipe,
  RecipeStep,
  Table,
} from 'aifn-compute/foundation/contracts'
import type { Description } from 'aifn-compute/learning/subgroups'
import { copy, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Truth } from './truth'
import { DomainError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Recipe, RecipeStep } from 'aifn-compute/foundation/contracts'

/** What a dataset is and where it came from: the contract's metadata plus the records modifiers keep. */
export interface DatasetMeta extends DatasetMetaContract {
  /** One name per column of `x`. */
  readonly featureNames: readonly string[]
  /**
   * The known generating process, where it has a closed form, as a model: the Bayes rule and posterior for
   * classification, the regression function and noise law for regression. Modifiers keep it consistent.
   */
  readonly truth?: Truth
  /**
   * How the dataset was made: the generator (`base`), its knobs, and every modifier applied since, in order. `seed` is
   * the seed of `recipe()`, or the path of the stream key the generator was called with.
   */
  readonly recipe?: Recipe
  /** The key of the stream the data were drawn from (synthetic data only). */
  readonly key?: Key
  /** Recipe knobs the generator could not use (e.g. `separation` for moons), reported rather than dropped. */
  readonly ignored?: readonly string[]
  /** Labels before `withLabelNoise` (int32, length n); `y` holds the observed, possibly flipped, labels. */
  readonly cleanLabels?: Tensor
  /** 1 for rows replaced by `withOutliers` (int32, length n). */
  readonly outliers?: Tensor
  /** 1 where `withMissing` removed an entry (int32, $n \times d$); those entries of `x` are NaN. */
  readonly missing?: Tensor
  /** `x` before `withMissing`, for drawing where the missing values were. */
  readonly complete?: Tensor
}

/**
 * A dataset: features `x` ($n \times d$, float64), optional labels or targets `y` (length $n$: int32 class indices, or
 * float64 targets), and metadata. Generators may add `t`, a continuous coordinate along a manifold for colouring, or
 * `f`, the noise-free regression function at `x`.
 */
export interface Dataset extends DatasetContract<Tensor, Tensor> {
  /** What the dataset is, where it came from and what modifiers recorded. */
  readonly meta: DatasetMeta
}

/**
 * The contract recipe of a generator call: its name, the stream it drew from and its plain knobs, with no modifiers.
 *
 * @param base The generator's registry key.
 * @param key The key of the stream the generator drew from; its path is the recipe's seed (0 without a key).
 * @param knobs The knobs the generator used, by name (stored, not copied).
 * @returns The recipe.
 */
export function generatorRecipe(base: string, key: Key | undefined, knobs: Record<string, unknown>): Recipe {
  return { base, seed: key?.path ?? 0, knobs, modifiers: [] }
}

/**
 * The recipe with one more modifier step, as a new recipe.
 *
 * @param recipe The dataset's recipe so far; without one, a recipe of base `'data'`, seed 0 and no knobs is started.
 * @param step The modifier's key and parameters.
 * @returns The recipe with `step` appended to its modifiers.
 */
export function appendStep(recipe: Recipe | undefined, step: RecipeStep): Recipe {
  const r = recipe ?? { base: 'data', seed: 0, knobs: {}, modifiers: [] }
  return { ...r, modifiers: [...r.modifiers, step] }
}

/**
 * A float64 matrix from a row-major array.
 *
 * @param data The entries, row by row: `rows * cols` values (used as they are, not copied).
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @returns The `rows` by `cols` matrix.
 */
export function matrix(data: Float64Array, rows: number, cols: number): Tensor {
  return fromData(data, [rows, cols])
}

/**
 * A float64 vector.
 *
 * @param data The values (copied).
 * @returns The vector.
 */
export function vector(data: ArrayLike<number>): Tensor {
  return fromData(Float64Array.from(data))
}

/**
 * An int32 vector, as labels are stored.
 *
 * @param data The values (copied, and truncated to 32-bit integers).
 * @returns The vector.
 */
export function labels(data: ArrayLike<number>): Tensor {
  return fromData(Int32Array.from(data))
}

/**
 * Values of a tensor or array as a fresh Float64Array (row-major).
 *
 * @param x A tensor of any shape, or an array of numbers.
 * @returns A copy of the values.
 */
export function values(x: Tensor | ArrayLike<number>): Float64Array {
  return isTensor(x) ? (copy(x, 'float64').data as Float64Array) : Float64Array.from(x)
}

/**
 * Split `n` into `k` near-equal group sizes (the first $n \bmod k$ groups get one extra), as scikit-learn does.
 *
 * @param n The total to split.
 * @param k The number of groups.
 * @returns The $k$ sizes, summing to `n`.
 */
export function splitSizes(n: number, k: number): number[] {
  const base = Math.floor(n / k)
  return Array.from({ length: k }, (_, j) => base + (j < n % k ? 1 : 0))
}

/**
 * Throw `DomainError` unless `n` is a non-negative integer.
 *
 * @param n The count to check.
 * @param what The caller's name, for the error message.
 */
export function checkCount(n: number, what: string): void {
  if (!Number.isInteger(n) || n < 0) throw new DomainError(what, `${what}: n must be a non-negative integer, got ${n}`)
}

/** A pattern planted in a table: the description, which targets it affects, and how (for captions and ranks). */
export interface PlantedPattern {
  /** The subgroup it covers, as a description over the table's columns. */
  readonly description: Description
  /** The target columns it changes. */
  readonly targets: readonly string[]
  /** One sentence: what is unusual inside it. */
  readonly effect: string
}

/**
 * A table for pattern mining (`output: 'table'`): named nominal (string) and numeric columns, the names of the target
 * columns among them, and the patterns planted in it (empty for real data).
 */
export interface TableData {
  /** Marks a table, as against a dataset. */
  readonly kind: 'table'
  /** The named columns. */
  readonly table: Table
  /** The names of the target columns. */
  readonly targets: readonly string[]
  /** The patterns planted in the table, empty for real data. */
  readonly planted: readonly PlantedPattern[]
  /** Its `name` and one-line `description`, and for real data its `source` and `url`. */
  readonly meta: {
    readonly name: string
    readonly description: string
    readonly source?: string
    readonly url?: string
  }
}
