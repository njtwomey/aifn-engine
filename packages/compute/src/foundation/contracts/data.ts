/**
 * Datasets, recipes and truths (design S §2.8). One `Dataset` shape for estimators, validation and the generators;
 * a recipe is data that a generic interpreter replays through the dataset registry; a truth is a model of the
 * generating process. `aifn-compute/learning/estimators` re-exports these types, and the generators of
 * `aifn-methods/data` return datasets and truths that extend them.
 */

import type { Distribution } from './distribution'
import type { Kinded } from './kinds'
import type { Decides, Expects, Model, Predicts, Task } from './model'
import type { Tensor, TensorWire } from './numbers'
import type { Info } from './registry'
import type { Space } from './space'

/** One named column of a table: a numeric tensor (`[n]` or `[n, k]`) or a list of category labels. */
export type Column = Tensor | readonly (string | number)[]

/** Named columns of equal length, e.g. `{ age: tensor([...]), city: ['Cork', 'Paris', ...] }`. */
export type Table = { readonly [name: string]: Column }

/** Features: a matrix `[n, d]`, or a table of named columns. */
export type Features = Tensor | Table

/** One step of how a dataset was made: the generator or modifier (`op`) and its parameters. */
export interface RecipeStep {
  /** The registry key of the generator or modifier. */
  op: string
  /** Its parameters, as plain data. */
  params: Record<string, unknown>
}

/**
 * A recipe (target): a registered base generator, a seed, the base's knobs, and modifiers applied in order, each
 * looked up in the dataset registry. Plain data, so it round-trips through a URL.
 */
export interface Recipe {
  /** The registry key of the base generator. */
  readonly base: string
  /** The seed of the root stream, or the path of the stream key the generator was called with. */
  readonly seed: number | string
  /** The base generator's knobs, by name. */
  readonly knobs: Readonly<Record<string, unknown>>
  /** The modifiers applied after the base, in order: each a registry key (`op`) and its `params`. */
  readonly modifiers: readonly { readonly op: string; readonly params: Readonly<Record<string, unknown>> }[]
}

/**
 * The truth of a synthetic problem (target): a model of the generating process, with the Bayes rule (`decide`), the
 * Bayes posterior or the conditional law of $y$ (`predictive`) and the regression function (`expect`), plus the
 * lowest risk any predictor can reach.
 */
export interface Truth<X = Tensor> extends Model, Decides<X>, Predicts<X, Distribution>, Expects<X> {
  /**
   * `changepoint`: a piecewise series whose inputs are times; `decide` gives the segment, `predictive` and `expect`
   * the law and mean of the value at each time. `spectrum`: a stationary series (plus sinusoids) whose inputs are
   * times, with a known power spectrum; `expect` gives the sinusoids at each time, `predictive` the marginal law.
   */
  readonly task: 'classification' | 'regression' | 'changepoint' | 'spectrum'
  /**
   * The Bayes error (classification), the Bayes risk under squared loss (regression), the mean squared error of
   * predicting each value from its segment's true parameters (changepoint), or the variance of the stochastic part
   * (spectrum: no function of time alone predicts it).
   */
  readonly bayesRisk: number
}

/** What a dataset is and where it came from (target). */
export interface DatasetMeta {
  /** A short name, e.g. `moons`. */
  readonly name: string
  /** One or two sentences for a caption: how the data were made and what the labels mean. */
  readonly description: string
  /** What the dataset is for: a model `Task`, or a kind of data no single task covers. */
  readonly task: Task | 'manifold' | 'sequence' | 'images' | 'recommendation' | 'text' | 'decision'
  /** One name per feature (column of `x`). */
  readonly featureNames?: readonly string[]
  /** One name per class, indexed by the integer label. */
  readonly labelNames?: readonly string[]
  /** The name of the target `y`, for axis labels. */
  readonly targetName?: string
  /** A citation for real data or for the generator's recipe. */
  readonly source?: string
  /** Where the data or its description can be found. */
  readonly url?: string
  /** The known generating process, where it has a closed form. */
  readonly truth?: Truth
  /** How the dataset was made: the provenance. */
  readonly recipe?: Recipe
}

/**
 * A dataset (target): features `x` ($n$ rows), optional targets `y`, group labels, a continuous coordinate `t` (for
 * colouring), the noise-free target `f`, and metadata. Row $i$ of every field belongs to example $i$.
 */
export interface Dataset<X extends Features = Features, Y = Tensor> extends Kinded<'dataset'> {
  /** The features, one row per example. */
  readonly x: X
  /** The targets: class labels or real values, one per example. */
  readonly y?: Y
  /** A group label per example, read by the splitters of grouped cross-validation (a group stays on one side). */
  readonly groups?: Column
  /** A continuous coordinate per example (the position along a manifold), for colouring. */
  readonly t?: Tensor
  /** The noise-free target per example (the regression function at `x`). */
  readonly f?: Tensor
  /** What the dataset is and where it came from. */
  readonly meta?: DatasetMeta
}

/**
 * What a dataset generator returns: a `Dataset` (the only form a recipe can start from), several (`datasets`), a
 * sequence with hidden states, a series, an image tensor, an image with the geometry it was drawn from (`scene`), a
 * set of binary patterns, ratings, a click log, logged bandit feedback (`log`), the loss sequence of an online game
 * (`game`), a catalogue, a text corpus, paired views of the same objects (`pairs`: two feature matrices whose row $i$
 * describes the same object, for contrastive and multi-view learning), a split (`split`: a whole population, such as
 * a finite table, with fixed train and test parts and its truth beside them), a stream of paired results with the
 * players' true skills (`matches`, for rating systems), or a table (`table`: named nominal and numeric columns, some
 * of them targets, with any planted patterns as truth, for subgroup discovery).
 */
export type DatasetOutput =
  | 'dataset'
  | 'datasets'
  | 'sequence'
  | 'series'
  | 'image'
  | 'scene'
  | 'patterns'
  | 'ratings'
  | 'clicks'
  | 'log'
  | 'game'
  | 'catalogue'
  | 'corpus'
  | 'pairs'
  | 'split'
  | 'table'
  | 'matches'

/**
 * Registry metadata of a dataset generator: its task, its knobs (the scalar options, with the generator's defaults),
 * whether its datasets carry a truth, and what it returns. A generator with `random: true` is called `(s, knobs)`,
 * any other `(knobs)`.
 */
export interface DatasetInfo extends Info {
  /** The entry kind of a dataset generator. */
  readonly kind: 'dataset'
  /** What its datasets are for, as `DatasetMeta.task`. */
  readonly task: DatasetMeta['task']
  /** Its scalar options, with the generator's defaults. */
  readonly knobs: Space
  /** True when its datasets carry a `truth`. */
  readonly truth: boolean
  /** What it returns. */
  readonly output: DatasetOutput
}

/**
 * Registry metadata of a dataset modifier, called `(s, dataset, params)`: its parameters, and what it needs of the
 * dataset (`labels`: integer class labels in `y`; a recipe skips it, and reports it, on data without them).
 */
export interface ModifierInfo extends Info {
  /** The entry kind of a dataset modifier. */
  readonly kind: 'modifier'
  /** Its parameters, with their defaults. */
  readonly params: Space
  /** `labels` when it needs integer class labels in `y`. */
  readonly needs?: 'labels'
}

/** A dataset on the wire; the truth, which holds functions, stays behind (its recipe rebuilds it). */
export interface DatasetWire {
  /** The brand of a dataset. */
  readonly kind: 'dataset'
  /** The features, as a tensor. */
  readonly x: TensorWire
  /** The targets, when the dataset has them. */
  readonly y?: TensorWire
  /** The metadata without the truth. */
  readonly meta?: Omit<DatasetMeta, 'truth'>
}
