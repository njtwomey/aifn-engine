/**
 * Datasets, recipes and truths (design S §2.8). One `Dataset` shape for estimators, validation and the generators;
 * a recipe is data that a generic interpreter replays through the dataset registry; a truth is a model of the
 * generating process. Today two dataset shapes exist (`aifn-compute/learning/estimators`' generic one and `aifn-methods/data`' concrete
 * one), and truths are bespoke objects; phase 1 moves both onto these.
 */

import type { Distribution } from './distribution'
import type { Kinded } from './kinds'
import type { Decides, Expects, Model, Predicts, Task } from './model'
import type { Tensor, TensorWire } from './numbers'
import type { Info } from './registry'
import type { Space } from './space'

/** One named column of a table: a numeric tensor ([n] or [n, k]) or a list of category labels. */
export type Column = Tensor | readonly (string | number)[]

/** Named columns of equal length, e.g. `{ age: tensor([...]), city: ['Cork', 'Paris', ...] }`. */
export type Table = { readonly [name: string]: Column }

/** Features: a matrix [n, d], or a table of named columns. */
export type Features = Tensor | Table

/** One step of how a dataset was made: the generator or modifier (`op`) and its parameters. */
export interface RecipeStep {
  op: string
  params: Record<string, unknown>
}

/**
 * A recipe (target): a registered base generator, a seed, the base's knobs, and modifiers applied in order, each
 * looked up in the dataset registry. Plain data, so it round-trips through a URL.
 */
export interface Recipe {
  readonly base: string
  readonly seed: number | string
  readonly knobs: Readonly<Record<string, unknown>>
  readonly modifiers: readonly { readonly op: string; readonly params: Readonly<Record<string, unknown>> }[]
}

/**
 * The truth of a synthetic problem (target): a model of the generating process, with the Bayes rule (`decide`), the
 * Bayes posterior or the conditional law of y (`predictive`) and the regression function (`expect`), plus the lowest
 * risk any predictor can reach.
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
  readonly task: Task | 'manifold' | 'sequence' | 'images' | 'recommendation' | 'text' | 'decision'
  readonly featureNames?: readonly string[]
  readonly labelNames?: readonly string[]
  readonly targetName?: string
  /** A citation for real data or for the generator's recipe. */
  readonly source?: string
  readonly url?: string
  /** The known generating process, where it has a closed form. */
  readonly truth?: Truth
  /** How the dataset was made: the provenance. */
  readonly recipe?: Recipe
}

/**
 * A dataset (target): features `x` (n rows), optional targets `y`, group labels, a continuous coordinate `t` (for
 * colouring), the noise-free target `f`, and metadata. Row i of every field belongs to example i.
 */
export interface Dataset<X extends Features = Features, Y = Tensor> extends Kinded<'dataset'> {
  readonly x: X
  readonly y?: Y
  readonly groups?: Column
  readonly t?: Tensor
  readonly f?: Tensor
  readonly meta?: DatasetMeta
}

/**
 * What a dataset generator returns: a `Dataset` (the only form a recipe can start from), several (`datasets`), a
 * sequence with hidden states, a series, an image tensor, an image with the geometry it was drawn from (`scene`), a set of binary patterns, ratings, a click log, logged
 * bandit feedback (`log`), the loss sequence of an online game (`game`), a catalogue, a text corpus, paired views of
 * the same objects (`pairs`: two feature matrices whose row i describes the same object, for contrastive and
 * multi-view learning), a split (`split`: a whole population, such as a finite table, with fixed train and test
 * parts and its truth beside them), a stream of paired results with the players' true skills (`matches`, for
 * rating systems), or a table (`table`: named nominal and numeric columns, some of them targets, with any planted
 * patterns as truth, for subgroup discovery).
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
  readonly kind: 'dataset'
  readonly task: DatasetMeta['task']
  readonly knobs: Space
  readonly truth: boolean
  readonly output: DatasetOutput
}

/**
 * Registry metadata of a dataset modifier, called `(s, dataset, params)`: its parameters, and what it needs of the
 * dataset (`labels`: integer class labels in `y`; a recipe skips it, and reports it, on data without them).
 */
export interface ModifierInfo extends Info {
  readonly kind: 'modifier'
  readonly params: Space
  readonly needs?: 'labels'
}

/** A dataset on the wire; the truth, which holds functions, stays behind (its recipe rebuilds it). */
export interface DatasetWire {
  readonly kind: 'dataset'
  readonly x: TensorWire
  readonly y?: TensorWire
  readonly meta?: Omit<DatasetMeta, 'truth'>
}
