/**
 * `aifn-methods/data`: datasets and test problems with known answers, for figures, tests and benchmarks.
 *
 * - `synthetic`: seeded synthetic datasets with known ground truth (point clouds, densities, regression, series,
 *   recommendation, weak supervision, evaluation test beds) and the modifiers that change them (label noise,
 *   prevalence, outliers, missing values, covariate shift).
 * - `real`: small real datasets embedded in the source (`iris`, `oldFaithful`, `karateClub`, `titanic`, ...); larger
 *   ones (an annotated electrocardiogram, fonts, hyphenation) are modules of their own under it.
 * - `signals`: deterministic test signals (chirps, tones, the Donoho–Johnstone functions, a voiced sound), series with
 *   a known power spectrum, and a synthetic electrocardiogram.
 * - `objectives`: test surfaces for optimisers (Rosenbrock, Himmelblau, Beale, a quadratic bowl, Rastrigin), with
 *   gradients, Hessians and known minima.
 * - `targets`: target log-densities for samplers and variational inference (a banana, a Gaussian, a Gaussian mixture,
 *   Neal's funnel), with closed-form gradients and known moments.
 *
 * The shared layer, exported here, holds the `Dataset` shape, the ground truth of the generating processes
 * (`Truth`: `classificationTruth`, `regressionTruth`, `spectralTruth` and the others, each a model with the Bayes rule
 * and the Bayes risk), exact class sizes (`classCounts`) and the recipe interpreter. A generator takes a stream first
 * when it draws, so the same seed gives the same dataset.
 *
 * Registries (design S §3), keyed by `info.key`: `datasetRegistry` (generators of `synthetic`, `real` and `signals`;
 * the font table, a large module, keeps its own `fontDatasetRegistry` in `aifn-methods/data/real/fonts`),
 * `modifierRegistry`, `objectiveRegistry` and `logDensityRegistry`; `generate` and `modify` call their entries.
 * Recipes replay them: `recipe`, `normaliseRecipe`, `describeRecipe`, `encodeRecipe`, `decodeRecipe`, `parseRecipe`,
 * `recipeBases`, `recipeOps` and `recipeSpace` (the space of a base and its knobs, derived from the registry).
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { DatasetEntry, LogDensityEntry, ModifierEntry, ObjectiveEntry } from './define'
import * as objectives from './objectives'
import * as real from './real'
import { recipeBook } from './recipe'
import * as signals from './signals'
import * as synthetic from './synthetic'
import * as targets from './targets'

export {
  additiveTruth,
  armaVariance,
  changepointTruth,
  classificationTruth,
  curve1dTruth,
  inverseTruth,
  regressionTruth,
  regimeTruth,
  spectralTruth,
  twoGaussianBayesError,
  type AdditiveModel,
  type AdditiveTruth,
  type ArmaParts,
  type ChangepointFamily,
  type ChangepointTruth,
  type ClassModel,
  type ClassificationTruth,
  type CoupledSpectra,
  type Curve1dLaw,
  type Curve1dModel,
  type Curve1dTruth,
  type InverseModel,
  type InverseSolution,
  type InverseTruth,
  type LabelOp,
  type Reference,
  type RegressionModel,
  type RegressionTruth,
  type RegimeModel,
  type RegimeTruth,
  type Row,
  type Segment,
  type SpectralLine,
  type SpectralModel,
  type SpectralTruth,
  type Truth,
} from './truth'
export { classCounts, type ClassSizeOptions, type ClassSizes } from './sizes'
export {
  type Dataset,
  type DatasetMeta,
  type PlantedPattern,
  type Recipe,
  type RecipeStep,
  type TableData,
} from './types'
export {
  generate,
  modify,
  type DatasetEntry,
  type LogDensityEntry,
  type ModifierEntry,
  type ObjectiveEntry,
} from './define'
export { type NormalisedRecipe, type RecipeBook, type RecipeInput, type RecipeStepInput } from './recipe'
export { blobs, moons } from './synthetic'

/** Every registered dataset generator (kind `dataset`) of `data/synthetic`, `data/real` and `data/signals`. */
export const datasetRegistry = entries('dataset', synthetic, real, signals) as Readonly<Record<string, DatasetEntry>>

/** Every registered dataset modifier (kind `modifier`). */
export const modifierRegistry = entries('modifier', synthetic) as Readonly<Record<string, ModifierEntry>>

/** Every registered test objective (kind `objective`). */
export const objectiveRegistry = entries('objective', objectives) as Readonly<Record<string, ObjectiveEntry>>

/** Every registered test log density for samplers (kind `log-density`). */
export const logDensityRegistry = entries('log-density', targets) as Readonly<Record<string, LogDensityEntry>>

const book = recipeBook(datasetRegistry, modifierRegistry)

/** The generators a recipe can start from (those returning a `Dataset`), by key. */
export const recipeBases: readonly string[] = book.bases
/** The modifiers a recipe can apply, by key. */
export const recipeOps: readonly string[] = book.ops
/** The space of a recipe's base and its knobs (`base`: a `variants` dimension over `recipeBases`), and its seed. */
export const recipeSpace = book.space
/** Build the dataset a recipe describes; `meta.recipe` holds the normalised recipe, `meta.ignored` what was dropped. */
export const recipe = book.make
/** A recipe with every knob and parameter explicit, and what normalising dropped. */
export const normaliseRecipe = book.normalise
/** One line describing a recipe. */
export const describeRecipe = book.describe
/** A recipe as a compact URL string (default knobs left out). */
export const encodeRecipe = book.encode
/** The normalised recipe in a string from `encodeRecipe`. */
export const decodeRecipe = book.decode
/** Check that a parsed JSON value is a recipe. */
export const parseRecipe = book.parse
