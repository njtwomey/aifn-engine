/**
 * `aifn-methods/learning`: supervised models over the estimator protocol of `aifn-compute/learning/estimators`, as
 * scikit-learn's estimators with lifelines' survival models and statsmodels' GLMs and GAMs.
 *
 * - `linear`: least squares and ridge regression, and the perceptron.
 * - `generalised`: generalised models on a likelihood, with penalised IRLS and smoothing selection shared by its
 *   children `glm` (any family and link, logistic and multinomial regression), `gam` (additive models, GAMLSS) and
 *   `ordinal` (cumulative-link and threshold models of ordered labels).
 * - `generative-classifiers`: naive Bayes, and linear and quadratic discriminant analysis.
 * - `kernel-methods`: support vector machines and the Crammer–Singer multiclass SVM.
 * - `gaussian-processes`: Gaussian process regression, classification and ordinal regression, sparse approximations,
 *   the relevance vector machine and the GP latent variable model.
 * - `neighbours`: $k$-nearest-neighbour classification and regression.
 * - `trees-and-ensembles`: CART trees, with the children `bagging` (bagging and random forests) and `boosting`
 *   (AdaBoost and gradient boosting).
 * - `reductions`: multiclass classifiers from any binary one (one-versus-rest, one-versus-one, output codes, nested
 *   dichotomies).
 * - `mixture-density`: the mixture density network, for targets with several branches given the input.
 * - `mixture-of-experts`: mixtures of experts and their hierarchical form, fitted by EM or Adam.
 * - `survival`: Cox proportional-hazards and accelerated-failure-time models of right-censored times.
 * - `preprocessing`: scalers, encoders, imputers, feature maps and resamplers for imbalanced classes.
 * - `weak-supervision`: learning without clean labels (label models, positive-unlabelled data, label proportions,
 *   multiple instances, confident learning).
 * - `transfer`: domain adaptation, label shift, continual learning and meta-learning on small problems.
 * - `explanation`: studies that compare data-attribution methods against planted label noise.
 *
 * The family itself re-exports the most used estimators of its modules, and `isotonicRegressor`, isotonic regression
 * of one feature. `learningModelRegistry` lists the estimator factories registered by the modules it gathers
 * (supervised models and preprocessing transformers), keyed by `info.key`.
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { ModelEntry } from 'aifn-compute/learning/estimators'
import * as gam from './generalised/gam'
import * as glm from './generalised/glm'
import * as ordinal from './generalised/ordinal'
import * as gaussianProcesses from './gaussian-processes'
import * as generativeClassifiers from './generative-classifiers'
import * as kernelMethods from './kernel-methods'
import * as linear from './linear'
import * as mixtureDensity from './mixture-density'
import * as neighbours from './neighbours'
import * as preprocessing from './preprocessing'
import * as reductions from './reductions'
import * as trees from './trees-and-ensembles'
import * as bagging from './trees-and-ensembles/bagging'
import * as boosting from './trees-and-ensembles/boosting'

export { glm } from './generalised/glm'
export { gam } from './generalised/gam'
export {
  binaryDecomposition,
  deepOrdinalRegression,
  ordinalRegression,
  thresholdOrdinalRegression,
} from './generalised/ordinal'
export { linearRegression, perceptron } from './linear'
export { gaussianNaiveBayes, linearDiscriminant, quadraticDiscriminant } from './generative-classifiers'
export { supportVectorMachine } from './kernel-methods'
export { gaussianProcessRegressor, gpClassifier, gpOrdinalRegression } from './gaussian-processes'
export { randomForest } from './trees-and-ensembles/bagging'
export { adaBoost, gradientBoosting } from './trees-and-ensembles/boosting'
export { kNearestNeighbours } from './neighbours'
export { mixtureDensityNetwork } from './mixture-density'
export { oneVersusRest } from './reductions'
export { standardScaler } from './preprocessing'
export { isotonicRegressor, type IsotonicParams, type IsotonicRegressor } from './isotonic'

/** Every registered estimator factory of `aifn-methods/learning`, keyed by `info.key` (kind `model`). */
export const learningModelRegistry = entries(
  'model',
  linear,
  glm,
  gam,
  ordinal,
  generativeClassifiers,
  kernelMethods,
  gaussianProcesses,
  trees,
  bagging,
  boosting,
  neighbours,
  reductions,
  preprocessing,
  mixtureDensity,
) as Readonly<Record<string, ModelEntry>>
