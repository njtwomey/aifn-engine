/**
 * `aifn-methods/learning`: supervised models over the estimator protocol of `aifn-compute/learning/estimators`. Groups:
 * generalised (glm, gam, ordinal), trees-and-ensembles (bagging, boosting); modules: linear, generative-classifiers,
 * kernel-methods, gaussian-processes, neighbours, reductions, preprocessing, mixture-density. `learningModelRegistry` lists every
 * registered estimator factory of the area (supervised models and preprocessing transformers) by key.
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
