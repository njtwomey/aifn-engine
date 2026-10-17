/**
 * `aifn-methods/learning/semi-supervised`: learning from a few labelled points and many unlabelled ones, by wrapping a
 * supervised learner or by changing its objective.
 *
 * - Wrappers around any probabilistic classifier: self-training, pseudo-labelling and Noisy Student
 *   (`selfTrainingSteps`, with the student noise `inputJitter` and `featureDropout`), co-training on two views
 *   (`coTrainingSteps`, combined by `productOfExperts`) and tri-training (`triTrainingSteps`, with its update rule
 *   `triTrainingRule` and the vote `majority`); the row helpers `rowArgmax`, `probabilities`, `accuracyOf` and
 *   `labelledSet`.
 * - The cluster assumption: cluster-then-label names each cluster of any clustering by the labels
 *   (`clusterClasses`, `clusterThenLabelProbabilities`).
 * - Thresholds on a line, where the value of unlabelled data is exact: the supervised `thresholdErm` (on
 *   `ermInterval`), the semi-supervised `lowDensityThreshold` on the troughs of `unlabelledDensity`, Kääriäinen's
 *   `medianThreshold`, and `thresholdErrors`.
 * - Manifold regularisation: `laplacianRegularisedLeastSquares`, kernel least squares with a graph penalty.
 * - Open-set filtering: `gaussianNoveltyScores`, the distance of each point to the nearest labelled class mean, with a
 *   chi-squared p-value.
 * - Low-density separation on fixed features: the transductive SVM in the primal with continuation
 *   (`transductiveSvmObjective`, `transductiveSvmFit`, `transductiveSvmStages`, `transductiveSvmDecision`, and the
 *   unlabelled losses `symmetricHinge`, `symmetricSigmoid`, `symmetricRamp`); entropy minimisation for a multinomial
 *   logistic model (`entropyMinimisationObjective`, `entropyMinimisationFit`, the warm-started
 *   `entropyMinimisationPath`, `entropyMinimisationProbabilities`); and the density-sensitive $\rho$-path distance
 *   (`rhoPathDistances`, `rhoPathExtend` to new points, `rhoQueryNeighbours`, `rhoPathNodes`, with `rhoEdgeCost` and
 *   `rhoDistance`). These take and return plain arrays, so a worker can run them.
 *
 * Classes are $0, \dots, K - 1$ and $-1$ marks an unlabelled point. Class probabilities are row-major $n \times K$
 * `Float64Array`s. Stepped methods are `Algorithm`s whose states are plain data, run with `trace` and drawing from the
 * step's stream.
 */

export {
  accuracyOf,
  labelledSet,
  probabilities,
  rowArgmax,
  selfTrainingSteps,
  type SelfTrainingOptions,
  type SelfTrainingProblem,
  type SelfTrainingState,
  type TrainingSet,
} from './self-training'
export {
  coTrainingSteps,
  majority,
  productOfExperts,
  triTrainingRule,
  triTrainingSteps,
  type CoTrainingOptions,
  type CoTrainingProblem,
  type CoTrainingState,
  type TriTrainingDecision,
  type TriTrainingProblem,
  type TriTrainingState,
} from './co-training'
export { featureDropout, inputJitter, type FeatureDropoutOptions } from './student-noise'
export { clusterClasses, clusterThenLabelProbabilities } from './cluster-then-label'
export {
  ermInterval,
  lowDensityThreshold,
  medianThreshold,
  thresholdErm,
  thresholdErrors,
  unlabelledDensity,
  type GridDensity,
  type LineSample,
  type ThresholdInterval,
} from './threshold-learning'
export { laplacianRegularisedLeastSquares, type LapRlsModel, type LapRlsOptions } from './laplacian-rls'
export { gaussianNoveltyScores, type GaussianNovelty } from './gaussian-novelty'
export {
  symmetricHinge,
  symmetricRamp,
  symmetricSigmoid,
  transductiveSvmDecision,
  transductiveSvmFit,
  transductiveSvmObjective,
  transductiveSvmStages,
  type FeatureRows,
  type TransductiveSvmProblem,
  type TransductiveSvmSolution,
  type TransductiveSvmWeights,
} from './transductive-svm'
export {
  entropyMinimisationFit,
  entropyMinimisationObjective,
  entropyMinimisationPath,
  entropyMinimisationProbabilities,
  type EntropyMinimisationFit,
  type EntropyMinimisationProblem,
  type EntropyMinimisationWeights,
} from './entropy-minimisation'
export {
  rhoDistance,
  rhoEdgeCost,
  rhoPathDistances,
  rhoPathExtend,
  rhoPathNodes,
  rhoQueryNeighbours,
  type PointRows,
  type QueryNeighbours,
  type RhoPathDistances,
  type RhoPathOptions,
} from './density-sensitive-distance'
export { semiSupervisedAlgorithms, semiSupervisedFunctions } from './registry'
