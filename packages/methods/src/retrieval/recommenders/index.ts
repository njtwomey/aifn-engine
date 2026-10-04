/**
 * `aifn-methods/retrieval/recommenders`: recommender models on user–item interactions. Popularity and neighbourhood
 * collaborative filtering; matrix factorisation by SGD and by alternating least squares, and implicit-feedback ALS
 * (Hu–Koren–Volinsky); gradient-trained models (logistic MF, BPR, factorisation machines, field-aware FM, Wide & Deep,
 * DeepFM, NCF, two-tower, SASRec); Matchbox (Bayesian bilinear ordinal ratings with feature traits, by ADF);
 * held-out ranking evaluation; a training run that streams its progress; and a
 * feedback-loop simulator of popularity concentration.
 */

export {
  alternatingLeastSquares,
  biasedFactorInit,
  biasedFactorPredict,
  biasedFactorScorer,
  factorScorer,
  alsFactors,
  implicitAls,
  implicitAlsObjective,
  matrixFactorisationSgd,
  randomFactors,
  type AlsOptions,
  type AlsState,
  type BiasedFactorParams,
  type Factors,
  type ImplicitAlsOptions,
  type MfSgdOptions,
} from './factorisation'
export {
  evaluateRanking,
  interactionMatrix,
  interactionsFromRows,
  itemsByUser,
  topK,
  type Interactions,
  type RankingReport,
  type Scorer,
} from './interactions'
export {
  neuralRecommender,
  type NeuralKind,
  type NeuralOptions,
  type NeuralRecommender,
  type RecommenderContext,
} from './models'
export {
  matchbox,
  matchboxPredict,
  matchboxRatings,
  matchboxRun,
  matchboxUpdate,
  type GaussianTable,
  type Matchbox,
  type MatchboxCheckpoint,
  type MatchboxData,
  type MatchboxOptions,
  type MatchboxPrediction,
  type MatchboxRatingsOptions,
  type MatchboxRun,
  type MatchboxRunOptions,
  type SparseFeatures,
} from './matchbox'
export { itemKnn, popularity, userKnn, type NeighbourhoodOptions } from './neighbourhood'
export {
  principalMap,
  RECOMMENDERS,
  recommenderRun,
  type RecommenderCheckpoint,
  type RecommenderData,
  type RecommenderHistory,
  type RecommenderKind,
  type RecommenderRunOptions,
  type RecommenderSnapshot,
} from './run'
export {
  feedbackLoop,
  worldFromFactors,
  type FeedbackLoopOptions,
  type FeedbackPolicy,
  type FeedbackSnapshot,
  type PolicyHistory,
  type SimulatedWorld,
} from './simulator'
export { recommenderAlgorithms, recommenderFunctions } from './registry'
