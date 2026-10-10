/**
 * `aifn-methods/retrieval/recommenders`: recommender models on user–item interactions, their held-out evaluation, a
 * streamed training run and a feedback-loop simulator.
 *
 * - Data and evaluation: `Interactions` as parallel index arrays (`interactionsFromRows`), the dense matrix
 *   (`interactionMatrix`) and each user's items (`itemsByUser`); every recommender is a `Scorer`, ranked by `topK` and
 *   measured by `evaluateRanking` (recall@$k$, NDCG@$k$, hit rate, coverage on items not seen in training).
 * - Baselines without training: `popularity`, and neighbourhood collaborative filtering by cosine similarity,
 *   `userKnn` and `itemKnn`.
 * - Matrix factorisation: explicit ratings by alternating least squares (`alternatingLeastSquares`, ALS-WR) or by SGD
 *   on the biased model (`matrixFactorisationSgd`, with `biasedFactorInit`, `biasedFactorPredict` and
 *   `biasedFactorScorer`); implicit feedback by Hu–Koren–Volinsky ALS (`implicitAls`, objective
 *   `implicitAlsObjective`), as the `implicit` library fits it. `randomFactors`, `alsFactors` and `factorScorer` turn
 *   factors into scorers.
 * - Gradient-trained models: `neuralRecommender` builds logistic MF, BPR, factorisation machines, field-aware FM, Wide
 *   & Deep, DeepFM, NCF, two-tower and SASRec as an initialiser, training rows, loss and scorer.
 * - Matchbox (Stern, Herbrich and Graepel, 2009), Bayesian bilinear ordinal ratings with feature traits, learned by
 *   ADF: the prior `matchbox`, `matchboxPredict` and `matchboxUpdate`, synthetic data from `matchboxRatings`, and the
 *   streamed run `matchboxRun` with cold-start users.
 * - Runs and simulation: `recommenderRun` trains any of `RECOMMENDERS` epoch by epoch, streaming held-out metrics,
 *   scores and `principalMap` embedding maps; `feedbackLoop` simulates policies retrained on their own clicks in a
 *   world from `worldFromFactors`, tracking exposure concentration.
 * - Registries: `recommenderAlgorithms` (the step-through factorisations) and `recommenderFunctions`.
 *
 * Users and items are 0-based indices, scores and factors row-major arrays with a row per user or item. Every random
 * draw comes from an explicit stream or seed, so runs are reproduced exactly.
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
