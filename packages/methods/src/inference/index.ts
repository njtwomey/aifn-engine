/**
 * `aifn-methods/inference`: named probabilistic models, built on the engines of `aifn-compute/inference` (exact chain
 * and tree inference, belief propagation, expectation propagation, variational inference, Gibbs sampling).
 *
 * - `sequence-models`: hidden Markov models (the occasionally dishonest casino) and linear-chain CRFs, over dense
 *   features or CRF++-style feature templates, with their training.
 * - `topic-models`: topic models of bag-of-words corpora: LDA by collapsed Gibbs sampling, the HDP, pLSA, a correlated
 *   and a dynamic topic model, and LSA and NMF topics.
 * - `rating-models`: skill ratings from games and answers: Elo, Glicko, Bradley–Terry, Plackett–Luce, item response
 *   theory and TrueSkill, online and in batch.
 * - `learner-models`: zero-inflated learner models (IRT-ZILM) for equitable ability estimation, with their baselines,
 *   equity measures and streamed experiments.
 * - `lattice-models`: the Ising model on a graph or lattice, with inference chosen by the graph's shape.
 * - `mixture-models`: the variational Bayesian Gaussian mixture, and Minka's clutter problem for expectation
 *   propagation.
 * - `conjugate-models`: coordinate-ascent variational inference for the normal-gamma model, beside its exact posterior.
 * - `classifier-models`: Bayesian linear classifiers: the Bayes point machine by expectation propagation, and
 *   AdPredictor by assumed-density filtering.
 *
 * The family's root re-exports one entry point of four of them: `dishonestCasino` and `linearChainCrf`, `ldaModel`,
 * `trueSkillUpdate` and `fitLearnerModel`.
 */

export { dishonestCasino, linearChainCrf } from './sequence-models'
export { ldaModel } from './topic-models'
export { trueSkillUpdate } from './rating-models'
export { fitLearnerModel } from './learner-models'
