/**
 * `aifn-methods/inference`: named probabilistic models on the engines of `aifn-compute/inference`: sequence models (HMM
 * problems, the linear-chain CRF), topic models (LDA), rating models (TrueSkill), learner models (IRT-ZILM), lattice
 * models (Ising), mixture models, conjugate models and classifier models (the Bayes point machine).
 */

export { dishonestCasino, linearChainCrf } from './sequence-models'
export { ldaModel } from './topic-models'
export { trueSkillUpdate } from './rating-models'
export { fitLearnerModel } from './learner-models'
