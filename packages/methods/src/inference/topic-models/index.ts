/**
 * `aifn-methods/inference/topic-models`: topic models. Latent Dirichlet allocation (the model, its match, collapsed
 * Gibbs sampling, labelled LDA through allowed topic sets, estimates, and its engine `ldaEngine` / `ldaEngines` for
 * `infer`); pLSA by EM; LSA and NMF topics; a correlated topic model (MAP EM); and the hierarchical Dirichlet process (`hdpGibbs`, the number of topics
 * inferred); a dynamic topic model whose topics drift over time slices (`dynamicTopicSteps`, `dynamicTopicRun`); and
 * `topicModelRun`, which streams any of the static models with its log-likelihood and NPMI coherence.
 */
export {
  ldaCollapsedGibbs,
  ldaEngine,
  ldaEngines,
  ldaEstimates,
  ldaModel,
  ldaOptions,
  matchLda,
  type LdaOptions,
  type LdaState,
} from './lda'
export {
  bagOfWordsCorpus,
  documentTermCounts,
  groupByLabel,
  topWords,
  type BagOfWordsCorpus,
  type Documents,
} from './corpus'
export { correlatedTopicSteps, topicCorrelations, type CorrelatedTopicOptions, type CorrelatedTopicState } from './ctm'
export { lsaTopics, nmfTopicModel, nmfTopics, nmfTopicSteps, type FactorTopics, type NmfTopicOptions } from './factor'
export {
  dynamicTopicRun,
  dynamicTopicSteps,
  type DynamicTopicCheckpoint,
  type DynamicTopicOptions,
  type DynamicTopicRunOptions,
  type DynamicTopicSnapshot,
  type DynamicTopicState,
} from './dtm'
export { hdpEstimates, hdpGibbs, type HdpOptions, type HdpState } from './hdp'
export { plsaSteps, type PlsaOptions, type PlsaState } from './plsa'
export {
  logLikelihoodPerToken,
  TOPIC_METHODS,
  topicModelRun,
  type TopicCheckpoint,
  type TopicMethod,
  type TopicModelRunOptions,
  type TopicSnapshot,
} from './run'
export { topicModelAlgorithms, topicModelFunctions } from './registry'
