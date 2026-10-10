/**
 * `aifn-methods/inference/topic-models`: topic models of bag-of-words corpora, by Gibbs sampling, EM and matrix
 * factorisation.
 *
 * - Latent Dirichlet allocation: `ldaCollapsedGibbs` samples the topic assignments with $\thetavec$ and $\phivec$
 *   integrated out (labelled LDA through each document's allowed topics), and `ldaEstimates` reads the topics and
 *   proportions from a state. `ldaModel` is LDA in the model language, `matchLda` recognises LDA-shaped models,
 *   `ldaOptions` reads their problem from bindings, and `ldaEngine` (in the table `ldaEngines`) lets `infer` run them.
 * - An unknown number of topics: `hdpGibbs`, the hierarchical Dirichlet process by direct-assignment Gibbs sampling,
 *   with `hdpEstimates`.
 * - Other static models: `plsaSteps` (pLSA by EM), `correlatedTopicSteps` (a correlated topic model by MAP EM, with
 *   `topicCorrelations`), `lsaTopics` (a truncated SVD, signed) and NMF (`nmfTopicSteps`, normalised by `nmfTopics`,
 *   or run to the end by `nmfTopicModel`).
 * - Topics that drift over time slices: `dynamicTopicSteps`, a dynamic topic model by MAP EM, and `dynamicTopicRun`,
 *   which streams its fit.
 * - Corpora and evaluation: `bagOfWordsCorpus` turns texts into word ids, `groupByLabel` joins short texts,
 *   `documentTermCounts` counts them, `topWords` lists each topic's heaviest words, and `logLikelihoodPerToken` scores
 *   a fit. `topicModelRun` streams any static model (`TopicMethod`, listed in `TOPIC_METHODS`) with its log-likelihood
 *   and NPMI coherence.
 * - `topicModelAlgorithms` and `topicModelFunctions` are the module's registry entries.
 *
 * A corpus is `Documents`, arrays of word ids in $\{0, \dots, V - 1\}$. Every static model reports a $K \times V$
 * topic-word matrix and a $D \times K$ document-topic matrix (the dynamic one a topic-word matrix per slice); the
 * step-through ones are `Algorithm`s run with no start, whose random starts and draws come from the run's stream. None
 * is differentiable.
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
