/**
 * The registry of `aifn-methods/inference/topic-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as corpus from './corpus'
import * as ctm from './ctm'
import * as dtm from './dtm'
import * as factor from './factor'
import * as hdp from './hdp'
import * as lda from './lda'
import * as plsa from './plsa'
import * as run from './run'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/topic-models')
const fn = definer<FunctionInfo>('function', 'inference/topic-models')
const notes = ['latent-dirichlet-allocation']

algorithm(
  {
    key: 'ldaCollapsedGibbs',
    name: 'LDA by collapsed Gibbs sampling',
    summary: 'Resamples each token’s topic from its full conditional with θ and φ integrated out; one sweep per step.',
    problem: 'factor-graph',
    state: { iterate: 'docTopic', objective: 'logLikelihood', flags: [] },
    random: true,
    notes: [...notes, 'gibbs-sampling'],
    cite: ['griffiths2004', 'blei2003'],
  },
  lda.ldaCollapsedGibbs,
)
algorithm(
  {
    key: 'hdpGibbs',
    name: 'HDP topic model by direct-assignment Gibbs sampling',
    summary:
      'LDA with unboundedly many topics: tokens join used topics or open new ones, and the shared topic weights are redrawn from table counts.',
    problem: 'corpus',
    state: { iterate: 'docTopic', objective: 'logLikelihood', flags: [] },
    random: true,
    notes: ['hierarchical-dirichlet-process', 'gibbs-sampling'],
    cite: ['teh2006hdp'],
  },
  hdp.hdpGibbs,
)
fn(
  {
    key: 'hdpEstimates',
    name: 'HDP point estimates',
    summary:
      'Topic–word distributions and document proportions over the topics in use, from the counts of an HDP state.',
    role: 'estimator',
    notes: ['hierarchical-dirichlet-process'],
    cite: ['teh2006hdp'],
  },
  hdp.hdpEstimates,
)
algorithm(
  {
    key: 'dynamicTopicSteps',
    name: 'Dynamic topic model (MAP variational EM)',
    summary:
      'Topics drift between time slices as a Gaussian random walk in their natural parameters; variational LDA per document, each topic chain at its posterior mode.',
    problem: 'corpus',
    state: { iterate: 'docTopic', objective: 'objective', flags: ['diverged'] },
    random: true,
    notes: ['dynamic-topic-model'],
    cite: ['blei2006dtm'],
  },
  dtm.dynamicTopicSteps,
)
fn(
  {
    key: 'dynamicTopicRun',
    name: 'Dynamic topic model run',
    summary: 'Fit the dynamic topic model step by step, streaming the bound, the fit and each slice’s topics.',
    role: 'simulation',
    random: true,
    notes: ['dynamic-topic-model'],
    cite: ['blei2006dtm'],
  },
  dtm.dynamicTopicRun,
)
fn({ key: 'ldaModel', name: 'LDA model specification', role: 'construction', notes, cite: ['blei2003'] }, lda.ldaModel)
fn({ key: 'matchLda', name: 'Match a model to LDA', role: 'property', notes }, lda.matchLda)
fn({ key: 'ldaOptions', name: 'LDA options from a model', role: 'construction', notes }, lda.ldaOptions)
fn(
  {
    key: 'ldaEstimates',
    name: 'LDA point estimates',
    summary: 'Posterior-mean topic–word and document–topic matrices from the counts of a Gibbs state.',
    role: 'estimator',
    notes,
    cite: ['griffiths2004'],
  },
  lda.ldaEstimates,
)

algorithm(
  {
    key: 'plsaSteps',
    name: 'pLSA by EM',
    summary: 'Alternate topic responsibilities of each (document, word) pair and the two conditional distributions.',
    problem: 'corpus',
    state: { iterate: 'docTopic', objective: 'logLikelihood', flags: ['converged'] },
    random: true,
    notes: ['probabilistic-latent-semantic-analysis', 'expectation-maximisation'],
    cite: ['hofmann1999'],
  },
  plsa.plsaSteps,
)
algorithm(
  {
    key: 'correlatedTopicSteps',
    name: 'Correlated topic model (MAP EM)',
    summary: 'Logistic-normal topic proportions: posterior modes per document by L-BFGS, then topics and (μ, Σ).',
    problem: 'corpus',
    state: { iterate: 'eta', objective: 'logLikelihood', flags: ['diverged'] },
    random: true,
    notes: ['correlated-topic-model'],
    cite: ['blei2007ctm'],
  },
  ctm.correlatedTopicSteps,
)
fn(
  {
    key: 'topicCorrelations',
    name: 'Topic correlations',
    summary: 'The correlation matrix of a correlated topic model’s logistic-normal covariance.',
    role: 'property',
    notes: ['correlated-topic-model'],
  },
  ctm.topicCorrelations,
)
fn(
  {
    key: 'lsaTopics',
    name: 'LSA topics',
    summary: 'Topics as the signed term loadings of a truncated SVD of the term × document counts.',
    role: 'fit',
    notes: ['latent-semantic-analysis'],
    cite: ['deerwester1990'],
  },
  factor.lsaTopics,
)
fn(
  {
    key: 'nmfTopicSteps',
    name: 'NMF topics, stepped',
    summary: 'Multiplicative-update NMF of the document × term counts (KL by default), whose parts read as topics.',
    role: 'construction',
    notes: ['non-negative-matrix-factorisation', 'topic-model-family'],
    cite: ['lee1999'],
  },
  factor.nmfTopicSteps,
)
fn(
  {
    key: 'nmfTopics',
    name: 'NMF topic distributions',
    role: 'transform',
    notes: ['non-negative-matrix-factorisation'],
  },
  factor.nmfTopics,
)
fn(
  {
    key: 'nmfTopicModel',
    name: 'NMF topic model',
    role: 'fit',
    notes: ['non-negative-matrix-factorisation'],
    cite: ['lee1999'],
  },
  factor.nmfTopicModel,
)
fn(
  { key: 'documentTermCounts', name: 'Document × term counts', role: 'transform', notes: ['latent-semantic-analysis'] },
  corpus.documentTermCounts,
)
fn(
  {
    key: 'groupByLabel',
    name: 'Documents from labelled sentences',
    summary: 'Join short texts sharing a label into longer documents with one dominant topic.',
    role: 'transform',
    notes: ['latent-dirichlet-allocation'],
  },
  corpus.groupByLabel,
)
fn(
  {
    key: 'bagOfWordsCorpus',
    name: 'Bag-of-words corpus',
    summary: 'Texts as lower-cased word-id documents without stop words, and their vocabulary.',
    role: 'transform',
    notes: ['latent-dirichlet-allocation'],
  },
  corpus.bagOfWordsCorpus,
)
fn(
  { key: 'topWords', name: 'Top words of each topic', role: 'property', notes: ['topic-model-evaluation'] },
  corpus.topWords,
)
fn(
  {
    key: 'logLikelihoodPerToken',
    name: 'Log-likelihood per token',
    tex: '\\frac{1}{N}\\sum_d \\sum_w \\log \\sum_k \\theta_{dk} \\phi_{kw}',
    role: 'estimator',
    notes: ['topic-model-evaluation'],
  },
  run.logLikelihoodPerToken,
)
fn(
  {
    key: 'topicModelRun',
    name: 'Topic model run',
    summary:
      'Fit LDA, the HDP, labelled LDA, pLSA, NMF, LSA or the CTM step by step, streaming fit, coherence and topics.',
    role: 'simulation',
    random: true,
    notes: [
      'topic-model-family',
      'topic-model-evaluation',
      'labelled-latent-dirichlet-allocation',
      'hierarchical-dirichlet-process',
    ],
    cite: ['ramage2009llda', 'blei2003', 'teh2006hdp'],
  },
  run.topicModelRun,
)

const sources = [lda, hdp, dtm, plsa, ctm, factor, corpus, run]
/** The algorithms of the module. */
export const topicModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  ...sources,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const topicModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  ...sources,
) as Table<FunctionInfo>
