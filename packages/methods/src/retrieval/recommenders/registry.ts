/** The registry of `aifn-methods/retrieval/recommenders`: the recommenders, their evaluation and the simulators. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as factorisation from './factorisation'
import * as interactions from './interactions'
import * as matchbox from './matchbox'
import * as models from './models'
import * as neighbourhood from './neighbourhood'
import * as run from './run'
import * as simulator from './simulator'

const MODULE = 'retrieval/recommenders'
const fn = definer<FunctionInfo>('function', MODULE)
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)

const NEIGHBOURS = ['neighbourhood-collaborative-filtering', 'recommender-systems']
const MF = ['matrix-factorisation-for-recommendation']

fn(
  {
    key: 'popularity',
    name: 'Popularity recommender',
    summary: 'Every user gets the catalogue ranked by training interactions: the baseline every model must beat.',
    role: 'fit',
    notes: ['progress-and-baselines-in-recommendation', 'popularity-bias'],
    cite: ['dacrema2019'],
  },
  neighbourhood.popularity,
)
fn(
  {
    key: 'userKnn',
    name: 'User-based collaborative filtering',
    summary: 'Score an item by the cosine-weighted interactions of the k most similar users.',
    role: 'fit',
    notes: NEIGHBOURS,
    cite: ['resnick1994grouplens'],
  },
  neighbourhood.userKnn,
)
fn(
  {
    key: 'itemKnn',
    name: 'Item-based collaborative filtering',
    summary: 'Score an item by its cosine similarity to the items the user interacted with, keeping k neighbours each.',
    role: 'fit',
    notes: NEIGHBOURS,
    cite: ['sarwar2001itemcf'],
  },
  neighbourhood.itemKnn,
)
algorithm(
  {
    key: 'alternatingLeastSquares',
    name: 'Alternating least squares (explicit ratings)',
    summary: 'Refit every user’s factors by ridge regression on the items fixed, then every item’s; never increases.',
    problem: 'least-squares',
    state: { iterate: 'P', objective: 'objective', flags: ['diverged'] },
    random: true,
    notes: ['alternating-least-squares', ...MF],
    cite: ['zhou2008als', 'koren2009mf'],
  },
  factorisation.alternatingLeastSquares,
)
algorithm(
  {
    key: 'implicitAls',
    name: 'Implicit-feedback ALS (Hu–Koren–Volinsky)',
    summary: 'Confidence-weighted factorisation of the whole 0/1 matrix, solved in closed form one side at a time.',
    problem: 'least-squares',
    state: { iterate: 'P', objective: 'objective', flags: ['diverged'] },
    random: true,
    notes: ['weighted-matrix-factorisation', 'explicit-and-implicit-feedback', 'alternating-least-squares'],
    cite: ['hu2008implicit'],
  },
  factorisation.implicitAls,
)
fn(
  {
    key: 'implicitAlsObjective',
    name: 'Implicit ALS objective',
    tex: '\\sum_{u,i} c_{ui}(p_{ui} - \\mathbf{x}_u^\\top \\mathbf{y}_i)^2 + \\lambda(\\|X\\|^2 + \\|Y\\|^2)',
    role: 'property',
    notes: ['weighted-matrix-factorisation'],
    cite: ['hu2008implicit'],
  },
  factorisation.implicitAlsObjective,
)
algorithm(
  {
    key: 'matrixFactorisationSgd',
    name: 'Biased matrix factorisation by SGD',
    summary: 'μ + b_u + b_i + p_uᵀq_i fitted to ratings by minibatch SGD with automatic gradients.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', grad: 'grads', flags: ['diverged'] },
    random: true,
    notes: [...MF, 'netflix-prize'],
    cite: ['koren2009mf'],
  },
  factorisation.matrixFactorisationSgd,
)
fn(
  {
    key: 'neuralRecommender',
    name: 'Gradient-trained recommenders',
    summary:
      'Logistic MF, BPR, FM, field-aware FM, Wide & Deep, DeepFM, NCF, two-tower and SASRec: parameters, loss and scorer.',
    role: 'construction',
    notes: [
      'bayesian-personalised-ranking',
      'factorisation-machines',
      'field-aware-factorisation-machines',
      'wide-and-deep',
      'deepfm',
      'neural-collaborative-filtering',
      'two-tower-model',
      'sasrec',
    ],
    cite: [
      'rendle2009bpr',
      'rendle2010fm',
      'juan2016ffm',
      'cheng2016wide',
      'guo2017deepfm',
      'he2017ncf',
      'covington2016youtube',
      'yi2019sampling',
      'kang2018sasrec',
    ],
  },
  models.neuralRecommender,
)
fn(
  {
    key: 'evaluateRanking',
    name: 'Held-out ranking evaluation',
    summary: 'Recall@k, NDCG@k, hit rate and coverage of each user’s ranking of the items they have not seen.',
    role: 'estimator',
    notes: ['recommendation-metrics', 'precision-and-recall-at-k', 'normalised-discounted-cumulative-gain'],
  },
  interactions.evaluateRanking,
)
fn(
  {
    key: 'recommenderRun',
    name: 'Recommender training run',
    summary: 'Train one recommender epoch by epoch, streaming held-out recall and NDCG, scores and embedding maps.',
    role: 'simulation',
    random: true,
    notes: ['recommender-systems', 'two-tower-model', 'matrix-factorisation-for-recommendation'],
  },
  run.recommenderRun,
)
fn(
  {
    key: 'feedbackLoop',
    name: 'Feedback-loop simulator',
    summary: 'Policies retrained on the clicks their own slates produced; exposure concentration round by round.',
    role: 'simulation',
    random: true,
    notes: ['feedback-loops-in-recommendation', 'popularity-bias', 'biases-in-recommender-feedback'],
    cite: ['chaney2018feedback'],
  },
  simulator.feedbackLoop,
)
fn(
  {
    key: 'worldFromFactors',
    name: 'Simulated click world',
    summary: 'True click probabilities σ(a·w_uᵀv_i + b) from user and item factors, for the feedback-loop simulator.',
    role: 'construction',
    notes: ['feedback-loops-in-recommendation'],
  },
  simulator.worldFromFactors,
)

const MATCHBOX = ['matchbox-recommender', 'hybrid-recommenders']
fn(
  {
    key: 'matchbox',
    name: 'Matchbox model (prior)',
    summary: 'Gaussian posteriors over user and item trait weights, biases and ordinal thresholds.',
    role: 'construction',
    random: true,
    notes: MATCHBOX,
    cite: ['stern2009matchbox'],
  },
  matchbox.matchbox,
)
fn(
  {
    key: 'matchboxPredict',
    name: 'Matchbox prediction',
    summary: 'Traits, the latent affinity and the probability of each rating level for a user–item pair.',
    role: 'property',
    notes: MATCHBOX,
    cite: ['stern2009matchbox'],
  },
  matchbox.matchboxPredict,
)
fn(
  {
    key: 'matchboxUpdate',
    name: 'Matchbox ADF update',
    summary:
      'One rating: EP on the two threshold comparisons, then sum and product factor messages back to every weight.',
    role: 'fit',
    notes: MATCHBOX,
    cite: ['stern2009matchbox'],
  },
  matchbox.matchboxUpdate,
)
fn(
  {
    key: 'matchboxRatings',
    name: 'Synthetic ordinal ratings with feature traits',
    summary: 'Bilinear affinities of traits linear in side features, plus noise, cut into equal-share levels.',
    role: 'simulation',
    random: true,
    notes: MATCHBOX,
  },
  matchbox.matchboxRatings,
)
fn(
  {
    key: 'matchboxRun',
    name: 'Streamed Matchbox training',
    summary: 'Online ADF over the ratings, with held-out RMSE, accuracy, cold-start error and item trait maps.',
    role: 'simulation',
    random: true,
    notes: MATCHBOX,
    cite: ['stern2009matchbox'],
  },
  matchbox.matchboxRun,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const sources = [neighbourhood, factorisation, models, interactions, run, simulator, matchbox]
/** The algorithms of the module. */
export const recommenderAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  ...sources,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const recommenderFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  ...sources,
) as Table<FunctionInfo>
