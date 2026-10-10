/**
 * The registry of `aifn-methods/inference/learner-models`: the catalogue entries (name, summary, role, notes and
 * citations) of its functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as evaluation from './evaluation'
import * as run from './run'
import * as zilm from './zilm'

const fn = definer<FunctionInfo>('function', 'inference/learner-models')
const IRT = ['item-response-theory']
const EQUITY = ['item-response-theory', 'group-fairness-metrics']

fn(
  {
    key: 'zilmProbability',
    name: 'IRT-ZILM probability of a correct answer',
    summary: 'Pr(Y = 1) = (1 − π) σ(a(θ − b)): the IRT probability scaled by the learning quality factor 1 − π.',
    role: 'property',
    notes: IRT,
    cite: ['lord1968', 'barton1981'],
  },
  zilm.zilmProbability,
)
fn(
  {
    key: 'structuralZeroPosterior',
    name: 'Structural-zero posterior',
    summary: 'The probability that an observed zero came from the context, π / (π + (1 − π)(1 − p)).',
    role: 'inference',
    notes: IRT,
  },
  zilm.structuralZeroPosterior,
)
fn(
  {
    key: 'fitLearnerModel',
    name: 'Learner models: IRT, linear KTM, IRT-ZILM',
    summary:
      'Abilities, difficulties, discriminations and context weights by penalised joint maximum likelihood (autodiff, L-BFGS); IRT-ZILM adds a zero-inflation probability logistic in conditions × item features.',
    role: 'fit',
    notes: IRT,
    cite: ['lord1968', 'barton1981'],
  },
  zilm.fitLearnerModel,
)
fn(
  {
    key: 'predictLearner',
    name: 'Learner-model predictions',
    summary: 'Pr(correct), the base probability p and the zero-inflation probability π for every student and item.',
    role: 'property',
    notes: IRT,
  },
  zilm.predictLearner,
)
fn(
  {
    key: 'abilityEquity',
    name: 'Ability bias by group',
    summary:
      'Mean signed error and RMSE of the ability estimates per group, and the gap between groups and a reference.',
    role: 'estimator',
    notes: EQUITY,
  },
  evaluation.abilityEquity,
)
fn(
  {
    key: 'parameterRecovery',
    name: 'Parameter recovery',
    summary: 'Pearson and Spearman correlations between true and estimated parameters.',
    role: 'estimator',
    notes: IRT,
  },
  evaluation.parameterRecovery,
)
fn(
  {
    key: 'responseScores',
    name: 'Held-out response scores',
    summary:
      'Accuracy, F₁, mean negative log-likelihood and Brier score of predicted probabilities of a correct answer.',
    role: 'estimator',
    notes: ['log-loss-and-brier-score'],
  },
  evaluation.responseScores,
)
fn(
  {
    key: 'structuralZeroAuroc',
    name: 'Structural-zero AUROC',
    summary: 'How well structural-zero posteriors separate context-caused zeros from incorrect answers.',
    role: 'estimator',
    notes: ['receiver-operating-characteristic-curve-and-area'],
  },
  evaluation.structuralZeroAuroc,
)
fn(
  {
    key: 'learnerModelRun',
    name: 'Streamed learner-model comparison',
    summary:
      'IRT, linear KTM and IRT-ZILM fitted to the same responses: ability bias by group, recovery, held-out scores and structural-zero posteriors.',
    role: 'simulation',
    random: true,
    notes: EQUITY,
  },
  run.learnerModelRun,
)
fn(
  {
    key: 'learnerEquitySweep',
    name: 'Learner-model equity sweep',
    summary: 'Each model’s ability bias by group and equity gap as the zero-inflation rate of the conditions grows.',
    role: 'simulation',
    random: true,
    notes: EQUITY,
  },
  run.learnerEquitySweep,
)

type Table = Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
/** The functions of the module. */
export const learnerModelFunctions: Table = entries<FunctionInfo>('function', zilm, evaluation, run) as Table
