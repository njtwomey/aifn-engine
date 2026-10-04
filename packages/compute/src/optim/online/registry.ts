/**
 * The registry of `aifn-compute/optim/online`: the experts algorithms and the online convex optimisers as algorithms
 * (`problem: 'sequence'`: a stream of losses), and the projections, regret traces, bounds and online-to-batch
 * conversion as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as convex from './convex'
import * as experts from './experts'
import * as regret from './regret'

const MODULE = 'optim/online'
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)
const fn = definer<FunctionInfo>('function', MODULE)

const EXPERTS = { iterate: 'weights', objective: 'regret', flags: [] } as const
const OCO = { iterate: 'w', objective: 'cumulative', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] } as const

algorithm(
  {
    key: 'hedge',
    name: 'Hedge (exponential weights)',
    tex: 'p_{t,i} \\propto e^{-\\eta L_{t-1,i}}',
    summary: 'Play the softmax of the negated cumulative losses; regret at most √(T ln N / 2) at the tuned rate.',
    problem: 'sequence',
    state: EXPERTS,
    notes: ['hedge-and-exponential-weights', 'lower-bounds-for-online-learning', 'online-learning'],
    cite: ['freund1997', 'cesabianchi2006'],
  },
  experts.hedge,
)
algorithm(
  {
    key: 'fixedShare',
    name: 'Fixed share',
    summary: 'Hedge that hands back a share α of the mass each round, to track the best switching expert.',
    problem: 'sequence',
    state: EXPERTS,
    notes: ['tracking-the-best-expert', 'hedge-and-exponential-weights'],
    cite: ['herbster1998', 'cesabianchi2006'],
  },
  experts.fixedShare,
)
algorithm(
  {
    key: 'weightedMajority',
    name: 'Weighted majority',
    summary: 'Predict by a weighted vote of the experts and multiply the weight of each expert that errs by β.',
    problem: 'sequence',
    state: { iterate: 'weights', objective: 'mistakes', flags: [] },
    notes: ['weighted-majority', 'online-learning'],
    cite: ['littlestone1994', 'littlestone1988'],
  },
  experts.weightedMajority,
)
algorithm(
  {
    key: 'onlineGradientDescent',
    name: 'Online gradient descent',
    tex: 'w_{t+1} = \\Pi_{\\mathcal W}(w_t - \\eta_t g_t)',
    summary: 'Step against the latest subgradient and project; regret (3/2)DG√T with ηₜ = D/(G√t).',
    problem: 'sequence',
    state: OCO,
    notes: ['online-gradient-descent', 'online-convex-optimisation'],
    cite: ['zinkevich2003', 'hazan2016'],
  },
  convex.onlineGradientDescent,
)
algorithm(
  {
    key: 'followTheRegularisedLeader',
    name: 'Follow the regularised leader',
    summary: 'Minimise the linearised cumulative loss plus a regulariser (L2, L1, or the adaptive FTRL-Proximal form).',
    problem: 'sequence',
    state: OCO,
    notes: ['follow-the-regularised-leader', 'adaptive-online-methods', 'online-convex-optimisation'],
    cite: ['shalevshwartz2012', 'mcmahan2017b'],
  },
  convex.followTheRegularisedLeader,
)
algorithm(
  {
    key: 'onlineNewtonStep',
    name: 'Online Newton step',
    summary: 'Precondition by Aₜ = εI + Σ gₛgₛᵀ and project in its norm; logarithmic regret on exp-concave losses.',
    problem: 'sequence',
    state: OCO,
    notes: ['online-newton-step-and-exp-concavity', 'online-convex-optimisation'],
    cite: ['hazan2007', 'hazan2016'],
  },
  convex.onlineNewtonStep,
)
algorithm(
  {
    key: 'onlineAdagrad',
    name: 'AdaGrad (online, diagonal)',
    summary: 'Scale each coordinate by the root of its summed squared gradients and project in that norm.',
    problem: 'sequence',
    state: OCO,
    notes: ['adaptive-online-methods', 'online-convex-optimisation'],
    cite: ['duchi2011', 'orabona2019'],
  },
  convex.onlineAdagrad,
)

fn(
  {
    key: 'projectOnto',
    name: 'Euclidean projection',
    tex: '\\Pi_{\\mathcal W}',
    summary: 'The nearest point of a ball, a box or the simplex.',
    role: 'transform',
    notes: ['online-gradient-descent'],
    cite: ['zinkevich2003'],
  },
  convex.projectOnto,
)
fn(
  {
    key: 'ogdStepSize',
    name: 'OGD step-size schedule',
    tex: '\\eta_t = D/(G\\sqrt t)',
    summary: 'The schedule that gives online gradient descent regret (3/2)DG√T.',
    role: 'construction',
    notes: ['online-gradient-descent'],
    cite: ['zinkevich2003'],
  },
  convex.ogdStepSize,
)
fn(
  {
    key: 'onlineToBatch',
    name: 'Online-to-batch conversion',
    tex: '\\bar w = \\tfrac1T \\textstyle\\sum_t w_t',
    summary: 'Average the iterates of one online pass; its excess risk is at most E[R_T]/T for a convex loss.',
    role: 'estimator',
    notes: ['online-to-batch-conversion'],
    cite: ['cesabianchi2004', 'shalevshwartz2012'],
  },
  convex.onlineToBatch,
)
fn(
  {
    key: 'regretTrace',
    name: 'Regret trace',
    tex: 'R_t = \\hat L_t - L^\\star_t',
    summary: 'Cumulative losses and regret over time against a comparator or the best expert so far.',
    role: 'estimator',
    notes: ['online-learning', 'hedge-and-exponential-weights', 'online-convex-optimisation'],
    cite: ['cesabianchi2006'],
  },
  regret.regretTrace,
)
fn(
  {
    key: 'bestSwitchingLoss',
    name: 'Best switching sequence',
    summary: 'The least loss of an expert sequence with at most m switches, by dynamic programming.',
    role: 'solver',
    notes: ['tracking-the-best-expert'],
    cite: ['herbster1998'],
  },
  regret.bestSwitchingLoss,
)
fn(
  {
    key: 'hedgeTunedRate',
    name: 'Hedge tuned rate',
    tex: '\\eta = \\sqrt{8 \\ln N / T}',
    summary: 'The learning rate that minimises Hedge’s regret bound for a known horizon.',
    role: 'construction',
    notes: ['hedge-and-exponential-weights'],
    cite: ['cesabianchi2006'],
  },
  regret.hedgeTunedRate,
)
fn(
  {
    key: 'hedgeRegretBound',
    name: 'Hedge regret bound',
    tex: '\\ln N / \\eta + \\eta T / 8',
    summary: 'Hedge’s regret bound for a constant, tuned or anytime learning rate.',
    role: 'property',
    notes: ['hedge-and-exponential-weights'],
    cite: ['cesabianchi2006'],
  },
  regret.hedgeRegretBound,
)
fn(
  {
    key: 'ogdRegretBound',
    name: 'OGD regret bound',
    tex: '\\tfrac32 D G \\sqrt T',
    summary: 'The regret bound of online gradient descent with ηₜ = D/(G√t).',
    role: 'property',
    notes: ['online-gradient-descent'],
    cite: ['zinkevich2003'],
  },
  regret.ogdRegretBound,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const all = { ...experts, ...convex, ...regret }

/** The algorithms of the module, keyed by factory name. */
export const onlineAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', all) as Table<AlgorithmInfo>

/** The functions of the module, keyed by name. */
export const onlineFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', all) as Table<FunctionInfo>
