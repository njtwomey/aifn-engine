/**
 * The algorithms of `aifn-compute/optim/proximal`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as alternating from './alternating'
import * as proximal from './proximal'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/proximal')

algorithm(
  {
    key: 'proximalGradient',
    name: 'Proximal gradient',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods'],
    cite: ['beck2009'],
  },
  proximal.proximalGradient,
)
algorithm(
  {
    key: 'ista',
    name: 'ISTA',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods', 'lasso'],
    cite: ['beck2009'],
  },
  proximal.ista,
)
algorithm(
  {
    key: 'fista',
    name: 'FISTA',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods', 'accelerated-gradient-methods'],
    cite: ['beck2009'],
  },
  proximal.fista,
)
algorithm(
  {
    key: 'projectedGradient',
    name: 'Projected gradient',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods'],
    cite: ['beck2009'],
  },
  proximal.projectedGradient,
)
algorithm(
  {
    key: 'alternatingProjectionsSteps',
    name: 'Alternating projections',
    summary:
      'Cycle through the projections onto convex sets until the point stops moving: a point of their intersection.',
    problem: 'system',
    state: { iterate: 'x', flags: ['converged'] },
    notes: ['label-propagation-for-label-proportions'],
    cite: ['boyd2003'],
  },
  alternating.alternatingProjectionsSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const proximalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', proximal, alternating) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'optim/proximal')
const PROX = ['proximal-gradient-methods']

fn({ key: 'projectBox', name: 'Projection onto a box', role: 'transform', notes: PROX }, proximal.projectBox)
fn(
  { key: 'projectNonnegative', name: 'Projection onto the non-negative orthant', role: 'transform', notes: PROX },
  proximal.projectNonnegative,
)
fn({ key: 'projectBall', name: 'Projection onto a ball', role: 'transform', notes: PROX }, proximal.projectBall)
fn(
  {
    key: 'projectSimplex',
    name: 'Projection onto the simplex',
    role: 'transform',
    notes: [...PROX, 'sampling-the-simplex'],
  },
  proximal.projectSimplex,
)
fn(
  {
    key: 'projectSimplexRows',
    name: 'Projection of each row onto the simplex',
    summary: 'Project every row of a matrix of class scores onto the probability simplex.',
    role: 'transform',
    notes: [...PROX, 'label-propagation-for-label-proportions'],
  },
  proximal.projectSimplexRows,
)
fn(
  {
    key: 'projectGroupSums',
    name: 'Projection onto fixed group sums',
    summary: 'Shift each disjoint group of coordinates by a common amount so the group sums to its target.',
    role: 'transform',
    notes: [...PROX, 'label-propagation-for-label-proportions'],
  },
  alternating.projectGroupSums,
)
fn(
  {
    key: 'alternatingProjections',
    name: 'Alternating projections (run)',
    tex: 'x \\leftarrow P_m(\\cdots P_2(P_1(x)))',
    summary: 'A point in the intersection of convex sets; with Dykstra’s corrections, the projection onto it.',
    role: 'solver',
    notes: ['label-propagation-for-label-proportions'],
    cite: ['boyd2003'],
  },
  alternating.alternatingProjections,
)
fn({ key: 'proxZero', name: 'Proximal operator of zero (identity)', role: 'transform', notes: PROX }, proximal.proxZero)
fn(
  {
    key: 'proxL1',
    name: 'Soft thresholding (prox of the L1 norm)',
    tex: '\\operatorname{sign}(x)\\max(|x| - \\lambda, 0)',
    role: 'transform',
    notes: [...PROX, 'subgradients'],
    cite: ['parikh2014'],
  },
  proximal.proxL1,
)
fn(
  {
    key: 'proxL2',
    name: 'Block soft thresholding (prox of the L2 norm)',
    role: 'transform',
    notes: PROX,
    cite: ['parikh2014'],
  },
  proximal.proxL2,
)
fn(
  {
    key: 'proxSquaredL2',
    name: 'Prox of the squared L2 norm (shrinkage)',
    role: 'transform',
    notes: PROX,
    cite: ['parikh2014'],
  },
  proximal.proxSquaredL2,
)
fn({ key: 'proxBox', name: 'Prox of a box indicator', role: 'transform', notes: PROX }, proximal.proxBox)
fn(
  { key: 'proxNonnegative', name: 'Prox of the non-negativity indicator', role: 'transform', notes: PROX },
  proximal.proxNonnegative,
)

/** The functions of the module, keyed by name. */
export const proximalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', proximal, alternating) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
