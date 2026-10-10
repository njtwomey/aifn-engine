/**
 * The registry of the `aifn-methods/learning/generalised` group's shared layer: IRLS and backfitting as traceable
 * algorithms, and the deviance, residual, penalty and smoothing-criterion functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as backfitting from './backfitting'
import * as irls from './irls'
import * as residuals from './residuals'
import * as smoothing from './smoothing'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/generalised')
const fn = definer<FunctionInfo>('function', 'learning/generalised')

algorithm(
  {
    key: 'irls',
    name: 'Iteratively reweighted least squares',
    summary: 'Fisher scoring for a GLM as weighted least squares on the working response, optionally penalised.',
    problem: 'least-squares',
    state: { iterate: 'coefficients', objective: 'deviance', flags: ['converged', 'diverged'] },
    notes: [
      'iteratively-reweighted-least-squares',
      'generalised-linear-model',
      'penalised-iteratively-reweighted-least-squares',
    ],
    cite: ['nelder1972', 'mccullagh1989', 'green1984'],
  },
  irls.irls,
)
algorithm(
  {
    key: 'backfitting',
    name: 'Backfitting',
    summary: 'Cycle over the terms of an additive model, smoothing the partial residuals of each in turn.',
    problem: 'least-squares',
    state: { iterate: 'coefficients', objective: 'deviance', flags: ['converged', 'diverged'] },
    notes: ['generalised-additive-model'],
    cite: ['hastie1990'],
  },
  backfitting.backfitting,
)
fn(
  {
    key: 'deviance',
    name: 'Deviance',
    role: 'estimator',
    notes: ['deviance-and-generalised-linear-model-diagnostics'],
    cite: ['mccullagh1989'],
  },
  irls.deviance,
)
fn(
  {
    key: 'residuals',
    name: 'GLM residuals',
    summary: 'Response, Pearson, deviance and working residuals.',
    role: 'estimator',
    notes: ['deviance-and-generalised-linear-model-diagnostics', 'residual-diagnostics'],
  },
  residuals.residuals,
)
fn(
  {
    key: 'penalisedFit',
    name: 'Penalised least-squares fit',
    role: 'fit',
    notes: ['smoothing-penalties-and-curvature', 'p-splines'],
    cite: ['wood2017'],
  },
  smoothing.penalisedFit,
)
fn(
  { key: 'penaltyMatrix', name: 'Penalty matrix', role: 'construction', notes: ['smoothing-penalties-and-curvature'] },
  smoothing.penaltyMatrix,
)
fn(
  {
    key: 'nullSpaceDimension',
    name: 'Penalty null-space dimension',
    role: 'property',
    notes: ['smoothing-penalties-and-curvature'],
  },
  smoothing.nullSpaceDimension,
)
fn(
  {
    key: 'smoothingCriterion',
    name: 'Smoothing criterion (GCV, UBRE, REML)',
    role: 'estimator',
    notes: ['additive-model-smoothing-parameter-selection'],
    cite: ['wood2017', 'craven1979'],
  },
  smoothing.smoothingCriterion,
)

/** The algorithms of the group's shared layer, keyed by factory name. */
export const generalisedAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  irls,
  backfitting,
) as Table<AlgorithmInfo>
/** The functions of the group's shared layer, keyed by name. */
export const generalisedFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  irls,
  residuals,
  smoothing,
) as Table<FunctionInfo>
