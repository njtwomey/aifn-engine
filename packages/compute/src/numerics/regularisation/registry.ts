/**
 * The algorithms and functions of `aifn-compute/numerics/regularisation`, registered with what each factory takes
 * (`problem`) and the roles of its state's fields (`state`), so a generic trace view picks default series and a worker
 * can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as landweber from './landweber'
import * as selection from './selection'
import * as tikhonov from './tikhonov'

const algorithm = definer<AlgorithmInfo>('algorithm', 'numerics/regularisation')

algorithm(
  {
    key: 'landweberSteps',
    name: 'Landweber iteration',
    summary: 'x ← x + ωAᵀ(b − Ax): gradient descent on the residual, regularised by stopping early.',
    problem: 'system',
    state: { iterate: 'x', objective: 'residualNorm', stepSize: 'step', flags: ['converged', 'diverged'] },
    cite: ['landweber1951', 'hansen1998'],
  },
  landweber.landweberSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const regularisationAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', landweber) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'numerics/regularisation')

fn(
  {
    key: 'tikhonov',
    name: 'Tikhonov regularisation',
    tex: '\\min_x \\|Ax - b\\|^2 + \\lambda \\|Lx\\|^2',
    summary: 'The least-squares solution with a penalty on its (semi)norm: ridge regression in standard form.',
    role: 'solver',
    cite: ['tikhonov1963', 'hansen1998'],
  },
  tikhonov.tikhonov,
)
fn(
  {
    key: 'tikhonovPath',
    name: 'Tikhonov path',
    summary: 'Tikhonov solutions, norms and filter factors for many weights from one SVD (a general L made standard).',
    role: 'solver',
    cite: ['hansen1998', 'elden1982'],
  },
  tikhonov.tikhonovPath,
)
fn(
  {
    key: 'filterFactors',
    name: 'Tikhonov filter factors',
    tex: 'f_i = \\frac{s_i^2}{s_i^2 + \\lambda}',
    summary: 'How much of each spectral component of the least-squares solution Tikhonov regularisation keeps.',
    role: 'transform',
    cite: ['hansen1998'],
  },
  tikhonov.filterFactors,
)
fn(
  {
    key: 'differenceOperator',
    name: 'Discrete derivative operator',
    summary: 'The (n − d) × n matrix of d-th differences, a regularisation matrix that penalises roughness.',
    role: 'construction',
    cite: ['hansen1998'],
  },
  tikhonov.differenceOperator,
)
fn(
  {
    key: 'convolutionMatrix',
    name: 'Convolution matrix',
    summary: 'The n × n matrix of a one-dimensional blur, with zero, periodic or reflected ends.',
    role: 'construction',
    cite: ['hansen1998'],
  },
  tikhonov.convolutionMatrix,
)
fn(
  {
    key: 'lCurve',
    name: 'L-curve',
    summary: 'Residual norm against solution norm on log scales, its curvature, and the corner of largest curvature.',
    role: 'estimator',
    cite: ['hansen1993', 'hansen1998'],
  },
  selection.lCurve,
)
fn(
  {
    key: 'discrepancyPrinciple',
    name: 'Discrepancy principle',
    tex: '\\|Ax_\\lambda - b\\| = \\tau\\delta',
    summary: 'The Tikhonov weight at which the residual norm matches the noise level.',
    role: 'estimator',
    cite: ['morozov1966', 'hansen1998'],
  },
  selection.discrepancyPrinciple,
)

/** The functions of the module, keyed by name. */
export const regularisationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', tikhonov, selection) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
