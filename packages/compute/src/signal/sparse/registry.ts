/**
 * The registry of `aifn-compute/signal/sparse`: the greedy pursuits, basis pursuit and its denoising form, iterative
 * hard thresholding, batch sparse coding and dictionary learning.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as atoms from './atoms'
import * as code from './code'
import * as convex from './convex'
import * as dictionary from './dictionary'
import * as pursuit from './pursuit'

const MODULE = 'signal/sparse'
const PURSUIT = ['sparse-approximation', 'matching-pursuit']
const CONVEX = ['sparse-approximation', 'basis-pursuit']
const LEARNING = ['dictionary-learning']
const fn = definer<FunctionInfo>('function', MODULE)
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)

fn(
  {
    key: 'mutualCoherence',
    name: 'Mutual coherence',
    tex: '\\mu(D) = \\max_{i \\ne j} \\frac{|d_i^\\top d_j|}{\\|d_i\\| \\|d_j\\|}',
    summary: 'The largest absolute cosine between two atoms; it bounds when pursuits find the sparsest code.',
    role: 'property',
    notes: PURSUIT,
    cite: ['donoho2003', 'tropp2004'],
  },
  atoms.mutualCoherence,
)
fn({ key: 'normaliseAtoms', name: 'Unit-norm atoms', role: 'transform', notes: PURSUIT }, atoms.normaliseAtoms)
fn(
  {
    key: 'hardThreshold',
    name: 'Hard thresholding',
    summary: 'Keep the s entries of largest magnitude and zero the rest.',
    role: 'transform',
    notes: CONVEX,
    cite: ['blumensath2009'],
  },
  atoms.hardThreshold,
)
algorithm(
  {
    key: 'matchingPursuitSteps',
    name: 'Matching pursuit',
    summary: 'Pick the atom most correlated with the residual and move its projection into the code.',
    problem: 'signal',
    state: { iterate: 'x', objective: 'residualNorm', flags: ['converged', 'stalled'] },
    notes: PURSUIT,
    cite: ['mallat1993'],
  },
  pursuit.matchingPursuitSteps,
)
algorithm(
  {
    key: 'orthogonalMatchingPursuitSteps',
    name: 'Orthogonal matching pursuit',
    summary: 'Add the atom most correlated with the residual to the support and refit the support by least squares.',
    problem: 'signal',
    state: { iterate: 'x', objective: 'residualNorm', flags: ['converged', 'stalled'] },
    notes: PURSUIT,
    cite: ['pati1993', 'tropp2004'],
  },
  pursuit.orthogonalMatchingPursuitSteps,
)
fn(
  { key: 'matchingPursuit', name: 'Matching pursuit', role: 'solver', notes: PURSUIT, cite: ['mallat1993'] },
  pursuit.matchingPursuit,
)
fn(
  {
    key: 'orthogonalMatchingPursuit',
    name: 'Orthogonal matching pursuit',
    role: 'solver',
    notes: PURSUIT,
    cite: ['pati1993', 'tropp2004'],
  },
  pursuit.orthogonalMatchingPursuit,
)
fn(
  {
    key: 'basisPursuit',
    name: 'Basis pursuit',
    tex: '\\min_x \\|x\\|_1 \\text{ s.t. } Dx = y',
    summary: 'The exact representation of least l1 norm, as a linear program.',
    role: 'solver',
    notes: CONVEX,
    cite: ['chen1998', 'donoho2003'],
  },
  convex.basisPursuit,
)
algorithm(
  {
    key: 'basisPursuitDenoisingSteps',
    name: 'Basis pursuit denoising (lasso) by FISTA',
    tex: '\\min_x \\tfrac{1}{2}\\|y - Dx\\|^2 + \\lambda\\|x\\|_1',
    summary: 'A gradient step on the squared error, then soft thresholding; accelerated by default.',
    problem: 'signal',
    state: { iterate: 'x', objective: 'value', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: CONVEX,
    cite: ['chen1998', 'tibshirani1996', 'beck2009'],
  },
  convex.basisPursuitDenoisingSteps,
)
fn(
  {
    key: 'basisPursuitDenoising',
    name: 'Basis pursuit denoising (lasso)',
    role: 'solver',
    notes: CONVEX,
    cite: ['chen1998', 'tibshirani1996'],
  },
  convex.basisPursuitDenoising,
)
algorithm(
  {
    key: 'iterativeHardThresholdingSteps',
    name: 'Iterative hard thresholding',
    summary: 'A gradient step on the squared error, then keep the s largest entries.',
    problem: 'signal',
    state: { iterate: 'x', objective: 'value', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: CONVEX,
    cite: ['blumensath2009'],
  },
  convex.iterativeHardThresholdingSteps,
)
fn(
  {
    key: 'iterativeHardThresholding',
    name: 'Iterative hard thresholding',
    role: 'solver',
    notes: CONVEX,
    cite: ['blumensath2009'],
  },
  convex.iterativeHardThresholding,
)
fn(
  {
    key: 'sparseCode',
    name: 'Sparse coding',
    summary: 'Code every column of a signal matrix over one dictionary by a chosen pursuit or solver.',
    role: 'transform',
    notes: [...PURSUIT, ...LEARNING],
  },
  code.sparseCode,
)
algorithm(
  {
    key: 'dictionaryLearningSteps',
    name: 'Dictionary learning (K-SVD, MOD)',
    tex: '\\min_{D, X} \\tfrac{1}{2}\\|Y - DX\\|_F^2 \\text{ s.t. } \\|x_i\\|_0 \\le s',
    summary: 'Alternate OMP coding with a K-SVD or method-of-optimal-directions dictionary update.',
    problem: 'objective',
    state: { iterate: 'D', objective: 'objective', flags: ['converged', 'diverged'] },
    random: true,
    notes: LEARNING,
    cite: ['aharon2006', 'engan1999'],
  },
  dictionary.dictionaryLearningSteps,
)
fn(
  {
    key: 'dictionaryLearning',
    name: 'Dictionary learning',
    role: 'fit',
    random: true,
    notes: LEARNING,
    cite: ['aharon2006', 'engan1999'],
  },
  dictionary.dictionaryLearning,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
/** The algorithms of the module, keyed by factory name. */
export const sparseAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  pursuit,
  convex,
  dictionary,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const sparseFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  atoms,
  pursuit,
  convex,
  code,
  dictionary,
) as Table<FunctionInfo>
