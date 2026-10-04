/**
 * The registry of `aifn-compute/numerics/factorisation`: non-negative matrix factorisation (multiplicative updates and HALS),
 * random projections with the Johnson–Lindenstrauss dimension, and canonical correlation analysis.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cca from './cca'
import * as nmf from './nmf'
import * as projection from './projection'

const MODULE = 'numerics/factorisation'
const notes = ['non-negative-matrix-factorisation']
const cite = ['lee1999', 'lee2001']
const fn = definer<FunctionInfo>('function', MODULE)

definer<AlgorithmInfo>('algorithm', MODULE)(
  {
    key: 'nmfSteps',
    name: 'NMF by multiplicative updates or HALS',
    summary:
      'Alternate W and H updates (Lee–Seung ratios of the gradient’s parts, or exact HALS column updates); never increases.',
    problem: 'objective',
    state: { iterate: 'W', objective: 'objective', flags: ['converged', 'diverged'] },
    random: true,
    notes,
    cite,
  },
  nmf.nmfSteps,
)
fn(
  {
    key: 'nmf',
    name: 'Non-negative matrix factorisation',
    summary: 'X ≈ WH with W, H ≥ 0, for the squared error or the generalised KL divergence.',
    role: 'fit',
    random: true,
    notes,
    cite,
  },
  nmf.nmf,
)
const jl = { notes: ['random-projections'], cite: ['johnson1984', 'dasgupta2003', 'achlioptas2003', 'li2006'] }
fn(
  {
    key: 'randomProjectionMatrix',
    name: 'Random projection matrix',
    summary: 'A k × d matrix of Gaussian or sparse ±1 entries with E[RᵀR] = I.',
    role: 'construction',
    random: true,
    ...jl,
  },
  projection.randomProjectionMatrix,
)
fn(
  {
    key: 'randomProjection',
    name: 'Random projection',
    summary: 'Rows mapped to k dimensions by a random matrix, keeping pairwise distances within 1 ± ε w.h.p.',
    role: 'transform',
    random: true,
    ...jl,
  },
  projection.randomProjection,
)
fn(
  {
    key: 'johnsonLindenstraussDimension',
    name: 'Johnson–Lindenstrauss dimension',
    summary: 'The target dimension 4 ln n / (ε²/2 − ε³/3) that keeps n points’ distances within 1 ± ε.',
    role: 'property',
    ...jl,
  },
  projection.johnsonLindenstraussDimension,
)
fn(
  {
    key: 'johnsonLindenstraussEpsilon',
    name: 'Johnson–Lindenstrauss distortion',
    summary: 'The smallest ε whose Johnson–Lindenstrauss dimension for n points fits in k dimensions.',
    role: 'property',
    ...jl,
  },
  projection.johnsonLindenstraussEpsilon,
)
fn(
  {
    key: 'distanceDistortion',
    name: 'Pairwise distance distortion',
    summary: 'Ratios of squared pairwise distances after and before a map, and the largest deviation from 1.',
    role: 'estimator',
    ...jl,
  },
  projection.distanceDistortion,
)
fn(
  {
    key: 'canonicalCorrelation',
    name: 'Canonical correlation analysis',
    summary: 'Paired directions of two views with maximal correlation, by the SVD of the whitened cross-covariance.',
    role: 'fit',
    notes: ['canonical-correlation-analysis'],
    cite: ['hotelling1936', 'hardoon2004'],
  },
  cca.canonicalCorrelation,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
/** The algorithms of the module. */
export const factorisationAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  nmf,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const factorisationFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  nmf,
  projection,
  cca,
) as Table<FunctionInfo>
