/**
 * The registry of `aifn-methods/unsupervised/embedding/linear`: SMACOF, the latent-model EM and FastICA as
 * step-through algorithms (the roles of their states' fields in `state`), and classical MDS, stress and Andrews curves
 * as functions with the notes they serve. The estimators (`pca`, `kernelPca`, `metricMds`, `factorAnalysis`,
 * `probabilisticPca`, `fastIca`) are registered as models beside their code.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as andrews from './andrews'
import * as latent from './latent'
import * as linear from './linear'

/** A registry table of the module: entries keyed by name, each a function with its `info`. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
/** Registers a function of the module under `unsupervised/embedding/linear`. */
const fn = definer<FunctionInfo>('function', 'unsupervised/embedding/linear')

definer<AlgorithmInfo>('algorithm', 'unsupervised/embedding/linear')(
  {
    key: 'smacofSteps',
    name: 'SMACOF',
    summary: 'Majorisation steps (the Guttman transform) that never increase metric stress.',
    problem: 'objective',
    state: { iterate: 'embedding', objective: 'stress', flags: ['converged'] },
    random: true,
    notes: ['multidimensional-scaling'],
    cite: ['kruskal1964', 'deleeuw2009'],
  },
  linear.smacofSteps,
)
fn(
  {
    key: 'classicalMds',
    name: 'Classical MDS',
    summary: 'The top eigenvectors of the double-centred squared distances.',
    role: 'fit',
    notes: ['multidimensional-scaling'],
    cite: ['torgerson1952'],
  },
  linear.classicalMds,
)
fn(
  {
    key: 'stress',
    name: 'Stress',
    role: 'estimator',
    notes: ['multidimensional-scaling', 'evaluating-embeddings'],
    cite: ['kruskal1964'],
  },
  linear.stress,
)
/** Registers a step-through algorithm of the module under `unsupervised/embedding/linear`. */
const algorithm = definer<AlgorithmInfo>('algorithm', 'unsupervised/embedding/linear')
algorithm(
  {
    key: 'latentGaussianSteps',
    name: 'EM for factor analysis and probabilistic PCA',
    summary:
      'EM on x = Wz + μ + ε with diagonal (factor analysis) or isotropic (PPCA) noise; the likelihood never falls.',
    problem: 'objective',
    state: { iterate: 'loadings', objective: 'logLikelihood', flags: ['converged', 'diverged'] },
    random: true,
    notes: ['factor-analysis', 'probabilistic-principal-component-analysis'],
    cite: ['rubin1982', 'ghahramani1996', 'tipping1999'],
  },
  latent.latentGaussianSteps,
)
algorithm(
  {
    key: 'fastIcaSteps',
    name: 'FastICA fixed-point iteration',
    summary: 'Whitened data, then the symmetric fixed-point update of the unmixing rows on a log-cosh contrast.',
    problem: 'objective',
    state: { iterate: 'unmixing', objective: 'change', flags: ['converged', 'diverged'] },
    random: true,
    notes: ['independent-component-analysis'],
    cite: ['hyvarinen1999', 'hyvarinen2000'],
  },
  latent.fastIcaSteps,
)
fn({ key: 'andrewsCurves', name: 'Andrews curves', role: 'transform', cite: ['andrews2003'] }, andrews.andrewsCurves)

/**
 * The step-through algorithms of the module, keyed by factory name: `smacofSteps`, `latentGaussianSteps` and
 * `fastIcaSteps`, each with its `info` (the roles of its state's fields, and whether it draws random numbers).
 *
 * @example The algorithms and what their states report
 * for (const [key, entry] of Object.entries(linearEmbeddingAlgorithms)) print(key, entry.info.state)
 */
export const linearEmbeddingAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  linear,
  latent,
) as Table<AlgorithmInfo>
/**
 * The functions of the module, keyed by name: `classicalMds`, `stress` and `andrewsCurves`, each with its `info`.
 *
 * @example The functions and their roles
 * for (const [key, entry] of Object.entries(linearEmbeddingFunctions)) print(key, entry.info.role)
 */
export const linearEmbeddingFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  linear,
  andrews,
) as Table<FunctionInfo>
