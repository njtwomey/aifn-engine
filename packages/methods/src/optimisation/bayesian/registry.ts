/**
 * Registry entries of `aifn-methods/optimisation/bayesian`: the Bayesian optimisation loop as a step-through algorithm,
 * and the acquisition functions as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as acquisitions from './acquisitions'
import * as entropy from './entropy'
import * as loop from './loop'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optimisation/bayesian')
const fn = definer<FunctionInfo>('function', 'optimisation/bayesian')

algorithm(
  {
    key: 'bayesianOptimisationSteps',
    name: 'Bayesian optimisation',
    summary:
      'Fit a GP to the evaluations, maximise an acquisition (EI, PI, UCB, Thompson, MES) on a grid, evaluate there.',
    problem: 'objective',
    state: { iterate: 'next', objective: 'best', flags: ['diverged'] },
    random: true,
    cite: ['jones1998', 'snoek2012', 'shahriari2016', 'frazier2018'],
  },
  loop.bayesianOptimisationSteps,
)

fn(
  {
    key: 'probabilityOfImprovement',
    name: 'Probability of improvement',
    summary: 'The posterior probability that a point beats the incumbent by a margin ξ.',
    role: 'estimator',
    cite: ['kushner1964'],
  },
  acquisitions.probabilityOfImprovement,
)

fn(
  {
    key: 'expectedImprovement',
    name: 'Expected improvement',
    summary: 'The posterior expectation of the improvement over the incumbent, in closed form.',
    role: 'estimator',
    cite: ['mockus1978', 'jones1998'],
  },
  acquisitions.expectedImprovement,
)

fn(
  {
    key: 'upperConfidenceBound',
    name: 'Upper confidence bound',
    summary: 'The optimistic bound μ + √β σ (its negated lower counterpart when minimising), as GP-UCB.',
    role: 'estimator',
    cite: ['srinivas2010'],
  },
  acquisitions.upperConfidenceBound,
)

fn(
  {
    key: 'ucbBeta',
    name: 'GP-UCB exploration schedule',
    summary: 'The weight β_t = 2 log(|D| t² π² / 6δ) of the GP-UCB regret bound on a finite candidate set.',
    role: 'construction',
    cite: ['srinivas2010'],
  },
  acquisitions.ucbBeta,
)

fn(
  {
    key: 'maxValueSamples',
    name: 'Optimal-value samples (Gumbel)',
    summary: 'Samples of the optimal value from a Gumbel law fitted to the product of the marginal posterior cdfs.',
    role: 'simulation',
    random: true,
    cite: ['wang2017mes'],
  },
  entropy.maxValueSamples,
)

fn(
  {
    key: 'maxValueEntropySearch',
    name: 'Max-value entropy search',
    summary: 'The information an evaluation gives about the optimal value, averaged over samples of it.',
    role: 'estimator',
    random: true,
    cite: ['wang2017mes'],
  },
  entropy.maxValueEntropySearch,
)

/** The algorithms of the module, keyed by factory name. */
export const bayesianAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', loop) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module (the acquisitions and their helpers), keyed by name. */
export const bayesianFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', acquisitions, entropy) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
