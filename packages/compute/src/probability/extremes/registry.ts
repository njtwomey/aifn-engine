/**
 * The registry of `aifn-compute/probability/extremes`: the generalised Pareto fit and the peaks-over-threshold
 * estimates.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as extremes from './extremes'

const fn = definer<FunctionInfo>('function', 'probability/extremes')
const notes = ['extreme-value-theory-for-anomalies']
const cite = ['pickands1975']

fn(
  {
    key: 'fitGeneralisedPareto',
    name: 'Generalised Pareto fit',
    summary: 'Maximum-likelihood shape and scale of excesses over a threshold, by the one-dimensional profile in ξ/σ.',
    role: 'fit',
    notes,
    cite,
  },
  extremes.fitGeneralisedPareto,
)
fn(
  {
    key: 'peaksOverThreshold',
    name: 'Peaks over threshold',
    summary: 'Keep the observations above a threshold and fit a generalised Pareto law to their excesses.',
    role: 'fit',
    notes,
    cite,
  },
  extremes.peaksOverThreshold,
)
fn(
  {
    key: 'tailProbability',
    name: 'Tail probability',
    tex: '\\zeta_u (1 + \\xi (x - u)/\\sigma)^{-1/\\xi}',
    summary: 'P(X > x) beyond the threshold under a peaks-over-threshold fit.',
    role: 'property',
    notes,
  },
  extremes.tailProbability,
)
fn(
  {
    key: 'tailQuantile',
    name: 'Tail quantile',
    summary: 'A high quantile beyond the data from a peaks-over-threshold fit: the threshold at a chosen risk.',
    role: 'property',
    notes,
  },
  extremes.tailQuantile,
)
fn(
  {
    key: 'meanExcess',
    name: 'Mean excess function',
    summary: 'The mean of x − u over the observations above each threshold u (the mean residual life plot).',
    role: 'estimator',
    notes,
  },
  extremes.meanExcess,
)

/** The functions of the module, keyed by name. */
export const extremesFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', extremes) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
