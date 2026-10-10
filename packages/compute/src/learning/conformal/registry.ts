/**
 * The registry of `aifn-compute/learning/conformal`: the conformal quantile, the regression and classification
 * procedures, Mondrian quantiles and the coverage summaries, all registered as functions with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as conformal from './conformal'

const fn = definer<FunctionInfo>('function', 'learning/conformal')
const CP = ['conformal-prediction']

fn(
  {
    key: 'conformalQuantile',
    name: 'Conformal quantile',
    tex: '\\hat q = s_{(\\lceil (n+1)(1-\\alpha) \\rceil)}',
    summary: 'The ⌈(n + 1)(1 − α)⌉-th smallest calibration score (+∞ when there are too few).',
    role: 'estimator',
    notes: CP,
    cite: ['vovk2005', 'angelopoulos2021'],
  },
  conformal.conformalQuantile,
)
fn(
  {
    key: 'splitConformalRegression',
    name: 'Split conformal regression',
    tex: '\\hat y \\pm \\hat q',
    summary: 'Intervals ŷ ± q̂ from the absolute residuals of held-out cases.',
    role: 'fit',
    notes: CP,
    cite: ['lei2018'],
  },
  conformal.splitConformalRegression,
)
fn(
  {
    key: 'conformalisedQuantileRegression',
    name: 'Conformalised quantile regression (CQR)',
    tex: '[l(x) - \\hat q,\\ u(x) + \\hat q]',
    summary: 'Widen a quantile model’s interval by the conformal quantile of max(l − y, y − u).',
    role: 'fit',
    notes: [...CP, 'quantile-calibration'],
    cite: ['romano2019'],
  },
  conformal.conformalisedQuantileRegression,
)
fn(
  {
    key: 'classificationScores',
    name: 'Classification nonconformity scores',
    summary: 'LAC (1 − p_y), APS (mass ranked at or above y) and RAPS (APS plus a rank penalty).',
    role: 'transform',
    notes: CP,
    cite: ['sadinle2019', 'romano2020', 'angelopoulos2021'],
    random: true,
  },
  conformal.classificationScores,
)
fn(
  {
    key: 'conformalClassification',
    name: 'Conformal classification sets',
    summary: 'Every class whose score is at most q̂: sets that cover the true label with probability ≥ 1 − α.',
    role: 'fit',
    notes: [...CP, 'selective-classification'],
    cite: ['romano2020', 'angelopoulos2021'],
    random: true,
  },
  conformal.conformalClassification,
)
fn(
  {
    key: 'mondrianQuantiles',
    name: 'Mondrian conformal quantiles',
    summary: 'One conformal quantile per group, so coverage holds within each group.',
    role: 'estimator',
    notes: CP,
    cite: ['vovk2005'],
  },
  conformal.mondrianQuantiles,
)
fn(
  {
    key: 'intervalCoverage',
    name: 'Interval coverage',
    summary: 'The share of targets inside their intervals, and the mean width.',
    role: 'estimator',
    notes: [...CP, 'quantile-calibration'],
    cite: ['angelopoulos2021'],
  },
  conformal.intervalCoverage,
)
fn(
  {
    key: 'setCoverage',
    name: 'Set coverage',
    summary: 'The share of labels inside their prediction sets, and the mean set size.',
    role: 'estimator',
    notes: CP,
    cite: ['angelopoulos2021'],
  },
  conformal.setCoverage,
)

/** The functions of the module, keyed by name. */
export const conformalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', conformal) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
