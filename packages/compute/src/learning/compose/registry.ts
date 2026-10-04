/** The functions of `aifn-compute/learning/compose`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as columns from './columns'
import * as pipeline from './pipeline'
import * as target from './target'

const fn = definer<FunctionInfo>('function', 'learning/compose')

fn(
  {
    key: 'pipeline',
    name: 'Pipeline',
    summary: 'Chain transforms and a final estimator, fitted together so preprocessing sees only the training fold.',
    role: 'construction',
    notes: ['data-leakage', 'feature-scaling'],
  },
  pipeline.pipeline,
)
fn(
  {
    key: 'columns',
    name: 'Column transformer',
    role: 'construction',
    notes: ['categorical-encoding', 'feature-scaling'],
  },
  columns.columns,
)
fn({ key: 'transformTarget', name: 'Transformed target', role: 'construction' }, target.transformTarget)
fn({ key: 'logTarget', name: 'Log target', role: 'construction' }, target.logTarget)
fn({ key: 'log1pTarget', name: 'log(1 + y) target', role: 'construction' }, target.log1pTarget)
fn({ key: 'powerTarget', name: 'Power target', role: 'construction' }, target.powerTarget)
fn({ key: 'affineTarget', name: 'Affine target', role: 'construction' }, target.affineTarget)
fn(
  { key: 'standardTarget', name: 'Standardised target', role: 'construction', notes: ['feature-scaling'] },
  target.standardTarget,
)
fn(
  {
    key: 'pushForward',
    name: 'Push a predictive through a target map',
    role: 'transform',
    notes: ['change-of-variables'],
  },
  target.pushForward,
)
fn(
  { key: 'transformedPredictive', name: 'Transformed predictive', role: 'transform', notes: ['change-of-variables'] },
  target.transformedPredictive,
)
fn(
  { key: 'logNormalPredictive', name: 'Log-normal predictive', role: 'transform', notes: ['log-normal-distribution'] },
  target.logNormalPredictive,
)

/** The functions of the module, keyed by name. */
export const composeFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', pipeline, columns, target) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
