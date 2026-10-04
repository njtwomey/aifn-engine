/** The registry of `aifn-methods/theory/bias-variance`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bv from './bias-variance'

const fn = definer<FunctionInfo>('function', 'theory/bias-variance')
const notes = ['bias-variance-decomposition']

fn(
  {
    key: 'biasVariance',
    name: 'Bias–variance by resampling',
    summary: 'Fit one model to many training sets; bias² of the mean fit, variance of the fits and noise, per x.',
    role: 'simulation',
    random: true,
    notes,
    cite: ['geman1992', 'hastie2009'],
  },
  bv.biasVariance,
)
fn(
  {
    key: 'biasVarianceSweep',
    name: 'Bias–variance across complexity',
    summary: 'The decomposition for each polynomial degree or neighbour count, with the training error.',
    role: 'simulation',
    random: true,
    notes,
    cite: ['geman1992'],
  },
  bv.biasVarianceSweep,
)
fn({ key: 'fitAndPredict', name: 'Fit and predict (resampling models)', role: 'fit', notes }, bv.fitAndPredict)

/** The functions of the module, keyed by name. */
export const biasVarianceFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', bv) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
