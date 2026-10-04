/** The functions of `aifn-compute/learning/validate`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cross from './cross'
import * as search from './search'
import * as splitters from './splitters'

const fn = definer<FunctionInfo>('function', 'learning/validate')
const CV = ['cross-validation']
const TSCV = ['time-series-cross-validation']

fn(
  { key: 'kFold', name: 'k-fold split', role: 'construction', notes: CV, cite: ['stone1974', 'kohavi1995'] },
  splitters.kFold,
)
fn(
  { key: 'stratifiedKFold', name: 'Stratified k-fold split', role: 'construction', notes: [...CV, 'class-imbalance'] },
  splitters.stratifiedKFold,
)
fn(
  { key: 'groupKFold', name: 'Group k-fold split', role: 'construction', notes: [...CV, 'data-leakage'] },
  splitters.groupKFold,
)
fn({ key: 'leaveOneOut', name: 'Leave-one-out split', role: 'construction', notes: CV }, splitters.leaveOneOut)
fn(
  { key: 'shuffleSplit', name: 'Shuffle split', role: 'construction', random: true, notes: CV },
  splitters.shuffleSplit,
)
fn({ key: 'repeated', name: 'Repeated splits', role: 'construction', random: true, notes: CV }, splitters.repeated)
fn(
  { key: 'expandingWindow', name: 'Expanding-window split', role: 'construction', notes: TSCV, cite: ['tashman2000'] },
  splitters.expandingWindow,
)
fn(
  {
    key: 'rollingOrigin',
    name: 'Rolling-origin split',
    role: 'construction',
    notes: TSCV,
    cite: ['tashman2000', 'hyndman2021'],
  },
  splitters.rollingOrigin,
)
fn({ key: 'assignment', name: 'Fold assignment', role: 'property', notes: CV }, splitters.assignment)
fn({ key: 'crossValidate', name: 'Cross-validate', role: 'estimator', notes: CV }, cross.crossValidate)
fn(
  { key: 'gridSearch', name: 'Grid search', role: 'solver', notes: ['hyperparameter-search', ...CV] },
  search.gridSearch,
)
fn(
  {
    key: 'randomSearch',
    name: 'Random search',
    role: 'solver',
    random: true,
    notes: ['hyperparameter-search'],
    cite: ['bergstra2012'],
  },
  search.randomSearch,
)
fn(
  {
    key: 'nested',
    name: 'Nested cross-validation',
    role: 'estimator',
    notes: [...CV, 'hyperparameter-search', 'data-leakage'],
    cite: ['varma2006', 'cawley2010'],
  },
  search.nested,
)

/** The functions of the module, keyed by name. */
export const validateFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', splitters, cross, search) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
