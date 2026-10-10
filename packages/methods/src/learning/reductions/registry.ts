/**
 * Registry entries of `aifn-methods/learning/reductions`: the code matrices, `codeDistance`, the class trees and
 * `softmaxScores`, each with its notes.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as multiclass from './multiclass'

const fn = definer<FunctionInfo>('function', 'learning/reductions')
const ECOC = ['error-correcting-output-codes', 'multiclass-classification']

fn(
  { key: 'oneVersusRestCode', name: 'One-versus-rest code', role: 'construction', notes: ['one-versus-rest', ...ECOC] },
  multiclass.oneVersusRestCode,
)
fn(
  { key: 'oneVersusOneCode', name: 'One-versus-one code', role: 'construction', notes: ['one-versus-one', ...ECOC] },
  multiclass.oneVersusOneCode,
)
fn({ key: 'exhaustiveCode', name: 'Exhaustive code', role: 'construction', notes: ECOC }, multiclass.exhaustiveCode)
fn({ key: 'randomCode', name: 'Random code', role: 'construction', random: true, notes: ECOC }, multiclass.randomCode)
fn({ key: 'codeDistance', name: 'Code distance', role: 'property', notes: ECOC }, multiclass.codeDistance)
fn(
  {
    key: 'dichotomyTree',
    name: 'Nested dichotomy tree',
    role: 'construction',
    notes: ['nested-dichotomies-and-tree-reductions'],
  },
  multiclass.dichotomyTree,
)
fn(
  {
    key: 'randomDichotomyTree',
    name: 'Random nested dichotomy',
    role: 'construction',
    random: true,
    notes: ['nested-dichotomies-and-tree-reductions'],
  },
  multiclass.randomDichotomyTree,
)
fn(
  {
    key: 'softmaxScores',
    name: 'Softmax of scores',
    role: 'transform',
    notes: ['multiclass-classification', 'softmax-and-log-sum-exp'],
  },
  multiclass.softmaxScores,
)

/** The functions of the module, keyed by name. */
export const reductionsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', multiclass) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
