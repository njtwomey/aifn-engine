/** The functions of `aifn-compute/logic/propositional`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as formulas from './formulas'

const fn = definer<FunctionInfo>('function', 'logic/propositional')
const notes = ['propositional-and-first-order-logic-for-learning']

fn(
  {
    key: 'entails',
    name: 'Propositional entailment',
    tex: '\\mathrm{KB} \\models \\varphi',
    role: 'solver',
    summary:
      'Whether a knowledge base entails a formula, by checking the formula in every model of the knowledge base; with the models and counter-models.',
    notes,
  },
  formulas.entails,
)
fn(
  {
    key: 'modelsOf',
    name: 'Models of a set of formulas',
    role: 'solver',
    summary: 'The rows of the truth table that make every formula of a set true.',
    notes,
  },
  formulas.modelsOf,
)
fn(
  {
    key: 'interpretations',
    name: 'Truth-table rows',
    role: 'construction',
    summary: 'All 2^n interpretations of n atoms in truth-table order, the first atom varying slowest, true first.',
    notes,
  },
  formulas.interpretations,
)
fn(
  {
    key: 'evaluateFormula',
    name: 'Truth value of a formula',
    role: 'property',
    summary: 'The truth value of a propositional formula under an interpretation of its atoms.',
    notes,
  },
  formulas.evaluateFormula,
)
fn(
  {
    key: 'formulaText',
    name: 'Print a formula',
    role: 'transform',
    summary: 'A formula as Unicode text or KaTeX, bracketed only where precedence needs it.',
    notes,
  },
  formulas.formulaText,
)

/** The functions of the module, keyed by name. */
export const propositionalFunctions = entries<FunctionInfo>('function', formulas) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
