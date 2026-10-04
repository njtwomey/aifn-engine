/** The functions of `aifn-compute/logic/terms`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as syntax from './syntax'
import * as terms from './terms'

const fn = definer<FunctionInfo>('function', 'logic/terms')

fn(
  {
    key: 'unify',
    name: 'Most general unifier',
    role: 'solver',
    summary:
      'The most general unifier of two first-order terms (Robinson; Martelli–Montanari), idempotent with the occurs check; null when they do not unify.',
  },
  terms.unify,
)
fn(
  {
    key: 'matchTerm',
    name: 'One-way matching',
    role: 'solver',
    summary:
      "A substitution θ of the pattern's variables with pattern θ = term, treating the term's variables as constants.",
  },
  terms.matchTerm,
)
fn({ key: 'applySubstitution', name: 'Apply a substitution', role: 'transform' }, terms.applySubstitution)
fn({ key: 'composeSubstitutions', name: 'Compose substitutions', role: 'transform' }, terms.composeSubstitutions)
fn({ key: 'renameVariables', name: 'Rename variables apart', role: 'transform' }, terms.renameVariables)
fn({ key: 'termToString', name: 'Print a term in Prolog syntax', role: 'transform' }, terms.termToString)
fn(
  {
    key: 'parseProgram',
    name: 'Read a Prolog program',
    role: 'transform',
    summary:
      'Clauses and queries from Prolog text: operators with standard priorities, lists, comments; errors with line and column.',
  },
  syntax.parseProgram,
)
fn({ key: 'parseQuery', name: 'Read a Prolog query', role: 'transform' }, syntax.parseQuery)
fn({ key: 'parseTerm', name: 'Read a Prolog term', role: 'transform' }, syntax.parseTerm)
fn({ key: 'clauseToString', name: 'Print a clause', role: 'transform' }, syntax.clauseToString)

/** The functions of the module, keyed by name. */
export const termsFunctions = entries<FunctionInfo>('function', terms, syntax) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
