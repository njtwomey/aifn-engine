/**
 * `aifn-compute/logic/terms`: first-order terms and unification. Atoms, numbers, variables, compound terms and lists as plain
 * data; the most general unifier (with or without the occurs check), one-way matching, substitution application and
 * composition, renaming apart; reading Prolog text (clauses, queries, terms; operators with standard priorities) and
 * printing terms and clauses back.
 */

export {
  applySubstitution,
  atom,
  composeSubstitutions,
  compound,
  indicator,
  INFIX,
  isGround,
  listItems,
  listTerm,
  matchTerm,
  NIL,
  numeral,
  PREFIX,
  renameVariables,
  substitutionToString,
  termsEqual,
  termToString,
  termVariables,
  unify,
  variable,
  type Atom,
  type Binding,
  type Compound,
  type Numeral,
  type PrintOptions,
  type Substitution,
  type Term,
  type UnifyOptions,
  type Variable,
} from './terms'
export {
  clauseToString,
  conjunction,
  conjuncts,
  parseProgram,
  parseQuery,
  parseTerm,
  PrologSyntaxError,
  type Clause,
  type Query,
} from './syntax'
export { termsFunctions } from './registry'
