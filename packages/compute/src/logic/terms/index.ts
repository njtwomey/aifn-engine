/**
 * `aifn-compute/logic/terms`: first-order terms and unification, and reading and printing them as Prolog text.
 *
 * - Terms as plain data: `atom`, `numeral`, `variable`, `compound`, lists with `listTerm` and `listItems` (`NIL` the
 *   empty list); `indicator` (a predicate's `name/arity`), `termVariables`, `isGround` and `termsEqual` (Prolog's
 *   `==`).
 * - Unification and substitutions: `unify`, the most general unifier (the occurs check optional; null when the terms
 *   do not unify), `matchTerm` (one-way, binding only the pattern's variables), `applySubstitution`,
 *   `composeSubstitutions`, and `renameVariables` to rename a clause apart.
 * - Prolog text: `parseProgram` (clauses and queries), `parseQuery`, `parseTerm`, with the standard operators
 *   (`INFIX`, `PREFIX`); `termToString`, `substitutionToString` and `clauseToString` print them back; `conjuncts` and
 *   `conjunction` convert between a body and its goals. A syntax error throws `PrologSyntaxError` with a line and
 *   column.
 * - `termsFunctions`: the module's registry entries.
 *
 * A variable is identified by its integer id, not its name: each clause and query read numbers its variables from 0,
 * so terms from different clauses are renamed apart before they are unified.
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
