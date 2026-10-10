/**
 * `aifn-compute/logic`: logic programming, from first-order terms to a small Prolog and to learning clauses from
 * examples.
 *
 * - `aifn-compute/logic/terms`: first-order terms and unification (the most general unifier, one-way matching,
 *   substitutions), and reading and printing Prolog text.
 * - `aifn-compute/logic/resolution`: a small Prolog: SLD resolution with backtracking, cut, negation as failure and
 *   `findall`, step by step (`sldSteps`) or to the end (`solveQuery`), and the SLD tree of a run.
 * - `aifn-compute/logic/induction`: inductive logic programming: FOIL, $\theta$-subsumption and the least general
 *   generalisation of terms and clauses.
 *
 * The family re-exports the entry points: `parseProgram`, `parseQuery`, `termToString`, `unify`, `prologProgram`,
 * `sldSteps`, `solveQuery`, `foilProblem`, `foilSteps` and `foil`.
 */

export { parseProgram, parseQuery, termToString, unify, type Clause, type Term } from './terms'
export { prologProgram, sldSteps, solveQuery } from './resolution'
export { foil, foilProblem, foilSteps } from './induction'
