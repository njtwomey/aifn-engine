/**
 * `aifn-compute/logic`: logic and logic programming, from propositional formulas and first-order terms to a small
 * Prolog and to learning clauses from examples.
 *
 * - `aifn-compute/logic/propositional`: propositional formulas, their models and entailment by truth table.
 * - `aifn-compute/logic/terms`: first-order terms and unification (the most general unifier, also step by step by
 *   Martelli and Montanari's rules, one-way matching, substitutions), and reading and printing Prolog text.
 * - `aifn-compute/logic/resolution`: a small Prolog: SLD resolution with backtracking, cut, negation as failure and
 *   `findall`, step by step (`sldSteps`) or to the end (`solveQuery`), the SLD tree of a run, and bottom-up
 *   evaluation of Datalog to its least Herbrand model.
 * - `aifn-compute/logic/induction`: inductive logic programming and concept learning: FOIL, $\theta$-subsumption and
 *   the least general generalisation (traced, and relative to background facts), Progol's bottom clauses, Metagol,
 *   learning from failures, rule mining on knowledge graphs, propositionalisation, and version spaces of attribute
 *   conjunctions.
 *
 * The family re-exports the entry points: `parseProgram`, `parseQuery`, `termToString`, `unify`, `prologProgram`,
 * `sldSteps`, `solveQuery`, `foilProblem`, `foilSteps` and `foil`.
 */

export { parseProgram, parseQuery, termToString, unify, type Clause, type Term } from './terms'
export { prologProgram, sldSteps, solveQuery } from './resolution'
export { foil, foilProblem, foilSteps } from './induction'
