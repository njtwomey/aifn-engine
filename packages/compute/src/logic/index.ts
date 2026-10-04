/**
 * `aifn-compute/logic`: logic programming. Children: terms (first-order terms, unification, reading and printing Prolog),
 * resolution (a small Prolog: SLD resolution with backtracking, cut, negation as failure and findall, step by step)
 * and induction (learning clauses from examples: FOIL, θ-subsumption and least general generalisation).
 */

export { parseProgram, parseQuery, termToString, unify, type Clause, type Term } from './terms'
export { prologProgram, sldSteps, solveQuery } from './resolution'
export { foil, foilProblem, foilSteps } from './induction'
