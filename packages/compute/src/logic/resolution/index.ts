/**
 * `aifn-compute/logic/resolution`: a small Prolog, SLD resolution with depth-first search and backtracking.
 *
 * - Programs: `prologProgram` indexes clauses by predicate and adds the library predicates the program does not define
 *   (`LIBRARY_SOURCE`: `member`, `append`, `length`, `between`, ...).
 * - Running a query: `sldSteps` is the step-through algorithm, each step one node of the SLD tree (the goal, clause
 *   and unifier; branches a cut removes reported as pruned); `solveQuery` runs it to the end with step and depth
 *   limits and says why it stopped; `formatSolution` prints a solution's bindings.
 * - Reading the search: `sldTree` builds the SLD tree from a trace's states, with when each node was created and
 *   closed.
 * - `resolutionAlgorithms` and `resolutionFunctions`: the module's registry entries.
 *
 * Built-ins include cut, negation as failure, if-then-else, `call/1`, `findall/3`, unification and identity tests,
 * arithmetic and type tests. Solutions come in standard Prolog order. A goal that raises an error (an unknown
 * predicate, unbound arithmetic) ends the run with the error in the state; nothing is thrown.
 */

export {
  formatSolution,
  prologProgram,
  sldSteps,
  solveQuery,
  type PrologProgram,
  type PrunedBranch,
  type SldChoice,
  type SldEvent,
  type SldGoal,
  type SldOptions,
  type SldSolution,
  type SldState,
  type SolveResult,
} from './sld'
export { sldTree, type SldNode, type SldNodeStatus } from './tree'
export { LIBRARY_SOURCE } from './library'
export { resolutionAlgorithms, resolutionFunctions } from './registry'
