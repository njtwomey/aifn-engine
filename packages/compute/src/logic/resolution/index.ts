/**
 * `aifn-compute/logic/resolution`: a small Prolog. SLD resolution with depth-first search and backtracking as a step-through
 * algorithm (`sldSteps`: each step one node of the SLD tree, with the goal, clause and unifier; cut-pruned branches
 * reported), `solveQuery` to run a query to the end with step and depth limits, programs with a library of list predicates
 * (`prologProgram`), and `sldTree` to read the search tree out of a trace.
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
