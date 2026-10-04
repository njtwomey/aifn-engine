/** The algorithms and functions of `aifn-compute/logic/resolution`. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as sld from './sld'
import * as tree from './tree'

const algorithm = definer<AlgorithmInfo>('algorithm', 'logic/resolution')
const fn = definer<FunctionInfo>('function', 'logic/resolution')

algorithm(
  {
    key: 'sldSteps',
    name: 'SLD resolution (Prolog)',
    summary:
      'Depth-first SLD resolution with backtracking: each step resolves the first goal with the next clause whose head unifies, runs a built-in, or backtracks; cut, negation as failure and findall included.',
    problem: 'logic-program',
    tags: ['prolog', 'logic programming', 'backtracking'],
    state: { iterate: 'goals', flags: ['terminated'] },
  },
  sld.sldSteps,
)
fn(
  {
    key: 'solveQuery',
    name: 'Solve a Prolog query',
    role: 'solver',
    summary:
      'Every solution of a query in standard Prolog order, with step and depth limits and a message saying why the search stopped.',
  },
  sld.solveQuery,
)
fn(
  {
    key: 'prologProgram',
    name: 'Prolog program',
    role: 'construction',
    summary:
      'Clauses indexed by predicate, with the library predicates (member, append, …) the program does not define.',
  },
  sld.prologProgram,
)
fn({ key: 'formatSolution', name: 'Print a solution', role: 'transform' }, sld.formatSolution)
fn(
  {
    key: 'sldTree',
    name: 'SLD tree of a run',
    role: 'transform',
    summary:
      'The search tree of a traced SLD run: nodes with goal, clause, unifier, resolvent, status and the steps that created and closed them.',
  },
  tree.sldTree,
)

/** The step-through algorithms of the module, keyed by factory name. */
export const resolutionAlgorithms = entries<AlgorithmInfo>('algorithm', sld) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

/** The functions of the module, keyed by name. */
export const resolutionFunctions = entries<FunctionInfo>('function', sld, tree) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
