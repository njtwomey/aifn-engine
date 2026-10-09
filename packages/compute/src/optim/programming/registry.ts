/**
 * The algorithms of `aifn-compute/optim/programming`, registered with what each factory takes (`problem`) and the
 * roles of its state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a
 * generic trace view picks default series and a worker can address an algorithm by key (design S §2.3). The one-call
 * solvers and constructions are registered as functions, with their role and the notes they belong to.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as linprog from './linprog'
import * as lp from './lp'
import * as assignment from './assignment'
import * as dp from './dp'
import * as sequences from './sequences'
import * as interior from './interior'
import * as milp from './milp'
import * as qp from './qp'
import * as simplex from './simplex'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/programming')

algorithm(
  {
    key: 'simplex',
    stability: 'stable',
    name: 'Simplex method',
    problem: 'linear-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'terminated'] },
    notes: ['linear-programming'],
  },
  simplex.simplex,
)
algorithm(
  {
    key: 'linearInteriorPoint',
    stability: 'stable',
    name: 'Interior point (LP)',
    problem: 'linear-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-programming'],
    cite: ['karmarkar1984'],
  },
  interior.linearInteriorPoint,
)
algorithm(
  {
    key: 'activeSet',
    stability: 'stable',
    name: 'Active set (QP)',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['quadratic-programming'],
    cite: ['nocedal2006'],
  },
  qp.activeSet,
)
algorithm(
  {
    key: 'quadraticInteriorPoint',
    stability: 'stable',
    name: 'Interior point (QP)',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['quadratic-programming'],
    cite: ['nocedal2006'],
  },
  qp.quadraticInteriorPoint,
)
algorithm(
  {
    key: 'boxQuadraticProgram',
    stability: 'stable',
    name: 'Box-constrained QP',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'objective', grad: 'grad', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['quadratic-programming'],
  },
  qp.boxQuadraticProgram,
)
algorithm(
  {
    key: 'branchAndBound',
    stability: 'stable',
    name: 'Branch and bound',
    problem: 'integer-program',
    state: { iterate: 'incumbent', objective: 'incumbentValue', flags: ['converged', 'terminated'] },
  },
  milp.branchAndBound,
)
algorithm(
  {
    key: 'gomory',
    stability: 'stable',
    name: 'Gomory cutting planes',
    problem: 'integer-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'terminated'] },
  },
  milp.gomory,
)
algorithm(
  {
    key: 'hungarianSteps',
    stability: 'stable',
    name: 'Hungarian algorithm',
    problem: 'assignment',
    state: { flags: ['converged'] },
    cite: ['kuhn1955'],
  },
  assignment.hungarianSteps,
)
algorithm(
  {
    key: 'dynamicProgram',
    stability: 'stable',
    name: 'Dynamic programming',
    problem: 'dynamic-program',
    state: { iterate: 'table', flags: ['converged'] },
    notes: ['bellman-equations'],
    cite: ['bellman1957'],
  },
  dp.dynamicProgram,
)

/** Every algorithm of the module, keyed by factory name. */
export const programmingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', assignment, dp, interior, milp, qp, simplex) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'optim/programming')
const LP = ['linear-programming']
const QP = ['quadratic-programming', 'karush-kuhn-tucker-conditions']

fn(
  {
    key: 'linprog',
    name: 'Solve a linear program',
    summary:
      'min cᵀx subject to linear inequalities, equalities and bounds, by the simplex or an interior-point method.',
    role: 'solver',
    notes: LP,
    cite: ['dantzig1963'],
  },
  linprog.linprog,
)
fn(
  { key: 'simplexSolve', name: 'Solve by the simplex method', role: 'solver', notes: LP, cite: ['dantzig1963'] },
  simplex.simplexSolve,
)
fn(
  {
    key: 'interiorPointSolve',
    name: 'Solve by a primal–dual interior-point method',
    role: 'solver',
    notes: [...LP, 'penalty-and-barrier-methods'],
    cite: ['karmarkar1984'],
  },
  interior.interiorPointSolve,
)
fn(
  {
    key: 'lpCentralPath',
    name: 'Central path of a linear program',
    role: 'solver',
    notes: [...LP, 'penalty-and-barrier-methods'],
  },
  interior.lpCentralPath,
)
fn(
  {
    key: 'dualityReport',
    name: 'LP duality report',
    summary: 'Primal and dual objectives, the gap and complementary slackness.',
    role: 'property',
    notes: [...LP, 'lagrangian-duality'],
  },
  lp.dualityReport,
)
fn({ key: 'standardForm', name: 'LP in standard form', role: 'transform', notes: LP }, lp.standardForm)
fn({ key: 'quadprog', name: 'Solve a quadratic program', role: 'solver', notes: QP }, qp.quadprog)
fn(
  { key: 'boxQuadprog', name: 'Solve a box-constrained QP', role: 'solver', notes: ['quadratic-programming'] },
  qp.boxQuadprog,
)
fn(
  {
    key: 'kktReport',
    name: 'KKT conditions report',
    summary: 'Stationarity, primal and dual feasibility and complementary slackness at a point.',
    role: 'property',
    notes: ['karush-kuhn-tucker-conditions', 'lagrange-multipliers'],
    cite: ['boyd2004'],
  },
  qp.kktReport,
)
fn({ key: 'milp', name: 'Solve a mixed-integer linear program', role: 'solver', notes: LP }, milp.milp)
fn(
  { key: 'branchAndBoundTree', name: 'Branch-and-bound tree', role: 'solver', returns: 'tree', notes: LP },
  milp.branchAndBoundTree,
)
fn(
  { key: 'hungarian', name: 'Hungarian method (assignment)', role: 'solver', cite: ['kuhn1955'] },
  assignment.hungarian,
)

fn(
  { key: 'lcsProgram', name: 'Longest common subsequence as a dynamic program', role: 'construction' },
  sequences.lcsProgram,
)
fn(
  { key: 'lcs', name: 'Longest common subsequence', role: 'solver', stability: 'stable', notes: ['rouge'] },
  sequences.lcs,
)
fn(
  {
    key: 'editDistanceProgram',
    name: 'Edit distance as a dynamic program',
    role: 'construction',
    notes: ['word-and-character-error-rates'],
  },
  sequences.editDistanceProgram,
)
fn(
  {
    key: 'editDistance',
    name: 'Levenshtein edit distance',
    role: 'solver',
    stability: 'stable',
    notes: ['word-and-character-error-rates'],
  },
  sequences.editDistance,
)
fn(
  { key: 'alignmentProgram', name: 'Sequence alignment as a dynamic program', role: 'construction' },
  sequences.alignmentProgram,
)
fn(
  { key: 'needlemanWunsch', name: 'Needleman–Wunsch global alignment', role: 'solver', stability: 'stable' },
  sequences.needlemanWunsch,
)
fn(
  { key: 'smithWaterman', name: 'Smith–Waterman local alignment', role: 'solver', stability: 'stable' },
  sequences.smithWaterman,
)

/** The functions of the module (solvers, reports and program constructions), keyed by name. */
export const programmingFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', linprog, simplex, interior, lp, qp, milp, assignment, dp, sequences) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
