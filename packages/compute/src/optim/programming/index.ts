/**
 * `aifn-compute/optim/programming`: mathematical programming, from linear and quadratic programs to integer
 * programs, assignment and dynamic programming, as `scipy.optimize`'s `linprog`, `milp` and `linear_sum_assignment`.
 *
 * - Linear programs (minimise $\cvec^\top\xvec$ subject to $\Amat_{\text{ub}}\xvec \le \bvec_{\text{ub}}$,
 *   $\Amat_{\text{eq}}\xvec = \bvec_{\text{eq}}$ and bounds): `linprog` in one call; `simplex` (two-phase tableau,
 *   every pivot visible) and `linearInteriorPoint` (Mehrotra's predictor–corrector on the homogeneous self-dual
 *   embedding, which certifies infeasibility and unboundedness) step by step; `simplexDuals`, `dualityReport` and
 *   `lpCentralPath` for duals, optimality certificates and the central path.
 * - Quadratic programs (minimise $\frac{1}{2}\xvec^\top\Qmat\xvec + \cvec^\top\xvec$ subject to
 *   $\Amat\xvec \le \bvec$, $\Emat\xvec = \evec$): `quadprog` in one call; `activeSet` (exact, from a feasible
 *   start) and `quadraticInteriorPoint` (from any start) step by step; `boxQuadraticProgram` and `boxQuadprog` for
 *   bounds only; `kktReport` to check a point and its multipliers.
 * - Integer programs: `milp` and `branchAndBound` (mixed-integer, the whole search tree kept, drawn by
 *   `branchAndBoundTree`), and `gomory` (fractional cutting planes, pure integer programs with integer data).
 * - Assignment: `hungarian` in one call, `hungarianSteps` through Munkres' steps; rectangular cost matrices allowed.
 * - Dynamic programming: `dynamicProgram` (row by row) and `dp` (at once) fill a table from a cell rule; the sequence
 *   programs on it are `lcs`, `editDistance`, `needlemanWunsch` and `smithWaterman`, each with its `…Program` for
 *   stepping, and the traceback moves `DIAGONAL`, `UP`, `LEFT` and `STOP`.
 *
 * Every method is an `Algorithm` factory over its problem, paired with a one-call runner. Problems are given as
 * tensors or plain arrays. The linear, quadratic and integer programs minimise (negate the objective to maximise;
 * `hungarian` takes `maximize`). Infeasibility, unboundedness, nonconvexity and step limits are reported in a
 * result's `status`, not thrown; ill-formed data (wrong sizes, non-finite entries) throws. LP duals follow scipy's
 * sign convention, and QP multipliers the Lagrangian's ($\lambdavec \ge \zeros$). `programmingAlgorithms` and
 * `programmingFunctions` register the module.
 */

export {
  dualityReport,
  type Bound,
  type DualityReport,
  type LinearProgram,
  type LinearProgramDuals,
  type StandardForm,
} from './lp'
export {
  simplex,
  simplexDuals,
  type LinearProgramMethod,
  type LinearProgramResult,
  type LinearProgramStatus,
  type LinprogOptions,
  type SimplexEvent,
  type SimplexOptions,
  type SimplexRule,
  type SimplexState,
  type SimplexStatus,
} from './simplex'
export {
  linearInteriorPoint,
  lpCentralPath,
  type CentralPath,
  type InteriorPointOptions,
  type InteriorPointState,
  type InteriorPointStatus,
} from './interior'
export { linprog } from './linprog'
export {
  activeSet,
  boxQuadraticProgram,
  boxQuadprog,
  kktReport,
  quadraticInteriorPoint,
  quadprog,
  type ActiveSetEvent,
  type ActiveSetOptions,
  type ActiveSetStart,
  type ActiveSetState,
  type BoxQuadraticProblem,
  type BoxQuadraticProgramOptions,
  type BoxQuadraticProgramStart,
  type BoxQuadraticProgramState,
  type KKTReport,
  type QuadprogOptions,
  type QuadraticInteriorPointOptions,
  type QuadraticInteriorPointState,
  type QuadraticProgram,
  type QuadraticProgramResult,
  type QuadraticProgramStatus,
} from './qp'
export {
  branchAndBound,
  branchAndBoundTree,
  gomory,
  milp,
  type BranchAndBoundOptions,
  type BranchAndBoundState,
  type BranchNode,
  type BranchNodeStatus,
  type Cut,
  type GomoryOptions,
  type GomoryState,
  type IntegerProgramStatus,
  type MixedIntegerProgram,
  type MixedIntegerResult,
  type NodeSelection,
  type SearchTreeEdgeData,
  type SearchTreeNodeData,
} from './milp'
export { dp, dynamicProgram, type DynamicProgram, type DynamicProgramState } from './dp'
export {
  alignmentProgram,
  DIAGONAL,
  editDistance,
  editDistanceProgram,
  LEFT,
  lcs,
  lcsProgram,
  needlemanWunsch,
  smithWaterman,
  STOP,
  UP,
  type AlignmentResult,
  type AlignmentScoring,
  type EditCosts,
  type EditDistanceResult,
  type EditOperation,
  type LCSResult,
  type Sequence,
} from './sequences'
export {
  hungarian,
  hungarianSteps,
  type AssignmentResult,
  type HungarianOptions,
  type HungarianPhase,
  type HungarianState,
} from './assignment'
export { programmingAlgorithms, programmingFunctions } from './registry'
