/**
 * `aifn-compute/optim/programming`: mathematical programming: linear programmes (simplex, interior point, `linprog`),
 * quadratic programmes (active set, interior point, box-constrained, `quadprog`), mixed-integer programmes (branch and
 * bound, Gomory cuts, `milp`), assignment (`hungarian`), and the dynamic-programming engine (`dynamicProgram`, `dp`)
 * with the sequence programmes on it (`lcs`, `editDistance`, `needlemanWunsch`, `smithWaterman`).
 * Every method is an `Algorithm` factory over its problem, paired with a one-call runner.
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
