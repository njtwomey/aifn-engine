/**
 * Integer and mixed-integer linear programming: branch and bound over the LP relaxation (Land and Doig, 1960, "An
 * automatic method of solving discrete programming problems", Econometrica 28(3)), with the whole search tree kept,
 * and Gomory's fractional cutting planes for pure integer programs (Gomory, 1958, "Outline of an algorithm for integer
 * solutions to linear programs", Bull. AMS 64(5); presentation as in Wolsey, 1998, "Integer Programming", §8.6).
 *
 * Both minimise, as `scipy.optimize.milp`, and both solve their linear programs with the simplex method of
 * `./simplex`: branch and bound solves each node's relaxation from scratch, while the cutting-plane method keeps one
 * optimal tableau and adds a row and a column per cut.
 */

import { treeFromParents, type Tree } from 'aifn-compute/graph'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import type { Index, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import type { RunOptions } from '../options'
import { intTensor, matrix, readVector, vector } from './input'
import { parseLP, toOriginal, type LinearProgram, type ParsedLP, type StandardForm } from './lp'
import { basicSolution, pivotTableau, simplex, simplexSolve } from './simplex'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A mixed-integer linear program: a `LinearProgram` whose variables marked in `integrality` must take integer values,
 * as `scipy.optimize.milp` (1 = integer, 0 = continuous; default: every variable integer).
 */
export interface MixedIntegerProgram extends LinearProgram {
  /** For each of the $n$ variables, non-zero (1) when it must be an integer, 0 when continuous. Default all 1. */
  integrality?: VectorLike
}

/** How a branch-and-bound node ended. */
export type BranchNodeStatus =
  /** Not yet processed. */
  | 'open'
  /** Its relaxation had a fractional integer variable; two children were created. */
  | 'branched'
  /** Its relaxation's solution was integral: a candidate incumbent (a leaf). */
  | 'integral'
  /** Pruned: its relaxation is infeasible. */
  | 'infeasible'
  /** Pruned by bound: its relaxation's value (or its parent's) is no better than the incumbent. */
  | 'bound'
  /** Its relaxation is unbounded. */
  | 'unbounded'

/** One node of the branch-and-bound tree. */
export interface BranchNode {
  /** The node's id: its index in `nodes`, in order of creation (the root is 0). */
  id: number
  /** The parent's id, or $-1$ for the root. */
  parent: number
  /** The number of branchings from the root (0 for the root). */
  depth: number
  /** The node's lower bounds on the variables (the problem's, tightened by branching), length $n$. */
  lower: Tensor
  /** The node's upper bounds on the variables, length $n$. */
  upper: Tensor
  /**
   * The branching decision that created this node, or null for the root: the `variable` (0-based), its fractional
   * `value` in the parent's relaxation, and the `direction` (`down` for $x_j \le \lfloor v \rfloor$, `up` for
   * $x_j \ge \lceil v \rceil$).
   */
  branch: { variable: number; value: number; direction: 'down' | 'up' } | null
  /** How the node ended, or `open`. */
  status: BranchNodeStatus
  /** A lower bound on the node's objective: its LP relaxation value once solved, the parent's until then. */
  bound: number
  /** The relaxation's solution, or null if not solved or infeasible. */
  x: Tensor | null
  /** The step at which the node was processed, or $-1$ while open. */
  order: number
  /** The variable branched on at this node, or $-1$. */
  branchedOn: number
}

/**
 * Node selection. `depth-first` dives (down branch first), `best-bound` takes the lowest bound (the deeper node on
 * ties), `breadth-first` takes the oldest open node.
 */
export type NodeSelection = 'depth-first' | 'best-bound' | 'breadth-first'

/** Options for `branchAndBound`. */
export interface BranchAndBoundOptions {
  /** Node selection (default `best-bound`). */
  strategy?: NodeSelection
  /** Branching variable: the `most-fractional` (default) or the `first-fractional` integer variable. */
  branching?: 'most-fractional' | 'first-fractional'
  /** A value is integral when within this of an integer (default 1e-6). */
  integralityTolerance?: Scalar
}

/**
 * Overall status of a branch-and-bound or cutting-plane run: `running`, or `optimal`, `infeasible` or `unbounded`.
 */
export type IntegerProgramStatus = 'running' | 'optimal' | 'infeasible' | 'unbounded'

/**
 * One state of branch and bound: the tree so far. `t` counts processed nodes; `converged` at a proven optimum,
 * `terminated` when the problem is infeasible or unbounded.
 */
export interface BranchAndBoundState extends Status {
  /** Every node created so far, indexed by id. */
  nodes: readonly BranchNode[]
  /** Ids of open nodes, in the order the strategy keeps them. */
  open: readonly number[]
  /** The best integral solution found, or null. */
  incumbent: Tensor | null
  /** Its objective, Infinity when there is none. */
  incumbentValue: Scalar
  /**
   * The smallest bound over open nodes and the incumbent: the optimum lies between `bestBound` and `incumbentValue`.
   */
  bestBound: Scalar
  /**
   * `incumbentValue` minus `bestBound`: `Infinity` with no incumbent while nodes are open, and NaN once a run ends
   * with none.
   */
  gap: Scalar
  /** The node processed by the last step, or $-1$. */
  current: Index
  /** Whether the search goes on, or how it ended. */
  status: IntegerProgramStatus
  /** Simplex steps over all relaxations. */
  lpSteps: Size
  /** The problem as parsed. */
  problem: ParsedLP
  /** 1 for each integer variable, 0 for a continuous one; int32, length $n$. */
  integer: Tensor
  /** True when `status` is `optimal`. */
  converged: boolean
  /** True when `status` is `infeasible` or `unbounded`. */
  terminated: boolean
}

/**
 * Read which variables must be integers. Throws `ShapeError` when `integrality` does not have $n$ entries.
 *
 * @param problem The mixed-integer program.
 * @param n The number of variables.
 * @returns 1 for each integer variable (a non-zero entry, or every variable when `integrality` is absent), else 0.
 */
function readIntegrality(problem: MixedIntegerProgram, n: number): Int32Array {
  if (problem.integrality === undefined) return new Int32Array(n).fill(1)
  return Int32Array.from(readVector(problem.integrality, 'integrality', n), (v) => (v ? 1 : 0))
}

/**
 * The relaxation at a node: the problem with the node's bounds and no integrality.
 *
 * @param lp The parsed problem.
 * @param lower The node's lower bounds, length $n$.
 * @param upper The node's upper bounds, length $n$.
 * @returns The linear program to solve at the node.
 */
function relaxation(lp: ParsedLP, lower: ArrayLike<number>, upper: ArrayLike<number>): LinearProgram {
  return {
    c: vector(lp.c),
    A_ub: matrix(lp.Aub.a, lp.Aub.m, lp.n),
    b_ub: vector(lp.bub),
    A_eq: matrix(lp.Aeq.a, lp.Aeq.m, lp.n),
    b_eq: vector(lp.beq),
    bounds: Array.from({ length: lp.n }, (_, j) => [lower[j], upper[j]] as const),
  }
}

/**
 * How far a value is from the nearest integer.
 *
 * @param v The value.
 * @returns The distance, between 0 and $1/2$.
 */
const fractionality = (v: number) => Math.abs(v - Math.round(v))

/**
 * The Status flags of an integer-programming state from its outcome.
 *
 * @param status The outcome.
 * @returns `converged` (when `optimal`) and `terminated` (when `infeasible` or `unbounded`).
 */
const flags = (status: IntegerProgramStatus) => ({
  converged: status === 'optimal',
  terminated: status === 'infeasible' || status === 'unbounded',
})

/**
 * Complete a branch-and-bound state: the best bound over the open nodes and the incumbent, the gap, and the status
 * (`optimal` or `infeasible` once no node is open, with or without an incumbent; `running` before).
 *
 * @param s The state without its summary fields.
 * @param override A status to report instead (an unbounded relaxation ends the run as `unbounded`).
 * @returns The state.
 */
function summarise(
  s: Omit<BranchAndBoundState, 'bestBound' | 'gap' | 'status' | 'converged' | 'terminated'>,
  override?: IntegerProgramStatus,
): BranchAndBoundState {
  let bestBound = s.incumbentValue
  for (const id of s.open) bestBound = Math.min(bestBound, s.nodes[id].bound)
  const finished = s.open.length === 0
  const status = override ?? (finished ? (s.incumbent ? 'optimal' : 'infeasible') : 'running')
  return { ...s, bestBound, gap: s.incumbentValue - bestBound, status, ...flags(status) }
}

/**
 * Pick (and remove) the next open node: the last added for `depth-first`, the first for `breadth-first`, and for
 * `best-bound` the lowest bound, the deeper node on ties.
 *
 * @param open The ids of the open nodes, oldest first (not modified).
 * @param nodes Every node, indexed by id.
 * @param strategy The node selection.
 * @returns The chosen id, and the remaining open ids in order.
 */
function select(open: readonly number[], nodes: readonly BranchNode[], strategy: NodeSelection): [number, number[]] {
  const rest = [...open]
  if (strategy === 'depth-first') return [rest.pop()!, rest]
  if (strategy === 'breadth-first') return [rest.shift()!, rest]
  let k = 0
  for (let i = 1; i < rest.length; i++) {
    const a = nodes[rest[i]]
    const b = nodes[rest[k]]
    // Lowest bound; among equal bounds the deeper node, which is likelier to be integral.
    if (a.bound < b.bound - 1e-12 || (Math.abs(a.bound - b.bound) <= 1e-12 && a.depth > b.depth)) k = i
  }
  const [id] = rest.splice(k, 1)
  return [id, rest]
}

/**
 * Branch and bound (Land and Doig, 1960) for the mixed-integer linear program `problem`, as a traceable algorithm
 * with no start. Each step processes one node: it solves the node's LP relaxation by the simplex method and prunes the
 * node (`infeasible`, or `bound` when its value cannot beat the incumbent), accepts it (`integral`, perhaps a new
 * incumbent), or branches on a fractional variable $x_j = v$ into $x_j \le \lfloor v \rfloor$ (`down`) and
 * $x_j \ge \lceil v \rceil$ (`up`). The whole tree is kept in `nodes`. The run is done when no node is open; an
 * unbounded relaxation ends it as `unbounded`. Throws as `parseLP` does for an ill-formed problem.
 *
 * @param problem The mixed-integer program, in the form of `scipy.optimize.milp`.
 * @param options The node selection, the branching variable and the integrality tolerance.
 * @returns The algorithm. Its start is ignored; once the run has ended, a step returns the state unchanged.
 *
 * @example Watch the bounds close on an integer optimum
 * // Maximise 5x + 4y subject to 6x + 4y <= 24 and x + 2y <= 6, x, y >= 0 integer. The relaxation's optimum is
 * // fractional; branch and bound proves the integer one.
 * const problem = { c: [-5, -4], A_ub: [[6, 4], [1, 2]], b_ub: [24, 6] }
 * const tr = trace(branchAndBound(problem), {}, 50)
 * for (const s of tr.steps.slice(1))
 *   print('node', s.current, s.nodes[s.current].status, ' incumbent', s.incumbentValue, ' best bound', s.bestBound)
 * print('x =', tr.steps.at(-1).incumbent)
 */
export function branchAndBound(
  problem: MixedIntegerProgram,
  options: BranchAndBoundOptions = {},
): Algorithm<object, BranchAndBoundState> {
  const strategy = options.strategy ?? 'best-bound'
  const branching = options.branching ?? 'most-fractional'
  const integralityTolerance = options.integralityTolerance ?? 1e-6
  const lp = parseLP(problem)
  const integer = intTensor(readIntegrality(problem, lp.n))
  return {
    name: 'branch-and-bound',
    init: () => {
      const root: BranchNode = {
        id: 0,
        parent: -1,
        depth: 0,
        lower: vector(lp.lower),
        upper: vector(lp.upper),
        branch: null,
        status: 'open',
        bound: -Infinity,
        x: null,
        order: -1,
        branchedOn: -1,
      }
      return summarise({
        nodes: [root],
        open: [0],
        incumbent: null,
        incumbentValue: Infinity,
        current: -1,
        t: 0,
        lpSteps: 0,
        problem: lp,
        integer,
      })
    },
    step: (s) => {
      if (s.status !== 'running') return s
      const [id, open] = select(s.open, s.nodes, strategy)
      const nodes = [...s.nodes]
      const node = nodes[id]
      let { incumbent, incumbentValue, lpSteps } = s
      const cutoff = (v: number) => v >= incumbentValue - 1e-9 * (1 + Math.abs(incumbentValue))
      const finish = (patch: Partial<BranchNode>, extra: Partial<BranchAndBoundState> = {}) => {
        nodes[id] = { ...node, order: s.t, ...patch }
        return summarise(
          {
            ...s,
            nodes,
            open: extra.open ?? open,
            incumbent,
            incumbentValue,
            lpSteps,
            current: id,
            t: s.t + 1,
          },
          extra.status,
        )
      }
      // A node whose inherited bound already cannot beat the incumbent is pruned without solving its relaxation.
      if (cutoff(node.bound)) return finish({ status: 'bound' })
      const r = simplexSolve(relaxation(lp, node.lower.data, node.upper.data))
      lpSteps += r.steps
      if (r.status === 'unbounded') return finish({ status: 'unbounded' }, { status: 'unbounded', open: [] })
      if (r.status !== 'optimal') return finish({ status: 'infeasible' })
      const x = r.x.data
      if (cutoff(r.objective)) return finish({ status: 'bound', bound: r.objective, x: r.x })
      // Choose a fractional integer variable.
      let j = -1
      for (let k = 0; k < lp.n; k++) {
        if (!integer.data[k] || fractionality(x[k]) <= integralityTolerance) continue
        if (branching === 'first-fractional') {
          j = k
          break
        }
        if (j < 0 || fractionality(x[k]) > fractionality(x[j])) j = k
      }
      if (j < 0) {
        const snapped = Float64Array.from(x, (v, k) => (integer.data[k] ? Math.round(v) : v))
        incumbent = vector(snapped)
        incumbentValue = dense.dot(lp.c, snapped)
        return finish({ status: 'integral', bound: r.objective, x: r.x })
      }
      const v = x[j]
      const make = (direction: 'down' | 'up', childId: number): BranchNode => {
        const lower = Float64Array.from(node.lower.data)
        const upper = Float64Array.from(node.upper.data)
        if (direction === 'down') upper[j] = Math.floor(v)
        else lower[j] = Math.ceil(v)
        return {
          id: childId,
          parent: id,
          depth: node.depth + 1,
          lower: vector(lower),
          upper: vector(upper),
          branch: { variable: j, value: v, direction },
          status: 'open',
          bound: r.objective,
          x: null,
          order: -1,
          branchedOn: -1,
        }
      }
      const down = make('down', nodes.length)
      const up = make('up', nodes.length + 1)
      nodes.push(down, up)
      // Depth-first pops from the end, so push `up` first to explore the down branch first.
      const children = strategy === 'depth-first' ? [up.id, down.id] : [down.id, up.id]
      return finish({ status: 'branched', bound: r.objective, x: r.x, branchedOn: j }, { open: [...open, ...children] })
    },
  }
}

/** Node data of a branch-and-bound search tree (see `branchAndBoundTree`). */
export interface SearchTreeNodeData {
  /** How the node ended, or `open`. */
  status: BranchNodeStatus
  /** Why the node was pruned: its relaxation is `infeasible`, or its `bound` cannot beat the incumbent; else null. */
  prunedBy: 'infeasible' | 'bound' | null
  /** Its relaxation's value, or the parent's while open (a lower bound on the node's objective). */
  bound: number
  /** The relaxation's solution, or null if not solved or infeasible. */
  x: number[] | null
  /** The step at which the node was processed, $-1$ while open. */
  order: number
  /** The variable branched on at this node, or $-1$. */
  branchedOn: number
}

/**
 * Edge data of a branch-and-bound search tree: the branching decision (as in `BranchNode.branch`: the 0-based
 * `variable`, its fractional `value` and the `direction`). The tree also gives each edge a TeX `label` such as
 * `$x_{1} \le 2$`, with the variable numbered from 1.
 */
export interface SearchTreeEdgeData {
  /** The variable branched on, 0-based. */
  variable: number
  /** Its fractional value in the parent's relaxation. */
  value: number
  /** `down` for the branch $x_j \le \lfloor v \rfloor$, `up` for $x_j \ge \lceil v \rceil$. */
  direction: 'down' | 'up'
}

/**
 * The branch-and-bound search tree as an `aifn-compute/graph` `Tree` (binary: the down branch
 * $x_j \le \lfloor v \rfloor$ is slot 0, the up branch $x_j \ge \lceil v \rceil$ slot 1). Node ids are the `BranchNode`
 * ids (the root is 0); nodes carry their bound, status and pruning reason, edges the branching decision with a TeX
 * label. Works on any state's `nodes`, so a figure can draw the tree as it grows.
 *
 * @param nodes The nodes of a branch-and-bound state (or `MixedIntegerResult.tree`), indexed by id.
 * @returns The tree; its edges are indexed by the child's id (null for the root).
 *
 * @example The tree of a small integer program
 * const r = milp({ c: [-5, -4], A_ub: [[6, 4], [1, 2]], b_ub: [24, 6] })
 * const tree = branchAndBoundTree(r.tree)
 * print('status =', tree.nodes.map((n) => n.status))
 * print('bound =', tree.nodes.map((n) => n.bound))
 * print('branch =', tree.edges.map((e) => (e ? e.label : '-')))
 */
export function branchAndBoundTree(nodes: readonly BranchNode[]): Tree<SearchTreeNodeData, SearchTreeEdgeData> {
  const tree = treeFromParents<SearchTreeNodeData, SearchTreeEdgeData>(
    nodes.map((n) => n.parent),
    {
      data: (i) => {
        const n = nodes[i]
        return {
          status: n.status,
          prunedBy: n.status === 'infeasible' || n.status === 'bound' ? n.status : null,
          bound: n.bound,
          x: n.x ? Array.from(n.x.data) : null,
          order: n.order,
          branchedOn: n.branchedOn,
        }
      },
      edge: (c) => {
        const b = nodes[c].branch!
        const rhs = b.direction === 'down' ? Math.floor(b.value) : Math.ceil(b.value)
        return { ...b, label: `$x_{${b.variable + 1}} ${b.direction === 'down' ? '\\le' : '\\ge'} ${rhs}$` }
      },
    },
  )
  for (const n of tree.nodes) if (n.parent !== null) n.slot = nodes[n.id].branch!.direction === 'down' ? 0 : 1
  return { ...tree, arity: 2 }
}

/** The result of `milp`. */
export interface MixedIntegerResult {
  /** How the search ended: `optimal`, `infeasible`, `unbounded`, or `limit` when it stopped at `maxSteps`. */
  status: Exclude<IntegerProgramStatus, 'running'> | 'limit'
  /** The best integral solution (NaN when none). */
  x: Tensor
  /** The objective of `x`: `-Infinity` when unbounded, NaN when there is no integral solution. */
  objective: Scalar
  /** The lower bound proved; equal to `objective` at an optimum. */
  bound: Scalar
  /** Nodes processed (the steps of `branchAndBound`). */
  nodes: Size
  /** The search tree, as processed. */
  tree: readonly BranchNode[]
  /** The same search tree as an `aifn-compute/graph` `Tree` (see `branchAndBoundTree`). */
  searchTree: Tree<SearchTreeNodeData, SearchTreeEdgeData>
}

/**
 * Solve a mixed-integer linear program by branch and bound (as `scipy.optimize.milp`, minimising; negate `c` to
 * maximise), processing at most `maxSteps` nodes (default 100 000). The result carries the whole search tree. Throws as
 * `parseLP` does for an ill-formed problem.
 *
 * @param problem The mixed-integer program, in the form of `scipy.optimize.milp`.
 * @param options The options of `branchAndBound`, and `maxSteps`, the most nodes processed.
 * @returns The best integral solution, its objective, the bound proved and the search tree.
 *
 * @example An integer optimum is not the rounded LP optimum
 * // Maximise 5x + 4y subject to 6x + 4y <= 24 and x + 2y <= 6, x, y >= 0.
 * const problem = { c: [-5, -4], A_ub: [[6, 4], [1, 2]], b_ub: [24, 6] }
 * print('LP relaxation:', linprog(problem).x)
 * const r = milp(problem)
 * print('integer optimum:', r.x, ' objective =', r.objective, ' nodes =', r.nodes)
 *
 * @example Only some variables integer
 * const r = milp({ c: [-5, -4], A_ub: [[6, 4], [1, 2]], b_ub: [24, 6], integrality: [1, 0] })
 * print('x integer, y continuous:', r.x, ' objective =', r.objective)
 */
export function milp(
  problem: MixedIntegerProgram,
  options: BranchAndBoundOptions & Pick<RunOptions, 'maxSteps'> = {},
): MixedIntegerResult {
  const s = run(branchAndBound(problem, options), {}, options.maxSteps ?? 100_000)
  const n = s.problem.n
  return {
    status: s.status === 'running' ? 'limit' : s.status,
    x: s.incumbent ?? vector(new Float64Array(n).fill(NaN)),
    objective: s.incumbent ? s.incumbentValue : s.status === 'unbounded' ? -Infinity : NaN,
    bound: s.bestBound,
    nodes: s.t,
    tree: s.nodes,
    searchTree: branchAndBoundTree(s.nodes),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Gomory fractional cuts.

/** A cutting plane in the original variables: $\avec^\top\xvec \le b$. */
export interface Cut {
  /** The coefficients $\avec$, length $n$. */
  a: Tensor
  /** The right-hand side $b$. */
  b: Scalar
}

/** Options for `gomory`. */
export interface GomoryOptions {
  /** A value is integral when within this of an integer (default 1e-9). */
  tolerance?: Scalar
}

/**
 * One state of the cutting-plane method: the optimal tableau of the relaxation with the cuts so far. `t` counts the
 * cuts added; `converged` when the relaxation's optimum is integral, `terminated` when infeasible or unbounded.
 */
export interface GomoryState extends Status {
  /**
   * The tableau, $(m + 1) \times (k + 1)$, in the layout of `SimplexState.tableau`; each cut adds a row and a column.
   */
  tableau: Tensor
  /** The basic column of each constraint row, int32, length $m$. */
  basis: Tensor
  /** Column labels; cut slacks are `g1`, `g2`, and so on. */
  labels: readonly string[]
  /** The relaxation's optimum in the original variables. */
  x: Tensor
  /** $\cvec^\top\xvec$ at that optimum. */
  objective: Scalar
  /** The cuts added so far, as $\avec^\top\xvec \le b$ in the original variables. */
  cuts: readonly Cut[]
  /** The tableau row the last cut was read from, or $-1$. */
  sourceRow: Index
  /** Dual simplex pivots made by the last step to restore feasibility. */
  dualPivots: Size
  /** Whether cuts go on, or how the method ended. */
  status: IntegerProgramStatus
  /**
   * Each tableau column as an affine function of $\xvec$: row $q$ is $(\gvec_q, h_q)$, with column $q$ equal to
   * $\gvec_q^\top\xvec + h_q$; shape `[k, n + 1]`. It is how cuts written in tableau columns are drawn in
   * $\xvec$-space.
   */
  affine: Tensor
  /** The standard form of the relaxation. */
  standard: StandardForm
  /** True when `status` is `optimal`. */
  converged: boolean
  /** True when `status` is `infeasible` or `unbounded`. */
  terminated: boolean
}

/**
 * The fractional part of a value, rounded to 0 when within `tol` of an integer.
 *
 * @param v The value.
 * @param tol The integrality tolerance.
 * @returns $v - \lfloor v \rfloor$, or 0 when that is below `tol` or above $1 -$ `tol`.
 */
const frac = (v: number, tol: number) => {
  const f = v - Math.floor(v)
  return f < tol || 1 - f < tol ? 0 : f
}

/**
 * Throw `DomainError` unless the cutting-plane method applies: integer constraint data and bounds (so that the slack
 * variables are integer too), and no free variable.
 *
 * @param lp The parsed problem.
 */
function checkIntegerData(lp: ParsedLP): void {
  const all = [lp.Aub.a, lp.bub, lp.Aeq.a, lp.beq]
  for (const a of all)
    for (let k = 0; k < a.length; k++)
      if (!Number.isInteger(a[k]))
        throw new DomainError('gomory', 'gomory: constraint data must be integers so that slacks are integer')
  for (let j = 0; j < lp.n; j++) {
    const lo = lp.lower[j]
    const hi = lp.upper[j]
    if (!Number.isFinite(lo) && !Number.isFinite(hi))
      throw new DomainError('gomory', 'gomory: free variables are not supported')
    if ((Number.isFinite(lo) && !Number.isInteger(lo)) || (Number.isFinite(hi) && !Number.isInteger(hi)))
      throw new DomainError('gomory', 'gomory: bounds must be integers')
  }
}

/**
 * The affine map from $\xvec$ to each standard-form column (no free variables): $x_j - l_j$ or $u_j - x_j$ for a
 * variable's column, $b_i - \avec_i^\top\xvec$ for the slack of a $\le$ row, and $u_j - x_j$ for the slack of an upper
 * bound.
 *
 * @param sf The standard form, with no free variables.
 * @returns Row-major, $N \times (n + 1)$: row $q$ holds the coefficients of column $q$ on $\xvec$, then its constant.
 */
function affineColumns(sf: StandardForm): Float64Array {
  const { lp, N } = sf
  const n = lp.n
  const w = n + 1
  const G = new Float64Array(N * w)
  let boundSlack = n + lp.Aub.m
  sf.variables.forEach((v, j) => {
    if (v.kind === 'lower') {
      G[v.column * w + j] = 1
      G[v.column * w + n] = -lp.lower[j]
      if (v.boundRow >= 0) {
        G[boundSlack * w + j] = -1
        G[boundSlack * w + n] = lp.upper[j]
        boundSlack++
      }
    } else if (v.kind === 'upper') {
      G[v.column * w + j] = -1
      G[v.column * w + n] = lp.upper[j]
    }
  })
  for (let i = 0; i < lp.Aub.m; i++) {
    const col = n + i
    for (let j = 0; j < n; j++) G[col * w + j] = -lp.Aub.a[i * n + j]
    G[col * w + n] = lp.bub[i]
  }
  return G
}

/**
 * Assemble a cutting-plane state from its tableau: the basic solution in the original variables and, unless the
 * caller gives one, the status (`optimal` when every basic value is integral, else `running`).
 *
 * @param sf The standard form of the relaxation.
 * @param tol The integrality tolerance.
 * @param t The tableau, row-major, $(m + 1) \times w$; copied into the state.
 * @param m The number of constraint rows.
 * @param w The row width (columns plus the right-hand side).
 * @param basis The basic column of each constraint row.
 * @param labels The label of each column.
 * @param affine Each column as an affine function of $\xvec$, row-major, $(w - 1) \times (n + 1)$.
 * @param extra The fields the caller sets: the cut count `t`, `cuts`, `sourceRow`, `dualPivots`, and optionally a
 *   `status` that overrides the integrality test.
 * @returns The state.
 */
function gomoryState(
  sf: StandardForm,
  tol: Scalar,
  t: Float64Array,
  m: number,
  w: number,
  basis: Int32Array,
  labels: readonly string[],
  affine: Float64Array,
  extra: Pick<GomoryState, 't' | 'cuts' | 'sourceRow' | 'dualPivots'> & { status?: IntegerProgramStatus },
): GomoryState {
  const z = basicSolution(t, m, w, basis)
  const x = toOriginal(sf, z)
  let status = extra.status
  if (!status) {
    status = 'optimal'
    for (let r = 0; r < m; r++) if (frac(t[r * w + w - 1], tol) > 0) status = 'running'
  }
  return {
    ...extra,
    standard: sf,
    status,
    ...flags(status),
    tableau: matrix(t, m + 1, w),
    basis: intTensor(basis),
    labels,
    x: vector(x),
    objective: dense.dot(sf.lp.c, x),
    affine: matrix(affine, w - 1, sf.lp.n + 1),
  }
}

/**
 * Gomory's fractional cutting-plane method (Gomory, 1958) for the pure integer program `problem` (every variable
 * integer), as a traceable algorithm with no start. The initial state is the LP relaxation's optimal simplex tableau.
 * Each step reads the row whose basic variable has the largest fractional part $f_0$, adds the cut
 * $\sum_j \operatorname{frac}(\bar a_j) z_j \ge f_0$ over the non-basic columns (every integer point satisfies it;
 * the current vertex does not), and restores feasibility by dual simplex pivots. The run is done when the relaxation's
 * optimum is integral (`optimal`) or a cut makes it `infeasible`. The data must be integers so that the slack
 * variables are integer too: `DomainError` is thrown, when the algorithm is made, for fractional data or bounds or a
 * free variable.
 *
 * @param problem The integer program, as a linear program whose variables must all be integers.
 * @param options The integrality tolerance.
 * @returns The algorithm. Its start is ignored; once the run has ended, a step returns the state unchanged.
 *
 * @example Cut a fractional vertex down to the integer optimum
 * // Maximise 5x + 4y subject to 6x + 4y <= 24 and x + 2y <= 6, x, y >= 0 integer.
 * const tr = trace(gomory({ c: [-5, -4], A_ub: [[6, 4], [1, 2]], b_ub: [24, 6] }), {}, 20)
 * print('x =', tr.steps.map((s) => Array.from(s.x.data)))
 * const s = tr.steps.at(-1)
 * print('status =', s.status, ' objective =', s.objective)
 * print('cuts a =', s.cuts.map((c) => Array.from(c.a.data)), ' b =', s.cuts.map((c) => c.b))
 */
export function gomory(problem: LinearProgram, options: GomoryOptions = {}): Algorithm<object, GomoryState> {
  const tol = options.tolerance ?? 1e-9
  checkIntegerData(parseLP(problem))
  return {
    name: 'gomory-cuts',
    init: () => {
      const lpState = run(simplex(problem), {}, 10_000)
      const sf = lpState.standard
      const m = lpState.basis.shape[0]
      const w = lpState.tableau.shape[1]
      const extra = { t: 0, cuts: [], sourceRow: -1, dualPivots: 0 }
      const status: IntegerProgramStatus | undefined =
        lpState.status === 'optimal' ? undefined : lpState.status === 'unbounded' ? 'unbounded' : 'infeasible'
      return gomoryState(
        sf,
        tol,
        Float64Array.from(lpState.tableau.data),
        m,
        w,
        Int32Array.from(lpState.basis.data),
        lpState.labels,
        affineColumns(sf),
        { ...extra, status },
      )
    },
    step: (s) => {
      if (s.status !== 'running') return s
      const m = s.basis.shape[0]
      const w = s.tableau.shape[1]
      const k = w - 1
      const t = s.tableau.data
      const n = s.standard.lp.n
      // The source row: the basic variable with the largest fractional part.
      let row = -1
      let f0 = 0
      for (let r = 0; r < m; r++) {
        const f = frac(t[r * w + k], tol)
        if (f > f0) {
          f0 = f
          row = r
        }
      }
      // New tableau with one more row (the cut) and one more column (its slack g ≥ 0).
      const m2 = m + 1
      const w2 = w + 1
      const t2 = new Float64Array((m2 + 1) * w2)
      const copyRow = (from: number, to: number) => {
        for (let j = 0; j < k; j++) t2[to * w2 + j] = t[from * w + j]
        t2[to * w2 + w2 - 1] = t[from * w + k]
      }
      for (let r = 0; r < m; r++) copyRow(r, r)
      copyRow(m, m2) // objective row
      const basic = new Set(s.basis.data)
      const fj = new Float64Array(k)
      for (let j = 0; j < k; j++) if (!basic.has(j)) fj[j] = frac(t[row * w + j], tol)
      // Cut row: −Σ fⱼ zⱼ + g = −f₀.
      for (let j = 0; j < k; j++) t2[m * w2 + j] = -fj[j]
      t2[m * w2 + k] = 1
      t2[m * w2 + w2 - 1] = -f0
      const basis = Int32Array.from([...s.basis.data, k])
      // The cut slack as an affine function of x: g = Σ fⱼ zⱼ − f₀.
      const aw = n + 1
      const affine = new Float64Array((k + 1) * aw)
      affine.set(s.affine.data)
      for (let j = 0; j < k; j++) {
        if (fj[j] === 0) continue
        for (let q = 0; q < aw; q++) affine[k * aw + q] += fj[j] * s.affine.data[j * aw + q]
      }
      affine[k * aw + n] -= f0
      const cut: Cut = {
        a: vector(Array.from({ length: n }, (_, q) => -affine[k * aw + q])),
        b: affine[k * aw + n],
      }
      // Dual simplex: while a basic variable is negative, pivot it out keeping the reduced costs non-negative.
      let pivots = 0
      for (;;) {
        let r = -1
        for (let i = 0; i < m2; i++)
          if (t2[i * w2 + w2 - 1] < -tol && (r < 0 || t2[i * w2 + w2 - 1] < t2[r * w2 + w2 - 1])) r = i
        if (r < 0) break
        let col = -1
        let best = Infinity
        for (let j = 0; j < w2 - 1; j++) {
          const a = t2[r * w2 + j]
          if (a >= -tol) continue
          const ratio = Math.max(0, t2[m2 * w2 + j]) / -a
          if (ratio < best - 1e-12) {
            best = ratio
            col = j
          }
        }
        if (col < 0 || pivots > 1000)
          return gomoryState(s.standard, tol, t2, m2, w2, basis, [...s.labels, `g${s.cuts.length + 1}`], affine, {
            t: s.t + 1,
            cuts: [...s.cuts, cut],
            sourceRow: row,
            dualPivots: pivots,
            status: 'infeasible',
          })
        pivotTableau(t2, m2 + 1, w2, r, col)
        basis[r] = col
        pivots++
      }
      return gomoryState(s.standard, tol, t2, m2, w2, basis, [...s.labels, `g${s.cuts.length + 1}`], affine, {
        t: s.t + 1,
        cuts: [...s.cuts, cut],
        sourceRow: row,
        dualPivots: pivots,
      })
    },
  }
}
