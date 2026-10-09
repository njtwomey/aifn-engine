/**
 * Finite Markov chains on states $0, \dots, n - 1$ with a row-stochastic transition matrix $\Pmat$,
 * $P_{ij} = \pr(X_{t+1} = j \mid X_t = i)$: the classification of states, stationary distributions, absorption and
 * hitting by first-step analysis, convergence to stationarity, and simulation.
 *
 * The notation and results are those of Norris (1997), "Markov Chains", and Levin, Peres and Wilmer (2017), "Markov
 * Chains and Mixing Times". Distributions are row vectors, so the stationary distribution solves
 * $\pivec\Pmat = \pivec$ and the distribution after $t$ steps is $\pvec_0\Pmat^t$. Matrices are small and dense;
 * every quantity other than a simulated path is computed exactly (linear solves and matrix powers), never by
 * simulation. Every function validates the matrix first (square, entries non-negative, each row summing to 1 within
 * $10^{-9} n$) and throws `DomainError` otherwise, as it does for an unknown state or an invalid step count.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { categorical, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, imagPart, realPart, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { fromEdges } from 'aifn-compute/graph'
import { stronglyConnectedComponents } from 'aifn-compute/graph/traversal'
import { eig, solveDense } from 'aifn-compute/numerics/linalg'
import { totalVariation } from 'aifn-compute/probability/information'

/** A row-major `Float64Array` of the dense helpers. */
type F64 = dense.F64

/** Rows must sum to one within this, times the number of states. */
const ROW_TOLERANCE = 1e-9

/**
 * The validated matrix of a chain as row-major data. Throws `DomainError` when it is not square, has no states, has an
 * entry that is negative or NaN, or has a row whose sum differs from 1 by more than $10^{-9} n$.
 *
 * @param P The transition matrix $\Pmat$, a tensor or nested array.
 * @param where The caller's name, for error messages.
 * @returns `p`, the entries of $\Pmat$ row-major ($n^2$ values), and `n`, the number of states.
 */
function readChain(P: MatrixLike, where: string): { p: F64; n: number } {
  const { data, m, n } = dense.toMatrixF64(P, where)
  if (m !== n) throw new DomainError(where, `${where}: the transition matrix must be square, got ${m}×${n}`)
  if (n === 0) throw new DomainError(where, `${where}: the chain has no states`)
  for (let i = 0; i < n; i++) {
    let row = 0
    for (let j = 0; j < n; j++) {
      const v = data[i * n + j]
      if (!(v >= 0)) throw new DomainError(where, `${where}: P[${i}, ${j}] = ${v} is not a probability`)
      row += v
    }
    if (Math.abs(row - 1) > ROW_TOLERANCE * n) throw new DomainError(where, `${where}: row ${i} sums to ${row}, not 1`)
  }
  return { p: data, n }
}

/**
 * An initial distribution as a fresh array. A state index becomes the point mass on that state; a vector must have $n$
 * non-negative entries summing to 1 within $10^{-9} n$. Throws `DomainError` otherwise.
 *
 * @param p0 A state (an integer in $[0, n)$) or a distribution over the $n$ states.
 * @param n The number of states.
 * @param where The caller's name, for error messages.
 * @returns The distribution, $n$ values.
 */
function readDistribution(p0: VectorLike | number, n: number, where: string): F64 {
  if (typeof p0 === 'number') {
    if (!Number.isInteger(p0) || p0 < 0 || p0 >= n) throw new DomainError(where, `${where}: no state ${p0}`)
    const e = new Float64Array(n)
    e[p0] = 1
    return e
  }
  const v = dense.toF64(p0, where)
  if (v.length !== n) throw new DomainError(where, `${where}: ${v.length} probabilities for ${n} states`)
  const total = v.reduce((a, b) => a + b, 0)
  if (v.some((x) => !(x >= 0)) || Math.abs(total - 1) > 1e-9 * n)
    throw new DomainError(where, `${where}: the initial distribution must be non-negative and sum to 1`)
  return Float64Array.from(v)
}

/**
 * A set of states, checked and normalised: duplicates removed and sorted ascending. Throws `DomainError` when the set
 * is empty or holds something other than an integer in $[0, n)$.
 *
 * @param states The states, an array or a vector.
 * @param n The number of states of the chain.
 * @param where The caller's name, for error messages.
 * @returns The distinct states, in ascending order.
 */
function readStates(states: readonly number[] | VectorLike, n: number, where: string): number[] {
  const list = Array.from(dense.toF64(states as VectorLike, where))
  if (list.length === 0) throw new DomainError(where, `${where}: the target set is empty`)
  for (const s of list)
    if (!Number.isInteger(s) || s < 0 || s >= n) throw new DomainError(where, `${where}: no state ${s}`)
  return [...new Set(list)].sort((a, b) => a - b)
}

/**
 * One step of a distribution, $\pvec \mapsto \pvec\Pmat$ for a row vector $\pvec$.
 *
 * @param p The distribution $\pvec$, $n$ values; not modified.
 * @param P The transition matrix, row-major, $n^2$ values.
 * @param n The number of states.
 * @returns $\pvec\Pmat$, a new array of $n$ values.
 */
const stepRow = (p: ArrayLike<number>, P: F64, n: number): F64 => dense.matTVec(P, p, n, n)

/**
 * The total-variation distance $\lVert \pvec - \qvec \rVert_{TV}$ for one distribution $\pvec$, or for each row of
 * a matrix of distributions, against $\qvec$.
 *
 * @param p One distribution of $n$ values, or (with `rows`) a row-major matrix of `rows` distributions.
 * @param q The reference distribution, $n$ values.
 * @param rows The number of rows of `p`; left out, `p` is one distribution.
 * @returns The distances: one value, or one per row.
 */
function tv(p: F64, q: F64, rows?: number): F64 {
  const d = totalVariation(rows === undefined ? dense.vec(p) : dense.mat(p, rows, q.length), dense.vec(q))
  return typeof d === 'number' ? Float64Array.of(d) : dense.data(d as Tensor)
}

/**
 * The transition graph: a directed edge $i \to j$ wherever $P_{ij} > 0$.
 *
 * @param P The transition matrix, row-major, $n^2$ values.
 * @param n The number of states (the graph's nodes).
 * @returns The directed graph.
 */
function graphOf(P: F64, n: number) {
  const edges: [number, number][] = []
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (P[i * n + j] > 0) edges.push([i, j])
  return fromEdges(n, edges, { directed: true })
}

/**
 * The states from which some state of `targets` can be reached (in zero or more steps), as a mask, by a backward search
 * along positive transitions. Paths may not pass through a `blocked` state (one where the chain is stopped), though a
 * blocked state is itself marked when it reaches.
 *
 * @param P The transition matrix, row-major, $n^2$ values.
 * @param n The number of states.
 * @param targets The target states; each is marked.
 * @param blocked A mask of $n$ entries, nonzero where the chain is stopped; left out, no state is blocked.
 * @returns A mask of $n$ entries, 1 where a target can be reached.
 */
function canReach(P: F64, n: number, targets: readonly number[], blocked?: Uint8Array): Uint8Array {
  const mask = new Uint8Array(n)
  const queue = [...targets]
  for (const t of targets) mask[t] = 1
  while (queue.length > 0) {
    const j = queue.pop()!
    for (let i = 0; i < n; i++)
      if (!mask[i] && P[i * n + j] > 0) {
        mask[i] = 1
        if (!blocked?.[i]) queue.push(i)
      }
  }
  return mask
}

/**
 * The absorbing states: those with no positive transition to another state (so $P_{ii} = 1$ within the row
 * tolerance).
 *
 * @param P The transition matrix, row-major, $n^2$ values.
 * @param n The number of states.
 * @returns The absorbing states, in ascending order.
 */
function absorbingStates(P: F64, n: number): number[] {
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let other = false
    for (let j = 0; j < n && !other; j++) if (j !== i && P[i * n + j] > 0) other = true
    if (!other) out.push(i)
  }
  return out
}

// ── Construction ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The transition matrix of a chain as a float64 tensor, after checking that it is square, non-negative and
 * row-stochastic (each row sums to 1 within $10^{-9} n$). Throws a `DomainError` naming the offending entry or row
 * otherwise.
 *
 * @param P The candidate transition matrix ($n \times n$), a tensor or nested array.
 * @returns A copy of $\Pmat$ as an $n \times n$ float64 tensor.
 *
 * @example A valid chain, and a row that does not sum to 1
 * print('P =', transitionMatrix([[0.9, 0.1], [0.5, 0.5]]))
 * try {
 *   transitionMatrix([[0.9, 0.2], [0.5, 0.5]])
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
export function transitionMatrix(P: MatrixLike): Tensor {
  const { p, n } = readChain(P, 'transitionMatrix')
  return dense.mat(Float64Array.from(p), n, n)
}

/**
 * The $k$-step transition matrix $\Pmat^k$ ($\Pmat^0 = \Imat$), by repeated squaring: entry $(i, j)$ is the
 * probability of being in $j$ after $k$ steps from $i$.
 *
 * @param P The transition matrix ($n \times n$).
 * @param steps The number of steps $k$, an integer $\ge 0$ (else `DomainError`).
 * @returns $\Pmat^k$, an $n \times n$ float64 tensor.
 *
 * @example The rows approach the stationary distribution $(5/6, 1/6)$
 * const P = [[0.9, 0.1], [0.5, 0.5]]
 * print('P^2 =', nStepTransition(P, 2))
 * print('P^20 =', nStepTransition(P, 20))
 */
export function nStepTransition(P: MatrixLike, steps: number): Tensor {
  const where = 'nStepTransition'
  const { p, n } = readChain(P, where)
  if (!Number.isInteger(steps) || steps < 0) throw new DomainError(where, `${where}: steps must be an integer ≥ 0`)
  let result = dense.identity(n)
  let base = Float64Array.from(p)
  for (let k = steps; k > 0; k = Math.floor(k / 2)) {
    if (k % 2 === 1) result = dense.matMul(result, base, n, n, n)
    if (k > 1) base = dense.matMul(base, base, n, n, n)
  }
  return dense.mat(result, n, n)
}

/**
 * The distribution $\pvec_0\Pmat^t$ of $X_t$ for an initial distribution $\pvec_0$ (or a start state), by $t$
 * vector-matrix products.
 *
 * @param P The transition matrix ($n \times n$).
 * @param start The initial distribution $\pvec_0$ ($n$ probabilities summing to 1), or a state, which stands for the
 *   point mass on it.
 * @param steps The number of steps $t$, an integer $\ge 0$ (else `DomainError`).
 * @returns $\pvec_0\Pmat^t$, $n$ values.
 *
 * @example From state 1 of a two-state chain
 * const P = [[0.9, 0.1], [0.5, 0.5]]
 * print('t = 0:', distributionAfter(P, 1, 0))
 * print('t = 1:', distributionAfter(P, 1, 1))
 * print('t = 10:', distributionAfter(P, 1, 10))
 * print('from (0.5, 0.5), t = 1:', distributionAfter(P, [0.5, 0.5], 1))
 */
export function distributionAfter(P: MatrixLike, start: VectorLike | number, steps: number): Tensor {
  const where = 'distributionAfter'
  const { p, n } = readChain(P, where)
  if (!Number.isInteger(steps) || steps < 0) throw new DomainError(where, `${where}: steps must be an integer ≥ 0`)
  let d = readDistribution(start, n, where)
  for (let t = 0; t < steps; t++) d = stepRow(d, p, n)
  return dense.vec(d)
}

// ── Classification ───────────────────────────────────────────────────────────────────────────────────────────────────

/** The result of `classifyStates`. */
export interface ChainClasses {
  /** The communicating classes (each a sorted list of states), closed classes first, then by smallest state. */
  classes: number[][]
  /**
   * Whether each class is closed (no positive transition leaves it). In a finite chain, closed $\Leftrightarrow$
   * recurrent.
   */
  closed: boolean[]
  /**
   * The period of each class: the gcd of the lengths of its cycles (1 = aperiodic; 0 for a single state with no
   * self-loop, which has no cycle).
   */
  periods: number[]
  /** The class index of each state, an int32 tensor of $n$ values indexing `classes`. */
  classOf: Tensor
  /** The states in closed classes (recurrent), in ascending order. */
  recurrent: number[]
  /** The states in classes that are not closed (transient), in ascending order. */
  transient: number[]
  /** The states with no positive transition to another state ($P_{ii} = 1$). */
  absorbing: number[]
  /** One class: every state reaches every other. */
  irreducible: boolean
  /** Every closed class has period 1. */
  aperiodic: boolean
}

/**
 * The greatest common divisor, by Euclid's algorithm; $\gcd(0, b) = \lvert b \rvert$.
 *
 * @param a An integer.
 * @param b An integer.
 * @returns The non-negative gcd of `a` and `b`.
 */
const gcd = (a: number, b: number): number => (b === 0 ? Math.abs(a) : gcd(b, a % b))

/**
 * The communicating classes of a chain (the strongly connected components of its transition graph, Tarjan's
 * algorithm), which of them are closed (recurrent) and which transient, and the period of each: the gcd over edges
 * $u \to w$ inside the class of $\ell(u) + 1 - \ell(w)$, with levels $\ell$ from a breadth-first search inside the
 * class. A class is closed when no positive transition leaves it, however small, so a tiny leak makes it transient.
 *
 * @param P The transition matrix ($n \times n$).
 * @returns The classes and the properties of each, the recurrent, transient and absorbing states, and whether the
 *   chain is irreducible and aperiodic.
 *
 * @example A transient state leading into a closed class of period 2
 * const c = classifyStates([[0.5, 0.5, 0], [0, 0, 1], [0, 1, 0]])
 * print('classes:', c.classes, 'closed:', c.closed, 'periods:', c.periods)
 * print('recurrent:', c.recurrent, 'transient:', c.transient)
 * print('irreducible:', c.irreducible, 'aperiodic:', c.aperiodic)
 *
 * @example Two absorbing states
 * const c = classifyStates([[1, 0, 0], [0.5, 0, 0.5], [0, 0, 1]])
 * print('classes:', c.classes, 'absorbing:', c.absorbing)
 */
export function classifyStates(P: MatrixLike): ChainClasses {
  const where = 'classifyStates'
  const { p, n } = readChain(P, where)
  const components = stronglyConnectedComponents(graphOf(p, n))
  let groups = components.members.map((m) => Array.from(toFlat(m)).sort((a, b) => a - b))
  // Closed: no positive transition leaves the class (structure, not a sum within the row tolerance, so a tiny leak
  // still makes the class transient).
  const isClosed = (members: number[]) => {
    const inside = new Set(members)
    return members.every((i) => {
      for (let j = 0; j < n; j++) if (p[i * n + j] > 0 && !inside.has(j)) return false
      return true
    })
  }
  groups = groups.sort((a, b) => Number(isClosed(b)) - Number(isClosed(a)) || a[0] - b[0])
  const closed = groups.map(isClosed)
  const classOf = new Int32Array(n)
  groups.forEach((g, c) => g.forEach((i) => (classOf[i] = c)))
  const periods = groups.map((members) => {
    const inClass = new Set(members)
    const level = new Map<number, number>([[members[0], 0]])
    const queue = [members[0]]
    while (queue.length > 0) {
      const u = queue.shift()!
      for (const w of members)
        if (p[u * n + w] > 0 && !level.has(w)) {
          level.set(w, level.get(u)! + 1)
          queue.push(w)
        }
    }
    let d = 0
    for (const u of members)
      for (const w of members) if (inClass.has(w) && p[u * n + w] > 0) d = gcd(d, level.get(u)! + 1 - level.get(w)!)
    // A single state without a self-loop has no cycle: its period is undefined, reported as 0.
    return d
  })
  const recurrent: number[] = []
  const transient: number[] = []
  for (let i = 0; i < n; i++) (closed[classOf[i]] ? recurrent : transient).push(i)
  const absorbing = absorbingStates(p, n)
  return {
    classes: groups,
    closed,
    periods,
    classOf: fromData(classOf, [n]),
    recurrent,
    transient,
    absorbing,
    irreducible: groups.length === 1,
    aperiodic: groups.every((_, c) => !closed[c] || periods[c] === 1),
  }
}

// ── Stationary distributions ─────────────────────────────────────────────────────────────────────────────────────────

/**
 * Solve $\pivec\Pmat = \pivec$, $\sum_i \pi_i = 1$ on the states `members` of a closed class ($\pivec$ is zero
 * elsewhere), with the last balance equation replaced by the normalisation. Rounding below 0 is clipped and the
 * result renormalised. Throws `DomainError` when the system is singular.
 *
 * @param p The transition matrix, row-major, $n^2$ values.
 * @param n The number of states.
 * @param members The states of the closed class.
 * @returns $\pivec$ over all $n$ states.
 */
function stationaryOn(p: F64, n: number, members: number[]): F64 {
  const k = members.length
  // Equations j: Σᵢ πᵢ P_ij − π_j = 0 (one is redundant, since they sum to zero); the last is replaced by Σπ = 1.
  const a = new Float64Array(k * k)
  const b = new Float64Array(k)
  for (let r = 0; r < k; r++)
    for (let c = 0; c < k; c++) a[r * k + c] = p[members[c] * n + members[r]] - (r === c ? 1 : 0)
  for (let c = 0; c < k; c++) a[(k - 1) * k + c] = 1
  b[k - 1] = 1
  const { x, singular } = solveDense(a, b, k)
  if (singular || x === null) throw new DomainError('stationaryDistribution', 'stationaryDistribution: singular system')
  const pi = new Float64Array(n)
  members.forEach((s, i) => (pi[s] = Math.max(0, x[i])))
  const total = pi.reduce((u, v) => u + v, 0)
  return pi.map((v) => v / total)
}

/**
 * The stationary distributions of a chain that are supported on one closed class each, as the rows of a
 * $k \times n$ tensor ($k$ closed classes, in the order of `classifyStates`). Every stationary distribution is a
 * mixture of these rows.
 *
 * @param P The transition matrix ($n \times n$).
 * @returns A $k \times n$ float64 tensor, one stationary distribution per row.
 *
 * @example Gambler's ruin on $\{0, 1, 2, 3\}$: one row per absorbing end
 * const P = [[1, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 1]]
 * print('rows:', stationaryDistributions(P))
 */
export function stationaryDistributions(P: MatrixLike): Tensor {
  const { p, n } = readChain(P, 'stationaryDistributions')
  const { classes, closed } = classifyStates(P)
  const rows = classes.filter((_, c) => closed[c]).map((members) => stationaryOn(p, n, members))
  const out = new Float64Array(rows.length * n)
  rows.forEach((r, i) => out.set(r, i * n))
  return dense.mat(out, rows.length, n)
}

/**
 * The stationary distribution $\pivec$ of a chain with one closed class: the unique probability vector with
 * $\pivec\Pmat = \pivec$, found by one linear solve (one balance equation replaced by $\sum_i \pi_i = 1$).
 * Transient states get $\pi_i = 0$. Throws `DomainError` when the chain has several closed classes ($\pivec$ is then
 * not unique; see `stationaryDistributions`).
 *
 * @param P The transition matrix ($n \times n$).
 * @returns $\pivec$, $n$ values.
 *
 * @example A two-state chain: $\pi_0 P_{01} = \pi_1 P_{10}$ gives $(5/6, 1/6)$
 * const pi = stationaryDistribution([[0.9, 0.1], [0.5, 0.5]])
 * print('pi =', pi)
 * print('5/6, 1/6 =', 5 / 6, 1 / 6)
 *
 * @example A transient state gets no mass
 * print('pi =', stationaryDistribution([[0.5, 0.5, 0], [0, 0, 1], [0, 1, 0]]))
 */
export function stationaryDistribution(P: MatrixLike): Tensor {
  const where = 'stationaryDistribution'
  const { p, n } = readChain(P, where)
  const { classes, closed } = classifyStates(P)
  const closedClasses = classes.filter((_, c) => closed[c])
  if (closedClasses.length > 1)
    throw new DomainError(
      where,
      `${where}: the chain has ${closedClasses.length} closed classes, so its stationary distribution is not unique`,
    )
  return dense.vec(stationaryOn(p, n, closedClasses[0]))
}

/**
 * Kac's formula: the mean return time to each state of an irreducible chain, $1/\pi_i$. A transient state of a chain
 * with one closed class gets $\infty$; a chain with several closed classes throws (see `stationaryDistribution`).
 *
 * @param P The transition matrix ($n \times n$).
 * @returns The expected number of steps to return to each state, $n$ values.
 *
 * @example With $\pivec = (5/6, 1/6)$, the returns take $6/5$ and 6 steps
 * print('mean return times:', meanReturnTimes([[0.9, 0.1], [0.5, 0.5]]))
 */
export function meanReturnTimes(P: MatrixLike): Tensor {
  const pi = dense.data(stationaryDistribution(P))
  return dense.vec(pi.map((v) => (v > 0 ? 1 / v : Infinity)))
}

// ── Absorption and hitting ───────────────────────────────────────────────────────────────────────────────────────────

/** The result of `absorption`. */
export interface Absorption {
  /** The transient (non-absorbing) states, in order: the rows of every matrix below. */
  transient: number[]
  /** The absorbing states, in order: the columns of `probabilities`. */
  absorbing: number[]
  /**
   * The fundamental matrix $\Nmat = (\Imat - \Qmat)^{-1}$ ($t \times t$ for $t$ transient states): $N_{ij}$ is the
   * expected number of visits to $j$ starting from $i$.
   */
  fundamental: Tensor
  /**
   * $\Bmat = \Nmat\Rmat$ ($t \times a$ for $a$ absorbing states): the probability of being absorbed in each absorbing
   * state, from each transient state.
   */
  probabilities: Tensor
  /** $\tvec = \Nmat\ones$ ($t$ values): the expected number of steps before absorption. */
  expectedSteps: Tensor
  /**
   * The variance of the number of steps before absorption, $(2\Nmat - \Imat)\tvec - \tvec \circ \tvec$ ($t$ values,
   * rounding below 0 clipped).
   */
  varianceSteps: Tensor
}

/**
 * Absorption analysis by first-step analysis (Kemeny and Snell, 1960). With the states ordered transient first,
 * $\Pmat = \begin{bmatrix} \Qmat & \Rmat \\ \zeros & \Imat \end{bmatrix}$, the fundamental matrix
 * $\Nmat = (\Imat - \Qmat)^{-1}$ counts expected visits, $\Bmat = \Nmat\Rmat$ gives the absorption probabilities
 * and $\tvec = \Nmat\ones$ the expected time to absorption. The absorbing states default to those with
 * $P_{ii} = 1$; any set may be given instead, in which case the chain is stopped on first entering it. Throws
 * `DomainError` when there is no absorbing state, or when a non-absorbing state cannot reach the absorbing set
 * ($\Imat - \Qmat$ is then singular).
 *
 * @param P The transition matrix ($n \times n$).
 * @param options `absorbing`, the states at which the chain is stopped (default: the states with no positive
 *   transition to another state).
 * @returns The transient and absorbing states, which index the rows and columns of the results, with
 *   $\Nmat$, $\Bmat$, $\tvec$ and the variance of the time to absorption.
 *
 * @example Gambler's ruin on $\{0, 1, 2, 3\}$ with a fair coin
 * const P = [[1, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 1]]
 * const a = absorption(P)
 * print('transient:', a.transient, 'absorbing:', a.absorbing)
 * print('B =', a.probabilities)
 * print('expected steps:', a.expectedSteps)
 * print('variance:', a.varianceSteps)
 *
 * @example Stopping a chain on a chosen set
 * const a = absorption([[0.9, 0.1], [0.5, 0.5]], { absorbing: [1] })
 * print('steps from 0 to reach 1:', a.expectedSteps)
 */
export function absorption(P: MatrixLike, options: { absorbing?: readonly number[] } = {}): Absorption {
  const where = 'absorption'
  const { p, n } = readChain(P, where)
  const absorbing = options.absorbing === undefined ? absorbingStates(p, n) : readStates(options.absorbing, n, where)
  if (absorbing.length === 0) throw new DomainError(where, `${where}: the chain has no absorbing state`)
  const isAbsorbing = new Uint8Array(n)
  absorbing.forEach((s) => (isAbsorbing[s] = 1))
  const transient = Array.from({ length: n }, (_, i) => i).filter((i) => !isAbsorbing[i])
  const reach = canReach(p, n, absorbing)
  const stuck = transient.filter((i) => !reach[i])
  if (stuck.length > 0)
    throw new DomainError(where, `${where}: state ${stuck[0]} cannot reach an absorbing state, so I − Q is singular`)
  const t = transient.length
  const a = absorbing.length
  const iq = new Float64Array(t * t)
  for (let r = 0; r < t; r++)
    for (let c = 0; c < t; c++) iq[r * t + c] = (r === c ? 1 : 0) - p[transient[r] * n + transient[c]]
  const solved = t === 0 ? new Float64Array(0) : solveDense(iq, dense.identity(t), t).x
  const fundamental = solved === null ? null : Float64Array.from(solved)
  if (fundamental === null) throw new DomainError(where, `${where}: I − Q is singular`)
  const B = new Float64Array(t * a)
  for (let r = 0; r < t; r++)
    for (let c = 0; c < a; c++) {
      let s = 0
      for (let k = 0; k < t; k++) s += fundamental[r * t + k] * p[transient[k] * n + absorbing[c]]
      B[r * a + c] = s
    }
  const steps = new Float64Array(t)
  for (let r = 0; r < t; r++) for (let k = 0; k < t; k++) steps[r] += fundamental[r * t + k]
  const variance = new Float64Array(t)
  for (let r = 0; r < t; r++) {
    let s = 0
    for (let k = 0; k < t; k++) s += (2 * fundamental[r * t + k] - (r === k ? 1 : 0)) * steps[k]
    variance[r] = Math.max(0, s - steps[r] ** 2)
  }
  return {
    transient,
    absorbing,
    fundamental: dense.mat(fundamental, t, t),
    probabilities: dense.mat(B, t, a),
    expectedSteps: dense.vec(steps),
    varianceSteps: dense.vec(variance),
  }
}

/**
 * The hitting probabilities $h_i = \pr(X_t \in A \text{ for some } t \ge 0 \mid X_0 = i)$ of a target set $A$: 1
 * on $A$, 0 on the states that cannot reach $A$, and on the rest the solution of $\hvec = \Pmat\hvec$ (the minimal
 * non-negative solution, by one linear solve once the states that cannot reach $A$ are removed). Values are clipped to
 * $[0, 1]$; a singular system gives NaN.
 *
 * @param P The transition matrix ($n \times n$).
 * @param target The target set $A$: a non-empty list of states (duplicates allowed).
 * @returns $\hvec$, $n$ values.
 *
 * @example Gambler's ruin on $\{0, 1, 2, 3\}$: the chance of reaching 3
 * const P = [[1, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 1]]
 * print('h =', hittingProbabilities(P, [3]))
 */
export function hittingProbabilities(P: MatrixLike, target: readonly number[] | VectorLike): Tensor {
  const where = 'hittingProbabilities'
  const { p, n } = readChain(P, where)
  const A = readStates(target, n, where)
  const inA = new Uint8Array(n)
  A.forEach((s) => (inA[s] = 1))
  const reach = canReach(p, n, A)
  const rest = Array.from({ length: n }, (_, i) => i).filter((i) => !inA[i] && reach[i])
  const h = new Float64Array(n)
  A.forEach((s) => (h[s] = 1))
  const k = rest.length
  if (k > 0) {
    const a = new Float64Array(k * k)
    const b = new Float64Array(k)
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) a[r * k + c] = (r === c ? 1 : 0) - p[rest[r] * n + rest[c]]
      for (const s of A) b[r] += p[rest[r] * n + s]
    }
    const x = solveDense(a, b, k).x ?? new Float64Array(k).fill(NaN)
    rest.forEach((s, i) => (h[s] = Math.min(1, Math.max(0, x[i]))))
  }
  return dense.vec(h)
}

/**
 * The expected hitting times $k_i = \expect[\min\{t \ge 0 : X_t \in A\} \mid X_0 = i]$ of a target set $A$: 0 on
 * $A$, $\infty$ where the chain may never reach $A$ ($h_i < 1$: it can reach a state that cannot reach $A$ without
 * entering $A$ first), and on the rest the solution of $\kvec = \ones + \Pmat\kvec$. A singular system gives NaN.
 *
 * @param P The transition matrix ($n \times n$).
 * @param target The target set $A$: a non-empty list of states (duplicates allowed).
 * @returns $\kvec$, $n$ values.
 *
 * @example Gambler's ruin: either end is reached in 2 steps on average, but 3 alone may never be
 * const P = [[1, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 1]]
 * print('to {0, 3}:', expectedHittingTimes(P, [0, 3]))
 * print('to {3}:', expectedHittingTimes(P, [3]))
 *
 * @example A two-state chain: from 0, state 1 is reached after $1/P_{01} = 10$ steps
 * print('to {1}:', expectedHittingTimes([[0.9, 0.1], [0.5, 0.5]], [1]))
 */
export function expectedHittingTimes(P: MatrixLike, target: readonly number[] | VectorLike): Tensor {
  const where = 'expectedHittingTimes'
  const { p, n } = readChain(P, where)
  const A = readStates(target, n, where)
  const inA = new Uint8Array(n)
  A.forEach((s) => (inA[s] = 1))
  const reach = canReach(p, n, A)
  // States that cannot reach A, and then every state that can reach one of them without entering A first (the chain
  // stops on A), may never hit A.
  const lost = Array.from({ length: n }, (_, i) => i).filter((i) => !reach[i])
  const escapes = lost.length > 0 ? canReach(p, n, lost, inA) : new Uint8Array(n)
  const k = new Float64Array(n)
  const rest: number[] = []
  for (let i = 0; i < n; i++) {
    if (inA[i]) k[i] = 0
    else if (escapes[i]) k[i] = Infinity
    else rest.push(i)
  }
  const m = rest.length
  if (m > 0) {
    const a = new Float64Array(m * m)
    const b = new Float64Array(m).fill(1)
    for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) a[r * m + c] = (r === c ? 1 : 0) - p[rest[r] * n + rest[c]]
    const x = solveDense(a, b, m).x ?? new Float64Array(m).fill(NaN)
    rest.forEach((s, i) => (k[s] = x[i]))
  }
  return dense.vec(k)
}

// ── Convergence ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** The result of `distanceToStationarity`. */
export interface StationarityDistance {
  /** $d(t) = \max_x \lVert \Pmat^t(x, \cdot) - \pivec \rVert_{TV}$ for $t = 0, \dots, T$ ($T + 1$ values). */
  worst: Tensor
  /**
   * $\lVert \Pmat^t(x, \cdot) - \pivec \rVert_{TV}$ for each start $x$: a $(T + 1) \times n$ tensor, row $t$ for
   * step $t$.
   */
  perStart: Tensor
  /** The stationary distribution used. */
  stationary: Tensor
}

/**
 * The total-variation distance to stationarity of the chain started from each state, for $t = 0, \dots, T$, and its
 * worst case $d(t) = \max_x \lVert \Pmat^t(x, \cdot) - \pivec \rVert_{TV}$, which never increases (Levin, Peres
 * and Wilmer, 2017, §4.4). Needs a unique stationary distribution (else `DomainError`).
 *
 * @param P The transition matrix ($n \times n$).
 * @param steps The last step $T$, an integer $\ge 0$ (else `DomainError`).
 * @returns The worst-case and per-start distances for each step, and the stationary distribution used.
 *
 * @example The worst case shrinks by the factor 0.4, the second eigenvalue, each step
 * const d = distanceToStationarity([[0.9, 0.1], [0.5, 0.5]], 4)
 * print('d(t) =', d.worst)
 * print('from each start:', d.perStart)
 */
export function distanceToStationarity(P: MatrixLike, steps: number): StationarityDistance {
  const where = 'distanceToStationarity'
  const { p, n } = readChain(P, where)
  if (!Number.isInteger(steps) || steps < 0) throw new DomainError(where, `${where}: steps must be an integer ≥ 0`)
  const pi = dense.data(stationaryDistribution(P))
  const worst = new Float64Array(steps + 1)
  const per = new Float64Array((steps + 1) * n)
  let Pt = dense.identity(n)
  for (let t = 0; t <= steps; t++) {
    const d = tv(Pt, pi, n)
    per.set(d, t * n)
    worst[t] = Math.max(...d)
    if (t < steps) Pt = dense.matMul(Pt, p, n, n, n)
  }
  return { worst: dense.vec(worst), perStart: dense.mat(per, steps + 1, n), stationary: dense.vec(pi) }
}

/** The result of `spectralGap`. */
export interface SpectralGap {
  /**
   * The real parts of the eigenvalues of $\Pmat$: first the eigenvalue nearest 1 (which is 1), then the others by
   * modulus, descending.
   */
  real: Tensor
  /** The imaginary parts of the eigenvalues, in the order of `real`. */
  imag: Tensor
  /**
   * The largest modulus among the eigenvalues other than the first 1:
   * $\lambda_\star = \max_{k \ge 2} \lvert \lambda_k \rvert$ (0 for a one-state chain).
   */
  secondModulus: number
  /** The spectral gap $1 - \lambda_2$, with $\lambda_2$ the largest real part among the other eigenvalues. */
  gap: number
  /** The absolute spectral gap $1 - \lambda_\star$. */
  absoluteGap: number
  /**
   * The relaxation time $1/(1 - \lambda_\star)$ ($\infty$ when the absolute gap is below $10^{-12}$, as for a
   * periodic chain or one with several closed classes).
   */
  relaxationTime: number
}

/**
 * The eigenvalues of $\Pmat$ (by `eig`) and its spectral gaps (Levin, Peres and Wilmer, 2017, §12.2). For an
 * irreducible, aperiodic chain $\lambda_\star < 1$, and the distance to stationarity falls like $\lambda_\star^t$;
 * for a reversible chain the relaxation time $1/(1 - \lambda_\star)$ bounds the mixing time on both sides (see
 * `mixingTime`).
 *
 * @param P The transition matrix ($n \times n$).
 * @returns The eigenvalues, $\lambda_\star$, the spectral and absolute spectral gaps, and the relaxation time.
 *
 * @example A two-state chain has eigenvalues 1 and $1 - P_{01} - P_{10} = 0.4$
 * const g = spectralGap([[0.9, 0.1], [0.5, 0.5]])
 * print('eigenvalues:', g.real)
 * print('gap:', g.gap, 'relaxation time:', g.relaxationTime)
 *
 * @example A periodic chain has $\lambda_\star = 1$
 * const g = spectralGap([[0, 1], [1, 0]])
 * print('eigenvalues:', g.real, 'absolute gap:', g.absoluteGap, 'relaxation time:', g.relaxationTime)
 */
export function spectralGap(P: MatrixLike): SpectralGap {
  const where = 'spectralGap'
  const { p, n } = readChain(P, where)
  const { values } = eig(dense.mat(p, n, n), { vectors: false })
  const re = Array.from(toFlat(realPart(values)))
  const im = Array.from(toFlat(imagPart(values)))
  const order = re.map((_, i) => i).sort((a, b) => Math.hypot(re[b], im[b]) - Math.hypot(re[a], im[a]))
  // Remove the one eigenvalue nearest 1 (it is exactly 1 for a stochastic matrix).
  let trivial = order[0]
  for (const i of order) if (Math.hypot(re[i] - 1, im[i]) < Math.hypot(re[trivial] - 1, im[trivial])) trivial = i
  const sorted = [trivial, ...order.filter((i) => i !== trivial)]
  const others = sorted.slice(1)
  const secondModulus = others.length ? Math.max(...others.map((i) => Math.hypot(re[i], im[i]))) : 0
  const lambda2 = others.length ? Math.max(...others.map((i) => re[i])) : 0
  const absoluteGap = 1 - secondModulus
  return {
    real: dense.vec(Float64Array.from(sorted, (i) => re[i])),
    imag: dense.vec(Float64Array.from(sorted, (i) => im[i])),
    secondModulus,
    gap: 1 - lambda2,
    absoluteGap,
    relaxationTime: absoluteGap > 1e-12 ? 1 / absoluteGap : Infinity,
  }
}

/** The result of `isReversible`. */
export interface Reversibility {
  /** Whether `maxViolation` is within the tolerance. */
  reversible: boolean
  /** $\max_{i, j} \lvert \pi_i P_{ij} - \pi_j P_{ji} \rvert$. */
  maxViolation: number
  /** The stationary distribution $\pivec$ that was checked. */
  stationary: Tensor
}

/**
 * Detailed balance: whether $\pi_i P_{ij} = \pi_j P_{ji}$ for all $i, j$, with $\pivec$ the stationary
 * distribution (which must be unique, else `DomainError`).
 *
 * @param P The transition matrix ($n \times n$).
 * @param options `tolerance`, the largest violation still counted as balanced (default $10^{-10}$).
 * @returns Whether the chain is reversible, the largest violation, and $\pivec$.
 *
 * @example Every two-state chain is reversible; a cycle with a drift is not
 * print('two-state:', isReversible([[0.9, 0.1], [0.5, 0.5]]).reversible)
 * const drift = isReversible([[0, 0.9, 0.1], [0.1, 0, 0.9], [0.9, 0.1, 0]])
 * print('drifting cycle:', drift.reversible, 'violation:', drift.maxViolation)
 */
export function isReversible(P: MatrixLike, options: { tolerance?: number } = {}): Reversibility {
  const { p, n } = readChain(P, 'isReversible')
  const pi = dense.data(stationaryDistribution(P))
  let worst = 0
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) worst = Math.max(worst, Math.abs(pi[i] * p[i * n + j] - pi[j] * p[j * n + i]))
  return { reversible: worst <= (options.tolerance ?? 1e-10), maxViolation: worst, stationary: dense.vec(pi) }
}

/** The result of `mixingTime`. */
export interface MixingTime {
  /**
   * $t_{mix}(\varepsilon) = \min\{t : d(t) \le \varepsilon\}$; $\infty$ when not reached within `maxSteps` (a
   * periodic chain never mixes).
   */
  time: number
  /** The threshold $\varepsilon$ used. */
  epsilon: number
  /** $d(t)$ for $t = 0$ up to `time` (or up to `maxSteps` when not reached). */
  distances: Tensor
  /**
   * For a reversible chain, the lower bound $(t_{rel} - 1) \ln(1/(2\varepsilon)) \le t_{mix}(\varepsilon)$ (Levin,
   * Peres and Wilmer, 2017, Theorem 12.5), with $t_{rel}$ the relaxation time; undefined otherwise.
   */
  lower?: number
  /**
   * For a reversible chain, the upper bound
   * $t_{mix}(\varepsilon) \le \lceil t_{rel} \ln(1/(\varepsilon \pi_{min})) \rceil$ (Theorem 12.4), with
   * $\pi_{min}$ the smallest positive $\pi_i$; undefined otherwise.
   */
  upper?: number
}

/**
 * The mixing time
 * $t_{mix}(\varepsilon) = \min\{t : \max_x \lVert \Pmat^t(x, \cdot) - \pivec \rVert_{TV} \le \varepsilon\}$
 * (default $\varepsilon = 1/4$), by computing $\Pmat^t$ until the worst distance falls to $\varepsilon$; with the relaxation-time bounds when the chain is reversible (detailed balance within $10^{-9}$).
 * Needs a unique stationary distribution (else `DomainError`).
 *
 * @param P The transition matrix ($n \times n$).
 * @param options `epsilon`, the threshold $\varepsilon$, in $(0, 1)$ (default 0.25; else `DomainError`); `maxSteps`,
 *   the last $t$ tried (default 10000).
 * @returns The mixing time, the distances $d(t)$ computed, and for a reversible chain the bounds.
 *
 * @example A two-state chain mixes in a few steps, within its bounds
 * const m = mixingTime([[0.9, 0.1], [0.5, 0.5]])
 * print('t_mix =', m.time, 'bounds:', m.lower, m.upper)
 * print('d(t) =', m.distances)
 * print('t_mix(0.01) =', mixingTime([[0.9, 0.1], [0.5, 0.5]], { epsilon: 0.01 }).time)
 *
 * @example A periodic chain never mixes
 * print('t_mix =', mixingTime([[0, 1], [1, 0]], { maxSteps: 20 }).time)
 */
export function mixingTime(P: MatrixLike, options: { epsilon?: number; maxSteps?: number } = {}): MixingTime {
  const where = 'mixingTime'
  const { p, n } = readChain(P, where)
  const epsilon = options.epsilon ?? 0.25
  const maxSteps = options.maxSteps ?? 10_000
  if (!(epsilon > 0 && epsilon < 1)) throw new DomainError(where, `${where}: ε must be in (0, 1)`)
  const pi = dense.data(stationaryDistribution(P))
  const distances: number[] = []
  let Pt = dense.identity(n)
  let time = Infinity
  for (let t = 0; t <= maxSteps; t++) {
    const w = Math.max(...tv(Pt, pi, n))
    distances.push(w)
    if (w <= epsilon) {
      time = t
      break
    }
    Pt = dense.matMul(Pt, p, n, n, n)
  }
  const out: MixingTime = { time, epsilon, distances: dense.vec(Float64Array.from(distances)) }
  if (isReversible(P, { tolerance: 1e-9 }).reversible) {
    const { relaxationTime } = spectralGap(P)
    const piMin = Math.min(...Array.from(pi).filter((v) => v > 0))
    out.lower = Math.max(0, (relaxationTime - 1) * Math.log(1 / (2 * epsilon)))
    out.upper = Math.ceil(relaxationTime * Math.log(1 / (epsilon * piMin)))
  }
  return out
}

// ── Simulation ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A path $X_0, X_1, \dots, X_T$ of the chain, each step a categorical draw from the current state's row of $\Pmat$.
 *
 * @param s The stream the path is drawn from; advanced by every draw.
 * @param P The transition matrix ($n \times n$).
 * @param start $X_0$ itself (a state, not drawn), or a distribution to draw it from.
 * @param steps The number of transitions $T$, an integer $\ge 0$ (else `DomainError`).
 * @returns The path as an int32 tensor of $T + 1$ states.
 *
 * @example Twenty steps of a sticky two-state chain
 * print('path:', simulateChain(stream(0), [[0.9, 0.1], [0.5, 0.5]], 0, 20))
 * print('from (0.5, 0.5):', simulateChain(stream(1), [[0.9, 0.1], [0.5, 0.5]], [0.5, 0.5], 10))
 */
export function simulateChain(s: Stream, P: MatrixLike, start: VectorLike | number, steps: number): Tensor {
  const where = 'simulateChain'
  const { p, n } = readChain(P, where)
  if (!Number.isInteger(steps) || steps < 0) throw new DomainError(where, `${where}: steps must be an integer ≥ 0`)
  const p0 = readDistribution(start, n, where)
  const path = new Int32Array(steps + 1)
  path[0] = typeof start === 'number' ? start : categorical(s, p0)
  for (let t = 1; t <= steps; t++) path[t] = categorical(s, p.subarray(path[t - 1] * n, (path[t - 1] + 1) * n))
  return fromData(path, [steps + 1])
}

/** One state of `markovChainSteps`. */
export interface MarkovChainState extends Status {
  /** The walker's current state $X_t$. */
  state: number
  /** The visits to each state among $X_0, \dots, X_t$ ($n$ counts). */
  visits: Tensor
  /** The exact distribution $\pvec_0\Pmat^t$ of $X_t$ ($n$ values). */
  distribution: Tensor
  /** $\lVert \pvec_0\Pmat^t - \pivec \rVert_{TV}$ (NaN when $\pivec$ is not unique). */
  distance: number
  /**
   * $\lVert \vvec/(t + 1) - \pivec \rVert_{TV}$ with $\vvec$ the `visits`: the distance of the occupation measure
   * (NaN when $\pivec$ is not unique).
   */
  occupationDistance: number
}

/**
 * A chain stepped one transition at a time: a walker drawn from the chain (its stream is the runner's), its visit
 * counts, and beside it the exact distribution $\pvec_0\Pmat^t$ and both distances to $\pivec$. The walker's
 * occupation measure converges to $\pivec$ for any irreducible chain (the ergodic theorem); $\pvec_0\Pmat^t$
 * converges only when the chain is also aperiodic. `init` takes $X_0$ from `start`: the state itself, or a draw from
 * the distribution given. The algorithm has no starting point (`run` is given `undefined`) and never stops by itself.
 *
 * @param P The transition matrix ($n \times n$).
 * @param options `start`, the state $X_0$ or the distribution $\pvec_0$ it is drawn from (default state 0).
 * @returns The algorithm, whose state at step $t$ is the walker's state, visit counts, $\pvec_0\Pmat^t$ and the two
 *   distances.
 *
 * @example The occupation measure and $\pvec_0\Pmat^t$ after 200 steps
 * const s = run(markovChainSteps([[0.9, 0.1], [0.5, 0.5]]), undefined, 200)
 * print('t =', s.t, 'state:', s.state, 'visits:', s.visits)
 * print('distance of p0 P^t:', s.distance)
 * print('distance of the occupation measure:', s.occupationDistance)
 *
 * @example A periodic chain: the occupation measure converges, $\pvec_0\Pmat^t$ does not
 * const s = run(markovChainSteps([[0, 1], [1, 0]]), undefined, 101)
 * print('distribution:', s.distribution, 'distance:', s.distance)
 * print('occupation distance:', s.occupationDistance)
 */
export function markovChainSteps(
  P: MatrixLike,
  options: { start?: VectorLike | number } = {},
): Algorithm<void, MarkovChainState> {
  const where = 'markovChainSteps'
  const { p, n } = readChain(P, where)
  const start = options.start ?? 0
  const p0 = readDistribution(start, n, where)
  let pi: F64 | null = null
  try {
    pi = dense.data(stationaryDistribution(P))
  } catch {
    pi = null
  }
  const make = (t: number, state: number, visits: F64, distribution: F64): MarkovChainState => ({
    t,
    state,
    visits: dense.vec(visits),
    distribution: dense.vec(distribution),
    distance: pi ? tv(distribution, pi)[0] : NaN,
    occupationDistance: pi
      ? tv(
          visits.map((v) => v / (t + 1)),
          pi,
        )[0]
      : NaN,
  })
  return {
    name: 'markov-chain',
    init: (_, s) => {
      const x0 = typeof start === 'number' ? start : categorical(s, p0)
      const visits = new Float64Array(n)
      visits[x0] = 1
      return make(0, x0, visits, p0)
    },
    step: (s, ctx) => {
      const next = categorical(ctx.stream, p.subarray(s.state * n, (s.state + 1) * n))
      const visits = Float64Array.from(dense.data(s.visits))
      visits[next] += 1
      return make(s.t + 1, next, visits, stepRow(dense.data(s.distribution), p, n))
    },
  }
}
