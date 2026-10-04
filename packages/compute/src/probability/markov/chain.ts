/**
 * Finite Markov chains on states 0 … n − 1 with a row-stochastic transition matrix P (P_ij = P(X_{t+1} = j | X_t = i)):
 * the classification of states (communicating classes, closed classes, periods), the stationary distribution
 * πP = π, absorption and hitting probabilities and times by first-step analysis, the distance to stationarity and the
 * mixing time, the spectral gap and the reversibility check, and a simulator. Matrices are small and dense; every
 * quantity is computed exactly (linear solves and matrix powers), never by simulation.
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

type F64 = dense.F64

/** Rows must sum to one within this. */
const ROW_TOLERANCE = 1e-9

/** The validated matrix of a chain as row-major data. */
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

function readStates(states: readonly number[] | VectorLike, n: number, where: string): number[] {
  const list = Array.from(dense.toF64(states as VectorLike, where))
  if (list.length === 0) throw new DomainError(where, `${where}: the target set is empty`)
  for (const s of list)
    if (!Number.isInteger(s) || s < 0 || s >= n) throw new DomainError(where, `${where}: no state ${s}`)
  return [...new Set(list)].sort((a, b) => a - b)
}

/** p ↦ pP for a row vector p. */
const stepRow = (p: ArrayLike<number>, P: F64, n: number): F64 => dense.matTVec(P, p, n, n)

/** ‖p − q‖_TV for one distribution p [n], or for each row of p [m, n], against q [n]. */
function tv(p: F64, q: F64, rows?: number): F64 {
  const d = totalVariation(rows === undefined ? dense.vec(p) : dense.mat(p, rows, q.length), dense.vec(q))
  return typeof d === 'number' ? Float64Array.of(d) : dense.data(d as Tensor)
}

/** The transition graph: an edge i → j wherever P_ij > 0. */
function graphOf(P: F64, n: number) {
  const edges: [number, number][] = []
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (P[i * n + j] > 0) edges.push([i, j])
  return fromEdges(n, edges, { directed: true })
}

/**
 * The states from which some state of `targets` can be reached (in zero or more steps), as a mask. Paths may not pass
 * through a `blocked` state (one where the chain is stopped), though a blocked state is itself marked when it reaches.
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

/** The absorbing states: no positive transition to another state (so P_ii = 1 within the row tolerance). */
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
 * The transition matrix of a chain as a float64 tensor [n, n], after checking that it is square, non-negative and
 * row-stochastic (each row sums to 1 within 1e-9 · n). Throws a `DomainError` naming the offending row otherwise.
 */
export function transitionMatrix(P: MatrixLike): Tensor {
  const { p, n } = readChain(P, 'transitionMatrix')
  return dense.mat(Float64Array.from(p), n, n)
}

/** The n-step transition matrix Pⁿ (P⁰ = I), by repeated squaring. */
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

/** The distribution p₀Pᵗ of X_t for an initial distribution p₀ (or a start state). */
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
  /** Whether each class is closed (no transition leaves it). In a finite chain, closed ⇔ recurrent. */
  closed: boolean[]
  /** The period of each class: the gcd of the lengths of its cycles (1 = aperiodic). */
  periods: number[]
  /** The class index of each state. */
  classOf: Tensor
  /** States in closed classes (recurrent) and the rest (transient), in order. */
  recurrent: number[]
  transient: number[]
  /** States with P_ii = 1. */
  absorbing: number[]
  /** One class: every state reaches every other. */
  irreducible: boolean
  /** Every closed class has period 1. */
  aperiodic: boolean
}

const gcd = (a: number, b: number): number => (b === 0 ? Math.abs(a) : gcd(b, a % b))

/**
 * The communicating classes of a chain (the strongly connected components of its transition graph, Tarjan's
 * algorithm), which of them are closed (recurrent) and which transient, and the period of each: the gcd over edges
 * u → w inside the class of level(u) + 1 − level(w), with levels from a breadth-first search inside the class.
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

/** Solve πP = π, Σπ = 1 on the states `members` of a closed class (π is zero elsewhere). */
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
 * The stationary distributions of a chain that are supported on one closed class each, as rows of a tensor [k, n]
 * (k closed classes). Every stationary distribution is a mixture of these rows.
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
 * The stationary distribution π of a chain with one closed class: the unique probability vector with πP = π, found by
 * one linear solve (one balance equation replaced by Σπ = 1). Transient states get π = 0. Throws when the chain has
 * several closed classes (π is then not unique; see `stationaryDistributions`).
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

/** Kac's formula: the mean return time to each state of an irreducible chain, 1/πᵢ. */
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
  /** The fundamental matrix N = (I − Q)⁻¹ [t, t]: N_ij is the expected number of visits to j starting from i. */
  fundamental: Tensor
  /** B = NR [t, a]: the probability of being absorbed in each absorbing state, from each transient state. */
  probabilities: Tensor
  /** t = N1 [t]: the expected number of steps before absorption. */
  expectedSteps: Tensor
  /** The variance of the number of steps before absorption, (2N − I)t − t∘t [t]. */
  varianceSteps: Tensor
}

/**
 * Absorption analysis by first-step analysis (Kemeny and Snell, 1960). With the states ordered transient first,
 * P = [[Q, R], [0, I]], the fundamental matrix N = (I − Q)⁻¹ counts expected visits, B = NR gives the absorption
 * probabilities and t = N1 the expected time to absorption. The absorbing states default to those with P_ii = 1; any
 * set may be given instead, in which case the chain is stopped on first entering it. Throws when a non-absorbing state
 * cannot reach the absorbing set (I − Q is then singular).
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
 * The hitting probabilities hᵢ = P(X_t ∈ A for some t ≥ 0 | X₀ = i) of a target set A: 1 on A, 0 on the states that
 * cannot reach A, and on the rest the solution of h = Ph (the minimal non-negative solution, by one linear solve).
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
 * The expected hitting times kᵢ = E[min{t ≥ 0 : X_t ∈ A} | X₀ = i] of a target set A: 0 on A, ∞ where the chain may
 * never reach A (hᵢ < 1), and on the rest the solution of k = 1 + Pk.
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
  /** d(t) = maxₓ ‖Pᵗ(x, ·) − π‖_TV for t = 0 … steps [steps + 1]. */
  worst: Tensor
  /** ‖Pᵗ(x, ·) − π‖_TV for each start x: [steps + 1, n]. */
  perStart: Tensor
  /** The stationary distribution used. */
  stationary: Tensor
}

/**
 * The total-variation distance to stationarity of the chain started from each state, for t = 0 … steps, and its worst
 * case d(t) = maxₓ ‖Pᵗ(x, ·) − π‖_TV, which never increases (Levin, Peres and Wilmer, 2017, §4.4). Needs a unique
 * stationary distribution.
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
  /** The eigenvalues of P, real and imaginary parts, sorted by modulus (descending); the first is 1. */
  real: Tensor
  imag: Tensor
  /** The largest modulus among the eigenvalues other than the first 1: λ⋆ = max_{k ≥ 2} |λ_k|. */
  secondModulus: number
  /** The spectral gap 1 − λ₂, with λ₂ the largest real part among the other eigenvalues. */
  gap: number
  /** The absolute spectral gap 1 − λ⋆. */
  absoluteGap: number
  /** The relaxation time 1/(1 − λ⋆) (∞ for a periodic or reducible chain). */
  relaxationTime: number
}

/**
 * The eigenvalues of P and its spectral gaps. For an irreducible, aperiodic chain λ⋆ < 1, and the distance to
 * stationarity falls like λ⋆ᵗ; for a reversible chain the relaxation time 1/(1 − λ⋆) bounds the mixing time on both
 * sides (see `mixingTime`).
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
  reversible: boolean
  /** maxᵢⱼ |πᵢP_ij − πⱼP_ji|. */
  maxViolation: number
  stationary: Tensor
}

/** Detailed balance: whether πᵢP_ij = πⱼP_ji for all i, j (within `tolerance`, default 1e-10). */
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
  /** t_mix(ε) = min{t : d(t) ≤ ε}; ∞ when not reached within `maxSteps` (a periodic chain never mixes). */
  time: number
  epsilon: number
  /** d(t) for t = 0 … time (or … maxSteps when not reached). */
  distances: Tensor
  /**
   * For a reversible chain: (t_rel − 1) ln(1/(2ε)) ≤ t_mix(ε) ≤ ⌈t_rel ln(1/(ε π_min))⌉ (Levin, Peres and Wilmer,
   * 2017, Theorems 12.4 and 12.5); undefined otherwise.
   */
  lower?: number
  upper?: number
}

/**
 * The mixing time t_mix(ε) = min{t : maxₓ ‖Pᵗ(x, ·) − π‖_TV ≤ ε} (default ε = 1/4), by computing Pᵗ until the worst
 * distance falls below ε; with the relaxation-time bounds when the chain is reversible.
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

/** A path X₀, X₁, …, X_steps of the chain (int32 [steps + 1]), X₀ drawn from `start` (a state or a distribution). */
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
  /** The walker's current state X_t. */
  state: number
  /** Visits to each state among X₀ … X_t [n]. */
  visits: Tensor
  /** The exact distribution p₀Pᵗ of X_t [n]. */
  distribution: Tensor
  /** ‖p₀Pᵗ − π‖_TV (NaN when π is not unique). */
  distance: number
  /** ‖visits/(t + 1) − π‖_TV, the distance of the occupation measure (NaN when π is not unique). */
  occupationDistance: number
}

/**
 * A chain stepped one transition at a time: a walker drawn from the chain (its stream is the runner's), its visit
 * counts, and beside it the exact distribution p₀Pᵗ and both distances to π. The walker's occupation measure
 * converges to π for any irreducible chain (the ergodic theorem); p₀Pᵗ converges only when the chain is also
 * aperiodic. `init` draws X₀ from `start` (a state, or a distribution; default state 0).
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
