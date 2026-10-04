/**
 * Label propagation on a graph: a few labelled nodes, many unlabelled ones, and class scores spread along weighted
 * edges.
 *
 * - **Harmonic label propagation** (Zhu, Ghahramani and Lafferty 2003, "Semi-supervised learning using Gaussian fields
 *   and harmonic functions", ICML; Zhu and Ghahramani 2002, CMU-CALD-02-107): the scores F minimise
 *   Σᵢⱼ Wᵢⱼ ‖Fᵢ − Fⱼ‖² with the labelled rows fixed to their one-hot labels, so every unlabelled row is the weighted
 *   average of its neighbours: F_U = (D_UU − W_UU)⁻¹ W_UL Y_L. Iterating F ← D⁻¹WF and re-clamping the labelled rows
 *   converges to it (scikit-learn's `LabelPropagation`).
 * - **Label spreading** (Zhou, Bousquet, Lal, Weston and Schölkopf 2004, "Learning with local and global consistency",
 *   NeurIPS): with S = D^{−1/2} W D^{−1/2} (self-loops dropped), iterate F ← αSF + (1 − α)Y, whose limit is
 *   F* = (1 − α)(I − αS)⁻¹Y. Labelled rows are not clamped: α ∈ (0, 1) trades fitting the graph against keeping the
 *   given labels (scikit-learn's `LabelSpreading`).
 *
 * Labels are integers 0 … C − 1, with −1 for an unlabelled node. Scores are returned with each row normalised to sum to
 * one (rows with no mass stay zero), as scikit-learn's `label_distributions_`.
 */

import type { MatrixLike, Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { solve, squaredDistances } from 'aifn-compute/numerics/linalg'
import { weightOf, type Graph } from '../graph'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'

/** Edge weights as affinities: a graph (its weights, see {@link graphAffinity}) or a symmetric n × n matrix W ≥ 0. */
export type Affinity = Graph | MatrixLike

/** How {@link graphAffinity} turns edge weights into affinities. */
export type AffinityKernel =
  | { kind: 'weights' }
  | { kind: 'connectivity' }
  /** exp(−w²/(2σ²)) of an edge whose weight w is a distance (a k-NN or ε-ball graph). */
  | { kind: 'heat'; sigma: number }

/** The n × n affinity matrix of a graph: its edge weights as they are, 1 per edge, or a heat kernel of distances. */
export function graphAffinity(g: Graph, kernel: AffinityKernel = { kind: 'weights' }): Tensor {
  if (kernel.kind === 'weights') return adjacencyMatrix(g, { weighted: true })
  if (kernel.kind === 'heat' && !(kernel.sigma > 0))
    throw new DomainError('graphAffinity', 'graphAffinity: the heat kernel needs σ > 0')
  const map = (w: number) => (kernel.kind === 'connectivity' ? 1 : Math.exp(-(w * w) / (2 * kernel.sigma ** 2)))
  return adjacencyMatrix({ ...g, edges: g.edges.map((e) => ({ ...e, weight: map(weightOf(e)) })) }, { weighted: true })
}

/** Options of {@link pointAffinity}. */
export interface PointAffinityOptions {
  /** The Gaussian kernel's rate γ in exp(−γ‖xᵢ − xⱼ‖²) (default 1). */
  gamma?: number
  /** Keep only edges to each point's k nearest neighbours, symmetrised (i ~ j when either is among the other's); 0 or absent keeps the full graph. */
  neighbours?: Size
}

/**
 * The affinity matrix of a point cloud X (n × d): W_ij = exp(−γ‖xᵢ − xⱼ‖²) for i ≠ j and W_ii = 0 (Zhou et al. 2004; the
 * label-proportion propagation of Poyiadzi et al. 2018 uses this form), optionally restricted to a symmetrised k-NN
 * graph.
 */
export function pointAffinity(x: MatrixLike, options: PointAffinityOptions = {}): Tensor {
  const { gamma = 1, neighbours = 0 } = options
  if (!(gamma > 0)) throw new DomainError('pointAffinity', 'pointAffinity: γ must be positive')
  const D = dense.data(squaredDistances(x))
  const n = Math.round(Math.sqrt(D.length))
  const W = Float64Array.from(D, (d, k) => (Math.floor(k / n) === k % n ? 0 : Math.exp(-gamma * d)))
  if (!(neighbours > 0)) return fromData(W, [n, n])
  const k = Math.min(neighbours, n - 1)
  const keep = new Uint8Array(n * n)
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D[i * n + a] - D[i * n + b] || a - b)
    for (let r = 0; r < k; r++) {
      keep[i * n + order[r]] = 1
      keep[order[r] * n + i] = 1
    }
  }
  return fromData(
    W.map((w, idx) => (keep[idx] ? w : 0)),
    [n, n],
  )
}

function affinityOf(w: Affinity, where: string): { n: number; W: Float64Array } {
  const isGraph = !isTensor(w) && !Array.isArray(w) && (w as Graph).kind === 'graph'
  const t = isGraph ? adjacencyMatrix(w as Graph, { weighted: true }) : (w as MatrixLike)
  const { data, m, n } = dense.toMatrixF64(t, where)
  if (m !== n) throw new ShapeError(where, `${where}: the affinity must be square, got ${m} × ${n}`)
  for (const v of data) if (!(v >= 0)) throw new DomainError(where, `${where}: affinities must be non-negative`)
  return { n, W: Float64Array.from(data) }
}

/** Labels as a one-hot n × C matrix Y, with zero rows for unlabelled nodes (−1). */
export function labelMatrix(labels: ArrayLike<number>, classes?: Size): Tensor {
  const n = labels.length
  let C = classes ?? 0
  if (classes === undefined) for (let i = 0; i < n; i++) C = Math.max(C, labels[i] + 1)
  const Y = new Float64Array(n * C)
  for (let i = 0; i < n; i++) {
    const l = labels[i]
    if (l === -1) continue
    if (!(Number.isInteger(l) && l >= 0 && l < C))
      throw new DomainError('labelMatrix', `labelMatrix: label ${l} of node ${i} is not −1 or in 0 … ${C - 1}`)
    Y[i * C + l] = 1
  }
  return fromData(Y, [n, C])
}

/** Class scores (n × C), rows summing to one, and the predicted class of every node (int32, −1 where a row is zero). */
export interface LabelScores {
  readonly scores: Tensor
  readonly labels: Tensor
}

function normalised(F: Float64Array, n: number, C: number): LabelScores {
  const out = new Float64Array(F)
  const lab = new Int32Array(n).fill(-1)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let c = 0; c < C; c++) s += out[i * C + c]
    if (s === 0) continue
    let best = 0
    for (let c = 0; c < C; c++) {
      out[i * C + c] /= s
      if (out[i * C + c] > out[i * C + best]) best = c
    }
    lab[i] = best
  }
  return { scores: fromData(out, [n, C]), labels: fromData(lab, [n]) }
}

function setup(w: Affinity, labels: ArrayLike<number>, classes: Size | undefined, where: string) {
  const { n, W } = affinityOf(w, where)
  if (labels.length !== n) throw new ShapeError(where, `${where}: ${labels.length} labels for ${n} nodes`)
  const Y = labelMatrix(labels, classes)
  const C = Y.shape[1]
  const known = Array.from({ length: n }, (_, i) => labels[i] !== -1)
  if (!known.some(Boolean)) throw new DomainError(where, `${where}: no node is labelled`)
  return { n, W, Y: dense.data(Y), C, known }
}

/** Options shared by the label-propagation functions. */
export interface LabelOptions {
  /** The number of classes C (default: the largest label + 1). */
  classes?: Size
}

/**
 * The harmonic solution F_U = (D_UU − W_UU)⁻¹ W_UL Y_L with the labelled rows fixed to their labels (module notes),
 * solved directly. An unlabelled node with no path to a labelled one (a component without labels) would make the
 * system singular; such nodes keep zero scores and the label −1, as the iteration leaves them.
 */
export function harmonicLabels(w: Affinity, labels: ArrayLike<number>, options: LabelOptions = {}): LabelScores {
  const { n, W, Y, C, known } = setup(w, labels, options.classes, 'harmonicLabels')
  // The nodes joined to some labelled node by a path of positive affinities (either direction).
  const reached = Uint8Array.from(known, (k) => (k ? 1 : 0))
  const queue = known.flatMap((k, i) => (k ? [i] : []))
  while (queue.length > 0) {
    const i = queue.pop()!
    for (let j = 0; j < n; j++)
      if (!reached[j] && (W[i * n + j] > 0 || W[j * n + i] > 0)) {
        reached[j] = 1
        queue.push(j)
      }
  }
  const U = known.flatMap((k, i) => (k || !reached[i] ? [] : [i]))
  const L = known.flatMap((k, i) => (k ? [i] : []))
  const F = new Float64Array(Y)
  if (U.length > 0) {
    const u = U.length
    const A = new Float64Array(u * u)
    const B = new Float64Array(u * C)
    U.forEach((i, a) => {
      let deg = 0
      for (let j = 0; j < n; j++) if (j !== i) deg += W[i * n + j]
      A[a * u + a] = deg
      U.forEach((j, b) => {
        if (j !== i) A[a * u + b] -= W[i * n + j]
      })
      for (const j of L) for (let c = 0; c < C; c++) B[a * C + c] += W[i * n + j] * Y[j * C + c]
    })
    const X = dense.data(solve(fromData(A, [u, u]), fromData(B, [u, C])) as Tensor)
    U.forEach((i, a) => F.set(X.subarray(a * C, (a + 1) * C), i * C))
  }
  return normalised(F, n, C)
}

/**
 * The spreading matrix S = D^{−1/2} W D^{−1/2} of an affinity with its diagonal (self-loops) removed, D the degrees
 * without self-loops; a node of degree 0 has a zero row.
 */
export function spreadingMatrix(w: Affinity): Tensor {
  const { n, W } = affinityOf(w, 'spreadingMatrix')
  for (let i = 0; i < n; i++) W[i * n + i] = 0
  const d = new Float64Array(n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) d[i] += W[i * n + j]
  const S = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (d[i] > 0 && d[j] > 0) S[i * n + j] = W[i * n + j] / Math.sqrt(d[i] * d[j])
  return fromData(S, [n, n])
}

/**
 * The random-walk matrix S = D⁻¹W of an affinity with its diagonal removed: row i holds the probabilities of a walker at
 * node i stepping to each neighbour, W_ij / Σ_j W_ij. A node of degree 0 has a zero row. Label propagation through
 * linear neighbourhoods (Wang and Zhang 2008) and its label-proportion form (Poyiadzi et al. 2018) use this S, where
 * label spreading uses the symmetric D^{−1/2}WD^{−1/2} ({@link spreadingMatrix}).
 */
export function randomWalkMatrix(w: Affinity): Tensor {
  const { n, W } = affinityOf(w, 'randomWalkMatrix')
  for (let i = 0; i < n; i++) W[i * n + i] = 0
  for (let i = 0; i < n; i++) {
    let d = 0
    for (let j = 0; j < n; j++) d += W[i * n + j]
    for (let j = 0; j < n; j++) W[i * n + j] = d > 0 ? W[i * n + j] / d : 0
  }
  return fromData(W, [n, n])
}

/**
 * The resolvent of spreading, R = (1 − α)(I − αS)⁻¹ = (1 − α) Σ_k (αS)^k, for an n × n propagation matrix S and α ∈ (0, 1):
 * F* = RY is the limit of F ← αSF + (1 − α)Y. With `scaled: false` the (1 − α) factor is left out, (I − αS)⁻¹. When S is
 * row-stochastic (a random-walk matrix), the scaled R is too, so RY averages Y over walks whose length is geometric
 * with parameter 1 − α.
 */
export function spreadingResolvent(S: MatrixLike, alpha: number, options: { scaled?: boolean } = {}): Tensor {
  checkAlpha(alpha, 'spreadingResolvent')
  const { data, m, n } = dense.toMatrixF64(S, 'spreadingResolvent')
  if (m !== n) throw new ShapeError('spreadingResolvent', `spreadingResolvent: S must be square, got ${m} × ${n}`)
  const A = new Float64Array(n * n)
  const I = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    I[i * n + i] = options.scaled === false ? 1 : 1 - alpha
    for (let j = 0; j < n; j++) A[i * n + j] = (i === j ? 1 : 0) - alpha * data[i * n + j]
  }
  return solve(fromData(A, [n, n]), fromData(I, [n, n])) as Tensor
}

/** Options of label spreading. */
export interface SpreadingOptions extends LabelOptions {
  /** The weight α ∈ (0, 1) of the graph against the given labels (default 0.2, as scikit-learn). */
  alpha?: number
}

function checkAlpha(alpha: number, where: string): void {
  if (!(alpha > 0 && alpha < 1)) throw new DomainError(where, `${where}: α must lie in (0, 1)`)
}

/** The limit of label spreading, F* = (1 − α)(I − αS)⁻¹Y (module notes), solved directly and row-normalised. */
export function labelSpreading(w: Affinity, labels: ArrayLike<number>, options: SpreadingOptions = {}): LabelScores {
  const alpha = options.alpha ?? 0.2
  checkAlpha(alpha, 'labelSpreading')
  const { n, Y, C } = setup(w, labels, options.classes, 'labelSpreading')
  const S = dense.data(spreadingMatrix(w))
  const A = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] = (i === j ? 1 : 0) - alpha * S[i * n + j]
  const F = dense.data(solve(fromData(A, [n, n]), fromData(dense.scale(1 - alpha, Y), [n, C])) as Tensor)
  return normalised(F, n, C)
}

/** One state of iterative label propagation or spreading. */
export interface LabelPropagationState extends Status {
  /** The raw scores F (n × C) after t iterations (before the final row normalisation). */
  readonly scores: Tensor
  /** The predicted class per node (int32; −1 where a row is still zero). */
  readonly labels: Tensor
  /** Σ |F_t − F_{t−1}|, scikit-learn's convergence measure (Infinity at t = 0). */
  readonly change: number
}

/** Options of the stepped forms. */
export interface LabelStepsOptions extends LabelOptions {
  /** Stop once the change falls below this (default 1e-3, as scikit-learn). */
  tolerance?: number
}

function stepsOf(
  name: string,
  n: number,
  C: number,
  Y: Float64Array,
  update: (F: Float64Array) => Float64Array,
  tolerance: number,
): Algorithm<void, LabelPropagationState> {
  const state = (F: Float64Array, t: number, change: number): LabelPropagationState => ({
    t,
    scores: fromData(F, [n, C]),
    labels: normalised(F, n, C).labels,
    change,
    converged: change < tolerance,
  })
  return {
    name,
    init: () => state(new Float64Array(Y), 0, Infinity),
    step: (s) => {
      const F = dense.data(s.scores)
      const next = update(F)
      let change = 0
      for (let i = 0; i < next.length; i++) change += Math.abs(next[i] - F[i])
      return state(next, s.t + 1, change)
    },
  }
}

/**
 * Label propagation as steps (Zhu and Ghahramani 2002): F ← D⁻¹WF, each row normalised, labelled rows reset to their
 * labels; step 0 holds only the given labels. The limit is {@link harmonicLabels}.
 */
export function labelPropagationSteps(
  w: Affinity,
  labels: ArrayLike<number>,
  options: LabelStepsOptions = {},
): Algorithm<void, LabelPropagationState> {
  const { n, W, Y, C, known } = setup(w, labels, options.classes, 'labelPropagationSteps')
  const P = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += W[i * n + j]
    for (let j = 0; j < n; j++) P[i * n + j] = s > 0 ? W[i * n + j] / s : 0
  }
  return stepsOf(
    'labelPropagation',
    n,
    C,
    Y,
    (F) => {
      const G = dense.matMul(P, F, n, n, C)
      for (let i = 0; i < n; i++) {
        if (known[i]) {
          G.set(Y.subarray(i * C, (i + 1) * C), i * C)
          continue
        }
        let s = 0
        for (let c = 0; c < C; c++) s += G[i * C + c]
        if (s > 0) for (let c = 0; c < C; c++) G[i * C + c] /= s
      }
      return G
    },
    options.tolerance ?? 1e-3,
  )
}

/** Label spreading as steps (Zhou et al. 2004): F ← αSF + (1 − α)Y from F₀ = Y. The limit is {@link labelSpreading}. */
export function labelSpreadingSteps(
  w: Affinity,
  labels: ArrayLike<number>,
  options: SpreadingOptions & LabelStepsOptions = {},
): Algorithm<void, LabelPropagationState> {
  const alpha = options.alpha ?? 0.2
  checkAlpha(alpha, 'labelSpreadingSteps')
  const { n, Y, C } = setup(w, labels, options.classes, 'labelSpreadingSteps')
  const S = dense.data(spreadingMatrix(w))
  const base = dense.scale(1 - alpha, Y)
  return stepsOf(
    'labelSpreading',
    n,
    C,
    Y,
    (F) => dense.axpy(alpha, dense.matMul(S, F, n, n, C), base),
    options.tolerance ?? 1e-3,
  )
}

/** Row-normalised scores and predicted classes of raw scores (the last step of the stepped forms). */
export function normaliseScores(scores: Tensor): LabelScores {
  const [n, C] = scores.shape
  return normalised(dense.data(scores), n, C)
}
