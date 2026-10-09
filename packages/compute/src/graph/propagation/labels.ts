/**
 * Label propagation on a graph: a few labelled nodes, many unlabelled ones, and class scores spread along weighted
 * edges.
 *
 * - **Harmonic label propagation** (Zhu, Ghahramani and Lafferty 2003, "Semi-supervised learning using Gaussian fields
 *   and harmonic functions", ICML; Zhu and Ghahramani 2002, CMU-CALD-02-107): the scores $\Fmat$ minimise
 *   $\sum_{ij} W_{ij} \lVert \fvec_i - \fvec_j \rVert^2$ ($\fvec_i$ the row of node $i$) with the labelled rows
 *   fixed to their one-hot labels, so every unlabelled row is the weighted average of its neighbours:
 *   $\Fmat_U = (\Dmat_{UU} - \Wmat_{UU})^{-1} \Wmat_{UL} \Ymat_L$, with $U$ the unlabelled nodes, $L$ the labelled
 *   ones and $\Dmat$ the diagonal matrix of degrees. Iterating $\Fmat \leftarrow \Dmat^{-1}\Wmat\Fmat$ and
 *   re-clamping the labelled rows converges to it (scikit-learn's `LabelPropagation`).
 * - **Label spreading** (Zhou, Bousquet, Lal, Weston and Schölkopf 2004, "Learning with local and global consistency",
 *   NeurIPS): with $\Smat = \Dmat^{-1/2} \Wmat \Dmat^{-1/2}$ (self-loops dropped), iterate
 *   $\Fmat \leftarrow \alpha\Smat\Fmat + (1 - \alpha)\Ymat$, whose limit is
 *   $\Fmat^* = (1 - \alpha)(\Imat - \alpha\Smat)^{-1}\Ymat$. Labelled rows are not clamped: $\alpha \in (0, 1)$
 *   trades fitting the graph against keeping the given labels (scikit-learn's `LabelSpreading`).
 *
 * The affinities $\Wmat$ ($n \times n$, non-negative) come from a graph's edge weights or are given as a matrix. Labels
 * are integers $0, \dots, C - 1$, with $-1$ for an unlabelled node, and $\Ymat$ is their one-hot $n \times C$ matrix.
 * Scores are returned with each row normalised to sum to one (rows with no mass stay zero), as scikit-learn's
 * `label_distributions_`.
 */

import type { MatrixLike, Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { solve, squaredDistances } from 'aifn-compute/numerics/linalg'
import { weightOf, type Graph } from '../graph'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'

/**
 * Edge weights as affinities: a graph (its weighted adjacency matrix, as `graphAffinity` with its default kernel), or
 * an $n \times n$ matrix $\Wmat$ of non-negative entries. Symmetry is expected but not checked.
 */
export type Affinity = Graph | MatrixLike

/** How {@link graphAffinity} turns edge weights into affinities. */
export type AffinityKernel =
  /** The edge weights as they are (1 where unset). */
  | { kind: 'weights' }
  /** 1 for every edge, whatever its weight. */
  | { kind: 'connectivity' }
  /**
   * $\exp(-w^2/(2\sigma^2))$ of an edge whose weight $w$ is a distance (a $k$-NN or $\varepsilon$-ball graph), with
   * `sigma` the bandwidth $\sigma > 0$.
   */
  | { kind: 'heat'; sigma: number }

/**
 * The $n \times n$ affinity matrix of a graph: its edge weights as they are, 1 per edge, or a heat kernel of
 * distances. Entry $(i, j)$ sums the mapped weights of the edges $i \to j$, and an undirected edge fills both $(i, j)$
 * and $(j, i)$, as `adjacencyMatrix` does. Throws `DomainError` for a heat kernel without $\sigma > 0$.
 *
 * @param g The graph, with $n$ nodes.
 * @param kernel How each edge weight becomes an affinity (default `{ kind: 'weights' }`, the weights themselves).
 * @returns The $n \times n$ affinity matrix, zero where there is no edge.
 *
 * @example Weights, connectivity and a heat kernel
 * // The path 0 - 1 - 2, whose edge weights are distances 1 and 2.
 * const edges = [{ from: 0, to: 1, weight: 1 }, { from: 1, to: 2, weight: 2 }]
 * const g = { kind: 'graph', nodes: 3, edges, directed: false }
 * print('weights:', graphAffinity(g))
 * print('connectivity:', graphAffinity(g, { kind: 'connectivity' }))
 * print('heat, sigma = 1:', graphAffinity(g, { kind: 'heat', sigma: 1 }))
 */
export function graphAffinity(g: Graph, kernel: AffinityKernel = { kind: 'weights' }): Tensor {
  if (kernel.kind === 'weights') return adjacencyMatrix(g, { weighted: true })
  if (kernel.kind === 'heat' && !(kernel.sigma > 0))
    throw new DomainError('graphAffinity', 'graphAffinity: the heat kernel needs σ > 0')
  const map = (w: number) => (kernel.kind === 'connectivity' ? 1 : Math.exp(-(w * w) / (2 * kernel.sigma ** 2)))
  return adjacencyMatrix({ ...g, edges: g.edges.map((e) => ({ ...e, weight: map(weightOf(e)) })) }, { weighted: true })
}

/** Options of {@link pointAffinity}. */
export interface PointAffinityOptions {
  /** The Gaussian kernel's rate $\gamma > 0$ in $\exp(-\gamma \lVert \xvec_i - \xvec_j \rVert^2)$ (default 1). */
  gamma?: number
  /**
   * Keep only the affinities to each point's $k$ nearest neighbours (at most $n - 1$), symmetrised ($i$ and $j$ keep
   * theirs when either is among the other's $k$ nearest); 0 or absent keeps the full graph.
   */
  neighbours?: Size
}

/**
 * The affinity matrix of a point cloud $\Xmat$ ($n \times d$):
 * $W_{ij} = \exp(-\gamma \lVert \xvec_i - \xvec_j \rVert^2)$ for $i \ne j$ and $W_{ii} = 0$ (Zhou et al. 2004; the
 * label-proportion propagation of Poyiadzi et al. 2018 uses this form), optionally restricted to a symmetrised
 * $k$-NN graph (ties to the smaller index). Throws `DomainError` unless $\gamma > 0$.
 *
 * @param x The points $\Xmat$, one per row: an $n \times d$ tensor or array of rows.
 * @param options The kernel's rate $\gamma$ (default 1) and the number of neighbours to keep (default all).
 * @returns The symmetric $n \times n$ affinity matrix $\Wmat$.
 *
 * @example Three points on a line, in full and on a 1-NN graph
 * // Points 0, 1 and 3: the pair 0, 3 is the furthest apart, and neither is the other's nearest.
 * const x = [[0], [1], [3]]
 * print('full:', pointAffinity(x))
 * print('1-NN:', pointAffinity(x, { neighbours: 1 }))
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

/**
 * The affinity matrix as a fresh row-major array, from a graph (its weighted adjacency matrix) or a matrix. Throws
 * `ShapeError` when it is not square and `DomainError` on a negative or NaN entry.
 *
 * @param w The graph or matrix of affinities.
 * @param where The caller's name, for error messages.
 * @returns The number of nodes $n$ and a copy of $\Wmat$, $n^2$ values that the caller may overwrite.
 */
function affinityOf(w: Affinity, where: string): { n: number; W: Float64Array } {
  const isGraph = !isTensor(w) && !Array.isArray(w) && (w as Graph).kind === 'graph'
  const t = isGraph ? adjacencyMatrix(w as Graph, { weighted: true }) : (w as MatrixLike)
  const { data, m, n } = dense.toMatrixF64(t, where)
  if (m !== n) throw new ShapeError(where, `${where}: the affinity must be square, got ${m} × ${n}`)
  for (const v of data) if (!(v >= 0)) throw new DomainError(where, `${where}: affinities must be non-negative`)
  return { n, W: Float64Array.from(data) }
}

/**
 * Labels as a one-hot $n \times C$ matrix $\Ymat$, with zero rows for unlabelled nodes ($-1$). Throws `DomainError`
 * on a label that is not $-1$ or an integer in $0, \dots, C - 1$.
 *
 * @param labels The class of each node, $0, \dots, C - 1$, or $-1$ for unlabelled.
 * @param classes The number of classes $C$ (default: the largest label plus 1).
 * @returns The $n \times C$ matrix with a 1 in column $c$ of each row labelled $c$.
 *
 * @example One-hot rows, and a zero row for the unlabelled node
 * print('Y =', labelMatrix([0, -1, 1]))
 * print('with 3 classes:', labelMatrix([0, -1, 1], 3))
 */
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

/** Class scores and the predicted class of every node. */
export interface LabelScores {
  /** The scores, $n \times C$, each row summing to one (or zero, for a node no label reached). */
  readonly scores: Tensor
  /** The predicted class of each node, its highest score (ties to the lower class); int32, $-1$ where a row is zero. */
  readonly labels: Tensor
}

/**
 * Scores with each row divided by its sum, and each row's predicted class.
 *
 * @param F The raw scores, row-major $n \times C$; read, not modified.
 * @param n The number of rows (nodes).
 * @param C The number of columns (classes).
 * @returns The normalised scores (rows summing to zero left as they are) and the class of each row's largest score
 *   ($-1$ for a zero row).
 */
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

/**
 * The common start of the label-propagation functions: the affinities and the one-hot labels, checked against each
 * other. Throws `ShapeError` when there is not one label per node and `DomainError` when no node is labelled.
 *
 * @param w The graph or matrix of affinities, as `affinityOf` reads it.
 * @param labels The class of each node, or $-1$ for unlabelled.
 * @param classes The number of classes $C$, or undefined for the largest label plus 1.
 * @param where The caller's name, for error messages.
 * @returns $n$; $\Wmat$ and $\Ymat$ as row-major arrays; $C$; and `known`, whether each node is labelled.
 */
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
  /** The number of classes $C$ (default: the largest label plus 1). */
  classes?: Size
}

/**
 * The harmonic solution $\Fmat_U = (\Dmat_{UU} - \Wmat_{UU})^{-1} \Wmat_{UL} \Ymat_L$ with the labelled rows fixed to
 * their labels (module notes), solved directly; self-loops are left out of $\Dmat$ and $\Wmat$. An unlabelled node
 * with no path to a labelled one (a component without labels) would make the system singular; such nodes keep zero
 * scores and the label $-1$, as the iteration leaves them.
 *
 * @param w The affinities: a graph (its edge weights) or an $n \times n$ non-negative matrix. Row $i$ gives the
 *   weights node $i$ averages its neighbours with.
 * @param labels The class of each node, $0, \dots, C - 1$, or $-1$ for unlabelled; at least one must be labelled.
 * @param options The number of classes $C$ (default: the largest label plus 1).
 * @returns The scores, row-normalised (the labelled rows are their one-hot labels), and the predicted classes.
 *
 * @example A path with one label at each end
 * // 0 - 1 - 2 - 3: node 1 is twice as close to class 0 as to class 1.
 * const W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
 * const { scores, labels } = harmonicLabels(W, [0, -1, -1, 1])
 * print('scores:', scores)
 * print('labels:', labels)
 *
 * @example A component with no label is left unlabelled
 * // Nodes 0 - 1 hold a label; nodes 2 - 3 are joined only to each other.
 * const W = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
 * print('labels:', harmonicLabels(W, [1, -1, -1, -1], { classes: 2 }).labels)
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
 * The spreading matrix $\Smat = \Dmat^{-1/2} \Wmat \Dmat^{-1/2}$ of an affinity with its diagonal (self-loops)
 * removed, $\Dmat$ the diagonal matrix of row sums (degrees) without self-loops; a node of degree 0 has a zero row
 * and column.
 *
 * @param w The affinities: a graph (its edge weights) or an $n \times n$ non-negative matrix.
 * @returns $\Smat$, $n \times n$, with $S_{ij} = W_{ij} / \sqrt{d_i d_j}$.
 *
 * @example The path 0 - 1 - 2
 * // Degrees 1, 2, 1, so each edge's entry is 1 / sqrt(2).
 * print('S =', spreadingMatrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
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
 * The random-walk matrix $\Smat = \Dmat^{-1}\Wmat$ of an affinity with its diagonal removed: row $i$ holds the
 * probabilities of a walker at node $i$ stepping to each neighbour, $W_{ij} / \sum_j W_{ij}$. A node of degree 0 has
 * a zero row. Label propagation through linear neighbourhoods (Wang and Zhang 2008) and its label-proportion form
 * (Poyiadzi et al. 2018) use this $\Smat$, where label spreading uses the symmetric
 * $\Dmat^{-1/2}\Wmat\Dmat^{-1/2}$ ({@link spreadingMatrix}).
 *
 * @param w The affinities: a graph (its edge weights) or an $n \times n$ non-negative matrix.
 * @returns $\Smat$, $n \times n$, each row summing to one (or zero for an isolated node).
 *
 * @example The path 0 - 1 - 2
 * print('S =', randomWalkMatrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
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
 * The resolvent of spreading, $\Rmat = (1 - \alpha)(\Imat - \alpha\Smat)^{-1} = (1 - \alpha) \sum_k (\alpha\Smat)^k$,
 * for an $n \times n$ propagation matrix $\Smat$ and $\alpha \in (0, 1)$: $\Fmat^* = \Rmat\Ymat$ is the limit of
 * $\Fmat \leftarrow \alpha\Smat\Fmat + (1 - \alpha)\Ymat$. With `scaled: false` the $(1 - \alpha)$ factor is left
 * out, $(\Imat - \alpha\Smat)^{-1}$. When $\Smat$ is row-stochastic (a random-walk matrix), the scaled $\Rmat$ is
 * too, so $\Rmat\Ymat$ averages $\Ymat$ over walks whose length is geometric with parameter $1 - \alpha$. Throws
 * `DomainError` for $\alpha$ outside $(0, 1)$ and `ShapeError` for a non-square $\Smat$.
 *
 * @param S The propagation matrix $\Smat$, $n \times n$: `spreadingMatrix` or `randomWalkMatrix` of an affinity.
 * @param alpha The weight $\alpha \in (0, 1)$ of the graph against the given labels.
 * @param options Whether to include the factor $(1 - \alpha)$.
 * @param options.scaled True (default) for $(1 - \alpha)(\Imat - \alpha\Smat)^{-1}$, false for
 *   $(\Imat - \alpha\Smat)^{-1}$.
 * @returns $\Rmat$, $n \times n$.
 *
 * @example A random-walk resolvent is row-stochastic
 * const R = spreadingResolvent(randomWalkMatrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), 0.5)
 * print('R =', R)
 * print('row sums:', matmul(R, tensor([1, 1, 1])))
 *
 * @example R Y is label spreading before normalisation
 * const W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
 * const R = spreadingResolvent(spreadingMatrix(W), 0.2)
 * print('R Y, normalised:', normaliseScores(matmul(R, labelMatrix([0, -1, -1, 1]))).scores)
 * print('labelSpreading:', labelSpreading(W, [0, -1, -1, 1]).scores)
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
  /** The weight $\alpha \in (0, 1)$ of the graph against the given labels (default 0.2, as scikit-learn). */
  alpha?: number
}

/**
 * Throws `DomainError` unless $0 < \alpha < 1$.
 *
 * @param alpha The spreading weight $\alpha$ to check.
 * @param where The caller's name, for error messages.
 */
function checkAlpha(alpha: number, where: string): void {
  if (!(alpha > 0 && alpha < 1)) throw new DomainError(where, `${where}: α must lie in (0, 1)`)
}

/**
 * The limit of label spreading, $\Fmat^* = (1 - \alpha)(\Imat - \alpha\Smat)^{-1}\Ymat$ (module notes), solved
 * directly and row-normalised, with $\Smat$ the `spreadingMatrix` of the affinities. Labelled rows are not clamped,
 * so a given label can be revised. Throws `DomainError` for $\alpha$ outside $(0, 1)$.
 *
 * @param w The affinities: a graph (its edge weights) or an $n \times n$ non-negative matrix.
 * @param labels The class of each node, $0, \dots, C - 1$, or $-1$ for unlabelled; at least one must be labelled.
 * @param options The weight $\alpha$ (default 0.2) and the number of classes $C$ (default: the largest label plus 1).
 * @returns The scores, row-normalised, and the predicted classes.
 *
 * @example A path with one label at each end
 * const W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
 * const { scores, labels } = labelSpreading(W, [0, -1, -1, 1])
 * print('scores:', scores)
 * print('labels:', labels)
 *
 * @example A larger alpha lets the graph overrule a label
 * // A triangle 0, 1, 2 of class 0 with a lone node 3 labelled 1 hanging off node 2.
 * const W = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
 * print('alpha = 0.2:', labelSpreading(W, [0, 0, 0, 1], { alpha: 0.2 }).labels)
 * print('alpha = 0.95:', labelSpreading(W, [0, 0, 0, 1], { alpha: 0.95 }).labels)
 */
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
  /** The raw scores $\Fmat$ ($n \times C$) after $t$ iterations (before the final row normalisation). */
  readonly scores: Tensor
  /** The predicted class per node (int32; $-1$ where a row is still zero). */
  readonly labels: Tensor
  /**
   * $\sum_{ic} \lvert (F_t)_{ic} - (F_{t-1})_{ic} \rvert$, scikit-learn's convergence measure (Infinity at $t = 0$).
   */
  readonly change: number
}

/** Options of the stepped forms: the stopping tolerance, with the number of classes. */
export interface LabelStepsOptions extends LabelOptions {
  /** Stop (`converged`) once the change falls below this (default 1e-3, as scikit-learn). */
  tolerance?: number
}

/**
 * A label iteration as a traceable algorithm: it starts from $\Ymat$, applies `update` once per step, and reports the
 * change and convergence.
 *
 * @param name The algorithm's name.
 * @param n The number of nodes.
 * @param C The number of classes.
 * @param Y The one-hot labels $\Ymat$, row-major $n \times C$: the scores at step 0 (copied).
 * @param update One iteration: the next scores from the current ones (row-major $n \times C$; must not modify its
 *   argument).
 * @param tolerance The change below which a state is `converged`.
 * @returns The algorithm, whose `init` ignores its argument.
 */
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
 * Label propagation as steps (Zhu and Ghahramani 2002): $\Fmat \leftarrow \Dmat^{-1}\Wmat\Fmat$, each unlabelled
 * row normalised, labelled rows reset to their labels; step 0 holds only the given labels. Self-loops stay in $\Wmat$
 * and $\Dmat$ here, which does not change the limit, {@link harmonicLabels}. The runners stop when the change falls
 * below the tolerance.
 *
 * @param w The affinities: a graph (its edge weights) or an $n \times n$ non-negative matrix.
 * @param labels The class of each node, $0, \dots, C - 1$, or $-1$ for unlabelled; at least one must be labelled.
 * @param options The tolerance (default 1e-3) and the number of classes $C$ (default: the largest label plus 1).
 * @returns The algorithm, whose `init` ignores its argument (run it with `undefined`); `normaliseScores` turns a
 *   state's `scores` into the final ones.
 *
 * @example Step by step towards the harmonic solution
 * const W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
 * const alg = labelPropagationSteps(W, [0, -1, -1, 1])
 * for (const t of [0, 1, 2]) print(`step ${t}:`, run(alg, undefined, t).scores)
 * const last = run(alg, undefined, 200)
 * print('converged after', last.t, 'steps:', normaliseScores(last.scores).scores)
 * print('harmonic:', harmonicLabels(W, [0, -1, -1, 1]).scores)
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

/**
 * Label spreading as steps (Zhou et al. 2004): $\Fmat \leftarrow \alpha\Smat\Fmat + (1 - \alpha)\Ymat$ from
 * $\Fmat_0 = \Ymat$, with $\Smat$ the `spreadingMatrix` of the affinities. The limit is {@link labelSpreading}. The
 * runners stop when the change falls below the tolerance. Throws `DomainError` for $\alpha$ outside $(0, 1)$.
 *
 * @param w The affinities: a graph (its edge weights) or an $n \times n$ non-negative matrix.
 * @param labels The class of each node, $0, \dots, C - 1$, or $-1$ for unlabelled; at least one must be labelled.
 * @param options The weight $\alpha$ (default 0.2), the tolerance (default 1e-3) and the number of classes $C$
 *   (default: the largest label plus 1).
 * @returns The algorithm, whose `init` ignores its argument (run it with `undefined`); `normaliseScores` turns a
 *   state's `scores` into the final ones.
 *
 * @example Run to convergence and compare with the closed form
 * const W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
 * const last = run(labelSpreadingSteps(W, [0, -1, -1, 1], { tolerance: 1e-9 }), undefined, 200)
 * print('steps:', last.t)
 * print('stepped:', normaliseScores(last.scores).scores)
 * print('closed form:', labelSpreading(W, [0, -1, -1, 1]).scores)
 */
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

/**
 * Row-normalised scores and predicted classes of raw scores (the last step of the stepped forms). A row summing to
 * zero stays zero, with class $-1$.
 *
 * @param scores Raw non-negative scores, an $n \times C$ tensor such as a `LabelPropagationState`'s `scores`.
 * @returns The scores with each row divided by its sum, and each row's class of largest score (ties to the lower).
 *
 * @example Normalise three rows, one of them empty
 * const { scores, labels } = normaliseScores(tensor([[2, 2, 4], [0, 0, 0], [1, 3, 0]]))
 * print('scores:', scores)
 * print('labels:', labels)
 */
export function normaliseScores(scores: Tensor): LabelScores {
  const [n, C] = scores.shape
  return normalised(dense.data(scores), n, C)
}
