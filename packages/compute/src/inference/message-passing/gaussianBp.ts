/**
 * Gaussian belief propagation on a pairwise Gaussian Markov random field in information form,
 * $p(\xvec) \propto \exp(-\tfrac{1}{2} \xvec^\top \Jmat \xvec + \hvec^\top \xvec)$, with scalar variables (Weiss &
 * Freeman 2001, "Correctness of belief propagation in Gaussian graphical models of arbitrary topology", Neural
 * Computation 13(10); Bickson 2008, "Gaussian belief propagation: theory and application", §2).
 *
 * The message from $i$ to $j$ is a Gaussian in $x_j$ with precision $\Lambda_{i \to j}$ and potential
 * $\eta_{i \to j}$. With the cavity precision
 * $\hat{\Lambda} = J_{ii} + \sum_{k \in N(i) \setminus j} \Lambda_{k \to i}$ and potential
 * $\hat{\eta} = h_i + \sum_{k \in N(i) \setminus j} \eta_{k \to i}$, the update is
 * $\Lambda_{i \to j} = -J_{ij}^2 / \hat{\Lambda}$ and $\eta_{i \to j} = -J_{ij} \hat{\eta} / \hat{\Lambda}$. The
 * marginal of $x_i$ has precision $J_{ii} + \sum_k \Lambda_{k \to i}$ and potential $h_i + \sum_k \eta_{k \to i}$.
 * At a fixed point the means are exact on any graph; the variances are exact on trees. Convergence is guaranteed when
 * the model is walk-summable (Malioutov, Johnson & Willsky 2006), as a diagonally dominant $\Jmat$ is.
 */

import type { MatrixLike, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { adjacency, fromEdges, type Graph } from 'aifn-compute/graph'
import { fromData, isTensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of {@link gaussianBeliefPropagationSteps}. */
export interface GaussianBpOptions {
  /**
   * `'flooding'` (default; every message from those of the previous sweep) or `'sequential'` (in edge order, each
   * from the newest messages).
   */
  schedule?: 'flooding' | 'sequential'
  /** Weight of the old message in each update, in $[0, 1)$. Default 0; outside $[0, 1)$ throws `DomainError`. */
  damping?: number
  /** A sweep that moves no message precision or potential by more than this has converged. Default 1e-10. */
  tolerance?: number
}

/**
 * The state of Gaussian BP; messages are indexed by directed edge (`from`, `to`). `t` counts sweeps; `diverged` means
 * a non-positive cavity precision $\hat{\Lambda}$ appeared (the model is not walk-summable here) or a message became
 * non-finite, and the run stops.
 */
export interface GaussianBpState extends Status {
  /** The number of variables. */
  n: Size
  /** The precision matrix $\Jmat$, as rows. */
  J: number[][]
  /** The potential vector $\hvec$. */
  h: number[]
  /** The undirected graph of the non-zero off-diagonal entries of $\Jmat$. */
  graph: Graph
  /** Directed edges: message $k$ travels from `from[k]` to `to[k]`; the two directions of an edge are adjacent. */
  from: Int32Array
  /** The receiving variable of each directed edge. */
  to: Int32Array
  /** The message schedule. */
  schedule: 'flooding' | 'sequential'
  /** The weight of the old message in each update. */
  damping: number
  /** The largest message change of a converged sweep. */
  tolerance: number
  /** $\Lambda_{i \to j}$ per directed edge. */
  messagePrecision: Tensor
  /** $\eta_{i \to j}$ per directed edge. */
  messageShift: Tensor
  /** Marginal means from the current messages (length $n$). */
  means: Tensor
  /** Marginal variances from the current messages (length $n$). */
  variances: Tensor
  /** The largest change of a message precision or potential in the last sweep ($\infty$ before the first). */
  change: number
  /** Whether the last sweep changed no message by more than `tolerance`. */
  converged: boolean
  /** Whether a cavity precision was not positive or a change was not finite. */
  diverged: boolean
}

/**
 * The marginal means and variances from a set of messages: precision $J_{ii} + \sum_k \Lambda_{k \to i}$ and potential
 * $h_i + \sum_k \eta_{k \to i}$ for each variable $i$.
 *
 * @param s The size, $\Jmat$, $\hvec$ and the receiving variable of each directed edge.
 * @param lambda The message precisions $\Lambda$, one per directed edge.
 * @param eta The message potentials $\eta$, one per directed edge.
 * @returns The means (potential over precision) and variances (one over precision), each of length $n$.
 */
function marginals(s: Pick<GaussianBpState, 'n' | 'J' | 'h' | 'to'>, lambda: Float64Array, eta: Float64Array) {
  const P = s.J.map((row, i) => row[i])
  const H = [...s.h]
  for (let k = 0; k < s.to.length; k++) {
    P[s.to[k]] += lambda[k]
    H[s.to[k]] += eta[k]
  }
  return {
    means: fromData(
      Float64Array.from(H, (hi, i) => hi / P[i]),
      [s.n],
    ),
    variances: fromData(
      Float64Array.from(P, (p) => 1 / p),
      [s.n],
    ),
  }
}

/**
 * Gaussian BP as a traceable algorithm on the precision $\Jmat$ ($n \times n$ symmetric; its off-diagonal non-zeros
 * are the graph's edges) and the potential $\hvec = \Jmat \muvec$ (length $n$); each step is one sweep over every
 * directed edge. Messages start at zero ($\Lambda = \eta = 0$), so the first means are $h_i / J_{ii}$. The run
 * converges when no message moves by more than `tolerance`, and stops as `diverged` when a cavity precision
 * $\hat{\Lambda}$ is not positive or a message is not finite. Throws `DomainError` at `init` when `damping` is not in
 * $[0, 1)$.
 *
 * @param precision The precision matrix $\Jmat$ ($n \times n$, symmetric, as a tensor or rows); the edges are the
 *   non-zero entries above the diagonal.
 * @param shift The potential $\hvec$ (length $n$), so that the mean $\muvec$ solves $\Jmat \muvec = \hvec$.
 * @param o The schedule, damping and tolerance (see `GaussianBpOptions`).
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; `t` counts sweeps.
 *
 * @example A chain of three is exact after two sweeps
 * // J is tridiagonal and h = J · [1, 1, 1], so the means are all 1.
 * const J = tensor([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
 * const alg = gaussianBeliefPropagationSteps(J, tensor([1, 0, 1]))
 * for (const sweeps of [0, 1, 2, 3]) print(`after ${sweeps} sweeps: means`, run(alg, undefined, sweeps).means)
 * const s = run(alg, undefined, 100)
 * print('converged after', s.t, 'sweeps:', s.converged)
 *
 * @example A model that is not walk-summable diverges
 * // Positive definite (eigenvalues 0.1, 0.1 and 2.8), but the partial correlations are too strong for BP.
 * const J = tensor([[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]])
 * const s = run(gaussianBeliefPropagationSteps(J, tensor([1, 1, 1])), undefined, 100)
 * print('sweeps =', s.t, 'diverged =', s.diverged)
 */
export function gaussianBeliefPropagationSteps(
  precision: MatrixLike,
  shift: VectorLike,
  o: GaussianBpOptions = {},
): Algorithm<void, GaussianBpState> {
  return {
    name: 'gaussian-belief-propagation',
    init: () => {
      const damping = o.damping ?? 0
      if (!(damping >= 0 && damping < 1))
        throw new DomainError('gaussianBeliefPropagation', 'gaussianBeliefPropagation: damping must be in [0, 1)')
      const J = isTensor(precision) ? toRows(precision) : Array.from(precision, (r) => Array.from(r))
      const h = isTensor(shift) ? toFlat(shift) : Array.from(shift)
      const n = J.length
      const pairs: [number, number][] = []
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (J[i][j] !== 0) pairs.push([i, j])
      const graph = fromEdges(n, pairs, { directed: false })
      const from = new Int32Array(2 * pairs.length)
      const to = new Int32Array(2 * pairs.length)
      pairs.forEach(([i, j], k) => {
        from[2 * k] = i
        to[2 * k] = j
        from[2 * k + 1] = j
        to[2 * k + 1] = i
      })
      const lambda = new Float64Array(from.length)
      const eta = new Float64Array(from.length)
      const base = { n, J, h, to }
      return {
        n,
        J,
        h,
        graph,
        from,
        to,
        schedule: o.schedule ?? 'flooding',
        damping,
        tolerance: o.tolerance ?? 1e-10,
        messagePrecision: fromData(lambda, [lambda.length]),
        messageShift: fromData(eta, [eta.length]),
        ...marginals(base, lambda, eta),
        t: 0,
        change: Infinity,
        converged: false,
        diverged: false,
      }
    },
    step: (s) => {
      const old = { lambda: Float64Array.from(s.messagePrecision.data), eta: Float64Array.from(s.messageShift.data) }
      const lambda = Float64Array.from(old.lambda)
      const eta = Float64Array.from(old.eta)
      // Incoming messages of each node, by directed-edge index.
      const incoming = adjacency(s.graph).map((_, i) => {
        const ks: number[] = []
        for (let k = 0; k < s.to.length; k++) if (s.to[k] === i) ks.push(k)
        return ks
      })
      const read = s.schedule === 'flooding' ? old : { lambda, eta }
      let change = 0
      let diverged = false
      for (let k = 0; k < s.from.length; k++) {
        const i = s.from[k]
        const j = s.to[k]
        let L = s.J[i][i]
        let E = s.h[i]
        for (const m of incoming[i]) {
          if (s.from[m] === j) continue
          L += read.lambda[m]
          E += read.eta[m]
        }
        if (!(L > 0)) diverged = true
        const nl = (-s.J[i][j] * s.J[i][j]) / L
        const ne = (-s.J[i][j] * E) / L
        const dl = (1 - s.damping) * nl + s.damping * old.lambda[k]
        const de = (1 - s.damping) * ne + s.damping * old.eta[k]
        change = Math.max(change, Math.abs(dl - old.lambda[k]), Math.abs(de - old.eta[k]))
        lambda[k] = dl
        eta[k] = de
      }
      return {
        ...s,
        messagePrecision: fromData(lambda, [lambda.length]),
        messageShift: fromData(eta, [eta.length]),
        ...marginals(s, lambda, eta),
        t: s.t + 1,
        change,
        converged: !diverged && change < s.tolerance,
        diverged: diverged || !Number.isFinite(change),
      }
    },
  }
}

/**
 * Run Gaussian BP for at most `maxSteps` sweeps (default 500): marginal means and variances, and the flags. On a tree
 * both are exact once converged; on a graph with loops the means are exact and the variances approximate. Throws
 * `DomainError` when `damping` is not in $[0, 1)$.
 *
 * @param precision The precision matrix $\Jmat$ ($n \times n$, symmetric, as a tensor or rows).
 * @param shift The potential $\hvec$ (length $n$), with $\Jmat \muvec = \hvec$.
 * @param options The schedule, damping and tolerance of `gaussianBeliefPropagationSteps`, and `maxSteps`, the most
 *   sweeps to run (default 500).
 * @returns The marginal means and variances, the sweeps run, and whether the run converged or diverged.
 *
 * @example Exact on a chain
 * // J is tridiagonal and h = J · [1, 1, 1]; the exact variances are the diagonal of J⁻¹.
 * const J = tensor([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
 * const r = gaussianBeliefPropagation(J, tensor([1, 0, 1]))
 * print('means =', r.means)
 * print('variances =', r.variances)
 * print('sweeps =', r.sweeps, 'converged =', r.converged)
 * const Jinv = div(tensor([[3, 2, 1], [2, 4, 2], [1, 2, 3]]), 4)
 * print('J J⁻¹ =', matmul(J, Jinv))
 * print('diagonal of J⁻¹ =', diagonal(Jinv))
 *
 * @example On a loop the means are exact, the variances not
 * // J = 0.7 I + 0.3 · 1 1ᵀ and h = J · [1, 1, 1]. By Sherman–Morrison each variance is (1 − 0.3 / 1.6) / 0.7.
 * const J = tensor([[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]])
 * const r = gaussianBeliefPropagation(J, tensor([1.6, 1.6, 1.6]))
 * print('means =', r.means)
 * print('BP variances =', r.variances)
 * print('exact variance =', (1 - 0.3 / 1.6) / 0.7)
 */
export function gaussianBeliefPropagation(
  precision: MatrixLike,
  shift: VectorLike,
  options: GaussianBpOptions & { maxSteps?: Size } = {},
): { means: Tensor; variances: Tensor; sweeps: Size; converged: boolean; diverged: boolean } {
  const { maxSteps = 500, ...rest } = options
  const s = run(gaussianBeliefPropagationSteps(precision, shift, rest), undefined, maxSteps)
  return { means: s.means, variances: s.variances, sweeps: s.t, converged: s.converged, diverged: s.diverged }
}
