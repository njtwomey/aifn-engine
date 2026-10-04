/**
 * Gaussian belief propagation on a pairwise Gaussian Markov random field in information form,
 * p(x) ∝ exp(−½ xᵀJx + hᵀx), with scalar variables (Weiss & Freeman 2001, "Correctness of belief propagation in
 * Gaussian graphical models of arbitrary topology", Neural Computation 13(10); Bickson 2008, "Gaussian belief
 * propagation: theory and application", §2).
 *
 * The message from i to j is a Gaussian in x_j with precision Λ_{i→j} and potential η_{i→j}:
 * Λ̂ = J_ii + Σ_{k∈N(i)\j} Λ_{k→i}, η̂ = h_i + Σ_{k∈N(i)\j} η_{k→i}, Λ_{i→j} = −J_ij²/Λ̂, η_{i→j} = −J_ij η̂/Λ̂.
 * At a fixed point the means are exact on any graph; the variances are exact on trees.
 */

import type { MatrixLike, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { adjacency, fromEdges, type Graph } from 'aifn-compute/graph'
import { fromData, isTensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'

/** Options of {@link gaussianBeliefPropagationSteps}. */
export interface GaussianBpOptions {
  /** `flooding` (all messages from the previous sweep) or `sequential` (in edge order, newest messages). */
  schedule?: 'flooding' | 'sequential'
  /** Weight of the old message, in [0, 1). Default 0. */
  damping?: number
  tolerance?: number
}

/**
 * The state of Gaussian BP; messages are indexed by directed edge (`from`, `to`). `t` counts sweeps; `diverged` means
 * a non-positive cavity precision Λ̂ appeared (the model is not walk-summable here) and the run stops.
 */
export interface GaussianBpState extends Status {
  n: Size
  J: number[][]
  h: number[]
  graph: Graph
  /** Directed edges: message k travels from[k] → to[k]. */
  from: Int32Array
  to: Int32Array
  schedule: 'flooding' | 'sequential'
  damping: number
  tolerance: number
  /** Λ_{i→j} and η_{i→j} per directed edge. */
  messagePrecision: Tensor
  messageShift: Tensor
  /** Marginal means and variances from the current messages. */
  means: Tensor
  variances: Tensor
  change: number
  converged: boolean
  diverged: boolean
}

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
 * Gaussian BP as a traceable algorithm on precision J (n × n symmetric; its off-diagonal non-zeros are the graph's
 * edges) and potential h = J μ (length n); each step is one sweep over every directed edge. Messages start at zero
 * (Λ = η = 0). The run converges when no message moves by more than `tolerance`, and stops as `diverged` when a
 * cavity precision Λ̂ is not positive.
 */
export function gaussianBeliefPropagationSteps(
  precision: MatrixLike,
  shift: VectorLike,
  o: GaussianBpOptions = {},
): Algorithm<void, GaussianBpState> {
  return {
    name: 'gaussian-belief-propagation',
    init: () => {
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
        damping: o.damping ?? 0,
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

/** Run Gaussian BP for at most `maxSteps` sweeps (default 500): marginal means and variances, and the flags. */
export function gaussianBeliefPropagation(
  precision: MatrixLike,
  shift: VectorLike,
  options: GaussianBpOptions & { maxSteps?: Size } = {},
): { means: Tensor; variances: Tensor; sweeps: Size; converged: boolean; diverged: boolean } {
  const { maxSteps = 500, ...rest } = options
  const s = run(gaussianBeliefPropagationSteps(precision, shift, rest), undefined, maxSteps)
  return { means: s.means, variances: s.variances, sweeps: s.t, converged: s.converged, diverged: s.diverged }
}
