/**
 * Weight-free propagation stacks, which isolate what message passing alone does with depth: oversmoothing and its
 * remedies, and the error that neighbour sampling adds.
 *
 * `propagationStack` applies $L$ layers to node features $\Xmat$ and returns every layer: `plain`
 * $\hat\Amat\Hmat$; `residual` $\tfrac12(\Hmat + \hat\Amat\Hmat)$; `initial-residual`
 * $(1 - \alpha)\hat\Amat\Hmat + \alpha\Xmat$, the propagation of APPNP and of GCNII with identity weights (Chen et al.
 * 2020, "Simple and deep graph convolutional networks", ICML, eq. 5); `pairnorm`, PairNorm after each propagation
 * (Zhao and Akoglu 2020); or `dropedge`, $\hat\Amat_\ell\Hmat$ with $\hat\Amat_\ell$ the matrix of a fresh random
 * subgraph at each layer (Rong et al. 2020). $\hat\Amat$ is the GCN matrix
 * $\tilde\Dmat^{-1/2}(\Amat + \Imat)\tilde\Dmat^{-1/2}$ (`gcnMatrix` of `aifn-compute/nn/graph`). With no weights or
 * nonlinearities, the stacks show the smoothing that `dirichletEnergy` measures.
 *
 * `meanPropagation` runs $L$ rounds of $\hvec'_v = \tfrac12\big(\hvec_v + \mathrm{mean}_{u \to v}\hvec_u\big)$ on the
 * full graph or on GraphSAGE's per-depth neighbour samples (`sampledComputationGraph`), so the sampled output at the
 * targets can be compared with the exact one.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { add, dense, mul, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { propagate } from 'aifn-compute/graph/propagation'
import { dropEdge, gcnMatrix, pairNorm } from 'aifn-compute/nn/graph'

/** The propagation rule of one weight-free layer. */
export type PropagationVariant = 'plain' | 'residual' | 'initial-residual' | 'pairnorm' | 'dropedge'

/** Options of `propagationStack`. */
export interface PropagationOptions {
  /** The number of layers $L$. */
  depth: number
  /** The rule (default `plain`). */
  variant?: PropagationVariant
  /** Initial residual: the weight $\alpha \in [0, 1]$ on $\Xmat$ (default 0.1). */
  alpha?: number
  /** PairNorm: the target root-mean-square row norm $s$ (default 1). */
  scale?: number
  /** DropEdge: the drop probability (default 0.2). */
  dropProbability?: number
  /** DropEdge: the random stream; layer $\ell$ thins the graph with `child(stream, l)`. Required for `dropedge`. */
  stream?: Stream
}

/** Rows of numbers, one per node. */
type Rows = number[][]

/**
 * The product $\Mmat\Hmat$ of a dense $V \times V$ matrix and $V$ rows, skipping zero entries.
 *
 * @param m The matrix, row-major.
 * @param h The rows of $\Hmat$.
 * @returns The rows of the product.
 */
function apply(m: ArrayLike<number>, h: Rows): Rows {
  const V = h.length
  const F = h[0]?.length ?? 0
  return h.map((_, i) => {
    const out = new Array<number>(F).fill(0)
    for (let j = 0; j < V; j++) {
      const w = m[i * V + j]
      if (w !== 0) for (let k = 0; k < F; k++) out[k] += w * h[j][k]
    }
    return out
  })
}

/**
 * Every layer of a weight-free propagation stack, $\Hmat^{(0)} = \Xmat, \dots, \Hmat^{(L)}$, as rows (see the file
 * comment). Throws `DomainError` for `dropedge` without a stream or a depth that is not a count.
 *
 * @param g The undirected graph, with $V$ nodes.
 * @param x The input features $\Xmat$, $V \times F$ (a tensor or rows).
 * @param options The depth, the rule and its parameter.
 * @returns $L + 1$ arrays of $V$ rows, layer 0 the input.
 *
 * @example Features on two triangles joined by an edge flatten with depth
 * import { fromEdges } from 'aifn-compute/graph'
 * import { dirichletEnergy } from 'aifn-compute/nn/graph'
 * const g = fromEdges(6, [[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [2, 3]], { directed: false })
 * const layers = propagationStack(g, [[3], [1], [2], [0], [1], [-1]], { depth: 4 })
 * print('energy by layer:', layers.map((h) => dirichletEnergy(g, h)))
 */
export function propagationStack(g: Graph, x: MatrixLike, options: PropagationOptions): Rows[] {
  const { depth, variant = 'plain', alpha = 0.1, scale = 1, dropProbability: p = 0.2 } = options
  if (!(Number.isInteger(depth) && depth >= 0))
    throw new DomainError('propagationStack', `propagationStack: depth = ${depth} must be a count`)
  if (variant === 'dropedge' && !options.stream)
    throw new DomainError('propagationStack', 'propagationStack: dropedge needs a stream')
  const { data, m: V, n: F } = dense.toMatrixF64(x, 'propagationStack')
  const x0: Rows = Array.from({ length: V }, (_, i) => Array.from(data.subarray(i * F, i * F + F)))
  const fixed = gcnMatrix(g).data
  const layers: Rows[] = [x0]
  for (let l = 1; l <= depth; l++) {
    const h = layers[l - 1]
    let next: Rows
    if (variant === 'dropedge') next = apply(gcnMatrix(dropEdge(child(options.stream!, l), g, p)).data, h)
    else {
      const ah = apply(fixed, h)
      if (variant === 'plain') next = ah
      else if (variant === 'residual') next = ah.map((row, i) => row.map((v, k) => 0.5 * (v + h[i][k])))
      else if (variant === 'initial-residual')
        next = ah.map((row, i) => row.map((v, k) => (1 - alpha) * v + alpha * x0[i][k]))
      else next = toRows(pairNorm(dense.mat(Float64Array.from(ah.flat()), V, F), scale) as Tensor) as Rows
    }
    layers.push(next)
  }
  return layers
}

/**
 * $L$ rounds of $\hvec'_v = \tfrac12(\hvec_v + \mathrm{mean}_{u \to v}\hvec_u)$, a fixed linear layer with no
 * weights. With `samples`, the round that produces depth-$k$ vectors aggregates over the depth-$(k + 1)$ sample, as in
 * GraphSAGE: the last round, which produces the targets' outputs, uses `samples[0]`. Differentiable in `h`. Throws
 * `DomainError` when `samples` does not hold one graph per round.
 *
 * @param g The full graph (used when `samples` is absent).
 * @param h The node features, $V \times F$.
 * @param layers The number of rounds $L$.
 * @param samples The sampled graph of each depth (from `sampledComputationGraph`), $L$ of them, or none.
 * @returns The features after $L$ rounds, $V \times F$ (exact only at the targets when sampled).
 *
 * @example One round on a triangle with a tail
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(4, [[0, 1], [0, 2], [1, 2], [2, 3]], { directed: false })
 * print(meanPropagation(g, tensor([[1], [2], [3], [4]]), 1))
 */
export function meanPropagation(g: Graph, h: Value, layers: Size, samples?: readonly Graph[]): Value {
  if (samples && samples.length !== layers)
    throw new DomainError('meanPropagation', `meanPropagation: ${samples.length} samples for ${layers} rounds`)
  let x = h
  for (let round = 1; round <= layers; round++) {
    const graph = samples ? samples[layers - round] : g
    x = mul(0.5, add(x, propagate(graph, x, { aggregate: 'mean' })))
  }
  return x
}
