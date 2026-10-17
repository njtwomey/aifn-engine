/**
 * Polynomial spectral filters on a graph in the Chebyshev basis, and the ChebNet layer built on them. A filter with
 * frequency response $g(\lambda) = \sum_{k=0}^{K} \theta_k T_k(2\lambda/\lambda_{\max} - 1)$ acts on a graph signal
 * as $\sum_k \theta_k T_k(\tilde\Lmat)\xvec$, with the rescaled Laplacian $\tilde\Lmat = 2\Lmat/\lambda_{\max} - \Imat$
 * (`rescaledLaplacian` of `aifn-compute/graph/matrices`), computed by the three-term recurrence
 *
 * $$T_k(\tilde\Lmat)\xvec = 2\tilde\Lmat\, T_{k-1}(\tilde\Lmat)\xvec - T_{k-2}(\tilde\Lmat)\xvec:$$
 *
 * $K$ products with $\tilde\Lmat$, no eigendecomposition, and a result exactly zero more than $K$ hops from the
 * support of $\xvec$ (Hammond, Vandergheynst and Gribonval 2011, "Wavelets on graphs via spectral graph theory", ACHA
 * 30(2), §6). The ChebNet layer $\sum_k T_k(\tilde\Lmat)\Hmat\Thetamat_k + \bvec$ is Defferrard, Bresson and
 * Vandergheynst (2016), "Convolutional neural networks on graphs with fast localized spectral filtering", NeurIPS,
 * §2.1. The coefficients of a target response come from `chebyshevInterpolation` and the response itself from
 * `chebyshevSeries` (`aifn-compute/numerics/interpolate`, on the domain $[0, \lambda_{\max}]$).
 *
 * The products are dense ($\tilde\Lmat$ is a $V \times V$ tensor): for graphs of up to a few hundred nodes. Every
 * function is differentiable in the signal and the weights.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child } from 'aifn-compute/foundation/random'
import { add, dense, matmul, mul, shapeOfValue, sub, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { rescaledLaplacian, type RescaleOptions } from 'aifn-compute/graph/matrices'
import { xavierUniform, zerosInit } from 'aifn-compute/nn/init'
import { tap, type Layer } from 'aifn-compute/nn/layers'

/**
 * Check that $\tilde\Lmat$ is square and the signal has one row per node. Throws `ShapeError` otherwise.
 *
 * @param lTilde The rescaled Laplacian.
 * @param x The signal.
 * @param where The caller's name, for error messages.
 */
function checkSignal(lTilde: Tensor, x: Value, where: string): void {
  const [n, m] = lTilde.shape
  const s = shapeOfValue(x)
  if (lTilde.shape.length !== 2 || n !== m || s[0] !== n)
    throw new ShapeError(where, `${where}: need an n × n Laplacian and a signal with n rows`, [lTilde.shape, s])
}

/**
 * The Chebyshev terms $T_0(\tilde\Lmat)\xvec, \dots, T_K(\tilde\Lmat)\xvec$ of a signal, by the recurrence; term $k$
 * is zero at every node more than $k$ hops from the support of $\xvec$. Differentiable in `x`. Throws `ShapeError`
 * for a signal without one row per node and `DomainError` for a negative order.
 *
 * @param lTilde The rescaled Laplacian $\tilde\Lmat$, $V \times V$.
 * @param x The signal: a $V$-vector or $V \times F$ features.
 * @param order The highest order $K$.
 * @returns The $K + 1$ terms, each shaped like `x`.
 *
 * @example An impulse at node 0 of a path spreads one hop per order
 * import { rescaledLaplacian } from 'aifn-compute/graph/matrices'
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * const { lTilde } = rescaledLaplacian(chainGraph(4))
 * for (const t of chebyshevTerms(lTilde, tensor([1, 0, 0, 0]), 2)) print(t)
 */
export function chebyshevTerms(lTilde: Tensor, x: Value, order: Size): Value[] {
  checkSignal(lTilde, x, 'chebyshevTerms')
  if (!(Number.isInteger(order) && order >= 0))
    throw new DomainError('chebyshevTerms', `chebyshevTerms: order = ${order} must be a count`)
  const terms: Value[] = [x]
  if (order >= 1) terms.push(matmul(lTilde, x))
  for (let k = 2; k <= order; k++) terms.push(sub(mul(2, matmul(lTilde, terms[k - 1])), terms[k - 2]))
  return terms
}

/**
 * The filtered signal $\sum_k \theta_k T_k(\tilde\Lmat)\xvec$: the filter whose response at eigenvalue $\lambda$ of
 * $\Lmat$ is `chebyshevSeries(theta, lambda, [0, lambdaMax])`. Differentiable in `x`. Throws as `chebyshevTerms`
 * does, and `DomainError` for no coefficients.
 *
 * @param lTilde The rescaled Laplacian $\tilde\Lmat$, $V \times V$.
 * @param x The signal: a $V$-vector or $V \times F$ features (each column filtered alike).
 * @param theta The coefficients $\theta_0, \dots, \theta_K$, e.g. from `chebyshevInterpolation`.
 * @returns The filtered signal, shaped like `x`.
 *
 * @example A low-pass filter fitted to $e^{-2\lambda}$ smooths an impulse
 * import { rescaledLaplacian } from 'aifn-compute/graph/matrices'
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * import { chebyshevInterpolation } from 'aifn-compute/numerics/interpolate'
 * const { lTilde, lambdaMax } = rescaledLaplacian(chainGraph(5))
 * const theta = chebyshevInterpolation((l) => Math.exp(-2 * l), 3, [0, lambdaMax])
 * print('filtered impulse:', chebyshevFilter(lTilde, tensor([0, 0, 1, 0, 0]), theta))
 */
export function chebyshevFilter(lTilde: Tensor, x: Value, theta: VectorLike): Value {
  const c = dense.toF64(theta, 'chebyshevFilter')
  if (c.length === 0) throw new DomainError('chebyshevFilter', 'chebyshevFilter: no coefficients')
  const terms = chebyshevTerms(lTilde, x, c.length - 1)
  let y: Value = mul(c[0], terms[0])
  for (let k = 1; k < terms.length; k++) y = add(y, mul(c[k], terms[k]))
  return y
}

/**
 * The ChebNet layer $\sum_{k=0}^{K} T_k(\tilde\Lmat)\Hmat\Thetamat_k + \bvec$ (Defferrard et al. 2016): $V \times F$
 * features to $V \times G$, with one $F \times G$ weight per order. Differentiable in `h`, `weights` and `bias`.
 * Throws as `chebyshevTerms` does, and `DomainError` for no weights.
 *
 * @param lTilde The rescaled Laplacian $\tilde\Lmat$, $V \times V$.
 * @param h The node features $\Hmat$, $V \times F$.
 * @param weights The weights $\Thetamat_0, \dots, \Thetamat_K$, each $F \times G$.
 * @param bias The bias $\bvec$, $G$ values added to every row (default none).
 * @returns The new features, $V \times G$.
 *
 * @example One weight per order on one channel is the scalar filter
 * import { rescaledLaplacian } from 'aifn-compute/graph/matrices'
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * const { lTilde } = rescaledLaplacian(chainGraph(3))
 * const h = tensor([[1], [0], [2]])
 * print('chebConv:', chebConv(lTilde, h, [tensor([[0.5]]), tensor([[-1]])]))
 * print('filter:', chebyshevFilter(lTilde, h, [0.5, -1]))
 */
export function chebConv(lTilde: Tensor, h: Value, weights: readonly Value[], bias?: Value): Value {
  if (weights.length === 0) throw new DomainError('chebConv', 'chebConv: needs at least one weight')
  const terms = chebyshevTerms(lTilde, h, weights.length - 1)
  let y: Value = matmul(terms[0], weights[0])
  for (let k = 1; k < weights.length; k++) y = add(y, matmul(terms[k], weights[k]))
  return bias === undefined ? y : add(y, bias)
}

/** Parameters of `ChebConv`: one weight per order and the bias. */
export type ChebConvParams = {
  /** $\Thetamat_0, \dots, \Thetamat_K$, each $F \times G$. */
  weights: Tensor[]
  /** The bias $\bvec$, $G$ values (absent with `bias: false`). */
  bias?: Tensor
}

/**
 * A ChebNet layer of order $K$ ($F \to G$) over a fixed graph, whose rescaled Laplacian is computed once; the weights
 * Glorot-uniform (each from its own child stream `'weight', k`) and the bias zero. `apply` is `chebConv`.
 *
 * @param g The undirected graph every call runs on.
 * @param inFeatures The input width $F$.
 * @param outFeatures The output width $G$.
 * @param order The highest order $K$ (the layer sees $K$ hops).
 * @param options The Laplacian and $\lambda_{\max}$ (as `rescaledLaplacian`), and `bias` (default true).
 * @returns The layer: `init` draws `ChebConvParams`, `apply` maps $V \times F$ features to $V \times G$.
 *
 * @example An order-2 layer on a path of four nodes
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * const layer = ChebConv(chainGraph(4), 2, 3, 2)
 * const p = layer.init(stream(0))
 * print(layer.label, ' weights:', p.weights.length)
 * print('output shape:', shapeOf(layer.apply(p, tensor([[1, 0], [0, 1], [1, 1], [0, 0]]))))
 */
export function ChebConv(
  g: Graph,
  inFeatures: Size,
  outFeatures: Size,
  order: Size,
  options: RescaleOptions & { bias?: boolean } = {},
): Layer<ChebConvParams> {
  if (!(Number.isInteger(order) && order >= 0))
    throw new DomainError('ChebConv', `ChebConv: order = ${order} must be a count`)
  const { bias = true, ...rescale } = options
  const { lTilde } = rescaledLaplacian(g, rescale)
  const fans = { fanIn: inFeatures, fanOut: outFeatures }
  return {
    kind: 'ChebConv',
    label: `ChebConv(${inFeatures} → ${outFeatures}, K = ${order})`,
    init: (s) => ({
      weights: Array.from({ length: order + 1 }, (_, k) =>
        xavierUniform()(child(s, 'weight', k), [inFeatures, outFeatures], fans),
      ),
      ...(bias ? { bias: zerosInit()(child(s, 'bias'), [outFeatures], fans) } : {}),
    }),
    apply: (p, x, ctx) => tap(ctx, chebConv(lTilde, x, p.weights, p.bias)),
  }
}
