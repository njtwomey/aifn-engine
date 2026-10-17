/**
 * The unstacked deep operator network of Lu et al. (2021, "Learning nonlinear operators via DeepONet based on the
 * universal approximation theorem of operators", Nature Machine Intelligence 3, §2 and Fig. 1d): a branch network maps
 * the values of the input function $u$ at $m$ fixed sensors to $p$ coefficients $b_k(u)$, a trunk network maps a query
 * point $y$ to $p$ basis values $t_k(y)$, and the prediction is $G(u)(y) \approx \sum_k b_k(u)\,t_k(y) + b_0$. With
 * one shared query set for all functions (the "aligned" data layout), the prediction for $N$ functions at $Q$ points
 * is the matrix product $\Bmat\Tmat^\top$ of the branch outputs ($N \times p$) and the trunk outputs ($Q \times p$),
 * so the trunk runs once per batch. The trunk's last layer has the activation too, as in the paper (the basis
 * functions are nonlinear features of $y$). Scale the query points to $[-1, 1]$ so that the trunk's activations start
 * out of saturation.
 *
 * The architecture is the paper's; the sizes are smaller, so a browser trains it in seconds (about 3.5 ms per Adam
 * step and 10 ms per L-BFGS step at the note's sizes). Train it with `methodTraining` of `aifn-compute/nn/training` on
 * `deepOnetLoss`; the antiderivative pairs are `antiderivativeData` of `aifn-methods/data/synthetic`.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  matmul,
  mean,
  square,
  sub,
  transpose,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Params } from 'aifn-compute/foundation/pytree'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'

/** Options of `deepOnet`. */
export type DeepOnetOptions = {
  /** The number of sensors $m$: the branch network's input width. */
  sensors: number
  /** The coordinates of a query point $y$ (default 1). */
  queryDimension?: number
  /** The latent width $p$: coefficients from the branch, basis functions from the trunk. */
  latent: number
  /** The hidden widths of the branch network (default `[40]`). */
  branchHidden?: readonly number[]
  /** The hidden widths of the trunk network (default `[40, 40]`). */
  trunkHidden?: readonly number[]
  /** The hidden activation of both networks (default `'tanh'`). */
  activation?: 'tanh' | 'relu' | 'gelu'
}

/** Parameters of a DeepONet: the two networks and the output bias $b_0$. */
export type DeepOnetParams = {
  /** The branch network's layers. */
  branch: Params[]
  /** The trunk network's layers. */
  trunk: Params[]
  /** The output bias $b_0$, a vector of one. */
  bias: Tensor
}

/** A DeepONet: `apply(p, u, y)` maps sensor values $[N, m]$ and query points $[Q, d]$ to $G(u)(y)$, $[N, Q]$. */
export type DeepOnet = {
  /** A short description, for captions. */
  readonly label: string
  /** Fresh parameters: the branch from the child stream `branch`, the trunk from `trunk`, the bias 0. */
  init(s: Stream): DeepOnetParams
  /** The coefficients $b(u)$, $[N, p]$. */
  coefficients(p: DeepOnetParams, u: Value): Value
  /** The basis $t(y)$, $[Q, p]$. */
  basis(p: DeepOnetParams, y: Value): Value
  /** The prediction $b(u)\,t(y)^\top + b_0$, $[N, Q]$. */
  apply(p: DeepOnetParams, u: Value, y: Value): Value
}

/**
 * An unstacked DeepONet with a branch network on the sensor values and a trunk network on the query point, both
 * `Mlp`s with Xavier-uniform weights. Differentiable in its parameters.
 *
 * @param options The sensors, query dimension, latent width $p$, hidden widths and activation.
 * @returns The network.
 *
 * @example The prediction is the coefficients times the basis, plus the bias
 * const net = deepOnet({ sensors: 5, latent: 3 })
 * const p = net.init(stream(0))
 * const u = tensor([[0, 0.5, 1, 0.5, 0], [1, 1, 1, 1, 1]])
 * const y = tensor([[-1], [0], [1]])
 * print('G(u)(y) =', net.apply(p, u, y))
 * print('b(u) t(y)T =', matmul(net.coefficients(p, u), transpose(net.basis(p, y))))
 */
export function deepOnet(options: DeepOnetOptions): DeepOnet {
  const { sensors, queryDimension = 1, latent, activation = 'tanh' } = options
  const branch = Mlp([sensors, ...(options.branchHidden ?? [40]), latent], { activation, init: xavierUniform() })
  const trunk = Mlp([queryDimension, ...(options.trunkHidden ?? [40, 40]), latent], {
    activation,
    outputActivation: activation,
    init: xavierUniform(),
  })
  const coefficients = (p: DeepOnetParams, u: Value) => branch.apply(p.branch as never, u)
  const basis = (p: DeepOnetParams, y: Value) => trunk.apply(p.trunk as never, y)
  return {
    label: `DeepONet(m = ${sensors}, p = ${latent})`,
    init: (s) => ({
      branch: branch.init(child(s, 'branch')) as Params[],
      trunk: trunk.init(child(s, 'trunk')) as Params[],
      bias: zeros([1]),
    }),
    coefficients,
    basis,
    apply: (p, u, y) => add(matmul(coefficients(p, u), transpose(basis(p, y))), p.bias),
  }
}

/** A batch of aligned operator data; the query points are shared and held by the loss. */
export type AlignedBatch = {
  /** The sensor values, $[N, m]$. */
  u: Tensor
  /** The targets $G(u)(y)$ at the shared queries, $[N, Q]$. */
  G: Tensor
}

/**
 * The mean squared error of a DeepONet on aligned data with the query points `y` shared by every function. The
 * batch holds `u` and `G` only, so minibatches split functions, not queries.
 *
 * @param net The network.
 * @param y The shared query points, $[Q, d]$.
 * @returns The loss of the parameters and a batch, differentiable in the parameters.
 *
 * @example The loss of a fresh network on two functions
 * const net = deepOnet({ sensors: 3, latent: 2 })
 * const loss = deepOnetLoss(net, tensor([[0], [1]]))
 * print('loss =', loss(net.init(stream(0)), { u: tensor([[0, 1, 0], [1, 0, 1]]), G: tensor([[0, 1], [1, 0]]) }))
 */
export function deepOnetLoss(net: DeepOnet, y: Tensor): (p: DeepOnetParams, batch: AlignedBatch) => Value {
  return (p: DeepOnetParams, batch: AlignedBatch) => mean(square(sub(net.apply(p, batch.u, y), batch.G)))
}
