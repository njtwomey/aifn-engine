/**
 * RealNVP (Dinh, Sohl-Dickstein and Bengio, 2017) on 2-d data: a normalising flow
 * $f = f_K \circ \dots \circ f_1$ from data $\xvec$ to a standard normal $\zvec$, each $f_k$ an affine coupling
 * bijector (`aifn-compute/probability/bijectors`' `affineCouplingBijector`) whose mask alternates between the
 * coordinates, with a multilayer perceptron conditioner giving the shift $\tvec$ and the log-scale
 * $\svec = c \tanh \rvec$ of its raw output $\rvec$ (bounded by $c$, for stable training). By the change of variables,
 * $\log p(\xvec) = \log \Gauss(f(\xvec); \zeros, \Imat) + \sum_k \log \lvert \det \Jmat_k \rvert$, with $\Jmat_k$ the
 * Jacobian of $f_k$ at its input, which training maximises on the data by Adam (`realNvpRun`). Sampling runs the
 * inverses backwards from $\zvec \sim \Gauss(\zeros, \Imat)$. Parameters are a list of conditioner parameters, one
 * per layer.
 */

import { zerosLike, type Params } from 'aifn-compute/foundation/pytree'
import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  mul,
  slice,
  square,
  sum,
  tanh,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { affineCouplingBijector, type Bijector } from 'aifn-compute/probability/bijectors'

/** A RealNVP flow on $D$-dimensional data, without its parameters. */
export interface RealNvp {
  /** The data dimension $D$. */
  readonly dimension: number
  /** The number of coupling layers $K$. */
  readonly layers: number
  /** Each layer's 0/1 mask, $D$ entries: 1 keeps a coordinate (and feeds it to the conditioner), 0 transforms it. */
  readonly masks: readonly (readonly number[])[]
  /** The conditioner's shape, shared by every layer: an MLP from $D$ inputs to $2D$ outputs (shift, raw log-scale). */
  readonly conditioner: Layer<Params[]>
  /** The bound $c$ on each log-scale. */
  readonly scale: number
}

/** Options of `realNvp`. */
export interface RealNvpOptions {
  /** Data dimension (default 2). */
  dimension?: number
  /** Coupling layers (default 6), alternating masks. */
  layers?: number
  /** Hidden widths of each conditioner (default [32, 32]). */
  hidden?: readonly number[]
  /** The bound $c$ on $\lvert s \rvert$ (default 2). */
  scale?: number
}

/**
 * A RealNVP flow: alternating masks $(1, 0, 1, \dots)$ and $(0, 1, 0, \dots)$, and one conditioner shape (ReLU
 * hidden layers) for every layer. Its parameters are drawn separately (`initRealNvp`).
 *
 * @param options The dimension, the number of layers, the conditioner's hidden widths and the log-scale bound.
 * @returns The flow.
 *
 * @example Three layers' masks on 2-d data
 * const flow = realNvp({ layers: 3, hidden: [8] })
 * print('masks', flow.masks, ' bound', flow.scale)
 */
export function realNvp(options: RealNvpOptions = {}): RealNvp {
  const { dimension = 2, layers = 6, hidden = [32, 32], scale = 2 } = options
  const masks = Array.from({ length: layers }, (_, k) =>
    Array.from({ length: dimension }, (_, j) => ((j + k) % 2 === 0 ? 1 : 0)),
  )
  return {
    dimension,
    layers,
    masks,
    conditioner: Mlp([dimension, ...hidden, 2 * dimension], { activation: 'relu' }),
    scale,
  }
}

/**
 * Initial parameters: one conditioner per layer, with its output layer zeroed so that every coupling starts as the
 * identity ($\svec = \tvec = \zeros$) and the initial density is the base's.
 *
 * @param flow The flow.
 * @param s The stream; layer $k$ draws from its child `('layer', k)`.
 * @returns The parameters, one conditioner's per layer.
 *
 * @example At the start the flow's density is the standard normal: $\log p(\zeros) = -\log 2\pi$
 * const flow = realNvp()
 * const params = initRealNvp(flow, stream(1))
 * print('log p at the origin:', flowLogDensityValues(flow, params, tensor([[0, 0]])))
 * print('-log 2 pi:', -Math.log(2 * Math.PI))
 */
export function initRealNvp(flow: RealNvp, s: Stream): Params[][] {
  return flow.masks.map((_, k) => {
    const p = flow.conditioner.init(child(s, 'layer', k))
    const last = p.length - 1
    return p.map((layer, i) => (i === last ? zerosLike(layer) : layer))
  })
}

/**
 * The coupling bijector of layer $k$ under parameters, oriented from the data side to the base side: its `forward`
 * moves towards $\zvec$, its `inverse` towards $\xvec$.
 *
 * @param flow The flow.
 * @param params The parameters, one conditioner's per layer.
 * @param k The layer, from 0 to $K - 1$.
 * @returns The bijector, on one point of $D$ values or a batch $[n, D]$.
 *
 * @example A layer's map, its inverse and its log-determinant at a 2-d point
 * const flow = realNvp({ layers: 2, hidden: [8] })
 * const params = flow.masks.map((_, k) => flow.conditioner.init(stream(k)))
 * const layer = couplingLayer(flow, params, 0)
 * const x = tensor([[0.5, -1]])
 * const y = layer.forward(x)
 * print('forward', y, ' inverse', layer.inverse(y), ' log |det J|', layer.logAbsDetJacobian(x))
 */
export function couplingLayer(flow: RealNvp, params: Params[][], k: number): Bijector & { readonly eventRank: 1 } {
  const D = flow.dimension
  return affineCouplingBijector(flow.masks[k], (kept: Value) => {
    const out = flow.conditioner.apply(params[k], kept)
    return { shift: slice(out, null, [0, D]), logScale: mul(flow.scale, tanh(slice(out, null, [D, 2 * D]))) }
  })
}

/**
 * $\log p(\xvec)$ of each row of `x`, by the change of variables: the base log density of $f(\xvec)$ plus every
 * layer's log-determinant. Differentiable in the parameters (the training loss is its negative mean).
 *
 * @param flow The flow.
 * @param params The parameters, one conditioner's per layer.
 * @param x The points, $[n, D]$ (traced inside a gradient).
 * @returns The log densities, $[n]$, traced when `params` or `x` are.
 *
 * @example The log density of two points under a random flow
 * const flow = realNvp({ layers: 2, hidden: [8] })
 * const params = flow.masks.map((_, k) => flow.conditioner.init(stream(k)))
 * print(flowLogDensity(flow, params, tensor([[0.5, -1], [0, 0]])))
 */
export function flowLogDensity(flow: RealNvp, params: Params[][], x: Value): Value {
  let z = x
  let logDet: Value = 0
  for (let k = 0; k < flow.layers; k++) {
    const b = couplingLayer(flow, params, k)
    logDet = add(logDet, b.logAbsDetJacobian(z))
    z = b.forward(z)
  }
  const logBase = mul(-0.5, add(sum(square(z), -1), flow.dimension * Math.log(2 * Math.PI)))
  return add(logBase, logDet)
}

/**
 * The data pushed through the flow: $\xvec$ after each layer, from the data (entry 0) to the base (entry $K$).
 *
 * @param flow The flow.
 * @param params The parameters, one conditioner's per layer.
 * @param x The points, $[n, D]$.
 * @returns $K + 1$ tensors, each $[n, D]$.
 *
 * @example A point layer by layer: each layer moves only the coordinate its mask leaves out
 * const flow = realNvp({ layers: 2, hidden: [8] })
 * const params = flow.masks.map((_, k) => flow.conditioner.init(stream(k)))
 * print(flowForward(flow, params, tensor([[0.5, -1]])))
 */
export function flowForward(flow: RealNvp, params: Params[][], x: Tensor): Tensor[] {
  const out = [x]
  let z: Value = x
  for (let k = 0; k < flow.layers; k++) {
    z = couplingLayer(flow, params, k).forward(z)
    out.push(unwrap(z) as Tensor)
  }
  return out
}

/**
 * Samples: base points $\zvec$ pulled back through the inverses, last layer first, so $f^{-1}(\zvec)$. Draw $\zvec$
 * from $\Gauss(\zeros, \Imat)$ for samples of the model.
 *
 * @param flow The flow.
 * @param params The parameters, one conditioner's per layer.
 * @param z The base points, $[n, D]$.
 * @returns The data-side points, $[n, D]$.
 *
 * @example Pulling a point's base image back recovers the point
 * const flow = realNvp({ layers: 2, hidden: [8] })
 * const params = flow.masks.map((_, k) => flow.conditioner.init(stream(k)))
 * const z = flowForward(flow, params, tensor([[0.5, -1]])).at(-1)
 * print('z', z, ' back', flowSample(flow, params, z))
 */
export function flowSample(flow: RealNvp, params: Params[][], z: Tensor): Tensor {
  let x: Value = z
  for (let k = flow.layers - 1; k >= 0; k--) x = couplingLayer(flow, params, k).inverse(x)
  return unwrap(x) as Tensor
}

/**
 * $\log p(\xvec)$ of each row as plain numbers: `flowLogDensity` outside any trace.
 *
 * @param flow The flow.
 * @param params The parameters, one conditioner's per layer.
 * @param x The points, $[n, D]$.
 * @returns The $n$ log densities.
 *
 * @example The density of a random flow on a few points
 * const flow = realNvp({ layers: 2, hidden: [8] })
 * const params = flow.masks.map((_, k) => flow.conditioner.init(stream(k)))
 * print(flowLogDensityValues(flow, params, tensor([[0, 0], [1, 1], [-1, 2]])))
 */
export const flowLogDensityValues = (flow: RealNvp, params: Params[][], x: Tensor): Float64Array =>
  Float64Array.from(toFlat(unwrap(flowLogDensity(flow, params, x)) as Tensor))
