/**
 * RealNVP (Dinh, Sohl-Dickstein and Bengio, 2017) on 2-d data: a normalising flow f = f_K ∘ … ∘ f₁ from data x to a
 * standard normal z, each fₖ an affine coupling bijector (`aifn-compute/probability/bijectors`' `affineCouplingBijector`) whose
 * mask alternates between the coordinates, with a multilayer perceptron conditioner giving the shift t and the
 * log-scale s = scale · tanh(raw) (bounded, for stable training). By the change of variables,
 *
 *   log p(x) = log N(f(x); 0, I) + Σₖ log |det ∂fₖ/∂x|,
 *
 * which the flow maximises on the data by Adam. Sampling runs the inverses backwards from z ~ N(0, I).
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

/** A RealNVP flow on D-dimensional data. */
export interface RealNvp {
  readonly dimension: number
  readonly layers: number
  readonly masks: readonly (readonly number[])[]
  readonly conditioner: Layer<Params[]>
  /** The bound on each log-scale. */
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
  /** The bound on |s| (default 2). */
  scale?: number
}

/** A RealNVP flow: alternating masks (1, 0, 1, …) and (0, 1, 0, …), and one conditioner shape for every layer. */
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
 * identity (s = t = 0) and the initial density is the base's.
 */
export function initRealNvp(flow: RealNvp, s: Stream): Params[][] {
  return flow.masks.map((_, k) => {
    const p = flow.conditioner.init(child(s, 'layer', k))
    const last = p.length - 1
    return p.map((layer, i) => (i === last ? zerosLike(layer) : layer))
  })
}

/** The coupling bijector of layer k (data side → base side) under parameters. */
export function couplingLayer(flow: RealNvp, params: Params[][], k: number): Bijector & { readonly eventRank: 1 } {
  const D = flow.dimension
  return affineCouplingBijector(flow.masks[k], (kept: Value) => {
    const out = flow.conditioner.apply(params[k], kept)
    return { shift: slice(out, null, [0, D]), logScale: mul(flow.scale, tanh(slice(out, null, [D, 2 * D]))) }
  })
}

/** log p(x) of each row of x [n, D], as a traced [n] value. */
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

/** The data pushed through the flow: x after each layer, from the data (entry 0) to the base (entry K). */
export function flowForward(flow: RealNvp, params: Params[][], x: Tensor): Tensor[] {
  const out = [x]
  let z: Value = x
  for (let k = 0; k < flow.layers; k++) {
    z = couplingLayer(flow, params, k).forward(z)
    out.push(unwrap(z) as Tensor)
  }
  return out
}

/** Samples: base points z [n, D] pulled back through the inverses, last layer first. */
export function flowSample(flow: RealNvp, params: Params[][], z: Tensor): Tensor {
  let x: Value = z
  for (let k = flow.layers - 1; k >= 0; k--) x = couplingLayer(flow, params, k).inverse(x)
  return unwrap(x) as Tensor
}

/** log p(x) as plain numbers. */
export const flowLogDensityValues = (flow: RealNvp, params: Params[][], x: Tensor): Float64Array =>
  Float64Array.from(toFlat(unwrap(flowLogDensity(flow, params, x)) as Tensor))
