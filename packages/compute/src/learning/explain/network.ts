/**
 * A feedforward network as plain data, for the explanations that look inside one: DeepLIFT and DeepSHAP propagate
 * multipliers through its layers, TCAV reads its hidden activations and differentiates the output with respect to them,
 * and the model-randomisation sanity check re-initialises its layers. A `DenseNetwork` is a stack of affine maps
 * a ↦ aW + b (W of shape [in, out], as `aifn-compute/nn`'s `Linear`), each but the last followed by the same elementwise
 * activation; the output layer is linear. `fromMlpParams` reads the parameter list of an `aifn-compute/nn` `Mlp`.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import {
  add,
  dense,
  div,
  exp,
  fromData,
  get,
  matmul,
  maximum,
  neg,
  shapeOfValue,
  slice,
  tanh,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The hidden activation of a `DenseNetwork`. */
export type DenseActivation = 'relu' | 'tanh' | 'sigmoid' | 'identity'

/** A feedforward network: layer l maps a [m, in_l] to a W_l + b_l [m, out_l]; hidden layers apply `activation`. */
export type DenseNetwork = {
  /** Weights per layer, [in, out] row-major (a Tensor or nested rows). */
  weights: readonly MatrixLike[]
  /** Biases per layer [out]. */
  biases: readonly ArrayLike<number>[]
  activation: DenseActivation
}

type Layer = { W: Float64Array; b: Float64Array; inputs: Size; outputs: Size }

/** The layers of a network as flat arrays, checked for matching shapes. */
export function denseLayers(net: DenseNetwork): Layer[] {
  if (net.weights.length === 0 || net.weights.length !== net.biases.length)
    throw new ShapeError('denseLayers', 'denseLayers: need one bias per weight matrix and at least one layer')
  const layers = net.weights.map((w, l) => {
    const { data, m, n } = dense.toMatrixF64(w, 'denseLayers')
    const b = Float64Array.from(net.biases[l])
    if (b.length !== n)
      throw new ShapeError('denseLayers', `denseLayers: layer ${l} has ${n} outputs, ${b.length} biases`)
    return { W: Float64Array.from(data), b, inputs: m, outputs: n }
  })
  for (let l = 1; l < layers.length; l++)
    if (layers[l].inputs !== layers[l - 1].outputs)
      throw new ShapeError(
        'denseLayers',
        `denseLayers: layer ${l} takes ${layers[l].inputs} inputs after ${layers[l - 1].outputs}`,
      )
  return layers
}

/** The activation as a number function. */
export function activate(kind: DenseActivation, z: number): number {
  switch (kind) {
    case 'relu':
      return z > 0 ? z : 0
    case 'tanh':
      return Math.tanh(z)
    case 'sigmoid':
      return 1 / (1 + Math.exp(-z))
    case 'identity':
      return z
  }
}

/** The activation's derivative at z. */
export function activateDerivative(kind: DenseActivation, z: number): number {
  switch (kind) {
    case 'relu':
      return z > 0 ? 1 : 0
    case 'tanh':
      return 1 - Math.tanh(z) ** 2
    case 'sigmoid': {
      const s = 1 / (1 + Math.exp(-z))
      return s * (1 - s)
    }
    case 'identity':
      return 1
  }
}

/** The activation over tensors and tracers (for autodiff). */
function activateValue(kind: DenseActivation, z: Value): Value {
  switch (kind) {
    case 'relu':
      return maximum(z, 0)
    case 'tanh':
      return tanh(z)
    case 'sigmoid':
      return div(1, add(1, exp(neg(z))))
    case 'identity':
      return z
  }
}

/**
 * The forward pass on rows X [m, d]: the pre-activations z_l and activations a_l of every layer (a_0 = X; the last
 * layer's a equals its z), each [m, out_l] row-major.
 */
export function denseForward(
  net: DenseNetwork,
  X: MatrixLike,
): { pre: Float64Array[]; post: Float64Array[]; sizes: Size[]; rows: Size } {
  const layers = denseLayers(net)
  const { data, m, n } = dense.toMatrixF64(X, 'denseForward')
  if (n !== layers[0].inputs)
    throw new ShapeError('denseForward', `denseForward: the network takes ${layers[0].inputs} inputs, got ${n}`)
  const pre: Float64Array[] = []
  const post: Float64Array[] = [Float64Array.from(data)]
  const sizes: Size[] = [n]
  layers.forEach((L, l) => {
    const a = post[l]
    const z = new Float64Array(m * L.outputs)
    for (let r = 0; r < m; r++)
      for (let j = 0; j < L.outputs; j++) {
        let s = L.b[j]
        for (let i = 0; i < L.inputs; i++) s += a[r * L.inputs + i] * L.W[i * L.outputs + j]
        z[r * L.outputs + j] = s
      }
    pre.push(z)
    const last = l === layers.length - 1
    post.push(last ? z : Float64Array.from(z, (v) => activate(net.activation, v)))
    sizes.push(L.outputs)
  })
  return { pre, post, sizes, rows: m }
}

/**
 * The network from layer `from` on as a differentiable function of that layer's activations [h] (or a batch [m, h]):
 * `from = 0` is the whole network on its input; `from = l` takes the activations a_l after layer l's nonlinearity.
 * Returns output `output` (default 0), a scalar per row.
 */
export function denseFunction(net: DenseNetwork, options: { from?: Size; output?: Size } = {}): (a: Tensor) => Value {
  const { from = 0, output = 0 } = options
  const layers = denseLayers(net)
  if (!(from >= 0 && from < layers.length))
    throw new DomainError('denseFunction', `denseFunction: from must lie in 0 … ${layers.length - 1}`)
  const Ws = layers.map((L) => fromData(L.W, [L.inputs, L.outputs]))
  const bs = layers.map((L) => fromData(L.b, [L.outputs]))
  return (a) => {
    let h: Value = a
    for (let l = from; l < layers.length; l++) {
      const z: Value = add(matmul(h, Ws[l]), bs[l])
      h = l === layers.length - 1 ? z : activateValue(net.activation, z)
    }
    // One row: the output entry; a batch: column `output`.
    return shapeOfValue(h).length <= 1 ? get(h, output) : slice(h, null, output)
  }
}

/** The network's output (column `output`) on each row of X [m, d], as numbers. */
export function denseOutput(net: DenseNetwork, X: MatrixLike, output: Size = 0): Float64Array {
  const f = denseForward(net, X)
  const k = f.sizes.at(-1) as Size
  const z = f.post.at(-1) as Float64Array
  return Float64Array.from({ length: f.rows }, (_, r) => z[r * k + output])
}

/**
 * A `DenseNetwork` from the parameter list of an `aifn-compute/nn` `Mlp` (Linear layers `{ weight [in, out], bias }`
 * interleaved with parameter-free activation layers).
 */
export function fromMlpParams(params: readonly object[], activation: DenseActivation): DenseNetwork {
  const linear = params.filter((p): p is { weight: Tensor; bias?: Tensor } => 'weight' in p)
  return {
    weights: linear.map((p) => p.weight),
    biases: linear.map((p) => (p.bias ? Float64Array.from(toFlat(p.bias)) : new Float64Array(p.weight.shape[1]))),
    activation,
  }
}
