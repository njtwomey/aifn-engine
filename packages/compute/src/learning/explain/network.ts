/**
 * A feedforward network as plain data, for the explanations that look inside one: DeepLIFT and DeepSHAP propagate
 * multipliers through its layers, TCAV reads its hidden activations and differentiates the output with respect to
 * them, and the model-randomisation sanity check re-initialises its layers. A `DenseNetwork` is a stack of affine maps
 * $\avec \mapsto \avec\Wmat + \bvec$ ($\Wmat$ of shape $\text{in} \times \text{out}$, as `aifn-compute/nn`'s
 * `Linear`), each but the last followed by the same elementwise activation; the output layer is linear.
 * `fromMlpParams` reads the parameter list of an `aifn-compute/nn` `Mlp`.
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

/**
 * The hidden activation of a `DenseNetwork`: $\max(z, 0)$, $\tanh z$, the logistic $1/(1 + e^{-z})$, or none.
 */
export type DenseActivation = 'relu' | 'tanh' | 'sigmoid' | 'identity'

/**
 * A feedforward network: layer $l$ maps activations $\Amat$ ($m \times \text{in}_l$) to
 * $\Amat\Wmat_l + \bvec_l$ ($m \times \text{out}_l$); hidden layers then apply `activation`.
 */
export type DenseNetwork = {
  /** Weights per layer, each $\text{in} \times \text{out}$ (a tensor or nested rows). */
  weights: readonly MatrixLike[]
  /** Biases per layer, $\text{out}$ values each. */
  biases: readonly ArrayLike<number>[]
  /** The activation after every layer but the last. */
  activation: DenseActivation
}

/**
 * One layer as flat arrays: `W`, the weights row-major ($\text{inputs} \times \text{outputs}$ values), `b`, the
 * biases, and the layer's numbers of `inputs` and `outputs`.
 */
type Layer = { W: Float64Array; b: Float64Array; inputs: Size; outputs: Size }

/**
 * The layers of a network as flat arrays (copies), checked for matching shapes. Throws `ShapeError` when there is no
 * layer, the numbers of weights and biases differ, a bias does not match its layer's outputs, or a layer's inputs do
 * not match the previous layer's outputs.
 *
 * @param net The network.
 * @returns One `Layer` per weight matrix, in order.
 *
 * @example A 2-2-1 network's layers
 * const net = { weights: [[[1, -1], [1, 1]], [[1], [2]]], biases: [[0, 0], [0]], activation: 'relu' }
 * print(denseLayers(net).map((L) => [L.inputs, L.outputs]))
 */
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

/**
 * The activation applied to a number.
 *
 * @param kind The activation.
 * @param z The pre-activation.
 * @returns The activation of $z$.
 *
 * @example The four activations at 1
 * for (const kind of ['relu', 'tanh', 'sigmoid', 'identity']) print(kind, activate(kind, 1))
 */
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

/**
 * The activation's derivative at $z$ (for ReLU, 0 at $z = 0$).
 *
 * @param kind The activation.
 * @param z The pre-activation.
 * @returns The derivative of the activation at $z$.
 *
 * @example Slopes at 0
 * for (const kind of ['relu', 'tanh', 'sigmoid', 'identity']) print(kind, activateDerivative(kind, 0))
 */
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

/**
 * The activation over tensors and tracers, elementwise (so that it can be differentiated).
 *
 * @param kind The activation.
 * @param z The pre-activations: a number, a tensor or a traced value.
 * @returns The activations, of the shape of `z`.
 */
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
 * The forward pass on a batch of rows, keeping every layer's values. Throws `ShapeError` when the rows do not have the
 * network's number of inputs (and as `denseLayers`).
 *
 * @param net The network.
 * @param X The rows ($m \times d$).
 * @returns `pre`, the pre-activations $\Zmat_l$ of each layer; `post`, the activations, with $\Amat_0 = \Xmat$ first
 *   and the last layer's equal to its $\Zmat$ (so `post` has one more entry than `pre`); each row-major,
 *   $m \times \text{out}_l$ values. `sizes` holds the width of each entry of `post`, and `rows` is $m$.
 *
 * @example Every layer's values for one row
 * const net = { weights: [[[1, -1], [1, 1]], [[1], [2]]], biases: [[0, 0], [0]], activation: 'relu' }
 * const f = denseForward(net, [[1, 2]])
 * print('pre =', f.pre)
 * print('post =', f.post, ' sizes =', f.sizes)
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
 * The network from layer `from` on, as a differentiable function of that layer's input. Throws `DomainError` when
 * `from` is not a layer index (and as `denseLayers`).
 *
 * @param net The network.
 * @param options Where to start and which output to return.
 * @param options.from The first layer applied (default 0): 0 is the whole network on its input; $l$ takes the
 *   activations $\avec_l$ after layer $l$'s nonlinearity, as `denseForward`'s `post[l]`.
 * @param options.output The output unit returned (default 0).
 * @returns A function of one activation vector (returning a number) or a batch of rows (returning one value per row).
 *
 * @example The output and its gradient, from the input and from the hidden layer
 * const net = { weights: [[[1, -1], [1, 1]], [[1], [2]]], biases: [[0, 0], [0]], activation: 'relu' }
 * const f = denseFunction(net)
 * print('f(x) =', f(tensor([1, 2])), ' grad =', grad(f)(tensor([1, 2])))
 * const head = denseFunction(net, { from: 1 })
 * print('head(a1) =', head(tensor([3, 1])), ' grad =', grad(head)(tensor([3, 1])))
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

/**
 * The network's output on each row, as numbers.
 *
 * @param net The network.
 * @param X The rows ($m \times d$).
 * @param output The output unit returned.
 * @returns The output for each row ($m$ values).
 *
 * @example A batch of rows
 * const net = { weights: [[[1, -1], [1, 1]], [[1], [2]]], biases: [[0, 0], [0]], activation: 'relu' }
 * print(denseOutput(net, [[1, 2], [2, 1], [-1, -1]]))
 */
export function denseOutput(net: DenseNetwork, X: MatrixLike, output: Size = 0): Float64Array {
  const f = denseForward(net, X)
  const k = f.sizes.at(-1) as Size
  const z = f.post.at(-1) as Float64Array
  return Float64Array.from({ length: f.rows }, (_, r) => z[r * k + output])
}

/**
 * A `DenseNetwork` from the parameter list of an `aifn-compute/nn` `Mlp`: the entries with a `weight` are its
 * `Linear` layers, in order, and the others (parameter-free activation layers) are skipped. A layer without a bias gets
 * zeros. The tensors are used as they are, not copied.
 *
 * @param params The parameter list: `Linear` layers `{ weight, bias }` (`weight` $\text{in} \times \text{out}$)
 *   interleaved with activation layers.
 * @param activation The hidden activation the `Mlp` was built with (the list does not record it).
 * @returns The network.
 *
 * @example A parameter list with an activation layer and a missing bias
 * const params = [{ weight: tensor([[1, -1], [1, 1]]), bias: tensor([0, 0]) }, {}, { weight: tensor([[1], [2]]) }]
 * const net = fromMlpParams(params, 'relu')
 * print('biases =', net.biases)
 * print('output =', denseOutput(net, [[1, 2]]))
 */
export function fromMlpParams(params: readonly object[], activation: DenseActivation): DenseNetwork {
  const linear = params.filter((p): p is { weight: Tensor; bias?: Tensor } => 'weight' in p)
  return {
    weights: linear.map((p) => p.weight),
    biases: linear.map((p) => (p.bias ? Float64Array.from(toFlat(p.bias)) : new Float64Array(p.weight.shape[1]))),
    activation,
  }
}
