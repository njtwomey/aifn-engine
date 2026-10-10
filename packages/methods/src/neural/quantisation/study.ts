/**
 * A quantisation study on a small trained network: train an MLP classifier in floating point (Adam, streamed), then
 * quantise its weight matrices to $b$ bits by round-to-nearest per tensor, per output channel, by GPTQ (Frantar et
 * al., 2023: each layer's error fed forward through the Hessian of its own inputs) and by AWQ (Lin et al., 2024: input
 * channels scaled by their activations' size before rounding), and measure the accuracy and the weight error at each
 * $b$. Round-to-nearest is symmetric; biases stay in floating point. GPTQ and AWQ calibrate on the inputs each layer
 * sees in the floating-point network, over the whole training set.
 *
 * The weight error is the signal-to-quantisation-noise ratio of all weights $w_i$ and their quantised values
 * $\hat w_i$, $10 \log_{10} \bigl(\sum_i w_i^2 / \sum_i (w_i - \hat w_i)^2\bigr)$ dB. Built from `aifn-compute/nn`
 * (the MLP, training) and `aifn-compute/nn/quantise`.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { ravel, type Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { heUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { awqQuantise, dequantise, gptqQuantise, quantisationParams, quantise } from 'aifn-compute/nn/quantise'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The quantisation methods compared, in their fixed order (and colour slots). */
export const QUANTISATION_METHODS = ['per-tensor', 'per-channel', 'gptq', 'awq'] as const
/** A quantisation method of the study: round-to-nearest per tensor or per output channel, GPTQ or AWQ. */
export type QuantisationMethod = (typeof QUANTISATION_METHODS)[number]

/** Options of `quantisationStudy`. */
export interface QuantisationStudyOptions {
  /** Hidden units per layer (default 16). */
  width?: Size
  /** Hidden layers (default 2). */
  depth?: Size
  /** Adam steps (default 400). */
  steps?: Size
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /** Minibatch size (default 32, at most the number of examples). */
  batchSize?: Size
  /** Bit widths to evaluate (default 2 to 8). */
  bits?: readonly Size[]
  /** The seed of the initialisation and the minibatches (default 0). */
  seed?: number
}

/** The evaluation at one bit width. */
export interface BitResult {
  /** The bit width $b$. */
  bits: Size
  /** Training-set accuracy of the network with quantised weights, by method. */
  accuracy: Record<QuantisationMethod, number>
  /** Signal-to-quantisation-noise ratio (dB) of all weights, by method. */
  sqnr: Record<QuantisationMethod, number>
}

/** A snapshot of the study: training progress, then the evaluations. */
export interface QuantisationSnapshot {
  /** Units of work done: Adam steps, then one per bit width evaluated. */
  done: Size
  /** Units of work in all: the steps plus the bit widths. */
  total: Size
  /** Training, evaluating the bit widths, or finished. */
  phase: 'training' | 'evaluating' | 'done'
  /** The recorded steps, about 200 over the run, starting at 0. */
  step: number[]
  /** The full-batch training loss at each recorded step. */
  loss: number[]
  /** The trained network's training-set accuracy in floating point (NaN while training). */
  accuracy: number
  /** Every parameter of the trained network, weights and biases, as one vector (empty while training). */
  theta: Float64Array
  /** Every weight (not bias) of the trained network, layer by layer (empty while training). */
  weights: Float64Array
  /** Each weight matrix's shape, inputs by outputs. */
  layers: [number, number][]
  /** The evaluations of the bit widths done so far. */
  results: BitResult[]
  /** The MLP's layer sizes, inputs first and the one logit last. */
  sizes: number[]
}

/**
 * The MLP of the study: `depth` hidden ReLU layers of `width` units and one logit, initialised He-uniform.
 *
 * @param inputs The number of input features.
 * @param width The units of each hidden layer.
 * @param depth The number of hidden layers.
 * @returns The `Mlp` and its layer sizes, inputs first.
 *
 * @example Two hidden layers of four units on two inputs
 * const { model, sizes } = studyModel(2, 4, 2)
 * print('sizes:', sizes)
 * print('logits of two points:', model.apply(model.init(stream(0)), tensor([[1, 0], [0, 1]])))
 */
export function studyModel(inputs: Size, width: Size, depth: Size) {
  const sizes = [inputs, ...Array.from({ length: depth }, () => width), 1]
  return { model: Mlp(sizes, { activation: 'relu', init: heUniform() }), sizes }
}

/** A Linear layer's parameters: the weight, inputs by outputs, and the bias. */
type Layer = { weight: Tensor; bias?: Tensor }

/**
 * The logits of the study's MLP and the input of each Linear layer (the calibration data GPTQ and AWQ need), from one
 * `model.apply` with a tap recording every layer's output. The Mlp's layers sit at paths $0, 1, 2, \dots$: Linear
 * layers at even indices, so Linear layer $k$ reads the input `x` ($k = 0$) or the activation at path $2k - 1$.
 *
 * @param model The study's MLP.
 * @param params Its parameters.
 * @param x The inputs, $n \times d$.
 * @returns The $n$ logits, and each Linear layer's inputs as a row-major $n \times d_{\text{in}}$ array.
 */
function forward(
  model: ReturnType<typeof studyModel>['model'],
  params: Params[],
  x: Tensor,
): { logits: Float64Array; inputs: Float64Array[] } {
  const outputs = new Map<string, Value>()
  const out = model.apply(params, x, {
    tap: (path, value) => {
      outputs.set(path, value)
      return value
    },
  })
  const linear = layersOf(params).length
  const inputs = Array.from({ length: linear }, (_, k) =>
    Float64Array.from(toFlat(k === 0 ? x : (outputs.get(String(2 * k - 1)) as Tensor))),
  )
  return { logits: Float64Array.from(toFlat(out as Tensor)), inputs }
}

/**
 * The Linear layers of the Mlp's parameters, in order: the parameters alternate Linear layers and `{}` for each
 * activation.
 *
 * @param params The Mlp's parameters.
 * @returns The Linear layers' parameters (the same objects, not copies).
 */
const layersOf = (params: Params[]) =>
  (params as unknown as Partial<Layer>[]).filter((p): p is Layer => p.weight !== undefined)

/**
 * The parameters with the Linear layers replaced, in order, by `layers`. `params` is not modified.
 *
 * @param params The Mlp's parameters.
 * @param layers One replacement for each Linear layer, in order.
 * @returns A new parameter list.
 */
function withLayers(params: Params[], layers: readonly Layer[]): Params[] {
  let k = 0
  return (params as unknown as Partial<Layer>[]).map((p) =>
    p.weight !== undefined ? layers[k++] : p,
  ) as unknown as Params[]
}

/**
 * The share of examples whose logit's sign matches the label: positive predicts 1.
 *
 * @param logits One logit per example.
 * @param y The labels, 0 or 1.
 * @returns The accuracy, in $[0, 1]$.
 */
const accuracyOf = (logits: ArrayLike<number>, y: ArrayLike<number>) => {
  let right = 0
  for (let i = 0; i < y.length; i++) right += (logits[i] > 0 ? 1 : 0) === y[i] ? 1 : 0
  return right / y.length
}

/**
 * The weight matrices quantised by one method at $b$ bits and dequantised back (biases kept in floating point, as is
 * usual). Round-to-nearest is symmetric, with one scale per matrix or per output column; GPTQ and AWQ take each
 * matrix transposed to outputs by inputs, with the layer's calibration inputs.
 *
 * @param layers The Linear layers, weights inputs by outputs.
 * @param inputs Each layer's inputs as `forward` returns them.
 * @param method The quantisation method.
 * @param bits The bit width $b$.
 * @returns New layers with the dequantised weights.
 */
function quantisedLayers(
  layers: readonly Layer[],
  inputs: readonly Float64Array[],
  method: QuantisationMethod,
  bits: Size,
): Layer[] {
  return layers.map((layer, k) => {
    const W = layer.weight
    const [din, dout] = W.shape
    if (method === 'gptq' || method === 'awq') {
      // GPTQ and AWQ take W as (out × in) and the layer's inputs X (n × in).
      const Wt = new Float64Array(dout * din)
      const w = toFlat(W)
      for (let c = 0; c < din; c++) for (let j = 0; j < dout; j++) Wt[j * din + c] = w[c * dout + j]
      const n = inputs[k].length / din
      const [Wo, Xk] = [fromData(Wt, [dout, din]), fromData(inputs[k], [n, din])]
      const r = method === 'gptq' ? gptqQuantise(Wo, Xk, { bits }) : awqQuantise(Wo, Xk, { bits })
      const q = toFlat(r.weights)
      const back = new Float64Array(din * dout)
      for (let c = 0; c < din; c++) for (let j = 0; j < dout; j++) back[c * dout + j] = q[j * din + c]
      return { ...layer, weight: fromData(back, [din, dout]) }
    }
    const p = quantisationParams(W, { bits, scheme: 'symmetric', ...(method === 'per-channel' ? { axis: 1 } : {}) })
    return { ...layer, weight: dequantise(quantise(W, p), p) }
  })
}

/**
 * Train a small ReLU MLP on binary labels by Adam (snapshots stream while it trains), then quantise its weights at each
 * bit width by round-to-nearest per tensor, per output channel, GPTQ and AWQ, reporting accuracy and weight SQNR.
 * Throws `DomainError` when the data have no labels. Training stops early if it diverges. Accuracy is measured on
 * the training data. Deterministic from the seed.
 *
 * @param data The inputs `x` ($n \times d$) and the labels `y` ($n$ values, 0 or 1; required).
 * @param options The network's width and depth, the training, the bit widths and the seed.
 * @returns A generator of snapshots: every few steps while training, one per bit width, and a last one when done.
 *
 * @example Size and error of a small classifier at 2, 4 and 8 bits
 * const x = normals(stream(0), [64, 2])
 * const y = fromData(Float64Array.from(toRows(x), ([a, b]) => (a * b > 0 ? 1 : 0)), [64])
 * let last
 * for (const s of quantisationStudy({ x, y }, { width: 8, steps: 150, bits: [2, 4, 8] })) last = s
 * print('float accuracy:', last.accuracy, ' weights:', last.weights.length)
 * for (const r of last.results) {
 *   print(`${r.bits} bits, ${(last.weights.length * r.bits) / 8} bytes of weights:`)
 *   for (const m of QUANTISATION_METHODS) print(`  ${m}: accuracy ${r.accuracy[m]}, SQNR ${r.sqnr[m].toFixed(1)} dB`)
 * }
 */
export function* quantisationStudy(
  data: { readonly x: Tensor; readonly y?: Tensor },
  options: QuantisationStudyOptions = {},
): Generator<QuantisationSnapshot> {
  const { width = 16, depth = 2, steps = 400, stepSize = 0.01, batchSize = 32, seed = 0 } = options
  const bitsList = options.bits ?? [2, 3, 4, 5, 6, 7, 8]
  if (!data.y) throw new DomainError('quantisationStudy', 'quantisationStudy: the data need labels y')
  const [n, d] = data.x.shape
  const y = toFlat(data.y)
  const target = fromData(Float64Array.from(y), [n, 1])
  const { model, sizes } = studyModel(d, width, depth)
  const loss = (params: Params[], batch: { x: Tensor; y: Tensor }): Value =>
    binaryCrossEntropyWithLogits(model.apply(params, batch.x), batch.y)
  const alg = trainingLoop<Params[], { x: Tensor; y: Tensor }>({
    loss,
    data: { x: data.x, y: target },
    batchSize: Math.max(1, Math.min(batchSize, n)),
    optimizer: adamRule({ stepSize }) as UpdateRule<unknown>,
  })
  const root = stream(`quantisation-study-${seed}`)
  let state = alg.init({ params: model.init(child(root, 'weights')) }, child(root, 'init'))
  const total = steps + bitsList.length
  const snap: QuantisationSnapshot = {
    done: 0,
    total,
    phase: 'training',
    step: [],
    loss: [],
    accuracy: NaN,
    theta: new Float64Array(0),
    weights: new Float64Array(0),
    layers: [],
    results: [],
    sizes,
  }
  const fullLoss = (params: Params[]) => {
    const v = unwrap(loss(params, { x: data.x, y: target }))
    return typeof v === 'number' ? v : toFlat(v as Tensor)[0]
  }
  const every = Math.max(1, Math.ceil(steps / 200))
  const yieldEvery = Math.max(5, Math.ceil(steps / 25))
  snap.step.push(0)
  snap.loss.push(fullLoss(state.params as Params[]))
  yield { ...snap }
  for (let t = 1; t <= steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    if (state.diverged) break
    if (t % every === 0 || t === steps) {
      snap.step.push(t)
      snap.loss.push(fullLoss(state.params as Params[]))
    }
    snap.done = t
    if (t % yieldEvery === 0) yield { ...snap, step: [...snap.step], loss: [...snap.loss] }
  }
  const params = state.params as Params[]
  const layers = layersOf(params)
  const float = forward(model, params, data.x)
  snap.accuracy = accuracyOf(float.logits, y)
  snap.theta = ravel(params).vector
  snap.layers = layers.map((l) => [l.weight.shape[0], l.weight.shape[1]] as [number, number])
  snap.weights = Float64Array.from(layers.flatMap((l) => Array.from(toFlat(l.weight))))
  snap.phase = 'evaluating'
  for (const bits of bitsList) {
    const accuracy = {} as Record<QuantisationMethod, number>
    const sqnr = {} as Record<QuantisationMethod, number>
    for (const method of QUANTISATION_METHODS) {
      const q = quantisedLayers(layers, float.inputs, method, bits)
      accuracy[method] = accuracyOf(forward(model, withLayers(params, q), data.x).logits, y)
      const qw = Float64Array.from(q.flatMap((l) => Array.from(toFlat(l.weight))))
      let se = 0
      let ss = 0
      snap.weights.forEach((v, i) => {
        se += (v - qw[i]) ** 2
        ss += v * v
      })
      sqnr[method] = 10 * Math.log10(ss / Math.max(se, 1e-300))
    }
    snap.results.push({ bits, accuracy, sqnr })
    snap.done++
    yield { ...snap, results: [...snap.results] }
  }
  snap.phase = 'done'
  snap.done = total
  yield { ...snap, results: [...snap.results] }
}
