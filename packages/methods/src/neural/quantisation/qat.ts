/**
 * Quantisation-aware training against post-training quantisation on a small trained network: train a ReLU MLP in
 * floating point, then at each bit width b either round its weights once (post-training quantisation, PTQ) or
 * fine-tune it with fake-quantised weights in the forward pass and the straight-through estimator in the backward
 * pass (quantisation-aware training, QAT; Jacob et al., 2018, §3; Krishnamoorthi, 2018), and compare the accuracy of
 * the two quantised networks. Weights use one symmetric per-tensor quantiser per matrix, fitted to the current weights
 * (min–max), as `torch.ao.quantization`'s default observer; biases stay in floating point.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { dequantise, fakeQuantise, quantisationParams, quantise } from 'aifn-compute/nn/quantise'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { studyModel } from './study'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `quantisationAwareTraining`. */
export interface QatOptions {
  /** Hidden units per layer and hidden layers. Defaults 16 and 2. */
  width?: Size
  depth?: Size
  /** Floating-point Adam steps and step size. Defaults 1000 and 0.01. */
  steps?: Size
  stepSize?: number
  /** Fine-tuning Adam steps per bit width, and their step size. Defaults 300 and 0.003. */
  qatSteps?: Size
  qatStepSize?: number
  /** Minibatch size of both phases. Default 32. */
  batchSize?: Size
  /** Bit widths to compare. Default 2 … 6. */
  bits?: readonly Size[]
  seed?: number
}

/** PTQ against QAT at one bit width. */
export interface QatBitResult {
  bits: Size
  /** Training accuracy of the network with its weights rounded once (PTQ) and after fine-tuning (QAT). */
  ptq: number
  qat: number
  /** QAT's full-batch loss of the quantised network at each recorded fine-tuning step (step 0 is PTQ's loss). */
  step: number[]
  loss: number[]
  /** Every weight (not bias) at each recorded step, in floating point (the shadow weights QAT updates). */
  weights: Float64Array[]
  /** The quantiser's step size (per layer) at each recorded step. */
  scales: number[][]
}

/** A snapshot of the run: floating-point training, then PTQ and QAT at each bit width. */
export interface QatSnapshot {
  done: Size
  total: Size
  phase: 'training' | 'fine-tuning' | 'done'
  /** Floating-point training loss at each recorded step, and the trained network's accuracy. */
  step: number[]
  loss: number[]
  accuracy: number
  /** Each weight matrix's [in, out] shape. */
  layers: [number, number][]
  results: QatBitResult[]
}

type Layer = { weight: Value; bias?: Value }

/** The Mlp's parameters alternate Linear layers and {} for each activation; only Linear layers have a weight. */
const hasWeight = (p: Partial<Layer>): p is Layer => p.weight !== undefined

/** One symmetric per-tensor b-bit quantiser fitted to a weight matrix's current values. */
const weightQuantiser = (w: Value, bits: Size) => quantisationParams(unwrap(w) as Tensor, { bits, scheme: 'symmetric' })

/** The parameters with every weight matrix fake-quantised (straight-through in the backward pass). */
const fakeQuantised = (params: Params[], bits: Size): Params[] =>
  (params as unknown as Partial<Layer>[]).map((p) =>
    hasWeight(p) ? { ...p, weight: fakeQuantise(p.weight, weightQuantiser(p.weight, bits)) } : p,
  ) as unknown as Params[]

/** The parameters with every weight matrix rounded to b bits once (what is deployed). */
const rounded = (params: Params[], bits: Size): Params[] =>
  (params as unknown as Partial<Layer>[]).map((p) => {
    if (!hasWeight(p)) return p
    const w = unwrap(p.weight) as Tensor
    const q = weightQuantiser(w, bits)
    return { ...p, weight: dequantise(quantise(w, q), q) }
  }) as unknown as Params[]

const weightsOf = (params: Params[]) =>
  Float64Array.from(
    (params as unknown as Partial<Layer>[])
      .filter(hasWeight)
      .flatMap((p) => Array.from(toFlat(unwrap(p.weight) as Tensor))),
  )

const scalesOf = (params: Params[], bits: Size) =>
  (params as unknown as Partial<Layer>[]).filter(hasWeight).map((p) => weightQuantiser(p.weight, bits).scale[0])

/**
 * Train a ReLU MLP on binary labels by Adam, then for each bit width compare post-training quantisation (round the
 * trained weights once) with quantisation-aware training (fine-tune from the trained weights with fake-quantised
 * weights and the straight-through estimator, then round). Snapshots stream through both phases.
 */
export function* quantisationAwareTraining(
  data: { readonly x: Tensor; readonly y?: Tensor },
  options: QatOptions = {},
): Generator<QatSnapshot> {
  const { width = 16, depth = 2, steps = 1000, stepSize = 0.01, batchSize = 32, seed = 0 } = options
  const { qatSteps = 300, qatStepSize = 0.003 } = options
  const bitsList = options.bits ?? [2, 3, 4, 5, 6]
  if (!data.y) throw new DomainError('quantisationAwareTraining', 'quantisationAwareTraining: the data need labels y')
  const [n, d] = data.x.shape
  const y = toFlat(data.y)
  const target = fromData(Float64Array.from(y), [n, 1])
  const { model } = studyModel(d, width, depth)
  const lossAt =
    (bits: Size | null) =>
    (params: Params[], batch: { x: Tensor; y: Tensor }): Value =>
      binaryCrossEntropyWithLogits(model.apply(bits === null ? params : fakeQuantised(params, bits), batch.x), batch.y)
  const scalar = (v: Value) => {
    const u = unwrap(v)
    return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
  }
  const fullLoss = (params: Params[], bits: Size | null) => scalar(lossAt(bits)(params, { x: data.x, y: target }))
  const accuracy = (params: Params[]) => {
    const logits = toFlat(unwrap(model.apply(params, data.x)) as Tensor)
    let right = 0
    for (let i = 0; i < n; i++) right += (logits[i] > 0 ? 1 : 0) === y[i] ? 1 : 0
    return right / n
  }
  const loop = (bits: Size | null, eta: number) =>
    trainingLoop<Params[], { x: Tensor; y: Tensor }>({
      loss: lossAt(bits),
      data: { x: data.x, y: target },
      batchSize: Math.max(1, Math.min(batchSize, n)),
      optimizer: adamRule({ stepSize: eta }) as UpdateRule<unknown>,
    })
  const root = stream(`quantisation-aware-training-${seed}`)
  const total = steps + bitsList.length * qatSteps
  const snap: QatSnapshot = {
    done: 0,
    total,
    phase: 'training',
    step: [],
    loss: [],
    accuracy: NaN,
    layers: [],
    results: [],
  }
  const copy = (): QatSnapshot => ({ ...snap, step: [...snap.step], loss: [...snap.loss], results: [...snap.results] })

  // Floating-point training.
  const alg = loop(null, stepSize)
  let state = alg.init({ params: model.init(child(root, 'weights')) }, child(root, 'init'))
  const every = Math.max(1, Math.ceil(steps / 200))
  const yieldEvery = Math.max(5, Math.ceil(steps / 25))
  snap.step.push(0)
  snap.loss.push(fullLoss(state.params as Params[], null))
  yield copy()
  for (let t = 1; t <= steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    if (state.diverged) break
    if (t % every === 0 || t === steps) {
      snap.step.push(t)
      snap.loss.push(fullLoss(state.params as Params[], null))
    }
    snap.done = t
    if (t % yieldEvery === 0) yield copy()
  }
  const trained = state.params as Params[]
  snap.accuracy = accuracy(trained)
  snap.layers = (trained as unknown as Partial<Layer>[])
    .filter(hasWeight)
    .map((p) => (unwrap(p.weight) as Tensor).shape.slice(0, 2) as [number, number])
  snap.phase = 'fine-tuning'
  snap.done = steps

  // PTQ and QAT at each bit width, each fine-tuning run starting from the trained network.
  const record = Math.max(1, Math.ceil(qatSteps / 40))
  for (const bits of bitsList) {
    const result: QatBitResult = {
      bits,
      ptq: accuracy(rounded(trained, bits)),
      qat: NaN,
      step: [0],
      loss: [fullLoss(trained, bits)],
      weights: [weightsOf(trained)],
      scales: [scalesOf(trained, bits)],
    }
    const fine = loop(bits, qatStepSize)
    let s = fine.init({ params: trained }, child(root, 'qat-init', bits))
    for (let t = 1; t <= qatSteps; t++) {
      s = fine.step(s, { t, stream: child(root, 'qat', bits, t) })
      if (s.diverged) break
      if (t % record === 0 || t === qatSteps) {
        const p = s.params as Params[]
        result.step.push(t)
        result.loss.push(fullLoss(p, bits))
        result.weights.push(weightsOf(p))
        result.scales.push(scalesOf(p, bits))
      }
      snap.done++
      if (t % Math.max(5, Math.ceil(qatSteps / 10)) === 0) yield { ...copy(), results: [...snap.results, result] }
    }
    result.qat = accuracy(rounded(s.params as Params[], bits))
    snap.results.push(result)
    yield copy()
  }
  snap.phase = 'done'
  snap.done = total
  yield copy()
}
