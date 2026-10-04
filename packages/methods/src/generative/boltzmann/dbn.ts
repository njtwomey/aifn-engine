/**
 * A deep belief network, lite (Hinton, Osindero and Teh, 2006; Bengio et al., 2007): a stack of RBMs trained greedily,
 * one layer at a time. Layer 1 is an RBM on the data; its hidden probabilities p(h¹ | v) become the data of layer 2,
 * and so on. The trained stack is a generative model whose top two layers form an RBM (an undirected associative
 * memory) and whose lower layers are directed sigmoid belief layers p(hˡ⁻¹ | hˡ) = σ(aˡ + Wˡ hˡ), each using the
 * generative weights of the RBM that was trained on it. Sampling runs Gibbs in the top RBM, then one ancestral pass
 * down.
 *
 * Each added layer improves a variational lower bound on the data's log-likelihood when it is initialised from the
 * layer below's transposed weights (Hinton, Osindero and Teh, 2006, §4); here new layers start from small random
 * weights, which keeps the greedy procedure and drops that guarantee (hence "lite"). The up-down (wake–sleep)
 * fine-tuning of the generative model is omitted. The optional fine-tune is discriminative instead (Hinton and
 * Salakhutdinov, 2006; Bengio et al., 2007): the stack's recognition weights initialise a sigmoid MLP with a softmax
 * head, trained by backpropagation on a few labelled rows against the same network from random weights.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream, units, type Stream } from 'aifn-compute/foundation/random'
import {
  dense,
  fromData,
  toFlat,
  unwrap,
  type MatrixLike,
  type Tensor,
  type Value,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { sigmoid } from 'aifn-compute/numerics/special'
import { linear } from 'aifn-compute/nn/layers'
import { adamRule, applyUpdates, type UpdateRule } from 'aifn-compute/optim/first-order'
import {
  contrastiveDivergenceStep,
  gibbsChain,
  hiddenProbabilities,
  rbm,
  rbmLogLikelihood,
  visibleProbabilities,
  type Rbm,
} from './rbm'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A deep belief network: RBMs whose visible layer is the hidden layer of the one before. */
export interface Dbn {
  readonly layers: readonly Rbm[]
}

/** The recognition (up) pass: p(hˡ | hˡ⁻¹) for each layer from one visible vector, layer by layer. */
export function dbnUp(model: Dbn, v: ArrayLike<number>): Float64Array[] {
  const out: Float64Array[] = []
  let x: ArrayLike<number> = v
  for (const layer of model.layers) {
    x = hiddenProbabilities(layer, x)
    out.push(x as Float64Array)
  }
  return out
}

/** Sample 0/1 units from probabilities. */
const bernoulli = (p: ArrayLike<number>, u: ArrayLike<number>) => Float64Array.from(p, (q, i) => (u[i] < q ? 1 : 0))

/**
 * Samples from the DBN: `count` chains of `gibbsSteps` block Gibbs steps in the top RBM from random states, then one
 * ancestral pass down through the directed layers. Returns the bottom layer's probabilities p(v | h¹) per sample, as
 * rows [count × D].
 */
export function dbnSample(model: Dbn, s: Stream, options: { count?: number; gibbsSteps?: number } = {}): Float64Array {
  const { count = 16, gibbsSteps = 200 } = options
  const L = model.layers.length
  if (L === 0) throw new DomainError('dbnSample', 'dbnSample: the network has no layers')
  const top = model.layers[L - 1]
  const D = model.layers[0].visible
  const out = new Float64Array(count * D)
  for (let c = 0; c < count; c++) {
    const start = Float64Array.from(units(child(s, 'start', c), top.visible), (q) => (q < 0.5 ? 1 : 0))
    const chain = gibbsChain(top, start, gibbsSteps, child(s, 'gibbs', c))
    let h: Float64Array = chain.v
    if (L === 1) h = visibleProbabilities(top, bernoulli(chain.h, units(child(s, 'top', c), top.hidden)))
    for (let l = L - 2; l >= 0; l--) {
      const p = visibleProbabilities(model.layers[l], h)
      h = l === 0 ? p : bernoulli(p, units(child(s, 'down', c, l), p.length))
    }
    out.set(h, c * D)
  }
  return out
}

/** Options of `dbnRun`. */
export interface DbnRunOptions {
  /** Hidden layer sizes, bottom to top (default [32, 16]). */
  layers?: readonly number[]
  /** CD epochs per layer (default 200), Gibbs steps k (default 1), step size (default 0.1), minibatch rows (default 10). */
  epochs?: number
  k?: number
  learningRate?: number
  batchSize?: number
  /** Gibbs steps in the top RBM before each shown sample (default 200), and samples shown (default 16). */
  sampleSteps?: number
  samples?: number
  /**
   * The discriminative fine-tune: Adam epochs (default 300; 0 skips it), step size (default 0.01), labelled rows per
   * class taken from the training half (default 2), and the share of rows held out to test (default 0.5).
   */
  fineTuneEpochs?: number
  fineTuneRate?: number
  labelsPerClass?: number
  testFraction?: number
  seed?: number | string
}

/** One checkpoint of the greedy phase: after `epoch` epochs of layer `layer`. */
export interface DbnCheckpoint {
  layer: number
  epoch: number
  /** Layer 1's weights (D × H₁, row-major) and samples from the stack trained so far (rows of D). */
  weights: Float64Array
  samples: Float64Array
}

/** A DBN run so far. */
export interface DbnRun {
  visible: number
  sizes: number[]
  /**
   * Per layer, the reconstruction error per epoch, and the exact log-likelihood of its RBM on its input (layers of at
   * most 16 units) at the checkpoint epochs, NaN at the others.
   */
  reconstructionError: number[][]
  logLikelihood: number[][]
  checkpoints: DbnCheckpoint[]
  /** Fine-tune: test accuracy per epoch from the DBN's weights and from random weights; labelled and test row counts. */
  fineTune: { pretrained: number[]; random: number[]; labelled: number; test: number } | null
  phase: 'pretraining' | 'fine-tuning' | 'done'
  done: boolean
}

type MlpParams = { weight: Tensor; bias: Tensor }[]

/** Greedy layer-wise training of a DBN on rows x [n, D] in [0, 1], then the optional fine-tune (module docs). */
export function* dbnRun(
  data: { x: MatrixLike; y?: VectorLike } | MatrixLike,
  options: DbnRunOptions = {},
): Generator<DbnRun, DbnRun> {
  const withLabels = typeof data === 'object' && data !== null && 'x' in data
  const x = withLabels ? (data as { x: MatrixLike }).x : (data as MatrixLike)
  const yIn = withLabels ? (data as { y?: VectorLike }).y : undefined
  const { layers = [32, 16], epochs = 200, k = 1, learningRate = 0.1, batchSize = 10 } = options
  const { sampleSteps = 200, samples = 16, fineTuneEpochs = 300, fineTuneRate = 0.01 } = options
  const { labelsPerClass = 2, testFraction = 0.5, seed = 0 } = options
  if (layers.length === 0) throw new DomainError('dbnRun', 'dbnRun: needs at least one hidden layer')
  const V = dense.toMatrixF64(x, 'dbnRun')
  const D = V.n
  const root = stream(seed)
  const y = yIn === undefined ? null : Int32Array.from(dense.toF64(yIn, 'dbnRun'))
  // The training half pretrains (without labels); the test half only scores the fine-tune.
  const shuffled = Array.from({ length: V.m }, (_, r) => r)
  const su = units(child(root, 'split'), V.m)
  shuffled.sort((p, q) => su[p] - su[q])
  const nTest = y ? Math.round(testFraction * V.m) : 0
  const testRows = shuffled.slice(0, nTest)
  const trainRows = shuffled.slice(nTest)
  const rows = (ids: number[]) => {
    const out = new Float64Array(ids.length * D)
    ids.forEach((r, q) => {
      for (let i = 0; i < D; i++) out[q * D + i] = Math.min(1, Math.max(0, V.data[r * D + i]))
    })
    return out
  }
  const trained: Rbm[] = []
  const run: DbnRun = {
    visible: D,
    sizes: [...layers],
    reconstructionError: layers.map(() => []),
    logLikelihood: layers.map(() => []),
    checkpoints: [],
    fineTune: null,
    phase: 'pretraining',
    done: false,
  }
  const snapshot = (): DbnRun => ({
    ...run,
    reconstructionError: run.reconstructionError.map((r) => [...r]),
    logLikelihood: run.logLikelihood.map((r) => [...r]),
    checkpoints: run.checkpoints.slice(),
    fineTune: run.fineTune && {
      ...run.fineTune,
      pretrained: [...run.fineTune.pretrained],
      random: [...run.fineTune.random],
    },
  })
  const checkpoint = (layer: number, epoch: number, current: Rbm) => {
    const stack: Dbn = { layers: [...trained, current] }
    run.checkpoints.push({
      layer,
      epoch,
      weights: Float64Array.from((trained[0] ?? current).W),
      samples: dbnSample(stack, child(root, 'samples', layer, epoch), { count: samples, gibbsSteps: sampleSteps }),
    })
  }
  let input = rows(trainRows)
  let width = D
  const every = Math.max(1, Math.round(epochs / 10))
  for (let l = 0; l < layers.length; l++) {
    const n = input.length / width
    const X = fromData(input, [n, width])
    let model = rbm(child(root, 'init', l), width, layers[l])
    const exact = layers[l] <= 16
    run.logLikelihood[l].push(exact ? rbmLogLikelihood(model, X) : NaN)
    run.reconstructionError[l].push(NaN)
    checkpoint(l, 0, model)
    yield snapshot()
    for (let e = 1; e <= epochs; e++) {
      const order = Array.from({ length: n }, (_, r) => r)
      const u = units(child(root, 'order', l, e), n)
      order.sort((p, q) => u[p] - u[q])
      let err = 0
      for (let b = 0; b < n; b += batchSize) {
        const ids = order.slice(b, b + batchSize)
        const batch = new Float64Array(ids.length * width)
        ids.forEach((r, q) => batch.set(input.subarray(r * width, (r + 1) * width), q * width))
        const step = contrastiveDivergenceStep(
          model,
          fromData(batch, [ids.length, width]),
          child(root, 'cd', l, e, b),
          {
            k,
            learningRate,
          },
        )
        model = step.rbm
        err += (step.reconstructionError * ids.length) / n
      }
      // The exact likelihood costs 2ᴴ terms per row, so it is computed at checkpoints only (NaN between).
      const shown = e % every === 0 || e === epochs
      run.logLikelihood[l].push(exact && shown ? rbmLogLikelihood(model, X) : NaN)
      run.reconstructionError[l].push(err)
      if (shown) {
        checkpoint(l, e, model)
        yield snapshot()
      }
    }
    trained.push(model)
    // The next layer's data: this layer's hidden probabilities.
    const next = new Float64Array(n * layers[l])
    for (let r = 0; r < n; r++)
      next.set(hiddenProbabilities(model, input.subarray(r * width, (r + 1) * width)), r * layers[l])
    input = next
    width = layers[l]
  }
  if (!y || fineTuneEpochs <= 0) {
    run.phase = 'done'
    run.done = true
    const last = snapshot()
    yield last
    return last
  }
  // Fine-tune: a few labelled rows per class from the training half.
  const classes = Math.max(...y) + 1
  const taken = new Array<number>(classes).fill(0)
  const labelled = trainRows.filter((r) => taken[y[r]] < labelsPerClass && ++taken[y[r]] > 0)
  const xl = fromData(rows(labelled), [labelled.length, D])
  const yl = fromData(
    Int32Array.from(labelled, (r) => y[r]),
    [labelled.length],
  )
  const xt = fromData(rows(testRows), [testRows.length, D])
  const ytest = testRows.map((r) => y[r])
  const sizes = [D, ...layers]
  const head = (s: Stream): MlpParams[number] => {
    const H = layers[layers.length - 1]
    const r = rbm(s, H, classes)
    return { weight: fromData(r.W, [H, classes]), bias: fromData(new Float64Array(classes), [classes]) }
  }
  const pretrained: MlpParams = [
    ...trained.map((m) => ({ weight: fromData(m.W, [m.visible, m.hidden]), bias: fromData(m.b, [m.hidden]) })),
    head(child(root, 'head')),
  ]
  // The same architecture from the same random scale as an untrained RBM (N(0, 0.1²) weights, zero biases).
  const random: MlpParams = [
    ...layers.map((h, l) => {
      const m = rbm(child(root, 'random', l), sizes[l], h)
      return { weight: fromData(m.W, [m.visible, m.hidden]), bias: fromData(m.b, [h]) }
    }),
    head(child(root, 'head')),
  ]
  const logits = (p: MlpParams, input: Tensor): Value => {
    let h: Value = input
    p.forEach((layer, l) => {
      h = linear(h, layer.weight, layer.bias)
      if (l < p.length - 1) h = sigmoid(h)
    })
    return h
  }
  const accuracy = (p: MlpParams) => {
    const z = toFlat(unwrap(logits(p, xt)) as Tensor)
    let hit = 0
    ytest.forEach((c, i) => {
      let best = 0
      for (let j = 1; j < classes; j++) if (z[i * classes + j] > z[i * classes + best]) best = j
      if (best === c) hit++
    })
    return hit / ytest.length
  }
  const loss = (p: MlpParams): Value => softmaxCrossEntropy(logits(p, xl), yl, { reduction: 'mean' })
  const trainer = (init: MlpParams) => {
    const rule = adamRule({ stepSize: fineTuneRate }) as UpdateRule<unknown>
    let state = rule.init(init as unknown as Params)
    return (p: MlpParams): MlpParams => {
      const { grad } = valueAndGrad(loss, {})(p)
      const u = rule.update(grad as unknown as Params, state as never)
      state = u.state
      return applyUpdates(p as unknown as Params, u.updates) as unknown as MlpParams
    }
  }
  run.phase = 'fine-tuning'
  run.fineTune = {
    pretrained: [accuracy(pretrained)],
    random: [accuracy(random)],
    labelled: labelled.length,
    test: nTest,
  }
  const stepP = trainer(pretrained)
  const stepR = trainer(random)
  let pP = pretrained
  let pR = random
  const chunk = Math.max(1, Math.round(fineTuneEpochs / 10))
  for (let e = 1; e <= fineTuneEpochs; e++) {
    pP = stepP(pP)
    pR = stepR(pR)
    run.fineTune.pretrained.push(accuracy(pP))
    run.fineTune.random.push(accuracy(pR))
    if (e % chunk === 0 && e < fineTuneEpochs) yield snapshot()
  }
  run.phase = 'done'
  run.done = true
  const last = snapshot()
  yield last
  return last
}
