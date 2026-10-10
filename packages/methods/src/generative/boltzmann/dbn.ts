/**
 * A deep belief network, lite (Hinton, Osindero and Teh, 2006; Bengio et al., 2007): a stack of RBMs trained greedily,
 * one layer at a time. Layer 1 is an RBM on the data; its hidden probabilities $p(\hvec^{(1)} \mid \vvec)$ become the
 * data of layer 2, and so on. The trained stack is a generative model whose top two layers form an RBM (an undirected
 * associative memory) and whose lower layers are directed sigmoid belief layers
 * $p(\hvec^{(l-1)} \mid \hvec^{(l)}) = \sigma(\avec^{(l)} + \Wmat^{(l)} \hvec^{(l)})$, each using the generative
 * weights of the RBM that was trained on it. Sampling runs Gibbs in the top RBM, then one ancestral pass down.
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
  /** The RBMs, bottom (on the data) to top. */
  readonly layers: readonly Rbm[]
}

/**
 * The recognition (up) pass: $p(\hvec^{(l)} = 1 \mid \hvec^{(l-1)})$ for each layer from one visible vector
 * ($\hvec^{(0)} = \vvec$), layer by layer. Each layer is fed the probabilities of the one below, not samples (a
 * mean-field pass).
 *
 * @param model The network.
 * @param v The visible vector, $D$ values.
 * @returns One array of hidden probabilities per layer, bottom to top.
 *
 * @example Hidden probabilities up a stack of two RBMs
 * const model = { layers: [rbm(stream(1), 4, 3), rbm(stream(2), 3, 2)] }
 * const [h1, h2] = dbnUp(model, [1, 0, 1, 0])
 * print('layer 1:', h1)
 * print('layer 2:', h2)
 */
export function dbnUp(model: Dbn, v: ArrayLike<number>): Float64Array[] {
  const out: Float64Array[] = []
  let x: ArrayLike<number> = v
  for (const layer of model.layers) {
    x = hiddenProbabilities(layer, x)
    out.push(x as Float64Array)
  }
  return out
}

/**
 * Sample 0/1 units from probabilities: unit $i$ is 1 when $u_i < p_i$.
 *
 * @param p The units' probabilities of being 1.
 * @param u Uniforms on $[0, 1)$, at least as many as `p`.
 * @returns The 0/1 states, one per entry of `p`.
 */
const bernoulli = (p: ArrayLike<number>, u: ArrayLike<number>) => Float64Array.from(p, (q, i) => (u[i] < q ? 1 : 0))

/**
 * Samples from the DBN: `count` chains of `gibbsSteps` block Gibbs steps in the top RBM from uniform random 0/1
 * states, then one ancestral pass down through the directed layers, sampling each layer but the bottom one. A network
 * of one layer samples its hidden units once more and goes down to the visible units. Throws `DomainError` for a
 * network with no layers.
 *
 * @param model The network.
 * @param s The stream; chain $c$ starts from `child(s, 'start', c)` and runs on `child(s, 'gibbs', c)`.
 * @param options `count`, the number of samples (default 16), and `gibbsSteps`, the Gibbs steps of each chain in the
 *   top RBM (default 200).
 * @returns The bottom layer's probabilities $p(\vvec = 1 \mid \hvec^{(1)})$ of each sample, row-major
 *   `count` $\times D$.
 *
 * @example Two samples from a stack of two untrained RBMs
 * // Untrained, the weights are small and the probabilities near 1/2 (exactly 1/2 when the units above are all 0).
 * const model = { layers: [rbm(stream(1), 4, 3), rbm(stream(2), 3, 2)] }
 * print('p(v = 1 | h), two rows of 4:', dbnSample(model, stream(3), { count: 2, gibbsSteps: 10 }))
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
  /** Hidden layer sizes, bottom to top (default $[32, 16]$). */
  layers?: readonly number[]
  /** CD epochs per layer (default 200). */
  epochs?: number
  /** Gibbs steps $k$ per CD update (default 1). */
  k?: number
  /** The CD step size (default 0.1). */
  learningRate?: number
  /** Rows per CD minibatch (default 10). */
  batchSize?: number
  /** Gibbs steps in the top RBM before each shown sample (default 200). */
  sampleSteps?: number
  /** Samples shown at each checkpoint (default 16). */
  samples?: number
  /** The discriminative fine-tune's Adam epochs (default 300; 0 skips it). */
  fineTuneEpochs?: number
  /** The fine-tune's Adam step size (default 0.01). */
  fineTuneRate?: number
  /** Labelled rows per class the fine-tune trains on, taken from the training rows (default 2). */
  labelsPerClass?: number
  /** The share of rows held out to test the fine-tune, when there are labels (default 0.5). */
  testFraction?: number
  /** The seed of the run's stream (default 0). */
  seed?: number | string
}

/** One checkpoint of the greedy phase: after `epoch` epochs of layer `layer`. */
export interface DbnCheckpoint {
  /** The layer being trained, from 0 at the bottom. */
  layer: number
  /** Its epochs so far (0 before its training). */
  epoch: number
  /** The bottom RBM's weights, row-major $D \times H_1$ (the layer being trained, while it is the bottom one). */
  weights: Float64Array
  /** Samples from the stack trained so far (`dbnSample`'s probabilities), `samples` rows of $D$, row-major. */
  samples: Float64Array
}

/** A DBN run so far. */
export interface DbnRun {
  /** The number of visible units $D$. */
  visible: number
  /** The hidden layer sizes, bottom to top. */
  sizes: number[]
  /**
   * Per layer, the mean reconstruction error per row in each epoch, from epoch 0 (NaN), as
   * `contrastiveDivergenceStep` reports it.
   */
  reconstructionError: number[][]
  /**
   * Per layer, the exact log-likelihood of its RBM on its input (layers of at most 16 units) at epoch 0 and the
   * checkpoint epochs, NaN at the others. Above the bottom layer the input is the probabilities of the layer below,
   * so the value is $-F - \log Z$ evaluated there, not a likelihood of binary data.
   */
  logLikelihood: number[][]
  /** The checkpoints so far, every layer's: epoch 0 and about every tenth of its epochs. */
  checkpoints: DbnCheckpoint[]
  /**
   * The fine-tune, null until it starts (and when there are no labels): test accuracy per epoch, from epoch 0, from
   * the DBN's weights and from random weights, and the labelled and test row counts.
   */
  fineTune: { pretrained: number[]; random: number[]; labelled: number; test: number } | null
  /** What the run is doing. */
  phase: 'pretraining' | 'fine-tuning' | 'done'
  /** True at the end of the run. */
  done: boolean
}

/** A sigmoid MLP's parameters: per layer, the weights (inputs $\times$ outputs) and the biases. */
type MlpParams = { weight: Tensor; bias: Tensor }[]

/**
 * Greedy layer-wise training of a DBN, then the optional discriminative fine-tune (see the file's introduction), as a
 * generator of snapshots. With labels, a random `testFraction` of the rows is held out first: pretraining sees the
 * other rows only, without their labels, and the held-out rows score the fine-tune. Entries are clipped to $[0, 1]$.
 * Throws `DomainError` when `layers` is empty. Deterministic in `seed`.
 *
 * @param data The rows, $n \times D$ with entries in $[0, 1]$: a matrix, or `{ x, y }` with integer class labels `y`
 *   ($n$ of them; without them there is no fine-tune).
 * @param options The layer sizes, the CD settings, the samples shown, the fine-tune and the seed.
 * @returns A generator of `DbnRun` snapshots: at every checkpoint of pretraining, about every tenth of the fine-tune,
 *   and at the end (the last has `done` set).
 *
 * @example Pretrain two layers on two patterns, then fine-tune on one labelled row of each
 * const x = tensor([
 *   [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1],
 *   [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1],
 * ])
 * const y = tensor([0, 1, 0, 1, 0, 1, 0, 1])
 * const options = { layers: [3, 2], epochs: 50, batchSize: 2, learningRate: 0.5, samples: 2, sampleSteps: 10 }
 * let run
 * for (const r of dbnRun({ x, y }, { ...options, fineTuneEpochs: 20, labelsPerClass: 1 })) run = r
 * print('layer 1 log-likelihood at the start and the end:', run.logLikelihood[0][0], run.logLikelihood[0].at(-1))
 * print('test accuracy by fine-tune epoch, from the DBN:', run.fineTune.pretrained)
 * print('… and from random weights:', run.fineTune.random)
 * print('phase:', run.phase)
 */
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
