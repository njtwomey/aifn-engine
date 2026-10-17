/**
 * Kingma, Rezende, Mohamed and Welling's M2 model ("Semi-Supervised Learning with Deep Generative Models", NeurIPS
 * 2014, §2.2 and §3.1.2) on small data, trained as a traceable algorithm, with a plain-data run for a worker.
 *
 * A class label $y \in \{0, \dots, K - 1\}$ has a uniform prior, a continuous code $\zvec \sim \Gauss(\zeros, \Imat)$
 * and a decoder $p_\theta(\xvec \mid y, \zvec)$; an encoder $q_\phi(\zvec \mid \xvec, y)$ and a classifier
 * $q_\phi(y \mid \xvec)$ approximate the posterior. The encoder and decoder are the module's conditional VAE
 * (`autoencoder({ kind: 'cvae' })`, which concatenates the one-hot label to the input and to the code); the classifier
 * is an `Mlp` to $K$ logits. The negative bound of a labelled pair is
 * $$
 * \Lcal(\xvec, y) = -\expect_{q(\zvec \mid \xvec, y)}[\log p(\xvec \mid y, \zvec)]
 *   + \KL(q(\zvec \mid \xvec, y) \,\|\, p(\zvec)) - \log p(y)
 * $$
 * (eq. 6), estimated with one reparameterised draw and the KL in closed form; that of an unlabelled input is
 * $\Ucal(\xvec) = \sum_y q(y \mid \xvec) \Lcal(\xvec, y) - \entropy[q(y \mid \xvec)]$ (eq. 7), summing over every
 * class exactly. Each step takes one update of all three networks on
 * $(\sum_{L} \Lcal + \tfrac{u}{B} \sum_{\text{batch}} \Ucal) / N + a \cdot \operatorname{mean}_{L}[-\log q(y \mid
 * \xvec)]$ with all $\ell$ labelled points, a batch of $B$ unlabelled ones and $N = \ell + u$. The last term is eq. 9's
 * classification loss $\alpha\, \expect_{\tilde p_l}[-\log q(y \mid \xvec)]$ divided by $N$: Kingma et al.'s
 * $\alpha = 0.1 N$ is $a = 0.1$, the default (with few labels a larger $a$, such as 5, keeps the classes from
 * swapping).
 *
 * Differences from the paper: Adam instead of AdaGrad; the reconstruction is Gaussian with a fixed standard deviation
 * (the cvae's `observationSd`) for real-valued toy data; no M1 stage. `mode: 'classifier'` trains only the classifier
 * on the labelled cross-entropy, the supervised baseline. States are plain data; a step is a pure function of its state
 * and stream.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, normals, permutation, stream, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  exp,
  float64Data,
  fromData,
  mean,
  mul,
  shapeOf,
  square,
  sub,
  sum,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live, type Algorithm } from 'aifn-compute/foundation/trace'
import { oneHot, softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { logSoftmax, softmax } from 'aifn-compute/numerics/special'
import { adamRule, applyUpdates, type UpdateRule } from 'aifn-compute/optim/first-order'
import {
  autoencoder,
  autoencoderEncode,
  decodeRaw,
  initAutoencoder,
  type Autoencoder,
  type AutoencoderParams,
} from './models'

/** The three networks of M2. */
export interface M2Model {
  /** Encoder $q(\zvec \mid \xvec, y)$ and decoder $p(\xvec \mid y, \zvec)$: the module's conditional VAE. */
  vae: Autoencoder
  /** Classifier $q(y \mid \xvec)$: inputs to $K$ logits. */
  classifier: Layer<Params[]>
  /** The number of classes $K$. */
  classes: number
}

/** M2's parameters, a pytree. */
export type M2Params = {
  /** The conditional VAE's encoder and decoder. */
  vae: AutoencoderParams
  /** The classifier's. */
  classifier: Params[]
}

/** Options of `m2Model`. */
export interface M2ModelOptions {
  /** The code width $L$ (default 1). */
  latent?: number
  /** Hidden widths of the encoder (the decoder mirrors them; default `[32, 32]`). */
  hidden?: readonly number[]
  /** Hidden widths of the classifier (default `[32, 32]`, tanh). */
  classifierHidden?: readonly number[]
  /** The decoder's Gaussian standard deviation $\sigma$ (default 0.1). */
  observationSd?: number
}

/**
 * The networks of M2 for inputs of width `inputs` and `classes` classes: a conditional VAE and a tanh `Mlp`
 * classifier with Xavier-uniform weights.
 *
 * @param inputs The input width $d$.
 * @param classes The number of classes $K$.
 * @param options The code width, the hidden widths and the decoder's standard deviation.
 * @returns The networks.
 *
 * @example Class probabilities and decoded points of an untrained M2
 * const model = m2Model(2, 2, { hidden: [8], classifierHidden: [8] })
 * const params = initM2(model, stream(1))
 * print('q(y | x):', m2ClassProbabilities(model, params, tensor([[0, 0], [1, 1]])))
 * print('decoded for class 1:', m2Decode(model, params, tensor([[-1], [1]]), 1))
 */
export function m2Model(inputs: number, classes: number, options: M2ModelOptions = {}): M2Model {
  return {
    vae: autoencoder({
      kind: 'cvae',
      inputs,
      classes,
      latent: options.latent ?? 1,
      hidden: options.hidden ?? [32, 32],
      observationSd: options.observationSd ?? 0.1,
    }),
    classifier: Mlp([inputs, ...(options.classifierHidden ?? [32, 32]), classes], {
      activation: 'tanh',
      init: xavierUniform(),
    }),
    classes,
  }
}

/**
 * Initial parameters of M2, from the stream's children `'vae'` and `'classifier'`.
 *
 * @param model The networks.
 * @param s The stream.
 * @returns The parameters.
 *
 * @example Parameters of both parts
 * const model = m2Model(2, 2, { hidden: [4], classifierHidden: [4] })
 * print('parts:', Object.keys(initM2(model, stream(1))))
 */
export function initM2(model: M2Model, s: Stream): M2Params {
  return { vae: initAutoencoder(model.vae, child(s, 'vae')), classifier: model.classifier.init(child(s, 'classifier')) }
}

/**
 * The number behind a scalar, traced or not.
 *
 * @param v A number or a rank-0 value.
 * @returns Its value (a tensor's first entry).
 */
const num = (v: Value): number => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : float64Data(r)[0]
}

/**
 * $\Lcal(\xvec_i, y_i)$ for every row, one reparameterised draw per row: the Gaussian negative log-likelihood of the
 * reconstruction, plus the closed-form KL divergence to $\Gauss(\zeros, \Imat)$, plus $\log K$ (the uniform label
 * prior). Differentiable in the parameters.
 *
 * @param model The networks.
 * @param params Their parameters (traced inside a gradient).
 * @param x The inputs, $n \times d$.
 * @param labels One-hot labels, $n \times K$.
 * @param eps The standard normal draws of the reparameterisation, $n \times L$.
 * @returns $n$ values of $\Lcal$.
 *
 * @example The bound of two labelled points
 * import { oneHot } from 'aifn-compute/learning/losses'
 * const model = m2Model(2, 2, { hidden: [4], classifierHidden: [4] })
 * const params = initM2(model, stream(1))
 * const x = tensor([[0, 0.5], [1, -0.5]])
 * print('L(x, y):', labelledBound(model, params, x, oneHot([0, 1], 2), normal(stream(2), 0, 1, { shape: [2, 1] })))
 */
export function labelledBound(model: M2Model, params: M2Params, x: Value, labels: Tensor, eps: Tensor): Value {
  const { vae } = model
  const d = vae.spec.inputs
  const sd = vae.spec.observationSd
  const { mean: mu, logVariance } = autoencoderEncode(vae, params.vae, x, labels)
  const lv = logVariance as Value
  const z = add(mu, mul(exp(mul(0.5, lv)), eps))
  const xhat = decodeRaw(vae, params.vae, z, labels)
  const reconstruction = add(
    mul(1 / (2 * sd * sd), sum(square(sub(x, xhat)), -1)),
    d * Math.log(sd * Math.sqrt(2 * Math.PI)),
  )
  const kl = mul(0.5, sum(sub(add(exp(lv), square(mu)), add(1, lv)), -1))
  return add(add(reconstruction, kl), Math.log(model.classes))
}

/**
 * $\Ucal = \sum_y q_y \Lcal_y - \entropy[q]$ for one input, from its class probabilities and per-class bounds; its
 * minimum over $q$ is $-\log \sum_y e^{-\Lcal_y}$, at $q = \operatorname{softmax}(-\Lcal)$.
 *
 * @param q The class probabilities $q(y \mid \xvec)$.
 * @param perClass The bounds $\Lcal(\xvec, y)$, one per class.
 * @returns $\Ucal(\xvec)$.
 *
 * @example The note's worked example and its minimum
 * print('U at q = (0.8, 0.2):', unlabelledBound([0.8, 0.2], [3.1, 5.4]))
 * const q = 1 / (1 + Math.exp(-2.3))
 * print('U at softmax(−L):', unlabelledBound([q, 1 - q], [3.1, 5.4]))
 * print('−log Σ e^−L:', -Math.log(Math.exp(-3.1) + Math.exp(-5.4)))
 */
export function unlabelledBound(q: readonly number[], perClass: readonly number[]): number {
  let u = 0
  for (let k = 0; k < q.length; k++) u += q[k] * perClass[k] + (q[k] > 0 ? q[k] * Math.log(q[k]) : 0)
  return u
}

/**
 * The classifier's class probabilities $q(y \mid \xvec)$, row-major.
 *
 * @param model The networks.
 * @param params Their parameters.
 * @param x The inputs, $n \times d$.
 * @returns Row-major $n \times K$ probabilities.
 *
 * @example Two rows of two probabilities
 * const model = m2Model(2, 2, { hidden: [4], classifierHidden: [4] })
 * print(m2ClassProbabilities(model, initM2(model, stream(1)), tensor([[0, 0], [1, 1]])))
 */
export function m2ClassProbabilities(model: M2Model, params: M2Params, x: Tensor): Float64Array {
  return Float64Array.from(float64Data(softmax(model.classifier.apply(params.classifier, x) as Tensor)))
}

/**
 * The decoder's means $\expect[\xvec \mid y, \zvec]$ at codes for one class.
 *
 * @param model The networks.
 * @param params Their parameters.
 * @param z The codes, $n \times L$.
 * @param y The class.
 * @returns Row-major $n \times d$ means.
 *
 * @example The same codes decoded for each class
 * const model = m2Model(2, 2, { hidden: [4], classifierHidden: [4] })
 * const params = initM2(model, stream(1))
 * for (const y of [0, 1]) print('class', y, m2Decode(model, params, tensor([[-1], [0], [1]]), y))
 */
export function m2Decode(model: M2Model, params: M2Params, z: Tensor, y: number): Float64Array {
  const n = shapeOf(z)[0]
  const labels = oneHot(new Array<number>(n).fill(y), model.classes)
  return Float64Array.from(float64Data(decodeRaw(model.vae, params.vae, z, labels) as Tensor))
}

/** The data of a run. */
export interface M2Problem {
  /** Labelled inputs, $\ell \times d$. */
  labelledX: Tensor
  /** Their classes in $0, \dots, K - 1$. */
  labelledY: ArrayLike<number>
  /** Unlabelled inputs, $u \times d$. */
  unlabelledX: Tensor
  /** The number of classes $K$. */
  classes: number
}

/** Options of `semiSupervisedVaeTraining`. */
export interface M2TrainingOptions {
  /** The networks. */
  model: M2Model
  /** `'m2'` (default) or `'classifier'`: the classifier alone on the labelled cross-entropy. */
  mode?: 'm2' | 'classifier'
  /**
   * $a = \alpha / N$, the weight of the mean labelled cross-entropy (default 0.1, Kingma et al.'s $\alpha = 0.1 N$).
   */
  classWeight?: number
  /** Unlabelled points per step (default 64, or all of them if fewer). */
  batch?: number
  /** The update rule (default Adam with step 3e-3). */
  optimizer?: UpdateRule<unknown>
}

/** The state after $t$ updates; the scalar parts describe the step that produced it (NaN at $t = 0$). */
export interface M2State extends Status {
  /** Updates made. */
  t: number
  /** The parameters. */
  params: M2Params
  /** The update rule's state. */
  optimizer: unknown
  /** The objective of the step. */
  loss: number
  /** Mean $\Lcal$ over the labelled points. */
  labelled: number
  /** Mean $\Ucal$ over the step's unlabelled batch. */
  unlabelled: number
  /** Mean labelled cross-entropy $-\log q(y \mid \xvec)$. */
  classification: number
  /** Whether the loss is not finite. */
  diverged: boolean
}

/**
 * M2 training (or its classifier-only baseline); see the file comment. Step $t$ draws the unlabelled batch from
 * `child(stream, 'batch')` and the reparameterisation noise from `child(stream, 'eps', 'labelled')` and
 * `child(stream, 'eps', k)` for class $k$. The state flags `diverged` when the loss is not finite.
 *
 * @param problem The labelled and unlabelled inputs and the number of classes.
 * @param options The networks, the mode, the classification weight, the batch and the update rule.
 * @returns The algorithm; start it from `{ params }`, or from `undefined` to initialise from the init stream's child
 *   `'params'`.
 *
 * @example Ten M2 updates on six points
 * const problem = {
 *   labelledX: tensor([[-1, 0.5], [1, -0.5]]),
 *   labelledY: [0, 1],
 *   unlabelledX: tensor([[-1.2, 0.4], [-0.8, 0.6], [0.9, -0.4], [1.1, -0.6]]),
 *   classes: 2,
 * }
 * const model = m2Model(2, 2, { hidden: [8], classifierHidden: [8] })
 * const s = run(semiSupervisedVaeTraining(problem, { model, classWeight: 5 }), undefined, 10)
 * print('loss:', s.loss, ' labelled bound:', s.labelled, ' unlabelled bound:', s.unlabelled)
 */
export function semiSupervisedVaeTraining(
  problem: M2Problem,
  options: M2TrainingOptions,
): Algorithm<{ params: M2Params } | undefined, M2State> {
  const { model } = options
  const mode = options.mode ?? 'm2'
  const a = options.classWeight ?? 0.1
  const rule = options.optimizer ?? adamRule({ stepSize: 3e-3 })
  const K = problem.classes
  const nL = shapeOf(problem.labelledX)[0]
  const [nU, d] = shapeOf(problem.unlabelledX) as [number, number]
  const B = Math.min(nU, options.batch ?? 64)
  const N = nL + nU
  const L = model.vae.spec.latent
  const yL = Array.from(problem.labelledY)
  const hotL = oneHot(yL, K)
  const hotU = Array.from({ length: K }, (_, k) => oneHot(new Array<number>(B).fill(k), K))
  const U = float64Data(problem.unlabelledX)
  const columns = Array.from({ length: K }, (_, k) =>
    fromData(
      Float64Array.from({ length: B * K }, (_, e) => (e % K === k ? 1 : 0)),
      [B, K],
    ),
  )

  return {
    name: `semiSupervisedVaeTraining(${mode})`,
    init(start, s) {
      const params = start?.params ?? initM2(model, child(s, 'params'))
      return {
        t: 0,
        params,
        optimizer: rule.init(params),
        loss: NaN,
        labelled: NaN,
        unlabelled: NaN,
        classification: NaN,
        diverged: false,
      }
    },
    step(state, ctx) {
      const s = ctx.stream
      const order = float64Data(permutation(child(s, 'batch'), nU))
      const xb = new Float64Array(B * d)
      for (let r = 0; r < B; r++) xb.set(U.subarray(order[r] * d, order[r] * d + d), r * d)
      const xU = fromData(xb, [B, d])
      const epsL = normals(child(s, 'eps', 'labelled'), [nL, L])
      const epsU = Array.from({ length: K }, (_, k) => normals(child(s, 'eps', k), [B, L]))
      const parts = { labelled: NaN, unlabelled: NaN, classification: NaN }

      const objective = (p: M2Params): Value => {
        const ce = softmaxCrossEntropy(model.classifier.apply(p.classifier, problem.labelledX), yL)
        parts.classification = num(ce)
        if (mode === 'classifier') return ce
        const lab = mean(labelledBound(model, p, problem.labelledX, hotL, epsL))
        parts.labelled = num(lab)
        const logits = model.classifier.apply(p.classifier, xU)
        const q = softmax(logits)
        const logq = logSoftmax(logits)
        // Σ_y q(y|x) L(x, y) − H(q), summed over the batch, one column of q per class.
        let total: Value = sum(mul(q, logq))
        for (let k = 0; k < K; k++) {
          const Lk = labelledBound(model, p, xU, hotU[k], epsU[k])
          total = add(total, sum(mul(sum(mul(q, columns[k]), -1), Lk)))
        }
        parts.unlabelled = num(total) / B
        return add(mul(1 / N, add(mul(nL, lab), mul(nU / B, total))), mul(a, ce))
      }

      const { value, grad: g } = valueAndGrad(objective)(state.params)
      const loss = num(value)
      const { updates, state: opt } = rule.update(
        g as unknown as Params,
        state.optimizer,
        state.params as unknown as Params,
      )
      const params = applyUpdates(state.params as unknown as Params, updates) as unknown as M2Params
      return {
        t: state.t + 1,
        params,
        optimizer: opt,
        loss,
        labelled: parts.labelled,
        unlabelled: parts.unlabelled,
        classification: parts.classification,
        diverged: !Number.isFinite(loss),
      }
    },
  }
}

// ── A plain-data run for a worker ────────────────────────────────────────────────────────────────────────────────────

/** Options of `semiSupervisedVaeRun`; plain data, so a worker task can carry them. */
export type SemiSupervisedVaeRunOptions = M2ModelOptions & {
  /** `'m2'` (default) or `'classifier'`, the supervised baseline. */
  mode?: 'm2' | 'classifier'
  /** The weight $a$ of the labelled cross-entropy (default 0.1). */
  classWeight?: number
  /** Unlabelled points per step (default 64). */
  batch?: number
  /** Updates (default 400). */
  steps?: number
  /** Adam's step size (default 3e-3). */
  rate?: number
  /** The key of the run's root stream, from which the initial parameters and every step derive (default 0). */
  seed?: number | string
  /** Steps between checkpoints (default 10); step 0 and the last are always kept. */
  every?: number
  /** True classes of the unlabelled points, for the accuracy. */
  truth?: ArrayLike<number>
  /** Further inputs ($m \times d$, e.g. a grid) whose class probabilities each checkpoint holds. */
  evaluate?: Tensor
  /** Codes ($c \times L$) each checkpoint decodes for every class. */
  codes?: Tensor
}

/** One checkpoint of `semiSupervisedVaeRun`: plain arrays only. */
export type SemiSupervisedVaeCheckpoint = {
  /** Updates made (0 for the initial networks). */
  step: number
  /** The step's objective (NaN at step 0). */
  loss: number
  /** Mean $\Lcal$ over the labelled points. */
  labelled: number
  /** Mean $\Ucal$ over the step's batch. */
  unlabelled: number
  /** Mean labelled cross-entropy. */
  classification: number
  /** $q(y \mid \xvec)$ on the unlabelled points, row-major $u \times K$. */
  probabilities: Float64Array
  /** $q(y \mid \xvec)$ on the `evaluate` inputs, row-major (empty without them). */
  evaluate: Float64Array
  /** The decoder's means at `codes` for each class, row-major $c \times d$ each (empty without codes). */
  decoded: Float64Array[]
  /** The arg max accuracy on the unlabelled points (NaN without `truth`). */
  accuracy: number
  /** Whether training diverged, which ends the run. */
  diverged: boolean
}

/**
 * Train M2 (or the classifier alone) from plain data and yield a checkpoint at step 0, every `every` steps, at the
 * last step and on divergence (which ends the run). The algorithm starts from `undefined` (parameters from the init
 * stream) under the root stream `stream(seed)`, with Adam at step size `rate`. Deterministic in `seed`.
 *
 * @param problem The labelled and unlabelled inputs and the number of classes.
 * @param options The networks, the mode and weight, the training, the seed and what each checkpoint holds.
 * @returns A generator of checkpoints.
 *
 * @example Accuracy of a short run on six points
 * const problem = {
 *   labelledX: tensor([[-1, 0.5], [1, -0.5]]),
 *   labelledY: [0, 1],
 *   unlabelledX: tensor([[-1.2, 0.4], [-0.8, 0.6], [0.9, -0.4], [1.1, -0.6]]),
 *   classes: 2,
 * }
 * const options = { hidden: [8], classifierHidden: [8], steps: 20, every: 10, truth: [0, 0, 1, 1] }
 * for (const c of semiSupervisedVaeRun(problem, options)) print('step', c.step, ' accuracy', c.accuracy)
 */
export function* semiSupervisedVaeRun(
  problem: M2Problem,
  options: SemiSupervisedVaeRunOptions = {},
): Generator<SemiSupervisedVaeCheckpoint> {
  const { steps = 400, rate = 3e-3, seed = 0, every = 10, truth, evaluate, codes } = options
  const K = problem.classes
  const model = m2Model(shapeOf(problem.unlabelledX)[1], K, options)
  const alg = semiSupervisedVaeTraining(problem, {
    model,
    mode: options.mode,
    classWeight: options.classWeight,
    batch: options.batch,
    optimizer: adamRule({ stepSize: rate }),
  })
  const accuracy = (p: Float64Array): number => {
    if (!truth) return NaN
    let right = 0
    for (let i = 0; i < truth.length; i++) {
      let best = 0
      for (let k = 1; k < K; k++) if (p[i * K + k] > p[i * K + best]) best = k
      if (best === truth[i]) right++
    }
    return truth.length ? right / truth.length : NaN
  }
  const it = live(alg, undefined, { stream: stream(seed) })
  for (let k = 0; k <= steps; k++) {
    const next = it.next()
    if (next.done) return
    const st = next.value.state
    if (k % every === 0 || k === steps || st.diverged) {
      const probabilities = m2ClassProbabilities(model, st.params, problem.unlabelledX)
      yield {
        step: k,
        loss: st.loss,
        labelled: st.labelled,
        unlabelled: st.unlabelled,
        classification: st.classification,
        probabilities,
        evaluate: evaluate ? m2ClassProbabilities(model, st.params, evaluate) : new Float64Array(0),
        decoded: codes ? Array.from({ length: K }, (_, y) => m2Decode(model, st.params, codes, y)) : [],
        accuracy: accuracy(probabilities),
        diverged: st.diverged,
      }
    }
    if (st.diverged) return
  }
}
