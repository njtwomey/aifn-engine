/**
 * A classifier read as an energy-based model: JEM (Grathwohl et al., 2019, "Your classifier is secretly an energy
 * based model and you should treat it like one").
 *
 * A softmax classifier with logits $f(\xvec) \in \reals^K$ sets
 * $p(y \mid \xvec) = \exp(f(\xvec)_y) / \sum_{y'} \exp(f(\xvec)_{y'})$. Adding any $c(\xvec)$ to every logit leaves
 * $p(y \mid \xvec)$ unchanged, so cross-entropy training fixes the logits only up to that shift. JEM spends the free
 * degree of freedom on the density of $\xvec$: it reads the same logits as $p(\xvec, y) \propto \exp(f(\xvec)_y)$,
 * whence $p(\xvec) \propto \sum_y \exp(f(\xvec)_y) = \exp(-E(\xvec))$ with the energy
 * $E(\xvec) = -\operatorname{logsumexp}_y f(\xvec)_y$, and $p(y \mid \xvec)$ is still the softmax. Training maximises
 * $\log p(y \mid \xvec) + \log p(\xvec)$: cross-entropy plus the contrastive-divergence term, with negatives from
 * persistent short-run Langevin (`aifn-compute/nn/training`'s `contrastiveDivergence`). Class-conditional samples come
 * from $p(\xvec \mid y) \propto \exp(f(\xvec)_y)$.
 *
 * Points are the rows of an $n \times d$ batch, and a classifier's parameters are those of its `layer`
 * (`net.layer.init(s)`).
 */

import { grad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  fromData,
  logsumexp,
  neg,
  slice,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { PersistentLangevinOptions } from 'aifn-compute/inference/stochastic'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import type { Activation } from 'aifn-compute/nn/functional'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { contrastiveDivergence, type ContrastiveDivergenceState } from 'aifn-compute/nn/training'
import type { UpdateRule } from 'aifn-compute/optim/first-order'

/** A classifier network, points $n \times d$ to logits $n \times K$. */
export type Classifier = {
  /** The network, an MLP; `layer.init(s)` draws its parameters. */
  readonly layer: Layer<Params[]>
  /** The number of classes $K$. */
  readonly classes: number
  /** The input dimension $d$. */
  readonly dimension: number
}

/** Options of `classifier`. */
export type ClassifierOptions = {
  /** Hidden widths. Default $[64, 64]$. */
  hidden?: readonly number[]
  /** Activation. Default SiLU (smooth, so the energy's gradient in x is too, as Langevin needs). */
  activation?: Activation
}

/**
 * An MLP classifier of $d$-dimensional points into $K$ classes, without parameters (`net.layer.init(s)` draws them).
 *
 * @param dimension The input dimension $d$.
 * @param classes The number of classes $K$.
 * @param options The hidden widths and the activation.
 * @returns The classifier.
 *
 * @example A classifier of 2-d points into 3 classes
 * const net = classifier(2, 3, { hidden: [8] })
 * const params = net.layer.init(stream(1))
 * print('classes:', net.classes, ' layers:', params.length)
 */
export function classifier(dimension: number, classes: number, options: ClassifierOptions = {}): Classifier {
  const { hidden = [64, 64], activation = 'silu' } = options
  return { layer: Mlp([dimension, ...hidden, classes], { activation }), classes, dimension }
}

/**
 * The logits $f(\xvec)$ of each row.
 *
 * @param net The classifier.
 * @param params Its parameters.
 * @param x The points, $n \times d$ (traced inside a gradient).
 * @returns The logits, $n \times K$.
 *
 * @example The logits of two points under an untrained classifier
 * const net = classifier(2, 3, { hidden: [8] })
 * print(classifierLogits(net, net.layer.init(stream(1)), tensor([[0, 0], [1, -1]])))
 */
export function classifierLogits(net: Classifier, params: Params[], x: Value): Value {
  return net.layer.apply(params, x)
}

/**
 * The energy $E(\xvec) = -\operatorname{logsumexp}_y f(\xvec)_y$ of each row, so that
 * $p(\xvec) \propto \exp(-E(\xvec))$. Differentiable in the parameters and in `x`.
 *
 * @param net The classifier.
 * @param params Its parameters.
 * @param x The points, $n \times d$ (traced inside a gradient).
 * @returns The $n$ energies.
 *
 * @example The energy is minus the log-sum-exp of the logits
 * const net = classifier(2, 3, { hidden: [8] })
 * const params = net.layer.init(stream(1))
 * const x = tensor([[0, 0], [1, -1]])
 * print('E(x) =', classifierEnergy(net, params, x))
 * print('-logsumexp f(x) =', neg(logsumexp(classifierLogits(net, params, x), -1)))
 */
export function classifierEnergy(net: Classifier, params: Params[], x: Value): Value {
  return neg(logsumexp(classifierLogits(net, params, x), -1))
}

/**
 * The class energy $-f(\xvec)_y$ of each row, so that $p(\xvec \mid y) \propto \exp(f(\xvec)_y)$.
 *
 * @param net The classifier.
 * @param params Its parameters.
 * @param x The points, $n \times d$ (traced inside a gradient).
 * @param y The class, from 0 to $K - 1$.
 * @returns The $n$ class energies.
 *
 * @example Minus the logit of class 1
 * const net = classifier(2, 3, { hidden: [8] })
 * const params = net.layer.init(stream(1))
 * const x = tensor([[0, 0], [1, -1]])
 * print('logits =', classifierLogits(net, params, x))
 * print('class-1 energy =', classEnergy(net, params, x, 1))
 */
export function classEnergy(net: Classifier, params: Params[], x: Value, y: number): Value {
  return neg(slice(classifierLogits(net, params, x), null, y))
}

/**
 * The score $\nabla_{\xvec} \log p$ at every row: of $p(\xvec) \propto \exp(-E(\xvec))$, or of
 * $p(\xvec \mid y) \propto \exp(f(\xvec)_y)$ when `y` is given. For Langevin samplers, which step
 * $\xvec \leftarrow \xvec + \epsilon \nabla_{\xvec} \log p + \sqrt{2\epsilon}\, \zvec$ with
 * $\zvec \sim \Gauss(\zeros, \Imat)$.
 *
 * @param net The classifier.
 * @param params Its parameters (constants: the score is a gradient in the points only).
 * @param y The class of $p(\xvec \mid y)$, from 0 to $K - 1$; left out, the score is that of $p(\xvec)$.
 * @returns The score function: points $n \times d$ to their scores, $n \times d$.
 *
 * @example An energy model's Langevin sample: twenty steps from uniform starts move down the energy
 * const net = classifier(2, 2, { hidden: [8] })
 * const params = net.layer.init(stream(1))
 * const score = classifierScore(net, params)
 * const step = 0.05
 * let x = uniformBox(2, 2)(stream(2), 4)
 * print('energies at the start =', classifierEnergy(net, params, x))
 * for (let t = 0; t < 20; t++) {
 *   const noise = normal(stream(t), 0, Math.sqrt(2 * step), { shape: [4, 2] })
 *   x = add(add(x, mul(step, score(x))), noise)
 * }
 * print('samples =', x)
 * print('energies at the end =', classifierEnergy(net, params, x))
 */
export function classifierScore(net: Classifier, params: Params[], y?: number): (x: Tensor) => Tensor {
  const energy = (x: Value) => (y === undefined ? classifierEnergy(net, params, x) : classEnergy(net, params, x, y))
  const g = grad((x: Value) => sum(energy(x)))
  return (x) =>
    fromData(
      Float64Array.from(toFlat(unwrap(g(x) as Value) as Tensor), (v) => -v),
      x.shape,
    )
}

/**
 * A sampler of points uniform on the box $[-b, b]^d$: JEM's restarts for its replay buffer.
 *
 * @param bound The box's half-width $b$.
 * @param d The dimension $d$.
 * @returns The sampler: a stream and a count $n$ to $n$ points, $n \times d$.
 *
 * @example Three points in the square $[-1, 1]^2$
 * print(uniformBox(1, 2)(stream(1), 3))
 */
export function uniformBox(bound: number, d: number): (s: Stream, n: number) => Tensor {
  return (s, n) => uniform(s, -bound, bound, { shape: [n, d] }) as Tensor
}

/** Options of `jemTraining`. */
export type JemTrainingOptions = {
  /** The classifier to train. */
  net: Classifier
  /** The training points, $N \times d$. */
  x: Tensor
  /** Their integer labels, $N$ of them, from 0 to $K - 1$. */
  y: Tensor
  /**
   * `jem` adds $\log p(\xvec)$ to the objective; `cross-entropy` trains $\log p(y \mid \xvec)$ alone. Default
   * `jem`.
   */
  objective?: 'jem' | 'cross-entropy'
  /** Points per minibatch. Default 64. */
  batchSize?: number
  /** The update rule. Default `contrastiveDivergence`'s, Adam with step size $10^{-3}$. */
  optimizer?: UpdateRule<unknown>
  /**
   * The negatives' sampler, over the defaults of 20 Langevin steps of size 0.02, restart probability 0.05, the box's
   * `bound`, and `fresh` points uniform on the box.
   */
  sampler?: Partial<PersistentLangevinOptions>
  /** Persistent chains kept in the replay buffer. Default 1000. */
  bufferSize?: number
  /** The energy-magnitude penalty of the CD term. Default 0. */
  regularisation?: number
  /**
   * Half-width of the data's box (restarts and clipping). Default from the data: $1.2$ times the largest absolute
   * coordinate.
   */
  bound?: number
}

/**
 * JEM training, or plain cross-entropy training of the same network with the same minibatches: `contrastiveDivergence`
 * with the energy $-\operatorname{logsumexp} f(\xvec)$, the supervised term softmax cross-entropy, and generative
 * weight 1 (JEM) or 0. `init` takes `{ params: net.layer.init(s) }`.
 *
 * @param options The network, the data, the objective, and the minibatch, optimiser, sampler and buffer settings.
 * @returns The algorithm, to run with `run`, `trace` or `live`; its state is `contrastiveDivergence`'s.
 *
 * @example A few JEM updates on two clusters
 * const s = stream(1)
 * const x = concat([normal(s, -1, 0.3, { shape: [16, 2] }), normal(stream(2), 1, 0.3, { shape: [16, 2] })], 0)
 * const y = tensor([...Array(16).fill(0), ...Array(16).fill(1)])
 * const net = classifier(2, 2, { hidden: [8] })
 * const sampler = { steps: 5 }
 * const alg = jemTraining({ net, x, y, batchSize: 8, bufferSize: 32, sampler })
 * const state = run(alg, { params: net.layer.init(stream(3)) }, 20, { stream: stream(4) })
 * print('cross-entropy =', state.supervisedLoss, ' CD term =', state.generativeLoss)
 * print('energy of data =', state.dataEnergy, ' of negatives =', state.sampleEnergy)
 */
export function jemTraining(
  options: JemTrainingOptions,
): Algorithm<{ params: Params[] }, ContrastiveDivergenceState<Params[]>> {
  const { net, x, y, objective = 'jem', batchSize = 64, bufferSize = 1000, regularisation = 0 } = options
  let extent = 0
  for (const v of toFlat(x)) extent = Math.max(extent, Math.abs(v))
  const bound = options.bound ?? 1.2 * extent
  const sampler: PersistentLangevinOptions = {
    steps: 20,
    stepSize: 0.02,
    reinitialise: 0.05,
    bound,
    fresh: uniformBox(bound, net.dimension),
    ...options.sampler,
  }
  return contrastiveDivergence<Params[], { x: Tensor; y: Tensor }>({
    energy: (params, points) => classifierEnergy(net, params, points),
    data: { x, y },
    batchSize,
    optimizer: options.optimizer,
    sampler,
    bufferSize,
    regularisation,
    generativeWeight: objective === 'jem' ? 1 : 0,
    supervised: (params, batch) => softmaxCrossEntropy(classifierLogits(net, params, batch.x), batch.y),
  })
}

/**
 * A shift $c(\xvec)$ added to every logit, of size `amount` $a$: `none` 0, `radial` $a \lVert \xvec \rVert^2$,
 * `tilt` $a x_1$, `bump` $a \exp(-\lVert \xvec \rVert^2 / 2)$.
 */
export type LogitShift = { kind: 'none' | 'radial' | 'tilt' | 'bump'; amount: number }

/**
 * $c(\xvec)$ at each row. Adding $c(\xvec)$ to every logit leaves
 * $p(y \mid \xvec) = \operatorname{softmax}(f(\xvec))_y$ unchanged, since the factor $\exp(c(\xvec))$ cancels
 * between numerator and denominator, but moves the energy to $E(\xvec) - c(\xvec)$ and so reshapes
 * $p(\xvec) \propto \exp(c(\xvec) - E(\xvec))$: the degree of freedom a classifier leaves free and JEM spends on the
 * density.
 *
 * @param shift The kind of shift and its size $a$.
 * @param x The points, $n \times d$.
 * @returns The $n$ values $c(\xvec)$.
 *
 * @example Each shift at three points
 * const x = tensor([[0, 0], [1, 0], [0, 2]])
 * for (const kind of ['none', 'radial', 'tilt', 'bump']) print(kind, logitShift({ kind, amount: 1 }, x))
 */
export function logitShift(shift: LogitShift, x: Tensor): Float64Array {
  const [n, d] = x.shape
  const v = toFlat(x)
  return Float64Array.from({ length: n }, (_, i) => {
    let r2 = 0
    for (let j = 0; j < d; j++) r2 += v[i * d + j] ** 2
    switch (shift.kind) {
      case 'none':
        return 0
      case 'radial':
        return shift.amount * r2
      case 'tilt':
        return shift.amount * v[i * d]
      case 'bump':
        return shift.amount * Math.exp(-r2 / 2)
    }
  })
}
