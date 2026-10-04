/**
 * A classifier read as an energy-based model: JEM (Grathwohl et al., 2019, "Your classifier is secretly an energy
 * based model and you should treat it like one").
 *
 * A softmax classifier with logits f(x) ∈ ℝ^K sets p(y | x) = exp(f(x)[y]) / Σ_y′ exp(f(x)[y′]). Adding any c(x) to
 * every logit leaves p(y | x) unchanged, so cross-entropy training fixes the logits only up to that shift. JEM spends
 * the free degree of freedom on the density of x: it reads the same logits as p(x, y) ∝ exp(f(x)[y]), whence
 * p(x) ∝ Σ_y exp(f(x)[y]) = exp(−E(x)) with the energy E(x) = −logsumexp_y f(x)[y], and p(y | x) is still the
 * softmax. Training maximises log p(y | x) + log p(x): cross-entropy plus the contrastive-divergence term, with negatives
 * from persistent short-run Langevin (`aifn-compute/nn/training`'s `contrastiveDivergence`). Class-conditional samples come
 * from p(x | y) ∝ exp(f(x)[y]).
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

/** A classifier network x [n, d] → logits [n, K]. */
export type Classifier = {
  readonly layer: Layer<Params[]>
  readonly classes: number
  readonly dimension: number
}

/** Options of `classifier`. */
export type ClassifierOptions = {
  /** Hidden widths. Default [64, 64]. */
  hidden?: readonly number[]
  /** Activation. Default SiLU (smooth, so the energy's gradient in x is too, as Langevin needs). */
  activation?: Activation
}

/** An MLP classifier of d-dimensional points into K classes. */
export function classifier(dimension: number, classes: number, options: ClassifierOptions = {}): Classifier {
  const { hidden = [64, 64], activation = 'silu' } = options
  return { layer: Mlp([dimension, ...hidden, classes], { activation }), classes, dimension }
}

/** The logits f(x), [n, K]. */
export function classifierLogits(net: Classifier, params: Params[], x: Value): Value {
  return net.layer.apply(params, x)
}

/** The energy E(x) = −logsumexp_y f(x)[y] of each row, [n]: p(x) ∝ exp(−E(x)). */
export function classifierEnergy(net: Classifier, params: Params[], x: Value): Value {
  return neg(logsumexp(classifierLogits(net, params, x), -1))
}

/** The class energy −f(x)[y] of each row, [n]: p(x | y) ∝ exp(f(x)[y]). */
export function classEnergy(net: Classifier, params: Params[], x: Value, y: number): Value {
  return neg(slice(classifierLogits(net, params, x), null, y))
}

/**
 * The score ∇ₓ log p at every row of x ([n, d]): of p(x) ∝ exp(−E(x)), or of p(x | y) ∝ exp(f(x)[y]) when `y` is
 * given. For Langevin samplers.
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

/** n points uniform on the box [−bound, bound]^d: JEM's restarts for its replay buffer. */
export function uniformBox(bound: number, d: number): (s: Stream, n: number) => Tensor {
  return (s, n) => uniform(s, -bound, bound, { shape: [n, d] }) as Tensor
}

/** Options of `jemTraining`. */
export type JemTrainingOptions = {
  net: Classifier
  /** Points [N, d] and integer labels [N]. */
  x: Tensor
  y: Tensor
  /** `jem` adds log p(x) to the objective; `cross-entropy` trains log p(y | x) alone. Default `jem`. */
  objective?: 'jem' | 'cross-entropy'
  batchSize?: number
  optimizer?: UpdateRule<unknown>
  /** The negatives' sampler (its `fresh` defaults to uniform on the data's box). */
  sampler?: Partial<PersistentLangevinOptions>
  bufferSize?: number
  /** The energy-magnitude penalty of the CD term. Default 0. */
  regularisation?: number
  /** Half-width of the data's box (restarts and clipping). Default from the data, 1.2 × the largest |coordinate|. */
  bound?: number
}

/**
 * JEM training, or plain cross-entropy training of the same network with the same minibatches: `contrastiveDivergence`
 * with the energy −logsumexp f(x), the supervised term softmax cross-entropy, and generative weight 1 (JEM) or 0.
 * `init` takes `{ params: net.layer.init(s) }`.
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

/** A shift c(x) added to every logit: `radial` a‖x‖², `tilt` a·x₁, `bump` a·exp(−‖x‖²/2). */
export type LogitShift = { kind: 'none' | 'radial' | 'tilt' | 'bump'; amount: number }

/**
 * c(x) at each row of x ([n, d]). Adding c(x) to every logit leaves p(y | x) = softmax(f(x))_y unchanged, since the
 * factor exp(c(x)) cancels between numerator and denominator, but moves the energy to E(x) − c(x) and so reshapes
 * p(x) ∝ exp(c(x) − E(x)): the degree of freedom a classifier leaves free and JEM spends on the density.
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
