/**
 * Fitted models and their capabilities (design S §2.7), after scikit-learn's mixins but checked by the compiler: a
 * figure can ask a model only for what it declares. A model is a plain object with its fitted state as public fields
 * and its capabilities as methods.
 */

import type { Trace } from './algorithm'
import type { Distribution } from './distribution'
import type { Kinded } from './kinds'
import type { Size, Tensor } from './numbers'
import type { Stream } from './random'
import type { Info } from './registry'
import type { Space } from './space'

/** The partial forward pass: logits, latent means, raw scores (the "head" a readout completes). */
export interface Fitted<X, H = Tensor> {
  /** The head for a batch of inputs: logits, latent means or raw scores, one per input. */
  forward(x: X): H
}

/** A decision per input: an argmax class, a cluster label, an action, or a point prediction. */
export interface Decides<X, Y = Tensor> {
  /** The decision for each input of the batch. */
  decide(x: X): Y
}

/** A predictive distribution over the batch of inputs. */
export interface Predicts<X, D extends Distribution = Distribution> {
  /** The predictive distribution of the target, batched over the inputs. */
  predictive(x: X): D
}

/** $\expect[f(y) \mid x]$ per input (the mean without `f`), shape of the predictive's batch. */
export interface Expects<X> {
  /** $\expect[f(y) \mid x]$ for each input, or $\expect[y \mid x]$ when `f` is left out. */
  expect(x: X, f?: (y: number) => number): Tensor
}

/** Per-class or per-item scores, e.g. logits `[N, K]` or a binary logit `[N]`. */
export interface Scores<X> {
  /** The scores of each input of the batch; higher means more likely or more relevant. */
  score(x: X): Tensor
}

/** A representation of the inputs: embeddings, projections, transformed features. */
export interface Transforms<X, Z = Tensor> {
  /** The representation of each input of the batch. */
  transform(x: X): Z
}

/** Draws from the predictive: `[n, ...batch]` with `n`, else `[...batch]`. */
export interface Samples<X, Y = Tensor> {
  /** Draws for the inputs `x` from the stream: `n` per input when given, else one. */
  sample(s: Stream, x: X, n?: Size): Y
}

/** A fitted model that kept the trace of its training loop. */
export interface Trained<S = unknown> {
  /** The trace of the training loop: its recorded series (a loss, a log-likelihood) and states. */
  training: Trace<S>
}

/** The names of the capabilities: one vocabulary for models, metrics (`info.capability`) and evaluation. */
export type Capability = 'forward' | 'decide' | 'predictive' | 'expect' | 'score' | 'transform' | 'sample'

/**
 * What a supervised or unsupervised task asks of a model; `preprocessing` is a fitted transformer (a scaler, an
 * encoder, a feature map) whose output feeds another model.
 */
export type Task =
  'classification' | 'regression' | 'clustering' | 'density' | 'embedding' | 'ranking' | 'forecasting' | 'preprocessing'

/**
 * Registry metadata of a model (an estimator factory): its task, the capabilities every model it fits has (checked by
 * a protocol test that fits it on tiny data), and its hyperparameter space (the scalar options of the factory, with
 * their defaults).
 */
export interface ModelInfo extends Info {
  /** The entry kind of a model. */
  readonly kind: 'model'
  /** What the model is for. */
  readonly task: Task
  /** The methods every model the factory fits has. */
  readonly capabilities: readonly Capability[]
  /** The factory's scalar options, with their defaults. */
  readonly hyper: Space
  /** True for models that cannot score new inputs (t-SNE, spectral clustering). */
  readonly transductive?: boolean
}

/**
 * A fitted model (target): the `kind` brand, a reference to its registry entry, and its capabilities as methods (any
 * of `Fitted`, `Decides`, `Predicts`, `Expects`, `Scores`, `Transforms`, `Samples`, `Trained`).
 */
export interface Model extends Kinded<'model'> {
  /** The registry metadata of the factory that fitted it. */
  readonly info?: ModelInfo
  /** Set for a model that cannot score new inputs (t-SNE, spectral clustering). */
  readonly transductive?: true
}
