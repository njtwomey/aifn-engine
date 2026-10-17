/**
 * Noise for a self-training student, as maps from a training set to the set actually fitted, for the `augment` option
 * of `selfTrainingSteps` (Noisy Student training, Xie et al., 2020, §2: input noise by data augmentation, model noise
 * by dropout and stochastic depth).
 *
 * `inputJitter` replaces each training row by `copies` copies with independent Gaussian noise added to every input
 * coordinate, the same label kept: the student must give the teacher's label to a neighbourhood of each point, a
 * stand-in for image augmentation on vectors. `featureDropout` is inverted dropout (Srivastava et al., 2014) on a fixed
 * feature layer, written as data: each row is replaced by `copies` copies in which every feature is zeroed with
 * probability $p$ and the rest are scaled by $1 / (1 - p)$, so a linear model fitted on them is trained under dropout
 * of its input features, as a network is under dropout of its last hidden layer; prediction uses the clean features.
 * Both replicate the training-time noise over a finite number of draws, which a minibatch optimiser would resample
 * every step instead.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import { add, concat, mul, shapeOf, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { dropout } from 'aifn-compute/nn/layers'
import type { TrainingSet } from './self-training'

/**
 * Gaussian input jitter: `copies` noisy copies of every row, copy $c$ drawing its noise from the child
 * `'jitter', c` of the stream it is given. Throws `DomainError` for a negative `sd` or a `copies` that is not a
 * positive integer.
 *
 * @param sd The standard deviation of the noise added to each input coordinate (0 copies the rows unchanged).
 * @param copies The number of copies per row (at least 1).
 * @returns A function from a training set and a stream to the jittered training set ($n$ times `copies` rows, copy by
 *   copy).
 *
 * @example Three jittered copies of two rows
 * import { dataset } from 'aifn-compute/learning/estimators'
 * const noisy = inputJitter(0.1, 3)(dataset(tensor([[1, 2], [3, 4]]), tensor([0, 1])), stream(1))
 * print('x:', noisy.x)
 * print('y:', noisy.y)
 */
export function inputJitter(sd: number, copies: number): (data: TrainingSet, s: Stream) => TrainingSet {
  if (!(sd >= 0)) throw new DomainError('inputJitter', 'inputJitter: sd must be non-negative')
  if (!(Number.isInteger(copies) && copies >= 1))
    throw new DomainError('inputJitter', 'inputJitter: copies must be a positive integer')
  return (data, s) => {
    const shape = shapeOf(data.x)
    const xs: Tensor[] = []
    const ys: Tensor[] = []
    for (let c = 0; c < copies; c++) {
      xs.push(sd > 0 ? add(data.x, normals(child(s, 'jitter', c), shape, 0, sd)) : data.x)
      ys.push(data.y)
    }
    return dataset(concat(xs, 0), concat(ys, 0))
  }
}

/** Options of `featureDropout`. */
export interface FeatureDropoutOptions {
  /**
   * How each copy's mask is drawn from its child stream `'mask', c`: `'bernoulli'` (default) by the engine's `dropout`
   * (`aifn-compute/nn/layers`), or `'uniform'` by zeroing the features whose uniform draw is below $p$, the draw of the
   * site's first version (the same law, different masks for one seed).
   */
  mask?: 'bernoulli' | 'uniform'
}

/**
 * Inverted dropout on the columns of the inputs: `copies` masked copies of every row. Throws `DomainError` for a rate
 * outside $[0, 1)$ or a `copies` that is not a positive integer.
 *
 * @param rate The probability $p \in [0, 1)$ that a feature is zeroed (0 returns the training set unchanged).
 * @param copies The number of masked copies per row (at least 1).
 * @param options How the masks are drawn; see `FeatureDropoutOptions`.
 * @returns A function from a training set (features as inputs) and a stream to the masked training set ($n$ times
 *   `copies` rows, copy by copy).
 *
 * @example At rate 0.5 every feature is zeroed or doubled
 * import { dataset } from 'aifn-compute/learning/estimators'
 * const masked = featureDropout(0.5, 2)(dataset(tensor([[1, 2], [3, 4]]), tensor([0, 1])), stream(1))
 * print('x:', masked.x)
 */
export function featureDropout(
  rate: number,
  copies: number,
  options: FeatureDropoutOptions = {},
): (data: TrainingSet, s: Stream) => TrainingSet {
  const { mask = 'bernoulli' } = options
  if (!(rate >= 0 && rate < 1)) throw new DomainError('featureDropout', 'featureDropout: rate must be in [0, 1)')
  if (!(Number.isInteger(copies) && copies >= 1))
    throw new DomainError('featureDropout', 'featureDropout: copies must be a positive integer')
  return (data, s) => {
    if (rate === 0) return data
    const shape = shapeOf(data.x)
    const keep = 1 / (1 - rate)
    const xs: Tensor[] = []
    const ys: Tensor[] = []
    for (let c = 0; c < copies; c++) {
      const sc = child(s, 'mask', c)
      if (mask === 'bernoulli') xs.push(dropout(sc, data.x, rate) as Tensor)
      else {
        const u = toFlat(uniform(sc, 0, 1, { shape }))
        xs.push(
          mul(
            data.x,
            tensor(
              u.map((v) => (v < rate ? 0 : keep)),
              shape,
            ),
          ),
        )
      }
      ys.push(data.y)
    }
    return dataset(concat(xs, 0), concat(ys, 0))
  }
}
