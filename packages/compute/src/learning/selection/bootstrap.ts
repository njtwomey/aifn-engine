/**
 * Bootstrap estimates of prediction error: the leave-one-out bootstrap and the .632 and .632+ estimators of Efron and
 * Tibshirani (1997), "Improvements on cross-validation: the .632+ bootstrap method", JASA 92; Hastie, Tibshirani and
 * Friedman (2009), §7.11.
 *
 * With $B$ bootstrap samples, $C_i$ the number in which row $i$ is out of bag and $\hat f^{*b}$ the fit to sample $b$,
 * the leave-one-out bootstrap error averages each row's out-of-bag losses:
 *
 * $$ \widehat{\mathrm{Err}}^{(1)} = \frac1n \sum_i \frac{1}{C_i} \sum_{b:\, i \notin b} L(y_i, \hat f^{*b}(\xvec_i)). $$
 *
 * It is pessimistic (each fit sees about $0.632 n$ distinct rows), and the training error $\overline{\mathrm{err}}$ is
 * optimistic; $\widehat{\mathrm{Err}}^{(.632)} = 0.368\, \overline{\mathrm{err}} + 0.632\,
 * \widehat{\mathrm{Err}}^{(1)}$ splits the difference. The .632+ estimator moves the weight towards
 * $\widehat{\mathrm{Err}}^{(1)}$ as the fit overfits, measured against the no-information error
 * $\hat\gamma = \tfrac{1}{n^2} \sum_i \sum_j L(y_i, \hat f(\xvec_j))$ (targets and predictions paired at random) by the
 * relative overfitting rate $\hat R = (\widehat{\mathrm{Err}}^{(1)} - \overline{\mathrm{err}}) / (\hat\gamma -
 * \overline{\mathrm{err}})$:
 *
 * $$ \widehat{\mathrm{Err}}^{(.632+)} = (1 - \hat w)\, \overline{\mathrm{err}} + \hat w\,
 * \widehat{\mathrm{Err}}^{(1)}, \qquad \hat w = \frac{0.632}{1 - 0.368 \hat R}. $$
 *
 * As in the paper (its equations 31 and 32), $\widehat{\mathrm{Err}}^{(1)}$ is first capped at $\hat\gamma$, giving
 * $\widehat{\mathrm{Err}}^{(1)'}$, and $\hat R' = (\widehat{\mathrm{Err}}^{(1)'} - \overline{\mathrm{err}}) / (\hat\gamma -
 * \overline{\mathrm{err}})$ is set to 0 unless $\widehat{\mathrm{Err}}^{(1)}$ and $\hat\gamma$ both exceed
 * $\overline{\mathrm{err}}$, so that $\hat R' \in [0, 1]$; then
 *
 * $$ \widehat{\mathrm{Err}}^{(.632+)} = \widehat{\mathrm{Err}}^{(.632)} + (\widehat{\mathrm{Err}}^{(1)'} -
 * \overline{\mathrm{err}}) \frac{0.368 \cdot 0.632\, \hat R'}{1 - 0.368 \hat R'}, $$
 *
 * which is the weighted form above whenever $\widehat{\mathrm{Err}}^{(1)} \le \hat\gamma$.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, choice, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { outputFor, rowCount, takeData, takeRows, type Decides, type Features } from 'aifn-compute/learning/estimators'
import type { CrossValidationData, Fittable } from 'aifn-compute/learning/validate'

/** A pointwise loss: the loss of each prediction against its target, both $m$ values, as $m$ values. */
export type PointwiseLoss = (y: Tensor, prediction: Tensor) => Tensor

/** Options of `bootstrapPredictionError`. */
export interface BootstrapPredictionErrorOptions {
  /** The number of bootstrap samples $B$ (default 200); ignored when `indices` are given. */
  B?: number
  /**
   * Randomness: sample $b$ is drawn from `child(stream, 'sample', b)` and fitted with `child(stream, 'fit', b)`; the
   * fit to all the rows gets `child(stream, 'full')`. Needed unless `indices` are given.
   */
  stream?: Stream
  /**
   * The loss: `'squared'` $(y - \hat y)^2$ (default), `'zeroOne'` $[y \ne \hat y]$, or a function giving each
   * prediction's loss.
   */
  loss?: 'squared' | 'zeroOne' | PointwiseLoss
  /** The bootstrap samples themselves, $B \times n$ row indices (one sample per row), instead of drawing them. */
  indices?: MatrixLike
}

/** Bootstrap estimates of prediction error, from `bootstrapPredictionError`. */
export interface BootstrapPredictionError {
  /** The training (apparent) error $\overline{\mathrm{err}}$ of the fit to all the rows. */
  apparent: number
  /** The leave-one-out bootstrap error $\widehat{\mathrm{Err}}^{(1)}$, over the rows that were ever out of bag. */
  looBootstrap: number
  /** $0.368\, \overline{\mathrm{err}} + 0.632\, \widehat{\mathrm{Err}}^{(1)}$. */
  err632: number
  /** The .632+ estimate. */
  err632plus: number
  /** The no-information error rate $\hat\gamma$: the mean loss over every pairing of a target with a prediction. */
  noInformation: number
  /** The relative overfitting rate $\hat R'$ in $[0, 1]$ that sets the .632+ weight. */
  relativeOverfitting: number
  /**
   * The .632+ weight $\hat w = 0.632 / (1 - 0.368 \hat R')$: the estimate is $(1 - \hat w)\, \overline{\mathrm{err}} +
   * \hat w\, \widehat{\mathrm{Err}}^{(1)}$ when $\widehat{\mathrm{Err}}^{(1)} \le \hat\gamma$.
   */
  weight: number
  /** Each row's mean out-of-bag loss, $n$ values; NaN for a row never out of bag. */
  perPoint: Tensor
  /** $C_i$: the number of bootstrap samples each row is out of, $n$ values (int32). */
  outOfBagCount: Tensor
  /** How many times each row is in each sample, $B \times n$ (int32); 0 marks out of bag. */
  inBag: Tensor
  /** The number of bootstrap samples $B$. */
  B: number
}

/**
 * The pointwise loss function a `loss` option names.
 *
 * @param loss `'squared'`, `'zeroOne'` or a function.
 * @returns A function of targets and predictions giving one loss per pair, as a working array.
 */
function lossOf(loss: 'squared' | 'zeroOne' | PointwiseLoss): (y: Tensor, p: Tensor) => Float64Array {
  if (loss === 'squared' || loss === 'zeroOne') {
    return (y, p) => {
      const a = dense.data(y)
      const b = dense.data(p)
      return Float64Array.from(a, (v, i) => (loss === 'squared' ? (v - b[i]) ** 2 : v === b[i] ? 0 : 1))
    }
  }
  return (y, p) => Float64Array.from(dense.data(loss(y, p)))
}

/**
 * Bootstrap estimates of a model's prediction error (Efron and Tibshirani, 1997): fit to $B$ bootstrap samples of the
 * rows, score each fit on the rows it did not see, and combine with the training error into the leave-one-out
 * bootstrap, .632 and .632+ estimates. The fitted model's `decide` gives the predictions. Rows never out of bag are
 * left out of $\widehat{\mathrm{Err}}^{(1)}$ (their `perPoint` is NaN); with $B$ of 50 or more that is rare. Throws
 * `DomainError` without a stream or indices, and `ShapeError` for indices of the wrong width.
 *
 * @param estimator What to fit: anything with `fit(data, options)` whose model has `decide`.
 * @param data The rows: inputs `x` and targets `y`. Not modified.
 * @param options The number of samples, the stream, the loss, or the samples themselves.
 * @returns The training, leave-one-out bootstrap, .632 and .632+ errors, the no-information rate, the relative
 *   overfitting rate and weight, and the per-row out-of-bag losses and counts.
 *
 * @example One-nearest-neighbour overfits completely: training error 0, and .632+ moves towards the honest error
 * // A 1-nearest-neighbour classifier on one feature.
 * const nearest = {
 *   name: '1-nn',
 *   fit: (d) => {
 *     const xs = toFlat(d.x)
 *     const ys = toFlat(d.y)
 *     const decide = (x) =>
 *       tensor(toFlat(x).map((v) => ys[xs.reduce((b, u, j) => (Math.abs(u - v) < Math.abs(xs[b] - v) ? j : b), 0)]))
 *     return { decide }
 *   },
 * }
 * const x = tensor([[0.1], [0.4], [0.5], [0.9], [1.2], [1.3], [1.8], [2.0], [2.2], [2.9]])
 * const y = tensor([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
 * const e = bootstrapPredictionError(nearest, { x, y }, { B: 50, stream: stream(0), loss: 'zeroOne' })
 * print('apparent:', e.apparent)
 * print('loo bootstrap:', e.looBootstrap)
 * print('no-information rate:', e.noInformation)
 * print('.632, .632+:', e.err632, e.err632plus)
 * print('out-of-bag counts:', e.outOfBagCount)
 */
export function bootstrapPredictionError<X extends Features, M extends Decides<X>>(
  estimator: Fittable<X, M>,
  data: CrossValidationData<X>,
  options: BootstrapPredictionErrorOptions = {},
): BootstrapPredictionError {
  const { stream, loss = 'squared' } = options
  const where = 'bootstrapPredictionError'
  const n = rowCount(data.x)
  const pointLoss = lossOf(loss)
  let samples: Int32Array[]
  if (options.indices !== undefined) {
    const { data: idx, m, n: width } = dense.toMatrixF64(options.indices, where)
    if (width !== n) throw new ShapeError(where, `${where}: ${n} rows but bootstrap samples of ${width}`)
    samples = Array.from({ length: m }, (_, b) => Int32Array.from(idx.subarray(b * n, b * n + n)))
  } else {
    if (!stream) throw new DomainError(where, `${where}: needs a stream (or indices)`)
    const B = options.B ?? 200
    if (!(Number.isInteger(B) && B >= 1)) throw new DomainError(where, `${where}: B must be a positive integer`)
    samples = Array.from({ length: B }, (_, b) => Int32Array.from(dense.data(choice(child(stream, 'sample', b), n, n))))
  }
  const B = samples.length
  const predict = (model: M, x: unknown) => outputFor(model, 'decide', x) as Tensor
  // The fit to all the rows: training error and the no-information rate.
  const full = estimator.fit(data, { stream: stream && child(stream, 'full') })
  const fitted = predict(full, data.x)
  const trainLoss = pointLoss(data.y, fitted)
  const apparent = trainLoss.reduce((a, b) => a + b, 0) / n
  // γ: every target against every prediction, n² pairs.
  const ys = dense.data(data.y)
  const ps = dense.data(fitted)
  const yPairs = new Float64Array(n * n)
  const pPairs = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      yPairs[i * n + j] = ys[i]
      pPairs[i * n + j] = ps[j]
    }
  const noInformation =
    pointLoss(fromData(yPairs, [n * n]), fromData(pPairs, [n * n])).reduce((a, b) => a + b, 0) / n ** 2
  // Out-of-bag losses, accumulated per row.
  const inBag = new Int32Array(B * n)
  const total = new Float64Array(n)
  const count = new Int32Array(n)
  samples.forEach((rows, b) => {
    for (const i of rows) {
      if (!(i >= 0 && i < n && Number.isInteger(i)))
        throw new DomainError(where, `${where}: row index ${i} out of range`)
      inBag[b * n + i]++
    }
    const out: number[] = []
    for (let i = 0; i < n; i++) if (inBag[b * n + i] === 0) out.push(i)
    if (out.length === 0) return
    const model = estimator.fit(takeData(data, rows), { stream: stream && child(stream, 'fit', b) })
    const p = predict(model, takeRows(data.x, out))
    const l = pointLoss(takeRows(data.y, out), p)
    out.forEach((i, r) => {
      total[i] += l[r]
      count[i]++
    })
  })
  const perPoint = Float64Array.from(total, (t, i) => (count[i] > 0 ? t / count[i] : NaN))
  let used = 0
  let sum = 0
  for (const v of perPoint)
    if (!Number.isNaN(v)) {
      used++
      sum += v
    }
  const looBootstrap = used > 0 ? sum / used : NaN
  const capped = Math.min(looBootstrap, noInformation)
  const R = looBootstrap > apparent && noInformation > apparent ? (capped - apparent) / (noInformation - apparent) : 0
  const err632 = 0.368 * apparent + 0.632 * looBootstrap
  const weight = 0.632 / (1 - 0.368 * R)
  return {
    apparent,
    looBootstrap,
    err632,
    err632plus: err632 + ((capped - apparent) * 0.368 * 0.632 * R) / (1 - 0.368 * R),
    noInformation,
    relativeOverfitting: R,
    weight,
    perPoint: fromData(perPoint, [n]),
    outOfBagCount: fromData(count, [n]),
    inBag: fromData(inBag, [B, n]),
    B,
  }
}
