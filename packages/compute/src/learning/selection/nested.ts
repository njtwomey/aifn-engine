/**
 * Confidence intervals for prediction error by nested cross-validation: Bates, Hastie and Tibshirani (2024),
 * "Cross-validation: what does it estimate and how well does it do it?", JASA 119, Algorithm 1, with the rescaling and
 * clamping of its §4.3.2 and the bias correction of its Appendix C.
 *
 * The naive interval $\widehat{\mathrm{Err}}^{(\mathrm{CV})} \pm z_{1-\alpha/2}\, \hat\sigma / \sqrt n$ treats the $n$
 * held-out losses as independent, which they are not (every model is trained on most of the same rows), and it covers
 * the prediction error $\mathrm{Err}_{XY}$ of the model fitted to these data far less often than it claims. Nested
 * cross-validation estimates the mean squared error of cross-validation instead, from the holdout identity (their
 * Lemma 3)
 *
 * $$ \mathbb{E}\big[(\widehat{\mathrm{Err}} - \mathrm{Err}_{XY})^2\big] = \mathbb{E}\big[(\widehat{\mathrm{Err}} -
 * \bar e^{(\mathrm{out})})^2\big] - \mathbb{E}\big[(\bar e^{(\mathrm{out})} - \mathrm{Err}_{XY})^2\big]. $$
 *
 * For each of $R$ random assignments of the rows to $K$ folds, and each outer fold $I_k$: $(K - 1)$-fold
 * cross-validation on the other folds gives the inner losses $e^{(\mathrm{in})}$; the model fitted on all but $I_k$
 * gives the outer losses $e^{(\mathrm{out})}$ on $I_k$. Then $a_{rk} = (\bar e^{(\mathrm{in})} -
 * \bar e^{(\mathrm{out})})^2$ estimates the first term and $b_{rk} = \widehat{\operatorname{var}}(e^{(\mathrm{out})}) /
 * \abs{I_k}$ the second, so that
 *
 * $$ \widehat{\mathrm{MSE}} = \bar a - \bar b, \qquad \widehat{\mathrm{Err}}^{(\mathrm{NCV})} = \text{the mean of
 * every } e^{(\mathrm{in})}. $$
 *
 * The inner fits of outer folds $j$ and $k$ coincide (both leave out $I_j \cup I_k$), so each repetition costs
 * $K(K - 1)/2$ inner fits and $K$ outer ones. $\widehat{\mathrm{MSE}}$ estimates the error of $(K - 1)$-fold
 * cross-validation on $n(K - 1)/K$ rows, so it is rescaled by $(K - 1)/K$, then clamped so that its square root lies
 * between $\widehat{\mathrm{se}} = \hat\sigma_{\mathrm{in}} / \sqrt n$ (as if the $n$ losses were independent) and
 * $\sqrt K\, \widehat{\mathrm{se}}$ (as if only $n/K$ were), $\hat\sigma_{\mathrm{in}}$ the standard deviation of the
 * inner losses. The point estimate is moved by the bias estimate (their equation 15)
 *
 * $$ \widehat{\mathrm{bias}} = \Big(1 + \frac{K - 2}{K}\Big) \big(\widehat{\mathrm{Err}}^{(\mathrm{NCV})} -
 * \widehat{\mathrm{Err}}^{(\mathrm{CV})}\big), $$
 *
 * with $\widehat{\mathrm{Err}}^{(\mathrm{CV})}$ the ordinary $K$-fold estimate from the outer losses of the same
 * repetitions, and the interval is
 * $\widehat{\mathrm{Err}}^{(\mathrm{NCV})} - \widehat{\mathrm{bias}} \pm z_{1-\alpha/2} \sqrt{\widehat{\mathrm{MSE}}}$
 * (their equation 10). The authors' R package `nestedcv` raises the bias factor's fraction to the power $1.5$ and
 * estimates $\widehat{\mathrm{Err}}^{(\mathrm{CV})}$ from separate random splits; this file follows the paper.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { outputFor, rowCount, takeData, takeRows, type Decides, type Features } from 'aifn-compute/learning/estimators'
import { kFold, type CrossValidationData, type Fittable } from 'aifn-compute/learning/validate'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { lossOf, type PointwiseLoss } from './bootstrap'

/** Options of `nestedCvInterval`. */
export interface NestedCvIntervalOptions {
  /** The number of folds $K \ge 3$ (default 10). */
  folds?: number
  /** The number of random fold assignments $R$ (default 50); ignored when `foldIds` are given. */
  repetitions?: number
  /**
   * The loss: `'squared'` $(y - \hat y)^2$ (default), `'zeroOne'` $[y \ne \hat y]$, or a function giving each
   * prediction's loss.
   */
  loss?: 'squared' | 'zeroOne' | PointwiseLoss
  /**
   * Randomness: repetition $r$'s folds are drawn from `child(stream, 'folds', r)`, and its fits from
   * `child(stream, 'repeat', r)` (outer fold $k$ from its child `('outer', k)`, the inner fit leaving out folds
   * $j < k$ from `('inner', j, k)`). Needed unless `foldIds` are given.
   */
  stream?: Stream
  /** The confidence level $1 - \alpha$ (default 0.95). */
  level?: number
  /** The fold assignments themselves, $R \times n$: the fold $0, \dots, K - 1$ of each row in each repetition. */
  foldIds?: MatrixLike
}

/** The result of `nestedCvInterval`. */
export interface NestedCvInterval {
  /** The point estimate of prediction error, $\widehat{\mathrm{Err}}^{(\mathrm{NCV})} - \widehat{\mathrm{bias}}$. */
  estimate: number
  /** The confidence interval for $\mathrm{Err}_{XY}$, the estimate $\pm z_{1-\alpha/2}$ `se`. */
  interval: [number, number]
  /** The standard error $\sqrt{\widehat{\mathrm{MSE}}}$, after rescaling and clamping. */
  se: number
  /** The estimate of the mean squared error of cross-validation, rescaled by $(K - 1)/K$ and clamped. */
  mse: number
  /** The plug-in estimate $\bar a - \bar b$ before rescaling and clamping (may be negative). */
  mseNested: number
  /** The bias estimate subtracted from $\widehat{\mathrm{Err}}^{(\mathrm{NCV})}$. */
  bias: number
  /** $\widehat{\mathrm{Err}}^{(\mathrm{NCV})}$: the mean of the inner losses, before the bias correction. */
  ncv: number
  /** $\widehat{\mathrm{Err}}^{(\mathrm{CV})}$: the mean of the outer losses, ordinary $K$-fold cross-validation. */
  cv: number
  /** The ratio of `se` to $\widehat{\mathrm{se}} = \hat\sigma_{\mathrm{in}} / \sqrt n$, in $[1, \sqrt K]$. */
  inflation: number
  /**
   * The naive interval $\widehat{\mathrm{Err}}^{(\mathrm{CV})} \pm z_{1-\alpha/2}\, \hat\sigma_{\mathrm{out}} /
   * \sqrt n$, $\hat\sigma_{\mathrm{out}}$ the standard deviation of the outer losses of every repetition.
   */
  naive: { se: number; interval: [number, number] }
  /** The terms $a_{rk}$, one per repetition and outer fold, repetition by repetition. */
  a: Tensor
  /** The terms $b_{rk}$, in the order of `a`. */
  b: Tensor
  /** The number of folds $K$. */
  folds: number
  /** The number of repetitions $R$. */
  repetitions: number
}

/**
 * The mean of some values.
 *
 * @param v The values.
 * @returns Their mean.
 */
const meanOf = (v: ArrayLike<number>) => {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i]
  return s / v.length
}

/**
 * The sample variance of some values, divided by $m - 1$.
 *
 * @param v The $m$ values.
 * @returns Their variance (NaN for one value).
 */
const varianceOf = (v: ArrayLike<number>) => {
  const mu = meanOf(v)
  let s = 0
  for (let i = 0; i < v.length; i++) s += (v[i] - mu) ** 2
  return s / (v.length - 1)
}

/**
 * A confidence interval for a model's prediction error by nested cross-validation (Bates, Hastie and Tibshirani,
 * 2024, Algorithm 1): repeated $K$-fold cross-validation inside each outer training set estimates how far
 * cross-validation strays from the error of the model fitted to these data, and widens the naive interval by that
 * much (between 1 and $\sqrt K$ times). Returns the bias-corrected point estimate and interval, the naive ones, and
 * every term of the computation. The fitted model's `decide` gives the predictions; the loss must be pointwise. Each
 * repetition fits $K(K + 1)/2$ models. For a 0-1 loss the paper forms its intervals after a variance-stabilising
 * transform (its Appendix G), which this function does not apply. Throws `DomainError` unless $K \ge 3$, without a
 * stream or fold assignments, and for a fold assignment with an empty fold; `ShapeError` for fold assignments of the
 * wrong width.
 *
 * @param estimator What to fit: anything with `fit(data, options)` whose model has `decide`.
 * @param data The rows: inputs `x` and targets `y`. Not modified.
 * @param options The folds, repetitions, loss, stream and level, or the fold assignments themselves.
 * @returns The estimate and interval, the standard error and MSE, the bias, the NCV and CV means, the inflation over
 *   the naive standard error, the naive interval, and the terms $a_{rk}$ and $b_{rk}$.
 *
 * @example Least squares on a line: the nested interval against the naive one
 * const line = {
 *   name: 'least squares',
 *   fit: (d) => {
 *     const xs = toFlat(d.x)
 *     const ys = toFlat(d.y)
 *     const mx = xs.reduce((a, b) => a + b, 0) / xs.length
 *     const my = ys.reduce((a, b) => a + b, 0) / ys.length
 *     const sxy = xs.reduce((s, v, i) => s + (v - mx) * (ys[i] - my), 0)
 *     const sxx = xs.reduce((s, v) => s + (v - mx) ** 2, 0)
 *     return { decide: (x) => tensor(toFlat(x).map((v) => my + (sxy / sxx) * (v - mx))) }
 *   },
 * }
 * const x = normals(stream(1), [40, 1])
 * const noise = toFlat(normals(stream(2), [40]))
 * const y = tensor(toFlat(x).map((v, i) => 2 * v + noise[i]))
 * const r = nestedCvInterval(line, { x, y }, { folds: 5, repetitions: 10, stream: stream(3) })
 * print('estimate:', r.estimate, ' bias:', r.bias)
 * print('nested interval:', r.interval, ' inflation:', r.inflation)
 * print('naive interval: ', r.naive.interval)
 */
export function nestedCvInterval<X extends Features, M extends Decides<X>>(
  estimator: Fittable<X, M>,
  data: CrossValidationData<X>,
  options: NestedCvIntervalOptions = {},
): NestedCvInterval {
  const where = 'nestedCvInterval'
  const { stream, loss = 'squared', level = 0.95 } = options
  const n = rowCount(data.x)
  const pointLoss = lossOf(loss)
  let assignments: Int32Array[]
  let K: number
  if (options.foldIds !== undefined) {
    const { data: ids, m, n: width } = dense.toMatrixF64(options.foldIds, where)
    if (width !== n) throw new ShapeError(where, `${where}: ${n} rows but fold assignments of ${width}`)
    assignments = Array.from({ length: m }, (_, r) => Int32Array.from(ids.subarray(r * n, r * n + n)))
    K = options.folds ?? Math.max(...assignments.map((f) => Math.max(...f))) + 1
  } else {
    if (!stream) throw new DomainError(where, `${where}: needs a stream (or foldIds)`)
    K = options.folds ?? 10
    const R = options.repetitions ?? 50
    if (!(Number.isInteger(R) && R >= 1))
      throw new DomainError(where, `${where}: repetitions must be a positive integer`)
    if (!(Number.isInteger(K) && K >= 3 && K <= n)) throw new DomainError(where, `${where}: folds must be 3 to n`)
    assignments = Array.from({ length: R }, (_, r) => {
      const ids = new Int32Array(n)
      kFold({ k: K, shuffle: true })
        .split({ n }, child(stream, 'folds', r))
        .forEach((s, f) => s.test.data.forEach((i) => (ids[i] = f)))
      return ids
    })
  }
  if (!(Number.isInteger(K) && K >= 3)) throw new DomainError(where, `${where}: folds must be at least 3`)
  if (!(level > 0 && level < 1)) throw new DomainError(where, `${where}: level must lie in (0, 1)`)
  const R = assignments.length
  const a: number[] = []
  const b: number[] = []
  const inner: number[] = []
  const outer: number[] = []
  assignments.forEach((fold, r) => {
    const members = Array.from({ length: K }, () => [] as number[])
    fold.forEach((f, i) => {
      if (!(f >= 0 && f < K && Number.isInteger(f))) throw new DomainError(where, `${where}: fold ${f} out of range`)
      members[f].push(i)
    })
    if (members.some((m) => m.length === 0)) throw new DomainError(where, `${where}: repetition ${r} has an empty fold`)
    const s = stream && child(stream, 'repeat', r)
    /** The losses on the rows of `test` of the model fitted on every fold but those in `out`. */
    const lossesOn = (out: readonly number[], test: readonly number[], key: (string | number)[]) => {
      const rows: number[] = []
      for (let i = 0; i < n; i++) if (!out.includes(fold[i])) rows.push(i)
      const model = estimator.fit(takeData(data, rows), { stream: s && child(s, ...key) })
      return test.map((t) => {
        const p = outputFor(model, 'decide', takeRows(data.x, members[t])) as Tensor
        return pointLoss(takeRows(data.y, members[t]), p)
      })
    }
    // e[j][k]: the losses on fold j of the model that leaves out folds j and k.
    const e: Float64Array[][] = Array.from({ length: K }, () => new Array<Float64Array>(K))
    for (let j = 0; j < K; j++)
      for (let k = j + 1; k < K; k++) [e[j][k], e[k][j]] = lossesOn([j, k], [j, k], ['inner', j, k])
    for (let k = 0; k < K; k++) {
      const [eOut] = lossesOn([k], [k], ['outer', k])
      const eIn: number[] = []
      for (let j = 0; j < K; j++) if (j !== k) eIn.push(...e[j][k])
      a.push((meanOf(eIn) - meanOf(eOut)) ** 2)
      b.push(varianceOf(eOut) / eOut.length)
      inner.push(...eIn)
      outer.push(...eOut)
    }
  })
  const z = normalQuantile(0.5 + level / 2)
  const ncv = meanOf(inner)
  const cv = meanOf(outer)
  const mseNested = meanOf(a) - meanOf(b)
  const seIn = Math.sqrt(varianceOf(inner) / n)
  const se = Math.min(Math.max(Math.sqrt(Math.max(0, ((K - 1) / K) * mseNested)), seIn), Math.sqrt(K) * seIn)
  const bias = (1 + (K - 2) / K) * (ncv - cv)
  const estimate = ncv - bias
  const naiveSe = Math.sqrt(varianceOf(outer) / n)
  return {
    estimate,
    interval: [estimate - z * se, estimate + z * se],
    se,
    mse: se * se,
    mseNested,
    bias,
    ncv,
    cv,
    inflation: se / seIn,
    naive: { se: naiveSe, interval: [cv - z * naiveSe, cv + z * naiveSe] },
    a: fromData(Float64Array.from(a), [a.length]),
    b: fromData(Float64Array.from(b), [b.length]),
    folds: K,
    repetitions: R,
  }
}
