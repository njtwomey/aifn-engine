/**
 * Isotonic regression as an estimator: the pool-adjacent-violators algorithm (PAVA) of
 * `aifn-compute/learning/calibration`, with out-of-sample prediction.
 *
 * The fit is the non-decreasing (or non-increasing) function $f$ of one feature that minimises the weighted squared
 * error $\sum_i w_i (y_i - f(x_i))^2$ (Ayer et al., 1955; Best and Chakravarti, 1990). Points with equal $x$ are first
 * pooled to their weighted mean, as scikit-learn's `IsotonicRegression`, and the fitted values at the distinct $x$ are
 * joined by straight lines or held as steps between them.
 */

import type { Estimator, FitOptions, Supervised } from 'aifn-compute/learning/estimators'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { dense, fromData } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { isotonicRegression as coreIsotonic } from 'aifn-compute/learning/calibration'
import { matrixShape } from 'aifn-compute/learning/estimators'
import { targets } from './util'

/** Hyperparameters of `isotonicRegressor`. */
export type IsotonicParams = {
  /** If true (default), fits a non-decreasing function. If false, fits a non-increasing function. */
  increasing?: boolean
  /**
   * A query below the smallest or above the largest fitted $x$: `'clip'` (default) gives the fitted value at that end,
   * `'nan'` gives NaN.
   */
  outOfBounds?: 'clip' | 'nan'
  /**
   * Between two fitted $x$: `'linear'` (default) interpolates their fitted values, `'step'` holds the value of the
   * left one (piecewise constant, continuous from the right).
   */
  interpolation?: 'linear' | 'step'
}

/**
 * Training data of `isotonicRegressor`: `x` an $n \times 1$ matrix, `y` the $n$ targets, and `weights` optional
 * non-negative weights $w_i$, one per point (default all 1).
 */
export type WeightedData = Supervised<Tensor, Tensor> & { weights?: Tensor }

/** A fitted isotonic regression. */
export interface IsotonicRegressor {
  /** The distinct training $x$, in increasing order. */
  thresholds: Tensor
  /** The fitted value at each of `thresholds`, monotone in the fitted direction. */
  values: Tensor
  /** Whether the fit is non-decreasing (true) or non-increasing. */
  increasing: boolean
  /** What a query outside the range of `thresholds` gives, as in `IsotonicParams`. */
  outOfBounds: 'clip' | 'nan'
  /** How a query between two thresholds is answered, as in `IsotonicParams`. */
  interpolation: 'linear' | 'step'
  /** The fitted function at each element of `x`, with the shape of `x` (NaN for a NaN query). */
  forward(x: Tensor): Tensor
  /** The same as `forward`. */
  predict(x: Tensor): Tensor
  /** The same as `forward`. */
  decide(x: Tensor): Tensor
}

/**
 * The number of entries of a sorted array that are at most `x`: the index at which `x` would be inserted to the right
 * of any equal entries (Python's `bisect.bisect_right`).
 *
 * @param arr The entries, in non-decreasing order.
 * @param x The value to place.
 * @returns An index from 0 to `arr.length`.
 */
function bisectRight(arr: ArrayLike<number>, x: number): number {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (x < arr[mid]) hi = mid
    else lo = mid + 1
  }
  return lo
}

/**
 * Isotonic regression of one feature (as scikit-learn's `IsotonicRegression`): `fit` sorts the points by $x$, pools
 * points with equal $x$ to their weighted mean, and runs pool-adjacent-violators on the result; prediction
 * interpolates the fitted values at the distinct $x$. `fit` throws `DomainError` unless `x` has exactly one column,
 * `ShapeError` when `y` or `weights` does not have one entry per row, and `DomainError` (from the core PAVA) for a
 * negative or NaN weight. With a single distinct $x$ every query, in range or not, gives its value.
 *
 * @param params The direction, the out-of-range rule and the interpolation, as `IsotonicParams`.
 * @returns The estimator: `fit({ x, y, weights })` returns an `IsotonicRegressor`.
 *
 * @example A noisy increasing trend becomes a monotone staircase
 * const x = tensor([[1], [2], [3], [4], [5], [6]])
 * const model = isotonicRegressor().fit({ x, y: tensor([1, 3, 2, 4, 3, 5]) })
 * print('thresholds =', model.thresholds)
 * print('values =', model.values)
 * print('at 2.5, 0 and 9 =', model.forward(tensor([[2.5], [0], [9]])))
 *
 * @example Step interpolation, and NaN outside the training range
 * const x = tensor([[0], [1], [2]])
 * const model = isotonicRegressor({ interpolation: 'step', outOfBounds: 'nan' }).fit({ x, y: tensor([0, 2, 4]) })
 * print('at 0.5, 1.5 and 3 =', model.forward(tensor([[0.5], [1.5], [3]])))
 *
 * @example A decreasing fit
 * const x = tensor([[1], [2], [3], [4]])
 * const model = isotonicRegressor({ increasing: false }).fit({ x, y: tensor([5, 3, 4, 1]) })
 * print('values =', model.values)
 */
export function isotonicRegressor(params: IsotonicParams = {}): Estimator<WeightedData, IsotonicRegressor> {
  const { increasing = true, outOfBounds = 'clip', interpolation = 'linear' } = params

  return {
    name: 'isotonic-regression',
    params: { increasing, outOfBounds, interpolation },
    fit({ x, y, weights }: WeightedData, _options: FitOptions = {}) {
      const where = 'isotonicRegression'
      const [n, d] = matrixShape(x, where)
      if (d !== 1) {
        throw new DomainError(where, `${where}: isotonic regression requires univariate input (d = 1), got d = ${d}`)
      }

      const xData = dense.data(x)
      const yData = targets(y, n, where)
      const wData = weights ? dense.data(weights) : undefined

      if (wData && wData.length !== n) {
        throw new ShapeError(where, `${where}: ${wData.length} weights for ${n} points`)
      }

      // 1. Sort points by x.
      const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => xData[a] - xData[b] || a - b)

      // 2. Collapse duplicate x coordinates into weighted means.
      const gx: number[] = []
      const gy: number[] = []
      const gw: number[] = []
      for (const i of order) {
        const last = gx.length - 1
        const xi = xData[i]
        const yi = yData[i]
        const wi = wData ? wData[i] : 1
        if (last >= 0 && gx[last] === xi) {
          const totalW = gw[last] + wi
          gy[last] = totalW > 0 ? (gw[last] * gy[last] + wi * yi) / totalW : (gy[last] + yi) / 2
          gw[last] = totalW
        } else {
          gx.push(xi)
          gy.push(yi)
          gw.push(wi)
        }
      }

      // 3. Run pool adjacent violators on the unique points.
      const coreFit = coreIsotonic(gy, {
        x: gx,
        weights: gw,
        increasing,
      })

      const uniqueX = Float64Array.from(gx)
      const uniqueY = dense.data(coreFit.fit)

      const predictArray = (query: Float64Array): Float64Array => {
        const m = query.length
        const out = new Float64Array(m)
        const K = uniqueX.length

        if (K === 0) {
          out.fill(NaN)
          return out
        }
        if (K === 1) {
          out.fill(uniqueY[0])
          return out
        }

        const minX = uniqueX[0]
        const maxX = uniqueX[K - 1]

        for (let i = 0; i < m; i++) {
          const q = query[i]

          if (Number.isNaN(q)) {
            out[i] = NaN
            continue
          }

          if (q < minX) {
            out[i] = outOfBounds === 'nan' ? NaN : uniqueY[0]
            continue
          }
          if (q > maxX) {
            out[i] = outOfBounds === 'nan' ? NaN : uniqueY[K - 1]
            continue
          }

          // In range [minX, maxX]
          const idx = bisectRight(uniqueX, q)
          if (idx === 0) {
            out[i] = uniqueY[0]
          } else if (idx >= K) {
            out[i] = uniqueY[K - 1]
          } else {
            const x0 = uniqueX[idx - 1]
            const x1 = uniqueX[idx]
            const y0 = uniqueY[idx - 1]
            const y1 = uniqueY[idx]

            if (q === x0) {
              out[i] = y0
            } else if (q === x1) {
              out[i] = y1
            } else if (interpolation === 'step') {
              out[i] = y0
            } else {
              // Linear interpolation
              const alpha = (q - x0) / (x1 - x0)
              out[i] = y0 + alpha * (y1 - y0)
            }
          }
        }
        return out
      }

      const forward = (qx: Tensor): Tensor => {
        const qData = dense.data(qx)
        const pred = predictArray(qData)
        return fromData(pred, qx.shape)
      }

      return {
        thresholds: fromData(uniqueX, [uniqueX.length]),
        values: fromData(uniqueY, [uniqueY.length]),
        increasing,
        outOfBounds,
        interpolation,
        forward,
        predict: forward,
        decide: forward,
      }
    },
  }
}
