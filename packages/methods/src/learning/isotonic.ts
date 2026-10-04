/**
 * Isotonic regression by the pool-adjacent-violators algorithm (PAVA).
 *
 * Fits a non-decreasing (or non-increasing) univariate function that minimises the weighted
 * squared error sum w_i (y_i - f(x_i))^2 subject to monotonicity constraints.
 * Supports out-of-sample prediction via piecewise linear or step-function interpolation.
 */

import type { Estimator, FitOptions, Supervised } from 'aifn-compute/learning/estimators'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { dense, fromData } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { isotonicRegression as coreIsotonic } from 'aifn-compute/learning/calibration'
import { matrixShape } from 'aifn-compute/learning/estimators'
import { targets } from './util'

export type IsotonicParams = {
  /** If true (default), fits a non-decreasing function. If false, fits a non-increasing function. */
  increasing?: boolean
  /** How to handle out-of-bounds queries: 'clip' to range extrema (default) or 'nan'. */
  outOfBounds?: 'clip' | 'nan'
  /** Interpolation between points: 'linear' (default, connects steps) or 'step' (piecewise constant). */
  interpolation?: 'linear' | 'step'
}

export type WeightedData = Supervised<Tensor, Tensor> & { weights?: Tensor }

export interface IsotonicRegressor {
  thresholds: Tensor
  values: Tensor
  increasing: boolean
  outOfBounds: 'clip' | 'nan'
  interpolation: 'linear' | 'step'
  forward(x: Tensor): Tensor
  predict(x: Tensor): Tensor
  decide(x: Tensor): Tensor
}

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
 * Fit an isotonic regression model using the Pool Adjacent Violators Algorithm (PAVA).
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
