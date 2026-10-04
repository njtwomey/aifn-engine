/**
 * Linear regression by least squares (ridge when `l2 > 0`) with a Gaussian predictive: a reference estimator that
 * exercises the capability design end to end.
 */

import { cholesky, choleskySolve, lstsq } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Decides, Expects, Fitted, Predicts, Samples } from 'aifn-compute/learning/estimators'
import type { Estimator, Supervised } from 'aifn-compute/learning/estimators'
import { gaussianPredictive, type AnyUnivariate } from 'aifn-compute/learning/estimators'
import { withExpectation, withSampling } from 'aifn-compute/learning/estimators'
import { matrixShape, targetValues } from 'aifn-compute/learning/estimators'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const values = dense.data

/** Hyperparameters of `linearRegression`. */
export interface LinearRegressionParams {
  /** Ridge penalty α on ‖w‖² (the intercept is not penalised), as scikit-learn's `Ridge(alpha)`. Default 0. */
  l2?: number
  /** Fit an intercept (default true). */
  intercept?: boolean
}

/** A fitted linear regression y = x·w + b + ε, ε ~ N(0, σ²). */
export interface LinearRegressionModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Expects<Tensor>,
    Samples<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'linear-regression'
  /** Coefficients w, shape [d]. */
  readonly weights: Tensor
  /** Intercept b (0 without an intercept). */
  readonly intercept: number
  /** Plug-in noise standard deviation σ̂ = √(RSS / (n − p)), p the rank plus the intercept; NaN when n ≤ p. */
  readonly noiseSd: number
  /** Residual sum of squares on the training data. */
  readonly rss: number
  /** Residual degrees of freedom n − p. */
  readonly residualDof: number
  /** Numerical rank of the centred design (d when `l2 > 0`). */
  readonly rank: number
  /** Singular values of the centred design (least squares only; null for ridge). */
  readonly singularValues: Tensor | null
  readonly l2: number
}

/**
 * Linear regression by least squares on the centred data, as scikit-learn's `LinearRegression` and `Ridge`: with
 * x̄ and ȳ the training means, w minimises ‖(X − x̄)w − (y − ȳ)‖² + α‖w‖² and b = ȳ − x̄·w. With α = 0 the
 * minimum-norm solution is found by SVD (rank deficiency is reported in `rank`); with α > 0 by Cholesky of
 * XcᵀXc + αI. The predictive is the plug-in Gaussian N(x·w + b, σ̂²), which ignores the uncertainty in w.
 *
 * Capabilities: `forward` and `decide` (the mean x·w + b), `predictive` (Gaussian), `expect`, `sample`.
 */
export function linearRegression(
  params: LinearRegressionParams = {},
): Estimator<Supervised<Tensor, Tensor>, LinearRegressionModel> {
  const { l2 = 0, intercept = true } = params
  if (!(l2 >= 0)) throw new DomainError('linearRegression', 'linearRegression: l2 must be non-negative')
  return {
    name: 'linear-regression',
    params: { l2, intercept },
    fit({ x, y }) {
      const [n, d] = matrixShape(x, 'linearRegression')
      const target = targetValues(y, 'linearRegression')
      if (target.length !== n)
        throw new ShapeError('linearRegression', `linearRegression: ${n} rows of x but ${target.length} targets`)
      const X = values(x)
      const xMean = new Float64Array(d)
      let yMean = 0
      if (intercept) {
        for (let i = 0; i < n; i++) {
          yMean += target[i] / n
          for (let j = 0; j < d; j++) xMean[j] += X[i * d + j] / n
        }
      }
      const Xc = Float64Array.from(X, (v, k) => v - xMean[k % d])
      const yc = Float64Array.from(target, (v) => v - yMean)
      let w: Float64Array
      let rank = d
      let singularValues: Tensor | null = null
      if (l2 === 0) {
        const ls = lstsq(fromData(Xc, [n, d]), fromData(yc, [n]))
        w = values(ls.x).slice()
        rank = ls.rank
        singularValues = ls.singularValues
      } else {
        // Normal equations (XcᵀXc + αI) w = Xcᵀyc: positive definite for α > 0.
        const A = new Float64Array(d * d)
        const b = new Float64Array(d)
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < d; j++) {
            b[j] += Xc[i * d + j] * yc[i]
            for (let k = 0; k <= j; k++) A[j * d + k] += Xc[i * d + j] * Xc[i * d + k]
          }
        }
        for (let j = 0; j < d; j++) A[j * d + j] += l2
        const { L } = cholesky(fromData(A, [d, d]), { jitter: false })
        w = values(choleskySolve(L, fromData(b, [d]))).slice()
      }
      let b0 = yMean
      for (let j = 0; j < d; j++) b0 -= xMean[j] * w[j]
      let rss = 0
      for (let i = 0; i < n; i++) {
        let f = b0
        for (let j = 0; j < d; j++) f += X[i * d + j] * w[j]
        rss += (target[i] - f) ** 2
      }
      const residualDof = n - rank - (intercept ? 1 : 0)
      const noiseSd = residualDof > 0 ? Math.sqrt(rss / residualDof) : NaN
      const weights = fromData(w, [d])
      const forward = (input: Tensor): Tensor => {
        const [m, cols] = matrixShape(input, 'linearRegression.forward')
        if (cols !== d)
          throw new ShapeError('linearRegression', `linearRegression: fitted on ${d} features, given ${cols}`)
        const Z = values(input)
        const out = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let f = b0
          for (let j = 0; j < d; j++) f += Z[i * d + j] * w[j]
          out[i] = f
        }
        return fromData(out, [m])
      }
      const base = {
        kind: 'model' as const,
        name: 'linear-regression' as const,
        weights,
        intercept: b0,
        noiseSd,
        rss,
        residualDof,
        rank,
        singularValues,
        l2,
        forward,
        decide: forward,
        predictive: (input: Tensor) => {
          const mean = forward(input)
          return gaussianPredictive(mean, fromData(new Float64Array(mean.shape[0]).fill(noiseSd), mean.shape))
        },
      }
      return withSampling(withExpectation(base))
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'linearRegression',
    module: 'learning/linear',
    name: 'Linear regression',
    summary: 'Least squares (ridge with an L2 penalty), with a Gaussian predictive.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({ l2: real(0, 100, { default: 0, label: 'L2 penalty' }), intercept: bool({ default: true }) }),
    notes: ['linear-regression', 'ridge-regression'],
    cite: ['hastie2009'],
  },
  linearRegression,
)
