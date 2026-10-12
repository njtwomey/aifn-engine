/**
 * Linear regression by least squares (ridge when `l2 > 0`) with a Gaussian predictive: a reference estimator that
 * exercises the capability design end to end, and the ridge path through one SVD.
 *
 * The data are centred before the solve, so the intercept is never penalised, as in scikit-learn's `LinearRegression`
 * and `Ridge`; least squares goes through the SVD (`lstsq`), so a rank-deficient design gives the minimum-norm
 * solution, and ridge through the Cholesky factor of the regularised normal equations. With $d_i$ the singular values
 * of the centred design, the effective degrees of freedom of ridge are $\sum_i d_i^2 / (d_i^2 + \lambda)$ (Hastie,
 * Tibshirani and Friedman, 2009, §3.4.1), from the rank at $\lambda = 0$ down to 0 as $\lambda$ grows.
 */

import { cholesky, choleskySolve, lstsq, svd } from 'aifn-compute/numerics/linalg'
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
  /**
   * Ridge penalty $\alpha$ on $\lVert \wvec \rVert^2$ (the intercept is not penalised), as scikit-learn's
   * `Ridge(alpha)`. Default 0, plain least squares.
   */
  l2?: number
  /** Fit an intercept (default true). */
  intercept?: boolean
}

/** A fitted linear regression $y = \xvec^\top\wvec + b + \varepsilon$, $\varepsilon \sim \Gauss(0, \sigma^2)$. */
export interface LinearRegressionModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Expects<Tensor>,
    Samples<Tensor, Tensor> {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'linear-regression'
  /** Coefficients $\wvec$, $d$ values. */
  readonly weights: Tensor
  /** Intercept $b$ (0 without an intercept). */
  readonly intercept: number
  /**
   * Plug-in noise standard deviation $\hat\sigma = \sqrt{\text{RSS} / (n - p)}$, $p$ the rank plus the intercept;
   * NaN when $n \le p$.
   */
  readonly noiseSd: number
  /** Residual sum of squares on the training data. */
  readonly rss: number
  /** Residual degrees of freedom $n - p$. */
  readonly residualDof: number
  /** Numerical rank of the centred design ($d$ when `l2 > 0`). */
  readonly rank: number
  /** Singular values of the centred design, $\min(n, d)$ values, largest first. */
  readonly singularValues: Tensor
  /**
   * The effective degrees of freedom $\sum_i d_i^2 / (d_i^2 + \lambda)$, $d_i$ the singular values and $\lambda$
   * the penalty `l2`: the rank for least squares, falling towards 0 as the penalty grows.
   */
  readonly dof: number
  /** The ridge penalty it was fitted with. */
  readonly l2: number
}

/**
 * Linear regression by least squares on the centred data, as scikit-learn's `LinearRegression` and `Ridge`: with
 * $\bar{\xvec}$ and $\bar{y}$ the training means, $\wvec$ minimises
 * $\lVert (\Xmat - \ones\bar{\xvec}^\top)\wvec - (\yvec - \bar{y}\ones) \rVert^2 + \alpha \lVert \wvec \rVert^2$ and
 * $b = \bar{y} - \bar{\xvec}^\top\wvec$. With $\alpha = 0$ the minimum-norm solution is found by SVD (rank
 * deficiency is reported in `rank`); with $\alpha > 0$ by Cholesky of $\Xmat_c^\top\Xmat_c + \alpha\Imat$, with
 * $\Xmat_c$ the centred design. The predictive is the plug-in Gaussian $\Gauss(\xvec^\top\wvec + b, \hat\sigma^2)$,
 * which ignores the uncertainty in $\wvec$. Throws `DomainError` at once for a negative or NaN `l2`, and `ShapeError`
 * from `fit` when `x` and `y` differ in rows.
 *
 * Capabilities: `forward` and `decide` (the mean $\xvec^\top\wvec + b$, $m$ values), `predictive` (Gaussian),
 * `expect`, `sample`.
 *
 * @param params The ridge penalty `l2` and whether to fit an intercept, as `LinearRegressionParams`.
 * @returns The estimator: `fit({ x, y })`, with `x` $n \times d$ and `y` $n$ values, returns a `LinearRegressionModel`.
 *
 * @example Ordinary least squares recovers a known slope and intercept
 * // 50 points on the line y = 2x + 1, with noise of standard deviation 0.1.
 * const x = normals(stream(0), [50, 1])
 * const y = add(add(mul(reshape(x, [50]), 2), 1), mul(normals(stream(1), [50]), 0.1))
 * const model = linearRegression().fit({ x, y })
 * print('slope =', model.weights, ' intercept =', model.intercept)
 * print('noise sd =', model.noiseSd)
 * print('predictive sd at x = 0 and 1:', model.predictive(tensor([[0], [1]])).stddev())
 *
 * @example Ridge shrinks the slope towards zero as the penalty grows
 * // 50 points on the line y = 2x + 1, with noise of standard deviation 0.1.
 * const x = normals(stream(0), [50, 1])
 * const y = add(add(mul(reshape(x, [50]), 2), 1), mul(normals(stream(1), [50]), 0.1))
 * for (const l2 of [0, 10, 100]) print('l2 =', l2, ' slope =', linearRegression({ l2 }).fit({ x, y }).weights)
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
      let singularValues: Tensor
      let dof: number
      if (l2 === 0) {
        const ls = lstsq(fromData(Xc, [n, d]), fromData(yc, [n]))
        w = values(ls.x).slice()
        rank = ls.rank
        singularValues = ls.singularValues
        dof = rank
      } else {
        singularValues = svd(fromData(Xc, [n, d])).S
        dof = effectiveDof(values(singularValues), l2)
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
        dof,
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

/**
 * The effective degrees of freedom of ridge, $\sum_i d_i^2 / (d_i^2 + \lambda)$.
 *
 * @param s The singular values $d_i$ of the centred design.
 * @param lambda The penalty $\lambda \ge 0$; at 0 the count of nonzero singular values.
 * @returns The degrees of freedom, between 0 and the number of singular values.
 *
 * @example Two directions, one strong and one weak: the weak one is penalised away first
 * for (const lambda of [0, 0.1, 1, 10, 100]) print('lambda =', lambda, ' dof =', effectiveDof([10, 0.5], lambda))
 */
export function effectiveDof(s: ArrayLike<number>, lambda: number): number {
  let dof = 0
  for (let i = 0; i < s.length; i++) if (s[i] > 0) dof += (s[i] * s[i]) / (s[i] * s[i] + lambda)
  return dof
}

/** The ridge path: one fit per penalty, from one SVD. */
export interface RidgePath {
  /** The penalties $\lambda$, in the order given. */
  readonly lambdas: number[]
  /** The weights, $d \times k$: column $t$ is the fit at `lambdas[t]`. */
  readonly coef: Tensor
  /** The intercept of each fit (0 without an intercept). */
  readonly intercept: number[]
  /** The effective degrees of freedom of each fit. */
  readonly dof: number[]
  /** The singular values of the centred design, largest first. */
  readonly singularValues: Tensor
}

/**
 * Ridge regression at many penalties through one SVD of the centred design,
 * $\Xmat_c = \Umat\operatorname{diag}(\dvec)\Vmat^\top$:
 * $\wvec(\lambda) = \Vmat \operatorname{diag}(d_i / (d_i^2 + \lambda)) \Umat^\top\yvec_c$, with the effective
 * degrees of freedom $\sum_i d_i^2 / (d_i^2 + \lambda)$. Each column equals `linearRegression({ l2: lambda })`
 * (scikit-learn's `Ridge(alpha)`), and $\lambda = 0$ gives the minimum-norm least-squares fit. Throws `DomainError`
 * for a negative or NaN penalty and `ShapeError` for mismatched shapes.
 *
 * @param x The design, $n \times d$.
 * @param y The target, $n$ values.
 * @param lambdas The penalties $\lambda \ge 0$, in any order (kept).
 * @param options Whether to fit an unpenalised intercept by centring (default true).
 * @param options.intercept Centre the data first, so the intercept is not penalised.
 * @returns The path: the penalties, the weights ($d \times k$), the intercepts, the degrees of freedom, and the
 *   singular values.
 *
 * @example Degrees of freedom fall from the rank towards zero as the penalty grows
 * const x = normals(stream(0), [40, 3])
 * const y = tensor(toArray(x).map((r) => r[0] - 2 * r[1] + 0.5 * r[2]))
 * const path = ridgePath(x, y, [0, 1, 10, 100, 1000])
 * print('dof:', path.dof)
 * print('first weight:', toArray(path.coef)[0])
 */
export function ridgePath(
  x: Tensor,
  y: Tensor,
  lambdas: readonly number[],
  { intercept = true }: { intercept?: boolean } = {},
): RidgePath {
  for (const l of lambdas)
    if (!(l >= 0)) throw new DomainError('ridgePath', 'ridgePath: every lambda must be non-negative')
  const [n, d] = matrixShape(x, 'ridgePath')
  const target = targetValues(y, 'ridgePath')
  if (target.length !== n) throw new ShapeError('ridgePath', `ridgePath: ${n} rows of x but ${target.length} targets`)
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
  const { U, S, V } = svd(fromData(Xc, [n, d]))
  const r = S.shape[0]
  const u = values(U)
  const s = values(S)
  const v = values(V)
  // Uᵀ y_c, the target's coordinates along the left singular vectors.
  const uy = new Float64Array(r)
  for (let i = 0; i < n; i++) for (let a = 0; a < r; a++) uy[a] += u[i * r + a] * (target[i] - yMean)
  // Singular values below this are treated as zero, as lstsq does, so λ = 0 gives the minimum-norm fit.
  const cutoff = Math.max(n, d) * Number.EPSILON * (s[0] ?? 0)
  const k = lambdas.length
  const coef = new Float64Array(d * k)
  const intercepts: number[] = []
  const dof: number[] = []
  lambdas.forEach((lambda, t) => {
    let df = 0
    for (let a = 0; a < r; a++) {
      if (s[a] <= cutoff) continue
      const shrink = s[a] / (s[a] * s[a] + lambda)
      df += (s[a] * s[a]) / (s[a] * s[a] + lambda)
      for (let j = 0; j < d; j++) coef[j * k + t] += v[j * r + a] * shrink * uy[a]
    }
    let b = yMean
    for (let j = 0; j < d; j++) b -= xMean[j] * coef[j * k + t]
    intercepts.push(b)
    dof.push(df)
  })
  return { lambdas: [...lambdas], coef: fromData(coef, [d, k]), intercept: intercepts, dof, singularValues: S }
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
