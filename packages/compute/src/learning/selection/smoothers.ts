/**
 * Leave-one-out and generalised cross-validation for linear smoothers, $\hat\yvec = \Smat\yvec$, without refitting.
 *
 * For a smoother whose fit with row $i$ left out is the fit of $\Smat$ renormalised without that row (least squares,
 * ridge, smoothing splines, kernel smoothers), the leave-one-out residual is
 *
 * $$ y_i - \hat f^{(-i)}(\xvec_i) = \frac{y_i - \hat y_i}{1 - S_{ii}}, $$
 *
 * and generalised cross-validation (Craven and Wahba, 1979) replaces each leverage $S_{ii}$ by their mean
 * $\operatorname{tr}\Smat / n$:
 *
 * $$ \mathrm{GCV} = \frac{\tfrac1n \sum_i (y_i - \hat y_i)^2}{(1 - \operatorname{tr}\Smat / n)^2}. $$
 *
 * Hastie, Tibshirani and Friedman (2009), §7.10 and §7.12. Ridge regression is such a smoother: from one SVD of the
 * (centred) design, $\Smat_\lambda = \tfrac1n \mathbf{1}\mathbf{1}^\top + \Umat \operatorname{diag}(d_a^2 / (d_a^2 +
 * \lambda)) \Umat^\top$, so a whole grid of penalties is cross-validated for the price of one fit, as scikit-learn's
 * `RidgeCV` does. A leverage of exactly 1 gives an infinite (or NaN) leave-one-out residual: it is reported, not
 * clamped.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type MatrixLike, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import { svd } from 'aifn-compute/numerics/linalg'

/** Cross-validation of a linear smoother, from `linearSmootherCv`. */
export interface LinearSmootherCv {
  /** The fitted values $\hat\yvec = \Smat\yvec$, $n$ values. */
  fitted: Tensor
  /** The residuals $\yvec - \hat\yvec$, $n$ values. */
  residuals: Tensor
  /** The leverages $S_{ii}$, the diagonal of the smoother, $n$ values. */
  leverage: Tensor
  /** The leave-one-out residuals $(y_i - \hat y_i) / (1 - S_{ii})$, $n$ values. */
  looResiduals: Tensor
  /** The leave-one-out mean squared error, the mean of the squared `looResiduals`. */
  loo: number
  /** Generalised cross-validation, the mean squared residual over $(1 - \operatorname{tr}\Smat / n)^2$. */
  gcv: number
  /** The effective degrees of freedom $\operatorname{tr}\Smat$. */
  df: number
}

/**
 * Leave-one-out and generalised cross-validation of a linear smoother $\hat\yvec = \Smat\yvec$, in closed form from the
 * smoother matrix: the leave-one-out residuals $(y_i - \hat y_i)/(1 - S_{ii})$ (exact for least squares, ridge,
 * smoothing splines and kernel smoothers) and GCV with $\operatorname{tr}\Smat$ as the degrees of freedom (Craven and
 * Wahba, 1979). Throws `ShapeError` when $\Smat$ is not $n \times n$ for $n$ targets.
 *
 * @param S The smoother (hat) matrix, $n \times n$: row $i$ holds the weights that make $\hat y_i$ from $\yvec$.
 * @param y The targets, $n$ values.
 * @returns The fitted values, residuals, leverages and leave-one-out residuals, and the two estimates of prediction
 *   error with the degrees of freedom.
 *
 * @example A three-point moving average: its leave-one-out fit is the mean of the two neighbours
 * const S = tensor([
 *   [1 / 2, 1 / 2, 0, 0],
 *   [1 / 3, 1 / 3, 1 / 3, 0],
 *   [0, 1 / 3, 1 / 3, 1 / 3],
 *   [0, 0, 1 / 2, 1 / 2],
 * ])
 * const cv = linearSmootherCv(S, tensor([1, 3, 2, 4]))
 * print('leverage:', cv.leverage)
 * print('loo residuals:', cv.looResiduals)
 * print('loo, gcv, df:', cv.loo, cv.gcv, cv.df)
 */
export function linearSmootherCv(S: MatrixLike, y: VectorLike): LinearSmootherCv {
  const target = dense.toF64(y, 'linearSmootherCv')
  const n = target.length
  const { data: s } = dense.toMatrixF64(S, 'linearSmootherCv', n, n)
  const fitted = dense.matVec(s, target, n, n)
  const residuals = new Float64Array(n)
  const leverage = new Float64Array(n)
  const loo = new Float64Array(n)
  let rss = 0
  let df = 0
  let looSum = 0
  for (let i = 0; i < n; i++) {
    residuals[i] = target[i] - fitted[i]
    leverage[i] = s[i * n + i]
    loo[i] = residuals[i] / (1 - leverage[i])
    rss += residuals[i] ** 2
    df += leverage[i]
    looSum += loo[i] ** 2
  }
  return {
    fitted: dense.vec(fitted),
    residuals: dense.vec(residuals),
    leverage: dense.vec(leverage),
    looResiduals: dense.vec(loo),
    loo: looSum / n,
    gcv: rss / n / (1 - df / n) ** 2,
    df,
  }
}

/** The SVD of a ridge design, shared by `ridgeSmoother` and `ridgeCv`. */
interface RidgeBasis {
  /** Rows. */
  n: number
  /** The left singular vectors of the (centred) design, $n \times r$ row-major, nonzero singular values only. */
  u: Float64Array
  /** Their number $r$. */
  r: number
  /** The squared nonzero singular values $d_a^2$. */
  d2: Float64Array
  /** The singular values of the (centred) design, all of them, largest first. */
  singularValues: Tensor
}

/**
 * The SVD of the design, centred when there is an intercept, keeping the directions whose singular value is above
 * $\max(n, d)\,\varepsilon\, d_1$ (as `lstsq`), so that $\lambda = 0$ gives the least-squares projection.
 *
 * @param x The design, $n \times d$.
 * @param intercept Centre the columns first (an unpenalised intercept).
 * @param where The caller's name, for error messages.
 * @returns The basis.
 */
function ridgeBasis(x: MatrixLike, intercept: boolean, where: string): RidgeBasis {
  const { data, m: n, n: d } = dense.toMatrixF64(x, where)
  if (intercept) {
    for (let j = 0; j < d; j++) {
      let mean = 0
      for (let i = 0; i < n; i++) mean += data[i * d + j] / n
      for (let i = 0; i < n; i++) data[i * d + j] -= mean
    }
  }
  const { U, S } = svd(fromData(data, [n, d]))
  const k = S.shape[0]
  const u = dense.data(U)
  const s = dense.data(S)
  const cutoff = Math.max(n, d) * Number.EPSILON * (s[0] ?? 0)
  let r = 0
  while (r < k && s[r] > cutoff) r++
  const kept = new Float64Array(n * r)
  for (let i = 0; i < n; i++) for (let a = 0; a < r; a++) kept[i * r + a] = u[i * k + a]
  return { n, u: kept, r, d2: Float64Array.from(s.subarray(0, r), (v) => v * v), singularValues: S }
}

/**
 * Check that a ridge penalty is non-negative.
 *
 * @param lambda The penalty.
 * @param where The caller's name, for the error message.
 */
function checkPenalty(lambda: number, where: string): void {
  if (!(lambda >= 0)) throw new DomainError(where, `${where}: every lambda must be non-negative, got ${lambda}`)
}

/**
 * The smoother matrix of ridge regression, $\hat\yvec = \Smat_\lambda\yvec$ for the fit minimising
 * $\lVert \yvec - b\mathbf{1} - \Xmat\wvec \rVert^2 + \lambda \lVert \wvec \rVert^2$ (scikit-learn's `Ridge(alpha)`),
 * from the SVD $\Xmat_c = \Umat\operatorname{diag}(\dvec)\Vmat^\top$ of the centred design:
 * $\Smat_\lambda = \tfrac1n \mathbf{1}\mathbf{1}^\top + \Umat\operatorname{diag}(d_a^2/(d_a^2 + \lambda))\Umat^\top$.
 * Its trace is the effective degrees of freedom. Throws `DomainError` for a negative or NaN penalty.
 *
 * @param x The design, $n \times d$.
 * @param lambda The penalty $\lambda \ge 0$; 0 gives the least-squares hat matrix (a projection).
 * @param options Whether the fit has an intercept.
 * @param options.intercept Fit an unpenalised intercept by centring (default true); without it, no centring and no
 *   $\tfrac1n \mathbf{1}\mathbf{1}^\top$ term.
 * @returns The smoother, $n \times n$, symmetric.
 *
 * @example The trace falls from the number of columns plus one towards one as the penalty grows
 * import { matrixTrace } from 'aifn-compute/numerics/linalg'
 * const x = tensor([[0, 1], [1, 0], [2, 2], [3, 1], [4, 3]])
 * for (const lambda of [0, 1, 100]) print('lambda', lambda, 'trace', matrixTrace(ridgeSmoother(x, lambda)))
 */
export function ridgeSmoother(
  x: MatrixLike,
  lambda: number,
  { intercept = true }: { intercept?: boolean } = {},
): Tensor {
  checkPenalty(lambda, 'ridgeSmoother')
  const { n, u, r, d2 } = ridgeBasis(x, intercept, 'ridgeSmoother')
  const S = new Float64Array(n * n).fill(intercept ? 1 / n : 0)
  const shrink = Float64Array.from(d2, (v) => v / (v + lambda))
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let v = 0
      for (let a = 0; a < r; a++) v += u[i * r + a] * shrink[a] * u[j * r + a]
      S[i * n + j] += v
    }
  return fromData(S, [n, n])
}

/** Ridge penalties cross-validated by leave-one-out and GCV, from `ridgeCv`. */
export interface RidgeCv {
  /** The penalties $\lambda$, in the order given. */
  lambdas: number[]
  /** The leave-one-out mean squared error at each penalty, $k$ values. */
  loo: Tensor
  /** Generalised cross-validation at each penalty, $k$ values. */
  gcv: Tensor
  /** The effective degrees of freedom $\operatorname{tr}\Smat_\lambda$ at each penalty (including the intercept). */
  df: Tensor
  /** The leave-one-out residuals, $k \times n$: row $t$ at `lambdas[t]` (squared, scikit-learn's `cv_results_`). */
  looResiduals: Tensor
  /** The position in `lambdas` of the smallest leave-one-out error (the first among ties), scikit-learn's `alpha_`. */
  bestLoo: number
  /** The position in `lambdas` of the smallest GCV (the first among ties). */
  bestGcv: number
  /** The singular values of the (centred) design, largest first. */
  singularValues: Tensor
}

/**
 * Leave-one-out and generalised cross-validation of ridge regression over a grid of penalties, from one SVD of the
 * centred design (the efficient leave-one-out of scikit-learn's `RidgeCV`, whose `cv_results_` are the squared
 * `looResiduals`). At penalty $\lambda$ the leverages are
 * $S_{ii} = \tfrac1n + \sum_a U_{ia}^2 d_a^2/(d_a^2 + \lambda)$ and the leave-one-out residuals
 * $(y_i - \hat y_i)/(1 - S_{ii})$ equal those of refitting without row $i$. Throws `DomainError` for a negative or NaN
 * penalty and `ShapeError` when `x` and `y` have different numbers of rows.
 *
 * @param x The design, $n \times d$.
 * @param y The targets, $n$ values.
 * @param lambdas The penalties $\lambda \ge 0$ to try, in any order (kept).
 * @param options Whether the fit has an intercept.
 * @param options.intercept Fit an unpenalised intercept by centring (default true).
 * @returns Both curves over the penalties, the degrees of freedom, the leave-one-out residuals and the best penalty
 *   by each criterion.
 *
 * @example LOO and GCV pick nearly the same penalty
 * const s = stream(1)
 * const x = normals(s, [30, 5])
 * const y = add(matmul(x, tensor([1, -1, 0.5, 0, 0])), mul(0.8, normals(s, [30])))
 * const lambdas = [0.01, 0.1, 1, 10, 100]
 * const cv = ridgeCv(x, y, lambdas)
 * print('loo:', cv.loo)
 * print('gcv:', cv.gcv)
 * print('df:', cv.df)
 * print('best by loo, gcv:', lambdas[cv.bestLoo], lambdas[cv.bestGcv])
 */
export function ridgeCv(
  x: MatrixLike,
  y: VectorLike,
  lambdas: readonly number[],
  { intercept = true }: { intercept?: boolean } = {},
): RidgeCv {
  for (const lambda of lambdas) checkPenalty(lambda, 'ridgeCv')
  const { n, u, r, d2, singularValues } = ridgeBasis(x, intercept, 'ridgeCv')
  const target = dense.toF64(y, 'ridgeCv')
  if (target.length !== n) throw new ShapeError('ridgeCv', `ridgeCv: ${n} rows of x but ${target.length} targets`)
  const yMean = intercept ? target.reduce((a, b) => a + b, 0) / n : 0
  // Uᵀ y_c, the centred target's coordinates along the left singular vectors.
  const uy = new Float64Array(r)
  for (let i = 0; i < n; i++) for (let a = 0; a < r; a++) uy[a] += u[i * r + a] * (target[i] - yMean)
  const k = lambdas.length
  const loo = new Float64Array(k)
  const gcv = new Float64Array(k)
  const df = new Float64Array(k)
  const residuals = new Float64Array(k * n)
  lambdas.forEach((lambda, t) => {
    const shrink = Float64Array.from(d2, (v) => v / (v + lambda))
    let trace = intercept ? 1 : 0
    for (let a = 0; a < r; a++) trace += shrink[a]
    let rss = 0
    let looSum = 0
    for (let i = 0; i < n; i++) {
      let fit = yMean
      let leverage = intercept ? 1 / n : 0
      for (let a = 0; a < r; a++) {
        const uia = u[i * r + a]
        fit += uia * shrink[a] * uy[a]
        leverage += uia * uia * shrink[a]
      }
      const e = target[i] - fit
      const l = e / (1 - leverage)
      residuals[t * n + i] = l
      rss += e * e
      looSum += l * l
    }
    loo[t] = looSum / n
    gcv[t] = rss / n / (1 - trace / n) ** 2
    df[t] = trace
  })
  // The first minimum; a NaN never wins.
  const argmin = (v: Float64Array) => v.reduce((best, x, i) => (x < v[best] || Number.isNaN(v[best]) ? i : best), 0)
  return {
    lambdas: [...lambdas],
    loo: fromData(loo, [k]),
    gcv: fromData(gcv, [k]),
    df: fromData(df, [k]),
    looResiduals: fromData(residuals, [k, n]),
    bestLoo: argmin(loo),
    bestGcv: argmin(gcv),
    singularValues,
  }
}
