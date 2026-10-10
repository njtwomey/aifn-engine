/**
 * Local likelihood logistic regression (Loader 1999, "Local regression and likelihood", ch. 4; as used by Yang et al.
 * 2024, AAAI): near a point $\xvec$ the log-odds are a polynomial in $\zvec - \xvec$, so the fit at $\xvec$ maximises
 * the kernel-weighted Bernoulli log-likelihood
 *
 * $\ell_{\xvec}(\betavec) = \sum_i w_{i,h}(\xvec) [y_i \log \sigma(\eta_i) + (1 - y_i) \log(1 - \sigma(\eta_i))]$,
 * $\eta_i = \langle \betavec, A_p(\xvec_i - \xvec) \rangle$,
 *
 * with $w_{i,h}(\xvec) = K((\xvec - \xvec_i)/h)$, $K$ the Gaussian kernel, and $A_p$ the polynomial basis of order
 * $p$ ($p = 1$: $[1, v_0, v_1]$; $p = 2$ adds $[v_0^2/2, v_0 v_1, v_1^2/2]$, Loader's Eq. 2.9). The estimate is
 * $\hat{r}(\xvec) = \sigma(\hat{\beta}_0)$. Its variance is the sandwich
 * $\var[\hat{\betavec}] = \Bmat^{-1}\Cmat\Bmat^{-1}$ with $\Bmat = \sum_i w_i p_i(1 - p_i) \avec_i\avec_i^\top$ and
 * $\Cmat = \sum_i (y_i - p_i)^2 w_i^2 \avec_i\avec_i^\top$ (Frölich 2006); the covariance of the fits at two points
 * $\xvec$ and $\xvec'$ is $\Bmat_{\xvec}^{-1} (\sum_i \svec_{i,\xvec} \svec_{i,\xvec'}^\top) \Bmat_{\xvec'}^{-1}$ with
 * $\svec_i = w_i(y_i - p_i)\avec_i$ the score of point $i$, which reduces to the sandwich when $\xvec = \xvec'$. The
 * fit is the binomial GLM with prior weights $w_i$, solved by Newton's method (`aifn-compute/optim/second-order`) on
 * the weighted negative log-likelihood with autodiff gradients and Hessian; the parametric noise test fits its global
 * logistic regression with the same routine (all weights 1). Labels are $y_i \in \{0, 1\}$.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { hessian, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { matmul, mul, reshape, sub, sum, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { inverse } from 'aifn-compute/numerics/linalg'
import { sigmoid, softplus } from 'aifn-compute/numerics/special'
import { newton } from 'aifn-compute/optim/second-order'

/**
 * The weighted logistic MLE: $\hat{\betavec}$ maximising $\sum_i w_i [y_i z_i - \log(1 + e^{z_i})]$,
 * $\zvec = \Amat\betavec$, by Newton's method with exact (autodiff) Hessians from $\betavec = \zeros$, for at most
 * 100 steps (gradient tolerance $10^{-10}$).
 *
 * @param design The design matrix $\Amat$, $n \times q$ row-major: row $i$ holds point $i$'s basis values.
 * @param y The labels, $y_i \in \{0, 1\}$; their number is $n$.
 * @param weights The prior weights $w_i \ge 0$, one per point.
 * @param q The number of coefficients (columns of $\Amat$).
 * @returns `beta`, $\hat{\betavec}$ ($q$ values), `p`, the fitted $p_i = \sigma(z_i)$, and whether Newton's method
 *   converged.
 *
 * @example A logistic fit with every weight 1
 * // The design is an intercept and one feature.
 * const design = Float64Array.from([1, -2, 1, -1, 1, 0, 1, 1, 1, 2, 1, 0.5])
 * const fit = weightedLogistic(design, [0, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1], 2)
 * print('beta =', fit.beta, ' converged:', fit.converged)
 * print('fitted p =', fit.p)
 */
export function weightedLogistic(
  design: Float64Array,
  y: ArrayLike<number>,
  weights: ArrayLike<number>,
  q: Size,
): { beta: Float64Array; p: Float64Array; converged: boolean } {
  const n = y.length
  const A = fromData(design, [n, q])
  const Y = fromData(Float64Array.from(y), [n])
  const W = fromData(Float64Array.from(weights), [n])
  const nll = (b: Value) => {
    const z = reshape(matmul(A, reshape(b, [q, 1])), [n])
    return sum(mul(W, sub(softplus(z), mul(Y, z))))
  }
  const vg = valueAndGrad(nll)
  const H = hessian(nll)
  const state = run(
    newton(
      (b) => {
        const r = vg(b)
        return { value: unwrap(r.value) as number, grad: r.grad as Tensor }
      },
      { hessian: (b) => H(b) as Tensor, tolerance: 1e-10 },
    ),
    { x0: new Float64Array(q) },
    100,
  )
  const beta = Float64Array.from(toFlat(state.x))
  const p = Float64Array.from(toFlat(sigmoid(reshape(matmul(A, fromData(beta, [q, 1])), [n])) as Tensor))
  return { beta, p, converged: state.converged === true }
}

/**
 * The local polynomial basis $A_p(\vvec)$ of order $p \in \{0, 1, 2\}$ (Loader's Eq. 2.9 for $p = 2$): 1, then
 * $v_0, \dots, v_{d-1}$ for $p \ge 1$, then $v_j^2/2$ and $v_j v_k$ ($j < k$) for $p = 2$, in row order of $j \le k$.
 *
 * @param v The offset $\vvec = \xvec_i - \xvec$, $d$ values.
 * @param degree The order $p$.
 * @returns The basis values: $1$, $1 + d$ or $1 + d + d(d + 1)/2$ of them.
 *
 * @example The three bases of a two-dimensional offset
 * for (const p of [0, 1, 2]) print('p =', p, ':', localPolynomialBasis([2, 3], p))
 */
export function localPolynomialBasis(v: ArrayLike<number>, degree: 0 | 1 | 2 = 1): number[] {
  const out = [1]
  if (degree >= 1) for (let j = 0; j < v.length; j++) out.push(v[j])
  if (degree >= 2)
    for (let j = 0; j < v.length; j++)
      for (let k = j; k < v.length; k++) out.push(j === k ? (v[j] * v[j]) / 2 : v[j] * v[k])
  return out
}

/** Options of a local logistic fit. */
export interface LocalLogisticOptions {
  /** The Gaussian kernel's bandwidth $h > 0$ (default 1). */
  bandwidth?: number
  /** The polynomial order $p$ (default 1, local linear). */
  degree?: 0 | 1 | 2
}

/** A local logistic fit at one point. */
export interface LocalLogisticFit {
  /** The point $\xvec$ the fit is at. */
  readonly at: readonly number[]
  /** $\hat{r}(\xvec) = \sigma(\hat{\beta}_0)$. */
  readonly estimate: number
  /** The coefficients $\hat{\betavec}$, $q$ values; the first is the local log-odds $\hat{\beta}_0$. */
  readonly coefficients: Float64Array
  /** $\var[\hat{\betavec}] = \Bmat^{-1}\Cmat\Bmat^{-1}$, $q \times q$, row-major. */
  readonly covariance: Float64Array
  /** The $(0, 0)$ entry of `covariance`, the variance of the local log-odds $\hat{\beta}_0$. */
  readonly logitVariance: number
  /** $\Bmat^{-1}$, $q \times q$, row-major, for covariances between points. */
  readonly breadInverse: Float64Array
  /**
   * The per-point scores $\svec_i = w_i(y_i - p_i)\avec_i$, $n \times q$, row-major, for covariances between points.
   */
  readonly scores: Float64Array
  /** The number of basis functions $q$. */
  readonly basisSize: Size
  /** The effective number of points $\sum_i w_i / \max_i w_i$ (small when the neighbourhood is empty). */
  readonly effectivePoints: number
}

/**
 * A matrix argument as row-major data with its dimensions. A tensor that is not a real matrix, or ragged rows, throw.
 *
 * @param x The points, $n \times d$: a rank-2 tensor or an array of rows.
 * @param where The caller's name, for error messages.
 * @returns `data` ($n d$ values, row-major), `n` and `d`.
 */
function rowsOf(x: MatrixLike, where: string) {
  const m = dense.toMatrixF64(x, where)
  return { data: m.data, n: m.m, d: m.n }
}

/**
 * Fit the local logistic model at one point (see the file's notes): the weighted logistic MLE on the local polynomial
 * basis of the offsets, with Gaussian kernel weights $\exp(-\tfrac{1}{2}\lVert \xvec_i - \xvec \rVert^2 / h^2)$, and
 * its sandwich covariance. Throws `DomainError` for a bandwidth that is not positive and `ShapeError` when the labels
 * or the point do not match the data.
 *
 * @param x The points, $n \times d$.
 * @param y The labels, $y_i \in \{0, 1\}$, one per point.
 * @param at The point $\xvec$ to fit at, $d$ values.
 * @param options The bandwidth and the polynomial order.
 * @returns The fit: the estimate, the coefficients and their covariance, and what `localLogisticCovariance` needs.
 *
 * @example The probability of class 1 along a line, with its uncertainty
 * const s = stream(12)
 * const xs = Array.from({ length: 80 }, () => [uniform(s, -3, 3)])
 * const y = xs.map(([v]) => (uniform(s) < 1 / (1 + Math.exp(-2 * v)) ? 1 : 0))
 * for (const at of [-2, 0, 2]) {
 *   const fit = localLogistic(xs, y, [at], { bandwidth: 0.8 })
 *   print('x =', at, ' estimate:', fit.estimate, ' true:', 1 / (1 + Math.exp(-2 * at)))
 *   print('  sd of the log-odds:', Math.sqrt(fit.logitVariance))
 * }
 */
export function localLogistic(
  x: MatrixLike,
  y: ArrayLike<number>,
  at: ArrayLike<number>,
  options: LocalLogisticOptions = {},
): LocalLogisticFit {
  const { bandwidth: h = 1, degree = 1 } = options
  if (!(h > 0)) throw new DomainError('localLogistic', 'localLogistic: the bandwidth must be positive')
  const { data, n, d } = rowsOf(x, 'localLogistic')
  if (y.length !== n) throw new ShapeError('localLogistic', `localLogistic: ${y.length} labels for ${n} points`)
  if (at.length !== d)
    throw new ShapeError('localLogistic', `localLogistic: the point has ${at.length} coordinates, not ${d}`)
  const q = localPolynomialBasis(new Float64Array(d), degree).length
  const design = new Float64Array(n * q)
  const w = new Float64Array(n)
  let top = 0
  let total = 0
  for (let i = 0; i < n; i++) {
    const v = Array.from({ length: d }, (_, j) => data[i * d + j] - at[j])
    design.set(localPolynomialBasis(v, degree), i * q)
    // The Gaussian kernel's normalising constant cancels in the estimate and in the sandwich.
    w[i] = Math.exp(-0.5 * v.reduce((s, u) => s + (u / h) ** 2, 0))
    top = Math.max(top, w[i])
    total += w[i]
  }
  const { beta, p } = weightedLogistic(design, y, w, q)
  const B = new Float64Array(q * q)
  const scores = new Float64Array(n * q)
  for (let i = 0; i < n; i++) {
    const a = design.subarray(i * q, (i + 1) * q)
    const bw = w[i] * p[i] * (1 - p[i])
    const r = w[i] * (y[i] - p[i])
    for (let j = 0; j < q; j++) {
      scores[i * q + j] = r * a[j]
      for (let k = 0; k < q; k++) B[j * q + k] += bw * a[j] * a[k]
    }
  }
  const Binv = dense.data(inverse(fromData(B, [q, q])) as Tensor)
  const covariance = sandwich(Binv, scores, Binv, scores, n, q)
  return {
    at: Array.from(at),
    estimate: 1 / (1 + Math.exp(-beta[0])),
    coefficients: beta,
    covariance,
    logitVariance: covariance[0],
    breadInverse: Float64Array.from(Binv),
    scores,
    basisSize: q,
    effectivePoints: top > 0 ? total / top : 0,
  }
}

/**
 * $\Amat^{-1} (\sum_i \svec_i \tvec_i^\top) \Bmat^{-1}$ for score rows $\svec_i$ and $\tvec_i$.
 *
 * @param Ainv $\Amat^{-1}$, $q \times q$, row-major.
 * @param s The scores $\svec_i$, $n \times q$, row-major.
 * @param Binv $\Bmat^{-1}$, $q \times q$, row-major.
 * @param t The scores $\tvec_i$, $n \times q$, row-major.
 * @param n The number of points.
 * @param q The number of coefficients.
 * @returns The $q \times q$ product, row-major.
 */
function sandwich(
  Ainv: ArrayLike<number>,
  s: Float64Array,
  Binv: ArrayLike<number>,
  t: Float64Array,
  n: number,
  q: number,
) {
  const C = new Float64Array(q * q)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < q; j++) for (let k = 0; k < q; k++) C[j * q + k] += s[i * q + j] * t[i * q + k]
  return dense.matMul(dense.matMul(Ainv, C, q, q, q), Binv, q, q, q)
}

/**
 * The covariance of the local log-odds $\hat{\beta}_0$ of two fits on the same data (see the file's notes). Throws
 * `ShapeError` when the fits differ in basis size or number of points; that they share the same data is not checked.
 *
 * @param a The fit at $\xvec$.
 * @param b The fit at $\xvec'$, on the same points and labels and with the same degree.
 * @returns $\cov(\hat{\beta}_0(\xvec), \hat{\beta}_0(\xvec'))$; with `a` and `b` the same fit, its `logitVariance`.
 *
 * @example Fits close together are correlated; fits far apart are not
 * const s = stream(13)
 * const xs = Array.from({ length: 80 }, () => [uniform(s, -3, 3)])
 * const y = xs.map(([v]) => (uniform(s) < 1 / (1 + Math.exp(-2 * v)) ? 1 : 0))
 * const [a, b, c] = [-1, -0.8, 2].map((at) => localLogistic(xs, y, [at], { bandwidth: 0.5 }))
 * const corr = (f, g) => localLogisticCovariance(f, g) / Math.sqrt(f.logitVariance * g.logitVariance)
 * print('correlation at x = -1 and -0.8:', corr(a, b))
 * print('correlation at x = -1 and 2:', corr(a, c))
 * print('covariance of a fit with itself and its variance:', localLogisticCovariance(a, a), a.logitVariance)
 */
export function localLogisticCovariance(a: LocalLogisticFit, b: LocalLogisticFit): number {
  if (a.basisSize !== b.basisSize || a.scores.length !== b.scores.length)
    throw new ShapeError('localLogisticCovariance', 'localLogisticCovariance: the fits must share data and degree')
  const q = a.basisSize
  return sandwich(a.breadInverse, a.scores, b.breadInverse, b.scores, a.scores.length / q, q)[0]
}

/**
 * The bandwidth from a grid that minimises the leave-one-out log loss, the mean of
 * $-\log \hat{p}_{-i}(y_i \mid \xvec_i)$ over a random subsample of `subsample` points (Yang et al. use sub-sampled
 * LOO-CV), each point refitted without itself. Probabilities are clipped to $[10^{-12}, 1 - 10^{-12}]$. A tie keeps the
 * earlier bandwidth.
 *
 * @param s The stream that draws the subsample.
 * @param x The points, $n \times d$.
 * @param y The labels, $y_i \in \{0, 1\}$, one per point.
 * @param bandwidths The grid of bandwidths to try, each positive.
 * @param options `degree`, the polynomial order (default 1), and `subsample`, the number of points left out in turn
 *   (default 100; all of them when there are fewer).
 * @returns The chosen `bandwidth` and the mean leave-one-out log loss of each bandwidth of the grid.
 *
 * @example On a bump, too small and too large a bandwidth both lose
 * // Class 1 is likely only near 0, which one global logistic curve (a very large bandwidth) cannot follow.
 * const s = stream(15)
 * const xs = Array.from({ length: 30 }, () => [uniform(s, -3, 3)])
 * const y = xs.map(([v]) => (uniform(s) < (Math.abs(v) < 1.2 ? 0.9 : 0.1) ? 1 : 0))
 * const { bandwidth, losses } = localLogisticBandwidth(stream(16), xs, y, [0.1, 1, 10], { subsample: 6 })
 * print('log loss per bandwidth:', losses)
 * print('chosen:', bandwidth)
 */
export function localLogisticBandwidth(
  s: Stream,
  x: MatrixLike,
  y: ArrayLike<number>,
  bandwidths: readonly number[],
  options: { degree?: 0 | 1 | 2; subsample?: Size } = {},
): { bandwidth: number; losses: number[] } {
  const { degree = 1, subsample = 100 } = options
  const { data, n, d } = rowsOf(x, 'localLogisticBandwidth')
  const pick = Array.from(toFlat(permutation(child(s, 'subsample'), n))).slice(0, Math.min(subsample, n))
  const losses = bandwidths.map((h) => {
    let loss = 0
    for (const i of pick) {
      const keep = Array.from({ length: n }, (_, j) => j).filter((j) => j !== i)
      const xs = new Float64Array(keep.length * d)
      keep.forEach((j, r) => xs.set(data.subarray(j * d, (j + 1) * d), r * d))
      const fit = localLogistic(
        fromData(xs, [keep.length, d]),
        keep.map((j) => y[j]),
        data.subarray(i * d, (i + 1) * d),
        {
          bandwidth: h,
          degree,
        },
      )
      const p = Math.min(1 - 1e-12, Math.max(1e-12, fit.estimate))
      loss -= y[i] === 1 ? Math.log(p) : Math.log(1 - p)
    }
    return loss / pick.length
  })
  let best = 0
  losses.forEach((l, k) => {
    if (l < losses[best]) best = k
  })
  return { bandwidth: bandwidths[best], losses }
}
