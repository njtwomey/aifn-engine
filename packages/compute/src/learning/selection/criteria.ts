/**
 * Penalised-fit criteria for choosing among models: the information criteria AIC (Akaike, 1974), AICc (Hurvich and
 * Tsai, 1989) and BIC (Schwarz, 1978), Mallows' $C_p$ (Mallows, 1973), and Stein's unbiased risk estimate (Stein,
 * 1981) with its closed forms for soft thresholding (Donoho and Johnstone, 1995) and the lasso (Zou, Hastie and
 * Tibshirani, 2007).
 *
 * Each trades the fit to the training data against a measure of complexity. For a model with maximised
 * log-likelihood $\hat\ell$ and $k$ parameters fitted to $n$ observations,
 *
 * $$ \mathrm{AIC} = -2\hat\ell + 2k, \quad \mathrm{AIC_c} = \mathrm{AIC} + \frac{2k(k + 1)}{n - k - 1}, \quad
 * \mathrm{BIC} = -2\hat\ell + k \log n, $$
 *
 * lower being better. The criteria are written with primitives, so they work elementwise over a tensor of candidates
 * (a degree sweep) with broadcasting, and are differentiable in their arguments.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  add,
  dense,
  div,
  fromData,
  lessEqual,
  log,
  mul,
  sub,
  tensor,
  where,
  type Tensor,
  type Value,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'

/**
 * A criterion's argument: one number, or one value per candidate (a tensor or an array); a traced value inside `grad`
 * and the other transforms.
 */
export type CandidateValues = Value | readonly number[]

/** The information criteria of one or many candidates, from `informationCriteria`. */
export interface InformationCriteria<T> {
  /** Akaike's criterion $-2\hat\ell + 2k$. */
  aic: T
  /** The small-sample correction $\mathrm{AIC} + 2k(k + 1)/(n - k - 1)$; NaN where $n \le k + 1$. */
  aicc: T
  /** Schwarz's criterion $-2\hat\ell + k \log n$. */
  bic: T
}

/**
 * A criterion's argument as a primitive's input: numbers stay numbers, arrays become tensors.
 *
 * @param v The argument.
 * @returns A primitive's input: a number, a tensor or a traced value.
 */
function asValue(v: CandidateValues): Value {
  return Array.isArray(v) ? tensor([...(v as readonly number[])]) : (v as Value)
}

/**
 * AIC, AICc and BIC from maximised log-likelihoods (Akaike, 1974; Hurvich and Tsai, 1989; Schwarz, 1978), elementwise
 * over candidates with broadcasting. $k$ counts every estimated parameter, including a noise variance if it is
 * estimated (statsmodels' OLS counts the coefficients only: pass its `df_model + k_constant` to match its `aic`).
 * Lower is better; differences between candidates, not the values, carry the meaning.
 *
 * @param args The log-likelihoods, parameter counts and sample sizes.
 * @param args.logLik The maximised log-likelihood $\hat\ell$ of each candidate.
 * @param args.k The number of estimated parameters $k$ of each candidate.
 * @param args.n The number of observations $n$.
 * @returns AIC, AICc and BIC: numbers for numbers in, otherwise tensors of the broadcast shape.
 *
 * @example Polynomial degrees 0 to 4 on the same 20 points: BIC penalises the extra terms harder
 * const logLik = tensor([-40.1, -22.3, -21.9, -20.8, -20.6])
 * const ic = informationCriteria({ logLik, k: tensor([2, 3, 4, 5, 6]), n: 20 })
 * print('aic:', ic.aic)
 * print('aicc:', ic.aicc)
 * print('bic:', ic.bic)
 */
export function informationCriteria(args: { logLik: number; k: number; n: number }): InformationCriteria<number>
export function informationCriteria(args: {
  logLik: CandidateValues
  k: CandidateValues
  n: CandidateValues
}): InformationCriteria<Tensor>
export function informationCriteria(args: {
  logLik: CandidateValues
  k: CandidateValues
  n: CandidateValues
}): InformationCriteria<number | Tensor> {
  const logLik = asValue(args.logLik)
  const k = asValue(args.k)
  const n = asValue(args.n)
  const deviance = mul(-2, logLik)
  const aic = add(deviance, mul(2, k))
  const room = sub(sub(n, k), 1)
  const aicc = where(lessEqual(room, 0), NaN, add(aic, div(mul(mul(2, k), add(k, 1)), room)))
  return { aic, aicc, bic: add(deviance, mul(k, log(n))) } as InformationCriteria<number | Tensor>
}

/** Mallows' $C_p$ in both of its forms, from `mallowsCp`. */
export interface MallowsCp<T> {
  /** The prediction-error form of Hastie, Tibshirani and Friedman (2009, §7.5), $\mathrm{RSS}/n + 2 d \sigma^2 / n$. */
  cp: T
  /** Mallows' original form $\mathrm{RSS}/\sigma^2 - n + 2d$, about $d$ for an unbiased model. */
  classical: T
}

/**
 * Mallows' $C_p$ (Mallows, 1973), elementwise over candidates with broadcasting, in two forms: the estimate of
 * in-sample prediction error $\mathrm{RSS}/n + 2 d \hat\sigma^2/n$ of Hastie, Tibshirani and Friedman (2009, eq. 7.26),
 * and the classical $\mathrm{RSS}/\hat\sigma^2 - n + 2d$. Both rank candidates alike. $\hat\sigma^2$ is usually the
 * residual variance of the largest model, and $d$ the number of coefficients, or a linear smoother's
 * $\operatorname{tr}\Smat$.
 *
 * @param args The residual sums of squares, degrees of freedom, noise variance and sample size.
 * @param args.rss The residual sum of squares of each candidate.
 * @param args.df The degrees of freedom $d$ of each candidate.
 * @param args.sigma2 The noise variance $\hat\sigma^2$, estimated from a low-bias model.
 * @param args.n The number of observations $n$.
 * @returns Both forms: numbers for numbers in, otherwise tensors.
 *
 * @example Adding a useless term lowers RSS a little and raises Cp
 * const cp = mallowsCp({ rss: tensor([90, 22, 21.5]), df: tensor([1, 2, 3]), sigma2: 1.1, n: 20 })
 * print('cp:', cp.cp)
 * print('classical:', cp.classical)
 */
export function mallowsCp(args: { rss: number; df: number; sigma2: number; n: number }): MallowsCp<number>
export function mallowsCp(args: {
  rss: CandidateValues
  df: CandidateValues
  sigma2: CandidateValues
  n: CandidateValues
}): MallowsCp<Tensor>
export function mallowsCp(args: {
  rss: CandidateValues
  df: CandidateValues
  sigma2: CandidateValues
  n: CandidateValues
}): MallowsCp<number | Tensor> {
  const rss = asValue(args.rss)
  const df = asValue(args.df)
  const sigma2 = asValue(args.sigma2)
  const n = asValue(args.n)
  const penalty = mul(mul(2, df), sigma2)
  return {
    cp: div(add(rss, penalty), n),
    classical: add(sub(div(rss, sigma2), n), mul(2, df)),
  } as MallowsCp<number | Tensor>
}

/**
 * Stein's unbiased risk estimate (Stein, 1981) of $\expect\lVert \hat\muvec - \muvec \rVert^2$ for an estimate
 * $\hat\muvec = g(\yvec)$ from $\yvec \sim \Gauss(\muvec, \sigma^2 \Imat)$ with $g$ weakly differentiable:
 *
 * $$ \mathrm{SURE} = -n\sigma^2 + \lVert \yvec - \hat\muvec \rVert^2 + 2\sigma^2 \sum_i \frac{\partial \hat\mu_i}{\partial
 * y_i}. $$
 *
 * The divergence $\sum_i \partial\hat\mu_i/\partial y_i$ is the estimate's degrees of freedom: $\operatorname{tr}\Smat$
 * for a linear smoother, the number of survivors for soft thresholding, the number of nonzero coefficients for the
 * lasso. Divide by $n$ for the risk per observation. Throws `ShapeError` for `y` and `fit` of different lengths.
 *
 * @param options The data, the estimate, its divergence and the noise variance.
 * @param options.y The observations $\yvec$, $n$ values.
 * @param options.fit The estimate $\hat\muvec = g(\yvec)$, $n$ values.
 * @param options.divergence The divergence $\sum_i \partial \hat\mu_i / \partial y_i$.
 * @param options.sigma2 The noise variance $\sigma^2$, assumed known.
 * @returns The estimate of the total squared-error risk $\expect\lVert \hat\muvec - \muvec \rVert^2$.
 *
 * @example Shrinking towards zero by a half: the divergence is n / 2
 * const y = tensor([2.1, -0.4, 1.3, 0.2, -1.8])
 * const fit = mul(0.5, y)
 * print('SURE of y / 2:', sure({ y, fit, divergence: 2.5, sigma2: 1 }))
 * print('SURE of y itself (n sigma^2):', sure({ y, fit: y, divergence: 5, sigma2: 1 }))
 */
export function sure({
  y,
  fit,
  divergence,
  sigma2,
}: {
  y: VectorLike
  fit: VectorLike
  divergence: number
  sigma2: number
}): number {
  const a = dense.toF64(y, 'sure')
  const b = dense.toF64(fit, 'sure')
  if (a.length !== b.length) throw new ShapeError('sure', `sure: ${a.length} observations but ${b.length} fitted`)
  let rss = 0
  for (let i = 0; i < a.length; i++) rss += (a[i] - b[i]) ** 2
  return -a.length * sigma2 + rss + 2 * sigma2 * divergence
}

/**
 * SURE for soft thresholding $\hat\mu_i = \operatorname{sign}(y_i)(|y_i| - \lambda)_+$ (Donoho and Johnstone, 1995),
 * in closed form at each threshold:
 * $-n\sigma^2 + \sum_i \min(y_i^2, \lambda^2) + 2\sigma^2\, \#\{i : |y_i| > \lambda\}$. Minimising it over $\lambda$ is
 * SureShrink's choice of threshold.
 *
 * @param y The observations $\yvec$, $n$ values (wavelet coefficients, or a sparse mean in noise).
 * @param lambda The threshold $\lambda \ge 0$, or several.
 * @param sigma2 The noise variance $\sigma^2$.
 * @returns The risk estimate: a number for one threshold, a tensor of one per threshold otherwise.
 *
 * @example A sparse mean in unit noise: SURE is lowest at a moderate threshold
 * const y = tensor([0.3, -1.1, 0.8, 4.2, -0.2, 3.5, 0.5, -0.9])
 * print('lambda:', [0, 0.5, 1, 1.5, 2, 3])
 * print('SURE:', softThresholdSure(y, [0, 0.5, 1, 1.5, 2, 3], 1))
 */
export function softThresholdSure(y: VectorLike, lambda: number, sigma2: number): number
export function softThresholdSure(y: VectorLike, lambda: VectorLike, sigma2: number): Tensor
export function softThresholdSure(y: VectorLike, lambda: number | VectorLike, sigma2: number): number | Tensor {
  const v = dense.toF64(y, 'softThresholdSure')
  const at = (l: number) => {
    let total = -v.length * sigma2
    for (const yi of v) total += Math.abs(yi) > l ? l * l + 2 * sigma2 : yi * yi
    return total
  }
  if (typeof lambda === 'number') return at(lambda)
  return dense.vec(dense.toF64(lambda, 'softThresholdSure').map(at))
}

/**
 * SURE for lasso fits (Zou, Hastie and Tibshirani, 2007), with the number of nonzero coefficients as the unbiased
 * estimate of the degrees of freedom (plus one for a fitted intercept):
 * $-n\sigma^2 + \lVert \yvec - \Xmat\hat\betavec - \hat\beta_0 \rVert^2 + 2\sigma^2 (\#\{j : \hat\beta_j \ne 0\} + 1)$.
 * Takes one fit or a path ($d \times k$ coefficients, as `elasticNetPath` of `aifn-methods/learning/linear` returns).
 * Throws `ShapeError` for mismatched shapes.
 *
 * @param args The design, targets, coefficients, intercepts and noise variance.
 * @param args.x The design $\Xmat$, $n \times d$.
 * @param args.y The targets $\yvec$, $n$ values.
 * @param args.coef The coefficients $\hat\betavec$: $d$ values for one fit, or $d \times k$ for $k$ fits.
 * @param args.intercept The fitted intercepts $\hat\beta_0$ (one per fit); left out for a fit without an intercept,
 *   which then counts no extra degree of freedom.
 * @param args.sigma2 The noise variance $\sigma^2$.
 * @returns The risk estimate and the degrees of freedom of each fit: numbers for one fit, tensors of $k$ otherwise.
 *
 * @example Two fits of a three-column design: the sparser one pays less for its degrees of freedom
 * const x = tensor([[1, 0, 0.5], [0, 1, 0.2], [1, 1, 0.1], [2, 0, 0.3], [0, 2, 0.9], [1, 2, 0.4]])
 * const y = tensor([1.1, -0.9, 0.2, 2.1, -2.2, -0.8])
 * const coef = tensor([[1, 0.9], [-1, -1.05], [0, 0.4]])
 * const s = lassoSure({ x, y, coef, intercept: [0, 0], sigma2: 0.05 })
 * print('df:', s.df)
 * print('SURE:', s.sure)
 */
export function lassoSure(args: {
  x: Tensor
  y: VectorLike
  coef: Tensor
  intercept?: number | readonly number[]
  sigma2: number
}): { sure: number | Tensor; df: number | Tensor } {
  const where = 'lassoSure'
  const { data: X, m: n, n: d } = dense.toMatrixF64(args.x, where)
  const y = dense.toF64(args.y, where)
  if (y.length !== n) throw new ShapeError(where, `${where}: ${n} rows of x but ${y.length} targets`)
  const single = args.coef.shape.length === 1
  const k = single ? 1 : args.coef.shape[1]
  const { data: B } = dense.toMatrixF64(single ? fromData(dense.data(args.coef), [d, 1]) : args.coef, where, d, k)
  const b0 =
    args.intercept === undefined ? undefined : typeof args.intercept === 'number' ? [args.intercept] : args.intercept
  if (b0 && b0.length !== k) throw new ShapeError(where, `${where}: ${k} fits but ${b0.length} intercepts`)
  const sures = new Float64Array(k)
  const dfs = new Float64Array(k)
  for (let t = 0; t < k; t++) {
    let df = b0 ? 1 : 0
    for (let j = 0; j < d; j++) if (B[j * k + t] !== 0) df++
    let rss = 0
    for (let i = 0; i < n; i++) {
      let fit = b0 ? b0[t] : 0
      for (let j = 0; j < d; j++) fit += X[i * d + j] * B[j * k + t]
      rss += (y[i] - fit) ** 2
    }
    sures[t] = -n * args.sigma2 + rss + 2 * args.sigma2 * df
    dfs[t] = df
  }
  return single ? { sure: sures[0], df: dfs[0] } : { sure: dense.vec(sures), df: dense.vec(dfs) }
}
