/**
 * Bayesian predictive criteria from posterior draws: WAIC (Watanabe, 2010), Pareto-smoothed importance-sampling
 * leave-one-out cross-validation (PSIS-LOO; Vehtari, Gelman and Gabry, 2017; Vehtari, Simpson, Gelman, Yao and Gabry,
 * 2024) and the deviance information criterion (DIC; Spiegelhalter, Best, Carlin and van der Linde, 2002).
 *
 * WAIC and PSIS-LOO read the pointwise log-likelihood $\ell_{si} = \log p(y_i \mid \thetavec^{(s)})$ as an
 * $S \times n$ matrix, a row per posterior draw and a column per observation, and estimate the expected log pointwise
 * predictive density $\mathrm{elpd} = \sum_i \log p(y_i \mid \yvec_{-i})$ for new data. Both start from the log
 * pointwise predictive density of the data themselves,
 * $\mathrm{lppd}_i = \log \tfrac1S \sum_s \exp \ell_{si}$, which is optimistic. WAIC subtracts the posterior variance
 * of $\ell_{\cdot i}$ (divided by $S - 1$). PSIS-LOO reweights the draws by the importance ratios
 * $r_{si} \propto 1/p(y_i \mid \thetavec^{(s)})$ that turn the posterior into the leave-one-out posterior, after
 * replacing the largest $M = \min(S/5, 3\sqrt{S})$ ratios by quantiles of a generalised Pareto distribution fitted to
 * them (Zhang and Stephens, 2009, with the weakly informative prior on its shape of Vehtari et al., 2024). The fitted
 * shape $\hat k$ is the diagnostic: the estimate is reliable for $\hat k < \min(1 - 1/\log_{10} S, 0.7)$.
 *
 * The PSIS computation follows arviz-stats 1.x (`psislw`, `loo` with `reff`) step for step, including the cap of the
 * smoothed ratios at the largest raw one and no truncation of the weights (the 2024 version of the method). Standard
 * errors are $\sqrt{n \operatorname{var}_i(\mathrm{elpd}_i)}$, the variance divided by $n - 1$. On the deviance scale
 * (`waic`, `looic`) the values and their standard errors are $-2$ and $2$ times those on the elpd scale.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, isTensor, type MatrixLike, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'

/** Pointwise contributions of an elpd estimate. */
export interface PointwiseElpd {
  /** Each observation's $\mathrm{elpd}_i$, $n$ values. */
  elpd: Tensor
  /** Each observation's effective number of parameters $p_i$, $n$ values. */
  p: Tensor
  /** Each observation's log predictive density $\mathrm{lppd}_i$, $n$ values. */
  lppd: Tensor
}

/** An estimate of the expected log pointwise predictive density, from `waic` or `psisLoo`. */
export interface ElpdEstimate {
  /** The estimate $\sum_i \mathrm{elpd}_i$ (higher is better). */
  elpd: number
  /** Its standard error, $\sqrt{n \operatorname{var}_i(\mathrm{elpd}_i)}$. */
  se: number
  /** The effective number of parameters $\sum_i p_i = \mathrm{lppd} - \mathrm{elpd}$. */
  p: number
  /** The log pointwise predictive density of the data, $\sum_i \mathrm{lppd}_i$. */
  lppd: number
  /** The pointwise values. */
  pointwise: PointwiseElpd
  /** The number of draws $S$. */
  draws: number
}

/** WAIC, from `waic`. */
export interface Waic extends ElpdEstimate {
  /** On the deviance scale, $-2\, \mathrm{elpd}$ (lower is better); its standard error is $2$ `se`. */
  waic: number
}

/** PSIS-LOO, from `psisLoo`. */
export interface PsisLoo extends ElpdEstimate {
  /** On the deviance scale, $-2\, \mathrm{elpd}$ (lower is better); its standard error is $2$ `se`. */
  looic: number
  /** The Pareto shape $\hat k_i$ of each observation's importance ratios, $n$ values; NaN where it could not be fitted. */
  khat: Tensor
  /** The threshold $\min(1 - 1/\log_{10} S, 0.7)$ above which an observation's estimate is unreliable. */
  khatThreshold: number
  /** The number of observations whose $\hat k_i$ exceeds the threshold (or is not finite). */
  bad: number
}

/** Pareto-smoothed importance weights, from `psisWeights`. */
export interface PsisWeights {
  /** The smoothed log weights, normalised so that their exponentials sum to 1, $S$ values. */
  logWeights: Tensor
  /** The fitted generalised Pareto shape $\hat k$; NaN when the tail could not be fitted (then nothing is smoothed). */
  khat: number
  /** The number $M$ of largest ratios replaced by smoothed values. */
  tailLength: number
}

/**
 * $\log \sum_i \exp a_i$, computed stably by factoring out the largest value.
 *
 * @param a The values.
 * @returns The log of the sum of their exponentials; $-\infty$ when all are $-\infty$, NaN with a NaN.
 */
function logSumExp(a: ArrayLike<number>): number {
  let m = -Infinity
  for (let i = 0; i < a.length; i++) if (a[i] > m || Number.isNaN(a[i])) m = a[i]
  if (!Number.isFinite(m)) return m
  let s = 0
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - m)
  return m + Math.log(s)
}

/**
 * The sample variance of an array, divided by its length minus one.
 *
 * @param a The values.
 * @returns The variance; NaN for fewer than two values.
 */
function sampleVariance(a: ArrayLike<number>): number {
  const n = a.length
  if (n < 2) return NaN
  let m = 0
  for (let i = 0; i < n; i++) m += a[i] / n
  let s = 0
  for (let i = 0; i < n; i++) s += (a[i] - m) ** 2
  return s / (n - 1)
}

/**
 * The pointwise log-likelihood draws as columns: column $i$ of an $S \times n$ matrix as its own array.
 *
 * @param logLik The $S \times n$ matrix.
 * @param where The caller's name, for error messages.
 * @returns The columns and the dimensions.
 */
function columns(logLik: MatrixLike, where: string): { cols: Float64Array[]; S: number; n: number } {
  const { data, m: S, n } = dense.toMatrixF64(logLik, where)
  if (S < 2) throw new DomainError(where, `${where}: needs at least two draws`)
  const cols = Array.from({ length: n }, (_, i) => Float64Array.from({ length: S }, (_, s) => data[s * n + i]))
  return { cols, S, n }
}

/**
 * Generalised Pareto fit by the empirical Bayes method of Zhang and Stephens (2009), with the weakly informative
 * prior on the shape of Vehtari et al. (2024): the shape is shrunk towards $0.5$ as if by 10 further observations. As
 * arviz-stats' `_gpdfit`.
 *
 * @param x The exceedances, sorted ascending, $M$ values.
 * @returns The shape $k$ and scale $\sigma$; both NaN when the first quartile does not exceed the minimum (not a
 *   Pareto tail), and $k$ infinite when the estimate breaks down.
 */
function generalisedParetoFit(x: Float64Array): { k: number; sigma: number } {
  const n = x.length
  const priorBs = 3
  const priorK = 10
  const m = 30 + Math.floor(Math.sqrt(n))
  const xstar = x[Math.floor(n / 4 + 0.5) - 1]
  if (xstar <= x[0]) return { k: NaN, sigma: NaN }
  const b = Float64Array.from(
    { length: m },
    (_, j) => (1 - Math.sqrt(m / (j + 0.5))) / (priorBs * xstar) + 1 / x[n - 1],
  )
  const lengths = new Float64Array(m)
  for (let j = 0; j < m; j++) {
    let k = 0
    for (let i = 0; i < n; i++) k += Math.log1p(-b[j] * x[i]) / n
    lengths[j] = n * (Math.log(-b[j] / k) - k - 1)
  }
  const lse = logSumExp(lengths)
  let w = Float64Array.from(lengths, (l) => Math.exp(l - lse))
  // Negligible weights are dropped before the posterior mean of b, as arviz-stats does.
  const keep = Array.from(w, (v) => v >= 10 * Number.EPSILON)
  const bKept = b.filter((_, j) => keep[j])
  w = w.filter((_, j) => keep[j])
  const total = w.reduce((a, v) => a + v, 0)
  let bPost = 0
  for (let j = 0; j < w.length; j++) bPost += (bKept[j] * w[j]) / total
  let k = 0
  for (let i = 0; i < n; i++) k += Math.log1p(-bPost * x[i]) / n
  const sigma = -k / bPost
  k = (n * k + priorK * 0.5) / (n + priorK)
  if (Number.isNaN(k)) return { k: Infinity, sigma: NaN }
  return { k, sigma }
}

/**
 * The quantile function of the generalised Pareto distribution with location $\mu$, scale $\sigma$ and shape $k$.
 *
 * @param p The probability.
 * @param k The shape.
 * @param sigma The scale; NaN results for a non-positive one.
 * @param mu The location.
 * @returns The quantile.
 */
function generalisedParetoQuantile(p: number, k: number, sigma: number, mu: number): number {
  if (!(sigma > 0)) return NaN
  return k === 0 ? mu - sigma * Math.log1p(-p) : mu + (sigma * Math.expm1(-k * Math.log1p(-p))) / k
}

/**
 * The PSIS tail length: $3\sqrt{S / r}$ when $S r > 225$, else $S / 5$, rounded down.
 *
 * @param S The number of draws.
 * @param rEff The relative efficiency $r$ of the draws.
 * @returns The tail length $M$.
 */
function tailLengthOf(S: number, rEff: number): number {
  return S * rEff > 225 ? Math.floor(3 * Math.sqrt(S / rEff)) : Math.floor(S / 5)
}

/**
 * Pareto-smoothed log weights of raw log importance ratios, in place: the $M$ largest are replaced by the expected
 * order statistics of a generalised Pareto fitted to them, capped at the largest raw ratio, then all are normalised.
 *
 * @param a The log ratios, $S$ values; overwritten with the normalised smoothed log weights.
 * @param M The tail length.
 * @returns The fitted shape $\hat k$.
 */
function smoothInPlace(a: Float64Array, M: number): number {
  const S = a.length
  let max = -Infinity
  for (const v of a) if (v > max) max = v
  for (let s = 0; s < S; s++) a[s] -= max
  const order = Array.from({ length: S }, (_, s) => s).sort((i, j) => a[i] - a[j])
  const tail = order.slice(S - M)
  const maxTail = a[tail[M - 1]]
  const minTail = a[tail[0]]
  let khat = NaN
  if (Math.abs(maxTail - minTail) >= Number.MIN_VALUE) {
    const cutoff = Math.exp(a[order[S - M - 1]])
    const fit = generalisedParetoFit(Float64Array.from(tail, (s) => Math.exp(a[s]) - cutoff))
    khat = fit.k
    if (Number.isFinite(fit.k)) {
      tail.forEach((s, r) => {
        const v = Math.log(generalisedParetoQuantile((r + 0.5) / M, fit.k, fit.sigma, cutoff))
        a[s] = v > maxTail ? maxTail : v
      })
    }
  }
  const lse = logSumExp(a)
  for (let s = 0; s < S; s++) a[s] -= lse
  return khat
}

/**
 * Pareto-smoothed importance sampling (Vehtari et al., 2024): normalised log weights from raw log importance ratios,
 * with the $M$ largest replaced by quantiles of a generalised Pareto distribution fitted to them, and the fitted shape
 * $\hat k$ as the diagnostic (reliable below about 0.7). As arviz-stats' `psislw` given the negated values. Throws
 * `DomainError` when the tail would have fewer than 5 draws ($S < 25$).
 *
 * @param logRatios The raw log importance ratios $\log r_s$, $S$ values (for leave-one-out, $-\ell_{si}$ for one
 *   observation $i$).
 * @param options The relative efficiency of the draws.
 * @param options.rEff The relative effective sample size $r$ of the draws ($\mathrm{ESS}/S$, default 1 for independent
 *   draws); it sets the tail length.
 * @returns The smoothed, normalised log weights, the shape $\hat k$ and the tail length.
 *
 * @example Heavy-tailed ratios: the largest weights are tamed and k-hat flags the tail
 * const s = stream(3)
 * const logRatios = mul(2, normals(s, [400]))
 * const w = psisWeights(logRatios)
 * print('k-hat:', w.khat, ' tail length:', w.tailLength)
 * print('largest raw weight share:', Math.exp(max(logRatios) - logsumexp(logRatios)))
 * print('largest smoothed weight share:', Math.exp(max(w.logWeights)))
 */
export function psisWeights(logRatios: VectorLike, { rEff = 1 }: { rEff?: number } = {}): PsisWeights {
  const a = dense.toF64(logRatios, 'psisWeights')
  const M = tailLengthOf(a.length, rEff)
  if (M < 5) throw new DomainError('psisWeights', `psisWeights: ${a.length} draws give a tail of ${M} (needs 5)`)
  const khat = smoothInPlace(a, M)
  return { logWeights: dense.vec(a), khat, tailLength: M }
}

/**
 * The widely applicable information criterion (Watanabe, 2010), as in Vehtari, Gelman and Gabry (2017), eq. 11–13:
 * $\mathrm{elpd}_i = \mathrm{lppd}_i - p_i$ with $p_i$ the posterior variance of $\ell_{\cdot i}$ over the $S$ draws
 * (divided by $S - 1$). Throws `DomainError` for fewer than two draws.
 *
 * @param logLik The pointwise log-likelihood, $S \times n$: row $s$ a posterior draw, column $i$ an observation.
 * @returns The elpd estimate with its standard error, the effective number of parameters, the lppd, WAIC on the
 *   deviance scale and the pointwise values.
 *
 * @example A normal mean with known unit variance: about one effective parameter
 * const s = stream(2)
 * const y = tensor([0.9, -1.2, 1.6, 0.1, -0.8, 1.9, 0.4, -1.5])
 * const mu = add(mean(y), mul(1 / Math.sqrt(8), normals(s, [500, 1])))
 * const logLik = sub(mul(-0.5, square(sub(y, mu))), 0.5 * Math.log(2 * Math.PI))
 * const w = waic(logLik)
 * print('elpd:', w.elpd, ' se:', w.se)
 * print('p_waic:', w.p)
 * print('waic:', w.waic)
 */
export function waic(logLik: MatrixLike): Waic {
  const { cols, S, n } = columns(logLik, 'waic')
  const lppd = new Float64Array(n)
  const p = new Float64Array(n)
  const elpd = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    lppd[i] = logSumExp(cols[i]) - Math.log(S)
    p[i] = sampleVariance(cols[i])
    elpd[i] = lppd[i] - p[i]
  }
  const summary = summarise(elpd, p, lppd, S)
  return { ...summary, waic: -2 * summary.elpd }
}

/**
 * The totals and standard error of pointwise elpd values.
 *
 * @param elpd The pointwise $\mathrm{elpd}_i$.
 * @param p The pointwise $p_i$.
 * @param lppd The pointwise $\mathrm{lppd}_i$.
 * @param S The number of draws.
 * @returns The estimate, without its deviance-scale value.
 */
function summarise(elpd: dense.F64, p: dense.F64, lppd: dense.F64, S: number): ElpdEstimate {
  const n = elpd.length
  const sum = (a: dense.F64) => a.reduce((x, y) => x + y, 0)
  return {
    elpd: sum(elpd),
    se: n > 1 ? Math.sqrt(n * sampleVariance(elpd)) : 0,
    p: sum(p),
    lppd: sum(lppd),
    pointwise: { elpd: dense.vec(elpd), p: dense.vec(p), lppd: dense.vec(lppd) },
    draws: S,
  }
}

/**
 * Pareto-smoothed importance-sampling leave-one-out cross-validation (Vehtari, Gelman and Gabry, 2017; Vehtari et al.,
 * 2024), as arviz-stats' `loo`: for each observation, the draws are reweighted by Pareto-smoothed ratios
 * $1/p(y_i \mid \thetavec^{(s)})$ (`psisWeights` of $-\ell_{\cdot i}$), and
 * $\mathrm{elpd}_i = \log \sum_s w_{si}\, p(y_i \mid \thetavec^{(s)})$. Each observation's $\hat k_i$ says whether its
 * estimate can be trusted; above `khatThreshold`, refit without it or use $K$-fold cross-validation. Throws
 * `DomainError` for fewer than 25 draws.
 *
 * @param logLik The pointwise log-likelihood, $S \times n$: row $s$ a posterior draw, column $i$ an observation.
 * @param options The relative efficiency of the draws.
 * @param options.rEff The relative effective sample size of the draws (default 1, for independent draws); arviz-stats
 *   estimates it from the chains when not given, so pass `reff=1` there to compare.
 * @returns The elpd estimate with its standard error, the effective number of parameters, LOOIC, the pointwise values
 *   and each observation's $\hat k_i$, with the threshold and the count above it.
 *
 * @example An outlier has by far the largest k-hat
 * const s = stream(4)
 * const y = tensor([0.9, -1.2, 1.6, 0.1, -0.8, 1.9, 0.4, -1.5, 0.3, -0.2, 1.1, -0.6, 0.7, -0.9, 0.2, 9])
 * const mu = add(mean(y), mul(1 / Math.sqrt(16), normals(s, [1000, 1])))
 * const logLik = sub(mul(-0.5, square(sub(y, mu))), 0.5 * Math.log(2 * Math.PI))
 * const loo = psisLoo(logLik)
 * print('elpd_loo:', loo.elpd, ' se:', loo.se, ' p_loo:', loo.p)
 * print('k-hat:', loo.khat)
 * print('above', loo.khatThreshold, ':', loo.bad)
 */
export function psisLoo(logLik: MatrixLike, { rEff = 1 }: { rEff?: number } = {}): PsisLoo {
  const { cols, S, n } = columns(logLik, 'psisLoo')
  const M = tailLengthOf(S, rEff)
  if (M < 5) throw new DomainError('psisLoo', `psisLoo: ${S} draws give a tail of ${M} (needs 5)`)
  const lppd = new Float64Array(n)
  const p = new Float64Array(n)
  const elpd = new Float64Array(n)
  const khat = new Float64Array(n)
  const threshold = Math.min(1 - 1 / Math.log10(S), 0.7)
  let bad = 0
  for (let i = 0; i < n; i++) {
    const ll = cols[i]
    const w = Float64Array.from(ll, (v) => -v)
    khat[i] = smoothInPlace(w, M)
    for (let s = 0; s < S; s++) w[s] += ll[s]
    elpd[i] = logSumExp(w)
    lppd[i] = logSumExp(ll) - Math.log(S)
    p[i] = lppd[i] - elpd[i]
    if (!(khat[i] <= threshold)) bad++
  }
  const summary = summarise(elpd, p, lppd, S)
  return { ...summary, looic: -2 * summary.elpd, khat: dense.vec(khat), khatThreshold: threshold, bad }
}

/** The deviance information criterion, from `dic`. */
export interface Dic {
  /** $\mathrm{DIC} = D(\bar\thetavec) + 2 p_D = \bar D + p_D$ (lower is better). */
  dic: number
  /** Spiegelhalter et al.'s effective number of parameters, $p_D = \bar D - D(\bar\thetavec)$. */
  pD: number
  /** Gelman's alternative $p_V = \operatorname{var}_s(D) / 2$, from the sample variance of the deviance draws. */
  pV: number
  /** DIC with $p_V$ in place of $p_D$: $\bar D + p_V$. */
  dicV: number
  /** The posterior mean deviance $\bar D = -2\, \overline{\log p(\yvec \mid \thetavec)}$. */
  meanDeviance: number
  /** The deviance at the posterior mean, $D(\bar\thetavec) = -2 \log p(\yvec \mid \bar\thetavec)$. */
  devianceAtMean: number
}

/**
 * The deviance information criterion (Spiegelhalter, Best, Carlin and van der Linde, 2002), with the deviance
 * $D(\thetavec) = -2 \log p(\yvec \mid \thetavec)$: $\mathrm{DIC} = \bar D + p_D$, $p_D = \bar D - D(\bar\thetavec)$,
 * and Gelman's $p_V = \operatorname{var}(D)/2$ (Gelman et al., 2013, §7.2) as an alternative. Unlike WAIC and PSIS-LOO
 * it uses a point estimate, so $p_D$ can be negative for a posterior mean that fits badly (a multimodal posterior).
 * Throws `DomainError` for fewer than two draws and `ShapeError` for mismatched lengths.
 *
 * @param logLikDraws The log-likelihood of all the data at each posterior draw: $S$ values, or an $S \times n$
 *   pointwise matrix whose rows are summed.
 * @param logLikAtMean The log-likelihood of all the data at the posterior mean $\bar\thetavec$: a number, or $n$
 *   pointwise values that are summed.
 * @returns DIC with $p_D$ and with $p_V$, and the deviances they are made of.
 *
 * @example A normal mean with known unit variance: both effective numbers of parameters are near 1
 * const s = stream(5)
 * const y = tensor([0.9, -1.2, 1.6, 0.1, -0.8, 1.9, 0.4, -1.5])
 * const mu = add(mean(y), mul(1 / Math.sqrt(8), normals(s, [2000, 1])))
 * const logLik = sub(mul(-0.5, square(sub(y, mu))), 0.5 * Math.log(2 * Math.PI))
 * const atMean = sub(mul(-0.5, square(sub(y, mean(y)))), 0.5 * Math.log(2 * Math.PI))
 * const d = dic(logLik, atMean)
 * print('DIC:', d.dic, ' p_D:', d.pD)
 * print('DIC (p_V):', d.dicV, ' p_V:', d.pV)
 */
export function dic(logLikDraws: VectorLike | MatrixLike, logLikAtMean: number | VectorLike): Dic {
  const rank2 = isTensor(logLikDraws) ? logLikDraws.shape.length === 2 : Array.isArray(logLikDraws[0])
  let totals: Float64Array
  let n = 0
  if (rank2) {
    const { data, m: S, n: cols } = dense.toMatrixF64(logLikDraws as MatrixLike, 'dic')
    n = cols
    totals = new Float64Array(S)
    for (let s = 0; s < S; s++) for (let i = 0; i < cols; i++) totals[s] += data[s * cols + i]
  } else totals = dense.toF64(logLikDraws as VectorLike, 'dic')
  const S = totals.length
  if (S < 2) throw new DomainError('dic', 'dic: needs at least two draws')
  let atMean: number
  if (typeof logLikAtMean === 'number') atMean = logLikAtMean
  else {
    const v = dense.toF64(logLikAtMean, 'dic')
    if (rank2 && v.length !== n) throw new ShapeError('dic', `dic: ${n} observations but ${v.length} at the mean`)
    atMean = v.reduce((a, b) => a + b, 0)
  }
  const meanDeviance = (-2 * totals.reduce((a, b) => a + b, 0)) / S
  const devianceAtMean = -2 * atMean
  const pD = meanDeviance - devianceAtMean
  const pV = 2 * sampleVariance(totals)
  return { dic: meanDeviance + pD, pD, pV, dicV: meanDeviance + pV, meanDeviance, devianceAtMean }
}
