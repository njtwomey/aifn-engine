/**
 * Convergence diagnostics for MCMC output (the autocorrelation function itself is `aifn-compute/probability/stats`'s
 * `autocorrelation`, by FFT): the integrated autocorrelation time, the effective sample size (Geyer's initial positive
 * and monotone sequences, multi-chain, split and rank-normalised), split $\hat R$ (rank-normalised and folded), and the
 * Monte Carlo standard error, following Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021) as implemented in Stan
 * and ArviZ.
 *
 * Every diagnostic takes $m$ chains of $n$ draws of one parameter and returns a number, or an $m \times n \times d$
 * tensor of $d$ parameters and returns a vector of $d$ values, one per parameter.
 *
 * Fixes relative to the site's `probabilistic-inference/_shared/mcmc.ts`: the autocovariance is computed once by FFT
 * ($O(n \log n)$) rather than lag by lag ($O(n^2)$ on slow chains); the Geyer sum adds the monotone correction;
 * several chains are combined with the between-chain variance rather than pooled; $\hat R$ is rank-normalised and
 * folded, so heavy tails and differing scales are caught; and constant or too-short chains give explicit values, not
 * $0/0$.
 */

import { normalQuantile, regularisedBetaInverse } from 'aifn-compute/numerics/special'
import { autocovariance, quantile, ranks } from 'aifn-compute/probability/stats'
import { dense, fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { F64 } from './util'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * MCMC draws: one chain (a vector or array), several chains of equal length (an $m \times n$ matrix or $m$ arrays), or
 * an $m \times n \times d$ tensor of $d$ parameters, as `sampleChains` returns.
 */
export type Chains = Tensor | ArrayLike<number> | readonly ArrayLike<number>[]

/**
 * Draws as $m$ chains $\times$ $n$ draws: `x` holds them row-major, chain $j$ in entries `j * n` to `j * n + n - 1`.
 */
type Grid = { m: number; n: number; x: F64 }

/**
 * The draws as one grid per parameter. Throws `ShapeError` for a tensor of rank above 3, or arrays of unequal length.
 *
 * @param chains The draws, in any of the forms of `Chains`; copied, not modified.
 * @param where The caller's name for error messages.
 * @returns `grids`, one for each of the $d$ parameters of an $m \times n \times d$ tensor and a single one otherwise,
 *   and `perParameter`, true for the rank-3 tensor.
 */
function toGrids(chains: Chains, where: string): { grids: Grid[]; perParameter: boolean } {
  if (isTensor(chains)) {
    const shape = chains.shape
    const flat = Float64Array.from(toFlat(chains))
    if (shape.length === 1) return { grids: [{ m: 1, n: shape[0], x: flat }], perParameter: false }
    if (shape.length === 2) return { grids: [{ m: shape[0], n: shape[1], x: flat }], perParameter: false }
    if (shape.length === 3) {
      const [m, n, d] = shape
      const grids = Array.from({ length: d }, (_, k) => {
        const x = new Float64Array(m * n)
        for (let i = 0; i < m * n; i++) x[i] = flat[i * d + k]
        return { m, n, x }
      })
      return { grids, perParameter: true }
    }
    throw new ShapeError(where, `${where}: expected draws of rank 1, 2 or 3, got shape [${shape.join(', ')}]`)
  }
  const list = chains as ArrayLike<number> | readonly ArrayLike<number>[]
  if (list.length > 0 && typeof list[0] !== 'number') {
    const rows = list as readonly ArrayLike<number>[]
    const n = rows[0].length
    const x = new Float64Array(rows.length * n)
    rows.forEach((r, i) => {
      if (r.length !== n) throw new ShapeError(where, `${where}: chains must have equal lengths`)
      x.set(Array.from(r), i * n)
    })
    return { grids: [{ m: rows.length, n, x }], perParameter: false }
  }
  const x = Float64Array.from(list as ArrayLike<number>)
  return { grids: [{ m: 1, n: x.length, x }], perParameter: false }
}

/**
 * Apply a per-grid diagnostic: a number for one parameter, a vector of $d$ values for $m \times n \times d$ draws.
 *
 * @param chains The draws, in any of the forms of `Chains`.
 * @param where The caller's name for error messages.
 * @param f The diagnostic of one parameter's grid.
 * @returns The value of `f` for a single parameter, or the vector of its values for each of the $d$ parameters.
 */
function perGrid(chains: Chains, where: string, f: (g: Grid) => number): number | Tensor {
  const { grids, perParameter } = toGrids(chains, where)
  const values = grids.map(f)
  return perParameter ? fromData(Float64Array.from(values), [values.length]) : values[0]
}

/**
 * Chain $j$ of a grid, as a view into it (not a copy).
 *
 * @param g The grid.
 * @param j The chain index, $0 \le j < m$.
 * @returns The $n$ draws of chain $j$.
 */
const row = (g: Grid, j: number) => g.x.subarray(j * g.n, (j + 1) * g.n)

/**
 * Split each chain into its first and last halves (dropping the middle draw when $n$ is odd): $2m$ chains of
 * $\lfloor n/2 \rfloor$.
 *
 * @param g The grid to split; not modified.
 * @returns A new grid whose chains $0, \dots, m - 1$ are the first halves and $m, \dots, 2m - 1$ the last halves.
 */
function split(g: Grid): Grid {
  const h = Math.floor(g.n / 2)
  const x = new Float64Array(2 * g.m * h)
  for (let j = 0; j < g.m; j++) {
    const r = row(g, j)
    x.set(r.subarray(0, h), j * h)
    x.set(r.subarray(g.n - h), (g.m + j) * h)
  }
  return { m: 2 * g.m, n: h, x }
}

/**
 * Rank normalisation (Vehtari et al., 2021, eq. 14): replace every draw by $\Phi^{-1}((r - 3/8)/(S + 1/4))$ where $r$
 * is its rank among all $S$ draws (ties averaged), so the diagnostics work for heavy tails and are invariant to
 * monotone maps.
 *
 * @param g The grid; its draws are ranked across all chains together, and it is not modified.
 * @returns A new grid of the same shape holding the normal scores.
 */
function rankNormalise(g: Grid): Grid {
  const r = dense.data(ranks(g.x))
  const S = g.x.length
  const x = new Float64Array(S)
  for (let i = 0; i < S; i++) x[i] = normalQuantile((r[i] - 0.375) / (S + 0.25)) as number
  return { m: g.m, n: g.n, x }
}

/**
 * Are the draws constant: is their range below $10^{-15} \max(1, \lvert \max x \rvert)$?
 *
 * @param x The draws.
 * @returns True when every draw is equal up to that relative tolerance (also for no draws).
 */
const isConstant = (x: F64) => {
  let lo = Infinity
  let hi = -Infinity
  for (const v of x) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return hi - lo < 1e-15 * Math.max(1, Math.abs(hi))
}

/**
 * The arithmetic mean.
 *
 * @param a The values (NaN for none).
 * @returns Their mean.
 */
function meanOf(a: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]
  return s / a.length
}

/**
 * The sample variance with $n - 1$ in the denominator.
 *
 * @param a The $n$ values ($n \ge 2$ for a finite result).
 * @returns Their unbiased sample variance.
 */
function sampleVariance(a: ArrayLike<number>): number {
  const m = meanOf(a)
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2
  return s / (a.length - 1)
}

/**
 * The effective sample size of the draws in `g` ($m$ chains $\times$ $n$), Stan's and ArviZ's estimator (Geyer, 1992;
 * Vehtari et al., 2021, eqs. 10–12): combined autocorrelations $\hat\rho_t = 1 - (W - \bar\gamma_t)/\widehat{\var}^+$
 * ($W$ the mean within-chain variance, $\bar\gamma_t$ the mean autocovariance at lag $t$, $\widehat{\var}^+$ the
 * pooled variance with the between-chain term), summed in pairs while the pair sums stay positive (initial positive
 * sequence) and forced non-increasing (initial monotone sequence); $\mathrm{ESS} = mn/\hat\tau$ with
 * $\hat\tau = -1 + 2\sum_{t \le T}\hat\rho_t + \hat\rho_{T+1}$, floored at $1/\log_{10}(mn)$ so ESS never exceeds
 * $mn\log_{10}(mn)$.
 *
 * @param g The draws, not modified.
 * @returns The ESS: NaN for $n < 4$ or any non-finite draw, $mn$ for constant draws.
 */
function essOf(g: Grid): number {
  const { m, n } = g
  if (n < 4 || !g.x.every(Number.isFinite)) return NaN
  if (isConstant(g.x)) return m * n
  // Biased (÷ n) autocovariances by FFT, one chain at a time.
  const acov: F64[] = []
  const chainMeans = new Float64Array(m)
  for (let j = 0; j < m; j++) {
    const r = row(g, j)
    acov.push(dense.data(autocovariance(r, { method: 'fft' })))
    chainMeans[j] = meanOf(r)
  }
  const meanAcov = (t: number) => {
    let s = 0
    for (let j = 0; j < m; j++) s += acov[j][t]
    return s / m
  }
  const meanVar = (meanAcov(0) * n) / (n - 1)
  let varPlus = (meanVar * (n - 1)) / n
  if (m > 1) varPlus += sampleVariance(chainMeans)
  const rho = new Float64Array(n)
  let rhoEven = 1
  rho[0] = rhoEven
  let rhoOdd = 1 - (meanVar - meanAcov(1)) / varPlus
  rho[1] = rhoOdd
  // Geyer's initial positive sequence.
  let t = 1
  while (t < n - 3 && rhoEven + rhoOdd > 0) {
    rhoEven = 1 - (meanVar - meanAcov(t + 1)) / varPlus
    rhoOdd = 1 - (meanVar - meanAcov(t + 2)) / varPlus
    if (rhoEven + rhoOdd >= 0) {
      rho[t + 1] = rhoEven
      rho[t + 2] = rhoOdd
    }
    t += 2
  }
  const maxT = t - 2
  // A last positive even-lag term improves the estimate for antithetic chains.
  if (rhoEven > 0) rho[maxT + 1] = rhoEven
  // Geyer's initial monotone sequence.
  for (let k = 1; k <= maxT - 2; k += 2) {
    if (rho[k + 1] + rho[k + 2] > rho[k - 1] + rho[k]) {
      rho[k + 1] = (rho[k - 1] + rho[k]) / 2
      rho[k + 2] = rho[k + 1]
    }
  }
  let sum = 0
  for (let k = 0; k <= maxT; k++) sum += rho[k]
  let tau = -1 + 2 * sum + (maxT + 1 < n ? rho[maxT + 1] : 0)
  tau = Math.max(tau, 1 / Math.log10(m * n))
  return rho.some(Number.isNaN) ? NaN : (m * n) / tau
}

/**
 * $\hat R$ of $m$ chains $\times$ $n$ (Gelman & Rubin, 1992; Vehtari et al., 2021, eq. 4):
 * $\sqrt{((n - 1)/n \cdot W + B/n)/W}$, with $W$ the mean within-chain variance and $B$ $n$ times the variance of the
 * chain means.
 *
 * @param g The draws, not modified.
 * @returns $\hat R$: NaN for $m < 2$, $n < 2$ or any non-finite draw; with $W = 0$, 1 when $B = 0$ and $\infty$
 *   otherwise.
 */
function rhatOf(g: Grid): number {
  const { m, n } = g
  if (m < 2 || n < 2 || !g.x.every(Number.isFinite)) return NaN
  const means = new Float64Array(m)
  let W = 0
  for (let j = 0; j < m; j++) {
    const r = row(g, j)
    means[j] = meanOf(r)
    W += sampleVariance(r)
  }
  W /= m
  const B = n * sampleVariance(means)
  if (W === 0) return B === 0 ? 1 : Infinity
  return Math.sqrt((B / W + n - 1) / n)
}

/**
 * What `effectiveSampleSize` measures: `'bulk'`, the efficiency of location estimates; `'tail'`, of the 5% and 95%
 * quantiles; `'mean'`, of the mean.
 */
export type EssMethod = 'bulk' | 'tail' | 'mean'

/**
 * The effective sample size of MCMC draws (Vehtari et al., 2021): `bulk` (default) on split, rank-normalised chains,
 * which measures the efficiency of location estimates; `tail`, the smaller ESS of the indicators of the 5% and 95%
 * quantiles, for interval estimates; `mean`, on split chains without ranks, for the posterior mean (used by the MCSE).
 * Chains combine through the between-chain variance, so chains stuck in different modes give a small ESS. A number
 * for one parameter, a vector of $d$ for $m \times n \times d$ draws. NaN for fewer than 4 draws per split chain (so
 * $n < 8$) or a non-finite draw; $mn$ for constant draws.
 *
 * @param chains The draws of $m$ chains of $n$ draws, in any of the forms of `Chains`.
 * @param options The `method`: `'bulk'`, `'tail'` or `'mean'`.
 * @returns The ESS of each parameter, comparable to the $mn$ draws.
 *
 * @example An AR(1) chain has far fewer effective draws
 * // x_t = phi x_{t-1} + z_t: the ESS of n draws is about n (1 - phi) / (1 + phi).
 * const ar1 = (phi, n) => {
 *   const z = toFlat(normals(stream(1), n))
 *   const x = [z[0]]
 *   for (let t = 1; t < n; t++) x.push(phi * x[t - 1] + z[t])
 *   return x
 * }
 * print('independent, 2000 draws:', effectiveSampleSize(ar1(0, 2000)))
 * print('phi = 0.9, 2000 draws:', effectiveSampleSize(ar1(0.9, 2000)))
 * print('n (1 - phi) / (1 + phi) =', (2000 * 0.1) / 1.9)
 *
 * @example Bulk, tail and mean ESS of several chains
 * const draws = normal(stream(2), 0, 1, { shape: [4, 100] })
 * for (const method of ['bulk', 'tail', 'mean']) print(`${method}:`, effectiveSampleSize(draws, { method }))
 */
export function effectiveSampleSize(
  chains: ArrayLike<number> | readonly ArrayLike<number>[],
  options?: { method?: EssMethod },
): number
export function effectiveSampleSize(chains: Chains, options?: { method?: EssMethod }): number | Tensor
export function effectiveSampleSize(chains: Chains, options: { method?: EssMethod } = {}): number | Tensor {
  const method = options.method ?? 'bulk'
  return perGrid(chains, 'effectiveSampleSize', (g) => {
    if (method === 'mean') return essOf(split(g))
    if (method === 'bulk') return essOf(rankNormalise(split(g)))
    const q = (p: number) => {
      const threshold = quantile(g.x, p) as number
      const indicator = g.x.map((v) => (v <= threshold ? 1 : 0))
      return essOf(split({ m: g.m, n: g.n, x: indicator }))
    }
    return Math.min(q(0.05), q(0.95))
  })
}

/**
 * The integrated autocorrelation time $\tau = 1 + 2 \sum_{t \ge 1} \rho_t$ (Sokal, 1997), estimated as
 * $mn / \mathrm{ESS}$ on the unsplit chains without ranks, so $n$ draws carry as much information about the mean as
 * $n/\tau$ independent ones. $\tau \approx 1$ for independent draws and $(1 + \phi)/(1 - \phi)$ for an AR(1) chain
 * with coefficient $\phi$. NaN for fewer than 4 draws per chain or a non-finite draw.
 *
 * @param chains The draws of $m$ chains of $n$ draws, in any of the forms of `Chains`.
 * @returns $\tau$ of each parameter: a number, or a vector of $d$ values for $m \times n \times d$ draws.
 *
 * @example AR(1) chains against $(1 + \phi)/(1 - \phi)$
 * for (const phi of [0, 0.5, 0.8]) {
 *   const z = toFlat(normals(stream(3), 2000))
 *   const x = [z[0]]
 *   for (let t = 1; t < 2000; t++) x.push(phi * x[t - 1] + z[t])
 *   print(`phi = ${phi}: tau =`, integratedAutocorrelationTime(x), 'exact', (1 + phi) / (1 - phi))
 * }
 */
export function integratedAutocorrelationTime(chains: ArrayLike<number> | readonly ArrayLike<number>[]): number
export function integratedAutocorrelationTime(chains: Chains): number | Tensor
export function integratedAutocorrelationTime(chains: Chains): number | Tensor {
  return perGrid(chains, 'integratedAutocorrelationTime', (g) => (g.m * g.n) / essOf(g))
}

/**
 * Which $\hat R$ `splitRhat` computes: `'rank'`, the larger of the rank-normalised bulk and folded split $\hat R$;
 * `'split'`, the classic split $\hat R$; `'basic'`, without splitting.
 */
export type RhatMethod = 'rank' | 'split' | 'basic'

/**
 * The potential scale reduction $\hat R$ (Gelman & Rubin, 1992): near 1 when the chains agree, above about 1.01 when
 * they have not mixed. `rank` (default; Vehtari et al., 2021) is the larger of the rank-normalised split $\hat R$
 * (bulk) and the rank-normalised split $\hat R$ of $\lvert x - \mathrm{median} \rvert$ (tails), so chains with equal
 * means but different spreads are caught; `split` is the classic split $\hat R$ (Gelman et al., BDA3, §11.4); `basic`
 * does not split. One chain is split in two, so a single chain has a split $\hat R$ too (`basic` gives NaN for it).
 * NaN for a non-finite draw or fewer than 2 draws per chain.
 *
 * @param chains The draws of $m$ chains of $n$ draws, in any of the forms of `Chains`.
 * @param options The `method`: `'rank'`, `'split'` or `'basic'`.
 * @returns $\hat R$ of each parameter: a number, or a vector of $d$ values for $m \times n \times d$ draws.
 *
 * @example Chains that agree, and chains that do not
 * const agree = normal(stream(4), 0, 1, { shape: [4, 200] })
 * const shifted = add(agree, tensor([[0], [0], [0], [1]]))
 * print('agreeing chains:', splitRhat(agree))
 * print('one chain shifted by 1 sd:', splitRhat(shifted))
 *
 * @example Equal means, different spreads: only the folded rank R-hat sees it
 * const spread = mul(normal(stream(5), 0, 1, { shape: [4, 200] }), tensor([[1], [1], [1], [3]]))
 * for (const method of ['basic', 'split', 'rank']) print(`${method}:`, splitRhat(spread, { method }))
 */
export function splitRhat(
  chains: ArrayLike<number> | readonly ArrayLike<number>[],
  options?: { method?: RhatMethod },
): number
export function splitRhat(chains: Chains, options?: { method?: RhatMethod }): number | Tensor
export function splitRhat(chains: Chains, options: { method?: RhatMethod } = {}): number | Tensor {
  const method = options.method ?? 'rank'
  return perGrid(chains, 'splitRhat', (g) => {
    if (method === 'basic') return rhatOf(g)
    const halves = split(g)
    if (method === 'split') return rhatOf(halves)
    const bulk = rhatOf(rankNormalise(halves))
    const sorted = Float64Array.from(halves.x).sort()
    const S = sorted.length
    const median = S % 2 ? sorted[(S - 1) / 2] : 0.5 * (sorted[S / 2 - 1] + sorted[S / 2])
    const folded = { m: halves.m, n: halves.n, x: halves.x.map((v) => Math.abs(v - median)) }
    const tail = rhatOf(rankNormalise(folded))
    return Math.max(bulk, tail)
  })
}

/**
 * Round to the nearest integer, halves to even (NumPy's `rint`).
 *
 * @param v The number to round.
 * @returns The nearest integer, the even one of the two when `v` is halfway between.
 */
function roundHalfEven(v: number): number {
  const r = Math.round(v)
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/**
 * The Monte Carlo standard error of the estimate of the posterior mean, or of the `quantile` $p$ when one is given.
 * Mean: $\mathrm{sd}/\sqrt{\mathrm{ESS}_\text{mean}}$, with sd the sample standard deviation of all draws and the
 * `mean` ESS (split chains, no ranks). Quantile (Vehtari et al., 2021, §4.3): the ESS of the indicator
 * $\indicator(x \le \hat q_p)$ on split chains gives a
 * $\Beta(\mathrm{ESS} \cdot p + 1, \mathrm{ESS} \cdot (1 - p) + 1)$ approximation to the distribution of the
 * empirical cdf at $\hat q_p$; half the distance between the order statistics at its 15.9% and 84.1% points
 * ($\pm 1$ sd of a normal) is the standard error. Both as ArviZ's `mcse`. Throws `DomainError` for a quantile outside
 * $(0, 1)$; NaN where the ESS is NaN.
 *
 * @param chains The draws of $m$ chains of $n$ draws, in any of the forms of `Chains`.
 * @param options The `quantile` $p \in (0, 1)$ whose estimate's error is wanted; left out, the mean's.
 * @returns The standard error of each parameter: a number, or a vector of $d$ values for $m \times n \times d$ draws.
 *
 * @example Independent draws: about $\mathrm{sd}/\sqrt{n}$
 * const draws = normal(stream(6), 0, 2, { shape: [4, 100] })
 * print('MCSE of the mean =', monteCarloStandardError(draws))
 * print('2 / sqrt(400) =', 2 / Math.sqrt(400))
 * print('MCSE of the median =', monteCarloStandardError(draws, { quantile: 0.5 }))
 * print('MCSE of the 95% quantile =', monteCarloStandardError(draws, { quantile: 0.95 }))
 */
export function monteCarloStandardError(
  chains: ArrayLike<number> | readonly ArrayLike<number>[],
  options?: { quantile?: number },
): number
export function monteCarloStandardError(chains: Chains, options?: { quantile?: number }): number | Tensor
export function monteCarloStandardError(chains: Chains, options: { quantile?: number } = {}): number | Tensor {
  const p = options.quantile
  if (p === undefined)
    return perGrid(chains, 'monteCarloStandardError', (g) => Math.sqrt(sampleVariance(g.x) / essOf(split(g))))
  if (!(p > 0 && p < 1))
    throw new DomainError(
      'monteCarloStandardError',
      `monteCarloStandardError: the quantile must lie in (0, 1), got ${p}`,
    )
  return perGrid(chains, 'monteCarloStandardError', (g) => {
    const threshold = quantile(g.x, p) as number
    const ess = essOf(split({ m: g.m, n: g.n, x: g.x.map((v) => (v <= threshold ? 1 : 0)) }))
    if (Number.isNaN(ess)) return NaN
    const a = ess * p + 1
    const b = ess * (1 - p) + 1
    const lo = regularisedBetaInverse(a, b, 0.1586553) as number
    const hi = regularisedBetaInverse(a, b, 0.8413447) as number
    const sorted = Float64Array.from(g.x).sort()
    const size = sorted.length
    const th1 = sorted[roundHalfEven(Math.max(lo * size, 0))]
    const th2 = sorted[roundHalfEven(Math.min(hi * size, size - 1))]
    return (th2 - th1) / 2
  })
}

/** A summary of one parameter's draws: mean, sd, the bulk and tail ESS, $\hat R$ and the MCSE of the mean. */
export type ChainSummary = {
  /** The mean of all draws. */
  mean: number
  /** The sample standard deviation of all draws ($n - 1$ in the denominator, over all $mn$). */
  sd: number
  /** The bulk ESS (`effectiveSampleSize` with `'bulk'`). */
  essBulk: number
  /** The tail ESS (`effectiveSampleSize` with `'tail'`). */
  essTail: number
  /** The rank-normalised split $\hat R$ (`splitRhat`). */
  rhat: number
  /** The Monte Carlo standard error of the mean (`monteCarloStandardError`). */
  mcse: number
}

/**
 * `mean`, `sd`, `essBulk`, `essTail`, `rhat` and `mcse` of one parameter's chains ($m \times n$ or one chain).
 * Throws `DomainError` for an $m \times n \times d$ tensor of $d > 1$ parameters.
 *
 * @param chains One parameter's draws: one chain, $m$ chains as arrays or an $m \times n$ matrix.
 * @returns The summary.
 *
 * @example Four chains of independent draws
 * summarise(normal(stream(7), 1, 2, { shape: [4, 200] }))
 */
export function summarise(chains: ArrayLike<number> | readonly ArrayLike<number>[] | Tensor): ChainSummary {
  const { grids } = toGrids(chains, 'summarise')
  if (grids.length !== 1) throw new DomainError('summarise', 'summarise: pass one parameter (rank ≤ 2)')
  const g = grids[0]
  return {
    mean: meanOf(g.x),
    sd: Math.sqrt(sampleVariance(g.x)),
    essBulk: effectiveSampleSize(chains as Tensor, { method: 'bulk' }) as number,
    essTail: effectiveSampleSize(chains as Tensor, { method: 'tail' }) as number,
    rhat: splitRhat(chains as Tensor) as number,
    mcse: monteCarloStandardError(chains as Tensor) as number,
  }
}
