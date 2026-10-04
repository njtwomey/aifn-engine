/**
 * Convergence diagnostics for MCMC output (the autocorrelation function itself is `aifn-compute/probability/stats`'s
 * `autocorrelation`, by FFT): the integrated autocorrelation time,
 * the effective sample size (Geyer's initial positive and monotone sequences, multi-chain, split and rank-normalised),
 * split R̂ (rank-normalised and folded), and the Monte Carlo standard error, following Vehtari, Gelman, Simpson,
 * Carpenter & Bürkner (2021) as implemented in Stan and ArviZ.
 *
 * Fixes relative to the site's `probabilistic-inference/_shared/mcmc.ts`: the autocovariance is computed once by FFT
 * (O(n log n)) rather than lag by lag (O(n²) on slow chains); the Geyer sum adds the monotone correction; several
 * chains are combined with the between-chain variance rather than pooled; R̂ is rank-normalised and folded, so heavy
 * tails and differing scales are caught; and constant or too-short chains give explicit values, not 0/0.
 */

import { normalQuantile, regularisedBetaInverse } from 'aifn-compute/numerics/special'
import { autocovariance, quantile, ranks } from 'aifn-compute/probability/stats'
import { dense, fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { F64 } from './util'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * MCMC draws: one chain (a vector or array), several chains of equal length (an m×n matrix or m arrays), or an
 * m×n×d tensor of d parameters, as `sampleChains` returns.
 */
export type Chains = Tensor | ArrayLike<number> | readonly ArrayLike<number>[]

/** Draws as m chains × n draws, row-major. */
type Grid = { m: number; n: number; x: F64 }

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

/** Apply a per-grid diagnostic: a number for one parameter, a vector of d values for m×n×d draws. */
function perGrid(chains: Chains, where: string, f: (g: Grid) => number): number | Tensor {
  const { grids, perParameter } = toGrids(chains, where)
  const values = grids.map(f)
  return perParameter ? fromData(Float64Array.from(values), [values.length]) : values[0]
}

const row = (g: Grid, j: number) => g.x.subarray(j * g.n, (j + 1) * g.n)

/** Split each chain into its first and last halves (dropping the middle draw when n is odd): 2m chains of ⌊n/2⌋. */
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
 * Rank normalisation (Vehtari et al., 2021, eq. 14): replace every draw by Φ⁻¹((r − 3/8)/(S + 1/4)) where r is its
 * rank among all S draws (ties averaged), so the diagnostics work for heavy tails and are invariant to monotone maps.
 */
function rankNormalise(g: Grid): Grid {
  const r = dense.data(ranks(g.x))
  const S = g.x.length
  const x = new Float64Array(S)
  for (let i = 0; i < S; i++) x[i] = normalQuantile((r[i] - 0.375) / (S + 0.25)) as number
  return { m: g.m, n: g.n, x }
}

const isConstant = (x: F64) => {
  let lo = Infinity
  let hi = -Infinity
  for (const v of x) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return hi - lo < 1e-15 * Math.max(1, Math.abs(hi))
}

function meanOf(a: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]
  return s / a.length
}

/** The sample variance with n − 1 in the denominator. */
function sampleVariance(a: ArrayLike<number>): number {
  const m = meanOf(a)
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2
  return s / (a.length - 1)
}

/**
 * The effective sample size of the draws in `g` (m chains × n), Stan's and ArviZ's estimator (Geyer, 1992; Vehtari et
 * al., 2021, eqs. 10–12): combined autocorrelations ρ̂ₜ = 1 − (W − mean autocovariance at lag t)/var⁺, summed in pairs
 * while the pair sums stay positive (initial positive sequence) and forced non-increasing (initial monotone sequence);
 * ESS = mn/τ̂ with τ̂ = −1 + 2Σρ̂ₜ, floored at 1/log₁₀(mn) so ESS never exceeds mn·log₁₀(mn).
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

/** R̂ of m chains × n (Gelman & Rubin, 1992; Vehtari et al., 2021, eq. 4): √(((n − 1)/n · W + B/n)/W). */
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

/** What `effectiveSampleSize` measures. */
export type EssMethod = 'bulk' | 'tail' | 'mean'

/**
 * The effective sample size of MCMC draws (Vehtari et al., 2021): `bulk` (default) on split, rank-normalised chains,
 * which measures the efficiency of location estimates; `tail`, the smaller ESS of the indicators of the 5% and 95%
 * quantiles, for interval estimates; `mean`, on split chains without ranks, for the posterior mean (used by the MCSE).
 * Chains combine through the between-chain variance, so chains stuck in different modes give a small ESS. A number
 * for one parameter, a vector of d for m×n×d draws. NaN for fewer than 4 draws per (split) chain; mn for constant
 * draws.
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
 * The integrated autocorrelation time τ = 1 + 2 Σₜ ρₜ (Sokal, 1997), estimated as mn / ESS on the unsplit chains
 * without ranks, so n draws carry as much information about the mean as n/τ independent ones. τ ≈ 1 for independent
 * draws and (1 + φ)/(1 − φ) for an AR(1) chain with coefficient φ.
 */
export function integratedAutocorrelationTime(chains: ArrayLike<number> | readonly ArrayLike<number>[]): number
export function integratedAutocorrelationTime(chains: Chains): number | Tensor
export function integratedAutocorrelationTime(chains: Chains): number | Tensor {
  return perGrid(chains, 'integratedAutocorrelationTime', (g) => (g.m * g.n) / essOf(g))
}

/** Which R̂ `splitRhat` computes. */
export type RhatMethod = 'rank' | 'split' | 'basic'

/**
 * The potential scale reduction R̂ (Gelman & Rubin, 1992): near 1 when the chains agree, above about 1.01 when they
 * have not mixed. `rank` (default; Vehtari et al., 2021) is the larger of the rank-normalised split R̂ (bulk) and the
 * rank-normalised split R̂ of |x − median| (tails), so chains with equal means but different spreads are caught;
 * `split` is the classic split R̂ (Gelman et al., BDA3, §11.4); `basic` does not split. One chain is split in two, so
 * a single chain has a split R̂ too.
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

/** Round to the nearest integer, halves to even (NumPy's `rint`). */
function roundHalfEven(v: number): number {
  const r = Math.round(v)
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/**
 * The Monte Carlo standard error of the estimate of the posterior mean, or of the `quantile` p when one is given.
 * Mean: sd / √ESS_mean, with sd the sample standard deviation of all draws and the `mean` ESS (split chains, no ranks).
 * Quantile (Vehtari et al., 2021, §4.3): the ESS of the indicator I(x ≤ q̂ₚ) on split chains gives a Beta(ESS·p + 1,
 * ESS·(1 − p) + 1) approximation to the distribution of the empirical cdf at q̂ₚ; half the distance between the order
 * statistics at its 15.9% and 84.1% points (±1 sd of a normal) is the standard error. Both as ArviZ's `mcse`.
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

/** A summary of one parameter's draws: mean, sd, the bulk and tail ESS, R̂ and the MCSE of the mean. */
export type ChainSummary = {
  mean: number
  sd: number
  essBulk: number
  essTail: number
  rhat: number
  mcse: number
}

/** `mean`, `sd`, `essBulk`, `essTail`, `rhat` and `mcse` of one parameter's chains (m×n or one chain). */
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
