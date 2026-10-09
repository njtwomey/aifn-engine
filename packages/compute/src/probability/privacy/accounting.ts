/**
 * Privacy accounting: the total privacy loss of a sequence of mechanisms.
 *
 * - `sequentialComposition`: $k$ mechanisms, the $i$-th $(\varepsilon_i, \delta_i)$-DP, compose to
 *   $(\sum_i \varepsilon_i, \sum_i \delta_i)$-DP (Dwork et al., 2006).
 * - `advancedComposition` (Dwork, Rothblum and Vadhan, 2010): $k$-fold composition of an $(\varepsilon, \delta)$-DP
 *   mechanism is
 *   $(\varepsilon \sqrt{2k \ln(1/\delta')} + k\varepsilon(e^\varepsilon - 1), k\delta + \delta')$-DP for any
 *   $\delta' > 0$.
 * - Rényi DP (Mironov, 2017): RDP of order $\alpha$ composes by adding. `rdpSubsampledGaussian` is the RDP of the
 *   sampled Gaussian mechanism (Poisson sampling rate $q$, noise multiplier $\sigma$) of Mironov, Talwar and Zhang
 *   (2019), computed as Opacus and TensorFlow Privacy do (exact sums for integer $\alpha$, the two-sided series for
 *   fractional $\alpha$); `rdpToEpsilon` converts to $(\varepsilon, \delta)$ by Balle et al. (2020, Thm. 21),
 *   minimised over the orders.
 * - Zero-concentrated DP (Bun and Steinke, 2016): the Gaussian mechanism is $\rho$-zCDP with
 *   $\rho = \Delta^2 / 2\sigma^2$, $\rho$ adds under composition, and $\rho$-zCDP implies
 *   $(\rho + 2\sqrt{\rho \ln(1/\delta)}, \delta)$-DP.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { normalLogCdf } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The basic composition theorem: mechanisms that are $(\varepsilon_i, \delta_i)$-DP, applied in sequence to the same
 * data, are together $(\sum_i \varepsilon_i, \sum_i \delta_i)$-DP.
 *
 * @param mechanisms The guarantee of each mechanism: its `epsilon`, and its `delta` (0 when left out).
 * @returns The total `epsilon` and `delta`.
 *
 * @example A Laplace release and a Gaussian one
 * print(sequentialComposition([{ epsilon: 0.5 }, { epsilon: 1, delta: 1e-6 }]))
 */
export function sequentialComposition(mechanisms: readonly { epsilon: number; delta?: number }[]): {
  epsilon: number
  delta: number
} {
  let epsilon = 0
  let delta = 0
  for (const m of mechanisms) {
    epsilon += m.epsilon
    delta += m.delta ?? 0
  }
  return { epsilon, delta }
}

/**
 * The advanced composition bound for $k$ uses of an $(\varepsilon, \delta)$-DP mechanism, at slack $\delta'$:
 * $(\varepsilon \sqrt{2k \ln(1/\delta')} + k\varepsilon(e^\varepsilon - 1), k\delta + \delta')$ (Dwork, Rothblum
 * and Vadhan, 2010). Tighter than sequential composition for many uses of a small $\varepsilon$. Throws a
 * `DomainError` unless $0 < \delta' < 1$.
 *
 * @param epsilon The $\varepsilon$ of one use.
 * @param delta The $\delta$ of one use.
 * @param k The number of uses $k$.
 * @param slack The extra failure probability $\delta'$ traded for the smaller $\varepsilon$.
 * @returns The total `epsilon` and `delta`.
 *
 * @example 100 uses at epsilon 0.1, against sequential composition
 * print('advanced:', advancedComposition(0.1, 0, 100, 1e-5))
 * print('sequential epsilon:', 100 * 0.1)
 */
export function advancedComposition(
  epsilon: number,
  delta: number,
  k: Size,
  slack: number,
): { epsilon: number; delta: number } {
  if (!(slack > 0 && slack < 1)) throw new DomainError('advancedComposition', 'advancedComposition: needs 0 < δ′ < 1')
  return {
    epsilon: epsilon * Math.sqrt(2 * k * Math.log(1 / slack)) + k * epsilon * Math.expm1(epsilon),
    delta: k * delta + slack,
  }
}

/** Opacus's default RDP orders: $1.1, 1.2, \dots, 10.9$ and $12, 13, \dots, 63$. */
export const DEFAULT_ORDERS: readonly number[] = [
  ...Array.from({ length: 99 }, (_, x) => 1 + (x + 1) / 10),
  ...Array.from({ length: 52 }, (_, x) => 12 + x),
]

/**
 * $\log(e^a + e^b)$, without overflow.
 *
 * @param a A log-value (may be $-\infty$).
 * @param b Another.
 * @returns The log of the sum.
 */
function logAdd(a: number, b: number): number {
  const lo = Math.min(a, b)
  const hi = Math.max(a, b)
  if (lo === -Infinity) return hi
  return Math.log1p(Math.exp(lo - hi)) + hi
}

/**
 * $\log(e^a - e^b)$ for $a \ge b$, without overflow; throws a `DomainError` when $a < b$.
 *
 * @param a The log of the larger term.
 * @param b The log of the term subtracted (may be $-\infty$).
 * @returns The log of the difference ($-\infty$ when $a = b$).
 */
function logSub(a: number, b: number): number {
  if (a < b) throw new DomainError('logSub', 'logSub: negative result')
  if (b === -Infinity) return a
  if (a === b) return -Infinity
  const d = Math.expm1(a - b)
  return Number.isFinite(d) ? Math.log(d) + b : a
}

/**
 * The generalised binomial coefficient $\binom{\alpha}{i} = \prod_{k=1}^{i} (\alpha - k + 1)/k$ for integer
 * $i \ge 0$ and real $\alpha$, with its sign (negative terms arise for fractional $\alpha$). The value itself, not
 * its log.
 *
 * @param alpha The upper argument $\alpha$, any real number.
 * @param i The lower argument $i$, a non-negative integer.
 * @returns The coefficient.
 */
function binomial(alpha: number, i: number): number {
  let c = 1
  for (let k = 1; k <= i; k++) c *= (alpha - k + 1) / k
  return c
}

/**
 * $\log \operatorname{erfc}(x)$, through the normal log cdf: $\operatorname{erfc}(x) = 2\Phi(-\sqrt{2}x)$.
 *
 * @param x The argument.
 * @returns The log of the complementary error function, accurate in the upper tail.
 */
const logErfc = (x: number) => Math.log(2) + (normalLogCdf(-x * Math.SQRT2) as number)

/**
 * $\log A_\alpha$ for integer $\alpha$, where the sampled Gaussian mechanism has RDP $\log A_\alpha / (\alpha - 1)$:
 * $A_\alpha = \sum_{i=0}^{\alpha} \binom{\alpha}{i} q^i (1 - q)^{\alpha - i} \exp((i^2 - i) / 2\sigma^2)$
 * (Mironov, Talwar and Zhang, 2019), summed in log space.
 *
 * @param q The sampling rate $q$, in $(0, 1)$.
 * @param sigma The noise multiplier $\sigma > 0$.
 * @param alpha The order $\alpha$, an integer above 1.
 * @returns $\log A_\alpha$.
 */
function logAInt(q: number, sigma: number, alpha: number): number {
  let logA = -Infinity
  for (let i = 0; i <= alpha; i++) {
    const coef = Math.log(binomial(alpha, i)) + i * Math.log(q) + (alpha - i) * Math.log(1 - q)
    logA = logAdd(logA, coef + (i * i - i) / (2 * sigma * sigma))
  }
  return logA
}

/**
 * $\log A_\alpha$ for fractional $\alpha$, by the two-sided series of Mironov, Talwar and Zhang (2019) as
 * Opacus computes it: two sums of binomial terms weighted by Gaussian tail probabilities (through `logErfc`) on either
 * side of $z_0 = \sigma^2 \ln(1/q - 1) + 1/2$, added or subtracted by the sign of the coefficient, until both terms
 * fall below $e^{-30}$.
 *
 * @param q The sampling rate $q$, in $(0, 1)$.
 * @param sigma The noise multiplier $\sigma > 0$.
 * @param alpha The order $\alpha > 1$, not an integer.
 * @returns $\log A_\alpha$.
 */
function logAFrac(q: number, sigma: number, alpha: number): number {
  let a0 = -Infinity
  let a1 = -Infinity
  const z0 = sigma * sigma * Math.log(1 / q - 1) + 0.5
  for (let i = 0; ; i++) {
    const coef = binomial(alpha, i)
    const logCoef = Math.log(Math.abs(coef))
    const j = alpha - i
    const t0 = logCoef + i * Math.log(q) + j * Math.log(1 - q)
    const t1 = logCoef + j * Math.log(q) + i * Math.log(1 - q)
    const e0 = Math.log(0.5) + logErfc((i - z0) / (Math.SQRT2 * sigma))
    const e1 = Math.log(0.5) + logErfc((z0 - j) / (Math.SQRT2 * sigma))
    const s0 = t0 + (i * i - i) / (2 * sigma * sigma) + e0
    const s1 = t1 + (j * j - j) / (2 * sigma * sigma) + e1
    if (coef > 0) {
      a0 = logAdd(a0, s0)
      a1 = logAdd(a1, s1)
    } else {
      a0 = logSub(a0, s0)
      a1 = logSub(a1, s1)
    }
    if (Math.max(s0, s1) < -30) break
  }
  return logAdd(a0, a1)
}

/**
 * The RDP $\varepsilon(\alpha)$ of `steps` compositions of the sampled Gaussian mechanism with Poisson sampling rate
 * $q$ and noise multiplier $\sigma$ (the noise's standard deviation over the $L_2$ sensitivity), at each order
 * $\alpha > 1$: one step's RDP times `steps`. The limits are exact: $q = 0$ gives 0, $\sigma = 0$ gives $\infty$,
 * and $q = 1$ gives the full Gaussian mechanism's $\alpha / 2\sigma^2$. Throws a `DomainError` for $q$ outside
 * $[0, 1]$ or an order not above 1.
 *
 * @param q The sampling rate $q$: each example is in a batch with probability $q$ (batch size over dataset size).
 * @param noiseMultiplier The noise multiplier $\sigma$.
 * @param steps The number of steps composed.
 * @param orders The orders $\alpha$ at which to evaluate (default `DEFAULT_ORDERS`).
 * @returns One RDP value per order.
 *
 * @example A few orders, after 1000 steps
 * print(rdpSubsampledGaussian(0.01, 1.1, 1000, [2, 4.5, 8, 32]))
 */
export function rdpSubsampledGaussian(
  q: number,
  noiseMultiplier: number,
  steps: Size,
  orders: readonly number[] = DEFAULT_ORDERS,
): Float64Array {
  if (!(q >= 0 && q <= 1)) throw new DomainError('rdpSubsampledGaussian', 'rdpSubsampledGaussian: needs 0 ≤ q ≤ 1')
  return Float64Array.from(orders, (alpha) => {
    if (!(alpha > 1)) throw new DomainError('rdpSubsampledGaussian', 'rdpSubsampledGaussian: orders must exceed 1')
    let rdp: number
    if (q === 0) rdp = 0
    else if (noiseMultiplier === 0) rdp = Infinity
    else if (q === 1) rdp = alpha / (2 * noiseMultiplier ** 2)
    else if (!Number.isFinite(alpha)) rdp = Infinity
    else {
      const logA = Number.isInteger(alpha) ? logAInt(q, noiseMultiplier, alpha) : logAFrac(q, noiseMultiplier, alpha)
      rdp = logA / (alpha - 1)
    }
    return rdp * steps
  })
}

/**
 * The $(\varepsilon, \delta)$ guarantee implied by RDP values at several orders (Balle et al., 2020, Thm. 21):
 * $\varepsilon = \min_\alpha [\varepsilon(\alpha) - (\log \delta + \log \alpha)/(\alpha - 1) + \log(1 - 1/\alpha)]$,
 * with the order that attains it. Throws a `ShapeError` unless there is one RDP value per order, and a `DomainError`
 * for $\delta$ outside $(0, 1)$.
 *
 * @param rdp The RDP $\varepsilon(\alpha)$ at each order, as `rdpSubsampledGaussian` returns it.
 * @param delta The target $\delta$, in $(0, 1)$.
 * @param orders The orders the RDP values belong to (default `DEFAULT_ORDERS`).
 * @returns The smallest `epsilon`, and the `order` that gives it (NaN when every value is infinite).
 *
 * @example DP-SGD's privacy from its RDP curve
 * const rdp = rdpSubsampledGaussian(0.01, 1.1, 1000)
 * print(rdpToEpsilon(rdp, 1e-5))
 */
export function rdpToEpsilon(
  rdp: ArrayLike<number>,
  delta: number,
  orders: readonly number[] = DEFAULT_ORDERS,
): { epsilon: number; order: number } {
  if (rdp.length !== orders.length) throw new ShapeError('rdpToEpsilon', 'rdpToEpsilon: one RDP value per order')
  if (!(delta > 0 && delta < 1)) throw new DomainError('rdpToEpsilon', 'rdpToEpsilon: needs 0 < δ < 1')
  let best = Infinity
  let order = NaN
  for (let k = 0; k < orders.length; k++) {
    const a = orders[k]
    const e = rdp[k] - (Math.log(delta) + Math.log(a)) / (a - 1) + Math.log((a - 1) / a)
    if (e < best) {
      best = e
      order = a
    }
  }
  return { epsilon: best, order }
}

/**
 * The $\varepsilon$ at $\delta$ after `steps` of DP-SGD with Poisson sampling rate $q$ and noise multiplier $\sigma$,
 * by RDP accounting (`rdpSubsampledGaussian` then `rdpToEpsilon`), as Opacus's `RDPAccountant`; it throws as they do.
 *
 * @param q The sampling rate $q$ (batch size over dataset size).
 * @param noiseMultiplier The noise multiplier $\sigma$.
 * @param steps The number of training steps.
 * @param delta The target $\delta$, in $(0, 1)$.
 * @param orders The RDP orders to minimise over (default `DEFAULT_ORDERS`).
 * @returns The privacy spent, $\varepsilon$.
 *
 * @example More noise, less privacy spent
 * print('sigma 0.8:', dpSgdEpsilon(256 / 60000, 0.8, 10000, 1e-5))
 * print('sigma 1.1:', dpSgdEpsilon(256 / 60000, 1.1, 10000, 1e-5))
 */
export function dpSgdEpsilon(
  q: number,
  noiseMultiplier: number,
  steps: Size,
  delta: number,
  orders: readonly number[] = DEFAULT_ORDERS,
): number {
  return rdpToEpsilon(rdpSubsampledGaussian(q, noiseMultiplier, steps, orders), delta, orders).epsilon
}

/**
 * The zCDP $\rho = \Delta^2 / (2\sigma^2)$ of the Gaussian mechanism with $L_2$ sensitivity $\Delta$ and noise
 * $\sigma$. The arguments are not checked.
 *
 * @param sensitivity The $L_2$ sensitivity $\Delta$.
 * @param sigma The noise standard deviation $\sigma$.
 * @returns $\rho$.
 *
 * @example Ten releases at sigma 4, composed and converted
 * const rho = 10 * gaussianZcdp(1, 4)
 * print('rho:', rho)
 * print('epsilon at 1e-5:', zcdpToEpsilon(rho, 1e-5))
 */
export const gaussianZcdp = (sensitivity: number, sigma: number): number => sensitivity ** 2 / (2 * sigma ** 2)

/**
 * The $(\varepsilon, \delta)$-DP implied by $\rho$-zCDP: $\varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)}$ (Bun
 * and Steinke, 2016, Prop. 1.3). Throws a `DomainError` for $\delta$ outside $(0, 1)$.
 *
 * @param rho The zCDP parameter $\rho \ge 0$ (summed over the composed mechanisms).
 * @param delta The target $\delta$, in $(0, 1)$.
 * @returns $\varepsilon$.
 *
 * @example One Gaussian release with sigma = sensitivity
 * print(zcdpToEpsilon(gaussianZcdp(1, 1), 1e-5))
 */
export function zcdpToEpsilon(rho: number, delta: number): number {
  if (!(delta > 0 && delta < 1)) throw new DomainError('zcdpToEpsilon', 'zcdpToEpsilon: needs 0 < δ < 1')
  return rho + 2 * Math.sqrt(rho * Math.log(1 / delta))
}
