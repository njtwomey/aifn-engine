/**
 * Differentially private mechanisms (Dwork, McSherry, Nissim and Smith, 2006; Dwork and Roth, 2014). A mechanism $M$ is
 * $(\varepsilon, \delta)$-differentially private when
 * $\prob(M(D) \in S) \le e^\varepsilon \prob(M(D') \in S) + \delta$ for all neighbouring datasets $D, D'$ and sets
 * $S$.
 *
 * - `laplaceMechanism`: $f(D) + \Laplace(0, \Delta_1/\varepsilon)$ per coordinate, $\varepsilon$-DP for a query of
 *   $L_1$ sensitivity $\Delta_1$.
 * - `gaussianMechanism`: $f(D) + \Gauss(0, \sigma^2)$ per coordinate. `classicGaussianSigma` is
 *   $\sigma = \Delta_2 \sqrt{2 \ln(1.25/\delta)} / \varepsilon$ (valid for $\varepsilon < 1$); `analyticGaussianSigma`
 *   (Balle and Wang, 2018) is the smallest $\sigma$ whose exact privacy profile `gaussianDelta`,
 *   $\delta(\varepsilon) = \Phi(a - b) - e^\varepsilon \Phi(-a - b)$ with $a = \Delta/2\sigma$ and
 *   $b = \varepsilon\sigma/\Delta$, is at most $\delta$, for any $\varepsilon > 0$.
 * - `exponentialMechanism` (McSherry and Talwar, 2007): picks candidate $r$ with probability
 *   $\propto \exp(\varepsilon u(r) / (2\Delta u))$.
 * - `randomisedResponse` (Warner, 1965): each bit kept with probability $e^\varepsilon/(1 + e^\varepsilon)$, else
 *   flipped; $\varepsilon$-DP per bit, with the unbiased estimate of the true share of ones from the reports.
 *
 * Randomness comes from an `aifn-compute/foundation/random` stream, the last argument. Parameters that must be positive
 * (sensitivities, $\varepsilon$, $\sigma$) are checked, and a bad one throws a `DomainError`.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalLogCdf } from 'aifn-compute/numerics/special'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A scalar result of a special function, read as a number.
 *
 * @param v The value, known to be a number.
 * @returns The same value, typed as a number.
 */
const num = (v: unknown) => v as number

/**
 * Throws a `DomainError` unless `v` is positive and finite.
 *
 * @param where The caller's name, for the error message.
 * @param name The parameter's name, for the error message.
 * @param v The value to check.
 */
const positive = (where: string, name: string, v: number) => {
  if (!(v > 0 && Number.isFinite(v)))
    throw new DomainError(where, `${where}: ${name} must be positive and finite, got ${v}`)
}

/**
 * The Laplace mechanism: $f(D) + \Laplace(0, b)$ noise on each coordinate, with scale $b = \Delta_1 / \varepsilon$:
 * $\varepsilon$-DP for a query of $L_1$ sensitivity $\Delta_1$. The noise is drawn by inversion from one uniform per
 * coordinate.
 *
 * @param value The true answer $f(D)$, one number per coordinate.
 * @param sensitivity The $L_1$ sensitivity $\Delta_1$: the most $\lVert f(D) - f(D') \rVert_1$ can be over
 *   neighbouring datasets.
 * @param epsilon The privacy parameter $\varepsilon > 0$; smaller is more private and noisier.
 * @param stream The random stream the noise is drawn from.
 * @returns The noisy answer, a new array of the same length.
 *
 * @example A count of 100 released with scale 1/0.5 = 2
 * print('released:', laplaceMechanism([100], 1, 0.5, stream(0)))
 * const draws = laplaceMechanism(new Float64Array(2000), 1, 0.5, stream(1))
 * print('mean |noise| (the scale, 2):', draws.reduce((a, x) => a + Math.abs(x), 0) / draws.length)
 */
export function laplaceMechanism(
  value: VectorLike,
  sensitivity: number,
  epsilon: number,
  stream: Stream,
): Float64Array {
  positive('laplaceMechanism', 'sensitivity', sensitivity)
  positive('laplaceMechanism', 'epsilon', epsilon)
  const v = dense.toF64(value, 'laplaceMechanism')
  const b = sensitivity / epsilon
  // Inversion: Lap(0, b) = −b sign(u − ½) log(1 − 2|u − ½|) for u uniform on (0, 1).
  const u = toFlat(uniform(stream, 0, 1, { shape: [v.length] }))
  return Float64Array.from(v, (x, i) => {
    const c = u[i] - 0.5
    return x - b * Math.sign(c) * Math.log1p(-2 * Math.abs(c))
  })
}

/**
 * The Gaussian mechanism: $f(D) + \Gauss(0, \sigma^2)$ noise on each coordinate. It is $(\varepsilon, \delta)$-DP for
 * the $\sigma$ that `classicGaussianSigma` or `analyticGaussianSigma` gives.
 *
 * @param value The true answer $f(D)$, one number per coordinate.
 * @param sigma The noise's standard deviation $\sigma > 0$ (not the noise multiplier: the sensitivity is in it).
 * @param stream The random stream the noise is drawn from.
 * @returns The noisy answer, a new array of the same length.
 *
 * @example A mean released at (1, 1e-5)-DP
 * const sigma = analyticGaussianSigma(0.01, 1, 1e-5)
 * print('sigma:', sigma)
 * print('released:', gaussianMechanism([0.42], sigma, stream(0)))
 */
export function gaussianMechanism(value: VectorLike, sigma: number, stream: Stream): Float64Array {
  positive('gaussianMechanism', 'sigma', sigma)
  const v = dense.toF64(value, 'gaussianMechanism')
  const noise = toFlat(normal(stream, 0, sigma, { shape: [v.length] }))
  return Float64Array.from(v, (u, i) => u + noise[i])
}

/**
 * The classic calibration of the Gaussian mechanism, $\sigma = \Delta_2 \sqrt{2 \ln(1.25/\delta)} / \varepsilon$
 * (Dwork and Roth, 2014, Thm. A.1), $(\varepsilon, \delta)$-DP for $\varepsilon < 1$. Throws a `DomainError`
 * outside $0 < \varepsilon < 1$ and $0 < \delta < 1$.
 *
 * @param sensitivity The $L_2$ sensitivity $\Delta_2$ of the query.
 * @param epsilon The privacy parameter $\varepsilon$, in $(0, 1)$.
 * @param delta The failure probability $\delta$, in $(0, 1)$.
 * @returns The noise standard deviation $\sigma$.
 *
 * @example Classic against analytic
 * print('classic:', classicGaussianSigma(1, 0.5, 1e-5))
 * print('analytic:', analyticGaussianSigma(1, 0.5, 1e-5))
 */
export function classicGaussianSigma(sensitivity: number, epsilon: number, delta: number): number {
  positive('classicGaussianSigma', 'sensitivity', sensitivity)
  if (!(epsilon > 0 && epsilon < 1))
    throw new DomainError('classicGaussianSigma', 'classicGaussianSigma: needs 0 < ε < 1')
  if (!(delta > 0 && delta < 1)) throw new DomainError('classicGaussianSigma', 'classicGaussianSigma: needs 0 < δ < 1')
  return (sensitivity * Math.sqrt(2 * Math.log(1.25 / delta))) / epsilon
}

/**
 * The exact privacy profile $\delta(\varepsilon)$ of the Gaussian mechanism with noise $\sigma$ and $L_2$ sensitivity
 * $\Delta$ (Balle and Wang, 2018, Thm. 8):
 * $\Phi(\Delta/2\sigma - \varepsilon\sigma/\Delta) - e^\varepsilon \Phi(-\Delta/2\sigma - \varepsilon\sigma/\Delta)$,
 * the smallest $\delta$ for which it is $(\varepsilon, \delta)$-DP. The second term is computed from the log cdf,
 * and a result below 0 by rounding is returned as 0. The arguments are not checked.
 *
 * @param sensitivity The $L_2$ sensitivity $\Delta$.
 * @param sigma The noise standard deviation $\sigma$.
 * @param epsilon The privacy parameter $\varepsilon \ge 0$.
 * @returns $\delta(\varepsilon)$, in $[0, 1]$.
 *
 * @example The profile falls as epsilon grows
 * for (const eps of [0, 1, 2, 4]) print('epsilon', eps, 'delta', gaussianDelta(1, 1, eps))
 */
export function gaussianDelta(sensitivity: number, sigma: number, epsilon: number): number {
  const a = sensitivity / (2 * sigma)
  const b = (epsilon * sigma) / sensitivity
  const first = num(normalCdf(a - b))
  const second = Math.exp(epsilon + num(normalLogCdf(-a - b)))
  return Math.max(first - second, 0)
}

/**
 * The smallest $\varepsilon$ at which the Gaussian mechanism with noise $\sigma$ and $L_2$ sensitivity $\Delta$ is
 * $(\varepsilon, \delta)$-DP: the inverse of `gaussianDelta` in $\varepsilon$ ($\delta(\varepsilon)$ decreases in
 * $\varepsilon$), by bisection to a relative tolerance of $10^{-13}$. 0 when $\delta(0) \le \delta$ already. Throws a
 * `DomainError` for a sensitivity or $\sigma$ that is not positive, or $\delta$ outside $(0, 1)$.
 *
 * @param sensitivity The $L_2$ sensitivity $\Delta$.
 * @param sigma The noise standard deviation $\sigma$.
 * @param delta The failure probability $\delta$, in $(0, 1)$.
 * @returns The smallest $\varepsilon$ (an upper end of the bisection bracket, so never below the true value).
 *
 * @example Round trip through gaussianDelta
 * const eps = gaussianEpsilon(1, 2, 1e-5)
 * print('epsilon:', eps)
 * print('delta back:', gaussianDelta(1, 2, eps))
 */
export function gaussianEpsilon(sensitivity: number, sigma: number, delta: number): number {
  positive('gaussianEpsilon', 'sensitivity', sensitivity)
  positive('gaussianEpsilon', 'sigma', sigma)
  if (!(delta > 0 && delta < 1)) throw new DomainError('gaussianEpsilon', 'gaussianEpsilon: needs 0 < δ < 1')
  if (gaussianDelta(sensitivity, sigma, 0) <= delta) return 0
  let lo = 0
  let hi = 1
  while (gaussianDelta(sensitivity, sigma, hi) > delta) hi *= 2
  for (let k = 0; k < 200 && hi - lo > 1e-13 * hi; k++) {
    const mid = (lo + hi) / 2
    if (gaussianDelta(sensitivity, sigma, mid) > delta) lo = mid
    else hi = mid
  }
  return hi
}

/**
 * The analytic Gaussian mechanism's $\sigma$ (Balle and Wang, 2018): the smallest $\sigma$ with
 * `gaussianDelta`$(\Delta, \sigma, \varepsilon) \le \delta$, by bisection on $\log \sigma$ over
 * $\Delta e^{-20}$ to $\Delta e^{20}$ ($\delta$ decreases in $\sigma$). Valid for every $\varepsilon > 0$, and never
 * larger than the classic $\sigma$. Throws a `DomainError` for a sensitivity or $\varepsilon$ that is not positive,
 * or $\delta$ outside $(0, 1)$.
 *
 * @param sensitivity The $L_2$ sensitivity $\Delta$.
 * @param epsilon The privacy parameter $\varepsilon > 0$.
 * @param delta The failure probability $\delta$, in $(0, 1)$.
 * @returns The noise standard deviation $\sigma$.
 *
 * @example Beyond the classic range, and the profile at the answer
 * const sigma = analyticGaussianSigma(1, 2, 1e-5)
 * print('sigma at epsilon = 2:', sigma)
 * print('delta at that sigma:', gaussianDelta(1, sigma, 2))
 */
export function analyticGaussianSigma(sensitivity: number, epsilon: number, delta: number): number {
  positive('analyticGaussianSigma', 'sensitivity', sensitivity)
  positive('analyticGaussianSigma', 'epsilon', epsilon)
  if (!(delta > 0 && delta < 1))
    throw new DomainError('analyticGaussianSigma', 'analyticGaussianSigma: needs 0 < δ < 1')
  let lo = Math.log(sensitivity) - 20
  let hi = Math.log(sensitivity) + 20
  for (let k = 0; k < 200; k++) {
    const mid = (lo + hi) / 2
    if (gaussianDelta(sensitivity, Math.exp(mid), epsilon) > delta) lo = mid
    else hi = mid
    if (hi - lo < 1e-13) break
  }
  return Math.exp(hi)
}

/**
 * The exponential mechanism's selection probabilities $p_r \propto \exp(\varepsilon u_r / (2\Delta u))$ over
 * candidates $r$ with utilities $u_r$, computed stably (shifted by the largest). Throws a `DomainError` for a
 * sensitivity or $\varepsilon$ that is not positive, or when no candidate has a finite utility; errors name
 * `exponentialMechanism`.
 *
 * @param utilities The utility $u_r$ of each candidate (higher is better); $-\infty$ rules a candidate out.
 * @param sensitivity The sensitivity $\Delta u$ of the utility: the most one candidate's utility can change between
 *   neighbouring datasets.
 * @param epsilon The privacy parameter $\varepsilon > 0$.
 * @returns The probabilities, one per candidate, summing to 1.
 *
 * @example Three candidates, more and less private
 * print('epsilon 1:', exponentialMechanismProbabilities([1, 2, 3], 1, 1))
 * print('epsilon 10:', exponentialMechanismProbabilities([1, 2, 3], 1, 10))
 */
export function exponentialMechanismProbabilities(
  utilities: VectorLike,
  sensitivity: number,
  epsilon: number,
): Float64Array {
  positive('exponentialMechanism', 'sensitivity', sensitivity)
  positive('exponentialMechanism', 'epsilon', epsilon)
  const u = dense.toF64(utilities, 'exponentialMechanism')
  const logits = Float64Array.from(u, (x) => (epsilon * x) / (2 * sensitivity))
  // A loop, not Math.max(...logits): spreading a long candidate list overflows the stack (review G2).
  let top = -Infinity
  for (const l of logits) if (l > top) top = l
  if (!(top > -Infinity) || Number.isNaN(top))
    throw new DomainError(
      'exponentialMechanism',
      'exponentialMechanism: needs at least one candidate with a finite utility',
    )
  const w = Float64Array.from(logits, (l) => Math.exp(l - top))
  const z = w.reduce((a, b) => a + b, 0)
  return w.map((x) => x / z)
}

/**
 * Pick a candidate by the exponential mechanism: $\varepsilon$-DP for utilities of sensitivity $\Delta u$. Draws one
 * uniform and walks the probabilities of `exponentialMechanismProbabilities`, which throws as that does.
 *
 * @param utilities The utility of each candidate (higher is better).
 * @param sensitivity The sensitivity $\Delta u$ of the utility.
 * @param epsilon The privacy parameter $\varepsilon > 0$.
 * @param stream The random stream the choice is drawn from.
 * @returns The index of the chosen candidate.
 *
 * @example The most useful candidate usually wins
 * const s = stream(3)
 * print('choices:', Array.from({ length: 10 }, () => exponentialMechanism([1, 2, 3], 1, 4, s)))
 */
export function exponentialMechanism(
  utilities: VectorLike,
  sensitivity: number,
  epsilon: number,
  stream: Stream,
): Size {
  const p = exponentialMechanismProbabilities(utilities, sensitivity, epsilon)
  let r = num(uniform(stream))
  for (let i = 0; i < p.length; i++) {
    r -= p[i]
    if (r < 0) return i
  }
  return p.length - 1
}

/**
 * The probability $e^\varepsilon / (1 + e^\varepsilon)$ that randomised response reports the true bit, computed as
 * $1 / (1 + e^{-\varepsilon})$.
 *
 * @param epsilon The privacy parameter $\varepsilon$ (not checked).
 * @returns The probability of telling the truth.
 *
 * @example Warner's coin, and more privacy
 * print('epsilon ln 3 (keep 3/4):', randomisedResponseKeep(Math.log(3)))
 * print('epsilon 0 (a fair coin):', randomisedResponseKeep(0))
 */
export const randomisedResponseKeep = (epsilon: number): number => 1 / (1 + Math.exp(-epsilon))

/**
 * Randomised response: each bit (0 or 1) reported truthfully with probability $e^\varepsilon/(1 + e^\varepsilon)$,
 * else flipped, which is $\varepsilon$-DP for each person's bit. Throws a `DomainError` for $\varepsilon$ that is not
 * positive.
 *
 * @param bits The true bits, each 0 or 1 (a flipped entry is `1 - x`).
 * @param epsilon The privacy parameter $\varepsilon > 0$.
 * @param stream The random stream the coin flips are drawn from.
 * @returns The reported bits, a new array of the same length.
 *
 * @example Ten bits at ln 3 (each kept with probability 3/4)
 * const truth = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
 * print('reported:', randomisedResponse(truth, Math.log(3), stream(2)))
 */
export function randomisedResponse(bits: VectorLike, epsilon: number, stream: Stream): Float64Array {
  positive('randomisedResponse', 'epsilon', epsilon)
  const b = dense.toF64(bits, 'randomisedResponse')
  const keep = randomisedResponseKeep(epsilon)
  const u = toFlat(uniform(stream, 0, 1, { shape: [b.length] }))
  return Float64Array.from(b, (x, i) => (u[i] < keep ? x : 1 - x))
}

/**
 * The unbiased estimate of the share of ones from randomised-response reports: $(\bar r - (1 - p)) / (2p - 1)$, with
 * $\bar r$ the share of reported ones and $p$ the probability of a truthful report. It may fall outside $[0, 1]$ for a
 * small sample.
 *
 * @param reports The reported bits.
 * @param epsilon The $\varepsilon$ the reports were made with (not checked: 0 divides by zero).
 * @returns The estimated share of true ones.
 *
 * @example Recovering a share of 0.3 from noisy reports
 * const truth = Array.from({ length: 2000 }, (_, i) => (i < 600 ? 1 : 0))
 * const reports = randomisedResponse(truth, 1, stream(4))
 * print('share reported:', reports.reduce((a, x) => a + x, 0) / reports.length)
 * print('estimate:', randomisedResponseEstimate(reports, 1))
 */
export function randomisedResponseEstimate(reports: VectorLike, epsilon: number): number {
  const r = dense.toF64(reports, 'randomisedResponseEstimate')
  const p = randomisedResponseKeep(epsilon)
  const mean = r.reduce((a, c) => a + c, 0) / r.length
  return (mean - (1 - p)) / (2 * p - 1)
}
