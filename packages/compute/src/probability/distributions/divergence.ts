/**
 * Divergences and entropies of univariate distributions that have no closed form, by adaptive quadrature over the
 * supports, and wrappers that use a closed form when one is registered and quadrature otherwise, reporting which.
 * Also the Monte Carlo KL estimate with its standard error, and the pointwise KL integrand for plotting.
 *
 * Definitions follow Cover and Thomas (2006), "Elements of Information Theory", 2nd ed., §2.3 and §8 (relative and
 * differential entropy); the Jensen–Shannon divergence is Lin (1991), "Divergence measures based on the Shannon
 * entropy", IEEE Trans. Inf. Theory 37(1). Integrals use `aifn-compute/numerics/quadrature`'s adaptive Gauss–Kronrod rule.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { integrate } from 'aifn-compute/numerics/quadrature'
import type { Stream } from 'aifn-compute/foundation/random'
import { logAddExp } from 'aifn-compute/numerics/special'
import { exp, mul, sub, toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import { hasKl, kl } from './kl'
import { intervalInside, supportInterval, type Interval } from 'aifn-compute/probability/bijectors'
import type { AnyUnivariate, Distribution } from './types'
import { guard, mask, outside } from './util'

/** A value found by quadrature: the estimate, its estimated absolute error, and whether the integrator converged. */
export type NumericalResult = { value: number; error: number; converged: boolean }

/** How a divergence or entropy was computed. */
export type DivergenceMethod = 'closed form' | 'quadrature'

/** A value with the method that produced it (`error` is 0 for a closed form). */
export type MethodResult = NumericalResult & { method: DivergenceMethod }

/** The Monte Carlo estimate of a mean and its standard error s/√n. */
export type MonteCarloEstimate = { value: number; standardError: number; draws: number }

const scalar = (v: Value): number => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/** An unbatched continuous univariate distribution, or an error naming `what`. */
function continuous(d: Distribution, what: string): AnyUnivariate {
  if (d.eventShape.length !== 0 || d.batchShape.length !== 0 || d.discrete)
    throw new DomainError(what, `${what}: needs unbatched continuous univariate distributions (got ${d.name})`)
  return d as AnyUnivariate
}

/** Probabilities whose quantiles split the integration range, so each piece holds a share of the mass. */
const SPLITS = [1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-4]

/**
 * ∫ f over `domain`, split at the quantiles of each distribution in `ds` (inside the domain), each piece by adaptive
 * Gauss–Kronrod (infinite ends mapped to finite ones), and at the finite ends of their supports. Non-finite integrand values count as 0: they occur only at
 * isolated points, such as a density that is infinite at a support end.
 */
function integrateOver(f: (x: number) => number, domain: Interval, ds: readonly AnyUnivariate[]): NumericalResult {
  const cuts = new Set<number>()
  const cut = (x: number) => {
    if (Number.isFinite(x) && x > domain.lower && x < domain.upper) cuts.add(x)
  }
  for (const d of ds) {
    // Support ends first: a density may jump there, and a rule whose nodes all miss a sliver of mass reads it as 0.
    const set = supportInterval(d.support)
    cut(set.lower)
    cut(set.upper)
    for (const u of SPLITS) cut(scalar(d.quantile(u)))
  }
  const ends = [domain.lower, ...[...cuts].sort((a, b) => a - b), domain.upper]
  const g = (x: number) => {
    const y = f(x)
    return Number.isFinite(y) ? y : 0
  }
  let value = 0
  let error = 0
  let converged = true
  for (let i = 0; i + 1 < ends.length; i++) {
    if (!(ends[i + 1] > ends[i])) continue
    const r = integrate(g, ends[i], ends[i + 1], { maxIntervals: 200 })
    value += r.value
    error += r.error
    converged &&= r.converged
  }
  return { value, error, converged }
}

/** The smallest interval holding both. */
const hull = (a: Interval, b: Interval): Interval => ({
  lower: Math.min(a.lower, b.lower),
  upper: Math.max(a.upper, b.upper),
  lowerOpen: a.lower < b.lower ? a.lowerOpen : b.lowerOpen,
  upperOpen: a.upper > b.upper ? a.upperOpen : b.upperOpen,
})

/**
 * KL(p ‖ q) = ∫ p (log p − log q) dx by adaptive quadrature over the support of p, for unbatched continuous
 * univariate distributions of any families (mixtures included). Infinite (exactly, with no error) when the support of
 * p is not inside that of q. When the integral diverges (e.g. a Cauchy p against a normal q) the result is not
 * converged and its value is meaningless.
 */
export function klNumerical(p: Distribution, q: Distribution): NumericalResult {
  const a = continuous(p, 'klNumerical')
  const b = continuous(q, 'klNumerical')
  const pSet = supportInterval(a.support)
  if (!intervalInside(pSet, supportInterval(b.support))) return { value: Infinity, error: 0, converged: true }
  return integrateOver(
    (x) => {
      const lp = scalar(a.logProb(x))
      if (lp === -Infinity) return 0
      return Math.exp(lp) * (lp - scalar(b.logProb(x)))
    },
    pSet,
    [a, b],
  )
}

/**
 * KL(p ‖ q) in nats: the closed form when one is registered for the pair (see `kl`), otherwise `klNumerical`
 * (continuous univariate pairs only). `method` says which was used.
 */
export function klAuto(p: Distribution, q: Distribution): MethodResult {
  if (hasKl(p, q)) return { value: scalar(kl(p, q)), error: 0, converged: true, method: 'closed form' }
  return { ...klNumerical(p, q), method: 'quadrature' }
}

/** Differential entropy H(p) = −∫ p log p dx by quadrature over the support of p. */
export function entropyNumerical(p: Distribution): NumericalResult {
  const a = continuous(p, 'entropyNumerical')
  const r = integrateOver(
    (x) => {
      const lp = scalar(a.logProb(x))
      return lp === -Infinity ? 0 : -Math.exp(lp) * lp
    },
    supportInterval(a.support),
    [a],
  )
  return r
}

/** H(p): the family's closed form (`entropy()`) when it has one, otherwise `entropyNumerical`. */
export function entropyAuto(p: Distribution): MethodResult {
  try {
    const value = scalar(p.entropy() as Value)
    if (!Number.isNaN(value)) return { value, error: 0, converged: true, method: 'closed form' }
  } catch {
    // No closed form (e.g. a mixture): fall through to quadrature.
  }
  return { ...entropyNumerical(p), method: 'quadrature' }
}

/**
 * Cross-entropy H(p, q) = −E_p[log q(X)] = H(p) + KL(p ‖ q). Closed form when both H(p) and KL(p ‖ q) have one;
 * otherwise −∫ p log q dx by quadrature (infinite when the support of p is not inside that of q).
 */
export function crossEntropyAuto(p: Distribution, q: Distribution): MethodResult {
  const h = entropyAuto(p)
  if (h.method === 'closed form' && hasKl(p, q))
    return { value: h.value + scalar(kl(p, q)), error: 0, converged: true, method: 'closed form' }
  const a = continuous(p, 'crossEntropyAuto')
  const b = continuous(q, 'crossEntropyAuto')
  const pSet = supportInterval(a.support)
  if (!intervalInside(pSet, supportInterval(b.support)))
    return { value: Infinity, error: 0, converged: true, method: 'quadrature' }
  const r = integrateOver(
    (x) => {
      const lp = scalar(a.logProb(x))
      return lp === -Infinity ? 0 : -Math.exp(lp) * scalar(b.logProb(x))
    },
    pSet,
    [a, b],
  )
  return { ...r, method: 'quadrature' }
}

/**
 * The Jensen–Shannon divergence JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m) with m = (p + q)/2, in nats, by quadrature over
 * the union of the supports. Symmetric, finite (at most log 2) even when the supports differ, and 0 only when p = q.
 */
export function jensenShannonNumerical(p: Distribution, q: Distribution): NumericalResult {
  const a = continuous(p, 'jensenShannonNumerical')
  const b = continuous(q, 'jensenShannonNumerical')
  const term = (l: number, lm: number) => (l === -Infinity ? 0 : Math.exp(l) * (l - lm))
  return integrateOver(
    (x) => {
      const lp = scalar(a.logProb(x))
      const lq = scalar(b.logProb(x))
      if (lp === -Infinity && lq === -Infinity) return 0
      const lm = logAddExp(lp, lq) - Math.LN2
      return 0.5 * (term(lp, lm) + term(lq, lm))
    },
    hull(supportInterval(a.support), supportInterval(b.support)),
    [a, b],
  )
}

/**
 * The integrand of KL(p ‖ q) at x: p(x) (log p(x) − log q(x)), with 0 where p(x) = 0 and +∞ where p(x) > 0 = q(x).
 * Its integral is KL(p ‖ q); it is negative wherever q(x) > p(x).
 */
export function klIntegrand(p: Distribution, q: Distribution, x: Value): Value {
  const lp = p.logProb(x) as Value
  const w = exp(lp)
  const ok = mask([w], (v) => v !== 0)
  return outside(ok, mul(w, guard(sub(lp, q.logProb(x) as Value), ok, 0)), 0)
}

/**
 * A Monte Carlo estimate of KL(p ‖ q) with its standard error: the mean of log p(xᵢ) − log q(xᵢ) over n draws from p
 * (stream `s`, the same draws as `klMonteCarlo`) and the sample standard deviation over √n. Unbatched distributions.
 */
export function klMonteCarloWithError(s: Stream, p: Distribution, q: Distribution, n: number): MonteCarloEstimate {
  if (n < 2) throw new DomainError('klMonteCarloWithError', 'klMonteCarloWithError: needs at least two draws')
  const x = p.sample(s, { shape: [n] })
  const d = unwrap(sub(p.logProb(x) as Value, q.logProb(x) as Value))
  const values = typeof d === 'number' ? [d] : toFlat(d)
  let mean = 0
  for (const v of values) mean += v / values.length
  let ss = 0
  for (const v of values) ss += (v - mean) ** 2
  return { value: mean, standardError: Math.sqrt(ss / (values.length - 1) / values.length), draws: values.length }
}
