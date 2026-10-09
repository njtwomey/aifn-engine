/**
 * Divergences and entropies of univariate distributions that have no closed form, by adaptive quadrature over the
 * supports, and wrappers that use a closed form when one is registered and quadrature otherwise, reporting which.
 * Also the Monte Carlo KL estimate with its standard error, and the pointwise KL integrand for plotting.
 *
 * Definitions follow Cover and Thomas (2006), "Elements of Information Theory", 2nd ed., §2.3 and §8 (relative and
 * differential entropy); the Jensen–Shannon divergence is Lin (1991), "Divergence measures based on the Shannon
 * entropy", IEEE Trans. Inf. Theory 37(1). Integrals use `aifn-compute/numerics/quadrature`'s adaptive Gauss–Kronrod
 * rule. The quadrature functions take unbatched continuous univariate distributions (else `DomainError`), return
 * numbers, and are not differentiable; a result says whether the integrator converged rather than throwing.
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

/**
 * A value found by quadrature: `value`, the estimate; `error`, its estimated absolute error (summed over the pieces
 * of the range); and `converged`, whether the integrator met its tolerance on every piece.
 */
export type NumericalResult = { value: number; error: number; converged: boolean }

/** How a divergence or entropy was computed. */
export type DivergenceMethod = 'closed form' | 'quadrature'

/**
 * A value with the method that produced it: the fields of `NumericalResult` (`error` is 0 and `converged` true for a
 * closed form) and `method`, `'closed form'` or `'quadrature'`.
 */
export type MethodResult = NumericalResult & { method: DivergenceMethod }

/**
 * The Monte Carlo estimate of a mean: `value`, the sample mean; `standardError`, $s/\sqrt{n}$ with $s$ the sample
 * standard deviation; and `draws`, the number $n$ of values averaged.
 */
export type MonteCarloEstimate = { value: number; standardError: number; draws: number }

/**
 * A value as a number: a number itself, or the first element of a tensor (read through a traced value).
 *
 * @param v The value, a scalar in practice.
 */
const scalar = (v: Value): number => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/**
 * The distribution as an unbatched continuous univariate one, or a `DomainError` naming `what`.
 *
 * @param d The distribution to check.
 * @param what The caller's name, for the error message.
 * @returns `d`, typed as univariate.
 */
function continuous(d: Distribution, what: string): AnyUnivariate {
  if (d.eventShape.length !== 0 || d.batchShape.length !== 0 || d.discrete)
    throw new DomainError(what, `${what}: needs unbatched continuous univariate distributions (got ${d.name})`)
  return d as AnyUnivariate
}

/** Probabilities whose quantiles split the integration range, so each piece holds a share of the mass. */
const SPLITS = [1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-4]

/**
 * $\int f(x)\,dx$ over `domain`, split at the finite ends of the supports of the distributions in `ds` and at their
 * quantiles (those inside the domain), each piece by adaptive Gauss–Kronrod (infinite ends mapped to finite ones, at
 * most 200 subintervals a piece). Non-finite integrand values count as 0: they occur only at isolated points, such as
 * a density that is infinite at a support end.
 *
 * @param f The integrand, a function of a number.
 * @param domain The interval to integrate over.
 * @param ds The distributions whose support ends and quantiles at the probabilities `SPLITS` split the range, so that
 *   each piece holds a share of the mass.
 * @returns The integral, the summed error estimates of the pieces, and whether every piece converged.
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

/**
 * The smallest interval holding both, each end open or closed as the interval it comes from.
 *
 * @param a One interval.
 * @param b The other.
 */
const hull = (a: Interval, b: Interval): Interval => ({
  lower: Math.min(a.lower, b.lower),
  upper: Math.max(a.upper, b.upper),
  lowerOpen: a.lower < b.lower ? a.lowerOpen : b.lowerOpen,
  upperOpen: a.upper > b.upper ? a.upperOpen : b.upperOpen,
})

/**
 * $\KL(p \,\Vert\, q) = \int p(x) (\log p(x) - \log q(x))\,dx$ by adaptive quadrature over the support of $p$, for
 * unbatched continuous univariate distributions of any families (mixtures included). Infinite (exactly, with no
 * error) when the support of $p$ is not inside that of $q$. When the integral diverges (e.g. a Cauchy $p$ against a
 * normal $q$) the result is not converged and its value is meaningless.
 *
 * @param p The first distribution, the one the integral is weighted by.
 * @param q The second distribution.
 * @returns The divergence in nats, its error estimate and whether the quadrature converged.
 *
 * @example Quadrature against the closed form
 * print('quadrature:', klNumerical(Normal(0, 1), Normal(1, 2)))
 * print('closed form:', kl(Normal(0, 1), Normal(1, 2)))
 *
 * @example Infinite when p has mass where q has none
 * print('KL(normal ‖ exponential):', klNumerical(Normal(0, 1), Exponential(1)))
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
 * $\KL(p \,\Vert\, q)$ in nats: the closed form when one is registered for the pair (see `kl`), otherwise
 * `klNumerical` (unbatched continuous univariate pairs only, else `DomainError`). `method` says which was used. The
 * value is a number: for a batched closed form it is the first element of the batch.
 *
 * @param p The first distribution, the one the expectation is under.
 * @param q The second distribution.
 * @returns The divergence, with its error estimate, convergence and method.
 *
 * @example A closed form, then quadrature
 * print('normal, normal:', klAuto(Normal(0, 1), Normal(1, 2)))
 * print('normal, Laplace:', klAuto(Normal(0, 1), Laplace(0, 1)))
 * const entropyOfNormal = 0.5 * Math.log(2 * Math.PI * Math.E)
 * print('by hand: log 2 + √(2/π) − H(N(0, 1)) =', Math.log(2) + Math.sqrt(2 / Math.PI) - entropyOfNormal)
 */
export function klAuto(p: Distribution, q: Distribution): MethodResult {
  if (hasKl(p, q)) return { value: scalar(kl(p, q)), error: 0, converged: true, method: 'closed form' }
  return { ...klNumerical(p, q), method: 'quadrature' }
}

/**
 * The differential entropy $\entropy(p) = -\int p(x) \log p(x)\,dx$ by quadrature over the support of $p$, for an
 * unbatched continuous univariate distribution (else `DomainError`).
 *
 * @param p The distribution.
 * @returns The entropy in nats, its error estimate and whether the quadrature converged.
 *
 * @example A normal's entropy, against its closed form
 * print('quadrature:', entropyNumerical(Normal(0, 2)))
 * print('log(2πe σ²) / 2 =', 0.5 * Math.log(2 * Math.PI * Math.E * 4))
 */
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

/**
 * $\entropy(p)$: the family's closed form (`entropy()`) when it has one (it neither throws nor gives NaN), otherwise
 * `entropyNumerical`. `method` says which was used. Discrete distributions with a closed form are accepted; for a
 * batched distribution the value is the first element of the batch.
 *
 * @param p The distribution.
 * @returns The entropy in nats, with its error estimate, convergence and method.
 *
 * @example A normal has a closed form; a mixture does not
 * print('normal:', entropyAuto(Normal(0, 1)))
 * print('mixture:', entropyAuto(Mixture([0.5, 0.5], [Normal(-5, 1), Normal(5, 1)])))
 */
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
 * The cross-entropy $\entropy(p, q) = -\expect_p[\log q(X)] = \entropy(p) + \KL(p \,\Vert\, q)$. Closed form when
 * both $\entropy(p)$ and $\KL(p \,\Vert\, q)$ have one; otherwise $-\int p(x) \log q(x)\,dx$ by quadrature
 * (infinite when the support of $p$ is not inside that of $q$), for unbatched continuous univariate pairs (else
 * `DomainError`).
 *
 * @param p The distribution the expectation is under.
 * @param q The distribution whose log-density is averaged.
 * @returns The cross-entropy in nats, with its error estimate, convergence and method.
 *
 * @example A closed form, then quadrature
 * print('normal, normal:', crossEntropyAuto(Normal(0, 1), Normal(1, 2)))
 * print('normal, Laplace:', crossEntropyAuto(Normal(0, 1), Laplace(0, 1)))
 * print('by hand: log 2 + √(2/π) =', Math.log(2) + Math.sqrt(2 / Math.PI))
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
 * The Jensen–Shannon divergence
 * $\operatorname{JS}(p, q) = \frac{1}{2}\KL(p \,\Vert\, m) + \frac{1}{2}\KL(q \,\Vert\, m)$ with
 * $m = (p + q)/2$, in nats, by quadrature over the hull of the supports. Symmetric, finite (at most $\log 2$) even
 * when the supports differ, and 0 only when $p = q$.
 *
 * @param p One unbatched continuous univariate distribution.
 * @param q The other.
 * @returns The divergence, its error estimate and whether the quadrature converged.
 *
 * @example Disjoint supports reach the bound log 2
 * print('JS =', jensenShannonNumerical(Uniform(0, 1), Uniform(1, 2)).value, ' log 2 =', Math.log(2))
 * print('JS of a distribution with itself =', jensenShannonNumerical(Normal(0, 1), Normal(0, 1)).value)
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
 * The integrand of $\KL(p \,\Vert\, q)$ at $x$: $p(x) (\log p(x) - \log q(x))$, with 0 where $p(x) = 0$ and
 * $+\infty$ where $p(x) > 0 = q(x)$. Its integral is $\KL(p \,\Vert\, q)$; it is negative wherever $q(x) > p(x)$.
 * Elementwise, and differentiable where $p(x) > 0$.
 *
 * @param p The first distribution.
 * @param q The second distribution.
 * @param x The points: a number or a tensor (a grid, for plotting).
 * @returns The integrand at each point, with the shape of `x` broadcast against the batches.
 *
 * @example Positive where p dominates, negative where q does
 * // N(0, 1) against N(1, 1): the log-ratio is 1/2 − x.
 * print('integrand at -1, 0, 1, 2:', klIntegrand(Normal(0, 1), Normal(1, 1), tensor([-1, 0, 1, 2])))
 */
export function klIntegrand(p: Distribution, q: Distribution, x: Value): Value {
  const lp = p.logProb(x) as Value
  const w = exp(lp)
  const ok = mask([w], (v) => v !== 0)
  return outside(ok, mul(w, guard(sub(lp, q.logProb(x) as Value), ok, 0)), 0)
}

/**
 * A Monte Carlo estimate of $\KL(p \,\Vert\, q)$ with its standard error: the mean of $\log p(x_i) - \log q(x_i)$
 * over $n$ draws from $p$ (the same draws as `klMonteCarlo` from the same stream) and the sample standard deviation
 * over $\sqrt{n}$. For unbatched distributions: a batch is pooled into one mean. Throws `DomainError` for $n < 2$.
 *
 * @param s The random stream the draws come from.
 * @param p The distribution the draws come from.
 * @param q The second distribution.
 * @param n The number of draws, at least 2.
 * @returns The estimate, its standard error and the number of draws.
 *
 * @example The estimate is within a few standard errors of the closed form
 * print('estimate:', klMonteCarloWithError(stream(0), Normal(0, 1), Normal(1, 2), 10000))
 * print('closed form =', kl(Normal(0, 1), Normal(1, 2)))
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
