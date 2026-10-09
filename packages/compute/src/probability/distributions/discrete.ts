/**
 * Discrete univariate families: mass functions on the integers. Log-masses are compositions of primitives,
 * differentiable in the continuous parameters (probabilities and rates); counts ($n$, $r$, population sizes) are
 * treated as constants. Cdfs use the regularised incomplete beta and gamma functions where a closed form exists
 * (differentiable in the probability or rate) and summation otherwise. Entropies without a closed form are sums over
 * the support (on an infinite support, cut off past the mean once a log-mass falls below $-45$) and are not
 * differentiable.
 *
 * Conventions follow scipy.stats: Geometric counts trials up to and including the first success ($k \ge 1$);
 * NegativeBinomial counts failures before the $r$-th success.
 *
 * Draws follow Devroye (1986), "Non-Uniform Random Variate Generation": inversion for the geometric (§X.2), the
 * gamma–Poisson mixture for the negative binomial (§X.4.5), sequential sampling without replacement for the
 * hypergeometric (§X.5); Bernoulli, binomial, Poisson and categorical draws are those of
 * `aifn-compute/foundation/random` and `aifn-compute/probability/samplers`. Rejection-style draws key each element by
 * its own child stream.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  bernoulli as bernoulliDraws,
  boundedIntegers,
  categorical as categoricalDraws,
  drawEach,
  units,
} from 'aifn-compute/foundation/random'
import {
  binomial as binomialDraws,
  gammaVariate as gammaDraws,
  poisson as poissonDraws,
} from 'aifn-compute/probability/samplers'
import {
  logChoose,
  logFactorial,
  logGamma,
  logSigmoid,
  logSoftmax,
  log1mexp,
  logit,
  regularisedBeta,
  regularisedGammaP,
  regularisedGammaQ,
  sigmoid,
  softplus,
} from 'aifn-compute/numerics/special'
import {
  add,
  div,
  exp,
  expm1,
  fromData,
  log,
  logsumexp,
  log1p,
  matmul,
  mul,
  neg,
  shapeOfValue,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  where,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Univariate } from './types'
import { atBatch, check, guard, isInteger, mask, outside, raw, rawMap, rawOnly, sumLast, univariate } from './util'
import { xlog1py, xlogy } from 'aifn-compute/numerics/special'

/**
 * Whether a value is a probability, in $[0, 1]$.
 *
 * @param x The value to test.
 */
const probability = (x: number) => x >= 0 && x <= 1
/**
 * Whether a value is a count: a non-negative integer.
 *
 * @param x The value to test.
 */
const count = (x: number) => isInteger(x) && x >= 0
/**
 * The elementwise floor $\lfloor k \rfloor$ of the raw values (a constant: no derivative flows through it).
 *
 * @param k The values (a number, a tensor or a traced value, read through its raw value).
 */
const floorOf = (k: Value): Raw => rawMap([k], Math.floor)
/**
 * The number of elements of a shape, the product of its sizes (1 for `[]`).
 *
 * @param shape The shape.
 */
const sizeOf = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/**
 * A draw tensor (the samplers return a number when every parameter is a number and no shape is given).
 *
 * @param x A sampler's draws: a number, or a tensor of them.
 * @returns `x` itself when it is a tensor, else a scalar (shape `[]`) tensor holding it.
 */
const drawn = (x: number | Tensor): Tensor => (typeof x === 'number' ? fromData(new Float64Array([x]), []) : x)

/**
 * The entropy $-\sum_k p_k \log p_k$ over $k = \mathit{lo}, \mathit{lo} + 1, \dots, \mathit{hi}$ for a scalar
 * log-mass, stopping early once the tail is negligible. Terms of zero mass contribute nothing.
 *
 * @param logPmf The log-mass $\log p_k$ at an integer $k$, as a number.
 * @param lo The first $k$ of the sum.
 * @param hi The last $k$ of the sum (inclusive).
 * @param tailFrom Past this $k$ the sum stops at the first term whose log-mass is below $-45$; the mean of the
 *   distribution, or `Infinity` (the default) to sum every term up to `hi`.
 * @returns The entropy in nats.
 */
function entropyBySum(logPmf: (k: number) => number, lo: number, hi: number, tailFrom = Infinity): number {
  let h = 0
  for (let k = lo; k <= hi; k++) {
    const lp = logPmf(k)
    if (lp > -Infinity) h -= Math.exp(lp) * lp
    // Beyond the bulk, stop when the terms are too small to change the sum.
    if (k > tailFrom && lp < -45) break
  }
  return h
}

// ── Bernoulli ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Bernoulli distribution on $\{0, 1\}$ with success probability $p$, $\pr(k) = p^k (1 - p)^{1 - k}$, or with
 * log-odds `{ logits }` (stable for extreme probabilities; `params` then holds the logits). The mean is $p$ and the
 * variance $p(1 - p)$; the mode is 1 when $p > 1/2$, else 0. An exponential family with $\eta = \operatorname{logit} p$
 * and $T(k) = k$. Throws `DomainError` when a probability is outside $[0, 1]$.
 *
 * @param p The success probability $p \in [0, 1]$ (a number, a tensor for a batch, or a traced value), or
 *   `{ logits }`, the log-odds $\log(p / (1 - p))$ (any real), which keeps $\log p$ and $\log(1 - p)$ accurate when
 *   $p$ is within rounding of 0 or 1.
 * @returns The distribution, with batch shape that of `p`.
 *
 * @example Mass, mean and variance
 * const d = Bernoulli(0.3)
 * print('P(1) =', d.prob(1), ' P(0) =', d.prob(0))
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example Logits keep the log-mass of an almost-certain event
 * // sigmoid(40) rounds to 1, so the probability form loses log P(0); the logit form keeps it.
 * print('from p:', Bernoulli(1 / (1 + Math.exp(-40))).logProb(0))
 * print('from logits:', Bernoulli({ logits: 40 }).logProb(0))
 */
export function Bernoulli<P extends Value>(p: P | { logits: P }): Univariate<P> {
  const fromLogits = typeof p === 'object' && p !== null && 'logits' in p
  const logits: Value = fromLogits ? (p as { logits: P }).logits : logit(p as P)
  const probs: Value = fromLogits ? sigmoid(logits) : (p as P)
  if (!fromLogits) check('Bernoulli', 'p', probs, probability, 'in [0, 1]')
  const logP = fromLogits ? logSigmoid(logits) : log(probs)
  const logQ = fromLogits ? logSigmoid(neg(logits)) : log1p(neg(probs))
  return univariate({
    name: 'Bernoulli',
    params: fromLogits ? { logits } : { probs },
    support: { type: 'integers', lower: 0, upper: 1 },
    discrete: true,
    logProb: (k) => {
      const ok = mask([k], (v) => v === 0 || v === 1)
      return outside(
        ok,
        where(
          mask([k], (v) => v === 1),
          logP,
          logQ,
        ),
        -Infinity,
      )
    },
    cdf: (k) =>
      where(
        mask([k], (v) => v < 0),
        0,
        where(
          mask([k], (v) => v < 1),
          sub(1, probs),
          1,
        ),
      ),
    quantile: (q) => rawOnly('Bernoulli.quantile', [q, probs], (u, s) => (probability(u) ? (u <= 1 - s ? 0 : 1) : NaN)),
    sample: (s, shape) => drawn(bernoulliDraws(s, raw(unwrap(probs), 'Bernoulli'), { shape })),
    mean: () => probs,
    variance: () => mul(probs, sub(1, probs)),
    entropy: () => neg(add(mul(probs, logP), mul(sub(1, probs), logQ))),
    mode: () => rawMap([probs], (s) => (s > 0.5 ? 1 : 0)),
    expFamily: {
      naturalParams: () => [logits],
      sufficientStats: (k) => [k],
      logPartition: () => softplus(logits),
      logBaseMeasure: () => 0,
    },
  })
}

// ── Binomial ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The log-mass of $\Binom(n, p)$ at $k$, as a number: $\log\binom{n}{k} + k \log p + (n - k)\log(1 - p)$, with
 * $0 \log 0 = 0$ so the ends of the support are finite at $p \in \{0, 1\}$.
 *
 * @param n The number of trials.
 * @param p The success probability.
 * @param k The number of successes, an integer in $0, \dots, n$ (not checked).
 * @returns $\log \pr(K = k)$.
 */
function binomialLogPmf(n: number, p: number, k: number): number {
  return (unwrap(logChoose(n, k)) as number) + xlogyScalar(k, p) + (n - k === 0 ? 0 : (n - k) * Math.log1p(-p))
}

/**
 * $x \log y$ for numbers, taken as 0 when $x = 0$ (even when $y = 0$).
 *
 * @param x The factor $x$.
 * @param y The argument $y$ of the logarithm.
 */
const xlogyScalar = (x: number, y: number) => (x === 0 ? 0 : x * Math.log(y))

/**
 * The binomial distribution: the number of successes in $n \ge 0$ trials (an integer, treated as a constant) with
 * success probability $p$, $\pr(k) = \binom{n}{k} p^k (1 - p)^{n - k}$ for $k = 0, \dots, n$. The mean is $np$ and
 * the variance $np(1 - p)$. The cdf is $I_{1 - p}(n - k, k + 1)$, differentiable in $p$; the entropy is a sum over the
 * support. An exponential family for fixed $n$, with $\eta = \operatorname{logit} p$ and $T(k) = k$. Throws
 * `DomainError` when $n$ is not a non-negative integer or $p$ is outside $[0, 1]$.
 *
 * @param n The number of trials $n$, a non-negative integer.
 * @param p The success probability of each trial, in $[0, 1]$.
 * @returns The distribution, with batch shape the broadcast of `n` and `p`.
 *
 * @example Four fair coins
 * const d = Binomial(4, 0.5)
 * print('P(2 heads) =', d.prob(2))
 * print('P(at most 2) =', d.cdf(2))
 * print('mean =', d.mean(), ' variance =', d.variance())
 */
export function Binomial<N extends Value, P extends Value>(n: N, p: P): Univariate<N | P> {
  check('Binomial', 'n', n, count, 'a non-negative integer')
  check('Binomial', 'p', p, probability, 'in [0, 1]')
  const valid = (k: Value) => mask([k, n], (v, m) => isInteger(v) && v >= 0 && v <= m)
  const middle = (k: Value) => mask([k, n], (v, m) => v >= 0 && v < m)
  return univariate({
    name: 'Binomial',
    params: { n, p },
    support: { type: 'integers', lower: 0, upper: n },
    discrete: true,
    logProb: (k) => {
      const ok = valid(k)
      const ks = guard(k, ok, 0)
      return outside(ok, add(add(logChoose(n, ks), xlogy(ks, p)), xlog1py(sub(n, ks), neg(p))), -Infinity)
    },
    cdf: (k) => {
      const mid = middle(k)
      const kf = where(mid, floorOf(k), 0)
      const inner = regularisedBeta(sub(n, kf), add(kf, 1), sub(1, p))
      return where(
        mask([k], (v) => v < 0),
        0,
        outside(mid, inner, 1),
      )
    },
    survival: (k) => {
      const mid = middle(k)
      const kf = where(mid, floorOf(k), 0)
      const inner = regularisedBeta(add(kf, 1), sub(n, kf), p)
      return where(
        mask([k], (v) => v < 0),
        1,
        outside(mid, inner, 0),
      )
    },
    sample: (s, shape) => drawn(binomialDraws(s, raw(n, 'Binomial'), raw(p, 'Binomial'), { shape })),
    mean: () => mul(n, p),
    variance: () => mul(mul(n, p), sub(1, p)),
    entropy: () => rawOnly('Binomial.entropy', [n, p], (m, q) => entropyBySum((k) => binomialLogPmf(m, q, k), 0, m)),
    mode: () => rawMap([n, p], (m, q) => Math.min(m, Math.floor((m + 1) * q))),
    expFamily: {
      naturalParams: () => [logit(p)],
      sufficientStats: (k) => [k],
      logPartition: () => mul(n, softplus(logit(p))),
      logBaseMeasure: (k) => logChoose(n, k),
    },
  })
}

// ── Poisson ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Poisson distribution with rate $\lambda > 0$, $\pr(k) = \lambda^k e^{-\lambda} / k!$ for $k = 0, 1, \dots$
 * The mean and variance are both $\lambda$, and the mode is $\lfloor \lambda \rfloor$. The cdf is
 * $Q(\lfloor k \rfloor + 1, \lambda)$ (the regularised upper incomplete gamma function), differentiable in
 * $\lambda$. An exponential family with $\eta = \log \lambda$ and $T(k) = k$. Throws `DomainError` when a rate is
 * not positive.
 *
 * @param rate The rate $\lambda > 0$, the expected count.
 * @returns The distribution, with batch shape that of `rate`.
 *
 * @example Mass and moments at rate 2
 * const d = Poisson(2)
 * print('P(0) =', d.prob(0), ' e^-2 =', Math.exp(-2))
 * print('P(2) =', d.prob(2))
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example Seeded draws
 * print('draws =', Poisson(2).sample(stream(1), { shape: [8] }))
 */
export function Poisson<L extends Value>(rate: L): Univariate<L> {
  check('Poisson', 'rate', rate, (x) => x > 0, 'positive')
  const valid = (k: Value) => mask([k], count)
  const nonNegative = (k: Value) => mask([k], (v) => v >= 0)
  return univariate({
    name: 'Poisson',
    params: { rate },
    support: { type: 'integers', lower: 0, upper: Infinity },
    discrete: true,
    logProb: (k) => {
      const ok = valid(k)
      const ks = guard(k, ok, 0)
      return outside(ok, sub(sub(xlogy(ks, rate), rate), logFactorial(ks)), -Infinity)
    },
    cdf: (k) => {
      const ok = nonNegative(k)
      return outside(ok, regularisedGammaQ(add(where(ok, floorOf(k), 0), 1), rate), 0)
    },
    survival: (k) => {
      const ok = nonNegative(k)
      return outside(ok, regularisedGammaP(add(where(ok, floorOf(k), 0), 1), rate), 1)
    },
    sample: (s, shape) => drawn(poissonDraws(s, raw(rate, 'Poisson'), { shape })),
    mean: () => rate,
    variance: () => rate,
    entropy: () =>
      rawOnly('Poisson.entropy', [rate], (l) =>
        entropyBySum(
          (k) => xlogyScalar(k, l) - l - (unwrap(logFactorial(k)) as number),
          0,
          Math.ceil(l + 40 * Math.sqrt(l) + 60),
          l,
        ),
      ),
    mode: () => rawMap([rate], Math.floor),
    expFamily: {
      naturalParams: () => [log(rate)],
      sufficientStats: (k) => [k],
      logPartition: () => rate,
      logBaseMeasure: (k) => neg(logFactorial(k)),
    },
  })
}

// ── Geometric ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The geometric distribution: the number of trials up to and including the first success, $k \ge 1$, with success
 * probability $p \in (0, 1]$ (scipy's geom): $\pr(k) = (1 - p)^{k - 1} p$. The mean is $1/p$, the variance
 * $(1 - p)/p^2$ and the cdf $1 - (1 - p)^{\lfloor k \rfloor}$. Draws are by inversion. An exponential family with
 * $\eta = \log(1 - p)$ and $T(k) = k$. Throws `DomainError` when $p$ is outside $(0, 1]$.
 *
 * @param p The success probability of each trial, in $(0, 1]$.
 * @returns The distribution, with batch shape that of `p`.
 *
 * @example Tosses of a fair coin until the first head
 * const d = Geometric(0.5)
 * print('P(1) =', d.prob(1), ' P(3) =', d.prob(3))
 * print('P(at most 3) =', d.cdf(3))
 * print('mean =', d.mean(), ' variance =', d.variance())
 */
export function Geometric<P extends Value>(p: P): Univariate<P> {
  check('Geometric', 'p', p, (x) => x > 0 && x <= 1, 'in (0, 1]')
  const valid = (k: Value) => mask([k], (v) => isInteger(v) && v >= 1)
  const atLeastOne = (k: Value) => mask([k], (v) => v >= 1)
  const logQ = log1p(neg(p))
  return univariate({
    name: 'Geometric',
    params: { p },
    support: { type: 'integers', lower: 1, upper: Infinity },
    discrete: true,
    logProb: (k) => {
      const ok = valid(k)
      return outside(ok, add(log(p), xlog1py(sub(guard(k, ok, 1), 1), neg(p))), -Infinity)
    },
    cdf: (k) => {
      const ok = atLeastOne(k)
      return outside(ok, neg(expm1(mul(where(ok, floorOf(k), 1), logQ))), 0)
    },
    survival: (k) => {
      const ok = atLeastOne(k)
      return outside(ok, exp(mul(where(ok, floorOf(k), 1), logQ)), 1)
    },
    sample: (s, shape) =>
      // Inversion: the smallest k with 1 − (1 − p)^k ≥ u is ⌈log(1 − u) / log(1 − p)⌉.
      rawMap([fromData(units(s, sizeOf(shape)), shape), raw(p, 'Geometric')], (u, q) =>
        q === 1 ? 1 : Math.max(1, Math.ceil(Math.log1p(-u) / Math.log1p(-q))),
      ) as Tensor,
    mean: () => div(1, p),
    variance: () => div(sub(1, p), square(p)),
    entropy: () => div(neg(add(xlogy(sub(1, p), sub(1, p)), xlogy(p, p))), p),
    mode: () => rawMap([p], () => 1),
    expFamily: {
      naturalParams: () => [logQ],
      sufficientStats: (k) => [k],
      logPartition: () => sub(logQ, log1mexp(logQ)),
      logBaseMeasure: () => 0,
    },
  })
}

// ── Negative binomial ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The negative binomial distribution: the number of failures $k \ge 0$ before the $r$-th success, for $r > 0$ (real
 * $r$ allowed, treated as a constant) and success probability $p \in (0, 1]$ (scipy's nbinom(r, p)):
 * $\pr(k) = \frac{\Gamma(k + r)}{\Gamma(r)\, k!} p^r (1 - p)^k$. The mean is $r(1 - p)/p$ and the variance
 * $r(1 - p)/p^2$. The cdf is $I_p(r, k + 1)$, differentiable in $p$; the entropy is a truncated sum. Draws are a
 * gamma–Poisson mixture. An exponential family for fixed $r$, with $\eta = \log(1 - p)$ and $T(k) = k$. Throws
 * `DomainError` when $r$ is not positive or $p$ is outside $(0, 1]$.
 *
 * @param r The number of successes $r > 0$ to wait for; a real $r$ gives the gamma–Poisson (overdispersed count)
 *   form.
 * @param p The success probability of each trial, in $(0, 1]$.
 * @returns The distribution, with batch shape the broadcast of `r` and `p`.
 *
 * @example Failures before the second head of a fair coin
 * const d = NegativeBinomial(2, 0.5)
 * print('P(0) =', d.prob(0), ' P(1) =', d.prob(1), ' P(2) =', d.prob(2))
 * print('mean =', d.mean(), ' variance =', d.variance())
 */
export function NegativeBinomial<R extends Value, P extends Value>(r: R, p: P): Univariate<R | P> {
  check('NegativeBinomial', 'r', r, (x) => x > 0, 'positive')
  check('NegativeBinomial', 'p', p, (x) => x > 0 && x <= 1, 'in (0, 1]')
  const valid = (k: Value) => mask([k], count)
  const nonNegative = (k: Value) => mask([k], (v) => v >= 0)
  const logPmf = (k: Value) =>
    add(sub(sub(logGamma(add(k, r)), logGamma(r)), logFactorial(k)), add(mul(r, log(p)), xlog1py(k, neg(p))))
  return univariate({
    name: 'NegativeBinomial',
    params: { r, p },
    support: { type: 'integers', lower: 0, upper: Infinity },
    discrete: true,
    logProb: (k) => {
      const ok = valid(k)
      return outside(ok, logPmf(guard(k, ok, 0)), -Infinity)
    },
    cdf: (k) => {
      const ok = nonNegative(k)
      return outside(ok, regularisedBeta(r, add(where(ok, floorOf(k), 0), 1), p), 0)
    },
    survival: (k) => {
      const ok = nonNegative(k)
      return outside(ok, regularisedBeta(add(where(ok, floorOf(k), 0), 1), r, sub(1, p)), 1)
    },
    sample: (s, shape) => {
      // A gamma–Poisson mixture: λ ~ Gamma(r, scale (1 − p)/p), k ~ Poisson(λ).
      const rr = raw(r, 'NegativeBinomial')
      const pp = raw(p, 'NegativeBinomial')
      return drawEach('NegativeBinomial', [rr, pp], { shape }, s, (e, a, q) => {
        if (q === 1) return 0
        const lambda = (gammaDraws(e, a) as number) * ((1 - q) / q)
        return lambda === 0 ? 0 : (poissonDraws(e, lambda) as number)
      }) as Tensor
    },
    mean: () => div(mul(r, sub(1, p)), p),
    variance: () => div(mul(r, sub(1, p)), square(p)),
    entropy: () =>
      rawOnly('NegativeBinomial.entropy', [r, p], (a, q) => {
        const m = (a * (1 - q)) / q
        const sd = Math.sqrt(a * (1 - q)) / q
        const lp = (k: number) => unwrap(logPmf(k)) as number
        return entropyBySum(lp, 0, Math.ceil(m + 60 * sd + 100), m)
      }),
    mode: () => rawMap([r, p], (a, q) => (a > 1 ? Math.floor(((a - 1) * (1 - q)) / q) : 0)),
    expFamily: {
      naturalParams: () => [log1p(neg(p))],
      sufficientStats: (k) => [k],
      logPartition: () => neg(mul(r, log(p))),
      logBaseMeasure: (k) => sub(sub(logGamma(add(k, r)), logGamma(r)), logFactorial(k)),
    },
  })
}

// Gamma(a, 1) draws through the random module's sampler (one number).

// ── Hypergeometric ───────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The hypergeometric distribution: the number of successes in $n$ draws without replacement from a population of $N$
 * items of which $K$ are successes (all integers, constants), scipy's hypergeom($M = N$, $n = K$, $N = n$):
 * $\pr(k) = \binom{K}{k}\binom{N - K}{n - k} / \binom{N}{n}$. The mean is $nK/N$ and the variance
 * $n \frac{K}{N}\left(1 - \frac{K}{N}\right)\frac{N - n}{N - 1}$. The cdf and entropy are sums over the support
 * $\max(0, n + K - N), \dots, \min(n, K)$ and are not differentiable; draws are sequential, one uniform per draw.
 * Throws `DomainError` when a parameter is not a non-negative integer.
 *
 * @param population The population size $N$.
 * @param successes The number $K$ of success items in the population.
 * @param draws The number $n$ of items drawn.
 * @returns The distribution, with batch shape the broadcast of the three parameters.
 *
 * @example Three cards from ten, four of them red
 * const d = Hypergeometric(10, 4, 3)
 * print('P(0 red) =', d.prob(0), ' P(1 red) =', d.prob(1))
 * print('mean =', d.mean(), ' variance =', d.variance())
 */
export function Hypergeometric<A extends Value, B extends Value, C extends Value>(
  population: A,
  successes: B,
  draws: C,
): Univariate<A | B | C> {
  check('Hypergeometric', 'population', population, count, 'a non-negative integer')
  check('Hypergeometric', 'successes', successes, count, 'a non-negative integer')
  check('Hypergeometric', 'draws', draws, count, 'a non-negative integer')
  const N = population
  const K = successes
  const n = draws
  const lower = rawMap([N, K, n], (a, b, c) => Math.max(0, c + b - a))
  const upper = rawMap([N, K, n], (_a, b, c) => Math.min(c, b))
  const logPmf = (k: Value) => sub(add(logChoose(K, k), logChoose(sub(N, K), sub(n, k))), logChoose(N, n))
  const scalarLogPmf = (a: number, b: number, c: number, k: number) =>
    (unwrap(logChoose(b, k)) as number) +
    (unwrap(logChoose(a - b, c - k)) as number) -
    (unwrap(logChoose(a, c)) as number)
  return univariate({
    name: 'Hypergeometric',
    params: { population, successes, draws },
    support: { type: 'integers', lower, upper },
    discrete: true,
    logProb: (k) => {
      const ok = mask([k, lower, upper], (v, lo, hi) => isInteger(v) && v >= lo && v <= hi)
      return outside(ok, logPmf(guard(k, ok, unwrap(lower) as number)), -Infinity)
    },
    cdf: (k) =>
      rawOnly('Hypergeometric.cdf', [k, N, K, n], (v, a, b, c) => {
        const lo = Math.max(0, c + b - a)
        let total = 0
        for (let j = lo; j <= Math.min(Math.floor(v), c, b); j++) total += Math.exp(scalarLogPmf(a, b, c, j))
        return Math.min(1, total)
      }),
    sample: (s, shape) =>
      // Sequential draws without replacement, one uniform per draw from the element's own key.
      drawEach(
        'Hypergeometric',
        [N, K, n].map((v) => raw(v, 'Hypergeometric')),
        { shape },
        s,
        (e, a, b, c) => {
          const u = units(e, c)
          let left = a
          let good = b
          let hits = 0
          for (let i = 0; i < c; i++) {
            if (u[i] * left < good) {
              hits++
              good--
            }
            left--
          }
          return hits
        },
      ) as Tensor,
    mean: () => div(mul(n, K), N),
    variance: () => {
      const f = div(K, N)
      return div(mul(mul(mul(n, f), sub(1, f)), sub(N, n)), sub(N, 1))
    },
    entropy: () =>
      rawOnly('Hypergeometric.entropy', [N, K, n], (a, b, c) =>
        entropyBySum((k) => scalarLogPmf(a, b, c, k), Math.max(0, c + b - a), Math.min(c, b)),
      ),
    mode: () => rawMap([N, K, n], (a, b, c) => Math.floor(((c + 1) * (b + 1)) / (a + 2))),
  })
}

// ── Discrete uniform ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The uniform distribution on the integers $a, a + 1, \dots, b$ (inclusive; scipy's randint($a$, $b + 1$)):
 * $\pr(k) = 1/(b - a + 1)$. The mean is $(a + b)/2$, the variance $((b - a + 1)^2 - 1)/12$ and the entropy
 * $\log(b - a + 1)$; the mode is NaN, as every point is one. Throws `DomainError` when a bound is not an integer or
 * $b < a$.
 *
 * @param low The smallest value $a$, an integer.
 * @param high The largest value $b$, an integer, at least `low`.
 * @returns The distribution, with batch shape the broadcast of `low` and `high`.
 *
 * @example A fair die
 * const d = DiscreteUniform(1, 6)
 * print('P(3) =', d.prob(3), ' P(at most 2) =', d.cdf(2))
 * print('mean =', d.mean(), ' variance =', d.variance())
 */
export function DiscreteUniform<A extends Value, B extends Value>(low: A, high: B): Univariate<A | B> {
  check('DiscreteUniform', 'low', low, isInteger, 'an integer')
  check('DiscreteUniform', 'high', high, isInteger, 'an integer')
  check('DiscreteUniform', 'high − low', sub(unwrap(high), unwrap(low)), (x) => x >= 0, 'non-negative')
  const size = add(sub(high, low), 1)
  return univariate({
    name: 'DiscreteUniform',
    params: { low, high },
    support: { type: 'integers', lower: low, upper: high },
    discrete: true,
    logProb: (k) => {
      const ok = mask([k, low, high], (v, a, b) => isInteger(v) && v >= a && v <= b)
      return outside(ok, atBatch(neg(log(size)), k), -Infinity)
    },
    cdf: (k) => {
      const c = div(add(sub(floorOf(k), low), 1), size)
      return where(
        mask([c], (v) => v < 0),
        0,
        where(
          mask([c], (v) => v > 1),
          1,
          c,
        ),
      )
    },
    sample: (s, shape) => {
      const lows = rawMap([fromData(new Float64Array(sizeOf(shape)), shape), raw(low, 'DiscreteUniform')], (_, a) => a)
      const widths = rawMap(
        [fromData(new Float64Array(sizeOf(shape)), shape), raw(low, 'DiscreteUniform'), raw(high, 'DiscreteUniform')],
        (_, a, b) => b - a + 1,
      )
      const offsets = boundedIntegers(s, toFlat(widths as Tensor), 'DiscreteUniform')
      return rawMap([lows, fromData(offsets, shape)], (a, k) => a + k) as Tensor
    },
    mean: () => mul(0.5, add(low, high)),
    variance: () => div(sub(square(size), 1), 12),
    entropy: () => log(size),
    mode: () => rawMap([low, high], () => NaN),
  })
}

// ── Categorical ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * One-hot rows for integer indices: a row of zeros with a 1 at the index (all zeros where the index is not an integer
 * in $0, \dots, K - 1$).
 *
 * @param k The indices (a number or a tensor; read through a traced value).
 * @param K The number of categories, the length of each row.
 * @returns A tensor of shape `[...shape of k, K]`.
 */
function oneHot(k: Value, K: number): Tensor {
  const r = unwrap(k)
  const values = typeof r === 'number' ? [r] : toFlat(r)
  const shape = typeof r === 'number' ? [] : r.shape
  const out = new Float64Array(values.length * K)
  values.forEach((v, i) => {
    if (isInteger(v) && v >= 0 && v < K) out[i * K + v] = 1
  })
  return fromData(out, [...shape, K])
}

/**
 * The categorical distribution on $\{0, \dots, K - 1\}$ with probabilities $\pr(k) = p_k / \sum_j p_j$ (the
 * weights are normalised here), or with `{ logits }`, $\pr(k) = e^{\ell_k} / \sum_j e^{\ell_j}$ (the stable form).
 * The mean and variance treat the categories as the numbers $0, \dots, K - 1$. An exponential family with
 * $\eta$ the log-probabilities and $T(k)$ the one-hot vector of $k$. Throws `ShapeError` for a scalar and
 * `DomainError` for a negative weight.
 *
 * @param p The weights $p_k \ge 0$ along the last axis (length $K$; a batch of shape `[..., K]` gives batch shape
 *   `[...]`), or `{ logits }`, unnormalised log-probabilities of the same shape.
 * @returns The distribution over category indices.
 *
 * @example Weights normalised to probabilities
 * const d = Categorical(tensor([1, 2, 1]))
 * print('P(1) =', d.prob(1), ' P(2) =', d.prob(2))
 * print('mean =', d.mean(), ' variance =', d.variance())
 * print('entropy =', d.entropy(), ' 1.5 log 2 =', 1.5 * Math.log(2))
 *
 * @example Seeded draws from logits
 * const d = Categorical({ logits: tensor([0, 0, Math.log(2)]) })
 * print('draws =', d.sample(stream(3), { shape: [10] }))
 */
export function Categorical<P extends Value>(p: P | { logits: P }): Univariate<P> {
  const fromLogits = typeof p === 'object' && p !== null && 'logits' in p
  const given: Value = fromLogits ? (p as { logits: P }).logits : (p as P)
  const shape = shapeOfValue(given)
  if (shape.length === 0) throw new ShapeError('Categorical', 'Categorical: needs a vector (or batch) of probabilities')
  if (!fromLogits) check('Categorical', 'probs', given, (x) => x >= 0, 'non-negative')
  const K = shape[shape.length - 1]
  const batchShape = shape.slice(0, -1)
  const logP = fromLogits ? logSoftmax(given as Tensor) : log(div(given, sum(given, -1, true)))
  const probs = exp(logP)
  const indices = fromData(
    Float64Array.from({ length: K }, (_, k) => k),
    [K],
  )
  const upper = fromData(
    Float64Array.from({ length: K * K }, (_, q) => (Math.floor(q / K) <= q % K ? 1 : 0)),
    [K, K],
  )
  const mean = sumLast(mul(probs, indices))
  return univariate({
    name: 'Categorical',
    params: fromLogits ? { logits: given } : { probs: given },
    batchShape,
    support: { type: 'integers', lower: 0, upper: K - 1 },
    discrete: true,
    logProb: (k) => {
      const ok = mask([k], (v) => isInteger(v) && v >= 0 && v < K)
      // Select log pₖ rather than multiply by the one-hot vector, since 0 · log 0 would give NaN for a class of
      // probability 0 other than k.
      return outside(ok, sumLast(where(oneHot(k, K), logP, 0)), -Infinity)
    },
    cdf: (k) => {
      const cumulative = matmul(probs, upper)
      const kc = rawMap([k], (v) => Math.min(K - 1, Math.max(0, Math.floor(v))))
      const inner = sumLast(mul(oneHot(kc, K), cumulative))
      return where(
        mask([k], (v) => v < 0),
        0,
        where(
          mask([k], (v) => v >= K - 1),
          1,
          inner,
        ),
      )
    },
    sample: (s, shape_) => {
      const weights = unwrap(probs) as Tensor
      const t = categoricalDraws(s, weights, { shape: shape_ })
      return typeof t === 'number' ? fromData(new Int32Array([t]), []) : t
    },
    mean: () => mean,
    variance: () => sub(sumLast(mul(probs, square(indices))), square(mean)),
    entropy: () => neg(sumLast(xlogy(probs, probs))),
    mode: () => {
      const flat = toFlat(unwrap(probs) as Tensor)
      const rows = flat.length / K
      const out = new Float64Array(rows)
      for (let r = 0; r < rows; r++) {
        let best = 0
        for (let j = 1; j < K; j++) if (flat[r * K + j] > flat[r * K + best]) best = j
        out[r] = best
      }
      return batchShape.length === 0 ? out[0] : fromData(out, batchShape)
    },
    expFamily: {
      naturalParams: () => [fromLogits ? given : logP],
      sufficientStats: (k) => [oneHot(k, K)],
      logPartition: () => (fromLogits ? logsumexp(given, -1) : mul(0, sumLast(probs))),
      logBaseMeasure: () => 0,
    },
  })
}
