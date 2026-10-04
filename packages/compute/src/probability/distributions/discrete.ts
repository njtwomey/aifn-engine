/**
 * Discrete univariate families (mass functions on the integers). Log-masses are compositions of primitives,
 * differentiable in the continuous parameters (probabilities and rates); counts (n, r, population sizes) are treated as
 * constants. Cdfs use the regularised incomplete beta and gamma functions where a closed form exists (differentiable in
 * the probability or rate) and summation otherwise. Entropies without a closed form are exact sums over the support
 * (truncated where the remaining mass is below 1e-17) and are not differentiable.
 *
 * Conventions follow scipy.stats: Geometric counts trials up to and including the first success (k ≥ 1);
 * NegativeBinomial counts failures before the r-th success.
 *
 * Draws follow Devroye (1986), "Non-Uniform Random Variate Generation": inversion for the geometric (§X.2), the
 * gamma–Poisson mixture for the negative binomial (§X.4.5), sequential sampling without replacement for the
 * hypergeometric (§X.5); Bernoulli, binomial, Poisson and categorical draws are those of `aifn-compute/foundation/random` and
 * `aifn-compute/probability/samplers`. Rejection-style draws key each element by its own child stream.
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

const probability = (x: number) => x >= 0 && x <= 1
const count = (x: number) => isInteger(x) && x >= 0
const floorOf = (k: Value): Raw => rawMap([k], Math.floor)
const sizeOf = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/** A draw tensor (the samplers return a number when every parameter is a number and no shape is given). */
const drawn = (x: number | Tensor): Tensor => (typeof x === 'number' ? fromData(new Float64Array([x]), []) : x)

/** −Σₖ pₖ log pₖ over k = lo, lo + 1, …, hi for a scalar log-mass, stopping early once the tail is negligible. */
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
 * The Bernoulli distribution on {0, 1} with success probability p, or with log-odds `{ logits }` (stable for extreme
 * probabilities; `params` then holds the logits). An exponential family with η = logit p and T(k) = k.
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

/** log pmf of Binomial(n, p) at k, as a number. */
function binomialLogPmf(n: number, p: number, k: number): number {
  return (unwrap(logChoose(n, k)) as number) + xlogyScalar(k, p) + (n - k === 0 ? 0 : (n - k) * Math.log1p(-p))
}

const xlogyScalar = (x: number, y: number) => (x === 0 ? 0 : x * Math.log(y))

/**
 * The binomial distribution: successes in n ≥ 0 trials (an integer, treated as a constant) with success probability
 * p. The cdf is I_{1−p}(n − k, k + 1), differentiable in p. An exponential family for fixed n, with η = logit p.
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
 * The Poisson distribution with rate λ > 0. The cdf is Q(⌊k⌋ + 1, λ), differentiable in λ. An exponential family with
 * η = log λ and T(k) = k.
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
 * The geometric distribution: the number of trials up to and including the first success, k ≥ 1, with success
 * probability p ∈ (0, 1] (scipy's geom). An exponential family with η = log(1 − p) and T(k) = k.
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
 * The negative binomial distribution: the number of failures k ≥ 0 before the r-th success, for r > 0 (real r
 * allowed, treated as a constant) and success probability p ∈ (0, 1] (scipy's nbinom(r, p)). The cdf is
 * I_p(r, k + 1), differentiable in p. An exponential family for fixed r, with η = log(1 − p).
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
 * The hypergeometric distribution: successes in n draws without replacement from a population of N items of which K
 * are successes (all integers, constants). scipy's hypergeom(M = N, n = K, N = n). The cdf and entropy are sums over
 * the support max(0, n + K − N) … min(n, K).
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

/** The uniform distribution on the integers low, low + 1, …, high (inclusive; scipy's randint(low, high + 1)). */
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

/** One-hot rows [..., K] for integer indices (0 where the index is invalid). */
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
 * The categorical distribution on {0, …, K − 1} with probabilities `probs` (last axis K, normalised here; a batch of
 * shape [..., K] gives batch shape [...]) or with `{ logits }` (unnormalised log-probabilities, the stable form). The
 * mean and variance treat the categories as the numbers 0 … K − 1. An exponential family with η = log-probabilities
 * and T(k) = one-hot(k).
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
