/**
 * `aifn-compute/probability/samplers`: draws from named families that need special functions or factorisations (decision D2):
 * gamma, log-gamma, beta, chi-square, Student t, Dirichlet, Poisson, binomial, multinomial and the multivariate
 * normal. Each takes the stream first, like the draws of `aifn-compute/foundation/random`, and is written on its blocks and
 * child keys (design K §6):
 *
 * - samplers with rejection loops (gamma and its relatives, Poisson, binomial, Student t) give element k the child key
 *   `child(s, '~', position + k)` and advance the stream by one word per element (`drawEach`), so the variable number
 *   of trials of one element never moves another element's draws;
 * - Dirichlet and multinomial rows key each component the same way (one word per component);
 * - the multivariate normal draws one block of d standard normals per row (`standardNormals`).
 *
 * None is differentiable: pathwise (reparameterised) draws are the `rsample` methods of `aifn-compute/probability/distributions`.
 */

import { cholesky, type CholeskyOptions } from 'aifn-compute/numerics/linalg'
import { logGamma, sigmoid } from 'aifn-compute/numerics/special'
import { broadcastShapes, broadcastTo, fromData, showShape, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Raw as Param, SampleOptions } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  batchIndex,
  checkBroadcast,
  child,
  drawEach,
  eventRows,
  standardNormals,
  units,
  type Drawn,
  type Spread,
  type Stream,
} from 'aifn-compute/foundation/random'

/** One uniform in [0, 1) from a stream (2 words). */
const unit = (s: Stream): number => units(s, 1)[0]

/** One standard normal from a stream (4 words; the second Box–Muller output is discarded). */
const standardNormal = (s: Stream): number => standardNormals(s, 1)[0]

/**
 * Draw `count` values, value k from its own child stream `child(s, '~', position + k)`, advancing `s` by one word per
 * value: the per-element keying of `drawEach`, for samplers that fill rows of an event (Dirichlet, multinomial).
 */
function eachChild(s: Stream, count: number): (k: number) => Stream {
  const start = s.position
  s.position += count
  return (k) => child(s, '~', start + k)
}

/**
 * log of a Gamma(shape, 1) draw. Marsaglia and Tsang (2000), "A simple method for generating gamma variables", ACM TOMS
 * 26(3); for shape < 1 the boost G(a) = G(a + 1) · U^{1/a} (their §6), kept in log space so that tiny shapes do not
 * underflow to 0.
 */
function logGammaDraw(s: Stream, shape: number): number {
  if (!(shape > 0)) return NaN
  if (shape < 1) return logGammaDraw(s, shape + 1) + Math.log(1 - unit(s)) / shape
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    const x = standardNormal(s)
    const t = 1 + c * x
    if (t <= 0) continue
    const v = t * t * t
    const u = unit(s)
    // The squeeze accepts about 98% of proposals without a logarithm.
    if (u < 1 - 0.0331 * x * x * x * x) return Math.log(d * v)
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return Math.log(d * v)
  }
}

/** A Beta(a, b) draw G_a/(G_a + G_b) = σ(log G_a − log G_b), from two log-gamma draws (no 0/0 for small a or b). */
function betaDraw(s: Stream, a: number, b: number): number {
  const la = logGammaDraw(s, a)
  const lb = logGammaDraw(s, b)
  return sigmoid(la - lb)
}

/** A Poisson(λ) draw: inversion for λ < 10, Hörmann's PTRS for λ ≥ 10 (see `poisson`). */
function poissonDraw(s: Stream, lambda: number): number {
  if (!(lambda >= 0)) return NaN
  if (lambda === 0) return 0
  if (lambda < 10) {
    const u = unit(s)
    let k = 0
    let p = Math.exp(-lambda)
    let cdf = p
    while (u >= cdf && k < 1000) {
      k++
      p *= lambda / k
      cdf += p
    }
    return k
  }
  const slam = Math.sqrt(lambda)
  const logLam = Math.log(lambda)
  const b = 0.931 + 2.53 * slam
  const a = -0.059 + 0.02483 * b
  const invAlpha = 1.1239 + 1.1328 / (b - 3.4)
  const vr = 0.9277 - 3.6224 / (b - 2)
  for (;;) {
    const u = unit(s) - 0.5
    const v = unit(s)
    const us = 0.5 - Math.abs(u)
    const k = Math.floor(((2 * a) / us + b) * u + lambda + 0.43)
    if (us >= 0.07 && v <= vr) return k
    if (k < 0 || (us < 0.013 && v > us)) continue
    if (Math.log(v) + Math.log(invAlpha) - Math.log(a / (us * us) + b) <= -lambda + k * logLam - logGamma(k + 1))
      return k
  }
}

/** A Binomial(n, p) draw: BINV inversion for small n·min(p, 1 − p), else the order-statistic recursion. */
function binomialDraw(s: Stream, n: number, p: number): number {
  if (!(Number.isInteger(n) && n >= 0 && p >= 0 && p <= 1)) return NaN
  if (n === 0 || p === 0) return 0
  if (p === 1) return n
  if (p > 0.5) return n - binomialDraw(s, n, 1 - p)
  if (n * p < 30) {
    const q = 1 - p
    const ratio = p / q
    for (;;) {
      let u = unit(s)
      let r = Math.pow(q, n)
      let k = 0
      while (u > r && k <= n) {
        u -= r
        k++
        r *= ((n - k + 1) / k) * ratio
      }
      // Rounding can leave u above the whole mass; redraw rather than return an impossible k.
      if (k <= n) return k
    }
  }
  const a = 1 + Math.floor(n / 2)
  const b = n + 1 - a
  const x = betaDraw(s, a, b)
  if (x >= p) return binomialDraw(s, a - 1, p / x)
  return a + binomialDraw(s, b - 1, (p - x) / (1 - x))
}

/** A chi-square draw: 2 · Gamma(k/2, 1). */
function chiSquareDraw(s: Stream, df: number): number {
  return 2 * Math.exp(logGammaDraw(s, df / 2))
}

/**
 * log of Gamma(shape, 1) draws, elementwise (NaN for shape ≤ 0). Marsaglia and Tsang (2000), "A simple method for
 * generating gamma variables", ACM TOMS 26(3); for shape < 1 the boost G(a) = G(a + 1) · U^{1/a} (their §6), kept in
 * log space so that tiny shapes do not underflow to 0.
 */
export function logGammaVariate<A extends Param, O extends SampleOptions = object>(
  s: Stream,
  shape: A,
  options?: O,
): Drawn<[A], O> {
  return drawEach('logGammaVariate', [shape], options, s, (e, a) => logGammaDraw(e, a)) as Drawn<[A], O>
}

/**
 * Gamma(shape, scale) draws (mean shape · scale), Marsaglia–Tsang, including shape < 1, elementwise over broadcast
 * parameters. For very small shapes a value can underflow to 0; use `logGammaVariate` when that matters. (Named
 * `…Variate` beside `logGammaVariate`, and so as not to collide with the gamma function Γ of `aifn-compute/numerics/special`.)
 */
export function gammaVariate<A extends Param, C extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  shape: A,
  scale?: C,
  options?: O,
): Drawn<[A, C], O> {
  return drawEach(
    'gammaVariate',
    [shape, scale ?? 1],
    options,
    s,
    (e, a, c) => c * Math.exp(logGammaDraw(e, a)),
  ) as Drawn<[A, C], O>
}

/**
 * Beta(a, b) draws as G_a / (G_a + G_b) with independent gamma draws, formed from their logarithms (a sigmoid of the
 * difference) so that small a or b do not produce 0/0. Elementwise over broadcast a and b.
 */
export function beta<A extends Param, B extends Param, O extends SampleOptions = object>(
  s: Stream,
  a: A,
  b: B,
  options?: O,
): Drawn<[A, B], O> {
  return drawEach('beta', [a, b], options, s, (e, x, y) => betaDraw(e, x, y)) as Drawn<[A, B], O>
}

/** Chi-square draws with k > 0 degrees of freedom, 2 · Gamma(k/2, 1), elementwise over broadcast k. */
export function chiSquare<K extends Param, O extends SampleOptions = object>(
  s: Stream,
  df: K,
  options?: O,
): Drawn<[K], O> {
  return drawEach('chiSquare', [df], options, s, (e, k) => chiSquareDraw(e, k)) as Drawn<[K], O>
}

/**
 * Student t draws with ν > 0 degrees of freedom (ν = ∞ gives a normal), location and scale, Z / √(χ²_ν / ν),
 * elementwise over broadcast parameters.
 */
export function studentT<
  K extends Param,
  L extends Param = number,
  C extends Param = number,
  O extends SampleOptions = object,
>(s: Stream, df: K, loc?: L, scale?: C, options?: O): Drawn<[K, L, C], O> {
  return drawEach('studentT', [df, loc ?? 0, scale ?? 1], options, s, (e, nu, m, c) => {
    const z = standardNormal(e)
    if (nu === Infinity) return m + c * z
    return m + (c * z) / Math.sqrt(chiSquareDraw(e, nu) / nu)
  }) as Drawn<[K, L, C], O>
}

/**
 * Dirichlet(α) draws for concentrations α (length K, all > 0; or a batch of shape [..., K]), as a float64 tensor of
 * shape `[...shape, K]` on the simplex (`[K]` for one vector and no shape). Normalised gamma draws, formed in log space
 * so that small αₖ give tiny components rather than NaN.
 */
export function dirichlet(s: Stream, alpha: Tensor | ArrayLike<number>, options?: SampleOptions): Tensor {
  const { batch, k, values } = eventRows(alpha, 'dirichlet')
  const shape = options?.shape ?? batch
  checkBroadcast('dirichlet', batch, shape)
  const rows = batchIndex(batch, shape)
  const out = new Float64Array(rows.length * k)
  const element = eachChild(s, rows.length * k)
  for (let r = 0; r < rows.length; r++) {
    const base = r * k
    let m = -Infinity
    for (let j = 0; j < k; j++) {
      out[base + j] = logGammaDraw(element(base + j), values[rows[r] * k + j])
      if (out[base + j] > m) m = out[base + j]
    }
    // Normalise by log-sum-exp (max-shifted), so tiny log-gamma values do not underflow to 0/0.
    let total = 0
    for (let j = 0; j < k; j++) total += Math.exp(out[base + j] - m)
    const z = m + Math.log(total)
    for (let j = 0; j < k; j++) out[base + j] = Math.exp(out[base + j] - z)
  }
  return fromData(out, [...shape, k])
}

/**
 * Poisson(λ) draws (NaN for invalid λ), elementwise over broadcast λ. For λ < 10, inversion by sequential search (one
 * uniform); for λ ≥ 10, the transformed rejection method PTRS of Hörmann (1993), "The transformed rejection method for
 * generating Poisson random variables", Insurance: Mathematics and Economics 12.
 */
export function poisson<L extends Param, O extends SampleOptions = object>(
  s: Stream,
  lambda: L,
  options?: O,
): Drawn<[L], O> {
  return drawEach('poisson', [lambda], options, s, (e, l) => poissonDraw(e, l)) as Drawn<[L], O>
}

/**
 * Binomial(n, p) draws for integer n ≥ 0 (NaN for invalid parameters), elementwise over broadcast n and p. For
 * n·min(p, 1 − p) < 30, inversion by sequential search (Kachitvichyanukul and Schmeiser 1988, BINV). Otherwise the
 * order-statistic recursion (Knuth, TAOCP vol. 2, §3.4.1): the a-th smallest of n uniforms is X ~ Beta(a, n + 1 − a);
 * the count below p splits into a binomial below X or above it, halving n at each level, so a draw costs O(log n) beta
 * draws and is exact.
 */
export function binomial<N extends Param, P extends Param, O extends SampleOptions = object>(
  s: Stream,
  n: N,
  p: P,
  options?: O,
): Drawn<[N, P], O> {
  return drawEach('binomial', [n, p], options, s, (e, m, q) => binomialDraw(e, m, q)) as Drawn<[N, P], O>
}

/**
 * Multinomial(n, p) draws for integer n ≥ 0 and probabilities p (length K, normalised internally; or a batch of shape
 * [..., K]), as a float64 tensor of counts of shape `[...shape, K]`, each row summing to n. n may be a tensor, broadcast
 * with p's batch shape. Sequential conditional binomials: countₖ ~ Binomial(n − Σ_{j<k} countⱼ, pₖ / Σ_{j≥k} pⱼ).
 */
export function multinomial(s: Stream, n: Param, p: Tensor | ArrayLike<number>, options?: SampleOptions): Tensor {
  const { batch: pBatch, k, values } = eventRows(p, 'multinomial')
  const nShape = typeof n === 'number' ? [] : n.shape
  const joint = broadcastShapes(nShape, pBatch)
  const shape = options?.shape ?? joint
  checkBroadcast('multinomial', joint, shape)
  const rows = batchIndex(pBatch, shape)
  const ns = typeof n === 'number' ? null : toFlat(broadcastTo(n, shape))
  const out = new Float64Array(rows.length * k)
  const element = eachChild(s, rows.length * k)
  for (let r = 0; r < rows.length; r++) {
    const base = r * k
    const offset = rows[r] * k
    let rest = 0
    for (let j = 0; j < k; j++) rest += values[offset + j]
    let left = ns ? ns[r] : (n as number)
    for (let j = 0; j < k && left > 0; j++) {
      if (j === k - 1) {
        out[base + j] = left
        break
      }
      const share = rest > 0 ? Math.min(1, values[offset + j] / rest) : 0
      out[base + j] = binomialDraw(element(base + j), left, share)
      left -= out[base + j]
      rest -= values[offset + j]
    }
  }
  return fromData(out, [...shape, k])
}

/**
 * Multivariate normal draws μ + L z with z ~ N(0, I), for a mean μ (length d, an array or tensor; or a batch of shape
 * [..., d]) and a covariance or Cholesky factor (d × d). Returns a float64 tensor of shape `[...shape, d]` (`[d]` for
 * one mean and no shape); each row uses d successive standard normals of one block.
 *
 * A covariance is factored with `aifn-compute/numerics/linalg`'s `cholesky`, without jitter unless `options.jitter` allows it (the
 * `cholesky` options). A covariance that does not factor is an error rather than a silently regularised draw: pass
 * `jitter`, or a Cholesky factor of your own.
 */
export function multivariateNormal(
  s: Stream,
  mean: Tensor | ArrayLike<number>,
  spread: Spread,
  options: SampleOptions & { jitter?: CholeskyOptions['jitter'] } = {},
): Tensor {
  const { batch, k: d, values } = eventRows(mean, 'multivariateNormal')
  let factor: Tensor
  if ('covariance' in spread) {
    const c = cholesky(spread.covariance, { jitter: options.jitter ?? false })
    if (c.failed)
      throw new DomainError(
        'multivariateNormal',
        `multivariateNormal: the covariance is not positive definite (pivot ${c.failedAt}); pass jitter or a Cholesky factor`,
      )
    factor = c.L
  } else factor = spread.choleskyFactor
  if (factor.shape.length !== 2 || factor.shape[0] !== d || factor.shape[1] !== d)
    throw new ShapeError(
      'multivariateNormal',
      `multivariateNormal: needs a ${d} × ${d} factor, got shape ${showShape(factor.shape)}`,
      [factor.shape],
    )
  const L = toFlat(factor)
  const shape = options.shape ?? batch
  checkBroadcast('multivariateNormal', batch, shape)
  const rows = batchIndex(batch, shape)
  const out = new Float64Array(rows.length * d)
  const z = standardNormals(s, rows.length * d)
  for (let r = 0; r < rows.length; r++) {
    for (let i = 0; i < d; i++) {
      let v = values[rows[r] * d + i]
      for (let j = 0; j <= i; j++) v += L[i * d + j] * z[r * d + j]
      out[r * d + i] = v
    }
  }
  return fromData(out, [...shape, d])
}
