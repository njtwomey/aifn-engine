/**
 * Draws from the named families that need special functions or factorisations (decision D2): gamma, log-gamma, beta,
 * chi-square, Student t, Dirichlet, Poisson, binomial, multinomial and the multivariate normal.
 *
 * Each takes the stream first, like the draws of `aifn-compute/foundation/random`, and is written on its blocks and
 * child keys (design K §6):
 *
 * - samplers with rejection loops (gamma and its relatives, Poisson, binomial, Student t) give element $k$ the child
 *   key `child(s, '~', position + k)` and advance the stream by one word per element (`drawEach`), so the variable
 *   number of trials of one element never moves another element's draws;
 * - Dirichlet and multinomial rows key each component the same way (one word per component);
 * - the multivariate normal draws one block of $d$ standard normals per row (`standardNormals`).
 *
 * Invalid parameters give NaN draws rather than an error, except for the multivariate normal, whose covariance must
 * factor. None is differentiable: pathwise (reparameterised) draws are the `rsample` methods of
 * `aifn-compute/probability/distributions`.
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

/**
 * One uniform in $[0, 1)$ from a stream (2 words).
 *
 * @param s The stream; advanced by 2 words.
 * @returns The uniform draw.
 */
const unit = (s: Stream): number => units(s, 1)[0]

/**
 * One standard normal from a stream (4 words; the second Box–Muller output is discarded).
 *
 * @param s The stream; advanced by 4 words.
 * @returns The standard normal draw.
 */
const standardNormal = (s: Stream): number => standardNormals(s, 1)[0]

/**
 * Reserve `count` values, value $k$ to be drawn from its own child stream `child(s, '~', position + k)`, advancing `s`
 * by one word per value: the per-element keying of `drawEach`, for samplers that fill rows of an event (Dirichlet,
 * multinomial).
 *
 * @param s The parent stream; its position is advanced by `count` at once (no words are drawn from it).
 * @param count The number of values to reserve keys for.
 * @returns A function from a value's index $k$ (0 to `count - 1`) to that value's child stream.
 */
function eachChild(s: Stream, count: number): (k: number) => Stream {
  const start = s.position
  s.position += count
  return (k) => child(s, '~', start + k)
}

/**
 * The logarithm of one $\GammaD(a, 1)$ draw. Marsaglia and Tsang (2000), "A simple method for generating gamma
 * variables", ACM TOMS 26(3); for $a < 1$ the boost $G_a = G_{a + 1} U^{1/a}$ (their §6), kept in log space so that
 * tiny shapes do not underflow to 0.
 *
 * @param s The element's stream, from which the proposals (a normal and a uniform each) are drawn until one is
 *   accepted.
 * @param shape The shape $a$; NaN is returned unless $a > 0$.
 * @returns $\log G_a$.
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

/**
 * A $\Beta(a, b)$ draw $G_a / (G_a + G_b) = \sigma(\log G_a - \log G_b)$, from two log-gamma draws (no $0/0$ for
 * small $a$ or $b$).
 *
 * @param s The element's stream; both gamma draws come from it, $G_a$ first.
 * @param a The first shape parameter, $a > 0$ (NaN otherwise).
 * @param b The second shape parameter, $b > 0$ (NaN otherwise).
 * @returns The draw, in $[0, 1]$.
 */
function betaDraw(s: Stream, a: number, b: number): number {
  const la = logGammaDraw(s, a)
  const lb = logGammaDraw(s, b)
  return sigmoid(la - lb)
}

/**
 * A $\Poisson(\lambda)$ draw: inversion for $\lambda < 10$ (capped at 1000 steps), Hörmann's PTRS for
 * $\lambda \ge 10$ (see `poisson`).
 *
 * @param s The element's stream.
 * @param lambda The rate $\lambda \ge 0$; NaN is returned otherwise, and 0 for $\lambda = 0$.
 * @returns The count drawn.
 */
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

/**
 * A $\Binom(n, p)$ draw: BINV inversion when $n \min(p, 1 - p) < 30$, else the order-statistic recursion (see
 * `binomial`). For $p > 1/2$ it draws $n - \Binom(n, 1 - p)$.
 *
 * @param s The element's stream; the recursion draws its beta variates and inner binomials from it in turn.
 * @param n The number of trials, an integer $n \ge 0$ (NaN otherwise).
 * @param p The success probability, in $[0, 1]$ (NaN otherwise).
 * @returns The number of successes, from 0 to $n$.
 */
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

/**
 * A chi-square draw: $2 G$ with $G \sim \GammaD(k/2, 1)$.
 *
 * @param s The element's stream.
 * @param df The degrees of freedom $k > 0$ (NaN otherwise).
 * @returns The draw.
 */
function chiSquareDraw(s: Stream, df: number): number {
  return 2 * Math.exp(logGammaDraw(s, df / 2))
}

/**
 * Logarithms of $\GammaD(a, 1)$ draws, elementwise over $a$ (NaN where $a \le 0$). Marsaglia and Tsang (2000), "A
 * simple method for generating gamma variables", ACM TOMS 26(3); for $a < 1$ the boost $G_a = G_{a + 1} U^{1/a}$
 * (their §6), kept in log space so that tiny shapes do not underflow to 0.
 *
 * @param s The stream; advanced by one word per draw, each draw using its own child key.
 * @param shape The shape $a$: a number or a tensor.
 * @param options `shape`, the output shape (default: the shape of `shape`).
 * @returns $\log G_a$: a number when `shape` is a number and no output shape is given, else a float64 tensor.
 *
 * @example Tiny shapes stay finite in log space
 * const s = stream(0)
 * print('log G, a = 2:', logGammaVariate(s, 2, { shape: [3] }))
 * print('log G, a = 0.001:', logGammaVariate(s, 0.001, { shape: [3] }))
 * print('G itself, a = 0.001:', gammaVariate(s, 0.001, 1, { shape: [3] }))
 */
export function logGammaVariate<A extends Param, O extends SampleOptions = object>(
  s: Stream,
  shape: A,
  options?: O,
): Drawn<[A], O> {
  return drawEach('logGammaVariate', [shape], options, s, (e, a) => logGammaDraw(e, a)) as Drawn<[A], O>
}

/**
 * $\GammaD(a, \theta)$ draws with shape $a$ and scale $\theta$ (mean $a\theta$), by Marsaglia and Tsang (2000),
 * including $a < 1$, elementwise over broadcast parameters (NaN where $a \le 0$). For very small shapes a value can
 * underflow to 0; use `logGammaVariate` when that matters. (Named `…Variate` beside `logGammaVariate`, and so as not
 * to collide with the gamma function $\Gamma$ of `aifn-compute/numerics/special`.)
 *
 * @param s The stream; advanced by one word per draw, each draw using its own child key.
 * @param shape The shape $a$: a number or a tensor.
 * @param scale The scale $\theta$ (default 1), which multiplies a $\GammaD(a, 1)$ draw: a number or a tensor.
 * @param options `shape`, the output shape (default: the broadcast shape of `shape` and `scale`).
 * @returns A number when the parameters are numbers and no output shape is given, else a float64 tensor.
 *
 * @example The sample mean is near $a\theta$
 * const x = gammaVariate(stream(0), 3, 2, { shape: [2000] })
 * print('first draws:', gammaVariate(stream(0), 3, 2, { shape: [4] }))
 * print('sample mean:', sum(x) / 2000, 'a * theta:', 3 * 2)
 *
 * @example One draw per scale
 * print('scales 1, 10, 100:', gammaVariate(stream(1), 2, tensor([1, 10, 100])))
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
 * $\Beta(a, b)$ draws as $G_a / (G_a + G_b)$ with independent gamma draws, formed from their logarithms (a sigmoid
 * of the difference) so that small $a$ or $b$ do not produce $0/0$. Elementwise over broadcast $a$ and $b$ (NaN
 * where either is not positive).
 *
 * @param s The stream; advanced by one word per draw, each draw using its own child key.
 * @param a The first shape parameter: a number or a tensor.
 * @param b The second shape parameter: a number or a tensor, broadcast with `a`.
 * @param options `shape`, the output shape (default: the broadcast shape of `a` and `b`).
 * @returns Draws in $[0, 1]$: a number when `a` and `b` are numbers and no output shape is given, else a float64
 *   tensor.
 *
 * @example The sample mean is near $a / (a + b)$
 * const x = beta(stream(0), 2, 5, { shape: [2000] })
 * print('first draws:', beta(stream(0), 2, 5, { shape: [4] }))
 * print('sample mean:', sum(x) / 2000, 'a / (a + b):', 2 / 7)
 */
export function beta<A extends Param, B extends Param, O extends SampleOptions = object>(
  s: Stream,
  a: A,
  b: B,
  options?: O,
): Drawn<[A, B], O> {
  return drawEach('beta', [a, b], options, s, (e, x, y) => betaDraw(e, x, y)) as Drawn<[A, B], O>
}

/**
 * Chi-square draws with $k > 0$ degrees of freedom, $2G$ with $G \sim \GammaD(k/2, 1)$, elementwise over broadcast
 * $k$ (NaN where $k \le 0$).
 *
 * @param s The stream; advanced by one word per draw, each draw using its own child key.
 * @param df The degrees of freedom $k$: a number or a tensor.
 * @param options `shape`, the output shape (default: the shape of `df`).
 * @returns A number when `df` is a number and no output shape is given, else a float64 tensor.
 *
 * @example The sample mean is near $k$
 * const x = chiSquare(stream(0), 4, { shape: [2000] })
 * print('first draws:', chiSquare(stream(0), 4, { shape: [4] }))
 * print('sample mean:', sum(x) / 2000)
 */
export function chiSquare<K extends Param, O extends SampleOptions = object>(
  s: Stream,
  df: K,
  options?: O,
): Drawn<[K], O> {
  return drawEach('chiSquare', [df], options, s, (e, k) => chiSquareDraw(e, k)) as Drawn<[K], O>
}

/**
 * Student t draws with $\nu > 0$ degrees of freedom ($\nu = \infty$ gives a normal), location $m$ and scale $c$:
 * $m + c Z / \sqrt{X / \nu}$ with $Z \sim \Gauss(0, 1)$ and $X \sim \ChiSq_\nu$, elementwise over broadcast
 * parameters.
 *
 * @param s The stream; advanced by one word per draw, each draw using its own child key.
 * @param df The degrees of freedom $\nu$: a number or a tensor; `Infinity` draws $m + cZ$.
 * @param loc The location $m$ (default 0): a number or a tensor.
 * @param scale The scale $c$ (default 1): a number or a tensor.
 * @param options `shape`, the output shape (default: the broadcast shape of the parameters).
 * @returns A number when the parameters are numbers and no output shape is given, else a float64 tensor.
 *
 * @example Heavy tails at $\nu = 1$, a normal at $\nu = \infty$
 * const s = stream(0)
 * print('nu = 1:', studentT(s, 1, 0, 1, { shape: [6] }))
 * print('nu = Infinity:', studentT(s, Infinity, 0, 1, { shape: [6] }))
 * print('loc 10, scale 0.1:', studentT(s, 5, 10, 0.1, { shape: [3] }))
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
 * $\Dir(\alphavec)$ draws for concentrations $\alphavec$ (length $K$, all $> 0$; or a batch of shape
 * $[\dots, K]$), as a float64 tensor of shape `[...shape, K]` on the simplex (`[K]` for one vector and no shape).
 * Normalised gamma draws, formed in log space (by a max-shifted log-sum-exp) so that small $\alpha_k$ give tiny
 * components rather than NaN. A row with a non-positive $\alpha_k$ is NaN.
 *
 * @param s The stream; advanced by one word per component drawn, each component using its own child key.
 * @param alpha The concentrations: an array of $K$ numbers, or a tensor whose last axis has length $K$ and whose
 *   leading axes are a batch.
 * @param options `shape`, the output shape without the last axis: the batch shape of `alpha` must broadcast to it
 *   (default: that batch shape).
 * @returns The draws, each row of $K$ components summing to 1.
 *
 * @example Three draws, and their row sums
 * const x = dirichlet(stream(0), [1, 2, 3], { shape: [3] })
 * print('draws:', x)
 * print('row sums:', sum(x, 1))
 *
 * @example Small concentrations put almost all the mass on one component
 * print('alpha = 0.01:', dirichlet(stream(1), [0.01, 0.01, 0.01]))
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
 * $\Poisson(\lambda)$ draws (NaN for $\lambda < 0$ or NaN), elementwise over broadcast $\lambda$. For
 * $\lambda < 10$, inversion by sequential search (one uniform); for $\lambda \ge 10$, the transformed rejection
 * method PTRS of Hörmann (1993), "The transformed rejection method for generating Poisson random variables",
 * Insurance: Mathematics and Economics 12.
 *
 * @param s The stream; advanced by one word per draw, each draw using its own child key.
 * @param lambda The rate $\lambda$: a number or a tensor.
 * @param options `shape`, the output shape (default: the shape of `lambda`).
 * @returns Counts: a number when `lambda` is a number and no output shape is given, else a float64 tensor.
 *
 * @example Inversion for a small rate, rejection for a large one
 * const s = stream(0)
 * print('lambda = 3:', poisson(s, 3, { shape: [8] }))
 * print('lambda = 50:', poisson(s, 50, { shape: [8] }))
 * print('one per rate:', poisson(s, tensor([0, 1, 100])))
 */
export function poisson<L extends Param, O extends SampleOptions = object>(
  s: Stream,
  lambda: L,
  options?: O,
): Drawn<[L], O> {
  return drawEach('poisson', [lambda], options, s, (e, l) => poissonDraw(e, l)) as Drawn<[L], O>
}

/**
 * $\Binom(n, p)$ draws for integer $n \ge 0$ (NaN for invalid parameters), elementwise over broadcast $n$ and $p$.
 * For $n \min(p, 1 - p) < 30$, inversion by sequential search (Kachitvichyanukul and Schmeiser 1988, BINV).
 * Otherwise the order-statistic recursion (Knuth, TAOCP vol. 2, §3.4.1): the $a$-th smallest of $n$ uniforms is
 * $X \sim \Beta(a, n + 1 - a)$; the count below $p$ splits into a binomial below $X$ or above it, halving $n$ at each
 * level, so a draw costs $O(\log n)$ beta draws and is exact.
 *
 * @param s The stream; advanced by one word per draw, each draw using its own child key.
 * @param n The number of trials: a number or a tensor of non-negative integers.
 * @param p The success probability, in $[0, 1]$: a number or a tensor, broadcast with `n`.
 * @param options `shape`, the output shape (default: the broadcast shape of `n` and `p`).
 * @returns Counts of successes: a number when `n` and `p` are numbers and no output shape is given, else a float64
 *   tensor.
 *
 * @example Inversion for a small mean, the recursion for a large one
 * const s = stream(0)
 * print('n = 10, p = 0.3:', binomial(s, 10, 0.3, { shape: [8] }))
 * print('n = 1000, p = 0.5:', binomial(s, 1000, 0.5, { shape: [4] }))
 * print('invalid n = 2.5:', binomial(s, 2.5, 0.5))
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
 * $\Mult(n, \pvec)$ draws for integer $n \ge 0$ and probabilities $\pvec$ (length $K$, normalised internally; or a
 * batch of shape $[\dots, K]$), as a float64 tensor of counts of shape `[...shape, K]`, each row summing to $n$. $n$
 * may be a tensor, broadcast with the batch shape of $\pvec$. Sequential conditional binomials:
 * $c_k \sim \Binom(n - \sum_{j<k} c_j, \, p_k / \sum_{j \ge k} p_j)$, the last component taking what is left.
 *
 * @param s The stream; advanced by one word per component of the output, each component using its own child key.
 * @param n The number of trials: a number, or a tensor broadcast with the batch shape of `p`.
 * @param p The weights: an array of $K$ non-negative numbers, or a tensor whose last axis has length $K$ and whose
 *   leading axes are a batch. They need not sum to 1.
 * @param options `shape`, the output shape without the last axis: the broadcast shape of `n` and the batch of `p` must
 *   broadcast to it (default: that broadcast shape).
 * @returns The counts, each row of $K$ summing to $n$.
 *
 * @example Ten trials over three categories, four times
 * const x = multinomial(stream(0), 10, [0.2, 0.3, 0.5], { shape: [4] })
 * print('counts:', x)
 * print('row sums:', sum(x, 1))
 *
 * @example Unnormalised weights, and one $n$ per row
 * print('weights 1, 1, 2:', multinomial(stream(1), tensor([5, 100]), [1, 1, 2]))
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
 * Multivariate normal draws $\muvec + \Lmat\zvec$ with $\zvec \sim \Gauss(\zeros, \Imat)$, for a mean $\muvec$
 * (length $d$, an array or tensor; or a batch of shape $[\dots, d]$) and a covariance or Cholesky factor
 * ($d \times d$). Returns a float64 tensor of shape `[...shape, d]` (`[d]` for one mean and no shape); each row uses
 * $d$ successive standard normals of one block.
 *
 * A covariance is factored with `aifn-compute/numerics/linalg`'s `cholesky`, without jitter unless `options.jitter`
 * allows it (the `cholesky` options). A covariance that does not factor is an error (`DomainError`) rather than a
 * silently regularised draw: pass `jitter`, or a Cholesky factor of your own. A factor that is not $d \times d$ throws
 * `ShapeError`.
 *
 * @param s The stream; advanced by the words of one block of $d$ standard normals per output row.
 * @param mean The mean $\muvec$: an array of $d$ numbers, or a tensor whose last axis has length $d$ and whose leading
 *   axes are a batch.
 * @param spread `{ covariance }`, a $d \times d$ covariance $\Sigmamat$ to factor, or `{ choleskyFactor }`, a
 *   lower-triangular $\Lmat$ with $\Lmat\Lmat^\top = \Sigmamat$ (only its lower triangle is read).
 * @param options `shape`, the output shape without the last axis (the batch shape of `mean` must broadcast to it;
 *   default: that batch shape); `jitter`, passed to `cholesky` when a covariance is factored (default `false`).
 * @returns The draws, a float64 tensor of shape `[...shape, d]`.
 *
 * @example Correlated pairs from a covariance
 * const x = multivariateNormal(stream(0), [0, 0], { covariance: tensor([[1, 0.9], [0.9, 1]]) }, { shape: [5] })
 * print('draws:', x)
 *
 * @example A covariance that does not factor is reported
 * try {
 *   multivariateNormal(stream(0), [0, 0], { covariance: tensor([[1, 2], [2, 1]]) })
 * } catch (e) {
 *   print('error:', e.message)
 * }
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
