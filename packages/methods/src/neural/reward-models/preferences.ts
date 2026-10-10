/**
 * Reward models from pairwise preferences, and what optimising against them does: synthetic preferences labelled by a
 * known gold reward, a Bradley–Terry reward model fitted to them (Bradley and Terry, 1952), and the exact best-of-$n$
 * curve that shows over-optimisation (Gao et al., 2023).
 *
 * Under the Bradley–Terry model a response $a$ is preferred to $b$ with probability
 * $\sigma\big((g(a) - g(b))/T\big)$ for a reward $g$ and temperature $T$. A reward model $r_\wvec(\xvec) =
 * \wvec^\top\phivec(\xvec)$ is fitted by maximising the likelihood of the observed choices, which is logistic
 * regression without an intercept on the differences $\phivec(\xvec_w) - \phivec(\xvec_l)$.
 *
 * Best-of-$n$ sampling draws $n$ responses and keeps the one the proxy reward ranks highest. As $n$ grows the policy
 * moves away from the base distribution, by at most $\log n - (n - 1)/n$ in KL divergence (Beirami et al., 2025), and
 * the proxy reward rises without limit; when the proxy is misspecified the gold reward rises, peaks and then falls, the
 * over-optimisation that Gao et al. (2023) measure.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, normals, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'

type F64 = dense.F64

/** Options of `syntheticPreferences`. */
export interface SyntheticPreferencesOptions {
  /** The seed of the root stream, `stream(seed)`. */
  seed: number
  /** The number of preference pairs $n$. */
  n: Size
  /** The dimension $d$ of each response's features. */
  dim: Size
  /** The gold reward $g(\xvec)$ that decides which response of a pair is preferred. */
  gold: (x: Float64Array) => number
  /**
   * `'gumbel'`: each response's utility is $g(\xvec)/T$ plus an independent standard Gumbel draw, and the larger wins,
   * so $a$ is chosen with probability $\sigma\big((g(a) - g(b))/T\big)$ (the Bradley–Terry model). `'none'`: the
   * response with the larger gold reward always wins.
   */
  noise: 'gumbel' | 'none'
  /** The temperature $T > 0$ of the choice: larger makes the labels noisier. Ignored without noise. */
  temperature: number
}

/** Preference pairs: row $i$ of `chosen` was preferred to row $i$ of `rejected`. */
export interface SyntheticPreferences {
  /** The preferred responses, $n \times d$. */
  chosen: Tensor
  /** The other response of each pair, $n \times d$. */
  rejected: Tensor
  /** $g$ of each chosen response, $n$ values. */
  goldChosen: Float64Array
  /** $g$ of each rejected response, $n$ values. */
  goldRejected: Float64Array
}

/**
 * Preference pairs between random responses, labelled by a gold reward under the Bradley–Terry model. The two
 * responses of each pair have standard normal features drawn from `child(stream(seed), 'points')` ($2n \times d$
 * values, the first response of pair $i$ in row $2i$); the Gumbel draws come from `child(stream(seed), 'noise')`.
 *
 * @param options The seed, the number of pairs, the dimension, the gold reward, the noise and the temperature.
 * @returns The chosen and rejected responses ($n \times d$ each) and their gold rewards.
 *
 * @example Noisy labels: the chosen response usually, not always, has the larger gold reward
 * const gold = (x) => x[0] + 0.5 * x[1]
 * const prefs = syntheticPreferences({ seed: 1, n: 1000, dim: 2, gold, noise: 'gumbel', temperature: 1 })
 * const agree = prefs.goldChosen.filter((g, i) => g > prefs.goldRejected[i]).length
 * print('pairs where the chosen has the larger gold reward:', agree, 'of 1000')
 */
export function syntheticPreferences(options: SyntheticPreferencesOptions): SyntheticPreferences {
  const { seed, n, dim, gold, noise, temperature } = options
  const where = 'syntheticPreferences'
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError(where, `${where}: n must be a positive integer, got ${n}`)
  if (!(Number.isInteger(dim) && dim >= 1))
    throw new DomainError(where, `${where}: dim must be a positive integer, got ${dim}`)
  if (noise === 'gumbel' && !(temperature > 0))
    throw new DomainError(where, `${where}: temperature must be positive, got ${temperature}`)
  const root: Stream = stream(seed)
  const points = dense.data(normals(child(root, 'points'), [2 * n, dim]))
  const u = noise === 'gumbel' ? dense.data(uniform(child(root, 'noise'), 0, 1, { shape: [2 * n] })) : null
  // A standard Gumbel draw from a uniform one.
  const gumbel = (v: number) => -Math.log(-Math.log(v))
  const chosen = new Float64Array(n * dim)
  const rejected = new Float64Array(n * dim)
  const goldChosen = new Float64Array(n)
  const goldRejected = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const a = points.slice(2 * i * dim, (2 * i + 1) * dim)
    const b = points.slice((2 * i + 1) * dim, (2 * i + 2) * dim)
    const ga = gold(a)
    const gb = gold(b)
    const aWins = u ? ga / temperature + gumbel(u[2 * i]) > gb / temperature + gumbel(u[2 * i + 1]) : ga >= gb
    chosen.set(aWins ? a : b, i * dim)
    rejected.set(aWins ? b : a, i * dim)
    goldChosen[i] = aWins ? ga : gb
    goldRejected[i] = aWins ? gb : ga
  }
  return { chosen: dense.mat(chosen, n, dim), rejected: dense.mat(rejected, n, dim), goldChosen, goldRejected }
}

/** Options of `fitBradleyTerry`. */
export interface FitBradleyTerryOptions {
  /** The L2 penalty $\lambda$ on the weights, so that the fit stays finite on separable data. Default $10^{-2}$. */
  l2?: number
  /** The features $\phivec(\xvec)$ the reward is linear in. Default the response's own features. */
  features?: (x: Float64Array) => Float64Array
}

/** A fitted Bradley–Terry reward model, $r(\xvec) = \wvec^\top\phivec(\xvec)$. */
export interface BradleyTerryReward {
  /** The weights $\wvec$, one per feature. */
  weights: Float64Array
  /** The reward of a response, $\wvec^\top\phivec(\xvec)$. */
  score(x: Float64Array): number
}

/**
 * Fit a Bradley–Terry reward model to preference pairs: the weights $\wvec$ minimising
 * $\sum_i -\log\sigma\big(\wvec^\top(\phivec(\xvec_{w,i}) - \phivec(\xvec_{l,i}))\big) + \frac{\lambda}{2}\lVert \wvec
 * \rVert^2$, logistic regression without an intercept on the feature differences. It is fitted by `logisticRegression`,
 * which needs both classes, so each difference is given twice, with label 1 and negated with label 0; that doubles the
 * likelihood, and the penalty passed is $2\lambda$ so that the minimiser is the one above.
 *
 * @param data The preference pairs: row $i$ of `chosen` was preferred to row $i$ of `rejected` (each $n \times d$).
 * @param options The penalty and the features.
 * @returns The weights and the reward function.
 *
 * @example Recover the direction of a linear gold reward
 * const gold = (x) => 2 * x[0] - x[1]
 * const prefs = syntheticPreferences({ seed: 2, n: 2000, dim: 2, gold, noise: 'gumbel', temperature: 1 })
 * const { weights, score } = fitBradleyTerry(prefs)
 * print('weights =', weights)
 * print('reward of (1, 0) =', score(new Float64Array([1, 0])))
 */
export function fitBradleyTerry(
  data: { chosen: Tensor; rejected: Tensor },
  options: FitBradleyTerryOptions = {},
): BradleyTerryReward {
  const { l2 = 1e-2, features = (x: Float64Array) => x } = options
  const where = 'fitBradleyTerry'
  if (!(l2 >= 0)) throw new DomainError(where, `${where}: l2 must be non-negative, got ${l2}`)
  const c = dense.toMatrixF64(data.chosen, where)
  const r = dense.toMatrixF64(data.rejected, where)
  if (c.m !== r.m || c.n !== r.n)
    throw new ShapeError(where, `${where}: chosen is ${c.m} × ${c.n} but rejected is ${r.m} × ${r.n}`)
  const row = (m: { data: F64; n: number }, i: number) => m.data.slice(i * m.n, (i + 1) * m.n)
  const diffs = Array.from({ length: c.m }, (_, i) => {
    const a = features(row(c, i))
    const b = features(row(r, i))
    return Float64Array.from(a, (v, k) => v - b[k])
  })
  const p = diffs[0]?.length ?? 0
  const x = new Float64Array(2 * c.m * p)
  diffs.forEach((dRow, i) => {
    x.set(dRow, i * p)
    x.set(
      dRow.map((v) => -v),
      (c.m + i) * p,
    )
  })
  const y = new Int32Array(2 * c.m).fill(1, 0, c.m)
  const model = logisticRegression({ intercept: false, l2: 2 * l2 }).fit(
    dataset(dense.mat(x, 2 * c.m, p), dense.vec(Float64Array.from(y))),
  )
  const weights = Float64Array.from(dense.data(model.weights))
  return { weights, score: (z) => dense.dot(weights, features(z)) }
}

/**
 * The usual formula for the KL divergence of the best-of-$n$ policy from the base policy, $\log n - (n - 1)/n$
 * (Stiennon et al., 2020). Beirami et al. (2025) prove it is an upper bound: the divergence is smaller whenever two
 * responses can share a proxy reward, and approaches the formula as the base distribution spreads over ever more
 * responses with distinct rewards.
 *
 * @param n The number of responses drawn, a positive integer.
 * @returns $\log n - (n - 1)/n$, in nats: $0$ for $n = 1$, increasing with $n$.
 *
 * @example The bound grows like log n
 * for (const n of [1, 2, 4, 16, 256]) print(`n = ${n}: KL bound =`, bestOfNKl(n))
 */
export function bestOfNKl(n: Size): number {
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError('bestOfNKl', `bestOfNKl: n must be a positive integer`)
  return Math.log(n) - (n - 1) / n
}

/** One point of a best-of-$n$ curve. */
export interface BestOfNPoint {
  /** The number of responses drawn. */
  n: Size
  /** `bestOfNKl(n)`, the bound on the KL divergence from the base policy. */
  kl: number
  /** The expected proxy reward of the best of $n$. */
  proxy: number
  /** The expected gold reward of the best of $n$. */
  gold: number
}

/**
 * The probability that each item of a pool, sorted by proxy reward in ascending order, is the best of $n$ drawn
 * without replacement: item $i$ (counting from 1) is the best when it is drawn and the other $n - 1$ come from the
 * $i - 1$ below it, with probability $\binom{i - 1}{n - 1} / \binom{M}{n}$. The weights are built in log space by the
 * ratio $\binom{i - 1}{n - 1} / \binom{i - 2}{n - 1} = (i - 1)/(i - n)$ and normalised, so no binomial overflows.
 *
 * @param M The pool's size.
 * @param n The number drawn, from 1 to $M$.
 * @returns $M$ probabilities summing to 1, zero below rank $n$.
 *
 * @example The best of 3 from a pool of 5
 * // Rank 5 (the top) is in 6 of the 10 subsets of three; rank 3 is the best only of {1, 2, 3}.
 * print(bestOfNWeights(5, 3))
 */
export function bestOfNWeights(M: Size, n: Size): Float64Array {
  const logW = new Float64Array(M).fill(-Infinity)
  logW[n - 1] = 0
  for (let i = n + 1; i <= M; i++) logW[i - 1] = logW[i - 2] + Math.log(i - 1) - Math.log(i - n)
  const top = logW[M - 1]
  let total = 0
  const w = logW.map((v) => {
    const e = Math.exp(v - top)
    total += e
    return e
  })
  return w.map((v) => v / total)
}

/**
 * The best-of-$n$ curve of a pool of $M$ responses: for each $n$, the KL bound `bestOfNKl(n)` and the exact expected
 * proxy and gold rewards of the response the proxy ranks highest among $n$ drawn at random without replacement,
 * $\sum_i g_{(i)} \binom{i - 1}{n - 1} / \binom{M}{n}$ with the pool sorted by proxy reward. Ties in the proxy are
 * broken at random, so tied responses share the mean of their gold rewards.
 *
 * @param options The pool and the values of $n$.
 * @param options.proxy The proxy reward of each response of the pool, $M$ values.
 * @param options.gold The gold reward of each response, $M$ values.
 * @param options.ns The numbers of responses to draw, each from 1 to $M$.
 * @returns One point per $n$, in the order of `ns`.
 *
 * @example Over-optimisation: the proxy keeps rising, the gold reward peaks
 * // The proxy is the response itself; the gold reward agrees for small x but penalises large ones.
 * const x = Float64Array.from(toArray(normals(stream(1), 4000)))
 * const gold = x.map((v) => v - 0.25 * v * v)
 * const curve = bestOfNCurve({ proxy: x, gold, ns: [1, 4, 16, 64, 256, 1024] })
 * for (const p of curve) print(`n = ${p.n}: KL bound`, p.kl, ' proxy', p.proxy, ' gold', p.gold)
 */
export function bestOfNCurve(options: {
  proxy: Float64Array
  gold: Float64Array
  ns: readonly Size[]
}): BestOfNPoint[] {
  const { proxy, gold, ns } = options
  const where = 'bestOfNCurve'
  const M = proxy.length
  if (gold.length !== M) throw new ShapeError(where, `${where}: ${M} proxy rewards but ${gold.length} gold rewards`)
  if (M === 0) throw new DomainError(where, `${where}: the pool is empty`)
  const order = Array.from({ length: M }, (_, i) => i).sort((a, b) => proxy[a] - proxy[b])
  const p = Float64Array.from(order, (i) => proxy[i])
  const g = Float64Array.from(order, (i) => gold[i])
  // Tied proxies share their mean gold reward: the best of a tied group is any of them with equal chance.
  for (let start = 0; start < M;) {
    let end = start + 1
    while (end < M && p[end] === p[start]) end++
    if (end - start > 1) {
      const mean = g.slice(start, end).reduce((a, b) => a + b, 0) / (end - start)
      g.fill(mean, start, end)
    }
    start = end
  }
  return ns.map((n) => {
    if (!(Number.isInteger(n) && n >= 1 && n <= M))
      throw new DomainError(where, `${where}: each n must be an integer from 1 to ${M}, got ${n}`)
    const w = bestOfNWeights(M, n)
    return { n, kl: bestOfNKl(n), proxy: dense.dot(w, p), gold: dense.dot(w, g) }
  })
}
