/**
 * Matchbox (Stern, Herbrich and Graepel, 2009): Bayesian recommendation from user and item features, learned online
 * by assumed-density filtering on a factor graph.
 *
 * A user is a sparse feature vector $\xvec$ (a one-hot id plus side features such as age band or region) and an item
 * is $\yvec$. Each of $K$ user traits is linear in the user's features, $s_k = \sum_i x_i u_{ki}$, and each item trait
 * in the item's, $t_k = \sum_j y_j v_{kj}$; biases $b = \sum_i x_i w_i + \sum_j y_j w'_j$ likewise. The latent rating
 * is the bilinear affinity $\tilde r = \sum_k s_k t_k + b$ plus Gaussian noise $\varepsilon$ of variance $\beta^2$, and
 * an ordinal rating $l \in \{0, \dots, L - 1\}$ is observed when $\tau_{l-1} < \tilde r + \varepsilon < \tau_l$, with
 * Gaussian thresholds $\tau_0 < \dots < \tau_{L-2}$ (and $\tau_{-1} = -\infty$, $\tau_{L-1} = \infty$). Every weight
 * and threshold has a factorised Gaussian posterior. Features let a new user or item borrow the traits of similar ones
 * (cold start).
 *
 * One rating is one update. Forward: the sum factors give Gaussian $s_k$, $t_k$ and $b$; the product $s_k t_k$ is
 * replaced by the Gaussian with its exact mean $\mu_s \mu_t$ and variance $v_s v_t + \mu_s^2 v_t + \mu_t^2 v_s$. The
 * two threshold comparisons are step factors on $\tilde r + \varepsilon - \tau$, iterated by EP with `stepTilted` and
 * the Gaussian message algebra of `aifn-compute/inference/expectation-propagation`. Backward: the message to
 * $\tilde r$ is split over the summands by the sum factor, the product factor sends $s_k$ the message with mean
 * $m \mu_t / (v_t + \mu_t^2)$ and variance $v / (v_t + \mu_t^2)$ (the variational message of Stern et al., 2009, §3),
 * and the sum factors pass each weight its share. Each posterior is multiplied by its message: ADF, one pass, no
 * stored messages. The item trait weights start at small random means, which breaks the symmetry of $s_k t_k$ (all
 * zero means would leave every product message flat). Infer.NET's Matchbox recommender learner implements the same
 * model.
 */

import { child, standardNormals, stream, units, type Stream } from 'aifn-compute/foundation/random'
import {
  divideGaussians,
  gaussianMoments,
  naturalGaussian,
  stepTilted,
} from 'aifn-compute/inference/expectation-propagation'
import { normalCdf } from 'aifn-compute/numerics/special'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A sparse feature vector: indices into the feature space and their values. */
export interface SparseFeatures {
  /** The indices of the non-zero features. */
  readonly index: readonly number[]
  /** Their values, parallel to `index`. */
  readonly value: readonly number[]
}

/** Factorised Gaussian posteriors, mean and variance per entry. */
export interface GaussianTable {
  /** The posterior mean of each entry. */
  readonly mean: Float64Array
  /** The posterior variance of each entry, parallel to `mean`. */
  readonly variance: Float64Array
}

/**
 * A Matchbox model: posteriors over the trait weights $\Umat$ ($K \times$ user features) and $\Vmat$ ($K \times$ item
 * features), the biases and the thresholds.
 */
export interface Matchbox {
  /** The number of traits $K$. */
  readonly traits: number
  /** The number of rating levels $L$. */
  readonly levels: number
  /** The standard deviation $\beta$ of the latent noise. */
  readonly beta: number
  /** The size of the user feature space. */
  readonly userFeatures: number
  /** The size of the item feature space. */
  readonly itemFeatures: number
  /** User trait weights $u_{ki}$, row-major: trait $k$'s weights at `k * userFeatures` onwards. */
  readonly U: GaussianTable
  /** Item trait weights $v_{kj}$, row-major: trait $k$'s weights at `k * itemFeatures` onwards. */
  readonly V: GaussianTable
  /** User bias weights $w_i$, one per user feature. */
  readonly userBias: GaussianTable
  /** Item bias weights $w'_j$, one per item feature. */
  readonly itemBias: GaussianTable
  /** The $L - 1$ thresholds $\tau_l$. */
  readonly thresholds: GaussianTable
}

/** Options of `matchbox`. */
export interface MatchboxOptions {
  /** Traits $K$ (default 2). */
  traits?: number
  /** Rating levels $L$, at least 2 (default 5). */
  levels?: number
  /** The latent noise's standard deviation $\beta$ (default 1). */
  beta?: number
  /** Prior variance of the trait weights (default 1). */
  traitVariance?: number
  /** Prior variance of the bias weights (default 1). */
  biasVariance?: number
  /** Prior variance of the thresholds (default 0.25). */
  thresholdVariance?: number
  /** The standard deviation of the item trait weights' random initial means (default 0.1). */
  symmetryBreaking?: number
}

/**
 * A table of $n$ Gaussians of one variance.
 *
 * @param n The number of entries.
 * @param variance The variance of every entry.
 * @param means The means, taken as they are (not copied); omitted, all 0.
 * @returns The table.
 */
const table = (n: number, variance: number, means?: Float64Array): GaussianTable => ({
  mean: means ?? new Float64Array(n),
  variance: new Float64Array(n).fill(variance),
})
/**
 * A copy of a table, so an update leaves the original as it was.
 *
 * @param t The table.
 * @returns New arrays with the same means and variances.
 */
const copy = (t: GaussianTable): GaussianTable => ({
  mean: Float64Array.from(t.mean),
  variance: Float64Array.from(t.variance),
})

/**
 * A Matchbox model with prior posteriors: zero-mean weights, except the item trait weights, whose means are drawn
 * with standard deviation `symmetryBreaking`; thresholds start evenly spaced, one unit apart, centred on 0. Throws
 * `DomainError` for fewer than two levels.
 *
 * @param s The stream of the item trait weights' initial means (`child(s, 'item traits')`).
 * @param userFeatures The size of the user feature space (ids and side features).
 * @param itemFeatures The size of the item feature space.
 * @param options The traits, levels, noise, prior variances and symmetry breaking.
 * @returns The model.
 *
 * @example The prior of five levels: thresholds centred on 0
 * const m = matchbox(stream(0), 3, 2)
 * print('thresholds:', m.thresholds.mean, ', variances:', m.thresholds.variance)
 * print('item trait means:', m.V.mean)
 */
export function matchbox(
  s: Stream,
  userFeatures: number,
  itemFeatures: number,
  options: MatchboxOptions = {},
): Matchbox {
  const { traits = 2, levels = 5, beta = 1, traitVariance = 1, biasVariance = 1, thresholdVariance = 0.25 } = options
  const { symmetryBreaking = 0.1 } = options
  if (levels < 2) throw new DomainError('matchbox', 'matchbox: at least two rating levels')
  const vMeans = Float64Array.from(
    standardNormals(child(s, 'item traits'), traits * itemFeatures),
    (e) => symmetryBreaking * e,
  )
  const tau = Float64Array.from({ length: levels - 1 }, (_, l) => l - (levels - 2) / 2)
  return {
    traits,
    levels,
    beta,
    userFeatures,
    itemFeatures,
    U: table(traits * userFeatures, traitVariance),
    V: table(traits * itemFeatures, traitVariance, vMeans),
    userBias: table(userFeatures, biasVariance),
    itemBias: table(itemFeatures, biasVariance),
    thresholds: table(levels - 1, thresholdVariance, tau),
  }
}

/**
 * Mean and variance of $\sum_i x_i w_i$ for weights at `offset + i` of a table.
 *
 * @param t The weights' posteriors.
 * @param x The features $x_i$.
 * @param offset Where the weights start in `t` (a trait's row of $\Umat$ or $\Vmat$).
 * @returns The mean and the variance.
 */
function sumForward(t: GaussianTable, x: SparseFeatures, offset = 0): [number, number] {
  let m = 0
  let v = 0
  x.index.forEach((i, q) => {
    m += x.value[q] * t.mean[offset + i]
    v += x.value[q] ** 2 * t.variance[offset + i]
  })
  return [m, v]
}

/**
 * Pass a message $\Gauss(m, v)$ on $\sum_i x_i w_i$ back to every weight through the sum factor and multiply it into
 * the posterior (in place): weight $i$ gets mean $(m - \sum_{j \ne i} x_j \mu_j) / x_i$ and variance
 * $(v + \sum_{j \ne i} x_j^2 v_j) / x_i^2$. Zero features, and messages whose variance is not positive and finite, are
 * skipped.
 *
 * @param t The weights' posteriors, updated in place.
 * @param x The features $x_i$.
 * @param m The message's mean $m$.
 * @param v The message's variance $v$.
 * @param offset Where the weights start in `t`.
 */
function sumBackward(t: GaussianTable, x: SparseFeatures, m: number, v: number, offset = 0) {
  const [sm, sv] = sumForward(t, x, offset)
  const updates = x.index.map((i, q) => {
    const a = x.value[q]
    if (a === 0) return null
    const k = offset + i
    const mean = (m - (sm - a * t.mean[k])) / a
    const variance = (v + (sv - a * a * t.variance[k])) / (a * a)
    return { k, mean, variance }
  })
  for (const u of updates) {
    if (!u || !Number.isFinite(u.variance) || u.variance <= 0) continue
    const p = 1 / t.variance[u.k] + 1 / u.variance
    t.mean[u.k] = (t.mean[u.k] / t.variance[u.k] + u.mean / u.variance) / p
    t.variance[u.k] = 1 / p
  }
}

/** The forward quantities of one user–item pair. */
interface Forward {
  /** Mean and variance of each user trait $s_k$. */
  s: [number, number][]
  /** Mean and variance of each item trait $t_k$. */
  t: [number, number][]
  /** Mean and variance of each product $s_k t_k$. */
  z: [number, number][]
  /** Mean and variance of the user bias. */
  bu: [number, number]
  /** Mean and variance of the item bias. */
  bi: [number, number]
  /** The mean of the latent affinity $\tilde r$ before the noise. */
  mean: number
  /** Its variance. */
  variance: number
}

/**
 * The forward pass of one pair: the traits, their products, the biases and the latent affinity, each a Gaussian.
 *
 * @param m The model.
 * @param x The user's features.
 * @param y The item's features.
 * @returns The forward quantities.
 */
function forward(m: Matchbox, x: SparseFeatures, y: SparseFeatures): Forward {
  const K = m.traits
  const s = Array.from({ length: K }, (_, k) => sumForward(m.U, x, k * m.userFeatures))
  const t = Array.from({ length: K }, (_, k) => sumForward(m.V, y, k * m.itemFeatures))
  const z = s.map(([ms, vs], k): [number, number] => {
    const [mt, vt] = t[k]
    return [ms * mt, vs * vt + ms * ms * vt + mt * mt * vs]
  })
  const bu = sumForward(m.userBias, x)
  const bi = sumForward(m.itemBias, y)
  const mean = z.reduce((a, [q]) => a + q, 0) + bu[0] + bi[0]
  const variance = z.reduce((a, [, q]) => a + q, 0) + bu[1] + bi[1]
  return { s, t, z, bu, bi, mean, variance }
}

/** A prediction: the traits' posteriors, the latent affinity and the probability of each rating level. */
export interface MatchboxPrediction {
  /** The posterior mean of each user trait $s_k$. */
  userTraits: Float64Array
  /** The posterior mean of each item trait $t_k$. */
  itemTraits: Float64Array
  /** The mean $\mu$ of the latent affinity $\tilde r$. */
  latentMean: number
  /** Its variance $v$. */
  latentVariance: number
  /** The probability of each level $0, \dots, L - 1$. */
  probabilities: Float64Array
  /** The expected rating level $\sum_l l \, \pr(l)$. */
  expected: number
}

/**
 * The predicted rating of a pair:
 * $\pr(\mathrm{rating} = l) = \Phi((\tau_l - \mu)/\sigma_l) - \Phi((\tau_{l-1} - \mu)/\sigma_{l-1})$ with
 * $\sigma_l^2 = v + \beta^2 + v_{\tau_l}$, for the pair's latent affinity $\Gauss(\mu, v)$ and the thresholds'
 * posterior means $\tau_l$ and variances $v_{\tau_l}$. Negative differences are set to 0 and the probabilities
 * renormalised.
 *
 * @param m The model.
 * @param x The user's features.
 * @param y The item's features.
 * @returns The traits, the latent affinity, and the probability and expectation of each level.
 *
 * @example A prior prediction is centred on the middle level
 * const m = matchbox(stream(0), 1, 1)
 * const p = matchboxPredict(m, { index: [0], value: [1] }, { index: [0], value: [1] })
 * print('probabilities:', p.probabilities)
 * print('expected level:', p.expected)
 */
export function matchboxPredict(m: Matchbox, x: SparseFeatures, y: SparseFeatures): MatchboxPrediction {
  const f = forward(m, x, y)
  const L = m.levels
  const below = (l: number) =>
    l < 0
      ? 0
      : l >= L - 1
        ? 1
        : (normalCdf(
            (m.thresholds.mean[l] - f.mean) / Math.sqrt(f.variance + m.beta ** 2 + m.thresholds.variance[l]),
          ) as number)
  const probabilities = Float64Array.from({ length: L }, (_, l) => Math.max(0, below(l) - below(l - 1)))
  const z = probabilities.reduce((a, b) => a + b, 0)
  probabilities.forEach((p, l) => (probabilities[l] = p / z))
  return {
    userTraits: Float64Array.from(f.s, ([q]) => q),
    itemTraits: Float64Array.from(f.t, ([q]) => q),
    latentMean: f.mean,
    latentVariance: f.variance,
    probabilities,
    expected: probabilities.reduce((a, p, l) => a + p * l, 0),
  }
}

/**
 * One ADF update on a rating `level` of item $\yvec$ by user $\xvec$ (see the file's introduction): EP on the (at most
 * two) threshold comparisons the level implies, then messages back to the thresholds, the biases and the trait
 * weights. Throws `DomainError` when `level` is not a whole number from 0 to $L - 1$.
 *
 * @param m The model; not modified.
 * @param x The user's features.
 * @param y The item's features.
 * @param level The observed rating level $l$, from 0 to $L - 1$.
 * @param options `iterations`, the EP sweeps over the threshold comparisons (default 4).
 * @returns The updated model, with new posterior tables.
 *
 * @example Repeated top ratings of one pair raise its expected level
 * let m = matchbox(stream(0), 1, 1)
 * const x = { index: [0], value: [1] }
 * const y = { index: [0], value: [1] }
 * print('before:', matchboxPredict(m, x, y).expected)
 * for (let t = 0; t < 5; t++) m = matchboxUpdate(m, x, y, 4)
 * print('after five ratings of 4:', matchboxPredict(m, x, y).expected)
 * print('top threshold:', m.thresholds.mean[3])
 */
export function matchboxUpdate(
  m: Matchbox,
  x: SparseFeatures,
  y: SparseFeatures,
  level: number,
  options: { iterations?: number } = {},
): Matchbox {
  const { iterations = 4 } = options
  const L = m.levels
  if (!(Number.isInteger(level) && level >= 0 && level < L))
    throw new DomainError('matchboxUpdate', `matchboxUpdate: level ${level}`)
  const f = forward(m, x, y)
  const thresholds = copy(m.thresholds)
  // The noisy rating r = r̃ + ε; its prior from above, and one site per active threshold comparison.
  const prior = naturalGaussian(f.mean, f.variance + m.beta ** 2)
  const sites = [
    { tau: level - 1, sign: 1, active: level > 0, msg: { precision: 0, shift: 0 } },
    { tau: level, sign: -1, active: level < L - 1, msg: { precision: 0, shift: 0 } },
  ].filter((q) => q.active)
  const marginal = () =>
    sites.reduce((g, q) => ({ precision: g.precision + q.msg.precision, shift: g.shift + q.msg.shift }), prior)
  // d = sign·(r − τ) > 0. EP over the (at most two) sites.
  const siteMessage = (q: (typeof sites)[number]) => {
    const cavity = gaussianMoments(divideGaussians(marginal(), q.msg))
    const tm = thresholds.mean[q.tau]
    const tv = thresholds.variance[q.tau]
    const dm = q.sign * (cavity.mean - tm)
    const dv = cavity.variance + tv
    const tilted = stepTilted(dm, dv, 0)
    const toD = gaussianMoments(divideGaussians(naturalGaussian(tilted.mean, tilted.variance), naturalGaussian(dm, dv)))
    return { cavity, toD }
  }
  for (let it = 0; it < iterations; it++)
    for (const q of sites) {
      const { toD } = siteMessage(q)
      if (!(toD.variance > 0) || !Number.isFinite(toD.variance)) continue
      // r = τ + sign·d: mean τ + sign·m_d, variance v_τ + v_d.
      const r = naturalGaussian(thresholds.mean[q.tau] + q.sign * toD.mean, thresholds.variance[q.tau] + toD.variance)
      q.msg = r
    }
  // The thresholds' messages: τ = r − sign·d, from each site's final cavity.
  for (const q of sites) {
    const { cavity, toD } = siteMessage(q)
    if (!(toD.variance > 0) || !Number.isFinite(toD.variance)) continue
    const mean = cavity.mean - q.sign * toD.mean
    const variance = cavity.variance + toD.variance
    const p = 1 / thresholds.variance[q.tau] + 1 / variance
    thresholds.mean[q.tau] = (thresholds.mean[q.tau] / thresholds.variance[q.tau] + mean / variance) / p
    thresholds.variance[q.tau] = 1 / p
  }
  const up = sites.reduce((g, q) => ({ precision: g.precision + q.msg.precision, shift: g.shift + q.msg.shift }), {
    precision: 0,
    shift: 0,
  })
  const next: Matchbox = {
    ...m,
    U: copy(m.U),
    V: copy(m.V),
    userBias: copy(m.userBias),
    itemBias: copy(m.itemBias),
    thresholds,
  }
  if (!(up.precision > 0)) return next
  // The message to r̃ through the noise factor, then split over the summands of r̃ = Σ zₖ + b_u + b_i.
  const M = up.shift / up.precision
  const Vr = 1 / up.precision + m.beta ** 2
  const toSummand = (mean: number, variance: number): [number, number] => [
    M - (f.mean - mean),
    Vr + (f.variance - variance),
  ]
  const [mbu, vbu] = toSummand(...f.bu)
  sumBackward(next.userBias, x, mbu, vbu)
  const [mbi, vbi] = toSummand(...f.bi)
  sumBackward(next.itemBias, y, mbi, vbi)
  f.z.forEach(([zm, zv], k) => {
    const [mz, vz] = toSummand(zm, zv)
    const [ms, vs] = f.s[k]
    const [mt, vt] = f.t[k]
    const et2 = vt + mt * mt
    const es2 = vs + ms * ms
    // The product factor's messages to sₖ and tₖ (Stern et al., 2009, Table 1).
    sumBackward(next.U, x, (mz * mt) / et2, vz / et2, k * m.userFeatures)
    sumBackward(next.V, y, (mz * ms) / es2, vz / es2, k * m.itemFeatures)
  })
  return next
}

// ── Synthetic ratings and a streamed run ──────────────────────────────────────────────────────────────────────────

/** Ratings drawn from a Matchbox-like model, with the true traits kept for checking. */
export interface MatchboxData {
  /** The number of users. */
  users: number
  /** The number of items. */
  items: number
  /** The number of rating levels $L$. */
  levels: number
  /** Side features per user. */
  userSide: number
  /** Side features per item. */
  itemSide: number
  /** The users' side features, dense and row-major, `users` rows of `userSide`. */
  userX: Float64Array
  /** The items' side features, `items` rows of `itemSide`. */
  itemY: Float64Array
  /** The ratings as (user, item, level) triples. */
  ratings: { user: Int32Array; item: Int32Array; level: Int32Array }
  /** The true user traits, row-major, `users` rows of $K$. */
  userTraits: Float64Array
  /** The true item traits, `items` rows of $K$. */
  itemTraits: Float64Array
  /** The $L - 1$ cut points of the noisy affinity between levels. */
  thresholds: Float64Array
}

/** Options of `matchboxRatings`. */
export interface MatchboxRatingsOptions {
  /** The number of users (default 200). */
  users?: number
  /** The number of items (default 100). */
  items?: number
  /** The number of true traits $K$ (default 2). */
  traits?: number
  /** The number of rating levels $L$ (default 5). */
  levels?: number
  /** Side features per user (default 3, standard normal); traits are linear in them plus noise. */
  userSide?: number
  /** Side features per item (default 3). */
  itemSide?: number
  /** The share of each trait's variance the side features explain (default 0.8); the rest is individual. */
  featureShare?: number
  /** Ratings per user (default 30), of distinct items while there are enough. */
  perUser?: number
  /** The standard deviation of the noise added to each affinity (default 0.5). */
  noise?: number
}

/**
 * Ratings on $L$ levels from the model: user traits $u_k = \sqrt{\rho} \, \avec_k^\top \fvec / \sqrt{d} +
 * \sqrt{1 - \rho} \, e$ and item traits $v_k = \sqrt{\rho} \, \bvec_k^\top \gvec / \sqrt{d'} + \sqrt{1 - \rho} \, e'$,
 * with side features $\fvec$, $\gvec$, weights $\avec_k$, $\bvec_k$ and noise $e$, $e'$ all standard normal and
 * $\rho$ the `featureShare`; the affinity $\sum_k u_k v_k$ plus noise is cut at its empirical quantiles into levels of
 * equal shares. Each user rates `perUser` items in a random order of the catalogue.
 *
 * @param s The stream every draw comes from.
 * @param options The sizes, the feature share and the noise.
 * @returns The side features, the ratings, and the true traits and thresholds.
 *
 * @example Equal shares of each level
 * const d = matchboxRatings(stream(0), { users: 20, items: 10, perUser: 5 })
 * const count = (l) => d.ratings.level.filter((v) => v === l).length
 * print('ratings:', d.ratings.level.length, '; per level:', [0, 1, 2, 3, 4].map(count))
 * print('thresholds:', d.thresholds)
 */
export function matchboxRatings(s: Stream, options: MatchboxRatingsOptions = {}): MatchboxData {
  const { users = 200, items = 100, traits: K = 2, levels = 5, userSide = 3, itemSide = 3 } = options
  const { featureShare = 0.8, perUser = 30, noise = 0.5 } = options
  const userX = standardNormals(child(s, 'user side'), users * userSide)
  const itemY = standardNormals(child(s, 'item side'), items * itemSide)
  const A = standardNormals(child(s, 'A'), K * userSide)
  const B = standardNormals(child(s, 'B'), K * itemSide)
  const eu = standardNormals(child(s, 'user noise'), users * K)
  const ei = standardNormals(child(s, 'item noise'), items * K)
  const mix = (side: Float64Array, W: Float64Array, e: Float64Array, n: number, d: number) => {
    const out = new Float64Array(n * K)
    for (let r = 0; r < n; r++)
      for (let k = 0; k < K; k++) {
        let a = 0
        for (let j = 0; j < d; j++) a += W[k * d + j] * side[r * d + j]
        out[r * K + k] = Math.sqrt(featureShare) * (a / Math.sqrt(d)) + Math.sqrt(1 - featureShare) * e[r * K + k]
      }
    return out
  }
  const userTraits = mix(userX, A, eu, users, userSide)
  const itemTraits = mix(itemY, B, ei, items, itemSide)
  const n = users * perUser
  const user = new Int32Array(n)
  const item = new Int32Array(n)
  const latent = new Float64Array(n)
  const eps = standardNormals(child(s, 'rating noise'), n)
  for (let u = 0; u < users; u++) {
    // perUser distinct items per user.
    const order = Array.from({ length: items }, (_, i) => i)
    const w = units(child(s, 'items of', u), items)
    order.sort((p, q) => w[p] - w[q])
    for (let q = 0; q < perUser; q++) {
      const r = u * perUser + q
      user[r] = u
      item[r] = order[q % items]
      let a = 0
      for (let k = 0; k < K; k++) a += userTraits[u * K + k] * itemTraits[item[r] * K + k]
      latent[r] = a + noise * eps[r]
    }
  }
  const sorted = Float64Array.from(latent).sort()
  const thresholds = Float64Array.from({ length: levels - 1 }, (_, l) => sorted[Math.floor(((l + 1) * n) / levels)])
  const level = Int32Array.from(latent, (v) => {
    let l = 0
    while (l < levels - 1 && v > thresholds[l]) l++
    return l
  })
  return {
    users,
    items,
    levels,
    userSide,
    itemSide,
    userX,
    itemY,
    ratings: { user, item, level },
    userTraits,
    itemTraits,
    thresholds,
  }
}

/**
 * The sparse features of a user or item: a one-hot id and, with `withSide`, the side features after the ids.
 *
 * @param id The user or item index, the one-hot feature.
 * @param count The number of users (or items): the side features start at this index.
 * @param side The side features, row-major, `d` per user or item.
 * @param d The side features per user or item.
 * @param withSide Whether to include the side features.
 * @returns The features.
 */
const features = (id: number, count: number, side: Float64Array, d: number, withSide: boolean): SparseFeatures => ({
  index: withSide ? [id, ...Array.from({ length: d }, (_, j) => count + j)] : [id],
  value: withSide ? [1, ...Array.from({ length: d }, (_, j) => side[id * d + j])] : [1],
})

/** Options of `matchboxRun`. */
export interface MatchboxRunOptions extends MatchboxOptions {
  /** Use the side features (default true); without them a new user or item is the prior. */
  useFeatures?: boolean
  /** Share of users held out entirely, the cold-start users (default 0.1). */
  coldUsers?: number
  /** Share of the other users' ratings held out for testing (default 0.2). */
  testShare?: number
  /** Passes over the training ratings (default 1: online ADF; more reuse ratings and overcount them). */
  passes?: number
  /** About how many checkpoints to take over the run (default 30), besides the one before training. */
  checkpoints?: number
  /** The root seed of the split, the prior and the order of the ratings (default 0). */
  seed?: number | string
}

/** One checkpoint of a Matchbox run. */
export interface MatchboxCheckpoint {
  /** Ratings learned from so far. */
  seen: number
  /** RMSE of the expected level on the held-out ratings of known users. */
  rmse: number
  /** Share of those ratings whose most probable level is right. */
  accuracy: number
  /** Their mean log-probability (each probability floored at $10^{-12}$). */
  logProbability: number
  /** The same RMSE on the cold-start users' ratings. */
  coldRmse: number
  /** Posterior mean item traits, row-major, `items` rows of $K$. */
  itemTraits: Float64Array
  /** The mean posterior variance of the item traits. */
  traitVariance: number
  /** The thresholds' posterior means. */
  thresholds: Float64Array
}

/** A Matchbox run so far. */
export interface MatchboxRun {
  /** Updates in the whole run: training ratings times passes. */
  total: number
  /** Updates made so far. */
  seen: number
  /** True once every update is made. */
  finished: boolean
  /** The number of rating levels $L$. */
  levels: number
  /** The number of traits $K$. */
  traits: number
  /** Baseline: RMSE of each item's training mean level on the known users' held-out ratings. */
  itemMeanRmse: number
  /** Baseline: RMSE of the global training mean on the cold-start users' ratings. */
  coldGlobalRmse: number
  /** The checkpoints so far, the first before any update. */
  checkpoints: MatchboxCheckpoint[]
  /** The test triples of known users. */
  test: { user: Int32Array; item: Int32Array; level: Int32Array }
  /** The predicted distributions of the test triples at the latest checkpoint, row-major, one row of $L$ each. */
  predictions: Float64Array
  /** True item traits, row-major, `items` rows of $K$ (for comparing up to rotation). */
  trueItemTraits: Float64Array
  /** The model's posteriors now. */
  model: Matchbox
}

/**
 * Train Matchbox online on `matchboxRatings` data and yield checkpoints: a share of users is held out entirely (cold
 * start) and a share of the others' ratings held out for testing; the rest are learned one `matchboxUpdate` at a time
 * in a random order. Each user's features are its one-hot id and, with `useFeatures`, its side features; likewise
 * each item's. The levels come from the data, overriding `options.levels`. Deterministic in `seed`.
 *
 * @param data The ratings and side features.
 * @param options The model's options, the use of features, the held-out shares, the passes, the number of
 *   checkpoints and the seed.
 * @returns A generator of the run so far, yielding before training and at each checkpoint, and returning the final
 *   state.
 *
 * @example Held-out error falls below the item-mean baseline
 * const data = matchboxRatings(stream(0), { users: 60, items: 30, perUser: 15 })
 * const runs = [...matchboxRun(data, { checkpoints: 3 })]
 * const last = runs[runs.length - 1]
 * print('item-mean RMSE:', last.itemMeanRmse, '; global-mean RMSE on cold users:', last.coldGlobalRmse)
 * for (const c of last.checkpoints) print('after', c.seen, 'ratings: RMSE', c.rmse, ', cold RMSE', c.coldRmse)
 */
export function* matchboxRun(
  data: MatchboxData,
  options: MatchboxRunOptions = {},
): Generator<MatchboxRun, MatchboxRun> {
  const { useFeatures = true, coldUsers = 0.1, testShare = 0.2, passes = 1, checkpoints = 30, seed = 0 } = options
  const root = stream(seed)
  const { users, items, levels, userSide, itemSide, userX, itemY } = data
  const K = options.traits ?? 2
  let model = matchbox(child(root, 'init'), users + userSide, items + itemSide, { ...options, traits: K, levels })
  const cold = new Set<number>()
  const cu = units(child(root, 'cold'), users)
  for (let u = 0; u < users; u++) if (cu[u] < coldUsers) cold.add(u)
  const tu = units(child(root, 'test'), data.ratings.user.length)
  const train: number[] = []
  const test: number[] = []
  const coldTest: number[] = []
  data.ratings.user.forEach((u, r) => (cold.has(u) ? coldTest : tu[r] < testShare ? test : train).push(r))
  const xOf = (u: number) => features(u, users, userX, userSide, useFeatures)
  const yOf = (i: number) => features(i, items, itemY, itemSide, useFeatures)
  const { user, item, level } = data.ratings
  // Baselines from the training ratings.
  const itemSum = new Float64Array(items)
  const itemCount = new Float64Array(items)
  let globalSum = 0
  for (const r of train) {
    itemSum[item[r]] += level[r]
    itemCount[item[r]]++
    globalSum += level[r]
  }
  const globalMean = globalSum / Math.max(1, train.length)
  const rmseOf = (rows: number[], predict: (r: number) => number) =>
    Math.sqrt(rows.reduce((a, r) => a + (predict(r) - level[r]) ** 2, 0) / Math.max(1, rows.length))
  const itemMeanRmse = rmseOf(test, (r) => (itemCount[item[r]] ? itemSum[item[r]] / itemCount[item[r]] : globalMean))
  const coldGlobalRmse = rmseOf(coldTest, () => globalMean)
  const shots: MatchboxCheckpoint[] = []
  let predictions = new Float64Array(test.length * levels)
  const checkpoint = (seen: number) => {
    let se = 0
    let hit = 0
    let lp = 0
    predictions = new Float64Array(test.length * levels)
    test.forEach((r, q) => {
      const p = matchboxPredict(model, xOf(user[r]), yOf(item[r]))
      predictions.set(p.probabilities, q * levels)
      se += (p.expected - level[r]) ** 2
      let best = 0
      for (let l = 1; l < levels; l++) if (p.probabilities[l] > p.probabilities[best]) best = l
      if (best === level[r]) hit++
      lp += Math.log(Math.max(1e-12, p.probabilities[level[r]]))
    })
    const coldRmse = rmseOf(coldTest, (r) => matchboxPredict(model, xOf(user[r]), yOf(item[r])).expected)
    const traits = new Float64Array(items * K)
    let tv = 0
    for (let i = 0; i < items; i++) {
      const f = forward(model, xOf(0), yOf(i))
      f.t.forEach(([mt, vt], k) => {
        traits[i * K + k] = mt
        tv += vt / (items * K)
      })
    }
    shots.push({
      seen,
      rmse: Math.sqrt(se / Math.max(1, test.length)),
      accuracy: hit / Math.max(1, test.length),
      logProbability: lp / Math.max(1, test.length),
      coldRmse,
      itemTraits: traits,
      traitVariance: tv,
      thresholds: Float64Array.from(model.thresholds.mean),
    })
  }
  const total = train.length * passes
  const snapshot = (seen: number, finished: boolean): MatchboxRun => ({
    total,
    seen,
    finished,
    levels,
    traits: K,
    itemMeanRmse,
    coldGlobalRmse,
    checkpoints: shots.slice(),
    test: {
      user: Int32Array.from(test, (r) => user[r]),
      item: Int32Array.from(test, (r) => item[r]),
      level: Int32Array.from(test, (r) => level[r]),
    },
    predictions,
    trueItemTraits: data.itemTraits,
    model,
  })
  checkpoint(0)
  yield snapshot(0, false)
  const every = Math.max(1, Math.round(total / checkpoints))
  let seen = 0
  for (let pass = 0; pass < passes; pass++) {
    const w = units(child(root, 'order', pass), train.length)
    const order = train.map((r, q) => ({ r, w: w[q] })).sort((a, b) => a.w - b.w)
    for (const { r } of order) {
      model = matchboxUpdate(model, xOf(user[r]), yOf(item[r]), level[r])
      seen++
      if (seen % every === 0 || seen === total) {
        checkpoint(seen)
        yield snapshot(seen, seen === total)
      }
    }
  }
  return snapshot(seen, true)
}
