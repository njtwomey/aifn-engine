/**
 * Matchbox (Stern, Herbrich and Graepel, 2009): Bayesian recommendation from user and item features, learned online
 * by assumed-density filtering on a factor graph.
 *
 * A user is a sparse feature vector x (a one-hot id plus side features such as age band or region) and an item is y.
 * Each of K user traits is linear in the user's features, sₖ = Σᵢ xᵢ uₖᵢ, and each item trait in the item's,
 * tₖ = Σⱼ yⱼ vₖⱼ; biases b = Σᵢ xᵢ wᵢ + Σⱼ yⱼ w′ⱼ likewise. The latent rating is the bilinear affinity
 * r̃ = Σₖ sₖtₖ + b plus Gaussian noise of variance β², and an ordinal rating l ∈ {0, …, L − 1} is observed when
 * τₗ₋₁ < r̃ + ε < τₗ, with Gaussian thresholds τ₀ < … < τ_{L−2}. Every weight and threshold has a factorised Gaussian
 * posterior. Features let a new user or item borrow the traits of similar ones (cold start).
 *
 * One rating is one update. Forward: the sum factors give Gaussian sₖ, tₖ and b; the product sₖtₖ is replaced by the
 * Gaussian with its exact mean μₛμₜ and variance vₛvₜ + μₛ²vₜ + μₜ²vₛ. The two threshold comparisons are step factors
 * on r̃ + ε − τ, iterated by EP with `stepTilted` and the Gaussian message algebra of
 * `aifn-compute/inference/expectation-propagation`. Backward: the message to r̃ is split over the summands by the sum factor,
 * the product factor sends sₖ the message with mean m μₜ/(vₜ + μₜ²) and variance v/(vₜ + μₜ²) (the variational
 * message of Stern et al., 2009, §3), and the sum factors pass each weight its share. Each posterior is multiplied by
 * its message: ADF, one pass, no stored messages. The item trait weights start at small random means, which breaks
 * the symmetry of sₖtₖ (all zero means would leave every product message flat).
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
  readonly index: readonly number[]
  readonly value: readonly number[]
}

/** Factorised Gaussian posteriors, mean and variance per entry. */
export interface GaussianTable {
  readonly mean: Float64Array
  readonly variance: Float64Array
}

/** A Matchbox model: posteriors over trait weights U [K × user features], V [K × item features], biases, thresholds. */
export interface Matchbox {
  readonly traits: number
  readonly levels: number
  readonly beta: number
  readonly userFeatures: number
  readonly itemFeatures: number
  readonly U: GaussianTable
  readonly V: GaussianTable
  readonly userBias: GaussianTable
  readonly itemBias: GaussianTable
  readonly thresholds: GaussianTable
}

/** Options of `matchbox`. */
export interface MatchboxOptions {
  /** Traits K (default 2), rating levels L (default 5), latent noise β (default 1). */
  traits?: number
  levels?: number
  beta?: number
  /** Prior variance of trait weights (default 1), of bias weights (default 1), of thresholds (default 0.25). */
  traitVariance?: number
  biasVariance?: number
  thresholdVariance?: number
  /** The standard deviation of the item trait weights' random initial means (default 0.1). */
  symmetryBreaking?: number
}

const table = (n: number, variance: number, means?: Float64Array): GaussianTable => ({
  mean: means ?? new Float64Array(n),
  variance: new Float64Array(n).fill(variance),
})
const copy = (t: GaussianTable): GaussianTable => ({
  mean: Float64Array.from(t.mean),
  variance: Float64Array.from(t.variance),
})

/** A Matchbox model with prior posteriors; thresholds start evenly spaced, one unit apart, centred on 0. */
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

/** Mean and variance of Σᵢ xᵢ wᵢ for weights at `offset + i` of a table. */
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
 * Pass a message N(m, v) on Σᵢ xᵢ wᵢ back to every weight through the sum factor and multiply it into the posterior
 * (in place): weight i gets mean (m − Σⱼ≠ᵢ xⱼμⱼ)/xᵢ and variance (v + Σⱼ≠ᵢ xⱼ²vⱼ)/xᵢ².
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
  s: [number, number][]
  t: [number, number][]
  z: [number, number][]
  bu: [number, number]
  bi: [number, number]
  /** The latent affinity r̃ before the noise. */
  mean: number
  variance: number
}

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
  userTraits: Float64Array
  itemTraits: Float64Array
  latentMean: number
  latentVariance: number
  probabilities: Float64Array
  /** The expected rating level Σ l P(l). */
  expected: number
}

/** P(rating = l) = Φ((τₗ − μ)/σₗ) − Φ((τₗ₋₁ − μ)/σₗ₋₁), σ² = v + β² + v_τ, for the pair's latent affinity N(μ, v). */
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

/** One ADF update on a rating `level` of item y by user x (module docs). Returns a new model; `iterations` EP sweeps over the two thresholds (default 4). */
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
  users: number
  items: number
  levels: number
  /** Side features per user [users × userSide] and per item [items × itemSide], dense. */
  userSide: number
  itemSide: number
  userX: Float64Array
  itemY: Float64Array
  /** The ratings as (user, item, level) triples. */
  ratings: { user: Int32Array; item: Int32Array; level: Int32Array }
  /** The true traits [users × K], [items × K], the true affinities' scale and thresholds. */
  userTraits: Float64Array
  itemTraits: Float64Array
  thresholds: Float64Array
}

/** Options of `matchboxRatings`. */
export interface MatchboxRatingsOptions {
  users?: number
  items?: number
  traits?: number
  levels?: number
  /** Side features per user and item (default 3 each, standard normal); traits are linear in them plus noise. */
  userSide?: number
  itemSide?: number
  /** How much of each trait the side features explain (default 0.8); the rest is individual. */
  featureShare?: number
  /** Ratings per user (default 30), latent noise sd (default 0.5). */
  perUser?: number
  noise?: number
}

/**
 * Ratings on L levels from the model: user traits uₖ = Aₖ·f + e, item traits vₖ = Bₖ·g + e′ (side features f, g
 * standard normal), affinity Σₖ uₖvₖ plus noise, cut at thresholds at the affinity's quantiles (equal shares).
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

/** The sparse features of user u: a one-hot id (index u) and, with `side`, the side features after the ids. */
const features = (id: number, count: number, side: Float64Array, d: number, withSide: boolean): SparseFeatures => ({
  index: withSide ? [id, ...Array.from({ length: d }, (_, j) => count + j)] : [id],
  value: withSide ? [1, ...Array.from({ length: d }, (_, j) => side[id * d + j])] : [1],
})

/** Options of `matchboxRun`. */
export interface MatchboxRunOptions extends MatchboxOptions {
  /** Use the side features (default true); without them a new user or item is the prior. */
  useFeatures?: boolean
  /** Share of users held out entirely (cold start, default 0.1) and of the other ratings held out (default 0.2). */
  coldUsers?: number
  testShare?: number
  /** Passes over the training ratings (default 1: online ADF; more reuse ratings and overcount them). */
  passes?: number
  checkpoints?: number
  seed?: number | string
}

/** One checkpoint of a Matchbox run. */
export interface MatchboxCheckpoint {
  seen: number
  /** Held-out ratings of known users: RMSE of the expected level, accuracy of the most probable, mean log-probability. */
  rmse: number
  accuracy: number
  logProbability: number
  /** The same RMSE on the cold-start users. */
  coldRmse: number
  /** Posterior mean item traits [items × K] and their mean variance. */
  itemTraits: Float64Array
  traitVariance: number
  thresholds: Float64Array
}

/** A Matchbox run so far. */
export interface MatchboxRun {
  total: number
  seen: number
  finished: boolean
  levels: number
  traits: number
  /** Baselines: RMSE of each item's training mean level (known users) and of the global mean (cold users). */
  itemMeanRmse: number
  coldGlobalRmse: number
  checkpoints: MatchboxCheckpoint[]
  /** The test triples of known users and the final predicted distributions [n × L]. */
  test: { user: Int32Array; item: Int32Array; level: Int32Array }
  predictions: Float64Array
  /** True item traits [items × K] (for comparing up to rotation). */
  trueItemTraits: Float64Array
  /** The model's posteriors now. */
  model: Matchbox
}

/** Train Matchbox online on `matchboxRatings` data and yield checkpoints. Deterministic in `seed`. */
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
