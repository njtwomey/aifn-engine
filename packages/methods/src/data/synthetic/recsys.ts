/**
 * Seeded recommendation data: Zipf popularity catalogues, low-rank ratings matrices, and click logs under the
 * position-based and cascade click models.
 */

import { aliasSample, aliasTable, normal, type Stream, child, uniform } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkCount, labels, matrix, vector, type DatasetMeta } from '../types'
import type { RecommenderData } from 'aifn-methods/retrieval/recommenders'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'

/**
 * Zipf popularity weights p_i ∝ i^{−s} for items i = 1, …, n (normalised to sum to one): the long-tailed popularity of
 * catalogues, words and queries (Zipf, 1949, "Human Behavior and the Principle of Least Effort").
 */
export function zipfWeights(items: number, exponent = 1): Tensor {
  checkCount(items, 'zipfWeights')
  const w = Float64Array.from({ length: items }, (_, i) => (i + 1) ** -exponent)
  const total = w.reduce((a, b) => a + b, 0)
  return vector(w.map((v) => v / total))
}

/** A draw from a Zipf catalogue. */
export interface ZipfCatalogue {
  /** Popularity p_i of each item (item 0 is the most popular). */
  weights: Tensor
  /** Item index of each interaction, int32, length n. */
  draws: Tensor
  /** Interactions per item, int32, length `items`. */
  counts: Tensor
  meta: DatasetMeta
}

/** n interactions with a catalogue of `items` whose popularity follows Zipf's law with the given exponent. */
export function zipfCatalogue(
  s: Stream,
  options: { items?: number; exponent?: number; n?: number } = {},
): ZipfCatalogue {
  const { items = 1000, exponent = 1, n = 10000 } = options
  checkCount(n, 'zipfCatalogue')
  const weights = zipfWeights(items, exponent)
  const draws = aliasSample(s, aliasTable(weights), { shape: [n] })
  const counts = new Int32Array(items)
  for (let i = 0; i < n; i++) counts[draws.data[i]]++
  return {
    weights,
    draws,
    counts: labels(counts),
    meta: {
      name: 'Zipf catalogue',
      description: `${n} interactions with ${items} items whose popularity falls as rank^−${exponent}.`,
      task: 'recommendation',
      featureNames: ['item'],
      source: 'Zipf (1949), Human Behavior and the Principle of Least Effort',
      key: s.key,
    },
  }
}

/** A synthetic ratings problem. */
export interface Ratings {
  users: number
  items: number
  /** Every user's rating of every item (users × items), including entries no one observed. */
  truth: Tensor
  /** The noise-free low-rank score before rounding (users × items). */
  score: Tensor
  /** Observed training entries as rows (user, item, rating), m × 3. */
  train: Tensor
  /** Observed held-out entries, as rows (user, item, rating). */
  test: Tensor
  /** Mean training rating. */
  mean: number
  /** The true user and item factors (users × rank, items × rank). */
  userFactors: Tensor
  itemFactors: Tensor
  meta: DatasetMeta
}

/** Options for `ratings`. */
export interface RatingsOptions {
  users?: number
  items?: number
  /** Rank of the taste model. Default 2. */
  rank?: number
  /** Fraction of entries observed. Default 0.45. */
  observed?: number
  /** Share of observed entries held out for testing. Default 0.25. */
  testShare?: number
  /** Noise standard deviation before rounding. Default 0.3. */
  noise?: number
  /** Ratings are rounded and clipped to this integer scale. Default [1, 5]. */
  scale?: readonly [number, number]
}

/**
 * A ratings matrix from a low-rank taste model: user and item factors w_u, v_i ~ N(0, I), score 3 + 0.9 w_uᵀv_i + ε,
 * rounded and clipped to the rating scale. A random fraction of entries is observed; of those, `testShare` are held
 * out. Rounding and clipping are part of the recipe (they are the data-generating process), not numerical guards.
 */
export function ratings(s: Stream, options: RatingsOptions = {}): Ratings {
  const { users = 12, items = 16, rank = 2, observed = 0.45, testShare = 0.25, noise = 0.3, scale = [1, 5] } = options
  const mid = (scale[0] + scale[1]) / 2
  const fs = child(s, 'factors')
  const W = Float64Array.from({ length: users * rank }, () => normal(fs))
  const V = Float64Array.from({ length: items * rank }, () => normal(fs))
  const score = new Float64Array(users * items)
  const truth = new Float64Array(users * items)
  const eps = child(s, 'noise')
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      let dot = 0
      for (let r = 0; r < rank; r++) dot += W[u * rank + r] * V[i * rank + r]
      score[u * items + i] = mid + 0.9 * dot
      truth[u * items + i] = Math.round(
        Math.min(scale[1], Math.max(scale[0], score[u * items + i] + noise * normal(eps))),
      )
    }
  const train: number[] = []
  const test: number[] = []
  const mask = child(s, 'mask')
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      const p = uniform(mask)
      if (p < observed * (1 - testShare)) train.push(u, i, truth[u * items + i])
      else if (p < observed) test.push(u, i, truth[u * items + i])
    }
  let mean = 0
  for (let k = 2; k < train.length; k += 3) mean += train[k]
  mean /= Math.max(1, train.length / 3)
  return {
    users,
    items,
    truth: matrix(truth, users, items),
    score: matrix(score, users, items),
    train: matrix(Float64Array.from(train), train.length / 3, 3),
    test: matrix(Float64Array.from(test), test.length / 3, 3),
    mean,
    userFactors: matrix(W, users, rank),
    itemFactors: matrix(V, items, rank),
    meta: {
      name: 'ratings',
      description: `${users} users rating ${items} items on a ${scale[0]}–${scale[1]} scale from a rank-${rank} taste model; ${Math.round(observed * 100)}% of ratings observed.`,
      task: 'recommendation',
      featureNames: ['user', 'item', 'rating'],
      key: s.key,
    },
  }
}

/** A click log: one row per (session, rank). */
export interface ClickLog {
  session: Tensor
  /** Rank in the shown list, 1-based. */
  rank: Tensor
  item: Tensor
  /** 1 if the user examined the result, else 0. */
  examined: Tensor
  clicked: Tensor
  /** P(examine | rank) under the model (for the cascade model, given the clicks above). */
  propensity: Tensor
  /** The true click probability given examination, per item. */
  relevance: Tensor
  meta: DatasetMeta
}

/** Options for `clickLog`. */
export interface ClickLogOptions {
  sessions?: number
  /** P(click | examined) per item; item 0 is the most relevant. Default ten items from 0.62 down to 0.15. */
  relevance?: readonly number[]
  /** `position`: examination π(k) = k^{−η} independent of relevance; `cascade`: users scan down and stop at a click. */
  model?: 'position' | 'cascade'
  /** Position-bias exponent η. Default 1. */
  eta?: number
  /** The logging ranker sorts items by relevance plus N(0, rankerNoise²), so rank is confounded with relevance. */
  rankerNoise?: number
}

/**
 * Simulated search sessions under a click model (Craswell, Zoeter, Taylor and Ramsey, 2008, "An experimental comparison
 * of click position-bias models", WSDM): each session ranks every item by noisy relevance; under the position-based
 * model the result at rank k is examined with probability k^{−η}; under the cascade model the user reads from the top
 * and stops after the first click. A click needs examination and then happens with the item's relevance.
 */
export function clickLog(s: Stream, options: ClickLogOptions = {}): ClickLog {
  const {
    sessions = 200,
    relevance = [0.62, 0.55, 0.5, 0.42, 0.4, 0.33, 0.3, 0.24, 0.2, 0.15],
    model = 'position',
    eta = 1,
    rankerNoise = 0.1,
  } = options
  checkCount(sessions, 'clickLog')
  const k = relevance.length
  const rows = sessions * k
  const cols = {
    session: new Int32Array(rows),
    rank: new Int32Array(rows),
    item: new Int32Array(rows),
    examined: new Float64Array(rows),
    clicked: new Float64Array(rows),
    propensity: new Float64Array(rows),
  }
  for (let q = 0; q < sessions; q++) {
    const r = child(s, 'session', q)
    const order = relevance
      .map((rel, i) => ({ i, score: rel + rankerNoise * normal(r) }))
      .sort((a, b) => b.score - a.score)
    let reading = true
    order.forEach(({ i }, pos) => {
      const row = q * k + pos
      const exam = model === 'position' ? (pos + 1) ** -eta : reading ? 1 : 0
      const examined = uniform(r) < exam
      const clicked = examined && uniform(r) < relevance[i]
      if (model === 'cascade' && clicked) reading = false
      cols.session[row] = q
      cols.rank[row] = pos + 1
      cols.item[row] = i
      cols.examined[row] = examined ? 1 : 0
      cols.clicked[row] = clicked ? 1 : 0
      cols.propensity[row] = exam
    })
  }
  return {
    session: labels(cols.session),
    rank: labels(cols.rank),
    item: labels(cols.item),
    examined: fromData(cols.examined),
    clicked: fromData(cols.clicked),
    propensity: fromData(cols.propensity),
    relevance: vector(relevance),
    meta: {
      name: 'click log',
      description: `${sessions} sessions of ${k} ranked results under the ${model === 'position' ? `position-based model (η = ${eta})` : 'cascade model'}.`,
      task: 'recommendation',
      featureNames: ['session', 'rank', 'item', 'examined', 'clicked', 'propensity'],
      source: 'Craswell et al. (2008), WSDM',
      key: s.key,
    },
  }
}

// ── Implicit feedback with known structure ──────────────────────────────────────────────────────────────────────────

/** Implicit feedback from a known low-rank taste model under popularity-biased exposure. */
export interface ImplicitFeedback extends RecommenderData {
  /** Training rows (user, item) [m, 2] and the held-out rows: each user's last `testPerUser` interactions. */
  train: Tensor
  test: Tensor
  /** A group per user and a category per item (int32): the side features. */
  userGroup: Tensor
  itemCategory: Tensor
  /** Each user's training items in the order they were consumed. */
  sequences: number[][]
  /** The true affinity w_uᵀv_i of every user for every item (users × items). */
  preference: Tensor
  /** The exposure weight of each item (sums to 1): how often the platform shows it, regardless of taste. */
  exposure: Tensor
  /** The true user and item factors (users × rank, items × rank). */
  userFactors: Tensor
  itemFactors: Tensor
  meta: DatasetMeta
}

/** Options of `implicitFeedback`. */
export interface ImplicitFeedbackOptions {
  users?: number
  items?: number
  /** Rank of the taste model (default 3). */
  rank?: number
  /** Item categories and user groups; factors cluster around a centre per category or group (default 5 and 3). */
  categories?: number
  groups?: number
  /** Interactions per user, uniform between the two (default [12, 24]). */
  perUser?: readonly [number, number]
  /** Zipf exponent of exposure over a random order of the items: 0 shows every item equally (default 1). */
  exposureBias?: number
  /** Temperature τ of the taste term exp(w_uᵀv_i/τ) (default 0.5). */
  temperature?: number
  /** Log-weight added to items of the previous item's category: sequential structure (default 1.5). */
  stickiness?: number
  /** Held-out interactions per user, the last ones in time (default 2). */
  testPerUser?: number
}

/**
 * Implicit feedback with a known latent structure: user factors w_u (around a centre per user group) and item factors
 * v_i (around a centre per item category) of rank r; item exposure e_i ∝ rank_i^{−b} over a random order of the items
 * (popularity that has nothing to do with taste); and each user's interactions drawn one after another without
 * replacement with probability ∝ e_i · exp(w_uᵀv_i/τ + γ·1[c(i) = c(previous item)]). Exposure bias makes popular
 * items over-represented in the log; the stickiness γ gives the sequences an order that only a sequential model can
 * use. The last `testPerUser` interactions of each user are held out (leave-last-out).
 */
export function implicitFeedback(s: Stream, options: ImplicitFeedbackOptions = {}): ImplicitFeedback {
  const {
    users = 80,
    items = 100,
    rank = 3,
    categories = 5,
    groups = 3,
    perUser = [12, 24],
    exposureBias = 1,
    temperature = 0.5,
    stickiness = 1.5,
    testPerUser = 2,
  } = options
  checkCount(users, 'implicitFeedback')
  checkCount(items, 'implicitFeedback')
  const fs = child(s, 'factors')
  const centre = (n: number, tag: string) =>
    Array.from({ length: n }, (_, c) => Array.from({ length: rank }, (_, r) => 1.2 * normal(child(fs, tag, c, r))))
  const groupCentres = centre(groups, 'group')
  const categoryCentres = centre(categories, 'category')
  const userGroup = Int32Array.from({ length: users }, (_, u) => u % groups)
  const itemCategory = Int32Array.from({ length: items }, (_, i) => i % categories)
  const W = new Float64Array(users * rank)
  const V = new Float64Array(items * rank)
  for (let u = 0; u < users; u++)
    for (let r = 0; r < rank; r++)
      W[u * rank + r] = groupCentres[userGroup[u]][r] + 0.6 * normal(child(fs, 'user', u, r))
  for (let i = 0; i < items; i++)
    for (let r = 0; r < rank; r++)
      V[i * rank + r] = categoryCentres[itemCategory[i]][r] + 0.6 * normal(child(fs, 'item', i, r))
  const scale = 1 / Math.sqrt(rank)
  const preference = new Float64Array(users * items)
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      let dot = 0
      for (let r = 0; r < rank; r++) dot += W[u * rank + r] * V[i * rank + r]
      preference[u * items + i] = scale * dot
    }
  // Exposure: Zipf weights over a random order of the items.
  const order = Array.from({ length: items }, (_, i) => ({ i, key: uniform(child(s, 'order', i)) })).sort(
    (a, b) => a.key - b.key,
  )
  const exposure = new Float64Array(items)
  order.forEach(({ i }, r) => (exposure[i] = (r + 1) ** -exposureBias))
  const total = exposure.reduce((a, b) => a + b, 0)
  exposure.forEach((v, i) => (exposure[i] = v / total))
  const train: number[] = []
  const test: number[] = []
  const sequences: number[][] = []
  const [lo, hi] = perUser
  for (let u = 0; u < users; u++) {
    const us = child(s, 'user', u)
    const n = Math.min(items, lo + Math.floor(uniform(child(us, 'length')) * (hi - lo + 1)))
    const taken = new Set<number>()
    const seq: number[] = []
    let previous = -1
    for (let t = 0; t < n; t++) {
      const w = new Float64Array(items)
      for (let i = 0; i < items; i++) {
        if (taken.has(i)) continue
        const sticky = previous >= 0 && itemCategory[i] === itemCategory[previous] ? stickiness : 0
        w[i] = exposure[i] * Math.exp(preference[u * items + i] / temperature + sticky)
      }
      const j = aliasSample(child(us, 'step', t), aliasTable(vector(w)), { shape: [1] }).data[0]
      taken.add(j)
      seq.push(j)
      previous = j
    }
    const cut = Math.max(1, seq.length - testPerUser)
    seq.slice(0, cut).forEach((i) => train.push(u, i))
    seq.slice(cut).forEach((i) => test.push(u, i))
    sequences.push(seq.slice(0, cut))
  }
  const rows = (a: number[]) => fromData(Int32Array.from(a), [a.length / 2, 2])
  return {
    users,
    items,
    train: rows(train),
    test: rows(test),
    userGroup: labels(userGroup),
    itemCategory: labels(itemCategory),
    sequences,
    preference: matrix(preference, users, items),
    exposure: vector(exposure),
    userFactors: matrix(W, users, rank),
    itemFactors: matrix(V, items, rank),
    meta: {
      name: 'implicit feedback',
      description: `${users} users and ${items} items: a rank-${rank} taste model with ${categories} item categories and ${groups} user groups, interactions drawn under Zipf(${exposureBias}) exposure; the last ${testPerUser} per user held out.`,
      task: 'recommendation',
      featureNames: ['user', 'item'],
      key: s.key,
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'zipfCatalogue',
    name: 'Zipf catalogue',
    summary: 'Item popularity following a Zipf law, with interactions drawn from it.',
    task: 'recommendation',
    output: 'catalogue',
    knobs: space({
      items: int(1, 100000, { default: 1000 }),
      exponent: real(0, 3, { default: 1 }),
      n: int(1, 1000000, { default: 10000 }),
    }),
    truth: false,
    random: true,
  },
  zipfCatalogue,
)

dataset(
  {
    key: 'ratings',
    name: 'Low-rank ratings',
    summary: 'A partly observed user × item rating matrix from a low-rank taste model, rounded to a 1–5 scale.',
    task: 'recommendation',
    output: 'ratings',
    knobs: space({
      users: int(1, 500, { default: 12 }),
      items: int(1, 500, { default: 16 }),
      rank: int(1, 10, { default: 2 }),
      observed: real(0.01, 1, { default: 0.45 }),
      testShare: real(0, 0.9, { default: 0.25 }),
      noise: real(0, 3, { default: 0.3 }),
    }),
    truth: false,
    random: true,
    notes: ['alternating-least-squares'],
  },
  ratings,
)

dataset(
  {
    key: 'clickLog',
    name: 'Click log',
    summary: 'Logged rankings and clicks under a position-bias or cascade click model.',
    task: 'recommendation',
    output: 'clicks',
    knobs: space({
      sessions: int(1, 100000, { default: 200 }),
      model: oneOf(['position', 'cascade']),
      eta: real(0, 5, { default: 1 }),
      rankerNoise: real(0, 2, { default: 0.1 }),
    }),
    truth: false,
    random: true,
    notes: ['click-models'],
  },
  clickLog,
)

dataset(
  {
    key: 'implicitFeedback',
    name: 'Implicit feedback (low-rank, exposure-biased)',
    summary: 'Interaction sequences from a clustered low-rank taste model under Zipf exposure, last items held out.',
    task: 'recommendation',
    output: 'log',
    knobs: space({
      users: int(2, 1000, { default: 80 }),
      items: int(2, 1000, { default: 100 }),
      rank: int(1, 10, { default: 3 }),
      categories: int(1, 20, { default: 5 }),
      groups: int(1, 20, { default: 3 }),
      exposureBias: real(0, 3, { default: 1 }),
      temperature: real(0.05, 5, { default: 0.5 }),
      stickiness: real(0, 5, { default: 1.5 }),
      testPerUser: int(1, 10, { default: 2 }),
    }),
    truth: false,
    random: true,
    notes: ['explicit-and-implicit-feedback', 'popularity-bias', 'biases-in-recommender-feedback'],
  },
  implicitFeedback,
)
