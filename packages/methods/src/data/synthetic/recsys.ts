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
 * Zipf popularity weights $p_i \propto i^{-s}$ for items $i = 1, \dots, n$ (normalised to sum to one): the
 * long-tailed popularity of catalogues, words and queries (Zipf, 1949, "Human Behavior and the Principle of Least
 * Effort"). Throws `DomainError` unless `items` is a non-negative integer.
 *
 * @param items The number of items $n$.
 * @param exponent The exponent $s$: 0 gives equal weights, larger values a steeper fall.
 * @returns The $n$ weights, the most popular item first.
 *
 * @example Each weight is the first divided by the rank
 * const w = zipfWeights(5)
 * print('weights:', w)
 * print('first / second:', toArray(w)[0] / toArray(w)[1])
 * print('exponent 2, first / second:', toArray(zipfWeights(5, 2))[0] / toArray(zipfWeights(5, 2))[1])
 */
export function zipfWeights(items: number, exponent = 1): Tensor {
  checkCount(items, 'zipfWeights')
  const w = Float64Array.from({ length: items }, (_, i) => (i + 1) ** -exponent)
  const total = w.reduce((a, b) => a + b, 0)
  return vector(w.map((v) => v / total))
}

/** A draw from a Zipf catalogue. */
export interface ZipfCatalogue {
  /** Popularity $p_i$ of each item (item 0 is the most popular), from `zipfWeights`. */
  weights: Tensor
  /** Item index of each interaction, int32, length n. */
  draws: Tensor
  /** Interactions per item, int32, length `items`. */
  counts: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/**
 * $n$ interactions with a catalogue of `items` whose popularity follows Zipf's law with the given exponent: each
 * interaction an independent draw of an item from the `zipfWeights`. Throws `DomainError` unless the counts are
 * non-negative integers.
 *
 * @param s The random stream the draws come from.
 * @param options `items`, the size of the catalogue (default 1000); `exponent`, the Zipf exponent $s$ (default 1);
 *   `n`, the number of interactions (default 10000).
 * @returns The weights, the item of each interaction, and the interactions per item.
 *
 * @example Counts follow the weights
 * const c = zipfCatalogue(stream(0), { items: 50, n: 2000 })
 * print('draws:', c.draws.shape, ' counts:', c.counts.shape, ' first draws:', toArray(c.draws).slice(0, 8))
 * print('counts of the top 4:', toArray(c.counts).slice(0, 4))
 * print('expected:          ', toArray(c.weights).slice(0, 4).map((w) => 2000 * w))
 */
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
  /** The number of users. */
  users: number
  /** The number of items. */
  items: number
  /** Every user's rating of every item (users $\times$ items), including entries no one observed. */
  truth: Tensor
  /** The noise-free low-rank score before rounding (users $\times$ items). */
  score: Tensor
  /** Observed training entries as rows (user, item, rating), $m \times 3$. */
  train: Tensor
  /** Observed held-out entries, as rows (user, item, rating). */
  test: Tensor
  /** Mean training rating (0 when no entry is in training). */
  mean: number
  /** The true user factors (users $\times$ rank). */
  userFactors: Tensor
  /** The true item factors (items $\times$ rank). */
  itemFactors: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/** Options for `ratings`. */
export interface RatingsOptions {
  /** Number of users. Default 12. */
  users?: number
  /** Number of items. Default 16. */
  items?: number
  /** Rank of the taste model. Default 2. */
  rank?: number
  /** Fraction of entries observed. Default 0.45. */
  observed?: number
  /** Share of observed entries held out for testing. Default 0.25. */
  testShare?: number
  /** Noise standard deviation before rounding. Default 0.3. */
  noise?: number
  /**
   * Ratings are clipped and rounded to this integer scale, `[lowest, highest]`, whose midpoint is the mean score.
   * Default `[1, 5]`.
   */
  scale?: readonly [number, number]
}

/**
 * A ratings matrix from a low-rank taste model: user and item factors $\wvec_u, \vvec_i \sim \Gauss(\zeros, \Imat)$,
 * score $m + 0.9\,\wvec_u^\top\vvec_i$ ($m$ the midpoint of the scale, 3 for 1 to 5), and rating the score plus
 * noise $\varepsilon \sim \Gauss(0, \sigma^2)$, clipped to the scale and rounded. Each entry is observed with
 * probability `observed`; of those, a share `testShare` is held out at random. Rounding and clipping are part of the
 * recipe (they are the data-generating process), not numerical guards.
 *
 * @param s The random stream: the factors come from its child `factors`, the noise from `noise`, the observed entries
 *   from `mask`.
 * @param options The sizes, rank, observed share, test share, noise and scale; see `RatingsOptions`.
 * @returns The full ratings and scores, the observed entries split into training and test rows, and the true factors.
 *
 * @example A small ratings matrix
 * const r = ratings(stream(0))
 * print('truth:', r.truth.shape, ' train:', r.train.shape, ' test:', r.test.shape)
 * print('first training rows (user, item, rating):', toArray(r.train).slice(0, 3))
 * print('mean training rating:', r.mean)
 * print('share observed:', (r.train.shape[0] + r.test.shape[0]) / (r.users * r.items), ' (asked 0.45)')
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
  /** The session of each row (int32), 0-based. */
  session: Tensor
  /** Rank in the shown list, 1-based. */
  rank: Tensor
  /** The item shown at that rank (int32). */
  item: Tensor
  /** 1 if the user examined the result, else 0. */
  examined: Tensor
  /** 1 if the user clicked the result, else 0. */
  clicked: Tensor
  /**
   * The probability of examination at that rank under the model: $k^{-\eta}$ for the position-based model, and for
   * the cascade model 1 until the session's first click and 0 after it.
   */
  propensity: Tensor
  /** The true click probability given examination, per item. */
  relevance: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/** Options for `clickLog`. */
export interface ClickLogOptions {
  /** Number of sessions. Default 200. */
  sessions?: number
  /**
   * The click probability given examination, per item; item 0 is the most relevant. Default ten items from 0.62 down
   * to 0.15.
   */
  relevance?: readonly number[]
  /**
   * `position` (default): examination $\pi(k) = k^{-\eta}$ at rank $k$, independent of relevance; `cascade`: users
   * scan down and stop at a click.
   */
  model?: 'position' | 'cascade'
  /** Position-bias exponent $\eta$. Default 1. */
  eta?: number
  /**
   * The logging ranker sorts items by relevance plus $\Gauss(0, \sigma^2)$ noise ($\sigma$ this value, default 0.1),
   * so rank is confounded with relevance.
   */
  rankerNoise?: number
}

/**
 * Simulated search sessions under a click model (Craswell, Zoeter, Taylor and Ramsey, 2008, "An experimental comparison
 * of click position-bias models", WSDM): each session ranks every item by noisy relevance; under the position-based
 * model the result at rank $k$ is examined with probability $k^{-\eta}$; under the cascade model the user reads from
 * the top and stops after the first click. A click needs examination and then happens with the item's relevance.
 * Throws `DomainError` unless `sessions` is a non-negative integer.
 *
 * @param s The random stream; session $q$ is drawn from its child `session` $q$.
 * @param options The number of sessions, relevances, click model, bias exponent and ranker noise; see
 *   `ClickLogOptions`.
 * @returns The log, one row per session and rank (sessions $\times$ items rows, by session then rank), with the
 *   examination propensities and the true relevances.
 *
 * @example Examination falls as 1 / rank
 * const log = clickLog(stream(0), { sessions: 400 })
 * print('rows:', log.rank.shape)
 * print('first ranks:', toArray(log.rank).slice(0, 4), ' their items:', toArray(log.item).slice(0, 4))
 * const rank = toArray(log.rank)
 * const examined = toArray(log.examined)
 * const rate = (k) => examined.filter((e, i) => rank[i] === k).reduce((a, e) => a + e, 0) / 400
 * print('examined at ranks 1, 2, 4:', [rate(1), rate(2), rate(4)])
 *
 * @example The cascade model stops at the first click
 * const log = clickLog(stream(0), { sessions: 100, model: 'cascade' })
 * const session = toArray(log.session)
 * const clicks = new Array(100).fill(0)
 * toArray(log.clicked).forEach((c, i) => (clicks[session[i]] += c))
 * print('most clicks in a session:', Math.max(...clicks))
 * print('sessions with a click:', clicks.filter((c) => c > 0).length, 'of 100')
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
  /** Training rows (user, item), int32, $m \times 2$: each user's interactions but the last `testPerUser`. */
  train: Tensor
  /** Held-out rows (user, item), int32: each user's last `testPerUser` interactions. */
  test: Tensor
  /** A group per user (int32), user $u$ in group $u \bmod g$: a side feature. */
  userGroup: Tensor
  /** A category per item (int32), item $i$ in category $i \bmod c$: a side feature. */
  itemCategory: Tensor
  /** Each user's training items in the order they were consumed. */
  sequences: number[][]
  /** The true affinity $\wvec_u^\top\vvec_i/\sqrt{r}$ of every user for every item (users $\times$ items). */
  preference: Tensor
  /** The exposure weight of each item (sums to 1): how often the platform shows it, regardless of taste. */
  exposure: Tensor
  /** The true user factors (users $\times$ rank). */
  userFactors: Tensor
  /** The true item factors (items $\times$ rank). */
  itemFactors: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/** Options of `implicitFeedback`. */
export interface ImplicitFeedbackOptions {
  /** Number of users (default 80). */
  users?: number
  /** Number of items (default 100). */
  items?: number
  /** Rank of the taste model (default 3). */
  rank?: number
  /** Item categories; item factors cluster around a centre per category (default 5). */
  categories?: number
  /** User groups; user factors cluster around a centre per group (default 3). */
  groups?: number
  /**
   * Interactions per user, an integer uniform between the two inclusive and at most the number of items (default
   * `[12, 24]`).
   */
  perUser?: readonly [number, number]
  /** Zipf exponent of exposure over a random order of the items: 0 shows every item equally (default 1). */
  exposureBias?: number
  /** Temperature $\tau$ of the taste term $\exp(\wvec_u^\top\vvec_i/(\sqrt{r}\,\tau))$ (default 0.5). */
  temperature?: number
  /** Log-weight added to items of the previous item's category: sequential structure (default 1.5). */
  stickiness?: number
  /** Held-out interactions per user, the last ones in time; every user keeps at least one for training (default 2). */
  testPerUser?: number
}

/**
 * Implicit feedback with a known latent structure: user factors $\wvec_u$ (around a centre per user group) and item
 * factors $\vvec_i$ (around a centre per item category) of rank $r$, each centre coordinate $\Gauss(0, 1.2^2)$ and
 * each factor its centre plus $\Gauss(0, 0.6^2)$ noise; item exposure $e_i \propto \rho_i^{-b}$, $\rho_i$ the item's
 * place in a random order of the items (popularity that has nothing to do with taste); and each user's interactions
 * drawn one after another without replacement with probability
 * $\propto e_i \exp(\wvec_u^\top\vvec_i/(\sqrt{r}\,\tau) + \gamma\,\indicator[c(i) = c(i')])$, $i'$ the
 * previous item and $c$ the category. Exposure bias makes popular items over-represented in the log; the stickiness
 * $\gamma$ gives the sequences an order that only a sequential model can use. The last `testPerUser` interactions of
 * each user are held out (leave-last-out). Throws `DomainError` unless `users` and `items` are non-negative integers.
 *
 * @param s The random stream: the factors come from its child `factors`, the exposure order from `order`, user $u$'s
 *   interactions from `user` $u$.
 * @param options The sizes, rank, clusters, interactions per user, exposure bias, temperature, stickiness and the
 *   held-out count; see `ImplicitFeedbackOptions`.
 * @returns The training and held-out interactions, the training sequences, the side features, and the true
 *   preferences, exposure and factors.
 *
 * @example Interactions and held-out items
 * const f = implicitFeedback(stream(0), { users: 30, items: 40 })
 * print('train:', f.train.shape, ' test:', f.test.shape, ' first rows (user, item):', toArray(f.train).slice(0, 3))
 * print('first user\'s training sequence:', f.sequences[0])
 *
 * @example Exposure bias over-represents the most exposed items
 * // The share of training interactions on the 4 items the platform shows most (10% of the catalogue).
 * const share = (f) => {
 *   const e = toArray(f.exposure)
 *   const top = new Set(e.map((_, i) => i).sort((a, b) => e[b] - e[a]).slice(0, 4))
 *   const items = toArray(f.train).map(([, i]) => i)
 *   return items.filter((i) => top.has(i)).length / items.length
 * }
 * print('exposure bias 2:', share(implicitFeedback(stream(0), { users: 30, items: 40, exposureBias: 2 })))
 * print('exposure bias 0:', share(implicitFeedback(stream(0), { users: 30, items: 40, exposureBias: 0 })))
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
