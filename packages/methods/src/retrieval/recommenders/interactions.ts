/**
 * Interaction data and evaluation for the recommenders of `aifn-methods/retrieval/recommenders`: user–item
 * interactions as index arrays, the dense $\mathit{users} \times \mathit{items}$ matrix they imply, top-$k$ lists that
 * skip items already seen, and the held-out evaluation (recall@$k$, NDCG@$k$, hit rate, catalogue coverage) through
 * `aifn-compute/learning/metrics`. Users and items are 0-based indices, and every matrix is row-major with a row per
 * user.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { hitRate, ndcg, recallAtK } from 'aifn-compute/learning/metrics'

/** Interactions of `users` users with `items` items: parallel arrays of user and item indices (and optional values). */
export type Interactions = {
  /** The number of users; user indices run from 0 to `users - 1`. */
  readonly users: Size
  /** The number of items in the catalogue. */
  readonly items: Size
  /** The user of each interaction. */
  readonly user: Int32Array
  /** The item of each interaction, parallel to `user`. */
  readonly item: Int32Array
  /** Explicit ratings, or interaction strengths for implicit models (default 1 each). */
  readonly value?: Float64Array
}

/**
 * Interactions from rows `[m, 2]` (user, item) or `[m, 3]` (user, item, value).
 *
 * @param rows The interactions, one per row, as a tensor or an array of rows; a third column becomes `value`.
 * @param users The number of users, which the rows do not determine.
 * @param items The number of items in the catalogue.
 * @returns The interactions as parallel index arrays (with `value` only when the rows have a third column).
 *
 * @example Three interactions with ratings
 * const d = interactionsFromRows([[0, 0, 5], [0, 2, 3], [1, 1, 4]], 2, 3)
 * print('user:', d.user, ' item:', d.item, ' value:', d.value)
 */
export function interactionsFromRows(
  rows: Tensor | readonly (readonly number[])[],
  users: Size,
  items: Size,
): Interactions {
  const flat = Array.isArray(rows)
    ? { data: (rows as number[][]).flat(), cols: (rows as number[][])[0]?.length ?? 2 }
    : { data: Array.from(toFlat(rows as Tensor)), cols: (rows as Tensor).shape[1] ?? 2 }
  const m = flat.data.length / flat.cols
  const user = new Int32Array(m)
  const item = new Int32Array(m)
  const value = flat.cols > 2 ? new Float64Array(m) : undefined
  for (let r = 0; r < m; r++) {
    user[r] = flat.data[r * flat.cols]
    item[r] = flat.data[r * flat.cols + 1]
    if (value) value[r] = flat.data[r * flat.cols + 2]
  }
  return { users, items, user, item, value }
}

/**
 * The dense $\mathit{users} \times \mathit{items}$ matrix of the interactions (values summed; 1 per interaction
 * without values).
 *
 * @param d The interactions.
 * @returns The matrix, row-major, row `u` at entries `u * items` to `u * items + items - 1`.
 *
 * @example A repeated interaction counts twice
 * print(interactionMatrix(interactionsFromRows([[0, 0], [0, 2], [0, 2], [1, 1]], 2, 3)))
 */
export function interactionMatrix(d: Interactions): Float64Array {
  const R = new Float64Array(d.users * d.items)
  for (let r = 0; r < d.user.length; r++) R[d.user[r] * d.items + d.item[r]] += d.value ? d.value[r] : 1
  return R
}

/**
 * The items each user interacted with, as sets.
 *
 * @param d The interactions.
 * @returns One set per user (empty for a user with no interactions), in user order.
 *
 * @example Each user's items
 * const seen = itemsByUser(interactionsFromRows([[0, 0], [0, 2], [2, 1]], 3, 3))
 * print(seen.map((items) => [...items]))
 */
export function itemsByUser(d: Interactions): Set<number>[] {
  const out = Array.from({ length: d.users }, () => new Set<number>())
  for (let r = 0; r < d.user.length; r++) out[d.user[r]].add(d.item[r])
  return out
}

/**
 * A function from a list of users to their scores for every item: a row-major array of `users.length` rows of
 * `items` scores, higher meaning recommended sooner.
 */
export type Scorer = (users: readonly number[]) => Float64Array

/**
 * The indices of the $k$ highest scores, skipping `exclude` and any score that is not finite; ties keep the lower
 * index first.
 *
 * @param scores One user's scores, one per item.
 * @param k How many indices to return; fewer when fewer items remain.
 * @param exclude Items never returned, such as those the user has already seen.
 * @returns The indices, best first.
 *
 * @example The top two, then the top two of the items not yet seen
 * const scores = [0.1, 0.9, 0.5, 0.9]
 * print('top 2:', topK(scores, 2))
 * print('top 2 unseen:', topK(scores, 2, new Set([1])))
 */
export function topK(scores: ArrayLike<number>, k: Size, exclude?: ReadonlySet<number>): number[] {
  const ids: number[] = []
  for (let i = 0; i < scores.length; i++) if (!exclude?.has(i) && Number.isFinite(scores[i])) ids.push(i)
  ids.sort((a, b) => scores[b] - scores[a] || a - b)
  return ids.slice(0, k)
}

/** Held-out ranking quality at a cut-off $k$. */
export type RankingReport = {
  /** Mean recall@$k$ over users with a held-out item. */
  recall: number
  /** Mean NDCG@$k$ (binary gains). */
  ndcg: number
  /** Share of users with at least one held-out item in their top $k$. */
  hitRate: number
  /** Share of the catalogue that appears in some evaluated user's top $k$. */
  coverage: number
  /** The number of users evaluated. */
  users: Size
}

/**
 * Evaluate a scorer on held-out interactions: for every user with a test item, rank all items the user did not
 * interact with in training by score and measure recall@$k$, NDCG@$k$ and the hit rate with
 * `aifn-compute/learning/metrics` (the training items are removed from the ranking, as is usual for top-$k$
 * recommendation). With no user to evaluate, every measure is NaN.
 *
 * @param score The recommender, called once with every evaluated user.
 * @param train The training interactions, whose items are left out of each user's ranking; it fixes the catalogue.
 * @param test The held-out interactions, each a relevant item.
 * @param k The cut-off $k$.
 * @returns The mean recall, NDCG and hit rate over the users with a held-out item, the catalogue coverage and the
 *   number of users evaluated.
 *
 * @example Popularity on three users who each held out the most popular item they had not seen
 * const train = interactionsFromRows([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]], 3, 4)
 * const test = interactionsFromRows([[0, 1], [1, 2], [2, 0]], 3, 4)
 * print(evaluateRanking(popularity(train), train, test, 1))
 */
export function evaluateRanking(score: Scorer, train: Interactions, test: Interactions, k: Size = 10): RankingReport {
  const seen = itemsByUser(train)
  const held = itemsByUser(test)
  const users = held.flatMap((s, u) => (s.size > 0 ? [u] : []))
  if (users.length === 0) return { recall: NaN, ndcg: NaN, hitRate: NaN, coverage: NaN, users: 0 }
  const I = train.items
  const S = score(users)
  const relevance = new Float64Array(users.length * I)
  const ranked = new Float64Array(users.length * I)
  const shown = new Set<number>()
  users.forEach((u, r) => {
    for (const i of held[u]) relevance[r * I + i] = 1
    for (let i = 0; i < I; i++) ranked[r * I + i] = seen[u].has(i) ? -Infinity : S[r * I + i]
    for (const i of topK(ranked.subarray(r * I, (r + 1) * I), k)) shown.add(i)
  })
  const rel = fromData(relevance, [users.length, I])
  const sc = fromData(ranked, [users.length, I])
  return {
    recall: recallAtK(rel, sc, { k }),
    ndcg: ndcg(rel, sc, { k, gain: 'linear' }),
    hitRate: hitRate(rel, sc, { k }),
    coverage: shown.size / I,
    users: users.length,
  }
}
