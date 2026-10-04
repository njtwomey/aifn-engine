/**
 * Interaction data and evaluation for the recommenders of `aifn-methods/retrieval/recommenders`: user–item
 * interactions as index arrays, the dense user × item matrix they imply, top-k lists that skip items already seen,
 * and the held-out evaluation (recall@k, NDCG@k, hit rate, catalogue coverage) through `aifn-compute/learning/metrics`.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { hitRate, ndcg, recallAtK } from 'aifn-compute/learning/metrics'

/** Interactions of `users` users with `items` items: parallel arrays of user and item indices (and optional values). */
export type Interactions = {
  readonly users: Size
  readonly items: Size
  readonly user: Int32Array
  readonly item: Int32Array
  /** Explicit ratings, or interaction strengths for implicit models (default 1 each). */
  readonly value?: Float64Array
}

/** Interactions from rows [m, 2] (user, item) or [m, 3] (user, item, value). */
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

/** The dense user × item matrix of the interactions (values summed; 1 per interaction without values). */
export function interactionMatrix(d: Interactions): Float64Array {
  const R = new Float64Array(d.users * d.items)
  for (let r = 0; r < d.user.length; r++) R[d.user[r] * d.items + d.item[r]] += d.value ? d.value[r] : 1
  return R
}

/** The items each user interacted with, as sets. */
export function itemsByUser(d: Interactions): Set<number>[] {
  const out = Array.from({ length: d.users }, () => new Set<number>())
  for (let r = 0; r < d.user.length; r++) out[d.user[r]].add(d.item[r])
  return out
}

/** A function from a list of users to their scores for every item, [users.length, items]. */
export type Scorer = (users: readonly number[]) => Float64Array

/** The indices of the k highest scores, skipping `exclude`; ties keep the lower index first. */
export function topK(scores: ArrayLike<number>, k: Size, exclude?: ReadonlySet<number>): number[] {
  const ids: number[] = []
  for (let i = 0; i < scores.length; i++) if (!exclude?.has(i) && Number.isFinite(scores[i])) ids.push(i)
  ids.sort((a, b) => scores[b] - scores[a] || a - b)
  return ids.slice(0, k)
}

/** Held-out ranking quality at a cut-off k. */
export type RankingReport = {
  /** Mean recall@k over users with a held-out item. */
  recall: number
  /** Mean NDCG@k (binary gains). */
  ndcg: number
  /** Share of users with at least one held-out item in their top k. */
  hitRate: number
  /** Share of the catalogue that appears in some user's top k. */
  coverage: number
  /** The number of users evaluated. */
  users: Size
}

/**
 * Evaluate a scorer on held-out interactions: for every user with a test item, rank all items the user did not
 * interact with in training by score and measure recall@k, NDCG@k and the hit rate with `aifn-compute/learning/metrics`
 * (the training items are removed from the ranking, as is usual for top-k recommendation).
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
