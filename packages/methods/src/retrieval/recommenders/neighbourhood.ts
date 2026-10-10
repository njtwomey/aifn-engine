/**
 * Recommenders without learned parameters: popularity, and user- and item-based neighbourhood collaborative filtering
 * (Resnick et al., 1994; Sarwar et al., 2001) with cosine similarity between interaction vectors. Neighbours are found
 * by `aifn-compute/numerics/neighbours`' exact search. Each returns a `Scorer`; the scores include the items a user
 * has already seen, which `topK` and `evaluateRanking` leave out.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'
import { interactionMatrix, type Interactions, type Scorer } from './interactions'

/**
 * The popularity recommender: every user gets the items ranked by their number of training interactions (their summed
 * values, when the interactions have values).
 *
 * @param train The training interactions.
 * @returns A scorer giving every user the same row of item counts.
 *
 * @example Item 1 is the most popular
 * const train = interactionsFromRows([[0, 1], [1, 1], [2, 1], [0, 0], [1, 2]], 3, 4)
 * print('scores:', popularity(train)([0]))
 * print("user 0's top 2 unseen:", topK(popularity(train)([0]), 2, itemsByUser(train)[0]))
 */
export function popularity(train: Interactions): Scorer {
  const counts = new Float64Array(train.items)
  for (let r = 0; r < train.item.length; r++) counts[train.item[r]] += train.value ? train.value[r] : 1
  return (users) => {
    const out = new Float64Array(users.length * train.items)
    users.forEach((_, r) => out.set(counts, r * train.items))
    return out
  }
}

/** Options of the neighbourhood recommenders. */
export type NeighbourhoodOptions = {
  /** Neighbours kept per user (user-kNN) or per item (item-kNN); default 20. */
  k?: Size
  /**
   * Shrink each similarity by $n/(n + \mathit{shrinkage})$, where $n$ is the overlap of the two vectors, the entries
   * positive in both (default 0: none).
   */
  shrinkage?: number
}

/**
 * Cosine similarities of each row of $\Rmat$ to its $k$ nearest other rows: neighbour ids and weights.
 *
 * @param R The matrix $\Rmat$, row-major, `rows` rows of `cols` entries (not modified).
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @param k The neighbours wanted per row; at most `rows - 1` are kept.
 * @param shrinkage The shrinkage of each similarity towards 0 by the overlap (0 for none).
 * @returns `ids` and `weights`, row-major with `k` entries per row (the neighbour ids, nearest first, and their
 *   similarities; 0 for an all-zero row), and the `k` actually kept.
 */
function nearestRows(R: Float64Array, rows: Size, cols: Size, k: Size, shrinkage: number) {
  const kk = Math.min(k, rows - 1)
  const nn = bruteForceNeighbours(fromData(R, [rows, cols]), fromData(R, [rows, cols]), kk, {
    metric: 'cosine',
    excludeSelf: true,
  })
  const ids = toFlat(nn.indices)
  const dist = toFlat(nn.distances)
  const weights = new Float64Array(rows * kk)
  for (let a = 0; a < rows; a++)
    for (let j = 0; j < kk; j++) {
      const b = ids[a * kk + j]
      // A row of zeros has no direction: its cosine distance is NaN, and it gets weight 0.
      const sim = Number.isFinite(dist[a * kk + j]) ? 1 - dist[a * kk + j] : 0
      let overlap = 0
      if (shrinkage > 0) for (let c = 0; c < cols; c++) if (R[a * cols + c] > 0 && R[b * cols + c] > 0) overlap++
      weights[a * kk + j] = shrinkage > 0 ? (sim * overlap) / (overlap + shrinkage) : sim
    }
  return { ids, weights, k: kk }
}

/**
 * User-based collaborative filtering: $\mathrm{score}(u, i) = \sum_{v \in N_k(u)} \mathrm{sim}(u, v) \, r_{vi}$, with
 * $N_k(u)$ the $k$ other users whose interaction vectors have the largest cosine similarity to $u$'s. The neighbours
 * are found once, when the scorer is built.
 *
 * @param train The training interactions; their matrix $r_{vi}$ is what neighbours contribute.
 * @param options The neighbours $k$ per user and the similarity shrinkage.
 * @returns A scorer over every item.
 *
 * @example User 0 is most like user 1, who also has item 2
 * const train = interactionsFromRows([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 3]], 3, 4)
 * const scores = userKnn(train, { k: 1 })([0])
 * print('scores:', scores)
 * print('top unseen:', topK(scores, 1, itemsByUser(train)[0]))
 */
export function userKnn(train: Interactions, options: NeighbourhoodOptions = {}): Scorer {
  const { k = 20, shrinkage = 0 } = options
  const R = interactionMatrix(train)
  const { users: U, items: I } = train
  const nn = nearestRows(R, U, I, k, shrinkage)
  return (users) => {
    const out = new Float64Array(users.length * I)
    users.forEach((u, r) => {
      for (let j = 0; j < nn.k; j++) {
        const v = nn.ids[u * nn.k + j]
        const w = nn.weights[u * nn.k + j]
        for (let i = 0; i < I; i++) out[r * I + i] += w * R[v * I + i]
      }
    })
    return out
  }
}

/**
 * Item-based collaborative filtering (Sarwar et al., 2001; Linden, Smith and York, 2003):
 * $\mathrm{score}(u, i) = \sum_{j \in I_u} \mathrm{sim}_k(i, j) \, r_{uj}$ over the items $I_u$ of user $u$, where
 * $\mathrm{sim}_k(i, j)$ is the cosine of the two items' user columns when $i$ is among $j$'s $k$ most similar items,
 * and 0 otherwise.
 *
 * @param train The training interactions.
 * @param options The neighbours $k$ kept per item and the similarity shrinkage.
 * @returns A scorer over every item.
 *
 * @example Items 0 and 2 are bought together, so a user with item 0 is offered item 2
 * const train = interactionsFromRows([[0, 0], [0, 2], [1, 0], [1, 2], [2, 1], [3, 0]], 4, 3)
 * const scores = itemKnn(train, { k: 1 })([3])
 * print('scores:', scores)
 * print('top unseen:', topK(scores, 1, itemsByUser(train)[3]))
 */
export function itemKnn(train: Interactions, options: NeighbourhoodOptions = {}): Scorer {
  const { k = 20, shrinkage = 0 } = options
  const R = interactionMatrix(train)
  const { users: U, items: I } = train
  const Rt = new Float64Array(I * U)
  for (let u = 0; u < U; u++) for (let i = 0; i < I; i++) Rt[i * U + u] = R[u * I + i]
  const nn = nearestRows(Rt, I, U, k, shrinkage)
  return (users) => {
    const out = new Float64Array(users.length * I)
    users.forEach((u, r) => {
      for (let j = 0; j < I; j++) {
        const ruj = R[u * I + j]
        if (ruj === 0) continue
        // Item j adds its similarity to each of its neighbours i.
        for (let q = 0; q < nn.k; q++) out[r * I + nn.ids[j * nn.k + q]] += ruj * nn.weights[j * nn.k + q]
      }
    })
    return out
  }
}
