/**
 * Recommenders without learned parameters: popularity, and user- and item-based neighbourhood collaborative filtering
 * (Resnick et al., 1994; Sarwar et al., 2001) with cosine similarity between interaction vectors. Neighbours are found
 * by `aifn-compute/numerics/neighbours`' exact search.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'
import { interactionMatrix, type Interactions, type Scorer } from './interactions'

/** The popularity recommender: every user gets the items ranked by their number of training interactions. */
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
  /** Shrink each similarity by n/(n + shrinkage), where n is the overlap of the two vectors (default 0: none). */
  shrinkage?: number
}

/** Cosine similarities of each row of R (rows × cols) to its k nearest other rows: neighbour ids and weights. */
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
 * User-based collaborative filtering: score(u, i) = Σ_{v ∈ N_k(u)} sim(u, v) r_vi, with N_k(u) the k users whose
 * interaction vectors have the largest cosine similarity to u's.
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
 * Item-based collaborative filtering (Sarwar et al., 2001; Linden, Smith and York, 2003): score(u, i) =
 * Σ_{j ∈ I_u} sim_k(i, j) r_uj, where sim_k keeps each item's k most similar items by the cosine of their user
 * columns and is 0 elsewhere.
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
