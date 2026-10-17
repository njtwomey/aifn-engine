/**
 * Low-rank and hashing approximations of the softmax attention matrix
 * $\Pmat = \operatorname{softmax}(\beta \Qmat\Kmat^\top)$ (rows normalised over keys, $\beta$ the inverse
 * temperature), each written as an explicit $n \times n$ matrix so that it can be compared with $\Pmat$:
 *
 * - `lowRankSeries` with `lowRankAt`: the best rank-$k$ approximation $\Pmat_k$ of $\Pmat$ (truncated SVD, the
 *   Eckart–Young floor for any rank-$k$ method), from one SVD;
 * - `linformerMatrix`: Linformer's $\operatorname{softmax}(\beta \Qmat (\Emat\Kmat)^\top)\Fmat$ with
 *   $\Emat = \Fmat$ a $k \times n$ projection along the sequence, here mean pooling over $k$ segments
 *   (`segmentMeans`, one of the projections the Linformer paper lists), so that the result is an attention matrix;
 * - `nystromMatrix`: Nyströmformer's $\tilde{\Fmat}\,\operatorname{pinv}(\tilde{\Amat})\,\tilde{\Bmat}$ with
 *   $\tilde{\Fmat} = \operatorname{softmax}(\beta \Qmat\tilde{\Kmat}^\top)$,
 *   $\tilde{\Amat} = \operatorname{softmax}(\beta \tilde{\Qmat}\tilde{\Kmat}^\top)$ and
 *   $\tilde{\Bmat} = \operatorname{softmax}(\beta \tilde{\Qmat}\Kmat^\top)$, for $m$ landmarks $\tilde{\Qmat}$,
 *   $\tilde{\Kmat}$, the segment means of the queries and keys;
 * - `lshAttentionMatrix`: Reformer-style attention restricted to keys that share an angular-LSH bucket with the query
 *   in at least one of $r$ hashing rounds, $h(\xvec) = \operatorname{argmax}([\xvec\Rmat; -\xvec\Rmat])$ for a
 *   random $d \times b/2$ matrix $\Rmat$ (`angularBuckets`, `angularRotations`).
 *
 * `rowTotalVariation` measures the mean over queries of $\tfrac12 \sum_j \lvert \hat{P}_{ij} - P_{ij} \rvert$, the
 * share of attention weight misplaced; unlike `totalVariation` of `aifn-compute/probability/information` it does not
 * renormalise the rows first, which matters for Nyström rows (they can have negative entries and sum only to about 1).
 * `topicTokens` generates queries and keys that follow $c$ topics in runs, so that $\Pmat$ has a low-rank block
 * structure. The exact matrix $\Pmat$ is the `weights` of `scaledDotProductAttention` (`aifn-compute/nn/attention`)
 * with `scale` $\beta$.
 *
 * Sources: Wang et al. (2020), "Linformer", §4 and Theorem 1; Xiong et al. (2021), "Nyströmformer", §3 (segment-means
 * landmarks, eq. (14)); Kitaev, Kaiser and Levskaya (2020), "Reformer", §2 (angular LSH, multi-round hashing); Andoni
 * et al. (2015) for the random-rotation hash. Differences: Linformer learns $\Emat$ and $\Fmat$, here they are fixed
 * mean pooling; Nyströmformer approximates the pseudo-inverse iteratively, here it is exact (`pinv`, with a cutoff);
 * Reformer shares queries and keys, sorts by bucket and attends within chunks, here queries and keys are separate and
 * each query attends to every key in its buckets (the idealised pattern that chunking approximates). Plain arrays,
 * one head, for speed on a figure's slider (all curves at $n = 128$ take about 50 ms; every result is plain data, so a
 * figure may run them in the compute worker).
 */

import { normals, type Stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { pinv, svd } from 'aifn-compute/numerics/linalg'

/** A matrix as rows of numbers. */
type Matrix = number[][]
/** A matrix as rows of numbers, read only. */
type ReadMatrix = readonly (readonly number[])[]

/**
 * The product $\Amat\Bmat$ of two matrices given as rows, skipping zero entries of $\Amat$ (the projections are
 * sparse).
 *
 * @param a The left factor, $n \times p$.
 * @param b The right factor, $p \times m$.
 * @returns $\Amat\Bmat$, $n \times m$.
 */
function rowsProduct(a: ReadMatrix, b: ReadMatrix): Matrix {
  const n = a.length
  const m = b[0].length
  const inner = b.length
  const out = Array.from({ length: n }, () => new Array<number>(m).fill(0))
  for (let i = 0; i < n; i++) {
    const ai = a[i]
    const oi = out[i]
    for (let t = 0; t < inner; t++) {
      const v = ai[t]
      if (v === 0) continue
      const bt = b[t]
      for (let j = 0; j < m; j++) oi[j] += v * bt[j]
    }
  }
  return out
}

/**
 * The transpose of a matrix given as rows.
 *
 * @param a The matrix, $n \times m$.
 * @returns $\Amat^\top$, $m \times n$.
 */
function rowsTranspose(a: ReadMatrix): Matrix {
  return a[0].map((_, j) => a.map((r) => r[j]))
}

/**
 * The row-wise softmax of $\beta$ times the scores; entries that are $-\infty$ get weight 0.
 *
 * @param scores The scores, one row per query.
 * @param beta The inverse temperature $\beta$.
 * @returns The weights, rows summing to 1.
 */
function softmaxRows(scores: ReadMatrix, beta: number): Matrix {
  return scores.map((row) => {
    let top = Number.NEGATIVE_INFINITY
    for (const x of row) if (beta * x > top) top = beta * x
    const e = row.map((x) => (x === Number.NEGATIVE_INFINITY ? 0 : Math.exp(beta * x - top)))
    const z = e.reduce((s, v) => s + v, 0)
    return e.map((v) => v / z)
  })
}

/**
 * Queries and keys for $c$ topics in runs: token $i$ belongs to topic $\lfloor ic/n \rfloor$, and its query and key
 * are the topic's centre plus independent Gaussian noise of standard deviation `spread`, scaled to unit length. With
 * $c = 0$ they are pure noise (no structure). The centres, the query noise and the key noise are three blocks of
 * normals drawn from `s` in that order.
 *
 * @param s The random stream.
 * @param n The number of tokens $n$.
 * @param d The width $d$.
 * @param topics The number of topics $c \ge 0$.
 * @param spread The noise standard deviation relative to a centre's unit-variance entries.
 * @returns The queries `q` and keys `k`, $n \times d$ each with rows of unit length, and each token's `topic`.
 *
 * @example Eight tokens in two topics
 * const { q, topic } = topicTokens(stream(6), 8, 4, 2, 0.3)
 * print('topic:', topic)
 * print('first query:', q[0])
 */
export function topicTokens(
  s: Stream,
  n: number,
  d: number,
  topics: number,
  spread: number,
): { q: Matrix; k: Matrix; topic: number[] } {
  const centres = toRows(normals(s, [Math.max(1, topics), d]) as Tensor)
  const nq = toRows(normals(s, [n, d]) as Tensor)
  const nk = toRows(normals(s, [n, d]) as Tensor)
  const topic = Array.from({ length: n }, (_, i) => (topics > 0 ? Math.floor((i * topics) / n) : 0))
  const unit = (noise: number[], i: number) => {
    const v = noise.map((z, t) => (topics > 0 ? centres[topic[i]][t] : 0) + spread * z)
    const r = Math.hypot(...v)
    return v.map((x) => x / r)
  }
  return { q: nq.map(unit), k: nk.map(unit), topic }
}

/** The SVD of a matrix as plain rows: $\Pmat = \Umat \operatorname{diag}(\sigmab) \Vmat^\top$. */
export type LowRankSeries = {
  /** The singular values $\sigma_1 \ge \sigma_2 \ge \dots$. */
  singularValues: number[]
  /** The left singular vectors $\Umat$, as rows (column $t$ is $\uvec_t$). */
  u: Matrix
  /** The right singular vectors $\Vmat$, as rows (column $t$ is $\vvec_t$). */
  v: Matrix
}

/**
 * The singular values of $\Pmat$ and the factors of its best rank-$k$ approximations, from one SVD; `lowRankAt`
 * builds $\Pmat_k$. The result is plain data, so it survives the compute worker.
 *
 * @param p The matrix to approximate, as rows.
 * @returns The singular values (descending) and the singular vectors.
 *
 * @example A constant attention matrix has rank one
 * print('singular values:', lowRankSeries([[0.5, 0.5], [0.5, 0.5]]).singularValues)
 */
export function lowRankSeries(p: ReadMatrix): LowRankSeries {
  const { U, S, V } = svd(tensor(p as number[][]))
  return { singularValues: toFlat(S as Tensor), u: toRows(U as Tensor), v: toRows(V as Tensor) }
}

/**
 * The best rank-$k$ approximation $\Pmat_k = \sum_{t \le k} \sigma_t \uvec_t \vvec_t^\top$ (Eckart–Young) from
 * the factors of `lowRankSeries`.
 *
 * @param series The SVD from `lowRankSeries`.
 * @param k The rank $k$ (capped at the number of singular values).
 * @returns $\Pmat_k$, as rows.
 *
 * @example Rank one of a 2 by 2 attention matrix
 * const s = lowRankSeries([[0.9, 0.1], [0.2, 0.8]])
 * print('P_1 =', lowRankAt(s, 1))
 * print('P_2 =', lowRankAt(s, 2))
 */
export function lowRankAt(series: LowRankSeries, k: number): Matrix {
  const { u, v, singularValues: sv } = series
  const r = Math.min(k, sv.length)
  return u.map((ui) =>
    v.map((vj) => {
      let x = 0
      for (let t = 0; t < r; t++) x += ui[t] * sv[t] * vj[t]
      return x
    }),
  )
}

/**
 * The $k \times n$ matrix that averages $n$ positions over $k$ consecutive segments of near-equal length.
 *
 * @param n The number of positions $n$.
 * @param k The number of segments $k$.
 * @returns The averaging matrix, as rows: row $a$ holds $1/\lvert \text{segment } a \rvert$ on its segment.
 *
 * @example Six positions in three segments
 * print('E =', segmentMeans(6, 3))
 */
export function segmentMeans(n: number, k: number): Matrix {
  return Array.from({ length: k }, (_, a) => {
    const lo = Math.round((a * n) / k)
    const hi = Math.round(((a + 1) * n) / k)
    return Array.from({ length: n }, (_, j) => (j >= lo && j < hi ? 1 / (hi - lo) : 0))
  })
}

/**
 * Linformer's attention as an $n \times n$ matrix: $\operatorname{softmax}(\beta \Qmat(\Emat\Kmat)^\top)\Emat$,
 * with $\Emat = \Fmat$ the $k \times n$ projection `e`.
 *
 * @param q The queries, $n \times d$.
 * @param k The keys, $n \times d$.
 * @param beta The inverse temperature $\beta$.
 * @param e The $k \times n$ projection along the sequence (such as `segmentMeans`).
 * @returns $\hat{\Pmat}$, $n \times n$.
 *
 * @example Fewer segments misplace more weight (one segment per token is exact attention)
 * const { q, k } = topicTokens(stream(6), 8, 4, 2, 0.5)
 * const P = linformerMatrix(q, k, 4, segmentMeans(8, 8))
 * for (const segments of [1, 2, 4]) {
 *   print(segments, 'segments:', rowTotalVariation(linformerMatrix(q, k, 4, segmentMeans(8, segments)), P))
 * }
 */
export function linformerMatrix(q: ReadMatrix, k: ReadMatrix, beta: number, e: ReadMatrix): Matrix {
  const projected = rowsProduct(e, k)
  return rowsProduct(softmaxRows(rowsProduct(q, rowsTranspose(projected)), beta), e)
}

/**
 * Nyströmformer's attention as an $n \times n$ matrix, with segment-mean landmarks of the queries and keys.
 *
 * @param q The queries, $n \times d$.
 * @param k The keys, $n \times d$.
 * @param beta The inverse temperature $\beta$.
 * @param landmarks The number of landmarks $m$ (at most $n$).
 * @param rtol Singular values of $\tilde{\Amat}$ below `rtol` times the largest are dropped from its pseudo-inverse
 *   (default 0.03). $\tilde{\Amat}$ is often ill-conditioned; Nyströmformer's iterative pseudo-inverse has the same
 *   regularising effect.
 * @returns $\hat{\Pmat} = \tilde{\Fmat}\tilde{\Amat}^{+}\tilde{\Bmat}$, $n \times n$. Its rows sum to about 1 but
 *   can have negative entries.
 *
 * @example Two landmarks for two topics
 * const { q, k } = topicTokens(stream(6), 8, 4, 2, 0.2)
 * const P = linformerMatrix(q, k, 4, segmentMeans(8, 8))
 * print('misplaced weight:', rowTotalVariation(nystromMatrix(q, k, 4, 2), P))
 */
export function nystromMatrix(q: ReadMatrix, k: ReadMatrix, beta: number, landmarks: number, rtol = 0.03): Matrix {
  const seg = segmentMeans(q.length, landmarks)
  const ql = rowsProduct(seg, q)
  const kl = rowsProduct(seg, k)
  const f = softmaxRows(rowsProduct(q, rowsTranspose(kl)), beta)
  const a = softmaxRows(rowsProduct(ql, rowsTranspose(kl)), beta)
  const b = softmaxRows(rowsProduct(ql, rowsTranspose(k)), beta)
  const aplus = toRows(pinv(tensor(a), { rtol }))
  return rowsProduct(rowsProduct(f, aplus), b)
}

/**
 * Angular LSH buckets: $h(\xvec) = \operatorname{argmax}([\xvec\Rmat; -\xvec\Rmat])$ for a $d \times b/2$
 * matrix $\Rmat$ of standard normals, one per round.
 *
 * @param xs The vectors to hash, $n \times d$ (any number of sets hashed with the same $\Rmat$: pass them
 *   concatenated).
 * @param rotations The $d \times b/2$ matrices $\Rmat$, one per round.
 * @returns The bucket of every vector in every round, rounds by $n$, values in $0, \dots, b - 1$.
 *
 * @example Opposite vectors land in opposite buckets
 * const R = angularRotations(stream(7), 2, 4, 1)
 * print('buckets:', angularBuckets([[1, 0], [-1, 0], [0, 1]], R))
 */
export function angularBuckets(xs: ReadMatrix, rotations: readonly ReadMatrix[]): number[][] {
  return rotations.map((r) =>
    rowsProduct(xs, r).map((row) => {
      let best = 0
      let top = Number.NEGATIVE_INFINITY
      row.forEach((v, t) => {
        if (v > top) [top, best] = [v, t]
        if (-v > top) [top, best] = [-v, t + row.length]
      })
      return best
    }),
  )
}

/**
 * Random $d \times b/2$ rotations for `angularBuckets`, one per round ($b$ even, at least 2), drawn from `s` round by
 * round.
 *
 * @param s The random stream.
 * @param d The width $d$.
 * @param buckets The number of buckets $b$.
 * @param rounds The number of hashing rounds $r$.
 * @returns The $r$ matrices, each $d \times b/2$, as rows.
 *
 * @example Two rounds of four buckets in two dimensions
 * print('R =', angularRotations(stream(7), 2, 4, 2))
 */
export function angularRotations(s: Stream, d: number, buckets: number, rounds: number): Matrix[] {
  return Array.from({ length: rounds }, () => toRows(normals(s, [d, Math.max(1, buckets / 2)]) as Tensor))
}

/**
 * LSH attention as an $n \times n$ matrix: query $i$ attends, with the softmax restricted to them, to the keys that
 * share its bucket in at least one round. A query whose buckets hold no key attends to the key at its own position.
 *
 * @param q The queries, $n \times d$.
 * @param k The keys, $n \times d$.
 * @param beta The inverse temperature $\beta$.
 * @param rotations The hashing rounds, from `angularRotations`.
 * @returns The weights $\hat{\Pmat}$ and the mean number of keys each query scores.
 *
 * @example More rounds score more keys
 * const { q, k } = topicTokens(stream(6), 16, 4, 2, 0.3)
 * for (const rounds of [1, 4]) {
 *   const lsh = lshAttentionMatrix(q, k, 4, angularRotations(stream(7), 4, 4, rounds))
 *   print(rounds, 'rounds:', lsh.keysPerQuery, 'keys per query')
 * }
 */
export function lshAttentionMatrix(
  q: ReadMatrix,
  k: ReadMatrix,
  beta: number,
  rotations: readonly ReadMatrix[],
): { weights: Matrix; keysPerQuery: number } {
  const n = q.length
  const hq = angularBuckets(q, rotations)
  const hk = angularBuckets(k, rotations)
  const scores = rowsProduct(q, rowsTranspose(k))
  let total = 0
  const masked = scores.map((row, i) => {
    const out = row.map((x, j) => (hq.some((h, r) => h[i] === hk[r][j]) ? x : Number.NEGATIVE_INFINITY))
    if (out.every((x) => x === Number.NEGATIVE_INFINITY)) out[i] = row[i]
    total += out.filter((x) => x !== Number.NEGATIVE_INFINITY).length
    return out
  })
  return { weights: softmaxRows(masked, beta), keysPerQuery: total / n }
}

/**
 * The mean over rows of the total variation $\tfrac12 \sum_j \lvert \hat{P}_{ij} - P_{ij} \rvert$: the share of
 * each query's attention weight that the approximation puts in the wrong place (0 for exact; at most 1 when both rows
 * are distributions). Rows are not renormalised.
 *
 * @param approx The approximation $\hat{\Pmat}$, as rows.
 * @param exact The exact $\Pmat$, as rows.
 * @returns The mean total variation over rows.
 *
 * @example Half a row's weight moved
 * print('TV =', rowTotalVariation([[1, 0], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]))
 */
export function rowTotalVariation(approx: ReadMatrix, exact: ReadMatrix): number {
  let s = 0
  exact.forEach((row, i) => {
    let r = 0
    row.forEach((p, j) => (r += Math.abs(approx[i][j] - p)))
    s += r / 2
  })
  return s / exact.length
}
