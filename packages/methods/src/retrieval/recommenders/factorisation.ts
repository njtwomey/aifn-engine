/**
 * Matrix factorisation for recommendation: a $\mathit{users} \times \mathit{items}$ matrix approximated by the inner
 * products $\pvec_u^\top \qvec_i$ of user factors $\pvec_u$ and item factors $\qvec_i$ in $k$ dimensions. Explicit
 * ratings are fitted by minibatch stochastic gradient descent on the squared error with biases (Funk, 2006; Koren,
 * Bell and Volinsky, 2009), or by alternating least squares with weighted regularisation (Zhou et al., 2008); implicit
 * feedback by the confidence-weighted alternating least squares of Hu, Koren and Volinsky (2008), the model of the
 * `implicit` library's `AlternatingLeastSquares`. The alternating methods are step-through algorithms (run them with
 * `run`) whose objective never increases. Factor matrices are row-major, a row per user or item.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { child, normal, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  fromData,
  mean,
  mul,
  square,
  sum,
  take,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { solveDense } from 'aifn-compute/numerics/linalg'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { sgdRule } from 'aifn-compute/optim/first-order'
import { interactionMatrix, type Interactions, type Scorer } from './interactions'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * User factors $\Pmat$ (`users` rows of `k`) and item factors $\Qmat$ (`items` rows of `k`) as row-major arrays, with
 * the dimensions `k`, `users` and `items`.
 */
export type Factors = { P: Float64Array; Q: Float64Array; k: Size; users: Size; items: Size }

/**
 * Factors drawn from $\Gauss(0, \mathit{scale}^2)$ for every user and item, from `child(s, 'users')` and
 * `child(s, 'items')`.
 *
 * @param s The stream the factors are drawn from.
 * @param users The number of users.
 * @param items The number of items.
 * @param k The factor dimension $k$.
 * @param scale The standard deviation of every entry.
 * @returns The factors.
 *
 * @example Two users and three items in two dimensions
 * const f = randomFactors(stream(0), 2, 3, 2)
 * print('P =', f.P)
 * print('Q =', f.Q)
 */
export function randomFactors(s: Stream, users: Size, items: Size, k: Size, scale = 0.1): Factors {
  const P = Float64Array.from(toFlat(normal(child(s, 'users'), 0, scale, { shape: [users * k] })))
  const Q = Float64Array.from(toFlat(normal(child(s, 'items'), 0, scale, { shape: [items * k] })))
  return { P, Q, k, users, items }
}

/**
 * The scorer of factors: $\mathrm{score}(u, i) = \pvec_u^\top \qvec_i$, plus $\mu + b_u + b_i$ when biases are given.
 *
 * @param f The factors.
 * @param bias The global bias $\mu$ and the user and item biases $b_u$ and $b_i$; omitted, no bias is added.
 * @returns A scorer over every item.
 *
 * @example Scores of rank-one factors, with and without biases
 * const f = { P: Float64Array.of(1, -1), Q: Float64Array.of(2, 0, -2), k: 1, users: 2, items: 3 }
 * print('no bias:', factorScorer(f)([0, 1]))
 * print('bias:', factorScorer(f, { global: 3, user: Float64Array.of(0, 1), item: new Float64Array(3) })([0, 1]))
 */
export function factorScorer(f: Factors, bias?: { global: number; user: Float64Array; item: Float64Array }): Scorer {
  return (users) => {
    const out = new Float64Array(users.length * f.items)
    users.forEach((u, r) => {
      for (let i = 0; i < f.items; i++) {
        let s = bias ? bias.global + bias.user[u] + bias.item[i] : 0
        for (let c = 0; c < f.k; c++) s += f.P[u * f.k + c] * f.Q[i * f.k + c]
        out[r * f.items + i] = s
      }
    })
    return out
  }
}

/**
 * Solve one $k \times k$ system $\Amat\xvec = \bvec$, the ridge system of one user or item (the shared step of both
 * alternating methods). A singular system gives the zero vector.
 *
 * @param A The matrix $\Amat$, row-major, $k^2$ values, with the ridge penalty already on its diagonal.
 * @param b The right-hand side $\bvec$, $k$ values.
 * @param k The dimension $k$.
 * @returns The solution $\xvec$, $k$ values.
 */
function solveRow(A: Float64Array, b: Float64Array, k: Size): Float64Array {
  const r = solveDense(A, b, k)
  return r.x ? Float64Array.from(r.x) : new Float64Array(k)
}

// ── Explicit ALS ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `alternatingLeastSquares`. */
export type AlsOptions = {
  /** Factor dimension $k$ (default 5). */
  factors?: Size
  /** Regularisation $\lambda$, scaled by each user's or item's number of ratings (ALS-WR; default 0.1). */
  regularisation?: number
  /**
   * Initial factors $\Pmat$ (`users` rows of $k$) and $\Qmat$ (`items` rows of $k$), row-major (default
   * $\Gauss(0, 0.1^2)$ from the init stream).
   */
  init?: { P: ArrayLike<number>; Q: ArrayLike<number> }
}

/** The state of the alternating methods. */
export interface AlsState extends Status {
  /** Sweeps taken so far. */
  t: Size
  /** User factors, `[users, k]`. */
  P: Tensor
  /** Item factors, `[items, k]`. */
  Q: Tensor
  /** The objective after the step. */
  objective: number
  /** The objective after the user half of the step (it lies between the two full-step values). */
  half: number
}

/**
 * The ALS-WR objective
 * $\sum_{(u,i)} (r_{ui} - \pvec_u^\top \qvec_i)^2 + \lambda (\sum_u n_u \lVert \pvec_u \rVert^2 + \sum_i n_i
 * \lVert \qvec_i \rVert^2)$, with $n_u$ and $n_i$ the numbers of ratings of user $u$ and item $i$.
 *
 * @param d The ratings, which must have values.
 * @param P The user factors, row-major, `users` rows of `k`.
 * @param Q The item factors, row-major, `items` rows of `k`.
 * @param k The factor dimension $k$.
 * @param lambda The regularisation $\lambda$.
 * @returns The objective.
 */
function alsObjective(d: Interactions, P: Float64Array, Q: Float64Array, k: Size, lambda: number): number {
  let loss = 0
  const nu = new Float64Array(d.users)
  const ni = new Float64Array(d.items)
  for (let r = 0; r < d.user.length; r++) {
    const u = d.user[r]
    const i = d.item[r]
    let s = 0
    for (let c = 0; c < k; c++) s += P[u * k + c] * Q[i * k + c]
    loss += (d.value![r] - s) ** 2
    nu[u]++
    ni[i]++
  }
  let reg = 0
  for (let u = 0; u < d.users; u++) for (let c = 0; c < k; c++) reg += nu[u] * P[u * k + c] ** 2
  for (let i = 0; i < d.items; i++) for (let c = 0; c < k; c++) reg += ni[i] * Q[i * k + c] ** 2
  return loss + lambda * reg
}

/**
 * One half-sweep of ALS-WR: refit the factors of every user (or every item) given the other side's, each by its ridge
 * regression $(\Fmat_a^\top \Fmat_a + \lambda n_a \Imat)^{-1} \Fmat_a^\top \rvec_a$ on the $n_a$ rows $\Fmat_a$ of
 * `fixed` it rated. An entity with no ratings gets zero factors.
 *
 * @param rows The ratings grouped by the side being refitted: for each entity, the indices of the other side it rated
 *   (`other`) and the ratings (`value`).
 * @param fixed The other side's factors, row-major with `k` per row (not modified).
 * @param k The factor dimension $k$.
 * @param lambda The regularisation $\lambda$.
 * @returns The refitted factors, row-major, one row of `k` per entry of `rows`.
 */
function alsHalf(
  rows: { other: number[]; value: number[] }[],
  fixed: Float64Array,
  k: Size,
  lambda: number,
): Float64Array {
  const out = new Float64Array(rows.length * k)
  rows.forEach(({ other, value }, a) => {
    if (other.length === 0) return
    const A = new Float64Array(k * k)
    const b = new Float64Array(k)
    other.forEach((j, n) => {
      for (let c = 0; c < k; c++) {
        b[c] += value[n] * fixed[j * k + c]
        for (let e = 0; e < k; e++) A[c * k + e] += fixed[j * k + c] * fixed[j * k + e]
      }
    })
    for (let c = 0; c < k; c++) A[c * k + c] += lambda * other.length
    out.set(solveRow(A, b, k), a * k)
  })
  return out
}

/**
 * Alternating least squares with weighted regularisation for explicit ratings (Zhou et al., 2008): minimise
 * $\sum_{(u,i)\ \mathrm{observed}} (r_{ui} - \pvec_u^\top \qvec_i)^2 + \lambda (\sum_u n_u \lVert \pvec_u \rVert^2 +
 * \sum_i n_i \lVert \qvec_i \rVert^2)$. With $\Qmat$ fixed the objective is a ridge regression in each $\pvec_u$,
 * solved exactly: $\pvec_u = (\Qmat_u^\top \Qmat_u + \lambda n_u \Imat)^{-1} \Qmat_u^\top \rvec_u$ over the items $u$
 * rated; then the same for every $\qvec_i$. Each step (users, then items) cannot increase the objective. The
 * algorithm's `init` draws the initial factors $\Gauss(0, 0.1^2)$ from its stream unless `options.init` gives them.
 * Throws `DomainError` when the ratings have no values.
 *
 * @param ratings The observed ratings, with values.
 * @param options The factor dimension $k$, the regularisation $\lambda$ and the initial factors.
 * @returns The step-through algorithm, whose state holds the factors and the objective after each sweep.
 *
 * @example Recovering a rank-one ratings matrix from 22 of its 30 cells
 * const p = [1, 2, -1, 0.5, 1.5, -2]
 * const q = [1, -1, 2, 0.5, -0.5]
 * const rows = []
 * for (let u = 0; u < 6; u++) for (let i = 0; i < 5; i++) if ((u + i) % 4 !== 0) rows.push([u, i, p[u] * q[i]])
 * const alg = alternatingLeastSquares(interactionsFromRows(rows, 6, 5), { factors: 1, regularisation: 0.001 })
 * const start = run(alg, undefined, 0, { stream: stream(0) })
 * const end = run(alg, undefined, 20, { stream: stream(0) })
 * print('objective:', start.objective, 'to', end.objective)
 * print('held-out cell (2, 2): truth', p[2] * q[2], ', predicted', factorScorer(alsFactors(end))([2])[2])
 */
export function alternatingLeastSquares(ratings: Interactions, options: AlsOptions = {}): Algorithm<void, AlsState> {
  const { factors: k = 5, regularisation: lambda = 0.1 } = options
  if (!ratings.value)
    throw new DomainError('alternatingLeastSquares', 'alternatingLeastSquares: explicit ratings need values')
  const byUser = Array.from({ length: ratings.users }, () => ({ other: [] as number[], value: [] as number[] }))
  const byItem = Array.from({ length: ratings.items }, () => ({ other: [] as number[], value: [] as number[] }))
  for (let r = 0; r < ratings.user.length; r++) {
    byUser[ratings.user[r]].other.push(ratings.item[r])
    byUser[ratings.user[r]].value.push(ratings.value[r])
    byItem[ratings.item[r]].other.push(ratings.user[r])
    byItem[ratings.item[r]].value.push(ratings.value[r])
  }
  const { users, items } = ratings
  return {
    name: 'alternating-least-squares',
    init: (_start, s) => {
      const f = options.init
        ? { P: Float64Array.from(options.init.P), Q: Float64Array.from(options.init.Q) }
        : randomFactors(s, users, items, k)
      const objective = alsObjective(ratings, f.P, f.Q, k, lambda)
      return { t: 0, P: fromData(f.P, [users, k]), Q: fromData(f.Q, [items, k]), objective, half: objective }
    },
    step: (state) => {
      const P = alsHalf(byUser, Float64Array.from(state.Q.data), k, lambda)
      const half = alsObjective(ratings, P, Float64Array.from(state.Q.data), k, lambda)
      const Q = alsHalf(byItem, P, k, lambda)
      const objective = alsObjective(ratings, P, Q, k, lambda)
      return {
        t: state.t + 1,
        P: fromData(P, [users, k]),
        Q: fromData(Q, [items, k]),
        objective,
        half,
        diverged: !Number.isFinite(objective),
      }
    },
  }
}

// ── Implicit ALS ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `implicitAls`. */
export type ImplicitAlsOptions = {
  /** Factor dimension $k$ (default 8). */
  factors?: Size
  /** Regularisation $\lambda$ (default 0.1). */
  regularisation?: number
  /** Confidence slope $\alpha$: $c_{ui} = 1 + \alpha r_{ui}$ (default 10). */
  alpha?: number
  /** Initial factors, row-major as for `AlsOptions` (default $\Gauss(0, 0.1^2)$ from the init stream). */
  init?: { P: ArrayLike<number>; Q: ArrayLike<number> }
}

/**
 * The objective of implicit ALS,
 * $\sum_{u,i} c_{ui} (p_{ui} - \xvec_u^\top \yvec_i)^2 + \lambda (\sum_u \lVert \xvec_u \rVert^2 + \sum_i \lVert
 * \yvec_i \rVert^2)$ over every cell of the matrix, with preference $p_{ui} = \indicator[r_{ui} > 0]$ and confidence
 * $c_{ui} = 1 + \alpha r_{ui}$; $\xvec_u$ and $\yvec_i$ are the user and item factors.
 *
 * @param R The interaction matrix $r_{ui}$, row-major, `users` rows of `items` (as `interactionMatrix` gives it).
 * @param f The factors: the user factors $\xvec_u$ in `P` and the item factors $\yvec_i$ in `Q`.
 * @param alpha The confidence slope $\alpha$.
 * @param lambda The regularisation $\lambda$.
 * @returns The objective.
 *
 * @example Zero factors cost the confidence of every observed cell
 * const R = interactionMatrix(interactionsFromRows([[0, 0], [1, 1]], 2, 2))
 * const zero = { P: new Float64Array(2), Q: new Float64Array(2), k: 1, users: 2, items: 2 }
 * print('objective:', implicitAlsObjective(R, zero, 10, 0.1), ' 2 (1 + 10) =', 22)
 */
export function implicitAlsObjective(R: Float64Array, f: Factors, alpha: number, lambda: number): number {
  const { users, items, k, P, Q } = f
  let loss = 0
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      let s = 0
      for (let c = 0; c < k; c++) s += P[u * k + c] * Q[i * k + c]
      const r = R[u * items + i]
      loss += (1 + alpha * r) * ((r > 0 ? 1 : 0) - s) ** 2
    }
  let reg = 0
  for (const v of P) reg += v * v
  for (const v of Q) reg += v * v
  return loss + lambda * reg
}

/**
 * One half-sweep of implicit ALS:
 * $\xvec_a = (\Ymat^\top \Ymat + \Ymat^\top (\Cmat_a - \Imat) \Ymat + \lambda \Imat)^{-1} \Ymat^\top \Cmat_a \pvec_a$
 * for every row $a$, with $\Cmat_a$ the diagonal of row $a$'s confidences; only the positive entries of the row add to
 * $\Ymat^\top \Ymat$, which is computed once.
 *
 * @param R The interaction matrix of the side being refitted, row-major, `rows` rows of `cols` (users by items, or
 *   its transpose for the item half).
 * @param rows The number of rows of `R`, the entities refitted.
 * @param cols The number of columns of `R`, the rows of $\Ymat$.
 * @param Y The other side's factors $\Ymat$, row-major, `cols` rows of `k` (not modified).
 * @param k The factor dimension $k$.
 * @param alpha The confidence slope $\alpha$.
 * @param lambda The regularisation $\lambda$.
 * @returns The refitted factors, row-major, `rows` rows of `k`.
 */
function implicitHalf(
  R: Float64Array,
  rows: Size,
  cols: Size,
  Y: Float64Array,
  k: Size,
  alpha: number,
  lambda: number,
) {
  const YtY = new Float64Array(k * k)
  for (let j = 0; j < cols; j++)
    for (let c = 0; c < k; c++) for (let e = 0; e < k; e++) YtY[c * k + e] += Y[j * k + c] * Y[j * k + e]
  const out = new Float64Array(rows * k)
  for (let a = 0; a < rows; a++) {
    const A = Float64Array.from(YtY)
    const b = new Float64Array(k)
    for (let j = 0; j < cols; j++) {
      const r = R[a * cols + j]
      if (r <= 0) continue
      const conf = 1 + alpha * r
      for (let c = 0; c < k; c++) {
        b[c] += conf * Y[j * k + c]
        for (let e = 0; e < k; e++) A[c * k + e] += (conf - 1) * Y[j * k + c] * Y[j * k + e]
      }
    }
    for (let c = 0; c < k; c++) A[c * k + c] += lambda
    out.set(solveRow(A, b, k), a * k)
  }
  return out
}

/**
 * Weighted matrix factorisation for implicit feedback by alternating least squares (Hu, Koren and Volinsky, 2008):
 * every cell of the $\mathit{users} \times \mathit{items}$ matrix is a preference $p_{ui} = \indicator[r_{ui} > 0]$
 * with confidence $c_{ui} = 1 + \alpha r_{ui}$, so unobserved cells count as weak negatives. With the item factors
 * fixed, each user's factors solve a weighted ridge regression exactly, and the $\Ymat^\top \Ymat$ term is shared by
 * all users, so a sweep costs $O(\mathit{nnz} \, k^2 + (\mathit{users} + \mathit{items}) k^3)$ besides a pass over the
 * dense matrix. Each step refits users, then items; the objective (`implicitAlsObjective`) never increases. The
 * `implicit` library's `AlternatingLeastSquares` fits the same model.
 *
 * @param train The interactions; their values, or 1 each, are the strengths $r_{ui}$.
 * @param options The factor dimension $k$, the regularisation $\lambda$, the confidence slope $\alpha$ and the initial
 *   factors.
 * @returns The step-through algorithm, whose state holds the factors and the objective after each sweep.
 *
 * @example Two taste groups: each user's top unseen item is the one held out from its group
 * const rows = []
 * for (let u = 0; u < 6; u++) for (let i = 0; i < 3; i++) if (i !== u % 3) rows.push([u, (u < 3 ? 0 : 3) + i])
 * const train = interactionsFromRows(rows, 6, 6)
 * const end = run(implicitAls(train, { factors: 2 }), undefined, 10, { stream: stream(0) })
 * const scores = factorScorer(alsFactors(end))([0, 1, 2, 3, 4, 5])
 * const seen = itemsByUser(train)
 * print('top unseen:', [0, 1, 2, 3, 4, 5].map((u) => topK(scores.subarray(u * 6, u * 6 + 6), 1, seen[u])[0]))
 * print('objective after 10 sweeps:', end.objective)
 */
export function implicitAls(train: Interactions, options: ImplicitAlsOptions = {}): Algorithm<void, AlsState> {
  const { factors: k = 8, regularisation: lambda = 0.1, alpha = 10 } = options
  const { users, items } = train
  const R = interactionMatrix(train)
  const Rt = new Float64Array(items * users)
  for (let u = 0; u < users; u++) for (let i = 0; i < items; i++) Rt[i * users + u] = R[u * items + i]
  const objective = (P: Float64Array, Q: Float64Array) =>
    implicitAlsObjective(R, { P, Q, k, users, items }, alpha, lambda)
  return {
    name: 'implicit-als',
    init: (_start, s) => {
      const f = options.init
        ? { P: Float64Array.from(options.init.P), Q: Float64Array.from(options.init.Q) }
        : randomFactors(s, users, items, k)
      const value = objective(f.P, f.Q)
      return { t: 0, P: fromData(f.P, [users, k]), Q: fromData(f.Q, [items, k]), objective: value, half: value }
    },
    step: (state) => {
      const Q0 = Float64Array.from(state.Q.data)
      const P = implicitHalf(R, users, items, Q0, k, alpha, lambda)
      const half = objective(P, Q0)
      const Q = implicitHalf(Rt, items, users, P, k, alpha, lambda)
      const value = objective(P, Q)
      return {
        t: state.t + 1,
        P: fromData(P, [users, k]),
        Q: fromData(Q, [items, k]),
        objective: value,
        half,
        diverged: !Number.isFinite(value),
      }
    },
  }
}

/**
 * The factors of an alternating state, as plain arrays (copies of the state's tensors).
 *
 * @param state A state of `alternatingLeastSquares` or `implicitAls`.
 * @returns The factors, with the dimensions read from the state's shapes.
 *
 * @example The factors after one sweep of implicit ALS
 * const train = interactionsFromRows([[0, 0], [1, 1], [2, 1]], 3, 2)
 * const f = alsFactors(run(implicitAls(train, { factors: 1 }), undefined, 1, { stream: stream(0) }))
 * print('k =', f.k, ', users =', f.users, ', items =', f.items)
 * print('P =', f.P, ', Q =', f.Q)
 */
export function alsFactors(state: AlsState): Factors {
  const [users, k] = state.P.shape
  return { P: Float64Array.from(state.P.data), Q: Float64Array.from(state.Q.data), k, users, items: state.Q.shape[0] }
}

// ── Explicit MF by SGD ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Parameters of the biased factor model: the global bias $\mu$ (a scalar), the user and item biases (`[users]` and
 * `[items]`), and the user and item factors (`[users, k]` and `[items, k]`).
 */
export type BiasedFactorParams = { global: Tensor; userBias: Tensor; itemBias: Tensor; P: Tensor; Q: Tensor }

/** Options of `matrixFactorisationSgd`. */
export type MfSgdOptions = {
  /**
   * Not read by `matrixFactorisationSgd`: the factor dimension is that of the initial parameters
   * (`biasedFactorInit`'s `factors`).
   */
  factors?: Size
  /** L2 penalty $\lambda$ on the factors and biases of each rating's user and item (default 0.02). */
  regularisation?: number
  /** SGD step size (default 0.05). */
  stepSize?: number
  /** Ratings per step (default 32). */
  batchSize?: Size
}

/**
 * $\hat r_{ui} = \mu + b_u + b_i + \pvec_u^\top \qvec_i$ for the rows $(u, i)$ of a batch: the biased factor model
 * (differentiable in the parameters).
 *
 * @param p The parameters.
 * @param users The user of each row.
 * @param items The item of each row, parallel to `users`.
 * @returns The predicted ratings, one per row.
 *
 * @example Predictions at the start: the mean rating plus small factor products
 * const ratings = interactionsFromRows([[0, 0, 4], [0, 1, 2], [1, 1, 3]], 2, 2)
 * const p = biasedFactorInit(stream(0), ratings, 2)
 * print(biasedFactorPredict(p, [0, 0, 1], [0, 1, 1]))
 */
export function biasedFactorPredict(p: BiasedFactorParams, users: Tensor | number[], items: Tensor | number[]): Value {
  const dot = sum(mul(take(p.P, users), take(p.Q, items)), -1)
  return add(add(p.global, add(take(p.userBias, users), take(p.itemBias, items))), dot)
}

/**
 * The biased matrix-factorisation model of the Netflix prize (Funk, 2006; Koren, Bell and Volinsky, 2009) fitted by
 * minibatch SGD through `aifn-compute/nn/training`: each step takes the gradient of the mean squared error of a batch
 * of ratings plus $\lambda$ times the batch mean of
 * $b_u^2 + b_i^2 + \lVert \pvec_u \rVert^2 + \lVert \qvec_i \rVert^2$, by automatic differentiation. The algorithm's
 * `init` takes `{ params }` (`biasedFactorInit`). Throws `DomainError` when the ratings have no values.
 *
 * @param ratings The observed ratings, with values.
 * @param options The penalty $\lambda$, the step size and the batch size (at most the number of ratings).
 * @returns The training algorithm, whose state holds the parameters and the batch loss.
 *
 * @example Fitting a rank-one ratings matrix around a mean of 3
 * const p = [1, 2, -1, 0.5]
 * const q = [1, -1, 2, 0.5]
 * const rows = []
 * for (let u = 0; u < 4; u++) for (let i = 0; i < 4; i++) rows.push([u, i, 3 + p[u] * q[i]])
 * const ratings = interactionsFromRows(rows, 4, 4)
 * const alg = matrixFactorisationSgd(ratings, { regularisation: 0, stepSize: 0.1 })
 * const params = biasedFactorInit(stream(0), ratings, 1)
 * const start = run(alg, { params }, 0, { stream: stream(1) })
 * const end = run(alg, { params }, 150, { stream: stream(1) })
 * print('mean squared error:', start.loss, 'to', end.loss)
 * print('truth of user 1:', q.map((qi) => 3 + p[1] * qi))
 * print('fitted:', biasedFactorScorer(end.params)([1]))
 */
export function matrixFactorisationSgd(
  ratings: Interactions,
  options: MfSgdOptions = {},
): Algorithm<{ params: BiasedFactorParams }, TrainingState<BiasedFactorParams>> {
  const { regularisation: lambda = 0.02, stepSize = 0.05, batchSize = 32 } = options
  if (!ratings.value)
    throw new DomainError('matrixFactorisationSgd', 'matrixFactorisationSgd: explicit ratings need values')
  const data = {
    user: fromData(Float64Array.from(ratings.user), [ratings.user.length]),
    item: fromData(Float64Array.from(ratings.item), [ratings.item.length]),
    rating: fromData(Float64Array.from(ratings.value), [ratings.value.length]),
  }
  return trainingLoop<BiasedFactorParams, typeof data>({
    loss: (p, b) => {
      const u = Array.from(toFlat(b.user))
      const i = Array.from(toFlat(b.item))
      const err = mean(square(add(biasedFactorPredict(p, u, i), mul(-1, b.rating))))
      const reg = add(
        add(mean(sum(square(take(p.P, u)), -1)), mean(sum(square(take(p.Q, i)), -1))),
        add(mean(square(take(p.userBias, u))), mean(square(take(p.itemBias, i)))),
      )
      return add(err, mul(lambda, reg))
    },
    data,
    batchSize: Math.min(batchSize, ratings.user.length),
    optimizer: sgdRule({ stepSize }) as never,
  })
}

/**
 * Initial parameters of the biased factor model: the mean rating as global bias, zero biases, and factors drawn from
 * $\Gauss(0, 0.1^2)$ by `randomFactors`.
 *
 * @param s The stream the factors are drawn from.
 * @param ratings The ratings, which fix the numbers of users and items and the mean (0 without values).
 * @param factors The factor dimension $k$.
 * @returns The parameters, the `params` to start `matrixFactorisationSgd` from.
 *
 * @example The global bias starts at the mean rating
 * const p = biasedFactorInit(stream(0), interactionsFromRows([[0, 0, 4], [0, 1, 2], [1, 1, 3]], 2, 2), 2)
 * print('global:', p.global, ', user biases:', p.userBias, ', P:', p.P)
 */
export function biasedFactorInit(s: Stream, ratings: Interactions, factors = 5): BiasedFactorParams {
  const f = randomFactors(s, ratings.users, ratings.items, factors)
  let m = 0
  for (const v of ratings.value ?? []) m += v / ratings.user.length
  return {
    global: fromData(Float64Array.of(m), []),
    userBias: fromData(new Float64Array(ratings.users), [ratings.users]),
    itemBias: fromData(new Float64Array(ratings.items), [ratings.items]),
    P: fromData(f.P, [ratings.users, factors]),
    Q: fromData(f.Q, [ratings.items, factors]),
  }
}

/**
 * The scorer of biased factor parameters, $\mu + b_u + b_i + \pvec_u^\top \qvec_i$ for every item, through
 * `factorScorer`.
 *
 * @param p The parameters (read once, when the scorer is built).
 * @returns A scorer over every item.
 *
 * @example Every item's score for user 0, before training
 * const p = biasedFactorInit(stream(0), interactionsFromRows([[0, 0, 4], [0, 1, 2], [1, 1, 3]], 2, 2), 2)
 * print(biasedFactorScorer(p)([0]))
 */
export function biasedFactorScorer(p: BiasedFactorParams): Scorer {
  const [users, k] = p.P.shape
  const items = p.Q.shape[0]
  return factorScorer(
    { P: Float64Array.from(p.P.data), Q: Float64Array.from(p.Q.data), k, users, items },
    {
      global: toFlat(p.global)[0],
      user: Float64Array.from(p.userBias.data),
      item: Float64Array.from(p.itemBias.data),
    },
  )
}
