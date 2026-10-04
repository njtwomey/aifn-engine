/**
 * Matrix factorisation for recommendation: a user × item matrix approximated by the inner products of user factors
 * p_u and item factors q_i in k dimensions. Explicit ratings are fitted by minibatch stochastic gradient descent on the
 * squared error with biases (Funk, 2006; Koren, Bell and Volinsky, 2009), or by alternating least squares with
 * weighted regularisation (Zhou et al., 2008); implicit feedback by the confidence-weighted alternating least squares of
 * Hu, Koren and Volinsky (2008). The alternating methods are step-through algorithms whose objective never increases.
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

/** User factors P [users, k] and item factors Q [items, k] as row-major arrays. */
export type Factors = { P: Float64Array; Q: Float64Array; k: Size; users: Size; items: Size }

/** Factors N(0, scale²) for every user and item, from `child(s, 'users')` and `child(s, 'items')`. */
export function randomFactors(s: Stream, users: Size, items: Size, k: Size, scale = 0.1): Factors {
  const P = Float64Array.from(toFlat(normal(child(s, 'users'), 0, scale, { shape: [users * k] })))
  const Q = Float64Array.from(toFlat(normal(child(s, 'items'), 0, scale, { shape: [items * k] })))
  return { P, Q, k, users, items }
}

/** The scorer of factors: score(u, i) = p_uᵀ q_i (plus biases when given). */
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
 * Solve the k × k ridge system (A + λI) x = b for every row of `target`, where A and b are built from the rows of
 * `fixed` an entity interacted with. The shared step of both alternating methods.
 */
function solveRow(A: Float64Array, b: Float64Array, k: Size): Float64Array {
  const r = solveDense(A, b, k)
  return r.x ? Float64Array.from(r.x) : new Float64Array(k)
}

// ── Explicit ALS ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `alternatingLeastSquares`. */
export type AlsOptions = {
  /** Factor dimension k (default 5). */
  factors?: Size
  /** Regularisation λ, scaled by each user's or item's number of ratings (ALS-WR; default 0.1). */
  regularisation?: number
  /** Initial factors P [users, k] and Q [items, k] (default N(0, 0.1²) from the init stream). */
  init?: { P: ArrayLike<number>; Q: ArrayLike<number> }
}

/** The state of the alternating methods. */
export interface AlsState extends Status {
  t: Size
  /** User factors [users, k] and item factors [items, k]. */
  P: Tensor
  Q: Tensor
  /** The objective after the step. */
  objective: number
  /** The objective after the user half of the step (it lies between the two full-step values). */
  half: number
}

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

/** One half-sweep of ALS-WR: refit every row of `target` given `fixed`, from the ratings grouped by the target's side. */
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
 * Σ_{(u,i) observed} (r_ui − p_uᵀq_i)² + λ(Σ_u n_u‖p_u‖² + Σ_i n_i‖q_i‖²). With Q fixed the objective is a ridge
 * regression in each p_u, solved exactly: p_u = (Q_uᵀQ_u + λn_u I)⁻¹ Q_uᵀ r_u over the items u rated; then the same for
 * every q_i. Each step (users, then items) cannot increase the objective. `init` takes a stream for the initial
 * factors N(0, 0.1²).
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
  /** Factor dimension k (default 8). */
  factors?: Size
  /** Regularisation λ (default 0.1). */
  regularisation?: number
  /** Confidence slope α: c_ui = 1 + α r_ui (default 10). */
  alpha?: number
  /** Initial factors (default N(0, 0.1²) from the init stream). */
  init?: { P: ArrayLike<number>; Q: ArrayLike<number> }
}

/**
 * The objective of implicit ALS: Σ_{u,i} c_ui (p_ui − x_uᵀy_i)² + λ(Σ‖x_u‖² + Σ‖y_i‖²) over every cell of the matrix,
 * with preference p_ui = 1[r_ui > 0] and confidence c_ui = 1 + α r_ui.
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

/** One half-sweep of implicit ALS: x_a = (YᵀY + Yᵀ(C_a − I)Y + λI)⁻¹ Yᵀ C_a p_a for every row a. */
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
 * every cell of the user × item matrix is a preference p_ui = 1[r_ui > 0] with confidence c_ui = 1 + α r_ui, so
 * unobserved cells count as weak negatives. With the item factors fixed, each user's factors solve a weighted ridge
 * regression exactly, and the YᵀY term is shared by all users, so a sweep costs O(nnz·k² + (users + items)k³). Each
 * step refits users, then items; the objective (`implicitAlsObjective`) never increases.
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

/** The factors of an alternating state. */
export function alsFactors(state: AlsState): Factors {
  const [users, k] = state.P.shape
  return { P: Float64Array.from(state.P.data), Q: Float64Array.from(state.Q.data), k, users, items: state.Q.shape[0] }
}

// ── Explicit MF by SGD ───────────────────────────────────────────────────────────────────────────────────────────────

/** Parameters of the biased factor model: global bias, user and item biases, user and item factors. */
export type BiasedFactorParams = { global: Tensor; userBias: Tensor; itemBias: Tensor; P: Tensor; Q: Tensor }

/** Options of `matrixFactorisationSgd`. */
export type MfSgdOptions = {
  factors?: Size
  /** L2 penalty λ on the factors and biases of each rating's user and item (default 0.02). */
  regularisation?: number
  /** SGD step size (default 0.05). */
  stepSize?: number
  /** Ratings per step (default 32). */
  batchSize?: Size
}

/** r̂_ui = μ + b_u + b_i + p_uᵀq_i for the rows (u, i) of a batch: the biased factor model. */
export function biasedFactorPredict(p: BiasedFactorParams, users: Tensor | number[], items: Tensor | number[]): Value {
  const dot = sum(mul(take(p.P, users), take(p.Q, items)), -1)
  return add(add(p.global, add(take(p.userBias, users), take(p.itemBias, items))), dot)
}

/**
 * The biased matrix-factorisation model of the Netflix prize (Funk, 2006; Koren, Bell and Volinsky, 2009) fitted by
 * minibatch SGD through `aifn-compute/nn/training`: each step takes the gradient of the mean squared error of a batch of
 * ratings, plus λ(b_u² + b_i² + ‖p_u‖² + ‖q_i‖²) for each rating in it, by automatic differentiation. `init` takes
 * `{ params }` (`biasedFactorInit`).
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

/** Initial parameters of the biased factor model: the mean rating as global bias, zero biases, factors N(0, 0.1²). */
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

/** The scorer of biased factor parameters. */
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
