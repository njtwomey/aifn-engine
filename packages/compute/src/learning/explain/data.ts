/**
 * Data attribution: which training points a model's behaviour depends on.
 *
 * - `influenceFunctions` (Koh and Liang, 2017, after Cook and Weisberg, 1982): the effect of upweighting a training
 *   point z on the test loss, I(z, z_test) = −∇ℓ(z_test, θ̂)ᵀ H⁻¹ ∇ℓ(z, θ̂), with H the Hessian of the training
 *   objective at its minimiser θ̂. Removing z changes the test loss by about −I/n. H⁻¹v is solved exactly (the Hessian
 *   by autodiff, for small models) or estimated by LiSSA (Agarwal, Bullins and Hazan, 2017): the recursion
 *   h ← v + (1 − damping) h − H_B h / scale on sampled minibatches B, Hessian-vector products by autodiff, gives
 *   H⁻¹v ≈ h/scale.
 * - `tracIn` (Pruthi et al., 2020): the sum over saved checkpoints θ_t of η_t ∇ℓ(z, θ_t) · ∇ℓ(z′, θ_t), how much the
 *   gradient steps on z lowered the loss on z′. Self-influence (z′ = z) is large for points the model struggled to fit,
 *   such as mislabelled ones.
 * - `dataShapley` (Ghorbani and Zou, 2019): the Shapley value of each training point in the game whose value is a
 *   model's validation score when trained on a subset, estimated by truncated Monte Carlo over permutations (TMC): a
 *   permutation's scan stops once the score is within `tolerance` of the full data's.
 * - `knnShapley` (Jia et al., 2019): exact Shapley values for the k-nearest-neighbour utility
 *   v(S) = (1/K) Σ_{k ≤ min(K, |S|)} 1[y of S's k-th nearest = y_test], by a recursion over the training points sorted
 *   by distance to the test point, O(n log n) per test point; averaged over the test set.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { grad, hessian, hvp, vmap } from 'aifn-compute/foundation/autodiff'
import { child, integers, permutation, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  dense,
  fromData,
  mean,
  mul,
  square,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The loss of one example (x [d], target y) at parameters θ [p]. */
export type ExampleLoss = (theta: Tensor, x: Tensor, y: Tensor) => Value

/** A labelled set: rows x [n, d] and targets y [n]. */
export type Examples = { x: MatrixLike; y: VectorLike }

const asTensors = (e: Examples, where: string) => {
  const X = dense.toMatrixF64(e.x, where)
  const y = Float64Array.from(dense.toF64(e.y, where))
  if (y.length !== X.m) throw new ShapeError(where, `${where}: one target per row`)
  return { x: fromData(Float64Array.from(X.data), [X.m, X.n]), y: fromData(y, [X.m]), n: X.m }
}

/** Per-example gradients ∇_θ ℓ(θ, xᵢ, yᵢ) as rows [n, p]. */
export function exampleGradients(loss: ExampleLoss, theta: VectorLike, examples: Examples): Tensor {
  const t = fromData(Float64Array.from(dense.toF64(theta, 'exampleGradients')))
  const { x, y } = asTensors(examples, 'exampleGradients')
  return vmap(grad(loss), { inAxes: [null, 0, 0] })(t, x, y) as Tensor
}

/** The training objective (1/n) Σ ℓ(θ, xᵢ, yᵢ) + (λ/2)‖θ‖² over the given rows. */
function objective(loss: ExampleLoss, x: Tensor, y: Tensor, l2: number) {
  const batch = vmap(loss, { inAxes: [null, 0, 0] })
  return (theta: Tensor): Value => {
    const v = mean(batch(theta, x, y) as Tensor)
    return l2 > 0 ? add(v, mul(l2 / 2, sum(square(theta)))) : v
  }
}

/** Options of `influenceFunctions`. */
export type InfluenceOptions = {
  /** `exact` (default; the full Hessian) or `lissa`. */
  method?: 'exact' | 'lissa'
  /** The L2 term (λ/2)‖θ‖² of the training objective (default 0). */
  l2?: number
  /** LiSSA: damping (default 0.01), scale (default 10), recursion depth (default 200), repeats (default 1), batch size (default 32) and stream. */
  damping?: number
  scale?: number
  depth?: Size
  repeats?: Size
  batch?: Size
  stream?: Stream
}

/**
 * Influence of each training point on the total loss over `test` at θ̂ (see the module comment): `influence` [n] is
 * I(zᵢ, test) = −g_testᵀ H⁻¹ ∇ℓ(zᵢ), `removal` [n] the predicted change of the test loss when zᵢ is removed (−I/n),
 * `selfInfluence` [n] = ∇ℓ(zᵢ)ᵀ H⁻¹ ∇ℓ(zᵢ) (exact method only; NaN for LiSSA), and `inverseHvp` = H⁻¹ g_test.
 */
export function influenceFunctions(
  loss: ExampleLoss,
  theta: VectorLike,
  train: Examples,
  test: Examples,
  options: InfluenceOptions = {},
): { influence: Float64Array; removal: Float64Array; selfInfluence: Float64Array; inverseHvp: Float64Array } {
  const th = Float64Array.from(dense.toF64(theta, 'influenceFunctions'))
  const p = th.length
  const T = asTensors(train, 'influenceFunctions')
  const Q = asTensors(test, 'influenceFunctions')
  const { method = 'exact', l2 = 0 } = options
  const G = toFlat(exampleGradients(loss, th, train))
  const Gt = toFlat(exampleGradients(loss, th, test))
  const v = new Float64Array(p)
  for (let k = 0; k < Q.n; k++) for (let j = 0; j < p; j++) v[j] += Gt[k * p + j]
  const thetaT = fromData(th, [p])
  let s: Float64Array
  let selfInfluence = new Float64Array(T.n).fill(NaN)
  if (method === 'exact') {
    const H = hessian(objective(loss, T.x, T.y, l2))(thetaT) as Tensor
    s = Float64Array.from(toFlat(solve(H, fromData(v, [p])) as Tensor))
    // H⁻¹ Gᵀ, one solve with n right-hand sides.
    const HiG = toFlat(solve(H, fromData(transpose(G, T.n, p), [p, T.n])) as Tensor)
    selfInfluence = Float64Array.from({ length: T.n }, (_, i) => {
      let a = 0
      for (let j = 0; j < p; j++) a += G[i * p + j] * HiG[j * T.n + i]
      return a
    })
  } else {
    const { damping = 0.01, scale = 10, depth = 200, repeats = 1, batch = 32 } = options
    if (!options.stream) throw new DomainError('influenceFunctions', 'influenceFunctions: LiSSA needs a stream')
    const Xd = Float64Array.from(toFlat(T.x))
    const yd = toFlat(T.y)
    const d = T.x.shape[1]
    s = new Float64Array(p)
    for (let r = 0; r < repeats; r++) {
      let h = Float64Array.from(v)
      for (let j = 0; j < depth; j++) {
        const st = child(options.stream, 'lissa', r, j)
        const b = Math.min(batch, T.n)
        const rows = new Float64Array(b * d)
        const ys = new Float64Array(b)
        for (let k = 0; k < b; k++) {
          const i = integers(st, T.n) as number
          rows.set(Xd.subarray(i * d, (i + 1) * d), k * d)
          ys[k] = yd[i]
        }
        const f = objective(loss, fromData(rows, [b, d]), fromData(ys, [b]), l2)
        const Hh = toFlat(hvp(f, thetaT, fromData(Float64Array.from(h), [p])) as Tensor)
        h = Float64Array.from(h, (hv, k) => v[k] + (1 - damping) * hv - Hh[k] / scale)
      }
      for (let k = 0; k < p; k++) s[k] += h[k] / scale / repeats
    }
  }
  const influence = Float64Array.from({ length: T.n }, (_, i) => {
    let a = 0
    for (let j = 0; j < p; j++) a += s[j] * G[i * p + j]
    return -a
  })
  return { influence, removal: Float64Array.from(influence, (I) => -I / T.n), selfInfluence, inverseHvp: s }
}

function transpose(a: ArrayLike<number>, m: Size, n: Size): Float64Array {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j * m + i] = a[i * n + j]
  return out
}

/**
 * TracInCP influences of training points on test points: Σ_t η_t ∇ℓ(zᵢ, θ_t) · ∇ℓ(z′ⱼ, θ_t) [n, m], and the
 * self-influence Σ_t η_t ‖∇ℓ(zᵢ, θ_t)‖² [n], over `checkpoints` θ_t with step sizes `rates` η_t (one number for all).
 */
export function tracIn(
  loss: ExampleLoss,
  checkpoints: readonly VectorLike[],
  rates: number | readonly number[],
  train: Examples,
  test?: Examples,
): { influence: Tensor | null; selfInfluence: Float64Array } {
  if (checkpoints.length === 0) throw new DomainError('tracIn', 'tracIn: needs at least one checkpoint')
  const eta = typeof rates === 'number' ? checkpoints.map(() => rates) : rates
  if (eta.length !== checkpoints.length) throw new ShapeError('tracIn', 'tracIn: one rate per checkpoint')
  const n = dense.toMatrixF64(train.x, 'tracIn').m
  const m = test ? dense.toMatrixF64(test.x, 'tracIn').m : 0
  const self = new Float64Array(n)
  const cross = new Float64Array(n * m)
  checkpoints.forEach((th, t) => {
    const G = toFlat(exampleGradients(loss, th, train))
    const p = G.length / n
    for (let i = 0; i < n; i++) {
      let a = 0
      for (let j = 0; j < p; j++) a += G[i * p + j] ** 2
      self[i] += eta[t] * a
    }
    if (test) {
      const Gt = toFlat(exampleGradients(loss, th, test))
      for (let i = 0; i < n; i++)
        for (let k = 0; k < m; k++) {
          let a = 0
          for (let j = 0; j < p; j++) a += G[i * p + j] * Gt[k * p + j]
          cross[i * m + k] += eta[t] * a
        }
    }
  })
  return { influence: test ? fromData(cross, [n, m]) : null, selfInfluence: self }
}

/** Options of `dataShapley`. */
export type DataShapleyOptions = {
  /** Permutations to scan (default 100). */
  permutations?: Size
  /** Truncate a scan once |v(N) − v(S)| ≤ tolerance (default 0: no truncation). */
  tolerance?: number
  /** Keep the running estimate every this many permutations (default 10). */
  every?: Size
}

/**
 * TMC data Shapley values of n points under `utility` (the score of a model trained on the given indices; the empty
 * set included): the mean marginal contribution over random permutations, its standard error, the running estimates
 * [snapshots, n] and the utility calls made.
 */
export function dataShapley(
  utility: (indices: readonly number[]) => number,
  n: Size,
  stream: Stream,
  options: DataShapleyOptions = {},
): { values: Float64Array; standardError: Float64Array; history: Tensor; calls: Size; truncated: number } {
  const { permutations = 100, tolerance = 0, every = 10 } = options
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError('dataShapley', 'dataShapley: n must be ≥ 1')
  let calls = 0
  const v = (s: readonly number[]) => {
    calls++
    return utility(s)
  }
  const full = v(Array.from({ length: n }, (_, i) => i))
  const empty = v([])
  const sumv = new Float64Array(n)
  const sumsq = new Float64Array(n)
  const history: Float64Array[] = []
  let skipped = 0
  for (let t = 0; t < permutations; t++) {
    const order = Array.from(toFlat(permutation(child(stream, 'permutation', t), n)))
    let prev = empty
    const subset: number[] = []
    for (let k = 0; k < n; k++) {
      const i = order[k]
      subset.push(i)
      let marginal = 0
      if (tolerance > 0 && Math.abs(full - prev) <= tolerance) skipped++
      else {
        const next = k === n - 1 ? full : v(subset)
        marginal = next - prev
        prev = next
      }
      sumv[i] += marginal
      sumsq[i] += marginal * marginal
    }
    if ((t + 1) % every === 0 || t === permutations - 1) history.push(Float64Array.from(sumv, (s) => s / (t + 1)))
  }
  const T = permutations
  const values = Float64Array.from(sumv, (s) => s / T)
  const standardError = Float64Array.from(sumsq, (q, i) =>
    T > 1 ? Math.sqrt(Math.max(0, q / T - values[i] ** 2) / (T - 1)) : NaN,
  )
  const h = new Float64Array(history.length * n)
  history.forEach((r, k) => h.set(r, k * n))
  return {
    values,
    standardError,
    history: fromData(h, [history.length, n]),
    calls,
    truncated: skipped / (T * n),
  }
}

/**
 * Exact KNN-Shapley values of training points (x [n, d], integer labels y [n]) for a K-nearest-neighbour classifier,
 * averaged over the test points: values [n] and the per-test values [m, n].
 */
export function knnShapley(train: Examples, test: Examples, K: Size): { values: Float64Array; perTest: Tensor } {
  const A = dense.toMatrixF64(train.x, 'knnShapley')
  const B = dense.toMatrixF64(test.x, 'knnShapley')
  const ya = dense.toF64(train.y, 'knnShapley')
  const yb = dense.toF64(test.y, 'knnShapley')
  if (A.n !== B.n) throw new ShapeError('knnShapley', 'knnShapley: train and test need the same features')
  if (!(Number.isInteger(K) && K >= 1)) throw new DomainError('knnShapley', 'knnShapley: K must be ≥ 1')
  const N = A.m
  const per = new Float64Array(B.m * N)
  const values = new Float64Array(N)
  const dist = new Float64Array(N)
  for (let t = 0; t < B.m; t++) {
    for (let i = 0; i < N; i++) {
      let s = 0
      for (let c = 0; c < A.n; c++) s += (A.data[i * A.n + c] - B.data[t * B.n + c]) ** 2
      dist[i] = s
    }
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => dist[a] - dist[b] || a - b)
    const match = (i: number) => (ya[order[i]] === yb[t] ? 1 : 0)
    const s = new Float64Array(N)
    s[N - 1] = match(N - 1) / N
    for (let i = N - 2; i >= 0; i--) s[i] = s[i + 1] + ((match(i) - match(i + 1)) / K) * (Math.min(K, i + 1) / (i + 1))
    for (let i = 0; i < N; i++) {
      per[t * N + order[i]] = s[i]
      values[order[i]] += s[i] / B.m
    }
  }
  return { values, perTest: fromData(per, [B.m, N]) }
}
