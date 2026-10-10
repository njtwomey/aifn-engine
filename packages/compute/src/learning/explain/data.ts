/**
 * Data attribution: which training points a model's behaviour depends on.
 *
 * - `influenceFunctions` (Koh and Liang, 2017, after Cook and Weisberg, 1982): the effect of upweighting a training
 *   point $z$ on the test loss,
 *   $I(z, z_{\text{test}}) = -\nabla\ell(z_{\text{test}}, \hat\thetavec)^\top \Hmat^{-1} \nabla\ell(z, \hat\thetavec)$,
 *   with $\Hmat$ the Hessian of the training objective at its minimiser $\hat\thetavec$. Removing $z$ changes the test
 *   loss by about $-I/n$. $\Hmat^{-1}\vvec$ is solved exactly (the Hessian by autodiff, for small models) or estimated
 *   by LiSSA (Agarwal, Bullins and Hazan, 2017): the recursion
 *   $\hvec \leftarrow \vvec + (1 - \text{damping})\hvec - \Hmat_B\hvec / \text{scale}$ on sampled minibatches $B$,
 *   Hessian-vector products by autodiff, gives $\Hmat^{-1}\vvec \approx \hvec/\text{scale}$.
 * - `tracIn` (Pruthi et al., 2020): the sum over saved checkpoints $\thetavec_t$ of
 *   $\eta_t \nabla\ell(z, \thetavec_t) \cdot \nabla\ell(z', \thetavec_t)$, how much the gradient steps on $z$ lowered
 *   the loss on $z'$. Self-influence ($z' = z$) is large for points the model struggled to fit, such as mislabelled
 *   ones.
 * - `dataShapley` (Ghorbani and Zou, 2019): the Shapley value of each training point in the game whose value is a
 *   model's validation score when trained on a subset, estimated by truncated Monte Carlo over permutations (TMC): a
 *   permutation's scan stops once the score is within `tolerance` of the full data's.
 * - `knnShapley` (Jia et al., 2019): exact Shapley values for the $K$-nearest-neighbour utility
 *   $v(S) = \frac{1}{K}\sum_{k \le \min(K, \lvert S \rvert)} \indicator[y_{(k)} = y_{\text{test}}]$, $y_{(k)}$ the
 *   label of $S$'s $k$-th nearest point, by a recursion over the training points sorted by distance to the test point,
 *   $O(n \log n)$ per test point; averaged over the test set.
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

/**
 * The loss of one example at parameters $\thetavec$ ($p$ values): `x` is the example's row ($d$ values) and `y` its
 * target (a scalar tensor). Differentiated with respect to `theta`.
 */
export type ExampleLoss = (theta: Tensor, x: Tensor, y: Tensor) => Value

/** A labelled set: rows `x` ($n \times d$) and one target per row in `y` ($n$ values). */
export type Examples = { x: MatrixLike; y: VectorLike }

/**
 * A labelled set as float64 tensors. Throws `ShapeError` when there is not one target per row.
 *
 * @param e The set.
 * @param where The caller's name, for error messages.
 * @returns `x` ($n \times d$), `y` ($n$) and the number of rows `n`.
 */
const asTensors = (e: Examples, where: string) => {
  const X = dense.toMatrixF64(e.x, where)
  const y = Float64Array.from(dense.toF64(e.y, where))
  if (y.length !== X.m) throw new ShapeError(where, `${where}: one target per row`)
  return { x: fromData(Float64Array.from(X.data), [X.m, X.n]), y: fromData(y, [X.m]), n: X.m }
}

/**
 * Per-example gradients $\nabla_{\thetavec} \ell(\thetavec, \xvec_i, y_i)$, in one `vmap(grad(loss))` call. Throws
 * `ShapeError` when there is not one target per row.
 *
 * @param loss The loss of one example.
 * @param theta The parameters $\thetavec$ ($p$ values).
 * @param examples The examples ($n$ rows and targets).
 * @returns The gradients ($n \times p$), one row per example.
 *
 * @example Each point pulls the fit towards its own target
 * // Fit a constant: loss(theta, x, y) = (theta x - y)^2 / 2, minimised at the mean of y.
 * const loss = (theta, x, y) => mul(0.5, square(sub(sum(mul(theta, x)), y)))
 * print(exampleGradients(loss, [2], { x: [[1], [1], [1]], y: [1, 2, 3] }))
 */
export function exampleGradients(loss: ExampleLoss, theta: VectorLike, examples: Examples): Tensor {
  const t = fromData(Float64Array.from(dense.toF64(theta, 'exampleGradients')))
  const { x, y } = asTensors(examples, 'exampleGradients')
  return vmap(grad(loss), { inAxes: [null, 0, 0] })(t, x, y) as Tensor
}

/**
 * The training objective $\frac{1}{n}\sum_i \ell(\thetavec, \xvec_i, y_i) + (\lambda/2)\lVert \thetavec \rVert^2$ over
 * the given rows, as a function of $\thetavec$ (the losses taken in one `vmap` call).
 *
 * @param loss The loss of one example.
 * @param x The rows ($n \times d$).
 * @param y The targets ($n$).
 * @param l2 The penalty $\lambda$ (0 for none).
 * @returns The objective, a differentiable function of $\thetavec$.
 */
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
  /** The $\lambda$ of the L2 term $(\lambda/2)\lVert \thetavec \rVert^2$ of the training objective (default 0). */
  l2?: number
  /**
   * LiSSA: the damping (default 0.01). It biases the estimate towards $(\Hmat + \text{damping} \cdot
   * \text{scale}\,\Imat)^{-1}\vvec$.
   */
  damping?: number
  /** LiSSA: the scale, which must exceed $\Hmat$'s largest eigenvalue for the recursion to converge (default 10). */
  scale?: number
  /** LiSSA: the recursion depth, one Hessian-vector product per step (default 200). */
  depth?: Size
  /** LiSSA: independent recursions averaged (default 1). */
  repeats?: Size
  /** LiSSA: training points per minibatch, drawn with replacement (default 32, at most $n$). */
  batch?: Size
  /** LiSSA: the random stream; step $j$ of repeat $r$ draws from `child(stream, 'lissa', r, j)`. Required for LiSSA. */
  stream?: Stream
}

/**
 * Influence of each training point on the total loss over the test set at $\hat\thetavec$ (see the file comment).
 * Throws `DomainError` for LiSSA without a `stream`, and `ShapeError` when a set does not have one target per row.
 *
 * @param loss The loss of one example.
 * @param theta The trained parameters $\hat\thetavec$ ($p$ values), a minimiser of the training objective (with
 *   `l2`); the formula assumes $\Hmat$ is positive definite there.
 * @param train The training set ($n$ points).
 * @param test The test set; its gradients are summed into $\gvec_{\text{test}}$.
 * @param options The method, the objective's penalty and LiSSA's settings.
 * @returns `influence`, $I(z_i, \text{test}) = -\gvec_{\text{test}}^\top \Hmat^{-1} \nabla\ell(z_i)$ ($n$);
 *   `removal`, the predicted change of the test loss when $z_i$ is removed, $-I/n$ ($n$); `selfInfluence`,
 *   $\nabla\ell(z_i)^\top \Hmat^{-1} \nabla\ell(z_i)$ ($n$; exact method only, NaN for LiSSA); and `inverseHvp`,
 *   $\Hmat^{-1}\gvec_{\text{test}}$ ($p$).
 *
 * @example Fitting a mean: predicted against actual leave-one-out changes
 * // Fit a constant: loss(theta, x, y) = (theta x - y)^2 / 2, minimised at the mean of y.
 * const loss = (theta, x, y) => mul(0.5, square(sub(sum(mul(theta, x)), y)))
 * const y = [1, 1.5, 2, 2, 2.5, 3]
 * const train = { x: y.map(() => [1]), y }
 * const test = { x: [[1]], y: [2.5] }
 * const r = influenceFunctions(loss, [2], train, test)
 * print('predicted =', r.removal)
 * // Without point i the fit is the mean of the others.
 * const actual = y.map((yi) => 0.5 * (2.5 - (12 - yi) / 5) ** 2 - 0.5 * (2.5 - 2) ** 2)
 * print('actual =', actual)
 *
 * @example LiSSA estimates the inverse Hessian-vector product
 * // Fit a constant: loss(theta, x, y) = (theta x - y)^2 / 2, minimised at the mean of y.
 * const loss = (theta, x, y) => mul(0.5, square(sub(sum(mul(theta, x)), y)))
 * const train = { x: [[1], [1], [1]], y: [1, 2, 3] }
 * const test = { x: [[1]], y: [2.5] }
 * print('exact:', influenceFunctions(loss, [2], train, test).inverseHvp)
 * const lissa = influenceFunctions(loss, [2], train, test, { method: 'lissa', depth: 100, stream: stream(0) })
 * print('lissa:', lissa.inverseHvp)
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

/**
 * The transpose of a row-major matrix.
 *
 * @param a The matrix, row-major ($m \times n$ values).
 * @param m Its number of rows.
 * @param n Its number of columns.
 * @returns The transpose, row-major ($n \times m$ values).
 */
function transpose(a: ArrayLike<number>, m: Size, n: Size): Float64Array {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j * m + i] = a[i * n + j]
  return out
}

/**
 * TracInCP influences of training points on test points,
 * $\sum_t \eta_t \nabla\ell(z_i, \thetavec_t) \cdot \nabla\ell(z'_j, \thetavec_t)$, and the self-influence
 * $\sum_t \eta_t \lVert \nabla\ell(z_i, \thetavec_t) \rVert^2$. Throws `DomainError` without a checkpoint and
 * `ShapeError` when the rates do not match the checkpoints.
 *
 * @param loss The loss of one example.
 * @param checkpoints The parameters $\thetavec_t$ saved during training ($p$ values each).
 * @param rates The step size $\eta_t$ in force at each checkpoint, or one number for all.
 * @param train The training points $z_i$ ($n$).
 * @param test The test points $z'_j$ ($m$); leave it out for the self-influence alone.
 * @returns `influence`, $n \times m$ (row $i$ for training point $i$), or `null` without `test`; and `selfInfluence`
 *   ($n$).
 *
 * @example The point the checkpoints fitted worst has the largest self-influence
 * // Fit a constant: loss(theta, x, y) = (theta x - y)^2 / 2, minimised at the mean of y.
 * const loss = (theta, x, y) => mul(0.5, square(sub(sum(mul(theta, x)), y)))
 * const train = { x: [[1], [1], [1]], y: [1, 2, 6] }
 * const r = tracIn(loss, [[0], [1.5], [3]], 0.5, train, { x: [[1]], y: [2] })
 * print('self =', r.selfInfluence)
 * print('on the test point =', r.influence)
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
  /**
   * Truncate a scan once $\lvert v(N) - v(S) \rvert \le \text{tolerance}$, crediting the remaining points 0 (default 0:
   * no truncation).
   */
  tolerance?: number
  /** Keep the running estimate every this many permutations (default 10). */
  every?: Size
}

/**
 * TMC data Shapley values of $n$ points under a utility: the mean marginal contribution over random permutations (see
 * the file comment). Throws `DomainError` when $n$ is not a positive integer.
 *
 * @param utility The score of a model trained on the points with the given indices (called on the empty set and the
 *   full set too). The index array is reused and grows after the call: copy it to keep it.
 * @param n The number of training points.
 * @param stream The random stream: permutation $t$ is drawn from `child(stream, 'permutation', t)`.
 * @param options The number of permutations, the truncation and the snapshots.
 * @returns `values` ($n$); `standardError`, of each mean over the permutations (NaN for one permutation); `history`,
 *   the running estimates every `every` permutations and at the end ($\text{snapshots} \times n$); `calls`, the
 *   utility calls made; and `truncated`, the fraction of marginal contributions skipped by truncation.
 *
 * @example Two redundant points share their worth
 * // Points 0 and 1 are worth 1 together or alone, point 2 adds 0.5, point 3 nothing.
 * const utility = (S) => (S.includes(0) || S.includes(1) ? 1 : 0) + (S.includes(2) ? 0.5 : 0)
 * const r = dataShapley(utility, 4, stream(0), { permutations: 60 })
 * print('values =', r.values)
 * print('standard error =', r.standardError, ' calls =', r.calls)
 *
 * @example Truncation saves calls once the full score is reached
 * const utility = (S) => (S.includes(0) || S.includes(1) ? 1 : 0) + (S.includes(2) ? 0.5 : 0)
 * const r = dataShapley(utility, 4, stream(0), { permutations: 60, tolerance: 1e-9 })
 * print('values =', r.values, ' calls =', r.calls, ' truncated =', r.truncated)
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
 * Exact KNN-Shapley values of training points for a $K$-nearest-neighbour classifier (Euclidean distance, ties broken
 * by index), averaged over the test points. Throws `ShapeError` when the sets' widths differ and `DomainError` when
 * $K$ is not a positive integer.
 *
 * @param train The training points ($n$ rows) and their class labels, compared for equality.
 * @param test The test points ($m$ rows) and their labels.
 * @param K The number of neighbours $K$.
 * @returns `values`, the mean over the test points ($n$), and `perTest`, each test point's values ($m \times n$).
 *
 * @example The two nearest points of the right class share the credit
 * const train = { x: [[0], [1], [2], [10]], y: [0, 0, 1, 1] }
 * const test = { x: [[0.5]], y: [0] }
 * print('K = 1:', knnShapley(train, test, 1).values)
 * print('K = 3:', knnShapley(train, test, 3).values)
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
