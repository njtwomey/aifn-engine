/**
 * Low-rank adaptation of a weight change: a target $\Delta\Wmat$ ($m \times n$) fitted by a scaled product
 * $s\,\Bmat\Amat$ of a $m \times r$ and a $r \times n$ factor, by first-order training from LoRA's or PiSSA's start.
 *
 * LoRA (Hu et al., 2022) freezes a pretrained weight $\Wmat_0$ and learns $\Wmat_0 + s\,\Bmat\Amat$, with
 * $\Bmat = \mathbf{0}$ and $\Amat$ random at the start, so training begins exactly at the pretrained model. The scale
 * is $s = \alpha / r$ (`'inverse'`, Hu et al.) or $s = \alpha / \sqrt{r}$ (`'inverse-sqrt'`, rank-stabilised LoRA,
 * Kalajdzievski, 2023, which keeps the update's size stable as $r$ grows). PiSSA (Meng et al., 2024) starts the adapter
 * at the principal singular directions instead. Here the frozen weight plays no part: the fit is of the change alone,
 * minimising
 *
 * $$\mathcal{L}(\Bmat, \Amat) = \tfrac{1}{2}\lVert s\,\Bmat\Amat - \Delta\Wmat \rVert_F^2,$$
 *
 * whose least value over rank-$r$ products is $\frac{1}{2}\sum_{i > r} \sigma_i^2$, the energy of the singular values
 * of $\Delta\Wmat$ past the $r$th (Eckart and Young, 1936). The gradients are written out by hand in `lowRankLoss`,
 * $\nabla_{\Bmat}\mathcal{L} = s\,\Rmat\Amat^\top$ and $\nabla_{\Amat}\mathcal{L} = s\,\Bmat^\top\Rmat$ with
 * $\Rmat = s\,\Bmat\Amat - \Delta\Wmat$ (the tests check them against the engine's autodiff), and stepped by
 * `aifn-compute/optim/first-order`'s `sgdRule` or `adamRule`.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, normal, stream, uniform } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { qr, svd } from 'aifn-compute/numerics/linalg'
import { adamRule, applyUpdates, sgdRule, type UpdateRule } from 'aifn-compute/optim/first-order'

type F64 = dense.F64

/** Options of `lowRankTarget`. */
export type LowRankTargetOptions = {
  /** The side $n$ of the square target. */
  size: Size
  /**
   * A number $d \ge 0$: singular values $\sigma_i = i^{-d}$ for $i = 1, \dots, n$ (a power law; $0$ is flat, larger is
   * closer to low rank). `'low-rank-plus-noise'`: `rank` unit singular values plus Gaussian noise.
   */
  decay: number | 'low-rank-plus-noise'
  /** The rank of the low-rank part, for `'low-rank-plus-noise'`. Default 4. */
  rank?: Size
  /**
   * For `'low-rank-plus-noise'`: the noise level $\nu$, entries $\mathcal{N}(0, \nu^2 / n)$, so the noise's singular
   * values reach about $2\nu$. Default 0.05.
   */
  noise?: number
  /** The seed of the random singular vectors and noise. */
  seed: number
}

/**
 * A random $n \times n$ orthogonal matrix, uniform (Haar): the $\Qmat$ of a Gaussian matrix's QR factorisation with
 * each column's sign set by the diagonal of $\Rmat$ (Mezzadri, 2007).
 *
 * @param s The stream the Gaussian matrix is drawn from.
 * @param n The size $n$.
 * @returns The matrix, row-major.
 */
function randomOrthogonal(s: ReturnType<typeof stream>, n: Size): F64 {
  const { Q, R } = qr(normal(s, 0, 1, { shape: [n, n] }) as Tensor)
  const q = Float64Array.from(toFlat(Q))
  const r = toFlat(R)
  for (let j = 0; j < n; j++) if (r[j * n + j] < 0) for (let i = 0; i < n; i++) q[i * n + j] = -q[i * n + j]
  return q
}

/**
 * A target weight change $\Delta\Wmat = \Umat\,\mathrm{diag}(\svec)\,\Vmat^\top$ ($n \times n$) with random orthogonal
 * $\Umat, \Vmat$ and a chosen spectrum: a power law $\sigma_i = i^{-d}$, or `rank` unit singular values plus Gaussian
 * noise. The same seed gives the same matrix.
 *
 * @param options The size, the spectrum, and for `'low-rank-plus-noise'` the rank and noise level, and the seed.
 * @returns The $n \times n$ target.
 *
 * @example A power-law spectrum
 * // With decay 1 the squared Frobenius norm is the sum of 1 / i^2 for i = 1 … 16.
 * const target = lowRankTarget({ size: 16, decay: 1, seed: 1 })
 * print('squared Frobenius norm:', sum(square(target)))
 * print('sum of 1 / i^2:', Array.from({ length: 16 }, (_, i) => 1 / (i + 1) ** 2).reduce((a, b) => a + b))
 *
 * @example Rank 2 plus noise
 * // Two unit singular values hold almost all the energy; the best rank-2 fit leaves only the noise's.
 * const target = lowRankTarget({ size: 16, decay: 'low-rank-plus-noise', rank: 2, noise: 0.05, seed: 1 })
 * print('squared Frobenius norm:', sum(square(target)))
 * const fit = lowRankFit(target, {
 *   rank: 2, scale: 'inverse', alpha: 2, optimiser: 'sgd', learningRate: 0, steps: 0, init: 'pissa', seed: 1,
 * })
 * print('singular values:', fit.targetSpectrum.slice(0, 4))
 */
export function lowRankTarget(options: LowRankTargetOptions): Tensor {
  const { size: n, decay, rank = 4, noise = 0.05, seed } = options
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('lowRankTarget', `lowRankTarget: size must be a positive integer, got ${n}`)
  const s = stream(seed)
  const U = randomOrthogonal(child(s, 'U'), n)
  const V = randomOrthogonal(child(s, 'V'), n)
  let sigma: F64
  if (decay === 'low-rank-plus-noise') {
    if (!(Number.isInteger(rank) && rank >= 0 && rank <= n))
      throw new DomainError('lowRankTarget', `lowRankTarget: rank must be an integer from 0 to ${n}, got ${rank}`)
    sigma = Float64Array.from({ length: n }, (_, i) => (i < rank ? 1 : 0))
  } else {
    if (!(decay >= 0)) throw new DomainError('lowRankTarget', `lowRankTarget: decay must be non-negative, got ${decay}`)
    sigma = Float64Array.from({ length: n }, (_, i) => (i + 1) ** -decay)
  }
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let v = 0
      for (let k = 0; k < n; k++) if (sigma[k] !== 0) v += U[i * n + k] * sigma[k] * V[j * n + k]
      out[i * n + j] = v
    }
  if (decay === 'low-rank-plus-noise') {
    const g = toFlat(normal(child(s, 'noise'), 0, noise / Math.sqrt(n), { shape: [n * n] }) as Tensor)
    for (let i = 0; i < n * n; i++) out[i] += g[i]
  }
  return fromData(out, [n, n])
}

/** Options of `lowRankFit`. */
export type LowRankFitOptions = {
  /** The rank $r$ of the adapter, from 1 to $\min(m, n)$. */
  rank: Size
  /**
   * The scale $s$: $\alpha / r$ (`'inverse'`, LoRA) or $\alpha / \sqrt{r}$ (`'inverse-sqrt'`, rank-stabilised
   * LoRA).
   */
  scale: 'inverse' | 'inverse-sqrt'
  /** The scale's numerator $\alpha$. */
  alpha: number
  /** The update rule: plain gradient descent (`sgdRule`) or Adam (`adamRule`). */
  optimiser: 'sgd' | 'adam'
  /** The step size of the rule. */
  learningRate: number
  /** The number of steps. */
  steps: Size
  /**
   * The start: `'lora'`, $\Bmat = \mathbf{0}$ and $\Amat$ uniform on $\pm 1/\sqrt{n}$ (PEFT's default, Kaiming uniform
   * with $a = \sqrt{5}$), so $s\,\Bmat\Amat = \mathbf{0}$; `'pissa'`, the factors of the target's best rank-$r$
   * approximation, $\Bmat = \Umat_r (\Smat_r / s)^{1/2}$ and $\Amat = (\Smat_r / s)^{1/2} \Vmat_r^\top$, so
   * $s\,\Bmat\Amat = \Umat_r \Smat_r \Vmat_r^\top$, the optimum (for a negative $s$ the roots are of
   * $\Smat_r / \lvert s \rvert$ and $\Bmat$ carries the sign).
   */
  init: 'lora' | 'pissa'
  /** The seed of LoRA's random $\Amat$ (unused by PiSSA). */
  seed: number
  /** Keep a checkpoint every this many steps, as well as the first and last. Default 10. */
  every?: Size
  /** Keep the update $s\,\Bmat\Amat$ in every checkpoint (default false; each is $m \times n$). */
  updates?: boolean
}

/** A checkpoint of `lowRankFit`. */
export type LowRankCheckpoint = {
  /** The step, from 0. */
  step: Size
  /** The singular values of $s\,\Bmat\Amat$, descending, $r$ of them (the rest are zero). */
  spectrum: Float64Array
  /** $s\,\Bmat\Amat$, when `updates` is set. */
  update?: Tensor
}

/** The result of `lowRankFit`. */
export type LowRankFitResult = {
  /**
   * The loss $\frac{1}{2}\lVert s\,\Bmat\Amat - \Delta\Wmat \rVert_F^2$ before each step and after the last:
   * `steps + 1` values.
   */
  losses: Float64Array
  /** The checkpoints, by step. */
  checkpoints: LowRankCheckpoint[]
  /**
   * The least loss any rank-$r$ update can reach, $\frac{1}{2}\sum_{i > r} \sigma_i^2$ over the target's singular
   * values.
   */
  floor: number
  /** The scale $s$ used. */
  scale: number
  /** The singular values of the target, descending. */
  targetSpectrum: Float64Array
}

/**
 * The loss $\frac{1}{2}\lVert s\,\Bmat\Amat - \Delta\Wmat \rVert_F^2$ and its gradients on row-major arrays.
 *
 * @param T The target $\Delta\Wmat$, $m \times n$.
 * @param B The left factor, $m \times r$.
 * @param A The right factor, $r \times n$.
 * @param m The rows of the target.
 * @param r The rank.
 * @param n The columns of the target.
 * @param s The scale.
 * @returns The loss, $\nabla_{\Bmat}\mathcal{L}$ ($m \times r$), $\nabla_{\Amat}\mathcal{L}$ ($r \times n$) and the
 *   unscaled product $\Bmat\Amat$.
 */
function lossAndGradients(T: ArrayLike<number>, B: F64, A: F64, m: Size, r: Size, n: Size, s: number) {
  const BA = dense.matMul(B, A, m, r, n)
  const R = new Float64Array(m * n)
  for (let i = 0; i < m * n; i++) R[i] = s * BA[i] - T[i]
  return {
    loss: 0.5 * dense.dot(R, R),
    gradB: dense.matMul(R, dense.transpose(A, r, n), m, n, r).map((x) => s * x),
    gradA: dense.matMul(dense.transpose(B, m, r), R, r, m, n).map((x) => s * x),
    BA,
  }
}

/** The loss of an adapter and its gradients, from `lowRankLoss`. */
export type LowRankLoss = {
  /** $\frac{1}{2}\lVert s\,\Bmat\Amat - \Delta\Wmat \rVert_F^2$. */
  loss: number
  /** $\nabla_{\Bmat}\mathcal{L} = s\,\Rmat\Amat^\top$, $m \times r$. */
  gradB: Tensor
  /** $\nabla_{\Amat}\mathcal{L} = s\,\Bmat^\top\Rmat$, $r \times n$. */
  gradA: Tensor
}

/**
 * The loss $\frac{1}{2}\lVert s\,\Bmat\Amat - \Delta\Wmat \rVert_F^2$ of a rank-$r$ adapter and its gradients,
 * written out by hand: with the residual $\Rmat = s\,\Bmat\Amat - \Delta\Wmat$,
 * $\nabla_{\Bmat}\mathcal{L} = s\,\Rmat\Amat^\top$ and $\nabla_{\Amat}\mathcal{L} = s\,\Bmat^\top\Rmat$. Each step of
 * `lowRankFit` computes these.
 *
 * @param target The change $\Delta\Wmat$, $m \times n$.
 * @param B The left factor $\Bmat$, $m \times r$.
 * @param A The right factor $\Amat$, $r \times n$.
 * @param scale The scale $s$.
 * @returns The loss and the two gradients.
 *
 * @example At LoRA's start only B moves
 * // With B = 0 the gradient for A is zero, and the one for B is −s ΔW Aᵀ.
 * const target = [[1, 0], [0, 0.5]]
 * const { loss, gradB, gradA } = lowRankLoss(target, [[0], [0]], [[0.5, -0.5]], 2)
 * print('loss =', loss, ' gradB =', gradB, ' gradA =', gradA)
 */
export function lowRankLoss(target: MatrixLike, B: MatrixLike, A: MatrixLike, scale: number): LowRankLoss {
  const where = 'lowRankLoss'
  const t = dense.toMatrixF64(target, where)
  const b = dense.toMatrixF64(B, where)
  const a = dense.toMatrixF64(A, where)
  if (b.m !== t.m || a.n !== t.n || b.n !== a.m)
    throw new ShapeError(
      where,
      `${where}: B (${b.m} × ${b.n}) times A (${a.m} × ${a.n}) does not match the target (${t.m} × ${t.n})`,
    )
  const r = b.n
  const out = lossAndGradients(t.data, b.data, a.data, t.m, r, t.n, scale)
  return { loss: out.loss, gradB: dense.mat(out.gradB, t.m, r), gradA: dense.mat(out.gradA, r, t.n) }
}

/**
 * The singular values of $\Bmat\Amat$ from the $r \times r$ core of the two QR factorisations: with
 * $\Bmat = \Qmat_B \Rmat_B$ and $\Amat^\top = \Qmat_A \Rmat_A$,
 * $\Bmat\Amat = \Qmat_B (\Rmat_B \Rmat_A^\top) \Qmat_A^\top$ has the singular values of $\Rmat_B \Rmat_A^\top$.
 *
 * @param B The left factor, $m \times r$, row-major.
 * @param A The right factor, $r \times n$, row-major.
 * @param m The rows of $\Bmat$.
 * @param r The rank.
 * @param n The columns of $\Amat$.
 * @param s The scale multiplying the product.
 * @returns The $r$ singular values of $s\,\Bmat\Amat$, descending.
 */
function productSpectrum(B: F64, A: F64, m: Size, r: Size, n: Size, s: number): Float64Array {
  const RB = toFlat(qr(dense.mat(B, m, r)).R)
  const RA = toFlat(qr(dense.mat(dense.transpose(A, r, n), n, r)).R)
  const core = dense.matMul(RB, dense.transpose(RA, r, r), r, r, r)
  return Float64Array.from(toFlat(svd(dense.mat(core, r, r)).S), (v) => Math.abs(s) * v)
}

/**
 * Fit a target weight change $\Delta\Wmat$ by a rank-$r$ adapter $s\,\Bmat\Amat$: `steps` steps of gradient descent or
 * Adam on $\frac{1}{2}\lVert s\,\Bmat\Amat - \Delta\Wmat \rVert_F^2$ from LoRA's start ($\Bmat = \mathbf{0}$) or
 * PiSSA's (the target's top-$r$ singular directions, already the optimum). From LoRA's start the loss falls to the
 * floor $\frac{1}{2}\sum_{i > r} \sigma_i^2$ as the adapter's spectrum grows towards the target's first $r$ singular
 * values; how fast depends on the scale convention, which with plain gradient descent sets the effective step size
 * ($s^2$ times the learning rate). From PiSSA's start the gradient is zero up to rounding, so gradient descent stays
 * put, but Adam, which divides each step by the gradient's own size, drifts away from the floor by steps of the
 * learning rate. The same options give the same result.
 *
 * @param target The change $\Delta\Wmat$, $m \times n$.
 * @param options The rank, the scale convention and $\alpha$, the optimiser and its step size, the number of steps, the
 *   start, the seed, the checkpoint spacing and whether to keep the updates.
 * @returns The loss at every step, the checkpoints (step and the adapter's spectrum), the floor, the scale and the
 *   target's singular values.
 *
 * @example LoRA's start climbs to the target's leading singular values
 * const target = lowRankTarget({ size: 32, decay: 1, seed: 1 })
 * const fit = lowRankFit(target, {
 *   rank: 4, scale: 'inverse', alpha: 8, optimiser: 'adam', learningRate: 0.01, steps: 300, init: 'lora', seed: 1,
 * })
 * print('loss: start', fit.losses[0], ' end', fit.losses.at(-1), ' floor', fit.floor)
 * print('adapter spectrum at the end:', fit.checkpoints.at(-1).spectrum)
 * print('target spectrum:', fit.targetSpectrum.slice(0, 4))
 *
 * @example PiSSA starts at the floor
 * const target = lowRankTarget({ size: 32, decay: 1, seed: 1 })
 * const fit = lowRankFit(target, {
 *   rank: 4, scale: 'inverse', alpha: 8, optimiser: 'adam', learningRate: 0.01, steps: 10, init: 'pissa', seed: 1,
 * })
 * print('loss at step 0:', fit.losses[0], ' floor:', fit.floor)
 */
export function lowRankFit(target: Tensor, options: LowRankFitOptions): LowRankFitResult {
  const where = 'lowRankFit'
  const { rank: r, alpha, optimiser, learningRate, steps, init, seed, every = 10, updates = false } = options
  const { data: T, m, n } = dense.toMatrixF64(target, where)
  if (!(Number.isInteger(r) && r >= 1 && r <= Math.min(m, n)))
    throw new DomainError(where, `${where}: rank must be an integer from 1 to ${Math.min(m, n)}, got ${r}`)
  if (!(Number.isInteger(steps) && steps >= 0))
    throw new DomainError(where, `${where}: steps must be a non-negative integer, got ${steps}`)
  if (!(Number.isInteger(every) && every >= 1))
    throw new DomainError(where, `${where}: every must be a positive integer, got ${every}`)
  if (!(alpha !== 0 && Number.isFinite(alpha))) throw new DomainError(where, `${where}: alpha must be non-zero`)
  const s = options.scale === 'inverse' ? alpha / r : alpha / Math.sqrt(r)
  const targetSvd = svd(target)
  const sigma = Float64Array.from(toFlat(targetSvd.S))
  let floor = 0
  for (let i = r; i < sigma.length; i++) floor += sigma[i] * sigma[i]
  floor /= 2

  let B: F64
  let A: F64
  if (init === 'pissa') {
    const { U, S, V } = targetSvd
    const u = toFlat(U)
    const v = toFlat(V)
    const k = S.shape[0]
    const root = Float64Array.from(toFlat(S).slice(0, r), (x) => Math.sqrt(x / Math.abs(s)) * Math.sign(s))
    const rootA = Float64Array.from(toFlat(S).slice(0, r), (x) => Math.sqrt(x / Math.abs(s)))
    B = Float64Array.from({ length: m * r }, (_, at) => u[Math.floor(at / r) * k + (at % r)] * root[at % r])
    A = Float64Array.from(
      { length: r * n },
      (_, at) => rootA[Math.floor(at / n)] * v[(at % n) * k + Math.floor(at / n)],
    )
  } else {
    const bound = 1 / Math.sqrt(n)
    B = new Float64Array(m * r)
    A = Float64Array.from(toFlat(uniform(child(stream(seed), 'A'), -bound, bound, { shape: [r * n] }) as Tensor))
  }

  const rule: UpdateRule =
    optimiser === 'adam' ? adamRule({ stepSize: learningRate }) : sgdRule({ stepSize: learningRate })
  let params = { A: dense.mat(A, r, n), B: dense.mat(B, m, r) }
  let state = rule.init(params)
  const losses = new Float64Array(steps + 1)
  const checkpoints: LowRankCheckpoint[] = []
  for (let t = 0; t <= steps; t++) {
    A = dense.data(params.A)
    B = dense.data(params.B)
    const { loss, gradB, gradA, BA } = lossAndGradients(T, B, A, m, r, n, s)
    losses[t] = loss
    if (t % every === 0 || t === steps)
      checkpoints.push({
        step: t,
        spectrum: productSpectrum(B, A, m, r, n, s),
        ...(updates
          ? {
              update: dense.mat(
                BA.map((x) => s * x),
                m,
                n,
              ),
            }
          : {}),
      })
    if (t === steps) break
    const out = rule.update({ A: dense.mat(gradA, r, n), B: dense.mat(gradB, m, r) }, state, params)
    params = applyUpdates(params, out.updates) as typeof params
    state = out.state
  }
  return { losses, checkpoints, floor, scale: s, targetSpectrum: sigma }
}
