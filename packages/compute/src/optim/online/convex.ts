/**
 * Online convex optimisation: each round $t = 1, \dots, T$ the learner plays $\wvec_t$ in a convex set $\Wcal$, the
 * environment reveals a convex loss $f_t$, and the learner pays $f_t(\wvec_t)$ and sees a subgradient
 * $\gvec_t \in \partial f_t(\wvec_t)$. Online gradient descent, follow the regularised leader (linearised, with L1
 * and L2 terms and an adaptive per-coordinate form), the online Newton step and diagonal AdaGrad, as step-through
 * algorithms whose states carry the regret $\sum_t f_t(\wvec_t) - f_t(\uvec)$ against a fixed comparator $\uvec$.
 *
 * Every method is an `Algorithm` started with `undefined` (its start point is an option), one step per round, that is
 * done after `rounds` rounds. The dimension $d$ is given as `dim`, or read from `start`. A non-finite loss or point
 * sets `diverged`; invalid options and a gradient of the wrong length throw `DomainError`. The references are Hazan
 * (2016), "Introduction to Online Convex Optimization", and Shalev-Shwartz (2012), "Online Learning and Online Convex
 * Optimization".
 */

import type { Schedule, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { eigh } from 'aifn-compute/numerics/linalg'

type F64 = dense.F64

/**
 * The loss of round $t$ (1-based) at a point $\wvec$: its value $f_t(\wvec)$ and a subgradient
 * $\gvec_t \in \partial f_t(\wvec)$ of length $d$.
 */
export type OnlineLoss = (t: number, w: Tensor) => { value: number; grad: VectorLike }

/**
 * The decision set $\Wcal$: a Euclidean ball (centred at 0 unless `center` is given), a box (bounds per coordinate or
 * shared), the probability simplex, or all of $\reals^d$ when omitted.
 */
export type OnlineDomain =
  | { kind: 'ball'; radius: number; center?: VectorLike }
  | { kind: 'box'; lower: number | VectorLike; upper: number | VectorLike }
  | { kind: 'simplex' }

/** Options shared by the online convex optimisers. */
export type OnlineOptions = {
  /** The decision set (default $\reals^d$). */
  domain?: OnlineDomain
  /** The first point $\wvec_1$, projected onto the domain (default the projection of 0). */
  start?: VectorLike
  /**
   * A fixed comparator $\uvec$ of length $d$: the state then tracks $\sum_t f_t(\uvec)$ and the regret
   * $\sum_t f_t(\wvec_t) - f_t(\uvec)$. The loss is called a second time each round, at $\uvec$.
   */
  comparator?: VectorLike
  /** The horizon $T$: the algorithm is done after $T$ rounds (default unbounded). */
  rounds?: number
}

/** One state of an online convex optimiser. */
export interface OnlineState extends Status {
  /** The point $\wvec_{t+1}$ played next. */
  w: Tensor
  /** The point played in the last round, $\wvec_t$ ($\wvec_1$ at $t = 0$). */
  played: Tensor
  /** The last round's loss $f_t(\wvec_t)$ (0 at $t = 0$). */
  loss: number
  /** The last subgradient $\gvec_t$ (zeros at $t = 0$). */
  grad: Tensor
  /** $\sum_{s \le t} f_s(\wvec_s)$. */
  cumulative: number
  /** $\sum_{s \le t} f_s(\uvec)$ for the comparator (NaN without one). */
  comparatorLoss: number
  /** The regret against the comparator, `cumulative` minus `comparatorLoss` (NaN without one). */
  regret: number
  /** The step size used in the last round (NaN at $t = 0$, or where a method has none). */
  stepSize: number
  /** The horizon `rounds` has been reached. */
  done: boolean
}

// ── Projections ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A box bound as $d$ values. Throws `DomainError` for a vector bound of the wrong length.
 *
 * @param v The bound: one number for every coordinate, or a vector of $d$.
 * @param d The dimension.
 * @param where The caller's name for error messages.
 * @returns The bound per coordinate (a new array for a number; the bound's own values for a vector).
 */
function boundsOf(v: number | VectorLike, d: number, where: string): F64 {
  if (typeof v === 'number') return new Float64Array(d).fill(v)
  const b = dense.toF64(v, where)
  if (b.length !== d) throw new DomainError(where, `${where}: a box bound has ${b.length} entries for dimension ${d}`)
  return b
}

/**
 * The Euclidean projection onto the probability simplex (sort-based; Held, Wolfe and Crowder, 1974).
 *
 * @param v The point to project; not modified.
 * @returns The projection, a new array.
 */
function projectSimplex(v: F64): F64 {
  const u = Float64Array.from(v).sort().reverse()
  let cumulative = 0
  let theta = 0
  for (let j = 0; j < u.length; j++) {
    cumulative += u[j]
    const t = (cumulative - 1) / (j + 1)
    if (u[j] - t > 0) theta = t
  }
  return v.map((x) => Math.max(0, x - theta))
}

/**
 * The Euclidean projection $\Pi_\Wcal(\vvec)$ onto a domain (the identity without one). Throws `DomainError` for a
 * box bound whose length is not that of `v`.
 *
 * @param v The point to project.
 * @param domain The decision set: a ball, a box or the probability simplex (default none).
 * @returns The projection, a new vector.
 *
 * @example Onto a ball, a box and the simplex
 * print('ball =', projectOnto([3, 4], { kind: 'ball', radius: 1 }))
 * print('box =', projectOnto([-1, 0.5, 2], { kind: 'box', lower: 0, upper: 1 }))
 * print('simplex =', projectOnto([0.5, 0.8, -0.2], { kind: 'simplex' }))
 * print('no domain =', projectOnto([0.5, 0.8, -0.2]))
 */
export function projectOnto(v: VectorLike, domain?: OnlineDomain): Tensor {
  const x = dense.toF64(v, 'projectOnto')
  return dense.vec(project(x, domain, 'projectOnto'))
}

/**
 * The Euclidean projection onto a domain, on plain arrays.
 *
 * @param x The point to project; not modified.
 * @param domain The decision set, or undefined for none (the identity).
 * @param where The caller's name for error messages.
 * @returns The projection, a new array.
 */
function project(x: F64, domain: OnlineDomain | undefined, where: string): F64 {
  if (domain === undefined) return Float64Array.from(x)
  const d = x.length
  if (domain.kind === 'box') {
    const lo = boundsOf(domain.lower, d, where)
    const hi = boundsOf(domain.upper, d, where)
    return x.map((v, i) => Math.min(hi[i], Math.max(lo[i], v)))
  }
  if (domain.kind === 'simplex') return projectSimplex(x)
  const c = domain.center === undefined ? new Float64Array(d) : dense.toF64(domain.center, where)
  const offset = dense.sub(x, c)
  const r = dense.norm(offset)
  return r <= domain.radius ? Float64Array.from(x) : dense.add(c, dense.scale(domain.radius / r, offset))
}

/**
 * The projection in the norm $\lVert \cdot \rVert_\Amat$ of a symmetric positive definite
 * $\Amat = \Vmat \diag(\lambdavec) \Vmat^\top$ onto a ball centred at 0:
 * $\argmin_{\lVert \wvec \rVert \le r} (\wvec - \yvec)^\top\Amat(\wvec - \yvec) = (\Amat + \mu\Imat)^{-1}\Amat\yvec$
 * with the $\mu \ge 0$ that puts it on the sphere (bisection on $\mu$).
 *
 * @param y The point $\yvec$ to project, of length $d$; not modified. Returned as a copy when it is inside the ball.
 * @param lambda The eigenvalues $\lambdavec$ of $\Amat$, all positive, in the order of the columns of `V`.
 * @param V The eigenvectors of $\Amat$ as the columns of a row-major $d \times d$ array, or null when $\Amat$ is the
 *   diagonal matrix $\diag(\lambdavec)$.
 * @param radius The radius $r$ of the ball.
 * @returns The projection, a new array.
 */
function projectBallInNorm(y: F64, lambda: F64, V: F64 | null, radius: number): F64 {
  const d = y.length
  // Coordinates in the eigenbasis (V = null: A is diagonal, the basis is the standard one).
  const z = V === null ? Float64Array.from(y) : dense.matTVec(V, y, d, d)
  if (dense.norm(y) <= radius) return Float64Array.from(y)
  const at = (mu: number) => z.map((v, i) => (lambda[i] * v) / (lambda[i] + mu))
  let lo = 0
  let hi = 1
  while (dense.norm(at(hi)) > radius) hi *= 2
  for (let k = 0; k < 200 && hi - lo > 1e-14 * Math.max(1, hi); k++) {
    const mid = (lo + hi) / 2
    if (dense.norm(at(mid)) > radius) lo = mid
    else hi = mid
  }
  const w = at(hi)
  return V === null ? w : dense.matVec(V, w, d, d)
}

// ── The shared driver ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The dimension of an online problem: `d` if given, else the length of `options.start`. Throws `DomainError` when
 * neither gives a positive integer. A `start` of another length than `d` is not checked here.
 *
 * @param options The method's options, read for `start`.
 * @param d The dimension the caller gave (`dim`), or undefined.
 * @param where The caller's name for error messages.
 * @returns The dimension $d$.
 */
function dimensionOf(options: OnlineOptions, d: number | undefined, where: string): number {
  const fromStart = options.start === undefined ? undefined : dense.toF64(options.start, where).length
  const dim = d ?? fromStart
  if (dim === undefined || !Number.isInteger(dim) || dim < 1)
    throw new DomainError(where, `${where}: give the dimension or a start point`)
  return dim
}

/**
 * Drive an online optimiser: `update(state data, w, g, t)` returns the next point and the step size used; the driver
 * evaluates the loss, books the regret and keeps the method's own plain-data `extra` state. Throws `DomainError` for
 * a comparator of the wrong length, and, when stepped, for a gradient of the wrong length.
 *
 * @param name The algorithm's name.
 * @param where The caller's name for error messages.
 * @param loss The loss of each round.
 * @param d The dimension.
 * @param options The domain, start point, comparator and horizon.
 * @param extra0 The method's own state before the first round (plain data: tensors and numbers).
 * @param update The method's update: from its state, the point played $\wvec_t$, the subgradient $\gvec_t$ and the
 *   round $t$, the next point (already projected), the step size used and its next state. It must not modify `w` or
 *   `g`.
 * @returns The algorithm, started with `undefined`, whose state is an `OnlineState` with the method's `extra`.
 */
function onlineGame<E extends Record<string, Tensor | number>>(
  name: string,
  where: string,
  loss: OnlineLoss,
  d: number,
  options: OnlineOptions,
  extra0: E,
  update: (extra: E, w: F64, g: F64, t: number) => { w: F64; stepSize: number; extra: E },
): Algorithm<void, OnlineState & { extra: E }> {
  const T = options.rounds ?? Infinity
  const u = options.comparator === undefined ? null : dense.toF64(options.comparator, where)
  if (u !== null && u.length !== d) throw new DomainError(where, `${where}: the comparator has the wrong dimension`)
  const w1 = project(
    options.start === undefined ? new Float64Array(d) : dense.toF64(options.start, where),
    options.domain,
    where,
  )
  return {
    name,
    init: () => ({
      t: 0,
      w: dense.vec(w1),
      played: dense.vec(w1),
      loss: 0,
      grad: dense.vec(new Float64Array(d)),
      cumulative: 0,
      comparatorLoss: u === null ? NaN : 0,
      regret: u === null ? NaN : 0,
      stepSize: NaN,
      extra: extra0,
      done: T === 0,
    }),
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1 }
      const t = s.t + 1
      const w = dense.data(s.w)
      const out = loss(t, dense.vec(Float64Array.from(w)))
      const g = dense.toF64(out.grad, where)
      if (g.length !== d) throw new DomainError(where, `${where}: round ${t} gave a gradient of length ${g.length}`)
      const value = out.value
      const cumulative = s.cumulative + value
      const comparatorLoss = u === null ? NaN : s.comparatorLoss + loss(t, dense.vec(Float64Array.from(u))).value
      const next = update(s.extra, w, g, t)
      const finite = next.w.every(Number.isFinite) && Number.isFinite(value)
      return {
        t,
        w: dense.vec(next.w),
        played: dense.vec(Float64Array.from(w)),
        loss: value,
        grad: dense.vec(Float64Array.from(g)),
        cumulative,
        comparatorLoss,
        regret: cumulative - comparatorLoss,
        stepSize: next.stepSize,
        extra: next.extra,
        done: t >= T,
        diverged: !finite,
      }
    },
    done: (s) => s.done,
  }
}

// ── Online gradient descent ──────────────────────────────────────────────────────────────────────────────────────────

/**
 * Online gradient descent (Zinkevich, 2003): $\wvec_{t+1} = \Pi_\Wcal(\wvec_t - \eta_t\gvec_t)$. With
 * $\eta_t = D/(G\sqrt{t})$ (`ogdStepSize`) its regret on convex losses is at most $\frac{3}{2}DG\sqrt{T}$, for a
 * domain of diameter $D$ and gradients bounded by $G$; with $\eta_t = 1/(\lambda t)$ on $\lambda$-strongly convex
 * losses it is $O((G^2/\lambda) \log T)$.
 *
 * @param loss The loss of each round, as a value and a subgradient at the point played.
 * @param options The `OnlineOptions` (domain, start, comparator, horizon), with `dim`, the dimension $d$ (needed
 *   without `start`), and `stepSize`, a constant $\eta$ or a schedule $\eta_t$ called with the round $t \ge 1$
 *   (default 0.1).
 * @returns The algorithm, started with `undefined`, one step per round.
 *
 * @example Regret against the best fixed point, well inside the bound
 * // The losses (w − zₜ)² with zₜ alternating 1, 3, ...: the best fixed point is u = 2.
 * const loss = (t, w) => {
 *   const z = t % 2 === 1 ? 1 : 3
 *   const x = toFlat(w)[0]
 *   return { value: (x - z) ** 2, grad: [2 * (x - z)] }
 * }
 * // The box [0, 4] has diameter 4, and on it the gradients are bounded by 8.
 * const options = { dim: 1, domain: { kind: 'box', lower: 0, upper: 4 }, comparator: [2], rounds: 100 }
 * const alg = onlineGradientDescent(loss, { ...options, stepSize: ogdStepSize(4, 8) })
 * const s = run(alg, undefined, 1000)
 * print('w =', s.w)
 * print('learner loss =', s.cumulative)
 * print('comparator loss =', s.comparatorLoss)
 * print('regret =', s.regret)
 * print('bound =', ogdRegretBound(100, 4, 8))
 */
export function onlineGradientDescent(
  loss: OnlineLoss,
  options: OnlineOptions & { dim?: number; stepSize?: number | Schedule } = {},
): Algorithm<void, OnlineState> {
  const where = 'onlineGradientDescent'
  const d = dimensionOf(options, options.dim, where)
  const eta = options.stepSize ?? 0.1
  const rate = typeof eta === 'number' ? () => eta : (t: number) => Number(eta(t))
  return onlineGame('online-gradient-descent', where, loss, d, options, {}, (extra, w, g, t) => {
    const stepSize = rate(t)
    return { w: project(dense.axpy(-stepSize, g, w), options.domain, where), stepSize, extra }
  })
}

/**
 * The step-size schedule $\eta_t = D/(G\sqrt{t})$ of online gradient descent ($t \ge 1$; a smaller $t$ is taken as
 * 1), with regret at most $\frac{3}{2}DG\sqrt{T}$. Throws `DomainError` unless both arguments are positive.
 *
 * @param diameter The diameter $D$ of the domain, the largest distance between two of its points.
 * @param gradientBound A bound $G$ on the norm of every subgradient.
 * @returns The schedule, a function of the round $t$.
 *
 * @example The step falls as one over the square root of the round
 * const eta = ogdStepSize(2, 1)
 * print('rounds 1, 2, 3, 4, 100:', [1, 2, 3, 4, 100].map(eta))
 */
export function ogdStepSize(diameter: number, gradientBound: number): Schedule {
  if (!(diameter > 0 && gradientBound > 0))
    throw new DomainError('ogdStepSize', 'ogdStepSize: the diameter and gradient bound must be positive')
  return (t) => diameter / (gradientBound * Math.sqrt(Math.max(t, 1)))
}

// ── Follow the regularised leader ────────────────────────────────────────────────────────────────────────────────────

/** Options of `followTheRegularisedLeader`. */
export type FtrlOptions = OnlineOptions & {
  /** The dimension $d$ (needed without `start`). */
  dim?: number
  /**
   * The rate $\eta > 0$ of the quadratic regulariser $\lVert \wvec \rVert^2/(2\eta)$ (default 0.1), or $\alpha$ of
   * the adaptive rate.
   */
  eta?: number
  /** An L1 penalty $\lambda_1\lVert \wvec \rVert_1$, which makes coordinates exactly zero (default 0). */
  l1?: number
  /** An extra L2 penalty $(\lambda_2/2)\lVert \wvec \rVert^2$ (default 0). */
  l2?: number
  /**
   * Per-coordinate adaptive rates (FTRL-Proximal; McMahan et al., 2013): coordinate $i$ uses
   * $\eta_{t,i} = \alpha/(\beta + \sqrt{\sum_s g_{s,i}^2})$ and a proximal regulariser centred on the past points.
   * Not with a ball domain. Default false.
   */
  adaptive?: boolean
  /** $\beta \ge 0$ of the adaptive rate (default 1). */
  beta?: number
}

/**
 * Follow the regularised leader on the linearised losses (Shalev-Shwartz, 2012; McMahan, 2017):
 * $\wvec_{t+1} = \argmin_{\wvec} \langle \gvec_{1:t}, \wvec \rangle + \lVert \wvec \rVert^2/(2\eta) + R(\wvec)$
 * with $R(\wvec) = \lambda_1\lVert \wvec \rVert_1 + (\lambda_2/2)\lVert \wvec \rVert^2$ and $\gvec_{1:t}$ the sum
 * of the subgradients so far: coordinate-wise soft thresholding of $\gvec_{1:t}$, then the projection onto a ball (a
 * rescaling) or a box (a clip). Its regret is at most
 * $\lVert \uvec \rVert^2/(2\eta) + \eta \sum_t \lVert \gvec_t \rVert^2$. With `adaptive` it is McMahan's
 * FTRL-Proximal, the per-coordinate AdaGrad form, whose state's `stepSize` is NaN (the rates differ by coordinate).
 * Without `adaptive` the leader does not depend on `start`, which sets only the first point. Throws `DomainError` for
 * invalid rates or penalties, for the simplex (use `hedge`), and for a ball with `adaptive`.
 *
 * @param loss The loss of each round, as a value and a subgradient at the point played.
 * @param options The `OnlineOptions` with the dimension, the rate, the L1 and L2 penalties and the adaptive form.
 * @returns The algorithm, started with `undefined`, one step per round.
 *
 * @example An L1 penalty holds a coordinate at exactly zero
 * // Linear losses whose gradients are (-1, 0.3) and (-1, -0.3) in turn: after 21 rounds they sum to (-21, 0.3).
 * const loss = (t, w) => {
 *   const g = [-1, t % 2 === 1 ? 0.3 : -0.3]
 *   const [a, b] = toFlat(w)
 *   return { value: g[0] * a + g[1] * b, grad: g }
 * }
 * for (const l1 of [0, 0.5]) {
 *   const s = run(followTheRegularisedLeader(loss, { dim: 2, eta: 0.1, l1, rounds: 21 }), undefined, 100)
 *   print(`l1 = ${l1}: w =`, s.w)
 * }
 */
export function followTheRegularisedLeader(loss: OnlineLoss, options: FtrlOptions = {}): Algorithm<void, OnlineState> {
  const where = 'followTheRegularisedLeader'
  const d = dimensionOf(options, options.dim, where)
  const eta = options.eta ?? 0.1
  const l1 = options.l1 ?? 0
  const l2 = options.l2 ?? 0
  const beta = options.beta ?? 1
  if (!(eta > 0) || !(l1 >= 0) || !(l2 >= 0) || !(beta >= 0))
    throw new DomainError(where, `${where}: η must be positive and λ₁, λ₂, β non-negative`)
  if (options.adaptive && options.domain?.kind === 'ball')
    throw new DomainError(where, `${where}: the adaptive form supports a box or no domain`)
  if (options.domain?.kind === 'simplex')
    throw new DomainError(where, `${where}: use hedge for the simplex (FTRL with the entropy)`)
  const soft = (z: number, scale: number) => (Math.abs(z) <= l1 ? 0 : -(z - Math.sign(z) * l1) / scale)
  const start = { z: dense.vec(new Float64Array(d)), squares: dense.vec(new Float64Array(d)) }
  return onlineGame('follow-the-regularised-leader', where, loss, d, options, start, (extra, w, g) => {
    const z = Float64Array.from(dense.data(extra.z))
    const n = Float64Array.from(dense.data(extra.squares))
    const next = new Float64Array(d)
    if (options.adaptive) {
      for (let i = 0; i < d; i++) {
        const before = n[i]
        n[i] += g[i] * g[i]
        const sigma = (Math.sqrt(n[i]) - Math.sqrt(before)) / eta
        z[i] += g[i] - sigma * w[i]
        next[i] = soft(z[i], (beta + Math.sqrt(n[i])) / eta + l2)
      }
    } else {
      for (let i = 0; i < d; i++) {
        z[i] += g[i]
        next[i] = soft(z[i], 1 / eta + l2)
      }
    }
    return {
      w: project(next, options.domain, where),
      stepSize: options.adaptive ? NaN : 1 / (1 / eta + l2),
      extra: { z: dense.vec(z), squares: dense.vec(n) },
    }
  })
}

// ── Second-order and adaptive methods ────────────────────────────────────────────────────────────────────────────────

/**
 * The online Newton step (Hazan, Agarwal and Kale, 2007):
 * $\Amat_t = \varepsilon\Imat + \sum_{s \le t} \gvec_s\gvec_s^\top$,
 * $\wvec_{t+1} = \Pi^{\Amat_t}(\wvec_t - \gamma^{-1}\Amat_t^{-1}\gvec_t)$, the projection in the norm of $\Amat_t$.
 * On $\alpha$-exp-concave losses with $\gamma = \frac{1}{2}\min(1/(4GD), \alpha)$ its regret is
 * $O(d(1/\alpha + GD) \log T)$. Supports a ball centred at 0, or no domain, and throws `DomainError` for any other
 * domain or a $\gamma$ or $\varepsilon$ that is not positive. Each round costs an eigendecomposition of the
 * $d \times d$ matrix $\Amat_t$; the state's `stepSize` is $1/\gamma$.
 *
 * @param loss The loss of each round, as a value and a subgradient at the point played.
 * @param options The `OnlineOptions` with `dim`, the dimension $d$ (needed without `start`), `gamma`, the
 *   $\gamma > 0$ of the step (default 0.5), and `epsilon`, the $\varepsilon > 0$ that starts $\Amat_0$ (default 1).
 * @returns The algorithm, started with `undefined`, one step per round.
 *
 * @example Squared loss on a regression with no noise
 * // The targets are exactly ⟨u, xₜ⟩ with u = (1, -1), so the comparator's loss is 0 and the regret is the learner's.
 * const xs = [[1, 0], [0, 1], [1, 1], [1, -1], [2, 1], [0.5, -1]]
 * const loss = (t, w) => {
 *   const x = xs[(t - 1) % xs.length]
 *   const [a, b] = toFlat(w)
 *   const r = a * x[0] + b * x[1] - (x[0] - x[1])
 *   return { value: r * r, grad: [2 * r * x[0], 2 * r * x[1]] }
 * }
 * const options = { dim: 2, domain: { kind: 'ball', radius: 2 }, comparator: [1, -1], rounds: 60 }
 * const ons = run(onlineNewtonStep(loss, { ...options, gamma: 0.5 }), undefined, 1000)
 * const ogd = run(onlineGradientDescent(loss, { ...options, stepSize: 0.1 }), undefined, 1000)
 * print('online Newton step: w =', ons.w, 'regret =', ons.regret)
 * print('online gradient descent: w =', ogd.w, 'regret =', ogd.regret)
 */
export function onlineNewtonStep(
  loss: OnlineLoss,
  options: OnlineOptions & { dim?: number; gamma?: number; epsilon?: number } = {},
): Algorithm<void, OnlineState> {
  const where = 'onlineNewtonStep'
  const d = dimensionOf(options, options.dim, where)
  const gamma = options.gamma ?? 0.5
  const epsilon = options.epsilon ?? 1
  if (!(gamma > 0 && epsilon > 0)) throw new DomainError(where, `${where}: γ and ε must be positive`)
  const domain = options.domain
  if (domain !== undefined && !(domain.kind === 'ball' && domain.center === undefined))
    throw new DomainError(where, `${where}: supports a ball centred at 0, or no domain`)
  const A0 = dense.scale(epsilon, dense.identity(d))
  return onlineGame('online-newton-step', where, loss, d, options, { A: dense.mat(A0, d, d) }, (extra, w, g) => {
    const A = Float64Array.from(dense.data(extra.A))
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) A[i * d + j] += g[i] * g[j]
    const { values, vectors } = eigh(dense.mat(A, d, d))
    const lambda = dense.data(values)
    const V = dense.data(vectors)
    // A⁻¹g = V diag(1/λ) Vᵀ g.
    const proj = dense.matTVec(V, g, d, d).map((v, i) => v / lambda[i])
    const y = dense.axpy(-1 / gamma, dense.matVec(V, proj, d, d), w)
    const next = domain === undefined ? y : projectBallInNorm(y, lambda, V, domain.radius)
    return { w: next, stepSize: 1 / gamma, extra: { A: dense.mat(A, d, d) } }
  })
}

/**
 * Diagonal AdaGrad as an online method (Duchi, Hazan and Singer, 2011):
 * $h_{t,i} = \delta + \sqrt{\sum_{s \le t} g_{s,i}^2}$ and
 * $\wvec_{t+1} = \Pi^{\Hmat_t}(\wvec_t - \eta \Hmat_t^{-1}\gvec_t)$, projected in the norm of
 * $\Hmat_t = \diag(\hvec_t)$ (a clip for a box, a secular equation for a ball centred at 0). Its regret is at most
 * $\sqrt{2} D_\infty \sum_i \lVert \gvec_{1:T,i} \rVert_2$ at $\eta = D_\infty/\sqrt{2}$, with
 * $\gvec_{1:T,i}$ coordinate $i$ of every subgradient and $D_\infty$ the domain's diameter in the max norm, far below
 * $DG\sqrt{T}$ when gradients are sparse. Throws `DomainError` for the simplex or a ball not centred at 0. The state's
 * `stepSize` is $\eta$.
 *
 * @param loss The loss of each round, as a value and a subgradient at the point played.
 * @param options The `OnlineOptions` with `dim`, the dimension $d$ (needed without `start`), `eta`, the rate $\eta$
 *   (default 0.5), and `delta`, the $\delta$ that keeps $h_{t,i}$ positive (default 1e-8).
 * @returns The algorithm, started with `undefined`, one step per round.
 *
 * @example A rare feature gets a larger step
 * // The second feature is non-zero one round in five; the targets are exactly ⟨u, xₜ⟩ with u = (1, 2).
 * const xs = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 1]]
 * const loss = (t, w) => {
 *   const x = xs[(t - 1) % xs.length]
 *   const [a, b] = toFlat(w)
 *   const r = a * x[0] + b * x[1] - (x[0] + 2 * x[1])
 *   return { value: r * r, grad: [2 * r * x[0], 2 * r * x[1]] }
 * }
 * const options = { dim: 2, domain: { kind: 'ball', radius: 3 }, comparator: [1, 2], rounds: 100 }
 * const ada = run(onlineAdagrad(loss, { ...options, eta: 1 }), undefined, 1000)
 * const ogd = run(onlineGradientDescent(loss, { ...options, stepSize: ogdStepSize(6, 12) }), undefined, 1000)
 * print('AdaGrad: w =', ada.w, 'regret =', ada.regret)
 * print('online gradient descent: w =', ogd.w, 'regret =', ogd.regret)
 */
export function onlineAdagrad(
  loss: OnlineLoss,
  options: OnlineOptions & { dim?: number; eta?: number; delta?: number } = {},
): Algorithm<void, OnlineState> {
  const where = 'onlineAdagrad'
  const d = dimensionOf(options, options.dim, where)
  const eta = options.eta ?? 0.5
  const delta = options.delta ?? 1e-8
  const domain = options.domain
  if (domain?.kind === 'simplex' || (domain?.kind === 'ball' && domain.center !== undefined))
    throw new DomainError(where, `${where}: supports a box, a ball centred at 0, or no domain`)
  return onlineGame(
    'online-adagrad',
    where,
    loss,
    d,
    options,
    { squares: dense.vec(new Float64Array(d)) },
    (e, w, g) => {
      const n = Float64Array.from(dense.data(e.squares), (v, i) => v + g[i] * g[i])
      const h = n.map((v) => delta + Math.sqrt(v))
      const y = w.map((v, i) => v - (eta * g[i]) / h[i])
      const next =
        domain === undefined
          ? y
          : domain.kind === 'box'
            ? project(y, domain, where)
            : projectBallInNorm(y, h, null, domain.radius)
      return { w: next, stepSize: eta, extra: { squares: dense.vec(n) } }
    },
  )
}

// ── Online to batch ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Online-to-batch conversion (Cesa-Bianchi, Conconi and Gentile, 2004): the average
 * $\bar{\wvec} = \frac{1}{k} \sum_t \wvec_t$ of the $k$ iterates $\wvec_{\text{from}+1}, \dots, \wvec_T$ of an
 * online learner run once over an i.i.d. sample. For a convex loss its expected excess risk is at most
 * $\expect[R_T]/T$. `from` (default 0) drops a burn-in prefix (suffix averaging). Throws `DomainError` unless `from`
 * is an integer in $[0, T)$.
 *
 * @param iterates The points played, in order: an array of $T$ vectors of length $d$, or a $T \times d$ matrix (one
 *   row per round). Not modified.
 * @param options `from`, the number of leading iterates to drop (default 0).
 * @returns The average $\bar{\wvec}$, a vector of length $d$.
 *
 * @example The average, and the average of a suffix
 * print('average =', onlineToBatch([[1, 0], [2, 2], [3, 4]]))
 * print('without the first =', onlineToBatch([[1, 0], [2, 2], [3, 4]], { from: 1 }))
 *
 * @example Averaging the points of online gradient descent
 * // The losses (w − zₜ)² with zₜ alternating 1, 3, ...: the best fixed point is u = 2.
 * const loss = (t, w) => {
 *   const z = t % 2 === 1 ? 1 : 3
 *   const x = toFlat(w)[0]
 *   return { value: (x - z) ** 2, grad: [2 * (x - z)] }
 * }
 * const tr = trace(onlineGradientDescent(loss, { dim: 1, stepSize: ogdStepSize(4, 8), rounds: 200 }), undefined, 1000)
 * const played = tr.steps.slice(1).map((s) => s.played)
 * print('last points played =', played.slice(-4))
 * print('average =', onlineToBatch(played))
 * print('average of the last 100 =', onlineToBatch(played, { from: 100 }))
 */
export function onlineToBatch(iterates: readonly VectorLike[] | Tensor, options: { from?: number } = {}): Tensor {
  const where = 'onlineToBatch'
  const rows = Array.isArray(iterates)
    ? (iterates as readonly VectorLike[]).map((r) => dense.toF64(r, where))
    : (() => {
        const { data, m, n } = dense.toMatrixF64(iterates as Tensor, where)
        return Array.from({ length: m }, (_, i) => data.subarray(i * n, (i + 1) * n))
      })()
  const from = options.from ?? 0
  if (!(Number.isInteger(from) && from >= 0 && from < rows.length))
    throw new DomainError(where, `${where}: from must be an integer in [0, ${rows.length})`)
  const d = rows[0].length
  const mean = new Float64Array(d)
  for (let t = from; t < rows.length; t++) for (let i = 0; i < d; i++) mean[i] += rows[t][i]
  return dense.vec(mean.map((v) => v / (rows.length - from)))
}
