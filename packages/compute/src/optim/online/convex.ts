/**
 * Online convex optimisation: each round t = 1, …, T the learner plays wₜ in a convex set 𝒲, the environment reveals
 * a convex loss fₜ, and the learner pays fₜ(wₜ) and sees a subgradient gₜ ∈ ∂fₜ(wₜ). Online gradient descent, follow
 * the regularised leader (linearised, with L1 and L2 terms and an adaptive per-coordinate form), the online Newton step
 * and diagonal AdaGrad, as step-through algorithms whose states carry the regret against a fixed comparator u.
 */

import type { Schedule, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { eigh } from 'aifn-compute/numerics/linalg'

type F64 = dense.F64

/** The loss of round t (1-based) at a point w: its value fₜ(w) and a subgradient gₜ ∈ ∂fₜ(w). */
export type OnlineLoss = (t: number, w: Tensor) => { value: number; grad: VectorLike }

/**
 * The decision set 𝒲: a Euclidean ball (centred at 0 unless `center` is given), a box (bounds per coordinate or
 * shared), the probability simplex, or all of ℝᵈ when omitted.
 */
export type OnlineDomain =
  | { kind: 'ball'; radius: number; center?: VectorLike }
  | { kind: 'box'; lower: number | VectorLike; upper: number | VectorLike }
  | { kind: 'simplex' }

/** Options shared by the online convex optimisers. */
export type OnlineOptions = {
  /** The decision set (default ℝᵈ). */
  domain?: OnlineDomain
  /** The first point w₁ (default the projection of 0). */
  start?: VectorLike
  /** A fixed comparator u: the state then tracks Σ fₜ(u) and the regret Σ fₜ(wₜ) − fₜ(u). */
  comparator?: VectorLike
  /** The horizon T (default unbounded). */
  rounds?: number
}

/** One state of an online convex optimiser. */
export interface OnlineState extends Status {
  /** The point wₜ₊₁ played next. */
  w: Tensor
  /** The point played in the last round, wₜ (w₁ at t = 0). */
  played: Tensor
  /** The last round's loss fₜ(wₜ) (0 at t = 0). */
  loss: number
  /** The last subgradient gₜ (zeros at t = 0). */
  grad: Tensor
  /** Σ_{s ≤ t} fₛ(wₛ). */
  cumulative: number
  /** Σ_{s ≤ t} fₛ(u) for the comparator (NaN without one). */
  comparatorLoss: number
  /** The regret against the comparator (NaN without one). */
  regret: number
  /** The step size used in the last round (NaN at t = 0, or where a method has none). */
  stepSize: number
  done: boolean
}

// ── Projections ──────────────────────────────────────────────────────────────────────────────────────────────────────

function boundsOf(v: number | VectorLike, d: number, where: string): F64 {
  if (typeof v === 'number') return new Float64Array(d).fill(v)
  const b = dense.toF64(v, where)
  if (b.length !== d) throw new DomainError(where, `${where}: a box bound has ${b.length} entries for dimension ${d}`)
  return b
}

/** The Euclidean projection onto the probability simplex (sort-based; Held, Wolfe and Crowder, 1974). */
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

/** The Euclidean projection Π𝒲(v) onto a domain (the identity without one). */
export function projectOnto(v: VectorLike, domain?: OnlineDomain): Tensor {
  const x = dense.toF64(v, 'projectOnto')
  return dense.vec(project(x, domain, 'projectOnto'))
}

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
 * The projection in the norm ‖·‖_A of a symmetric positive definite A = V diag(λ) Vᵀ onto a ball centred at 0:
 * argmin_{‖w‖ ≤ r} (w − y)ᵀA(w − y) = (A + μI)⁻¹Ay with the μ ≥ 0 that puts it on the sphere (bisection on μ).
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

function dimensionOf(options: OnlineOptions, d: number | undefined, where: string): number {
  const fromStart = options.start === undefined ? undefined : dense.toF64(options.start, where).length
  const dim = d ?? fromStart
  if (dim === undefined || !Number.isInteger(dim) || dim < 1)
    throw new DomainError(where, `${where}: give the dimension or a start point`)
  return dim
}

/**
 * Drive an online optimiser: `update(state data, w, g, t)` returns the next point and the step size used; the driver
 * evaluates the loss, books the regret and keeps the method's own plain-data `extra` state.
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
 * Online gradient descent (Zinkevich, 2003): wₜ₊₁ = Π𝒲(wₜ − ηₜgₜ). With ηₜ = D/(G√t) (`ogdStepSize`) its regret on
 * convex losses is at most (3/2)DG√T, for a domain of diameter D and gradients bounded by G; with ηₜ = 1/(λt) on
 * λ-strongly convex losses it is O((G²/λ) log T).
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

/** The step-size schedule ηₜ = D/(G√t) of online gradient descent (t ≥ 1), with regret at most (3/2)DG√T. */
export function ogdStepSize(diameter: number, gradientBound: number): Schedule {
  if (!(diameter > 0 && gradientBound > 0))
    throw new DomainError('ogdStepSize', 'ogdStepSize: the diameter and gradient bound must be positive')
  return (t) => diameter / (gradientBound * Math.sqrt(Math.max(t, 1)))
}

// ── Follow the regularised leader ────────────────────────────────────────────────────────────────────────────────────

/** Options of `followTheRegularisedLeader`. */
export type FtrlOptions = OnlineOptions & {
  dim?: number
  /** The rate η of the quadratic regulariser ‖w‖²/(2η) (default 0.1), or α of the adaptive rate. */
  eta?: number
  /** An L1 penalty λ₁‖w‖₁, which makes coordinates exactly zero (default 0). */
  l1?: number
  /** An extra L2 penalty (λ₂/2)‖w‖² (default 0). */
  l2?: number
  /**
   * Per-coordinate adaptive rates (FTRL-Proximal; McMahan et al., 2013): coordinate i uses
   * ηₜ,ᵢ = α/(β + √Σₛ g²ₛ,ᵢ) and a proximal regulariser centred on the past points. Default false.
   */
  adaptive?: boolean
  /** β of the adaptive rate (default 1). */
  beta?: number
}

/**
 * Follow the regularised leader on the linearised losses (Shalev-Shwartz, 2012; McMahan, 2017):
 * wₜ₊₁ = argmin_w ⟨g₁:ₜ, w⟩ + ‖w‖²/(2η) + λ₁‖w‖₁ + (λ₂/2)‖w‖², which is coordinate-wise soft thresholding of the
 * cumulative gradient, then the projection onto a ball (a rescaling) or a box (a clip). Its regret is at most
 * ‖u‖²/(2η) + η Σ‖gₜ‖². With `adaptive` it is McMahan's FTRL-Proximal, the per-coordinate AdaGrad form.
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
 * The online Newton step (Hazan, Agarwal and Kale, 2007): Aₜ = εI + Σₛ≤ₜ gₛgₛᵀ, wₜ₊₁ = Π^{Aₜ}(wₜ − γ⁻¹Aₜ⁻¹gₜ), the
 * projection in the norm of Aₜ. On α-exp-concave losses with γ = ½ min(1/(4GD), α) its regret is
 * O(d(1/α + GD) log T). Supports a ball centred at 0, or no domain.
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
 * Diagonal AdaGrad as an online method (Duchi, Hazan and Singer, 2011): hₜ,ᵢ = δ + √Σₛ≤ₜ g²ₛ,ᵢ and
 * wₜ₊₁ = Π^{Hₜ}(wₜ − η Hₜ⁻¹gₜ), projected in the norm of Hₜ = diag(hₜ) (a clip for a box, a secular equation for a
 * ball centred at 0). Its regret is at most √2 D_∞ Σᵢ ‖g₁:T,ᵢ‖₂ at η = D_∞/√2, far below D G √T when gradients are
 * sparse.
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
 * Online-to-batch conversion (Cesa-Bianchi, Conconi and Gentile, 2004): the average w̄ = (1/k) Σ wₜ of the iterates
 * w_{from+1} … w_T [T, d] of an online learner run once over an i.i.d. sample. For a convex loss its expected excess
 * risk is at most E[R_T]/T. `from` (default 0) drops a burn-in prefix (suffix averaging).
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
