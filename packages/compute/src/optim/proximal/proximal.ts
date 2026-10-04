/**
 * Proximal and projected gradient methods for composite objectives F(x) = f(x) + g(x), with f smooth (value and
 * gradient) and g handled through its proximal operator prox_{ηg}(v) = argmin_x g(x) + ‖x − v‖²/(2η). Includes ISTA,
 * FISTA, projected gradient, and the proximal operators and projections used with them.
 *
 * Parikh & Boyd (2014), "Proximal Algorithms", Foundations and Trends in Optimization 1(3), §4.2 and §6;
 * Beck & Teboulle (2009), "A fast iterative shrinkage-thresholding algorithm for linear inverse problems", SIAM J.
 * Imaging Sciences 2(1).
 */

import type { Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import type { IterateState, ObjectiveFn, StoppingOptions, VectorLike } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

const { axpy, data, dot, norm, sub, toF64, vec } = dense
type F64 = dense.F64

/**
 * A closed convex function g given by its value and proximal operator. `value(x)` is g(x) (Infinity outside the
 * domain of an indicator); `prox(v, eta)` returns prox_{ηg}(v).
 */
export type Prox = {
  name: string
  value: (x: Vector) => number
  prox: (v: Vector, eta: number) => Vector
}

/** A bound given as a number (all coordinates) or a vector. */
export type Bound = number | VectorLike

const bound = (b: Bound, i: number, arr: F64 | null) => (arr ? arr[i] : (b as number))
const boundArray = (b: Bound, where: string) => (typeof b === 'number' ? null : toF64(b, where))

/** Projection onto the box lo ≤ x ≤ hi (bounds may be ±Infinity), coordinatewise clipping. */
export function projectBox(lo: Bound, hi: Bound): (x: Vector) => Vector {
  const l = boundArray(lo, 'projectBox')
  const h = boundArray(hi, 'projectBox')
  return (x) => vec(toF64(x, 'projectBox').map((v, i) => Math.min(Math.max(v, bound(lo, i, l)), bound(hi, i, h))))
}

/** Projection onto the nonnegative orthant, max(x, 0). */
export function projectNonnegative(): (x: Vector) => Vector {
  return (x) => vec(toF64(x, 'projectNonnegative').map((v) => Math.max(v, 0)))
}

/** Projection onto the Euclidean ball ‖x − c‖ ≤ r (centre default 0). */
export function projectBall(radius: number, center?: VectorLike): (x: Vector) => Vector {
  const c = center === undefined ? null : toF64(center, 'projectBall')
  return (x) => {
    const v = toF64(x, 'projectBall')
    const d = c ? sub(v, c) : v
    const length = norm(d)
    if (length <= radius) return vec(v)
    const k = radius / length
    return vec(d.map((di, i) => (c ? c[i] : 0) + k * di))
  }
}

/**
 * Project v[start … start + length) onto the simplex {x ≥ 0, Σx = z} in place, by sorting (Duchi, Shalev-Shwartz,
 * Singer & Chandra, 2008, "Efficient projections onto the l1-ball", Figure 1).
 */
function simplexInPlace(v: Float64Array, start: number, length: number, z: number): void {
  const u = v
    .slice(start, start + length)
    .sort()
    .reverse()
  let cumulative = 0
  let theta = 0
  for (let j = 0; j < length; j++) {
    cumulative += u[j]
    const t = (cumulative - z) / (j + 1)
    if (u[j] - t > 0) theta = t
  }
  for (let j = start; j < start + length; j++) v[j] = Math.max(v[j] - theta, 0)
}

/** Euclidean projection onto the simplex {x ≥ 0, Σx = z} (default z = 1), by sorting (Duchi et al. 2008, Figure 1). */
export function projectSimplex(z = 1): (x: Vector) => Vector {
  return (x) => {
    const v = Float64Array.from(toF64(x, 'projectSimplex'))
    simplexInPlace(v, 0, v.length, z)
    return vec(v)
  }
}

/**
 * Projection of every row of an n × c matrix (flattened row-major, length n·c) onto the simplex {x ≥ 0, Σx = z}: the
 * projection onto the product of n simplices, as for a matrix of class distributions.
 */
export function projectSimplexRows(columns: number, z = 1): (x: Vector) => Vector {
  return (x) => {
    const v = Float64Array.from(toF64(x, 'projectSimplexRows'))
    if (v.length % columns !== 0)
      throw new DomainError('projectSimplexRows', `projectSimplexRows: ${v.length} entries are not rows of ${columns}`)
    for (let start = 0; start < v.length; start += columns) simplexInPlace(v, start, columns, z)
    return vec(v)
  }
}

export function proxZero(): Prox {
  return { name: 'zero', value: () => 0, prox: (v) => v }
}

/** g(x) = λ‖x‖₁: the prox is soft thresholding, sign(v)·max(|v| − ηλ, 0). */
export function proxL1(lambda: number): Prox {
  return {
    name: 'l1',
    value: (x) => lambda * toF64(x, 'proxL1').reduce((a, v) => a + Math.abs(v), 0),
    prox: (v, eta) => vec(toF64(v, 'proxL1').map((vi) => Math.sign(vi) * Math.max(Math.abs(vi) - eta * lambda, 0))),
  }
}

/** g(x) = λ‖x‖₂ (the unsquared norm, as in the group lasso): the prox is block soft thresholding, max(0, 1 − ηλ/‖v‖)v. */
export function proxL2(lambda: number): Prox {
  return {
    name: 'l2',
    value: (x) => lambda * norm(toF64(x, 'proxL2')),
    prox: (v, eta) => {
      const a = toF64(v, 'proxL2')
      const length = norm(a)
      const k = length > 0 ? Math.max(0, 1 - (eta * lambda) / length) : 0
      return vec(a.map((ai) => k * ai))
    },
  }
}

/** g(x) = (λ/2)‖x‖₂² (ridge): the prox is shrinkage, v / (1 + ηλ). */
export function proxSquaredL2(lambda: number): Prox {
  return {
    name: 'squared-l2',
    value: (x) => {
      const a = toF64(x, 'proxSquaredL2')
      return 0.5 * lambda * dot(a, a)
    },
    prox: (v, eta) => vec(toF64(v, 'proxSquaredL2').map((vi) => vi / (1 + eta * lambda))),
  }
}

/** The indicator of lo ≤ x ≤ hi (0 inside, Infinity outside): the prox is the projection onto the box. */
export function proxBox(lo: Bound, hi: Bound): Prox {
  const l = boundArray(lo, 'proxBox')
  const h = boundArray(hi, 'proxBox')
  const project = projectBox(lo, hi)
  return {
    name: 'box',
    value: (x) => (toF64(x, 'proxBox').every((v, i) => v >= bound(lo, i, l) && v <= bound(hi, i, h)) ? 0 : Infinity),
    prox: (v) => project(v),
  }
}

/** The indicator of x ≥ 0: the prox is max(v, 0). */
export function proxNonnegative(): Prox {
  return {
    name: 'nonnegative',
    value: (x) => (toF64(x, 'proxNonnegative').every((v) => v >= 0) ? 0 : Infinity),
    prox: projectNonnegative(),
  }
}

/** The state of `proximalGradient` (and so of `ista`, `fista` and `projectedGradient`). */
export type ProximalGradientState = IterateState & {
  /** F(x) = f(x) + g(x) (the `value` field) splits into `smoothValue` f(x) and `penalty` g(x). */
  smoothValue: number
  penalty: number
  /** The point y where the gradient step was taken: x itself without acceleration, the extrapolated point in FISTA. */
  y: Vector
  /** ∇f(y). */
  gradY: Vector
  /** FISTA's momentum sequence t_k (1 without acceleration). */
  momentum: number
  /** The step size η = 1/L used on the last step (NaN at t = 0). */
  stepSize: number
  /** The Lipschitz estimate L for the next step (from backtracking, or 1/stepSize). */
  lipschitz: number
  /**
   * ‖G_η(y)‖ = ‖y − x⁺‖/η, the gradient mapping's norm; zero exactly at a minimiser of F, so it is the stopping test
   * (NaN at t = 0).
   */
  mappingNorm: number
  /** Backtracking trials on the last step: each L tried and whether the quadratic upper bound held. */
  trials: { lipschitz: number; accepted: boolean }[]
}

/** Options for `proximalGradient`. */
export type ProximalGradientOptions = Omit<StoppingOptions, 'tolerance'> & {
  /** Fixed step η (≤ 1/L for a gradient that is L-Lipschitz). Used as the first guess 1/L₀ with `backtracking`. */
  stepSize?: number
  /**
   * Find L by backtracking, L ← L·`increase` until f(x⁺) ≤ f(y) + ∇f(y)ᵀ(x⁺ − y) + (L/2)‖x⁺ − y‖² (Beck & Teboulle,
   * 2009, §3, with the estimate carried between steps). Default false.
   */
  backtracking?: boolean
  /** Growth factor of L in backtracking. Default 2. */
  increase?: number
  /** Nesterov / FISTA extrapolation. Default false (ISTA). */
  accelerated?: boolean
  /** Stop when the gradient mapping's norm is at most this. Default 1e-6. */
  tolerance?: number
}

/**
 * Proximal gradient descent on F = f + g: x⁺ = prox_{ηg}(y − η∇f(y)), with y = x (ISTA) or, with `accelerated`, the
 * FISTA extrapolation y_{k+1} = x_k + ((t_k − 1)/t_{k+1})(x_k − x_{k−1}), t_{k+1} = (1 + √(1 + 4t_k²))/2
 * (Beck & Teboulle, 2009). F(x_k) − F* is O(1/k) for ISTA and O(1/k²) for FISTA. `f` returns `{ value, grad }`; `g`
 * is a `Prox`. `init` takes `{ x0 }`.
 */
export function proximalGradient(
  f: ObjectiveFn,
  g: Prox,
  options: ProximalGradientOptions = {},
): Algorithm<StartOptions, ProximalGradientState> {
  const {
    stepSize: lr = 0.1,
    backtracking = false,
    increase = 2,
    accelerated = false,
    tolerance = 1e-6,
    divergeAbove = DEFAULT_DIVERGE,
  } = options
  const name = accelerated ? `fista-${g.name}` : `ista-${g.name}`
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const penalty = g.value(vec(x))
      return {
        t: 0,
        x: vec(x),
        value: value + penalty,
        smoothValue: value,
        penalty,
        y: vec(x),
        gradY: vec(grad),
        momentum: 1,
        stepSize: NaN,
        lipschitz: 1 / lr,
        mappingNorm: NaN,
        trials: [],
        evaluations: 1,
        converged: false,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s) => {
      const y = data(s.y)
      const gy = data(s.gradY)
      // f(y): at y = x it is the stored smooth value; FISTA's extrapolated y needs it only for backtracking.
      let evaluations = 0
      let fy = s.smoothValue
      if (backtracking && accelerated && s.t > 0) {
        fy = evaluate(f, y, name).value
        evaluations++
      }
      let L = s.lipschitz
      const trials: { lipschitz: number; accepted: boolean }[] = []
      let xNext: F64
      let fx: { value: number; grad: F64 }
      for (;;) {
        const eta = 1 / L
        xNext = toF64(g.prox(vec(axpy(-eta, gy, y)), eta), name)
        fx = evaluate(f, xNext, name)
        evaluations++
        if (!backtracking) break
        const d = sub(xNext, y)
        const ok = fx.value <= fy + dot(gy, d) + (L / 2) * dot(d, d) + 1e-12 * Math.abs(fy)
        trials.push({ lipschitz: L, accepted: ok })
        if (ok || trials.length >= 60) break
        L *= increase
      }
      const eta = 1 / L
      const x = data(s.x)
      const penalty = g.value(vec(xNext))
      const mappingNorm = norm(sub(y, xNext)) / eta
      let momentum = 1
      let yNext = xNext
      let gradY = fx.grad
      if (accelerated) {
        momentum = (1 + Math.sqrt(1 + 4 * s.momentum * s.momentum)) / 2
        yNext = axpy((s.momentum - 1) / momentum, sub(xNext, x), xNext)
        if (yNext.some((v, i) => v !== xNext[i])) {
          gradY = evaluate(f, yNext, name).grad
          evaluations++
        }
      }
      return {
        t: s.t + 1,
        x: vec(xNext),
        value: fx.value + penalty,
        smoothValue: fx.value,
        penalty,
        y: vec(yNext),
        gradY: vec(gradY),
        momentum,
        stepSize: eta,
        lipschitz: L,
        mappingNorm,
        trials,
        evaluations: s.evaluations + evaluations,
        converged: mappingNorm <= tolerance,
        diverged: divergedAt(fx.value, xNext, divergeAbove) || Number.isNaN(penalty),
      }
    },
    done: (s) => s.converged || s.diverged,
  }
}

/** ISTA (iterative shrinkage-thresholding): `proximalGradient` without acceleration. */
export function ista(
  f: ObjectiveFn,
  g: Prox,
  options: Omit<ProximalGradientOptions, 'accelerated'> = {},
): Algorithm<StartOptions, ProximalGradientState> {
  return proximalGradient(f, g, { ...options, accelerated: false })
}

/** FISTA (Beck & Teboulle, 2009): `proximalGradient` with Nesterov extrapolation. */
export function fista(
  f: ObjectiveFn,
  g: Prox,
  options: Omit<ProximalGradientOptions, 'accelerated'> = {},
): Algorithm<StartOptions, ProximalGradientState> {
  return proximalGradient(f, g, { ...options, accelerated: true })
}

/**
 * Projected gradient descent onto a closed convex set C: x ← P_C(x − η∇f(x)), i.e. proximal gradient with g the
 * indicator of C, whose prox is the projection `project` (e.g. `projectBox`, `projectBall`, `projectSimplex`).
 * `init` takes `{ x0 }` (projected first is not required: the first step lands in C).
 */
export function projectedGradient(
  f: ObjectiveFn,
  project: (x: Vector) => Vector,
  options: ProximalGradientOptions = {},
): Algorithm<StartOptions, ProximalGradientState> {
  const alg = proximalGradient(f, { name: 'projection', value: () => 0, prox: (v) => project(v) }, options)
  return { ...alg, name: options.accelerated ? 'accelerated-projected-gradient' : 'projected-gradient' }
}
