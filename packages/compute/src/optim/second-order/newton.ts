/**
 * Second-order methods with a supplied Hessian: damped Newton (a Newton direction, made a descent direction by adding
 * a multiple of the identity when the Hessian is not positive definite, and a line search), and a trust-region method
 * with the dogleg step.
 */

import { cholesky, choleskySolve } from 'aifn-compute/numerics/linalg'
import type { Matrix, Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import {
  backtrackingSearch,
  strongWolfeSearch,
  type BacktrackingOptions,
  type LineSearchResult,
} from 'aifn-compute/optim/line-search'
import type { Hessian, IterateState, ObjectiveFn, StoppingOptions } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'

const { allFinite, axpy, data, dot, mat, matVec, norm, scale, sub, toF64, toMatrixF64, vec } = dense
type F64 = dense.F64

/** Evaluates the Hessian as a working array, checked for shape. */
function hessianAt(h: Hessian, x: F64, where: string): F64 {
  return toMatrixF64(h(vec(x)), where, x.length, x.length).data
}

/**
 * Solves (H + τI)p = −g with the smallest τ on the ladder 0, β, 2β, 4β, … (β = 10⁻³·max|Hᵢᵢ|, or 10⁻³) that makes
 * H + τI positive definite: Nocedal & Wright (2006), Algorithm 3.3, "Cholesky with added multiple of the identity".
 */
export function shiftedNewtonDirection(H: F64, g: F64): { direction: F64; shift: number } | null {
  const n = g.length
  if (!allFinite(H)) return null
  let minDiagonal = Infinity
  let maxDiagonal = 0
  for (let i = 0; i < n; i++) {
    minDiagonal = Math.min(minDiagonal, H[i * n + i])
    maxDiagonal = Math.max(maxDiagonal, Math.abs(H[i * n + i]))
  }
  const beta = 1e-3 * (maxDiagonal > 0 ? maxDiagonal : 1)
  let tau = minDiagonal > 0 ? 0 : beta - minDiagonal
  for (let attempt = 0; attempt < 80; attempt++) {
    const shifted = Float64Array.from(H)
    for (let i = 0; i < n; i++) shifted[i * n + i] += tau
    const c = cholesky(mat(shifted, n, n), { jitter: false })
    if (!c.failed) {
      const solved = data(choleskySolve(c.L, vec(scale(-1, g))))
      return { direction: Float64Array.from(solved), shift: tau }
    }
    tau = Math.max(2 * tau, beta)
  }
  return null
}

/** The state of `newton`. */
export type NewtonState = IterateState & {
  grad: Vector
  gradNorm: number
  /** ∇²f(x). */
  hessian: Matrix
  /** The last Newton direction p = −(H + τI)⁻¹∇f (zeros at t = 0). */
  direction: Vector
  /** The shift τ added to the Hessian's diagonal on the last step (0 when it was positive definite). */
  shift: number
  /** The accepted step length on the last step (1 is a full Newton step); NaN at t = 0. */
  stepSize: number
  /** The Newton decrement squared, −∇fᵀp = ∇fᵀ(H + τI)⁻¹∇f, on the last step: twice the predicted decrease. */
  decrement: number
  /** The last line search with its trials; null for a pure Newton step or at t = 0. */
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower f (x unchanged); the run stops. */
  stalled: boolean
}

/** Options for `newton`. */
export type NewtonOptions = StoppingOptions & {
  /** The Hessian ∇²f(x). */
  hessian: Hessian
  /**
   * `'backtracking'` (default) damps the step with an Armijo backtracking search from α = 1; `'strong-wolfe'` uses a
   * strong Wolfe search; `'none'` takes the full Newton step (pure Newton's method).
   */
  lineSearch?: 'backtracking' | 'strong-wolfe' | 'none'
  lineSearchOptions?: BacktrackingOptions
}

/**
 * Damped Newton's method: p = −(H + τI)⁻¹∇f(x), with τ = 0 when the Hessian H is positive definite and otherwise the
 * smallest shift that makes it so (Nocedal & Wright, Algorithm 3.3), then x ← x + αp with α from a backtracking line
 * search (α = 1 near a minimiser, where convergence is quadratic). `init` takes `{ x0 }`.
 *
 * If no shift makes H + τI factorable (a non-finite Hessian), the state is flagged `diverged`.
 */
export function newton(f: ObjectiveFn, options: NewtonOptions): Algorithm<StartOptions, NewtonState> {
  const {
    hessian,
    lineSearch = 'backtracking',
    tolerance = DEFAULT_TOLERANCE,
    divergeAbove = DEFAULT_DIVERGE,
  } = options
  const name = lineSearch === 'none' ? 'newton' : 'damped-newton'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const H = hessianAt(hessian, x, name)
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        hessian: mat(H, x.length, x.length),
        direction: vec(new Float64Array(x.length)),
        shift: 0,
        stepSize: NaN,
        decrement: NaN,
        lineSearch: null,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s) => {
      const n = s.x.shape[0]
      const x = data(s.x)
      const g = data(s.grad)
      const solved = shiftedNewtonDirection(data(s.hessian), g)
      if (!solved) return { ...s, t: s.t + 1, diverged: true }
      const { direction: p, shift } = solved
      let next: F64
      let value: number
      let grad: F64
      let alpha = 1
      let search: LineSearchResult | null = null
      let evaluations = 0
      if (lineSearch === 'none') {
        next = axpy(1, p, x)
        ;({ value, grad } = evaluate(f, next, name))
        evaluations = 1
      } else {
        const found =
          lineSearch === 'backtracking'
            ? backtrackingSearch(f, x, s.value, g, p, options.lineSearchOptions)
            : strongWolfeSearch(f, x, s.value, g, p)
        ;({ x: next, value, grad } = found)
        search = found.result
        alpha = found.result.alpha
        evaluations = found.result.evaluations
      }
      const gradNorm = norm(grad)
      return {
        t: s.t + 1,
        x: vec(next),
        value,
        grad: vec(grad),
        gradNorm,
        hessian: mat(hessianAt(hessian, next, name), n, n),
        direction: vec(p),
        shift,
        stepSize: alpha,
        decrement: -dot(g, p),
        lineSearch: search,
        stalled: search !== null && alpha === 0,
        evaluations: s.evaluations + evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, next, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Trust region with the dogleg step.

/** Which step the dogleg took. */
export type DoglegKind = 'newton' | 'dogleg' | 'cauchy' | 'none'

/** The state of `trustRegion`. */
export type TrustRegionState = IterateState & {
  grad: Vector
  gradNorm: number
  hessian: Matrix
  /** The trust-region radius Δ for the next step. */
  radius: number
  /** The step p proposed on the last step (whether or not it was accepted). */
  step: Vector
  /** `newton` (the full step fits), `dogleg` (the path meets the boundary), `cauchy` (steepest descent). */
  stepKind: DoglegKind
  /** Actual reduction f(x) − f(x + p) on the last step. */
  actual: number
  /** Predicted reduction of the quadratic model, −(gᵀp + ½pᵀHp). */
  predicted: number
  /** ρ = actual / predicted: near 1 when the model is good. */
  ratio: number
  /** Whether the last proposed step was accepted (x moved). */
  accepted: boolean
}

/** Options for `trustRegion`. */
export type TrustRegionOptions = StoppingOptions & {
  hessian: Hessian
  /** Initial radius Δ₀. Default 1. */
  radius?: number
  /** Largest radius. Default 1e3. */
  maxRadius?: number
  /** Accept a step when ρ > η. Default 0.15 (Nocedal & Wright suggest η ∈ [0, ¼)). */
  eta?: number
}

/**
 * The dogleg step for the model m(p) = gᵀp + ½pᵀBp within ‖p‖ ≤ Δ (Nocedal & Wright, §4.1): the full Newton step if
 * B is positive definite and the step fits; otherwise the path from the origin to the Cauchy point and on towards the
 * Newton step, cut at the boundary. When B is not positive definite the Cauchy point is used (eq. 4.11–4.12).
 */
function dogleg(g: F64, B: F64, radius: number): { p: F64; kind: DoglegKind } {
  const n = g.length
  const gNorm = norm(g)
  if (gNorm === 0) return { p: new Float64Array(n), kind: 'none' }
  const Bg = matVec(B, g, n, n)
  const gBg = dot(g, Bg)
  // Cauchy point: minimise the model along −g within the region.
  const tauC = gBg <= 0 ? 1 : Math.min(1, gNorm ** 3 / (radius * gBg))
  const cauchy = scale((-tauC * radius) / gNorm, g)
  const c = cholesky(mat(B, n, n), { jitter: false })
  if (c.failed) return { p: cauchy, kind: 'cauchy' }
  const newtonStep = Float64Array.from(data(choleskySolve(c.L, vec(scale(-1, g)))))
  if (norm(newtonStep) <= radius) return { p: newtonStep, kind: 'newton' }
  // The unconstrained minimiser along −g.
  const pu = scale(-dot(g, g) / gBg, g)
  if (norm(pu) >= radius) return { p: scale(-radius / gNorm, g), kind: 'cauchy' }
  // Solve ‖pu + s(pb − pu)‖ = Δ for s in [0, 1].
  const d = sub(newtonStep, pu)
  const a = dot(d, d)
  const b = 2 * dot(pu, d)
  const cc = dot(pu, pu) - radius * radius
  const sStar = (-b + Math.sqrt(b * b - 4 * a * cc)) / (2 * a)
  return { p: axpy(sStar, d, pu), kind: 'dogleg' }
}

/**
 * A trust-region method with the dogleg step (Nocedal & Wright, Algorithm 4.1): each step minimises the quadratic
 * model within radius Δ, compares the actual and predicted reductions (ρ), accepts the step when ρ > η, and grows
 * Δ (ρ > ¾ at the boundary) or shrinks it (ρ < ¼). Rejected steps leave x unchanged. `init` takes `{ x0 }`.
 */
export function trustRegion(f: ObjectiveFn, options: TrustRegionOptions): Algorithm<StartOptions, TrustRegionState> {
  const {
    hessian,
    radius: radius0 = 1,
    maxRadius = 1e3,
    eta = 0.15,
    tolerance = DEFAULT_TOLERANCE,
    divergeAbove = DEFAULT_DIVERGE,
  } = options
  const name = 'trust-region-dogleg'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        hessian: mat(hessianAt(hessian, x, name), x.length, x.length),
        radius: radius0,
        step: vec(new Float64Array(x.length)),
        stepKind: 'none',
        actual: NaN,
        predicted: NaN,
        ratio: NaN,
        accepted: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s) => {
      const n = s.x.shape[0]
      const x = data(s.x)
      const g = data(s.grad)
      const B = data(s.hessian)
      if (!allFinite(B)) return { ...s, t: s.t + 1, diverged: true }
      const { p, kind } = dogleg(g, B, s.radius)
      const predicted = -(dot(g, p) + 0.5 * dot(p, matVec(B, p, n, n)))
      const trial = axpy(1, p, x)
      const e = evaluate(f, trial, name)
      const actual = s.value - e.value
      // A non-finite trial value gives ρ = NaN, which fails every test below: the step is rejected and Δ shrinks.
      const ratio = predicted > 0 ? actual / predicted : actual >= 0 ? 1 : -1
      const pNorm = norm(p)
      let radius = s.radius
      if (!(ratio >= 0.25)) radius = 0.25 * pNorm
      else if (ratio > 0.75 && Math.abs(pNorm - s.radius) <= 1e-12 * s.radius)
        radius = Math.min(2 * s.radius, maxRadius)
      const accepted = ratio > eta
      const common = { t: s.t + 1, radius, step: vec(p), stepKind: kind, actual, predicted, ratio, accepted }
      if (!accepted) {
        // Δ collapsing to zero means no step can make progress at x (e.g. rounding at the minimum).
        return { ...s, ...common, evaluations: s.evaluations + 1, converged: s.converged || radius < 1e-14 }
      }
      const gradNorm = norm(e.grad)
      return {
        ...common,
        x: vec(trial),
        value: e.value,
        grad: vec(e.grad),
        gradNorm,
        hessian: mat(hessianAt(hessian, trial, name), n, n),
        evaluations: s.evaluations + 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(e.value, trial, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged,
  }
}
