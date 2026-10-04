/**
 * Conjugate gradients: the linear method for symmetric positive definite systems Ax = b (equivalently, minimising
 * ½xᵀAx − bᵀx), and the nonlinear methods of Fletcher–Reeves and Polak–Ribière for general smooth objectives.
 */

import type { Tensor, Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { RunOptions, StartOptions } from '../options'
import { strongWolfeSearch, type LineSearchResult, type StrongWolfeOptions } from 'aifn-compute/optim/line-search'
import type { IterateState, ObjectiveFn, StoppingOptions, VectorLike } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'
import { operatorOf, type LinearOperator } from 'aifn-compute/numerics/linalg'

const { axpy, data, dot, norm, scale, sub, toF64, vec } = dense
type F64 = dense.F64

/** Which β the nonlinear method uses. */
export type ConjugateGradientVariant = 'fletcher-reeves' | 'polak-ribiere'

/** The state of `conjugateGradient`. */
export type ConjugateGradientState = IterateState & {
  grad: Vector
  gradNorm: number
  /** The search direction for the next step, p = −∇f + βp_prev. */
  direction: Vector
  /** The β used to build `direction` (0 on a restart). */
  beta: number
  /** True when `direction` was reset to −∇f (every n steps, or when it was not a descent direction). */
  restarted: boolean
  stepSize: number
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower f (x unchanged); the run stops. */
  stalled: boolean
}

/** Options for `conjugateGradient`. */
export type ConjugateGradientOptions = StoppingOptions & {
  /** Default `'polak-ribiere'` (with β⁺ = max(β, 0)). */
  variant?: ConjugateGradientVariant
  /** Restart with steepest descent every this many steps. Default n (the dimension). */
  restartEvery?: number
  /** Line-search options; default c₂ = 0.1, as the Fletcher–Reeves descent guarantee needs c₂ < ½. */
  lineSearchOptions?: StrongWolfeOptions
}

/**
 * Nonlinear conjugate gradients (Nocedal & Wright, §5.2): x ← x + αp with α from a strong Wolfe search, then
 * p ← −∇f + βp with β_FR = ‖g′‖²/‖g‖² (Fletcher & Reeves, 1964) or β_PR⁺ = max(0, g′ᵀ(g′ − g)/‖g‖²) (Polak & Ribière,
 * 1969). The first trial step length follows Nocedal & Wright eq. 3.60. `init` takes `{ x0 }`.
 */
export function conjugateGradient(
  f: ObjectiveFn,
  options: ConjugateGradientOptions = {},
): Algorithm<StartOptions, ConjugateGradientState> {
  const { variant = 'polak-ribiere', tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const lineOptions = { c2: 0.1, ...options.lineSearchOptions }
  const name = `conjugate-gradient-${variant}`
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
        direction: vec(scale(-1, grad)),
        beta: 0,
        restarted: true,
        stepSize: NaN,
        lineSearch: null,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (st) => {
      const n = st.x.shape[0]
      const restartEvery = options.restartEvery ?? n
      const x = data(st.x)
      const g = data(st.grad)
      const p = data(st.direction)
      const slope = dot(g, p)
      // Nocedal & Wright eq. 3.60: α₀ = α_prev · (g_prevᵀp_prev)/(gᵀp); the first step has length 1/‖g‖·min(1, ‖g‖).
      const previousSlope = st.lineSearch?.initialSlope
      const alpha0 =
        st.t === 0 || previousSlope === undefined || !(st.stepSize > 0)
          ? Math.min(1, 1 / Math.max(norm(g), 1e-300))
          : Math.min(1, (1.01 * st.stepSize * previousSlope) / slope)
      const found = strongWolfeSearch(f, x, st.value, g, p, { ...lineOptions, alpha0: alpha0 > 0 ? alpha0 : 1 })
      const g1 = found.grad
      const gg = dot(g, g)
      let beta = variant === 'fletcher-reeves' ? dot(g1, g1) / gg : Math.max(0, dot(g1, sub(g1, g)) / gg)
      let direction = axpy(beta, p, scale(-1, g1))
      let restarted = false
      if ((st.t + 1) % restartEvery === 0 || !(dot(direction, g1) < 0) || !Number.isFinite(beta)) {
        beta = 0
        direction = scale(-1, g1)
        restarted = true
      }
      const gradNorm = norm(g1)
      const stalled = found.result.alpha === 0
      return {
        t: st.t + 1,
        x: vec(found.x),
        value: found.value,
        grad: vec(g1),
        gradNorm,
        direction: vec(direction),
        beta,
        restarted,
        stepSize: found.result.alpha,
        lineSearch: found.result,
        stalled,
        evaluations: st.evaluations + found.result.evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(found.value, found.x, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Linear conjugate gradients.

/** The state of `linearConjugateGradient`. */
export type LinearConjugateGradientState = {
  t: number
  x: Vector
  /** The residual r = b − Ax. */
  residual: Vector
  residualNorm: number
  /** The next search direction p (A-conjugate to all earlier ones in exact arithmetic). */
  direction: Vector
  /** The step length α = rᵀr / pᵀAp of the last step (NaN at t = 0). */
  alpha: number
  /** β = r′ᵀr′ / rᵀr of the last step (NaN at t = 0). */
  beta: number
  /** The quadratic ½xᵀAx − bᵀx that CG minimises. */
  value: number
  /** Matrix–vector products so far. */
  products: number
  converged: boolean
  /** True when pᵀAp ≤ 0 was met: A is not positive definite and the run stops. */
  indefinite: boolean
}

/** Options for `linearConjugateGradient`. */
export type LinearConjugateGradientOptions = {
  /** Stop when ‖r‖ ≤ tolerance · ‖b‖. Default 1e-10. */
  tolerance?: number
}

/**
 * Conjugate gradients for Ax = b with A symmetric positive definite (Hestenes & Stiefel, 1952; Nocedal & Wright,
 * Algorithm 5.2): at most n steps in exact arithmetic, with the error in the A-norm reduced at the rate
 * (√κ − 1)/(√κ + 1) per step for condition number κ. `A` is an n×n matrix or a function v ↦ Av; `b` has length n.
 * `init` takes `{ x0 }` (default zeros).
 */
export function linearConjugateGradient(
  A: LinearOperator,
  b: VectorLike,
  options: LinearConjugateGradientOptions = {},
): Algorithm<{ x0?: VectorLike }, LinearConjugateGradientState> {
  const bb = toF64(b, 'linearConjugateGradient')
  const n = bb.length
  const apply = operatorOf(A, n, 'linearConjugateGradient')
  const bNorm = norm(bb)
  const tol = (options.tolerance ?? 1e-10) * (bNorm > 0 ? bNorm : 1)
  const quadratic = (x: F64, Ax: F64) => 0.5 * dot(x, Ax) - dot(bb, x)
  return {
    name: 'linear-conjugate-gradient',
    init: ({ x0 } = {}) => {
      const x = x0 === undefined ? new Float64Array(n) : toF64(x0, 'linearConjugateGradient')
      const Ax = apply(x)
      const r = sub(bb, Ax)
      const residualNorm = norm(r)
      return {
        t: 0,
        x: vec(x),
        residual: vec(r),
        residualNorm,
        direction: vec(Float64Array.from(r)),
        alpha: NaN,
        beta: NaN,
        value: quadratic(x, Ax),
        products: 1,
        converged: residualNorm <= tol,
        indefinite: false,
      }
    },
    step: (s) => {
      const x = data(s.x)
      const r = data(s.residual)
      const p = data(s.direction)
      const Ap = apply(p)
      const pAp = dot(p, Ap)
      if (!(pAp > 0)) return { ...s, t: s.t + 1, products: s.products + 1, indefinite: true }
      const rr = dot(r, r)
      const alpha = rr / pAp
      const x1 = axpy(alpha, p, x)
      const r1 = axpy(-alpha, Ap, r)
      const beta = dot(r1, r1) / rr
      const residualNorm = norm(r1)
      // The quadratic from the updated residual: ½xᵀAx − bᵀx = −½xᵀ(r + b).
      const value = -0.5 * dot(x1, axpy(1, bb, r1))
      return {
        t: s.t + 1,
        x: vec(x1),
        residual: vec(r1),
        residualNorm,
        direction: vec(axpy(beta, p, r1)),
        alpha,
        beta,
        value,
        products: s.products + 1,
        converged: residualNorm <= tol,
        indefinite: false,
      }
    },
    done: (s) => s.converged || s.indefinite,
  }
}

/** The result of `solveConjugateGradient`. */
export type ConjugateGradientSolution = {
  x: Tensor
  residualNorm: number
  steps: number
  converged: boolean
  indefinite: boolean
}

/**
 * Solves Ax = b (A symmetric positive definite, n×n or v ↦ Av) by linear conjugate gradients, running at most
 * `maxSteps` steps (default 10n, allowing for rounding). See `linearConjugateGradient`.
 */
export function solveConjugateGradient(
  A: LinearOperator,
  b: VectorLike,
  options: LinearConjugateGradientOptions & { x0?: VectorLike } & Pick<RunOptions, 'maxSteps'> = {},
): ConjugateGradientSolution {
  const alg = linearConjugateGradient(A, b, options)
  const n = toF64(b, 'solveConjugateGradient').length
  const s = run(alg, { x0: options.x0 }, options.maxSteps ?? 10 * n)
  return { x: s.x, residualNorm: s.residualNorm, steps: s.t, converged: s.converged, indefinite: s.indefinite }
}
