/**
 * Quasi-Newton methods: BFGS, which keeps a dense inverse-Hessian approximation, and L-BFGS, which keeps only the
 * last m curvature pairs (s, y) and applies the approximation by the two-loop recursion. Both take steps from a strong
 * Wolfe line search, which guarantees yᵀs > 0 and so keeps the approximation positive definite. OWL-QN extends
 * L-BFGS, with the same pairs and two-loop recursion, to f(x) + C‖x‖₁, whose L1 term is not differentiable at 0.
 */

import type { Matrix, Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import { strongWolfeSearch, type LineSearchResult, type StrongWolfeOptions } from 'aifn-compute/optim/line-search'
import type { IterateState, ObjectiveFn, StoppingOptions } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

const { axpy, data, dot, mat, matVec, norm, scale, sub, toF64, vec } = dense
type F64 = dense.F64

/** One curvature pair: the step s = x_{k+1} − x_k, the gradient change y = ∇f_{k+1} − ∇f_k, and ρ = 1/(yᵀs). */
export type CurvaturePair = { s: Vector; y: Vector; rho: number }

/** Options for `bfgs` and `lbfgs`. */
export type QuasiNewtonOptions = StoppingOptions & {
  /** Options for the strong Wolfe line search (default c₁ = 1e-4, c₂ = 0.9). */
  lineSearchOptions?: StrongWolfeOptions
}

/** The state of `bfgs`. */
export type BfgsState = IterateState & {
  grad: Vector
  gradNorm: number
  /** The inverse-Hessian approximation H_k (n×n, symmetric positive definite). */
  inverseHessian: Matrix
  /** The search direction p = −H∇f of the last step (zeros at t = 0). */
  direction: Vector
  /** The last curvature pair, or null at t = 0. */
  pair: CurvaturePair | null
  /** True when the last update was skipped because yᵀs was not safely positive. */
  skipped: boolean
  stepSize: number
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower f (x unchanged); the run stops. */
  stalled: boolean
}

/** The shared outer step: search along p, then report the pair. */
function searchAlong(f: ObjectiveFn, x: F64, value: number, g: F64, p: F64, options: StrongWolfeOptions | undefined) {
  const found = strongWolfeSearch(f, x, value, g, p, options)
  const s = sub(found.x, x)
  const y = sub(found.grad, g)
  return { found, s, y, sy: dot(s, y) }
}

/**
 * BFGS (Broyden, Fletcher, Goldfarb and Shanno, 1970; Nocedal & Wright, Algorithm 6.1): p = −H∇f, a strong Wolfe step,
 * then the inverse update H ← (I − ρsyᵀ)H(I − ρysᵀ) + ρssᵀ. Before the first update H₀ = I is rescaled to
 * (yᵀs / yᵀy)I (eq. 6.20). An update with yᵀs ≤ 10⁻¹⁰‖s‖‖y‖ is skipped and flagged. `init` takes `{ x0 }`.
 */
export function bfgs(f: ObjectiveFn, options: QuasiNewtonOptions = {}): Algorithm<StartOptions, BfgsState> {
  const { tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'bfgs'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const n = x.length
      const { value, grad } = evaluate(f, x, name)
      const H = new Float64Array(n * n)
      for (let i = 0; i < n; i++) H[i * n + i] = 1
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        inverseHessian: mat(H, n, n),
        direction: vec(new Float64Array(n)),
        pair: null,
        skipped: false,
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
      const x = data(st.x)
      const g = data(st.grad)
      let H = data(st.inverseHessian)
      const p = scale(-1, matVec(H, g, n, n))
      const { found, s, y, sy } = searchAlong(f, x, st.value, g, p, options.lineSearchOptions)
      const skipped = !(sy > 1e-10 * norm(s) * norm(y))
      if (!skipped) {
        if (st.t === 0) H = scale(sy / dot(y, y), H)
        const rho = 1 / sy
        const Hy = matVec(H, y, n, n)
        const yHy = dot(y, Hy)
        // Expanded form of (I − ρsyᵀ)H(I − ρysᵀ) + ρssᵀ, using the symmetry of H.
        const next = new Float64Array(n * n)
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++)
            next[i * n + j] = H[i * n + j] - rho * (s[i] * Hy[j] + Hy[i] * s[j]) + (rho * rho * yHy + rho) * s[i] * s[j]
        H = next
      }
      const gradNorm = norm(found.grad)
      return {
        t: st.t + 1,
        x: vec(found.x),
        value: found.value,
        grad: vec(found.grad),
        gradNorm,
        inverseHessian: mat(H === data(st.inverseHessian) ? Float64Array.from(H) : H, n, n),
        direction: vec(p),
        pair: { s: vec(s), y: vec(y), rho: 1 / sy },
        skipped,
        stepSize: found.result.alpha,
        lineSearch: found.result,
        stalled: found.result.alpha === 0,
        evaluations: st.evaluations + found.result.evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(found.value, found.x, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

/** The state of `lbfgs`. */
export type LbfgsState = IterateState & {
  grad: Vector
  gradNorm: number
  /** The stored curvature pairs, oldest first (at most `memory`). */
  pairs: CurvaturePair[]
  /** The initial-Hessian scale γ = sᵀy / yᵀy of the newest pair (1 before any pair). */
  gamma: number
  /** The search direction of the last step (zeros at t = 0). */
  direction: Vector
  stepSize: number
  lineSearch: LineSearchResult | null
  /** True when the last pair was not stored because yᵀs was not safely positive. */
  skipped: boolean
  /** True when the last line search could not lower f (x unchanged); the run stops. */
  stalled: boolean
}

/** Options for `lbfgs`. */
export type LbfgsOptions = QuasiNewtonOptions & {
  /** Number of curvature pairs kept, m. Default 10. */
  memory?: number
}

/**
 * The two-loop recursion (Nocedal & Wright, Algorithm 7.4): returns H·q for the L-BFGS inverse Hessian built from the
 * pairs (oldest first) on the initial matrix γI.
 */
function twoLoop(q0: F64, pairs: { s: F64; y: F64; rho: number }[], gamma: number): F64 {
  let q = Float64Array.from(q0)
  const alphas = new Array<number>(pairs.length)
  for (let i = pairs.length - 1; i >= 0; i--) {
    const { s, y, rho } = pairs[i]
    alphas[i] = rho * dot(s, q)
    q = axpy(-alphas[i], y, q)
  }
  let r = scale(gamma, q)
  for (let i = 0; i < pairs.length; i++) {
    const { s, y, rho } = pairs[i]
    const beta = rho * dot(y, r)
    r = axpy(alphas[i] - beta, s, r)
  }
  return r
}

/**
 * Limited-memory BFGS (Liu & Nocedal, 1989; Nocedal & Wright, Algorithm 7.5): the direction −H∇f comes from the
 * two-loop recursion over the last m pairs with H₀ = γI, γ = sᵀy/yᵀy of the newest pair; steps satisfy the strong
 * Wolfe conditions. The first step, with no pairs, is scaled to length 1 along −∇f. `init` takes `{ x0 }`.
 */
export function lbfgs(f: ObjectiveFn, options: LbfgsOptions = {}): Algorithm<StartOptions, LbfgsState> {
  const { memory = 10, tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'lbfgs'
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
        pairs: [],
        gamma: 1,
        direction: vec(new Float64Array(x.length)),
        stepSize: NaN,
        lineSearch: null,
        skipped: false,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (st) => lbfgsStep(f, st, memory, tolerance, divergeAbove, options.lineSearchOptions),
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

/** The L-BFGS direction −H·v from the stored pairs; with no pairs, −v scaled to unit length. */
function lbfgsDirection(v: F64, st: { pairs: CurvaturePair[]; gamma: number }): F64 {
  const pairs = st.pairs.map((q) => ({ s: data(q.s), y: data(q.y), rho: q.rho }))
  // Without curvature information, a unit-length steepest-descent step (as scipy's L-BFGS-B starts).
  const gamma = pairs.length ? st.gamma : 1 / Math.max(norm(v), 1e-300)
  return scale(-1, twoLoop(v, pairs, gamma))
}

/** The history after a step: the pair (s, y) is stored, the oldest dropped beyond `memory`, unless yᵀs ≤ 10⁻¹⁰‖s‖‖y‖. */
function updateHistory(st: { pairs: CurvaturePair[]; gamma: number }, s: F64, y: F64, memory: number) {
  const sy = dot(s, y)
  const skipped = !(sy > 1e-10 * norm(s) * norm(y))
  if (skipped) return { pairs: st.pairs, gamma: st.gamma, skipped }
  return { pairs: [...st.pairs, { s: vec(s), y: vec(y), rho: 1 / sy }].slice(-memory), gamma: sy / dot(y, y), skipped }
}

/** One L-BFGS step: the two-loop direction, a strong Wolfe step, and the history update. */
function lbfgsStep<S extends LbfgsState>(
  f: ObjectiveFn,
  st: S,
  memory: number,
  tolerance: number,
  divergeAbove: number,
  lineSearchOptions: StrongWolfeOptions | undefined,
): S {
  const x = data(st.x)
  const g = data(st.grad)
  const p = lbfgsDirection(g, st)
  const { found, s, y } = searchAlong(f, x, st.value, g, p, lineSearchOptions)
  const history = updateHistory(st, s, y, memory)
  const gradNorm = norm(found.grad)
  return {
    ...st,
    t: st.t + 1,
    x: vec(found.x),
    value: found.value,
    grad: vec(found.grad),
    gradNorm,
    ...history,
    direction: vec(p),
    stepSize: found.result.alpha,
    lineSearch: found.result,
    stalled: found.result.alpha === 0,
    evaluations: st.evaluations + found.result.evaluations,
    converged: gradNorm <= tolerance,
    diverged: divergedAt(found.value, found.x, divergeAbove),
  }
}

// ── OWL-QN ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `owlqn`. */
export type OwlqnOptions = LbfgsOptions & {
  /** The L1 strength C ≥ 0 in F(x) = f(x) + C‖x‖₁. Default 0, where OWL-QN is L-BFGS. */
  l1?: number
  /** The sufficient-decrease constant of the backtracking search, c₁ (default 1e-4). */
  decrease?: number
  /** The backtracking factor (default 0.5) and the most halvings per step (default 50). */
  backtrack?: number
  maxBacktracks?: number
}

/** The state of `owlqn`. */
export type OwlqnState = LbfgsState & {
  /** `value` is F(x) = f(x) + C‖x‖₁; `smoothValue` is f(x). */
  smoothValue: number
  /** ∇f(x), the smooth part's gradient, from which the curvature pairs are taken. `grad` is the pseudo-gradient ◇F. */
  smoothGrad: Vector
  /** The number of non-zero coordinates of x. */
  nonzero: number
  /** Backtracking halvings in the last step (0 when the first trial point was accepted). */
  backtracks: number
}

/**
 * The pseudo-gradient ◇F of F(x) = f(x) + C‖x‖₁ (Andrew & Gao 2007, eq. 4) from g = ∇f(x): ∂ᵢf + C·sign(xᵢ) where
 * xᵢ ≠ 0; at xᵢ = 0, the one-sided derivative ∂ᵢf ± C that is negative on its side if there is one, else 0. −◇F is
 * the steepest-descent direction of F, and ◇F = 0 exactly where x minimises F.
 */
export function pseudoGradient(x: ArrayLike<number>, g: ArrayLike<number>, l1: number): F64 {
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) {
    if (x[i] > 0) out[i] = g[i] + l1
    else if (x[i] < 0) out[i] = g[i] - l1
    else if (g[i] + l1 < 0) out[i] = g[i] + l1
    else if (g[i] - l1 > 0) out[i] = g[i] - l1
  }
  return out
}

const l1Norm = (x: ArrayLike<number>) => {
  let a = 0
  for (let i = 0; i < x.length; i++) a += Math.abs(x[i])
  return a
}
const countNonzero = (x: ArrayLike<number>) => {
  let k = 0
  for (let i = 0; i < x.length; i++) if (x[i] !== 0) k++
  return k
}

/**
 * Orthant-wise limited-memory quasi-Newton (OWL-QN; Andrew & Gao 2007, "Scalable training of L1-regularized
 * log-linear models", ICML) for F(x) = f(x) + C‖x‖₁ with f smooth. Each step:
 *
 * 1. the pseudo-gradient v = ◇F(x) (`pseudoGradient`);
 * 2. the L-BFGS direction d = −H v from the same two-loop recursion and pair history as `lbfgs`, with the pairs taken
 *    from ∇f alone (the L1 term adds no curvature);
 * 3. d projected onto the orthant of steepest descent: dᵢ = 0 wherever sign(dᵢ) ≠ sign(−vᵢ);
 * 4. the orthant ξᵢ = sign(xᵢ), or sign(−vᵢ) where xᵢ = 0;
 * 5. a backtracking search along the projected path x(α) = π(x + αd; ξ), where π zeroes every coordinate that leaves
 *    its orthant, accepting F(x(α)) ≤ F(x) + c₁ vᵀ(x(α) − x).
 *
 * Coordinates that cross zero stop at zero, so the iterates are sparse. The run converges when ‖◇F‖ ≤ tolerance. With
 * C = 0 the steps are exactly those of `lbfgs` (strong Wolfe search included), as in libLBFGS. `init` takes `{ x0 }`.
 */
export function owlqn(f: ObjectiveFn, options: OwlqnOptions = {}): Algorithm<StartOptions, OwlqnState> {
  const {
    memory = 10,
    tolerance = DEFAULT_TOLERANCE,
    divergeAbove = DEFAULT_DIVERGE,
    l1 = 0,
    decrease = 1e-4,
    backtrack = 0.5,
    maxBacktracks = 50,
  } = options
  const name = 'owlqn'
  if (!(l1 >= 0)) throw new DomainError(name, `${name}: l1 must be ≥ 0, got ${l1}`)
  const total = (value: number, x: ArrayLike<number>) => value + (l1 > 0 ? l1 * l1Norm(x) : 0)
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const v = pseudoGradient(x, grad, l1)
      const F = total(value, x)
      const gradNorm = norm(v)
      return {
        t: 0,
        x: vec(x),
        value: F,
        grad: vec(v),
        gradNorm,
        smoothValue: value,
        smoothGrad: vec(grad),
        nonzero: countNonzero(x),
        pairs: [],
        gamma: 1,
        direction: vec(new Float64Array(x.length)),
        stepSize: NaN,
        lineSearch: null,
        skipped: false,
        stalled: false,
        backtracks: 0,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(F, x, divergeAbove),
      }
    },
    step: (st) => {
      if (l1 === 0) {
        // Plain L-BFGS: the smooth objective is the whole objective, so value and grad carry over unchanged.
        const next = lbfgsStep(f, st, memory, tolerance, divergeAbove, options.lineSearchOptions)
        return {
          ...next,
          backtracks: 0,
          smoothValue: next.value,
          smoothGrad: next.grad,
          nonzero: countNonzero(data(next.x)),
        }
      }
      const x = data(st.x)
      const g = data(st.smoothGrad)
      const v = data(st.grad)
      const n = x.length
      let d = lbfgsDirection(v, st)
      for (let i = 0; i < n; i++) if (d[i] * v[i] >= 0) d[i] = 0
      // If the projection leaves no descent (possible with a poor H), fall back to steepest descent of F.
      if (!(dot(d, v) < 0)) d = scale(-1 / Math.max(norm(v), 1e-300), v)
      const orthant = new Float64Array(n)
      for (let i = 0; i < n; i++) orthant[i] = x[i] !== 0 ? Math.sign(x[i]) : Math.sign(-v[i])
      let alpha = 1
      let evaluations = 0
      let accepted: { x: F64; value: number; grad: F64; F: number } | null = null
      let k = 0
      for (; k <= maxBacktracks; k++) {
        const xn = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          const z = x[i] + alpha * d[i]
          xn[i] = Math.sign(z) === orthant[i] ? z : 0
        }
        const out = evaluate(f, xn, name)
        evaluations++
        const F = total(out.value, xn)
        let decreaseTerm = 0
        for (let i = 0; i < n; i++) decreaseTerm += v[i] * (xn[i] - x[i])
        if (Number.isFinite(F) && F <= st.value + decrease * decreaseTerm) {
          accepted = { x: xn, value: out.value, grad: out.grad, F }
          break
        }
        alpha *= backtrack
      }
      if (!accepted)
        return { ...st, t: st.t + 1, stalled: true, backtracks: k, evaluations: st.evaluations + evaluations }
      const s = sub(accepted.x, x)
      const y = sub(accepted.grad, g)
      const history = updateHistory(st, s, y, memory)
      const pv = pseudoGradient(accepted.x, accepted.grad, l1)
      const gradNorm = norm(pv)
      return {
        t: st.t + 1,
        x: vec(accepted.x),
        value: accepted.F,
        grad: vec(pv),
        gradNorm,
        smoothValue: accepted.value,
        smoothGrad: vec(accepted.grad),
        nonzero: countNonzero(accepted.x),
        ...history,
        direction: vec(d),
        stepSize: alpha,
        lineSearch: null,
        backtracks: k,
        stalled: false,
        evaluations: st.evaluations + evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(accepted.F, accepted.x, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}
