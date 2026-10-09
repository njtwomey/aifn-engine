/**
 * Quasi-Newton methods: BFGS, which keeps a dense inverse-Hessian approximation, and L-BFGS, which keeps only the
 * last $m$ curvature pairs $(\svec, \yvec)$ and applies the approximation by the two-loop recursion. Both take steps
 * from a strong Wolfe line search, which guarantees $\yvec^\top\svec > 0$ and so keeps the approximation positive
 * definite. OWL-QN extends L-BFGS, with the same pairs and two-loop recursion, to
 * $f(\xvec) + C\lVert \xvec \rVert_1$, whose L1 term is not differentiable at 0.
 *
 * They need only the gradient: the curvature is learnt from how it changes between iterates. A pair with
 * $\yvec^\top\svec \le 10^{-10}\lVert \svec \rVert\lVert \yvec \rVert$ is skipped and flagged rather than allowed
 * to spoil the approximation. Non-convergence, divergence and a line search that cannot lower $f$ (`stalled`) are
 * reported in the state, not thrown. Nocedal & Wright (2006), "Numerical Optimization", 2nd ed., chapters 6 and 7.
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

/**
 * One curvature pair: the step `s` ($\svec = \xvec_{k+1} - \xvec_k$), the gradient change `y`
 * ($\yvec = \nabla f_{k+1} - \nabla f_k$), and `rho` ($\rho = 1/(\yvec^\top\svec)$).
 */
export type CurvaturePair = { s: Vector; y: Vector; rho: number }

/** Options for `bfgs` and `lbfgs`. */
export type QuasiNewtonOptions = StoppingOptions & {
  /** Options for the strong Wolfe line search (default $c_1 = 10^{-4}$, $c_2 = 0.9$). */
  lineSearchOptions?: StrongWolfeOptions
}

/** The state of `bfgs`. */
export type BfgsState = IterateState & {
  /** $\nabla f(\xvec)$. */
  grad: Vector
  /** $\lVert \nabla f(\xvec) \rVert_2$, compared with `tolerance`. */
  gradNorm: number
  /** The inverse-Hessian approximation $\Hmat_k$ ($n \times n$, symmetric positive definite). */
  inverseHessian: Matrix
  /** The search direction $\pvec = -\Hmat\nabla f$ of the last step (zeros at $t = 0$). */
  direction: Vector
  /** The last curvature pair (recorded even when the update was skipped), or null at $t = 0$. */
  pair: CurvaturePair | null
  /** True when the last update was skipped because $\yvec^\top\svec$ was not safely positive. */
  skipped: boolean
  /** The step length $\alpha$ the line search accepted on the last step; NaN at $t = 0$. */
  stepSize: number
  /** The last line search, with its trial points; null at $t = 0$. */
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower $f$ ($\xvec$ unchanged); the run stops. */
  stalled: boolean
}

/**
 * The shared outer step: search along $\pvec$, then report the pair.
 *
 * @param f The objective, returning value and gradient.
 * @param x The current point $\xvec$ (not modified).
 * @param value $f(\xvec)$.
 * @param g $\nabla f(\xvec)$.
 * @param p The search direction $\pvec$.
 * @param options The strong Wolfe search's options, or undefined for its defaults.
 * @returns The search's outcome `found`, the step `s` and gradient change `y` it made, and `sy`, $\svec^\top\yvec$.
 */
function searchAlong(f: ObjectiveFn, x: F64, value: number, g: F64, p: F64, options: StrongWolfeOptions | undefined) {
  const found = strongWolfeSearch(f, x, value, g, p, options)
  const s = sub(found.x, x)
  const y = sub(found.grad, g)
  return { found, s, y, sy: dot(s, y) }
}

/**
 * BFGS (Broyden, Fletcher, Goldfarb and Shanno, 1970; Nocedal & Wright, Algorithm 6.1):
 * $\pvec = -\Hmat\nabla f$, a strong Wolfe step, then the inverse update
 * $\Hmat \leftarrow (\Imat - \rho\svec\yvec^\top)\Hmat(\Imat - \rho\yvec\svec^\top) + \rho\svec\svec^\top$. The
 * first step is taken with $\Hmat_0 = \Imat$, which is rescaled to $(\yvec^\top\svec / \yvec^\top\yvec)\Imat$ just
 * before the first step's update (eq. 6.20). An update with
 * $\yvec^\top\svec \le 10^{-10}\lVert \svec \rVert\lVert \yvec \rVert$ is skipped and flagged. The $n \times n$
 * matrix costs $O(n^2)$ memory and time a step; `lbfgs` avoids it.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The line-search options and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Rosenbrock's function, and the inverse Hessian it learns
 * // At the minimum (1, 1) the true inverse Hessian is [[0.5, 1], [1, 2.005]].
 * const rosenbrock = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     value: (1 - a) ** 2 + 100 * (b - a * a) ** 2,
 *     grad: [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)],
 *   }
 * }
 * const s = run(bfgs(rosenbrock), { x0: [-1.2, 1] }, 1000)
 * print('x =', s.x, ' f =', s.value)
 * print('steps =', s.t, ' evaluations =', s.evaluations)
 * print('inverse Hessian approximation =', s.inverseHessian)
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
  /** $\nabla f(\xvec)$ (for `owlqn`, the pseudo-gradient). */
  grad: Vector
  /** The norm of `grad`, compared with `tolerance`. */
  gradNorm: number
  /** The stored curvature pairs, oldest first (at most `memory`). */
  pairs: CurvaturePair[]
  /**
   * The initial-Hessian scale $\gamma = \svec^\top\yvec / \yvec^\top\yvec$ of the newest stored pair (1 before any
   * pair).
   */
  gamma: number
  /** The search direction of the last step (zeros at $t = 0$). */
  direction: Vector
  /** The step length $\alpha$ accepted on the last step; NaN at $t = 0$. */
  stepSize: number
  /** The last strong Wolfe line search, with its trial points; null at $t = 0$ (and for `owlqn` with $C > 0$). */
  lineSearch: LineSearchResult | null
  /** True when the last pair was not stored because $\yvec^\top\svec$ was not safely positive. */
  skipped: boolean
  /** True when the last line search could not lower $f$ ($\xvec$ unchanged); the run stops. */
  stalled: boolean
}

/** Options for `lbfgs`. */
export type LbfgsOptions = QuasiNewtonOptions & {
  /** Number of curvature pairs kept, $m$. Default 10. */
  memory?: number
}

/**
 * The two-loop recursion (Nocedal & Wright, Algorithm 7.4): returns $\Hmat\qvec$ for the L-BFGS inverse Hessian built
 * from the pairs (oldest first) on the initial matrix $\gamma\Imat$, in $O(mn)$ without forming $\Hmat$.
 *
 * @param q0 The vector $\qvec$ to multiply, $n$ values (not modified).
 * @param pairs The curvature pairs as working arrays, oldest first.
 * @param gamma The scale $\gamma$ of the initial matrix.
 * @returns $\Hmat\qvec$.
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
 * Limited-memory BFGS (Liu & Nocedal, 1989; Nocedal & Wright, Algorithm 7.5): the direction $-\Hmat\nabla f$ comes
 * from the two-loop recursion over the last $m$ pairs with $\Hmat_0 = \gamma\Imat$,
 * $\gamma = \svec^\top\yvec / \yvec^\top\yvec$ of the newest pair; steps satisfy the strong Wolfe conditions. The
 * first step, with no pairs, is scaled to length 1 along $-\nabla f$. Memory and time per step are $O(mn)$, so it
 * suits large $n$. It is the default method of `minimize`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The number of pairs kept (`memory`), the line-search options and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Rosenbrock's function with different memories
 * const rosenbrock = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     value: (1 - a) ** 2 + 100 * (b - a * a) ** 2,
 *     grad: [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)],
 *   }
 * }
 * for (const memory of [3, 10]) {
 *   const s = run(lbfgs(rosenbrock, { memory }), { x0: [-1.2, 1] }, 1000)
 *   print(`memory ${memory}: x =`, s.x, ' steps =', s.t, ' evaluations =', s.evaluations, ' pairs =', s.pairs.length)
 * }
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

/**
 * The L-BFGS direction $-\Hmat\vvec$ from the stored pairs; with no pairs, $-\vvec$ scaled to unit length.
 *
 * @param v The vector $\vvec$ to turn into a direction: the gradient (or the pseudo-gradient for OWL-QN).
 * @param st The stored pairs and the scale $\gamma$.
 * @returns The direction, $n$ values.
 */
function lbfgsDirection(v: F64, st: { pairs: CurvaturePair[]; gamma: number }): F64 {
  const pairs = st.pairs.map((q) => ({ s: data(q.s), y: data(q.y), rho: q.rho }))
  // Without curvature information, a unit-length steepest-descent step (as scipy's L-BFGS-B starts).
  const gamma = pairs.length ? st.gamma : 1 / Math.max(norm(v), 1e-300)
  return scale(-1, twoLoop(v, pairs, gamma))
}

/**
 * The history after a step: the pair $(\svec, \yvec)$ is stored, the oldest dropped beyond `memory`, unless
 * $\yvec^\top\svec \le 10^{-10}\lVert \svec \rVert\lVert \yvec \rVert$.
 *
 * @param st The current pairs and scale $\gamma$ (not modified).
 * @param s The step $\svec$ just taken.
 * @param y The gradient change $\yvec$ over that step.
 * @param memory The most pairs to keep, $m$.
 * @returns The new `pairs` and `gamma`, unchanged when the pair was `skipped`.
 */
function updateHistory(st: { pairs: CurvaturePair[]; gamma: number }, s: F64, y: F64, memory: number) {
  const sy = dot(s, y)
  const skipped = !(sy > 1e-10 * norm(s) * norm(y))
  if (skipped) return { pairs: st.pairs, gamma: st.gamma, skipped }
  return { pairs: [...st.pairs, { s: vec(s), y: vec(y), rho: 1 / sy }].slice(-memory), gamma: sy / dot(y, y), skipped }
}

/**
 * One L-BFGS step: the two-loop direction, a strong Wolfe step, and the history update. Shared by `lbfgs` and by
 * `owlqn` with $C = 0$.
 *
 * @param f The objective, returning value and gradient.
 * @param st The state before the step; its other fields are carried over.
 * @param memory The most pairs to keep, $m$.
 * @param tolerance The gradient-norm tolerance for `converged`.
 * @param divergeAbove The value above which the state is flagged `diverged`.
 * @param lineSearchOptions The strong Wolfe search's options, or undefined for its defaults.
 * @returns The state after the step.
 */
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
  /**
   * The L1 strength $C \ge 0$ in $F(\xvec) = f(\xvec) + C\lVert \xvec \rVert_1$. Default 0, where OWL-QN is L-BFGS.
   * A negative value throws `DomainError`.
   */
  l1?: number
  /** The sufficient-decrease constant of the backtracking search, $c_1$ (default 1e-4). */
  decrease?: number
  /** The backtracking factor applied to $\alpha$ after each rejected trial (default 0.5). */
  backtrack?: number
  /** The most backtracking reductions per step (default 50); when all fail the run stops `stalled`. */
  maxBacktracks?: number
}

/** The state of `owlqn`. */
export type OwlqnState = LbfgsState & {
  /** `value` is $F(\xvec) = f(\xvec) + C\lVert \xvec \rVert_1$; `smoothValue` is $f(\xvec)$. */
  smoothValue: number
  /**
   * $\nabla f(\xvec)$, the smooth part's gradient, from which the curvature pairs are taken. `grad` is the
   * pseudo-gradient $\diamond F$.
   */
  smoothGrad: Vector
  /** The number of non-zero coordinates of $\xvec$. */
  nonzero: number
  /** Backtracking halvings in the last step (0 when the first trial point was accepted). */
  backtracks: number
}

/**
 * The pseudo-gradient $\diamond F$ of $F(\xvec) = f(\xvec) + C\lVert \xvec \rVert_1$ (Andrew & Gao 2007, eq. 4) from
 * $\gvec = \nabla f(\xvec)$: $g_i + C\sgn(x_i)$ where $x_i \ne 0$; at $x_i = 0$, $g_i + C$ when it is negative (the
 * right derivative: moving up decreases $F$), $g_i - C$ when it is positive (the left derivative), else 0. Then
 * $-\diamond F$ is the steepest-descent direction of $F$, and $\diamond F = \zeros$ exactly where $\xvec$ minimises a
 * convex $F$.
 *
 * @param x The point $\xvec$, $n$ values.
 * @param g The smooth part's gradient $\nabla f(\xvec)$, $n$ values.
 * @param l1 The L1 strength $C \ge 0$.
 * @returns $\diamond F(\xvec)$, a new array of $n$ values.
 *
 * @example At zero, a coordinate moves only if its gradient beats the penalty
 * // C = 1. Coordinate 0: x = 0, |g| = 0.5 < C, so it stays. Coordinate 1: x = 0, g = −3, so g + C = −2.
 * // Coordinate 2: x = 1 > 0, so g + C = 1.2.
 * print(pseudoGradient([0, 0, 1], [0.5, -3, 0.2], 1))
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

/**
 * $\lVert \xvec \rVert_1$.
 *
 * @param x The vector, $n$ values.
 * @returns The sum of the absolute values of its entries.
 */
const l1Norm = (x: ArrayLike<number>) => {
  let a = 0
  for (let i = 0; i < x.length; i++) a += Math.abs(x[i])
  return a
}
/**
 * The number of non-zero entries of a vector.
 *
 * @param x The vector, $n$ values.
 * @returns How many entries are not exactly 0.
 */
const countNonzero = (x: ArrayLike<number>) => {
  let k = 0
  for (let i = 0; i < x.length; i++) if (x[i] !== 0) k++
  return k
}

/**
 * Orthant-wise limited-memory quasi-Newton (OWL-QN; Andrew & Gao 2007, "Scalable training of L1-regularized
 * log-linear models", ICML) for $F(\xvec) = f(\xvec) + C\lVert \xvec \rVert_1$ with $f$ smooth. Each step:
 *
 * 1. the pseudo-gradient $\vvec = \diamond F(\xvec)$ (`pseudoGradient`);
 * 2. the L-BFGS direction $\dvec = -\Hmat\vvec$ from the same two-loop recursion and pair history as `lbfgs`, with the
 *    pairs taken from $\nabla f$ alone (the L1 term adds no curvature);
 * 3. $\dvec$ projected onto the orthant of steepest descent: $d_i = 0$ wherever $\sgn(d_i) \ne \sgn(-v_i)$ (and if
 *    that leaves no descent, $\dvec$ is $-\vvec$ scaled to unit length);
 * 4. the orthant $\xi_i = \sgn(x_i)$, or $\sgn(-v_i)$ where $x_i = 0$;
 * 5. a backtracking search along the projected path $\xvec(\alpha) = \pi(\xvec + \alpha\dvec; \xivec)$, where $\pi$
 *    zeroes every coordinate that leaves its orthant, accepting
 *    $F(\xvec(\alpha)) \le F(\xvec) + c_1\vvec^\top(\xvec(\alpha) - \xvec)$.
 *
 * Coordinates that cross zero stop at zero, so the iterates are sparse. The run converges when
 * $\lVert \diamond F \rVert \le$ `tolerance`, and stops `stalled` when no backtracking trial is accepted. With
 * $C = 0$ the steps are exactly those of `lbfgs` (strong Wolfe search included), as in libLBFGS.
 *
 * @param f The smooth part $f$, returning `{ value, grad }` at a point; the L1 term is added by the method.
 * @param options The L1 strength $C$ (`l1`), the backtracking options, the number of pairs kept, the line-search
 *   options (used when $C = 0$) and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example The L1 penalty zeroes small coordinates
 * // f(x) = ½‖x − c‖² with c = (3, 0.5, −2, −0.2) and C = 1: the minimiser soft-thresholds c to (2, 0, −1, 0).
 * const c = [3, 0.5, -2, -0.2]
 * const f = (x) => {
 *   const d = Array.from(x.data, (v, i) => v - c[i])
 *   return { value: 0.5 * d.reduce((s, v) => s + v * v, 0), grad: d }
 * }
 * const s = run(owlqn(f, { l1: 1 }), { x0: [0, 0, 0, 0] }, 100)
 * print('x =', s.x, ' non-zero =', s.nonzero)
 * print('F(x) =', s.value, ' f(x) =', s.smoothValue)
 * const plain = run(owlqn(f), { x0: [0, 0, 0, 0] }, 100)
 * print('without the penalty: x =', plain.x)
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
