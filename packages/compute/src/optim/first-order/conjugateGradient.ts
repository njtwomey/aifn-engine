/**
 * Conjugate gradients: the linear method for symmetric positive definite systems $\Amat\xvec = \bvec$ (equivalently,
 * minimising $\tfrac12\xvec^\top\Amat\xvec - \bvec^\top\xvec$), and the nonlinear methods of Fletcher–Reeves and
 * Polak–Ribière for general smooth objectives.
 *
 * Both build each search direction from the new gradient (or residual) and the previous direction,
 * $\pvec \leftarrow -\nabla f + \beta\pvec$, so they keep only a few vectors and never form a matrix. The linear
 * method needs only products $\vvec \mapsto \Amat\vvec$, and reports an indefinite $\Amat$ rather than throwing; the
 * nonlinear one takes its step lengths from a strong Wolfe line search and restarts from steepest descent when the
 * direction stops being one of descent. Nocedal & Wright (2006), "Numerical Optimization", 2nd ed., chapter 5.
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

/** Which $\beta$ the nonlinear method uses: Fletcher–Reeves or Polak–Ribière (clipped at 0). */
export type ConjugateGradientVariant = 'fletcher-reeves' | 'polak-ribiere'

/** The state of `conjugateGradient`. */
export type ConjugateGradientState = IterateState & {
  /** $\nabla f(\xvec)$. */
  grad: Vector
  /** $\lVert \nabla f(\xvec) \rVert_2$, compared with `tolerance`. */
  gradNorm: number
  /** The search direction for the next step, $\pvec = -\nabla f + \beta\pvec_{\text{prev}}$. */
  direction: Vector
  /** The $\beta$ used to build `direction` (0 on a restart). */
  beta: number
  /**
   * True when `direction` was reset to $-\nabla f$ (every `restartEvery` steps, or when it was not a descent direction
   * or $\beta$ was not finite). True at $t = 0$.
   */
  restarted: boolean
  /** The step length $\alpha$ the line search accepted on the last step; NaN at $t = 0$. */
  stepSize: number
  /** The last line search, with its trial points; null at $t = 0$. */
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower $f$ ($\xvec$ unchanged); the run stops. */
  stalled: boolean
}

/** Options for `conjugateGradient`. */
export type ConjugateGradientOptions = StoppingOptions & {
  /** Which $\beta$ to use. Default `'polak-ribiere'` (with $\beta^+ = \max(\beta, 0)$). */
  variant?: ConjugateGradientVariant
  /** Restart with steepest descent every this many steps. Default $n$ (the dimension). */
  restartEvery?: number
  /**
   * Line-search options; default $c_2 = 0.1$, as the Fletcher–Reeves descent guarantee needs $c_2 < \tfrac12$. The
   * first trial step `alpha0` is chosen by the method and overrides any given here.
   */
  lineSearchOptions?: StrongWolfeOptions
}

/**
 * Nonlinear conjugate gradients (Nocedal & Wright, §5.2): $\xvec \leftarrow \xvec + \alpha\pvec$ with $\alpha$ from a
 * strong Wolfe search, then $\pvec \leftarrow -\gvec_{k+1} + \beta\pvec$ with
 * $\beta_{\mathrm{FR}} = \lVert \gvec_{k+1} \rVert^2 / \lVert \gvec_k \rVert^2$ (Fletcher & Reeves, 1964) or
 * $\beta_{\mathrm{PR}}^+ = \max(0, \gvec_{k+1}^\top(\gvec_{k+1} - \gvec_k) / \lVert \gvec_k \rVert^2)$ (Polak &
 * Ribière, 1969), where $\gvec_k = \nabla f(\xvec_k)$. The first trial step length follows Nocedal & Wright eq. 3.60.
 * The run stops when the gradient norm reaches `tolerance`, on divergence, or when a line search cannot lower $f$.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The choice of $\beta$, the restart period, the line-search options and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Rosenbrock's function from the standard start
 * // (1 − a)² + 100(b − a²)², least at (1, 1).
 * const rosenbrock = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     value: (1 - a) ** 2 + 100 * (b - a * a) ** 2,
 *     grad: [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)],
 *   }
 * }
 * for (const variant of ['fletcher-reeves', 'polak-ribiere']) {
 *   const s = run(conjugateGradient(rosenbrock, { variant }), { x0: [-1.2, 1] }, 1000)
 *   print(`${variant}: x =`, s.x, ' steps =', s.t, ' evaluations =', s.evaluations)
 * }
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
  /** Steps taken. */
  t: number
  /** The current iterate $\xvec$. */
  x: Vector
  /** The residual $\rvec = \bvec - \Amat\xvec$. */
  residual: Vector
  /** $\lVert \rvec \rVert_2$. */
  residualNorm: number
  /** The next search direction $\pvec$ ($\Amat$-conjugate to all earlier ones in exact arithmetic). */
  direction: Vector
  /** The step length $\alpha = \rvec^\top\rvec / \pvec^\top\Amat\pvec$ of the last step (NaN at $t = 0$). */
  alpha: number
  /** $\beta = \rvec_{k+1}^\top\rvec_{k+1} / \rvec_k^\top\rvec_k$ of the last step (NaN at $t = 0$). */
  beta: number
  /** The quadratic $\tfrac12\xvec^\top\Amat\xvec - \bvec^\top\xvec$ that CG minimises. */
  value: number
  /** Matrix–vector products so far. */
  products: number
  /** $\lVert \rvec \rVert \le$ `tolerance` $\cdot \lVert \bvec \rVert$; the run stops. */
  converged: boolean
  /**
   * True when $\pvec^\top\Amat\pvec \le 0$ was met: $\Amat$ is not positive definite and the run stops, with
   * $\xvec$ left as it was.
   */
  indefinite: boolean
}

/** Options for `linearConjugateGradient`. */
export type LinearConjugateGradientOptions = {
  /**
   * Stop when $\lVert \rvec \rVert \le$ `tolerance` $\cdot \lVert \bvec \rVert$ (or $\le$ `tolerance` when
   * $\bvec = \zeros$). Default 1e-10.
   */
  tolerance?: number
}

/**
 * Conjugate gradients for $\Amat\xvec = \bvec$ with $\Amat$ symmetric positive definite (Hestenes & Stiefel, 1952;
 * Nocedal & Wright, Algorithm 5.2): at most $n$ steps in exact arithmetic, with the error in the $\Amat$-norm reduced
 * at the rate $(\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ per step for condition number $\kappa$. One product with
 * $\Amat$ per step. An indefinite $\Amat$ is reported (`indefinite`), not thrown.
 *
 * @param A The operator $\Amat$: an $n \times n$ matrix, or a function returning $\Amat\vvec$ for a vector $\vvec$.
 *   Only products with it are used, so it is never factored.
 * @param b The right-hand side $\bvec$, of length $n$.
 * @param options The relative residual tolerance.
 * @returns The algorithm; `init` takes `{ x0 }`, and $\xvec_0 = \zeros$ when it is left out.
 *
 * @example A two-by-two system in two steps
 * // A = [[4, 1], [1, 3]], b = (1, 2): the solution is (1/11, 7/11).
 * const alg = linearConjugateGradient([[4, 1], [1, 3]], [1, 2])
 * for (const steps of [0, 1, 2]) {
 *   const s = run(alg, {}, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' residual norm =', s.residualNorm)
 * }
 * print('1/11, 7/11 =', [1 / 11, 7 / 11])
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
  /** The last iterate, the solution when `converged`. */
  x: Tensor
  /** $\lVert \bvec - \Amat\xvec \rVert_2$ at `x`. */
  residualNorm: number
  /** Steps taken. */
  steps: number
  /** The relative residual reached the tolerance. */
  converged: boolean
  /** $\Amat$ was found not positive definite, and the run stopped. */
  indefinite: boolean
}

/**
 * Solves $\Amat\xvec = \bvec$ ($\Amat$ symmetric positive definite, $n \times n$ or $\vvec \mapsto \Amat\vvec$) by
 * linear conjugate gradients, running at most `maxSteps` steps (default $10n$, allowing for rounding). Failure to
 * converge and an indefinite $\Amat$ are reported in the result, not thrown. See `linearConjugateGradient`.
 *
 * @param A The operator $\Amat$: an $n \times n$ matrix, or a function returning $\Amat\vvec$ for a vector $\vvec$.
 * @param b The right-hand side $\bvec$, of length $n$.
 * @param options The relative residual `tolerance`, the starting point `x0` (default zeros) and `maxSteps`.
 * @returns The last iterate, its residual norm, the steps taken, and whether it converged or met indefiniteness.
 *
 * @example Solve with a matrix, and with a function
 * print(solveConjugateGradient([[4, 1], [1, 3]], [1, 2]))
 * // A diagonal operator diag(1, 2, 3) given as a product, never as a matrix.
 * print(solveConjugateGradient((v) => mul(tensor([1, 2, 3]), v), [1, 1, 1]).x)
 *
 * @example An indefinite matrix is reported
 * // [[1, 2], [2, 1]] has eigenvalues 3 and −1.
 * const { converged, indefinite } = solveConjugateGradient([[1, 2], [2, 1]], [1, 2])
 * print('converged =', converged, ' indefinite =', indefinite)
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
