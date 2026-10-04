/**
 * Implicit differentiation: the derivative of a solution with respect to the parameters of the equation it solves,
 * without differentiating the solver (design K §4.3).
 *
 * If $x^\star$ solves $r(p, x^\star) = 0$ and $\partial r/\partial x$ is invertible there, the implicit function theorem gives
 * $\partial x^\star/\partial p = -(\partial r/\partial x)^{-1} \partial r/\partial p$ (Krantz and Parks, 2002). The vjp of the solution map is therefore
 * $\bar{p} = -(\partial r/\partial p)^\top w$, where $w$ solves $(\partial r/\partial x)^\top w = \bar{x}$. For a fixed point $x^\star = F(p, x^\star)$, $r = x - F$ gives
 * $\bar{p} = (\partial F/\partial p)^\top w$ with $(I - \partial F/\partial x)^\top w = \bar{x}$ (Christianson, 1994; Blondel et al., 2022).
 *
 * The solver runs on raw values only and is never traced, so any solver works (Newton, L-BFGS, a `dense` loop). The
 * adjoint system is solved densely for small problems (the Jacobian from `jacobian`, then `aifn-compute/numerics/linalg`'s
 * `solve`, LU with partial pivoting, which raises `LinAlgError('singular')` for a singular system) and
 * otherwise iteratively with vjp calls only (jvp calls in forward mode): the fixed-point iteration
 * $w \leftarrow (\partial F/\partial x)^\top w + \bar{x}$ for `implicitFixedPoint`, BiCGSTAB for `implicitRoot`. Both iterations are written with
 * primitives and batch under `vmap`: the examples iterate together, each stopping on its own values. A solution that does not satisfy its
 * equation, or an adjoint solve that does not converge, raises `NumericalError('not-converged')`: never a gradient of
 * the wrong point.
 */

import {
  batchExamples,
  defineCustomVjp,
  jacobian,
  jvp,
  vjp,
  type Cotangents,
  type TreeOf,
} from 'aifn-compute/foundation/autodiff'
import { NumericalError } from 'aifn-compute/foundation/errors'
import {
  add,
  avalOf,
  concat,
  div,
  dot,
  eye,
  mul,
  neg,
  norm,
  reshape,
  slice,
  sub,
  sum,
  toFlat,
  transpose,
  unwrap,
  lessEqual,
  // `where` is the solvers' name for errors here.
  where as select,
  type Aval,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { treeFlatten, treeUnflatten, type TreeDef } from 'aifn-compute/foundation/pytree'
import { solve as linearSolve } from 'aifn-compute/numerics/linalg'

// ── Trees as vectors ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Compute the total number of elements in a tensor shape.
 *
 * @param shape The shape array of dimensions.
 * @returns The total number of elements (product of dimensions).
 */
const count = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)

/** A tree's leaves as one vector, and the way back; written with primitives, so traced values pass through. */
type Vectorised = { n: number; toVec: (tree: unknown) => Value; fromVec: (v: Value) => unknown }

/**
 * Flatten a pytree of tensor leaves into a single 1D vector and provide the inverse unflattening.
 *
 * @param like A representative instance of the pytree.
 * @returns An object with total dimension $n$, `toVec` flattener, and `fromVec` unflattening function.
 */
function vectorise(like: unknown): Vectorised {
  const flat = treeFlatten(like)
  const avals: Aval[] = flat.leaves.map(avalOf)
  const treedef: TreeDef = flat.treedef
  const n = avals.reduce((s, a) => s + count(a.shape), 0)
  return {
    n,
    toVec: (tree) => {
      const parts = treeFlatten(tree).leaves.map((l, i) => reshape(l, [count(avals[i].shape)]))
      return parts.length === 1 ? parts[0] : concat(parts, 0)
    },
    fromVec: (v) => {
      let off = 0
      const leaves = avals.map((a) => {
        const size = count(a.shape)
        const piece = slice(v, [off, off + size])
        off += size
        return a.number ? sum(piece) : reshape(piece, a.shape)
      })
      return treeUnflatten(treedef, leaves)
    },
  }
}

/**
 * Compute the Euclidean norm $\|v\|_2$ of a value as a raw scalar number.
 *
 * @param v The tensor or scalar value.
 * @returns The Euclidean norm as a JavaScript number.
 */
const size = (v: Value): number => unwrap(norm(v)) as number

// ── Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `implicitFixedPoint` and `implicitRoot`. */
export type ImplicitOptions = {
  /**
   * How the adjoint system is solved: `dense` (form the Jacobian, then a pivoted solve), `iterative` (vjp calls only),
   * or `auto` (default): dense when x has at most `denseMax` elements.
   */
  solve?: 'auto' | 'dense' | 'iterative'
  /** The largest x solved densely under `auto` (default 64). */
  denseMax?: number
  /** Absolute tolerance of the iterative adjoint solve: $\|\Delta\| \le \text{atol} + \text{rtol} \cdot \|\bar{x}\|$ (default 1e-12). */
  atol?: number
  /** Relative tolerance of the iterative adjoint solve (default 1e-10). */
  rtol?: number
  /** Iterations of the iterative adjoint solve before `NumericalError('not-converged')` (default 1000). */
  maxIter?: number
  /**
   * Check that the solver's answer satisfies its equation, $\|r\| \le \text{atol} + \text{rtol} \cdot \|x\|$, before differentiating at it
   * (default `{ atol: 1e-6, rtol: 1e-6 }`); `false` skips the check.
   */
  check?: { atol: number; rtol: number } | false
}

/**
 * Verify that a proposed solution satisfies its residual equation, throwing `NumericalError` if not.
 *
 * @param where The caller function name for error messages.
 * @param r The residual value $r(p, x)$.
 * @param x The candidate solution value $x$.
 * @param check The tolerance settings or `false` to disable the check.
 */
function checkSolution(where: string, r: Value, x: Value, check: ImplicitOptions['check']): void {
  if (check === false) return
  const { atol, rtol } = check ?? { atol: 1e-6, rtol: 1e-6 }
  const e = size(r)
  if (!(e <= atol + rtol * size(x)))
    throw new NumericalError(
      where,
      `${where}: the solver's answer does not satisfy its equation (residual ${e.toExponential(2)}); its derivative would be wrong`,
      'not-converged',
    )
}

/**
 * Determine whether an adjoint system of size $n$ should be solved densely or iteratively.
 *
 * @param n The total number of variables in the solution vector.
 * @param o The implicit differentiation options.
 * @returns True if a dense linear solve should be used, false for iterative.
 */
const solvesDensely = (n: number, o: ImplicitOptions) =>
  o.solve === 'dense' || (o.solve !== 'iterative' && n <= (o.denseMax ?? 64))

/**
 * The Jacobian of the vector map $v \mapsto \text{toVec}(h(\text{fromVec}(v)))$ at $x$, as an $n \times n$ tensor.
 *
 * @param h The mapping function.
 * @param x The point at which to evaluate the Jacobian.
 * @param vec The vectorisation helper holding dimensions and tree converters.
 * @returns An $n \times n$ tensor representing the Jacobian matrix.
 */
function denseJacobian<X>(h: (x: X) => unknown, x: X, vec: Vectorised): Value {
  const J = jacobian((v: Value) => vec.toVec(h(vec.fromVec(v) as X)), { mode: 'reverse' })(vec.toVec(x)) as Value
  return reshape(J, [vec.n, vec.n])
}

/**
 * Construct a `NumericalError` indicating non-convergence of an adjoint iterative solve.
 *
 * @param where The caller name for error messages.
 * @param iterations The number of iterations completed before termination.
 * @returns A `NumericalError` configured with the `'not-converged'` code.
 */
function notConverged(where: string, iterations: number): NumericalError {
  return new NumericalError(
    where,
    `${where}: the adjoint solve did not converge in ${iterations} iterations; its gradient is not returned`,
    'not-converged',
  )
}

// ── implicitFixedPoint ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The fixed point $x^\star = F(p, x^\star)$ found by `solver(p, x0)`, differentiable in $p$ by the implicit function
 * theorem (the derivative in $x_0$ is zero). The solver runs on raw values and is never differentiated; $F$ must be
 * written with primitives. The adjoint $(\Imat - \partial F/\partial x)^\top w = \bar{x}$ is solved densely for small
 * $x$, otherwise by $w \leftarrow (\partial F/\partial x)^\top w + \bar{x}$, which converges whenever the fixed-point
 * iteration itself contracts.
 *
 * @param solver A function computing the fixed point $x^\star$ from parameters $p$ and initial state $x_0$.
 * @param F The fixed-point map $F(p, x)$ satisfying $x^\star = F(p, x^\star)$.
 * @param options Options controlling adjoint solve strategy, tolerances, and solution checks.
 * @returns A differentiable function computing the fixed point for parameters $p$ and initial guess $x_0$.
 *
 * @example Differentiate through a contracting fixed point iteration
 * const solve = (p, x0) => {
 *   let x = Number(x0)
 *   for (let k = 0; k < 30; k++) x = 0.5 * x + Number(p)
 *   return x
 * }
 * const fp = implicitFixedPoint(solve, (p, x) => add(mul(0.5, x), p))
 * print('d(x*)/dp =', grad((p) => fp(p, 0))(3))
 */
export function implicitFixedPoint<P, X>(
  solver: (p: P, x0: X) => X,
  F: (p: P, x: X) => X,
  options: ImplicitOptions = {},
): (p: P, x0: X) => X {
  const where = 'implicitFixedPoint'
  const solve = (p: P, x0: X): X => {
    const x = solver(p, x0)
    const vec = vectorise(x)
    checkSolution(where, sub(vec.toVec(F(p, x)), vec.toVec(x)), vec.toVec(x), options.check)
    return x
  }
  return defineCustomVjp<[P, X], X, { p: P; x: X }>({
    name: where,
    f: solve,
    fwd: (p, x0) => {
      const x = solve(p, x0)
      return { out: x, residuals: { p, x } }
    },
    residualsFrom: ([p], x) => ({ p, x }),
    // Reverse: (I − ∂F/∂x)ᵀ w = x̄, then p̄ = (∂F/∂p)ᵀ w.
    bwd: ({ p, x }, xbar) => {
      const vec = vectorise(x)
      const b = vec.toVec(xbar)
      let w: Value
      if (solvesDensely(vec.n, options)) {
        const J = denseJacobian((y: X) => F(p, y), x, vec)
        w = linearSolve(sub(eye(vec.n), transpose(J)), b)
      } else {
        const { pullback } = vjp((y: X) => F(p, y), x)
        w = neumann((v) => vec.toVec(pullback(vec.fromVec(v) as never)), b, options, where)
      }
      const pbar = vjp((q: P) => F(q, x), p).pullback(vec.fromVec(w) as never)
      return [pbar, null] as unknown as Cotangents<[P, X]>
    },
    // Forward: (I − ∂F/∂x) ẋ = (∂F/∂p) ṗ.
    jvp: ({ p, x }, [dp]) => {
      const vec = vectorise(x)
      const b = vec.toVec(jvp((q: P) => F(q, x), p, dp).tangent)
      let dx: Value
      if (solvesDensely(vec.n, options)) {
        const J = denseJacobian((y: X) => F(p, y), x, vec)
        dx = linearSolve(sub(eye(vec.n), J), b)
      } else {
        dx = neumann((v) => vec.toVec(jvp((y: X) => F(p, y), x, vec.fromVec(v) as X).tangent), b, options, where)
      }
      return vec.fromVec(dx) as TreeOf<X, Value>
    },
  })
}

/**
 * Solve $(\Imat - \Kmat) w = b$ for a contraction $\Kmat$ given as products, by the iteration $w \leftarrow b + \Kmat w$.
 * Inside `vmap` the examples iterate together until every one has converged (iterating past convergence is harmless for
 * a contraction).
 *
 * @param K The linear contraction operator.
 * @param b The right-hand side vector.
 * @param o Options specifying tolerances and maximum iterations.
 * @param where Caller name for error messages.
 * @returns The converged solution vector $w$.
 */
function neumann(K: (v: Value) => Value, b: Value, o: ImplicitOptions, where: string): Value {
  const { atol = 1e-12, rtol = 1e-10, maxIter = 1000 } = o
  const tol = add(atol, mul(rtol, norm(b)))
  let w = b
  for (let k = 0; k < maxIter; k++) {
    const next = add(b, K(w))
    const converged = lessEqual(norm(sub(next, w)), tol)
    w = next
    if (everyExample(converged, where)) return w
  }
  throw notConverged(where, maxIter)
}

// ── implicitRoot ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The root $x^\star$ of $\text{residual}(p, x) = 0$ found by `solver(p)`, differentiable in $p$ by the implicit
 * function theorem: $\bar{p} = -(\partial r/\partial p)^\top w$ with $(\partial r/\partial x)^\top w = \bar{x}$.
 * The solver runs on raw values and is never differentiated; the residual must be written with primitives. For a
 * minimiser, the residual is the gradient of the objective (a hyperparameter gradient through `optim.minimize`).
 * The adjoint is solved densely for small $x$, otherwise by BiCGSTAB with vjp calls.
 *
 * @param solver A function computing the root $x^\star$ from parameter $p$.
 * @param residual The residual function $r(p, x)$ satisfying $r(p, x^\star) = 0$.
 * @param options Options controlling adjoint solve strategy, tolerances, and solution checks.
 * @returns A differentiable function computing the root for parameter $p$.
 *
 * @example Differentiate through an untraced square root solver
 * const newtonSqrt = (p) => {
 *   let x = 1
 *   for (let k = 0; k < 20; k++) x = 0.5 * (x + Number(p) / x)
 *   return x
 * }
 * const root = implicitRoot(newtonSqrt, (p, x) => sub(mul(x, x), p))
 * print('d(sqrt(4))/dp =', grad(root)(4))
 */
export function implicitRoot<P, X>(
  solver: (p: P) => X,
  residual: (p: P, x: X) => X,
  options: ImplicitOptions = {},
): (p: P) => X {
  const where = 'implicitRoot'
  const solve = (p: P): X => {
    const x = solver(p)
    const vec = vectorise(x)
    checkSolution(where, vec.toVec(residual(p, x)), vec.toVec(x), options.check)
    return x
  }
  return defineCustomVjp<[P], X, { p: P; x: X }>({
    name: where,
    f: solve,
    fwd: (p) => {
      const x = solve(p)
      return { out: x, residuals: { p, x } }
    },
    residualsFrom: ([p], x) => ({ p, x }),
    // Reverse: (∂r/∂x)ᵀ w = x̄, then p̄ = −(∂r/∂p)ᵀ w.
    bwd: ({ p, x }, xbar) => {
      const vec = vectorise(x)
      const b = vec.toVec(xbar)
      let w: Value
      if (solvesDensely(vec.n, options)) {
        w = linearSolve(transpose(denseJacobian((y: X) => residual(p, y), x, vec)), b)
      } else {
        const { pullback } = vjp((y: X) => residual(p, y), x)
        w = bicgstab((v) => vec.toVec(pullback(vec.fromVec(v) as never)), b, options, where)
      }
      const pbar = vjp((q: P) => residual(q, x), p).pullback(vec.fromVec(neg(w)) as never)
      return [pbar] as unknown as Cotangents<[P]>
    },
    // Forward: (∂r/∂x) ẋ = −(∂r/∂p) ṗ.
    jvp: ({ p, x }, [dp]) => {
      const vec = vectorise(x)
      const b = neg(vec.toVec(jvp((q: P) => residual(q, x), p, dp).tangent))
      let dx: Value
      if (solvesDensely(vec.n, options)) {
        dx = linearSolve(
          denseJacobian((y: X) => residual(p, y), x, vec),
          b,
        )
      } else {
        dx = bicgstab(
          (v) => vec.toVec(jvp((y: X) => residual(p, y), x, vec.fromVec(v) as X).tangent),
          b,
          options,
          where,
        )
      }
      return vec.fromVec(dx) as TreeOf<X, Value>
    },
  })
}

/**
 * Solve $\Amat w = b$ for a linear map $\Amat$ given only as products $v \mapsto \Amat v$, by BiCGSTAB (van der Vorst,
 * 1992). Written with primitives, so inside `vmap` every example runs its own iteration in lockstep: the scalars are
 * per example, an example that has converged keeps its answer (`where` on a per-example mask) and has its search state
 * reset so that it stays finite, and the loop stops when every example has converged. Convergence is read from the
 * values of each example.
 *
 * @param A The linear operator as a function $v \mapsto \Amat v$.
 * @param b The right-hand side vector.
 * @param o Options specifying convergence tolerances and maximum iterations.
 * @param where Caller name for error messages.
 * @returns The converged solution vector $w$.
 */
function bicgstab(A: (v: Value) => Value, b: Value, o: ImplicitOptions, where: string): Value {
  const { atol = 1e-12, rtol = 1e-10, maxIter = 1000 } = o
  const tol = add(atol, mul(rtol, norm(b)))
  const small = (v: Value) => lessEqual(norm(v), tol)
  // A denominator that is 1 for the examples masked out (no 0/0, so no NaN in values or derivatives).
  const safe = (mask: Value, d: Value) => select(mask, 1, d)
  let x: Value = mul(0, b)
  let r: Value = b
  let done: Value = small(r)
  if (everyExample(done, where)) return x
  const rhat = b
  let rho: Value = 1
  let alpha: Value = 1
  let omega: Value = 1
  let v: Value = mul(0, b)
  let p: Value = mul(0, b)
  for (let k = 0; k < maxIter; k++) {
    // Converged examples restart from a clean state each step; their x and r are frozen below.
    rho = select(done, 1, rho)
    alpha = select(done, 1, alpha)
    omega = select(done, 1, omega)
    p = select(done, 0, p)
    v = select(done, 0, v)
    const rhoNext = dot(rhat, r)
    const beta = mul(div(rhoNext, rho), div(alpha, omega))
    rho = rhoNext
    p = add(r, mul(beta, sub(p, mul(omega, v))))
    v = A(p)
    alpha = div(rho, safe(done, dot(rhat, v)))
    const h = add(x, mul(alpha, p))
    const s = sub(r, mul(alpha, v))
    const early = small(s)
    // Examples done before this step, or done at h.
    const finished = select(done, 1, early)
    const t = A(s)
    omega = div(dot(t, s), safe(finished, dot(t, t)))
    const xNext = add(h, mul(omega, s))
    const rNext = sub(s, mul(omega, t))
    x = select(done, x, select(early, h, xNext))
    r = select(done, r, select(early, s, rNext))
    done = select(finished, 1, small(rNext))
    if (everyExample(done, where)) return x
  }
  throw notConverged(where, maxIter)
}

/**
 * True when the per-example flag `flag` (a comparison of values) holds for every example: one example outside `vmap`,
 * each batch element inside it. The flags are read from the values, which every level of tracing carries.
 *
 * @param flag The boolean or numeric predicate value to evaluate.
 * @param where Caller name for error messages.
 * @returns True if all examples evaluate to nonzero, false otherwise.
 */
function everyExample(flag: Value, where: string): boolean {
  const examples = batchExamples(flag)
  if (examples === null)
    throw new NumericalError(
      where,
      `${where}: the iterative adjoint solve has no concrete values to test`,
      'not-converged',
    )
  return examples.every((e) => (typeof e === 'number' ? e !== 0 : toFlat(e).every((v) => v !== 0)))
}
