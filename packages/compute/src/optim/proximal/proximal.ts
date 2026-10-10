/**
 * Proximal and projected gradient methods for composite objectives $F(\xvec) = f(\xvec) + g(\xvec)$, with $f$ smooth
 * (value and gradient) and $g$ handled through its proximal operator
 * $\prox_{\eta g}(\vvec) = \argmin_{\xvec} g(\xvec) + \lVert \xvec - \vvec \rVert^2/(2\eta)$. Includes ISTA,
 * FISTA, projected gradient, and the proximal operators and projections used with them.
 *
 * A proximal operator is a `Prox`: the value of $g$ with its prox. A projection is a function from a point to its
 * Euclidean projection onto a closed convex set $\Ccal$, the prox of the set's indicator (0 on $\Ccal$, $\infty$ off
 * it). Both take and return vectors, and neither modifies its input.
 *
 * Parikh and Boyd (2014), "Proximal Algorithms", Foundations and Trends in Optimization 1(3), §4.2 and §6; Beck and
 * Teboulle (2009), "A fast iterative shrinkage-thresholding algorithm for linear inverse problems", SIAM J. Imaging
 * Sciences 2(1).
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
 * A closed convex function $g$ given by its value and proximal operator. `value(x)` is $g(\xvec)$ (Infinity outside
 * the domain of an indicator); `prox(v, eta)` returns $\prox_{\eta g}(\vvec)$.
 */
export type Prox = {
  /** A short name for $g$ (`'l1'`, `'box'`, ...), which names the algorithms built on it. */
  name: string
  /** $g(\xvec)$. */
  value: (x: Vector) => number
  /** $\prox_{\eta g}(\vvec)$, for a step $\eta > 0$. */
  prox: (v: Vector, eta: number) => Vector
}

/** A bound given as a number (all coordinates) or a vector. */
export type Bound = number | VectorLike

/**
 * Coordinate `i` of a bound.
 *
 * @param b The bound as given.
 * @param i The coordinate.
 * @param arr The bound as an array, as `boundArray` returns it, or null when `b` is a number.
 * @returns `arr[i]`, or `b` itself when it is a number.
 */
const bound = (b: Bound, i: number, arr: F64 | null) => (arr ? arr[i] : (b as number))
/**
 * A vector bound as an array, read once.
 *
 * @param b The bound as given.
 * @param where The caller's name for error messages.
 * @returns The bound's values, or null when `b` is a number.
 */
const boundArray = (b: Bound, where: string) => (typeof b === 'number' ? null : toF64(b, where))

/**
 * Projection onto the box $\boldsymbol{\ell} \le \xvec \le \uvec$ (bounds may be $\pm\infty$), by coordinatewise
 * clipping.
 *
 * @param lo The lower bound $\boldsymbol{\ell}$: one number for every coordinate, or a vector of one per coordinate.
 * @param hi The upper bound $\uvec$, in the same form; at least `lo` coordinate by coordinate.
 * @returns The projection, a function from a point to the clipped point.
 *
 * @example Clip to a box
 * print('clipped to [0, 1] =', projectBox(0, 1)(tensor([-0.5, 0.3, 2])))
 * print('per-coordinate bounds =', projectBox([0, -Infinity], [1, 0])(tensor([2, 3])))
 */
export function projectBox(lo: Bound, hi: Bound): (x: Vector) => Vector {
  const l = boundArray(lo, 'projectBox')
  const h = boundArray(hi, 'projectBox')
  return (x) => vec(toF64(x, 'projectBox').map((v, i) => Math.min(Math.max(v, bound(lo, i, l)), bound(hi, i, h))))
}

/**
 * Projection onto the nonnegative orthant, $\max(\xvec, 0)$ coordinatewise.
 *
 * @returns The projection, a function from a point to its nonnegative part.
 *
 * @example Negative coordinates go to zero
 * print('projection =', projectNonnegative()(tensor([-1, 2, -3, 0.5])))
 */
export function projectNonnegative(): (x: Vector) => Vector {
  return (x) => vec(toF64(x, 'projectNonnegative').map((v) => Math.max(v, 0)))
}

/**
 * Projection onto the Euclidean ball $\lVert \xvec - \cvec \rVert_2 \le r$: a point outside moves along the ray from
 * the centre to the sphere; a point inside is returned unchanged.
 *
 * @param radius The radius $r$.
 * @param center The centre $\cvec$, a vector of the points' length (default the origin).
 * @returns The projection, a function from a point to its projection.
 *
 * @example Onto the unit ball and onto a ball about another centre
 * print('onto the unit ball =', projectBall(1)(tensor([3, 4])))
 * print('already inside =', projectBall(1)(tensor([0.3, 0.4])))
 * print('radius 1 about (1, 1) =', projectBall(1, [1, 1])(tensor([1, 3])))
 */
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
 * Project the entries `v[start]` to `v[start + length - 1]` onto the simplex
 * $\{\xvec \ge 0, \sum_i x_i = z\}$ in place, by sorting (Duchi, Shalev-Shwartz, Singer and Chandra, 2008,
 * "Efficient projections onto the l1-ball", Figure 1): $x_i = \max(v_i - \theta, 0)$ with the threshold $\theta$
 * found from the sorted entries.
 *
 * @param v The array holding the block; the block is overwritten with its projection, the rest is left as it is.
 * @param start The index of the block's first entry.
 * @param length The number of entries in the block.
 * @param z The sum $z$ of the simplex, positive.
 */
function simplexInPlace(v: Float64Array, start: number, length: number, z: number): void {
  if (z <= 0) {
    for (let j = start; j < start + length; j++) v[j] = 0
    return
  }
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

/**
 * Euclidean projection onto the simplex $\{\xvec \ge 0, \sum_i x_i = z\}$, by sorting (Duchi et al., 2008,
 * Figure 1), in $O(n \log n)$.
 *
 * @param z The sum $z$ of the simplex, positive: 1 (the default) for probability vectors.
 * @returns The projection, a function from a point to its projection.
 *
 * @example Project onto the probability simplex
 * const p = projectSimplex()(tensor([0.5, 0.8, -0.2]))
 * print('projection =', p)
 * print('sum =', sum(p))
 * print('onto the simplex of sum 2 =', projectSimplex(2)(tensor([0.5, 0.8, -0.2])))
 */
export function projectSimplex(z = 1): (x: Vector) => Vector {
  return (x) => {
    const v = Float64Array.from(toF64(x, 'projectSimplex'))
    simplexInPlace(v, 0, v.length, z)
    return vec(v)
  }
}

/**
 * Projection of every row of an $n \times c$ matrix (flattened row-major, length $nc$) onto the simplex
 * $\{\xvec \ge 0, \sum_i x_i = z\}$: the projection onto the product of $n$ simplices, as for a matrix of class
 * distributions. The projection throws `DomainError` when the length of its input is not a multiple of $c$.
 *
 * @param columns The number of columns $c$, the length of each row.
 * @param z The sum $z$ of every row, positive (default 1).
 * @returns The projection, a function from the flattened matrix to the flattened projection.
 *
 * @example Two rows, each made a distribution
 * const P = projectSimplexRows(3)(tensor([0.5, 0.8, -0.2, 1, 1, 1]))
 * print('rows =', reshape(P, [2, 3]))
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

/**
 * The zero function $g = 0$, whose prox is the identity: proximal gradient with it is plain gradient descent.
 *
 * @returns The `Prox` of $g = 0$.
 *
 * @example The prox leaves its argument as it is
 * const g = proxZero()
 * print('prox =', g.prox(tensor([1, -2]), 0.5))
 * print('g(x) =', g.value(tensor([1, -2])))
 */
export function proxZero(): Prox {
  return { name: 'zero', value: () => 0, prox: (v) => v }
}

/**
 * $g(\xvec) = \lambda \lVert \xvec \rVert_1$: the prox is soft thresholding,
 * $\sgn(v_i) \max(\lvert v_i \rvert - \eta\lambda, 0)$ coordinatewise.
 *
 * @param lambda The weight $\lambda \ge 0$ of the norm.
 * @returns The `Prox` of $g$.
 *
 * @example Soft thresholding
 * const g = proxL1(1)
 * print('threshold 1 =', g.prox(tensor([3, -0.5, 1.5, -2]), 1))
 * print('threshold 0.5 =', g.prox(tensor([3, -0.5, 1.5, -2]), 0.5))
 * print('g(x) =', g.value(tensor([3, -0.5, 1.5, -2])))
 */
export function proxL1(lambda: number): Prox {
  return {
    name: 'l1',
    value: (x) => lambda * toF64(x, 'proxL1').reduce((a, v) => a + Math.abs(v), 0),
    prox: (v, eta) => vec(toF64(v, 'proxL1').map((vi) => Math.sign(vi) * Math.max(Math.abs(vi) - eta * lambda, 0))),
  }
}

/**
 * $g(\xvec) = \lambda \lVert \xvec \rVert_2$ (the unsquared norm, as in the group lasso): the prox is block soft
 * thresholding, $\max(0, 1 - \eta\lambda/\lVert \vvec \rVert_2)\,\vvec$, which sends the whole vector to zero
 * when its norm is at most $\eta\lambda$.
 *
 * @param lambda The weight $\lambda \ge 0$ of the norm.
 * @returns The `Prox` of $g$.
 *
 * @example Block soft thresholding shrinks the whole vector
 * const g = proxL2(1)
 * print('shrunk towards 0 =', g.prox(tensor([3, 4]), 1))
 * print('a norm of at most 1 goes to 0 =', g.prox(tensor([0.3, 0.4]), 1))
 * print('g(x) =', g.value(tensor([3, 4])))
 */
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

/**
 * $g(\xvec) = (\lambda/2) \lVert \xvec \rVert_2^2$ (ridge): the prox is shrinkage, $\vvec / (1 + \eta\lambda)$.
 *
 * @param lambda The weight $\lambda \ge 0$.
 * @returns The `Prox` of $g$.
 *
 * @example Shrinkage by a constant factor
 * const g = proxSquaredL2(1)
 * print('prox =', g.prox(tensor([2, 4]), 1))
 * print('g(x) =', g.value(tensor([2, 4])))
 */
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

/**
 * The indicator of $\boldsymbol{\ell} \le \xvec \le \uvec$ (0 inside, Infinity outside): the prox is the projection
 * onto the box, whatever the step.
 *
 * @param lo The lower bound: one number for every coordinate, or a vector of one per coordinate.
 * @param hi The upper bound, in the same form.
 * @returns The `Prox` of the indicator.
 *
 * @example The prox clips; the value says whether a point is inside
 * const g = proxBox(0, 1)
 * print('prox =', g.prox(tensor([-0.5, 0.3, 2]), 0.1))
 * print('g inside =', g.value(tensor([0.5, 0.5])))
 * print('g outside =', g.value(tensor([2, 0.5])))
 */
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

/**
 * The indicator of $\xvec \ge 0$: the prox is $\max(\vvec, 0)$ coordinatewise, whatever the step.
 *
 * @returns The `Prox` of the indicator.
 *
 * @example The prox is the nonnegative part
 * const g = proxNonnegative()
 * print('prox =', g.prox(tensor([-1, 2]), 1))
 * print('g inside =', g.value(tensor([1, 2])))
 * print('g outside =', g.value(tensor([-1, 2])))
 */
export function proxNonnegative(): Prox {
  return {
    name: 'nonnegative',
    value: (x) => (toF64(x, 'proxNonnegative').every((v) => v >= 0) ? 0 : Infinity),
    prox: projectNonnegative(),
  }
}

/** The state of `proximalGradient` (and so of `ista`, `fista` and `projectedGradient`). */
export type ProximalGradientState = IterateState & {
  /** $F(\xvec) = f(\xvec) + g(\xvec)$ (the `value` field) splits into `smoothValue` $f(\xvec)$ and `penalty`. */
  smoothValue: number
  /** $g(\xvec)$ (0 for `projectedGradient`). */
  penalty: number
  /**
   * The point $\yvec$ where the next gradient step is taken: $\xvec$ itself without acceleration, the extrapolated
   * point in FISTA.
   */
  y: Vector
  /** $\nabla f(\yvec)$. */
  gradY: Vector
  /** FISTA's momentum sequence $t_k$ (1 without acceleration). */
  momentum: number
  /** The step size $\eta = 1/L$ used on the last step (NaN at $t = 0$). */
  stepSize: number
  /** The Lipschitz estimate $L$ for the next step (from backtracking, or 1/`stepSize`). */
  lipschitz: number
  /**
   * $\lVert G_\eta(\yvec) \rVert = \lVert \yvec - \xvec^+ \rVert/\eta$, the gradient mapping's norm; zero exactly
   * at a minimiser of $F$, so it is the stopping test (NaN at $t = 0$).
   */
  mappingNorm: number
  /** Backtracking trials on the last step: each $L$ tried and whether the quadratic upper bound held. */
  trials: { lipschitz: number; accepted: boolean }[]
}

/** Options for `proximalGradient`. */
export type ProximalGradientOptions = Omit<StoppingOptions, 'tolerance'> & {
  /**
   * Fixed step $\eta$ ($\le 1/L$ for a gradient that is $L$-Lipschitz). Used as the first guess $1/L_0$ with
   * `backtracking`. Default 0.1.
   */
  stepSize?: number
  /**
   * Find $L$ by backtracking, $L \leftarrow L \cdot$ `increase` until
   * $f(\xvec^+) \le f(\yvec) + \nabla f(\yvec)^\top(\xvec^+ - \yvec) + (L/2)\lVert \xvec^+ - \yvec \rVert^2$ (Beck
   * and Teboulle, 2009, §3, with the estimate carried between steps, so $L$ never decreases). After 60 trials the last
   * one is taken. Default false.
   */
  backtracking?: boolean
  /** Growth factor of $L$ in backtracking. Default 2. */
  increase?: number
  /** Nesterov / FISTA extrapolation. Default false (ISTA). */
  accelerated?: boolean
  /** Stop when the gradient mapping's norm is at most this. Default 1e-6. */
  tolerance?: number
}

/**
 * Proximal gradient descent on $F = f + g$: $\xvec^+ = \prox_{\eta g}(\yvec - \eta\nabla f(\yvec))$, with
 * $\yvec = \xvec$ (ISTA) or, with `accelerated`, the FISTA extrapolation
 * $\yvec_{k+1} = \xvec_k + ((t_k - 1)/t_{k+1})(\xvec_k - \xvec_{k-1})$, $t_{k+1} = (1 + \sqrt{1 + 4t_k^2})/2$
 * (Beck and Teboulle, 2009). With $\eta \le 1/L$, $F(\xvec_k) - F^*$ is $O(1/k)$ for ISTA and $O(1/k^2)$ for FISTA.
 * `init` takes `{ x0 }`. It converges when the gradient mapping's norm is within `tolerance`, and flags divergence
 * when $f$ or the iterate is not finite, $\lvert f \rvert$ exceeds `divergeAbove`, or $g$ is NaN.
 *
 * @param f The smooth part: takes a point and returns `{ value, grad }`, $f(\xvec)$ and $\nabla f(\xvec)$.
 * @param g The nonsmooth part, as a `Prox` (`proxL1`, `proxBox`, ...).
 * @param options The step size or backtracking, acceleration, and the stopping and divergence thresholds.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example A smooth term plus an L1 penalty, solved by soft thresholding
 * // f(x) = ½(x₁ − 3)² + 2(x₂ − 0.2)², g = ‖x‖₁: the minimiser soft-thresholds (3, 0.2) at (1, 1/4).
 * const f = (x) => {
 *   const [p, q] = toFlat(x)
 *   return { value: 0.5 * (p - 3) ** 2 + 2 * (q - 0.2) ** 2, grad: [p - 3, 4 * (q - 0.2)] }
 * }
 * const s = run(proximalGradient(f, proxL1(1), { stepSize: 0.25 }), { x0: [0, 0] }, 500)
 * print('x =', s.x)
 * print('F(x) =', s.value)
 * print('steps =', s.t)
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

/**
 * ISTA (iterative shrinkage-thresholding): `proximalGradient` without acceleration.
 *
 * @param f The smooth part: takes a point and returns `{ value, grad }`.
 * @param g The nonsmooth part, as a `Prox`.
 * @param options The options of `proximalGradient`, without `accelerated`.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example A lasso, with the step found by backtracking
 * // Lasso: f(x) = ½‖Ax − b‖² with A = [[1, 0.9], [0.9, 1]] and g = 0.1‖x‖₁; the gradient of f is Aᵀ(Ax − b).
 * const b = [1, 2]
 * const f = (x) => {
 *   const [p, q] = toFlat(x)
 *   const r = [p + 0.9 * q - b[0], 0.9 * p + q - b[1]]
 *   return { value: 0.5 * (r[0] ** 2 + r[1] ** 2), grad: [r[0] + 0.9 * r[1], 0.9 * r[0] + r[1]] }
 * }
 * // Start from a step of 10, far too long: backtracking raises L until the step is safe.
 * const s = run(ista(f, proxL1(0.1), { stepSize: 10, backtracking: true }), { x0: [0, 0] }, 5000)
 * print('x =', s.x)
 * print('L found =', s.lipschitz)
 * print('steps =', s.t)
 */
export function ista(
  f: ObjectiveFn,
  g: Prox,
  options: Omit<ProximalGradientOptions, 'accelerated'> = {},
): Algorithm<StartOptions, ProximalGradientState> {
  return proximalGradient(f, g, { ...options, accelerated: false })
}

/**
 * FISTA (Beck and Teboulle, 2009): `proximalGradient` with Nesterov extrapolation.
 *
 * @param f The smooth part: takes a point and returns `{ value, grad }`.
 * @param g The nonsmooth part, as a `Prox`.
 * @param options The options of `proximalGradient`, without `accelerated`.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example FISTA reaches the lasso solution in fewer steps than ISTA
 * // Lasso: f(x) = ½‖Ax − b‖² with A = [[1, 0.9], [0.9, 1]] and g = 0.1‖x‖₁; the gradient of f is Aᵀ(Ax − b).
 * const b = [1, 2]
 * const f = (x) => {
 *   const [p, q] = toFlat(x)
 *   const r = [p + 0.9 * q - b[0], 0.9 * p + q - b[1]]
 *   return { value: 0.5 * (r[0] ** 2 + r[1] ** 2), grad: [r[0] + 0.9 * r[1], 0.9 * r[0] + r[1]] }
 * }
 * for (const alg of [ista(f, proxL1(0.1), { stepSize: 0.25 }), fista(f, proxL1(0.1), { stepSize: 0.25 })]) {
 *   const s = run(alg, { x0: [0, 0] }, 5000)
 *   print(alg.name, 'x =', s.x, 'steps =', s.t)
 * }
 */
export function fista(
  f: ObjectiveFn,
  g: Prox,
  options: Omit<ProximalGradientOptions, 'accelerated'> = {},
): Algorithm<StartOptions, ProximalGradientState> {
  return proximalGradient(f, g, { ...options, accelerated: true })
}

/**
 * Projected gradient descent onto a closed convex set $\Ccal$: $\xvec \leftarrow P_\Ccal(\xvec - \eta\nabla f(\xvec))$,
 * that is proximal gradient with $g$ the indicator of $\Ccal$, whose prox is the projection `project` (such as
 * `projectBox`, `projectBall`, `projectSimplex`). `init` takes `{ x0 }` (projected first is not required: the first
 * step lands in $\Ccal$). The state's `penalty` is always 0, so `value` is $f(\xvec)$, and the initial `value` is
 * $f(\xvec_0)$ even when $\xvec_0$ is outside $\Ccal$.
 *
 * @param f The objective: takes a point and returns `{ value, grad }`, $f(\xvec)$ and $\nabla f(\xvec)$.
 * @param project The Euclidean projection onto $\Ccal$, from a point to its projection.
 * @param options The options of `proximalGradient`; `accelerated` gives accelerated projected gradient.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example The closest point of the unit ball to (2, -1)
 * const f = (x) => {
 *   const [a, b] = toFlat(x)
 *   return { value: 0.5 * ((a - 2) ** 2 + (b + 1) ** 2), grad: [a - 2, b + 1] }
 * }
 * const s = run(projectedGradient(f, projectBall(1), { stepSize: 0.5 }), { x0: [0, 0] }, 500)
 * print('x =', s.x)
 * print('(2, -1)/sqrt(5) =', [2 / Math.sqrt(5), -1 / Math.sqrt(5)])
 */
export function projectedGradient(
  f: ObjectiveFn,
  project: (x: Vector) => Vector,
  options: ProximalGradientOptions = {},
): Algorithm<StartOptions, ProximalGradientState> {
  const alg = proximalGradient(f, { name: 'projection', value: () => 0, prox: (v) => project(v) }, options)
  return { ...alg, name: options.accelerated ? 'accelerated-projected-gradient' : 'projected-gradient' }
}
